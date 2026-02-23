import os
import time
import random
from collections import defaultdict

import numpy as np
import torch
import networkx as nx
import scipy.stats as stats
from tqdm import tqdm
import torch.multiprocessing as mp

from common import utils
from .base import SearchAgent

mp.set_start_method('spawn', force=True)


def default_dd_list():
    return defaultdict(list)


worker_model = None
worker_graphs = None
worker_embs = None
worker_args = None


def init_greedy_worker(model, graphs, embs, args):
    """
    Initializer function for each worker process in the pool.
    This runs ONCE per worker and loads the large data into its global scope.
    """
    global worker_model, worker_graphs, worker_embs, worker_args
    print(f"[{time.strftime('%H:%M:%S')}] Worker PID {os.getpid()} initializing...", flush=True)
    worker_model = model
    worker_graphs = graphs
    worker_embs = embs
    worker_args = args
    print(f"[{time.strftime('%H:%M:%S')}] Worker PID {os.getpid()} initialization complete.", flush=True)


def run_greedy_trial(trial_idx):
    """
    Executes a single greedy search trial.
    It now accesses the large data from global variables, avoiding data transfer.
    """
    global worker_model, worker_graphs, worker_embs, worker_args

    random.seed(int.from_bytes(os.urandom(4), 'little') + trial_idx)
    np.random.seed(int.from_bytes(os.urandom(4), 'little') + trial_idx)

    ps = np.array([len(g) for g in worker_graphs], dtype=np.float32)
    ps /= np.sum(ps)
    graph_dist = stats.rv_discrete(values=(np.arange(len(worker_graphs)), ps))

    graph_idx = np.arange(len(worker_graphs))[graph_dist.rvs()]
    graph = worker_graphs[graph_idx]
    start_node = random.choice(list(graph.nodes))

    neigh = [start_node]
    is_directed = graph.is_directed() if hasattr(graph, "is_directed") else False
    if worker_args.graph_type == "undirected" or not is_directed:
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
    elif worker_args.graph_type == "directed":
        frontier = list((set(graph.successors(start_node)) | set(graph.predecessors(start_node))) - set(neigh))
    visited = {start_node}

    trial_patterns = defaultdict(list)
    trial_counts = defaultdict(default_dd_list)

    while len(neigh) < worker_args.max_pattern_size and frontier:
        cand_neighs, anchors = [], []
        for cand_node in frontier:
            cand_neigh = graph.subgraph(neigh + [cand_node])
            cand_neighs.append(cand_neigh)
            if worker_args.node_anchored:
                anchors.append(neigh[0])

        if not cand_neighs:
            break

        with torch.no_grad():
            cand_embs = worker_model.emb_model(utils.batch_nx_graphs(
                cand_neighs, anchors=anchors if worker_args.node_anchored else None))

        best_score = float("inf")
        best_node = None

        for cand_node, cand_emb in zip(frontier, cand_embs):
            score = 0
            for emb_batch in worker_embs:
                with torch.no_grad():
                    if worker_args.method_type == "order":
                        pred = worker_model.predict((emb_batch.to(utils.get_device()), cand_emb)).unsqueeze(1)
                        score -= torch.sum(torch.argmax(worker_model.clf_model(pred), axis=1)).item()
                    elif worker_args.method_type == "mlp":
                        pred = worker_model(emb_batch.to(utils.get_device()), cand_emb.unsqueeze(0).expand(len(emb_batch), -1))
                        score += torch.sum(pred[:, 0]).item()

            if score < best_score:
                best_score = score
                best_node = cand_node

        if best_node is None:
            break

        if worker_args.graph_type == "undirected" or not is_directed:
            frontier = list(((set(frontier) | set(graph.neighbors(best_node))) - visited) - {best_node})
        elif worker_args.graph_type == "directed":
            frontier = list(((set(frontier) | set(graph.successors(best_node)) | set(graph.predecessors(best_node))) - visited) - {best_node})

        visited.add(best_node)
        neigh.append(best_node)

        if len(neigh) >= worker_args.min_pattern_size:
            neigh_g = graph.subgraph(neigh).copy()
            neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
            for v_idx, v in enumerate(neigh_g.nodes):
                neigh_g.nodes[v]["anchor"] = 1 if worker_args.node_anchored and v == neigh[0] else 0

            trial_patterns[len(neigh_g)].append((best_score, neigh_g))
            trial_counts[len(neigh_g)][utils.wl_hash(neigh_g, node_anchored=worker_args.node_anchored)].append(neigh_g)

    return trial_patterns, trial_counts


class GreedySearchAgent(SearchAgent):
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, rank_method="counts",
        model_type="order", out_batch_size=20, n_beams=1, n_workers=4):
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            embs, node_anchored=node_anchored, analyze=analyze,
            model_type=model_type, out_batch_size=out_batch_size)
        self.rank_method = rank_method
        self.n_beams = n_beams
        self.n_workers = n_workers
        print("Rank Method:", rank_method)
        if self.n_workers > 1:
            print(f"Using {self.n_workers} worker processes for parallel search.")

    def run_search(self, n_trials=1000):
        """
        Overridden run_search that uses an initializer to avoid repetitive data transfer.
        """
        self.cand_patterns = defaultdict(list)
        self.counts = defaultdict(lambda: defaultdict(list))
        self.n_trials = n_trials

        init_args = (self.model, self.dataset, self.embs, self.args)

        args_for_pool = range(n_trials)

        if self.n_workers > 1:
            print(f"Starting {n_trials} search trials on {self.n_workers} cores...")
            with mp.Pool(processes=self.n_workers, initializer=init_greedy_worker, initargs=init_args) as pool:
                results = list(tqdm(pool.imap_unordered(run_greedy_trial, args_for_pool), total=n_trials))
        else:
            print(f"Starting {n_trials} search trials sequentially (n_workers={self.n_workers})...")
            init_greedy_worker(*init_args)
            results = [run_greedy_trial(i) for i in tqdm(range(n_trials))]

        print("Aggregating results from all worker processes...")
        for trial_patterns, trial_counts in results:
            for size, scored_patterns in trial_patterns.items():
                self.cand_patterns[size].extend(scored_patterns)
            for size, hashed_patterns in trial_counts.items():
                for h, graphs in hashed_patterns.items():
                    self.counts[size][h].extend(graphs)

        return self.finish_search()

    def finish_search(self):
        """
        Processes the aggregated results from all trials to find the most frequent patterns.
        This method remains unchanged.
        """
        if self.analyze:
            pass

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size + 1):
            if self.rank_method == "hybrid":
                if self.counts[pattern_size]:
                    cur_rank_method = "margin" if len(max(
                        self.counts[pattern_size].values(), key=len, default=[])) < 3 else "counts"
                else:
                    cur_rank_method = "margin"
            else:
                cur_rank_method = self.rank_method

            print(f"Ranking patterns of size {pattern_size} using method: '{cur_rank_method}'")

            if cur_rank_method == "margin":
                wl_hashes = set()
                cands = self.cand_patterns[pattern_size]
                cand_patterns_uniq_size = []
                for score, pattern in sorted(cands, key=lambda x: x[0]):
                    wl_hash = utils.wl_hash(pattern, node_anchored=self.node_anchored)
                    if wl_hash not in wl_hashes:
                        wl_hashes.add(wl_hash)
                        cand_patterns_uniq_size.append(pattern)
                        if len(cand_patterns_uniq_size) >= self.out_batch_size:
                            break
                cand_patterns_uniq.extend(cand_patterns_uniq_size)

            elif cur_rank_method == "counts":
                sorted_counts = sorted(self.counts[pattern_size].items(), key=lambda x: len(x[1]), reverse=True)
                for _, neighs in sorted_counts[:self.out_batch_size]:
                    cand_patterns_uniq.append(random.choice(neighs))
            else:
                print("Unrecognized rank method")

        return cand_patterns_uniq


class MemoryEfficientGreedyAgent(GreedySearchAgent):
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, rank_method="counts",
        model_type="order", out_batch_size=20, batch_size=64):
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            embs, node_anchored=node_anchored, analyze=analyze,
            rank_method=rank_method, model_type=model_type,
            out_batch_size=out_batch_size)
        self.batch_size = batch_size
        self.use_fp16 = torch.cuda.is_available()

    def _grow_pattern(self, graph, start_node):
        neigh = [start_node]
        visited = {start_node}
        frontier = set(graph.neighbors(start_node))

        while frontier and len(neigh) < self.max_pattern_size:
            best_score = float('inf')
            best_node = None

            for i in range(0, len(frontier), self.batch_size):
                batch_nodes = list(frontier)[i:i+self.batch_size]
                cand_neighs = [graph.subgraph(neigh + [n]) for n in batch_nodes]
                anchors = [neigh[0]] * len(cand_neighs) if self.node_anchored else None

                with torch.no_grad():
                    cand_embs = self.model.emb_model(utils.batch_nx_graphs(
                        cand_neighs, anchors=anchors))

                    if self.use_fp16:
                        cand_embs = self._half_tensor(cand_embs)

                    for node, emb in zip(batch_nodes, cand_embs):
                        score = 0
                        for emb_batch in self.embs:
                            if self.use_fp16:
                                emb_batch = self._half_tensor(emb_batch)

                            if self.model_type == "order":
                                pred = self.model.predict((
                                    emb_batch.to(utils.get_device()),
                                    emb)).unsqueeze(1)
                                if self.use_fp16:
                                    pred = pred.float()
                                score -= torch.sum(torch.argmax(
                                    self.model.clf_model(pred), axis=1)).item()
                            elif self.model_type == "mlp":
                                pred = self.model(
                                    emb_batch.to(utils.get_device()),
                                    emb.unsqueeze(0).expand(len(emb_batch), -1)
                                    )
                                if self.use_fp16:
                                    pred = pred.float()
                                score += torch.sum(pred[:, 0]).item()

                        if score < best_score:
                            best_score = score
                            best_node = node

            if best_node is None:
                break

            neigh.append(best_node)
            visited.add(best_node)
            frontier = set((frontier | set(graph.neighbors(best_node))) -
                     visited - {best_node})

        if len(neigh) >= self.min_pattern_size:
            pattern = graph.subgraph(neigh).copy()
            pattern.remove_edges_from(nx.selfloop_edges(pattern))
            for v in pattern.nodes:
                pattern.nodes[v]["anchor"] = 1 if v == neigh[0] else 0

            if self.analyze:
                emb = self.model.emb_model(utils.batch_nx_graphs(
                    [pattern], anchors=[neigh[0]] if self.node_anchored else None)).squeeze(0)
                self.analyze_embs.append([emb.detach().cpu().numpy()])

            self.cand_patterns[len(pattern)].append((best_score, pattern))
            if self.rank_method in ["counts", "hybrid"]:
                self.counts[len(pattern)][utils.wl_hash(pattern,
                    node_anchored=self.node_anchored)].append(pattern)

            return pattern
        return None

    def step(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        new_beam_sets = []
        processed_graphs = set()

        for beam_set in tqdm(self.beam_sets):
            if isinstance(beam_set, (list, tuple)) and len(beam_set) > 0:
                if isinstance(beam_set[0], (list, tuple)):
                    graph_idx = beam_set[0][-1]
                else:
                    graph_idx = beam_set[-1]
                processed_graphs.add(graph_idx)

            patterns = []
            try:
                states = [beam_set] if not isinstance(beam_set[0], (list, tuple)) else beam_set
                for state in states:
                    if len(state) >= 5:
                        _, neigh, frontier, visited, graph_idx = state
                        graph = self.dataset[graph_idx]

                        for node in list(frontier)[:self.batch_size]:
                            pattern = self._grow_pattern(graph, node)
                            if pattern is not None:
                                patterns.append(pattern)

                if patterns:
                    patterns.sort(key=len, reverse=True)
                    new_beam_sets.append(patterns[:self.n_beams])

            except Exception as e:
                print(f"Error processing beam: {e}")
                continue

        print(f"Processing beams from {len(processed_graphs)} distinct graphs")
        self.beam_sets = [b for b in new_beam_sets if b]
