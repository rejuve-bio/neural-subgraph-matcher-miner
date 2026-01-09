import argparse
import csv
from itertools import combinations
import time
import os

from deepsnap.batch import Batch
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm

from common import data
from common import models
from common import utils
from common import combined_syn
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder

import matplotlib.pyplot as plt

import random
from scipy.io import mmread
import scipy.stats as stats
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import defaultdict
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx
import pickle
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from sklearn.decomposition import PCA
from functools import lru_cache
import torch.nn as nn
class SearchAgent:
    """ Class for search strategies to identify frequent subgraphs in embedding space.

    The problem is formulated as a search. The first action chooses a seed node to grow from.
    Subsequent actions chooses a node in dataset to connect to the existing subgraph pattern,
    increasing the pattern size by 1.

    See paper for rationale and algorithm details.
    """
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20):
        """ Subgraph pattern search by walking in embedding space.

        Args:
            min_pattern_size: minimum size of frequent subgraphs to be identified.
            max_pattern_size: maximum size of frequent subgraphs to be identified.
            model: the trained subgraph matching model (PyTorch nn.Module).
            dataset: the DeepSNAP dataset for which to mine the frequent subgraph pattern.
            embs: embeddings of sampled node neighborhoods (see paper).
            node_anchored: an option to specify whether to identify node_anchored subgraph patterns.
                node_anchored search procedure has to use a node_anchored model (specified in subgraph
                matching config.py).
            analyze: whether to enable analysis visualization.
            model_type: type of the subgraph matching model (requires to be consistent with the model parameter).
            out_batch_size: the number of frequent subgraphs output by the mining algorithm for each size.
                They are predicted to be the out_batch_size most frequent subgraphs in the dataset.
        """
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.model = model
        self.dataset = dataset
        self.embs = embs
        self.node_anchored = node_anchored
        self.analyze = analyze
        self.model_type = model_type
        self.out_batch_size = out_batch_size

    def run_search(self, n_trials=1000): 
        self.cand_patterns = defaultdict(list)
        self.counts = defaultdict(lambda: defaultdict(list))
        self.n_trials = n_trials

        self.init_search()
        while not self.is_search_done():
            self.step()
        return self.finish_search()

    def init_search():
        raise NotImplementedError

    def step(self):
        """ Abstract method for executing a search step.
        Every step adds a new node to the subgraph pattern.
        Run_search calls step at least min_pattern_size times to generate a pattern of at least this
        size. To be inherited by concrete search strategy implementations.
        """
        raise NotImplementedError

class MCTSSearchAgent(SearchAgent):
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20, c_uct=0.7):
        """ MCTS implementation of the subgraph pattern search.
        Uses MCTS strategy to search for the most common pattern.

        Args:
            c_uct: the exploration constant used in UCT criteria (See paper).
        """
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            embs, node_anchored=node_anchored, analyze=analyze,
            model_type=model_type, out_batch_size=out_batch_size)
        self.c_uct = c_uct
        assert not analyze

    def init_search(self):
        self.wl_hash_to_graphs = defaultdict(list)
        self.cum_action_values = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(float))
        self.visited_seed_nodes = set()
        self.max_size = self.min_pattern_size

    def is_search_done(self):
        return self.max_size == self.max_pattern_size + 1

    def has_min_reachable_nodes(self, graph, start_node, n):
        for depth_limit in range(n+1):
            edges = nx.bfs_edges(graph, start_node, depth_limit=depth_limit)
            nodes = set([v for u, v in edges])
            if len(nodes) + 1 >= n:
                return True
        return False

    def step(self):
        ps = np.array([len(g) for g in self.dataset], dtype=float)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        print("Size", self.max_size)
        print(len(self.visited_seed_nodes), "distinct seeds")
        for simulation_n in tqdm(range(self.n_trials //
            (self.max_pattern_size+1-self.min_pattern_size))):
            # pick seed node
            best_graph_idx, best_start_node, best_score = None, None, -float("inf")
            for cand_graph_idx, cand_start_node in self.visited_seed_nodes:
                state = cand_graph_idx, cand_start_node
                my_visit_counts = sum(self.visit_counts[state].values())
                q_score = (sum(self.cum_action_values[state].values()) /
                    (my_visit_counts or 1))
                uct_score = self.c_uct * np.sqrt(np.log(simulation_n or 1) /
                    (my_visit_counts or 1))
                node_score = q_score + uct_score
                if node_score > best_score:
                    best_score = node_score
                    best_graph_idx = cand_graph_idx
                    best_start_node = cand_start_node
            # if existing seed beats choosing a new seed
            if best_score >= self.c_uct * np.sqrt(np.log(simulation_n or 1)):
                graph_idx, start_node = best_graph_idx, best_start_node
                assert best_start_node in self.dataset[graph_idx].nodes
                graph = self.dataset[graph_idx]
            else:
                found = False
                while not found:
                    graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
                    graph = self.dataset[graph_idx]
                    start_node = random.choice(list(graph.nodes))
                    # don't pick isolated nodes or small islands
                    if self.has_min_reachable_nodes(graph, start_node,
                        self.min_pattern_size):
                        found = True
                self.visited_seed_nodes.add((graph_idx, start_node))
            neigh = [start_node]
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
            visited = set([start_node])
            neigh_g = nx.Graph()
            neigh_g.add_node(start_node, anchor=1)
            cur_state = graph_idx, start_node
            state_list = [cur_state]
            while frontier and len(neigh) < self.max_size:
                cand_neighs, anchors = [], []
                for cand_node in frontier:
                    cand_neigh = graph.subgraph(neigh + [cand_node])
                    cand_neighs.append(cand_neigh)
                    if self.node_anchored:
                        anchors.append(neigh[0])
                cand_embs = self.model.emb_model(utils.batch_nx_graphs(
                    cand_neighs, anchors=anchors if self.node_anchored else None))
                best_v_score, best_node_score, best_node = 0, -float("inf"), None
                for cand_node, cand_emb in zip(frontier, cand_embs):
                    score, n_embs = 0, 0
                    for emb_batch in self.embs:
                        score += torch.sum(self.model.predict((
                            emb_batch.to(utils.get_device()), cand_emb))).item()
                        n_embs += len(emb_batch)
                    EPS = 1e-10  
                    if n_embs > 0:
                        v_score = -np.log(score/n_embs + 1) + 1
                    else:
                        v_score = 0  
                    neigh_g = graph.subgraph(neigh + [cand_node]).copy()
                    neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                    for v in neigh_g.nodes:
                        neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                    next_state = utils.wl_hash(neigh_g,
                        node_anchored=self.node_anchored)
                    # compute node score
                    parent_visit_counts = sum(self.visit_counts[cur_state].values())
                    my_visit_counts = sum(self.visit_counts[next_state].values())
                    q_score = (sum(self.cum_action_values[next_state].values()) /
                        (my_visit_counts or 1))
                    uct_score = self.c_uct * np.sqrt(np.log(parent_visit_counts or
                        1) / (my_visit_counts or 1))
                    node_score = q_score + uct_score
                    if node_score > best_node_score:
                        best_node_score = node_score
                        best_v_score = v_score
                        best_node = cand_node
                frontier = list(((set(frontier) |
                    set(graph.neighbors(best_node))) - visited) -
                    set([best_node]))
                visited.add(best_node)
                neigh.append(best_node)

                # update visit counts, wl cache
                neigh_g = graph.subgraph(neigh).copy()
                neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                for v in neigh_g.nodes:
                    neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                prev_state = cur_state
                cur_state = utils.wl_hash(neigh_g, node_anchored=self.node_anchored)
                state_list.append(cur_state)
                self.wl_hash_to_graphs[cur_state].append(neigh_g)

            # backprop value
            for i in range(0, len(state_list) - 1):
                self.cum_action_values[state_list[i]][
                    state_list[i+1]] += best_v_score
                self.visit_counts[state_list[i]][state_list[i+1]] += 1
        self.max_size += 1

    def finish_search(self):
        counts = defaultdict(lambda: defaultdict(int))
        for _, v in self.visit_counts.items():
            for s2, count in v.items():
                counts[len(random.choice(self.wl_hash_to_graphs[s2]))][s2] += count

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size+1):
            for wl_hash, count in sorted(counts[pattern_size].items(), key=lambda
                x: x[1], reverse=True)[:self.out_batch_size]:
                cand_patterns_uniq.append(random.choice(
                    self.wl_hash_to_graphs[wl_hash]))
                print("- outputting", count, "motifs of size", pattern_size)
        return cand_patterns_uniq
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
    if worker_args.graph_type == "undirected":
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
    elif worker_args.graph_type == "directed":
        frontier = list(set(graph.successors(start_node)) - set(neigh))
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
                        score += torch.sum(pred[:,0]).item()

            if score < best_score:
                best_score = score
                best_node = cand_node

        if best_node is None:
            break

        if worker_args.graph_type == "undirected":
            frontier = list(((set(frontier) | set(graph.neighbors(best_node))) - visited) - {best_node})
        elif worker_args.graph_type == "directed":
            frontier = list(((set(frontier) | set(graph.successors(best_node))) - visited) - {best_node})      
              
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
                                score += torch.sum(pred[:,0]).item()
                                
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

class MemoryEfficientMCTSAgent(MCTSSearchAgent):
    """Memory-efficient MCTS implementation with legacy AMP support"""
    
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20, c_uct=0.7, memory_limit=1000000):
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            embs, node_anchored=node_anchored, analyze=analyze,
            model_type=model_type, out_batch_size=out_batch_size, c_uct=c_uct)
        self.memory_limit = memory_limit
        self.wl_hash_to_graphs = self._create_lru_cache(maxsize=10000)
        self.use_fp16 = torch.cuda.is_available()
        
    def _half_tensor(self, tensor):
        """Helper to convert tensor to FP16 if CUDA is available"""
        return tensor.half() if self.use_fp16 else tensor
        
    def _create_lru_cache(self, maxsize):
        """Create a size-limited LRU cache for storing graph patterns"""
        from functools import lru_cache
        return lru_cache(maxsize=maxsize)
        
    def _stream_neighborhood(self, graph, start_node, max_nodes=1000):
        """Stream neighborhoods instead of loading all at once"""
        visited = {start_node}
        frontier = set(graph.neighbors(start_node))
        while frontier and len(visited) < max_nodes:
            node = frontier.pop()
            if node not in visited:
                visited.add(node)
                frontier.update(n for n in graph.neighbors(node) 
                              if n not in visited)
                yield node
                
    def _batch_embeddings(self, cand_neighs, batch_size=64):
        """Process embeddings in batches with FP16 support"""
        for i in range(0, len(cand_neighs), batch_size):
            batch = cand_neighs[i:i+batch_size]
            # Filter out graphs with no edges
            valid_batch = [g for g in batch if g.number_of_edges() > 0]
        
            # Skip if no valid graphs in this batch
            if not valid_batch:
                continue
            
            anchors = None
            if self.node_anchored:
                anchors = [list(g.nodes)[0] for g in valid_batch]
        
            with torch.no_grad():
                embs = self.model.emb_model(utils.batch_nx_graphs(
                    valid_batch, anchors=anchors))
                if self.use_fp16:
                    embs = self._half_tensor(embs)
                for emb in embs:
                    yield emb
    def step(self):
        """Memory-efficient implementation of the MCTS step with FP16 support"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        ps = np.array([len(g) for g in self.dataset], dtype=np.float32)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        print("Size", self.max_size)
        print(len(self.visited_seed_nodes), "distinct seeds")
        
        for simulation_n in tqdm(range(self.n_trials // 
            (self.max_pattern_size+1-self.min_pattern_size))):
            
            if simulation_n % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
            graph = self.dataset[graph_idx] 
            
            seed_scores = []
            for _ in range(min(10, graph.number_of_nodes())):
                start_node = random.choice(list(graph.nodes))
                n_reachable = sum(1 for _ in self._stream_neighborhood(
                    graph, start_node, max_nodes=self.min_pattern_size))
                seed_scores.append((start_node, n_reachable))
            start_node = max(seed_scores, key=lambda x: x[1])[0]
            
            neigh = [start_node]
            visited = {start_node}
            frontier = set()
            
            for next_node in self._stream_neighborhood(graph, start_node):
                if len(neigh) >= self.max_size:
                    break
                    
                cand_neigh = graph.subgraph(neigh + [next_node])
                if self.node_anchored:
                    for v in cand_neigh.nodes:
                        cand_neigh.nodes[v]["anchor"] = 1 if v == neigh[0] else 0

                if cand_neigh.number_of_edges() > 0:
                    try:
                        cand_emb = next(self._batch_embeddings([cand_neigh]))
        
                        score = 0
                        n_embs = 0
                        for emb_batch in self.embs:
                            if self.use_fp16:
                                emb_batch = self._half_tensor(emb_batch)
                            pred = self.model.predict((
                                emb_batch.to(utils.get_device()), cand_emb))
                            if self.use_fp16:
                                pred = pred.float()
                            score += torch.sum(pred).item()
                            n_embs += len(emb_batch)
            
                        if n_embs > 0 and score/n_embs > 0.5:  
                            neigh.append(next_node)
                            visited.add(next_node)
                            frontier.update(n for n in graph.neighbors(next_node) 
                                if n not in visited)
                    except StopIteration:
                        pass
                if len(neigh) >= self.min_pattern_size:
                    pattern = graph.subgraph(neigh).copy()
                    pattern_hash = utils.wl_hash(pattern,
                        node_anchored=self.node_anchored)
                    self.visit_counts[len(pattern)][pattern_hash] += 1
                    
            self.max_size += 1

class BeamSearchAgent(SearchAgent):
    """Beam Search implementation for subgraph pattern mining.
    
    Beam search maintains a fixed-size set of the most promising candidates at each step,
    providing a good balance between exploration quality and computational efficiency.
    """
    
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20, beam_width=5, batch_size=64):
        """Initialize the beam search agent.
        
        Args:
            min_pattern_size: Minimum size of patterns to find.
            max_pattern_size: Maximum size of patterns to find.
            model: Trained subgraph matching model.
            dataset: DeepSNAP dataset to mine for patterns.
            embs: Embeddings of sampled node neighborhoods.
            node_anchored: Whether to identify node-anchored patterns.
            analyze: Whether to enable analysis visualization.
            model_type: Type of subgraph matching model.
            out_batch_size: Number of patterns to output for each size.
            beam_width: Number of candidates to maintain at each step.
            batch_size: Size of batches for processing embeddings.
        """
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            embs, node_anchored=node_anchored, analyze=analyze,
            model_type=model_type, out_batch_size=out_batch_size)
        self.beam_width = beam_width
        self.batch_size = batch_size
        self.use_fp16 = torch.cuda.is_available()
    
    def _half_tensor(self, tensor):
        """Convert tensor to half precision if CUDA is available."""
        return tensor.half() if self.use_fp16 else tensor
    
    def init_search(self):
        """Initialize search data structures."""
        self.pattern_beams = {size: [] for size in range(
            self.min_pattern_size, self.max_pattern_size + 1)}
        self.cand_patterns = defaultdict(list)
        self.pattern_counts = defaultdict(lambda: defaultdict(list))
        self.trials_completed = 0
        self.current_size = self.min_pattern_size
        self.analyze_embs = [] if self.analyze else None
    
    def is_search_done(self):
        """Check if search is complete."""
        return self.trials_completed >= self.n_trials
    
    def _compute_pattern_score(self, pattern, anchor=None):
        """Compute score for a pattern using the trained model."""
        if pattern.number_of_edges() == 0:
            return float('inf')  # Invalid pattern
            
        with torch.no_grad():
            anchors = [anchor] if self.node_anchored and anchor else None
            emb = self.model.emb_model(utils.batch_nx_graphs([pattern], anchors=anchors)).squeeze(0)
            
            if self.use_fp16:
                emb = self._half_tensor(emb)
                
            score = 0
            n_embs = 0
            
            for emb_batch in self.embs:
                n_embs += len(emb_batch)
                if self.use_fp16:
                    emb_batch = self._half_tensor(emb_batch)
                    
                if self.model_type == "order":
                    pred = self.model.predict((emb_batch.to(utils.get_device()), emb)).unsqueeze(1)
                    if self.use_fp16:
                        pred = pred.float()
                    score -= torch.sum(torch.argmax(self.model.clf_model(pred), axis=1)).item()
                elif self.model_type == "mlp":
                    pred = self.model(
                        emb_batch.to(utils.get_device()),
                        emb.unsqueeze(0).expand(len(emb_batch), -1))
                    if self.use_fp16:
                        pred = pred.float()
                    score += torch.sum(pred[:,0]).item()
            
            return score / max(1, n_embs)  # Normalize by number of embeddings
    
    def _sample_seed_node(self):
        """Sample a seed node from the dataset."""
        # Weight sampling by graph size
        ps = np.array([len(g) for g in self.dataset], dtype=np.float32)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))
        
        # Sample a graph
        graph_idx = graph_dist.rvs()
        graph = self.dataset[graph_idx]
        
        # Sample node with enough neighbors
        candidates = []
        for _ in range(min(10, graph.number_of_nodes())):
            node = random.choice(list(graph.nodes))
            subgraph = graph.subgraph(list(nx.ego_graph(graph, node, radius=2)))
            if subgraph.number_of_nodes() >= self.min_pattern_size:
                candidates.append((node, subgraph.number_of_nodes()))
        
        if not candidates:
            # Fallback to random node
            return graph_idx, random.choice(list(graph.nodes))
        
        # Choose node with largest 2-hop neighborhood
        node = max(candidates, key=lambda x: x[1])[0]
        return graph_idx, node
    
    def _grow_patterns(self, beam):
        """Grow patterns in the current beam by one node."""
        new_candidates = []
        
        for score, pattern, graph_idx, seed_node in beam:
            graph = self.dataset[graph_idx]
            
            # Find nodes that can be added to the pattern
            pattern_nodes = set(pattern.nodes)
            frontier = set()
            for node in pattern_nodes:
                frontier.update(n for n in graph.neighbors(node) if n not in pattern_nodes)
            
            # Process frontier nodes in batches
            for i in range(0, len(frontier), self.batch_size):
                batch_nodes = list(frontier)[i:i+self.batch_size]
                
                for node in batch_nodes:
                    # Create new pattern with added node
                    new_pattern_nodes = list(pattern_nodes) + [node]
                    new_pattern = graph.subgraph(new_pattern_nodes).copy()
                    
                    # Skip if no new edges were added
                    if new_pattern.number_of_edges() <= pattern.number_of_edges():
                        continue
                        
                    # Set anchor if needed
                    if self.node_anchored:
                        for v in new_pattern.nodes:
                            new_pattern.nodes[v]["anchor"] = 1 if v == seed_node else 0
                    
                    # Compute pattern score
                    new_score = self._compute_pattern_score(new_pattern, anchor=seed_node)
                    
                    # Add to candidates
                    new_candidates.append((new_score, new_pattern, graph_idx, seed_node))
        
        # Return top-k candidates
        return sorted(new_candidates, key=lambda x: x[0])[:self.beam_width]
    
    def step(self):
        """Execute one step of beam search."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize beam if needed
        if not self.pattern_beams[self.current_size]:
            # Sample seed nodes and create initial patterns
            initial_beam = []
            num_seeds = min(self.beam_width * 2, self.n_trials - self.trials_completed)
            
            for _ in range(num_seeds):
                graph_idx, seed_node = self._sample_seed_node()
                graph = self.dataset[graph_idx]
                
                # Create pattern from seed node and its 1-hop neighbors
                neighbors = list(graph.neighbors(seed_node))
                if not neighbors:
                    continue
                    
                # Start with seed node and its first neighbor
                initial_nodes = [seed_node, neighbors[0]]
                pattern = graph.subgraph(initial_nodes).copy()
                
                if pattern.number_of_edges() == 0:
                    continue
                    
                # Set anchor if needed
                if self.node_anchored:
                    for v in pattern.nodes:
                        pattern.nodes[v]["anchor"] = 1 if v == seed_node else 0
                
                # Grow pattern to minimum size
                current_pattern = pattern
                current_nodes = set(initial_nodes)
                
                while len(current_nodes) < self.min_pattern_size and neighbors:
                    # Add next neighbor
                    next_node = neighbors.pop(0)
                    if next_node in current_nodes:
                        continue
                        
                    current_nodes.add(next_node)
                    current_pattern = graph.subgraph(list(current_nodes)).copy()
                    
                    # Set anchor if needed
                    if self.node_anchored:
                        for v in current_pattern.nodes:
                            current_pattern.nodes[v]["anchor"] = 1 if v == seed_node else 0
                
                if len(current_pattern) < self.min_pattern_size:
                    continue
                    
                # Compute pattern score
                score = self._compute_pattern_score(current_pattern, anchor=seed_node)
                initial_beam.append((score, current_pattern, graph_idx, seed_node))
            
            # Sort and keep top beam_width patterns
            self.pattern_beams[self.current_size] = sorted(
                initial_beam, key=lambda x: x[0])[:self.beam_width]
            self.trials_completed += len(initial_beam)
        
        # Grow patterns
        current_beam = self.pattern_beams[self.current_size]
        if current_beam and self.current_size < self.max_pattern_size:
            next_beam = self._grow_patterns(current_beam)
            
            if next_beam:
                self.pattern_beams[self.current_size + 1] = next_beam
        
        # Record patterns from current beam
        for score, pattern, graph_idx, seed_node in current_beam:
            # Add to candidate patterns
            self.cand_patterns[len(pattern)].append((score, pattern))
            
            # Track pattern counts by WL hash
            pattern_hash = utils.wl_hash(pattern, node_anchored=self.node_anchored)
            self.pattern_counts[len(pattern)][pattern_hash].append(pattern)
            
            # Save embedding for analysis if needed
            if self.analyze:
                with torch.no_grad():
                    anchors = [seed_node] if self.node_anchored else None
                    emb = self.model.emb_model(utils.batch_nx_graphs(
                        [pattern], anchors=anchors)).squeeze(0)
                    self.analyze_embs.append(emb.detach().cpu().numpy())
        
        # Move to next pattern size or wrap around
        self.current_size += 1
        if self.current_size > self.max_pattern_size:
            self.current_size = self.min_pattern_size
    
    def finish_search(self):
        """Finish search and return identified patterns."""
        if self.analyze:
            print("Saving analysis info in results/analyze.p")
            with open("results/analyze.p", "wb") as f:
                pickle.dump((self.cand_patterns, self.analyze_embs), f)
            
            # Create visualization
            analysis_data = np.array(self.analyze_embs)
            if len(analysis_data) > 0 and len(analysis_data[0]) >= 2:
                xs, ys = analysis_data[:, 0], analysis_data[:, 1]
                plt.scatter(xs, ys, color="red", label="motif")
                plt.legend()
                plt.savefig("plots/analyze.png")
                plt.close()
        
        # Collect results
        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size + 1):
            pattern_counts = [(h, len(ps)) for h, ps in self.pattern_counts[pattern_size].items()]
            
            # Take the most frequent patterns
            for wl_hash, count in sorted(pattern_counts, key=lambda x: x[1], reverse=True)[:self.out_batch_size]:
                patterns = self.pattern_counts[pattern_size][wl_hash]
                if patterns:
                    cand_patterns_uniq.append(random.choice(patterns))
                    print(f"- outputting {count} motifs of size {pattern_size}")
        
        return cand_patterns_uniq