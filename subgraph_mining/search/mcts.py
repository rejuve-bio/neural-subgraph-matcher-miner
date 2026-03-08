import random
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
import networkx as nx
import scipy.stats as stats
from tqdm import tqdm

from common import utils
from .base import SearchAgent


class MCTSSearchAgent(SearchAgent):
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20, c_uct=0.7):
        """MCTS implementation of the subgraph pattern search."""
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
        self.counts = defaultdict(lambda: defaultdict(list))

    def is_search_done(self):
        return self.max_size == self.max_pattern_size + 1

    def has_min_reachable_nodes(self, graph, start_node, n):
        for depth_limit in range(n + 1):
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
            (self.max_pattern_size + 1 - self.min_pattern_size))):
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
                    if n_embs > 0:
                        v_score = -np.log(score / n_embs + 1) + 1
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
                cur_state = utils.wl_hash(neigh_g, node_anchored=self.node_anchored)
                state_list.append(cur_state)
                self.wl_hash_to_graphs[cur_state].append(neigh_g)

            # backprop value
            for i in range(0, len(state_list) - 1):
                self.cum_action_values[state_list[i]][
                    state_list[i + 1]] += best_v_score
                self.visit_counts[state_list[i]][state_list[i + 1]] += 1
        self.max_size += 1

    def finish_search(self):
        counts = defaultdict(lambda: defaultdict(int))
        for _, v in self.visit_counts.items():
            for s2, count in v.items():
                counts[len(random.choice(self.wl_hash_to_graphs[s2]))][s2] += count

        for wl_hash, graphs in self.wl_hash_to_graphs.items():
            if graphs:
                size = len(graphs[0])
                self.counts[size][wl_hash].extend(graphs)

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size + 1):
            for wl_hash, count in sorted(counts[pattern_size].items(), key=lambda x: x[1], reverse=True)[:self.out_batch_size]:
                cand_patterns_uniq.append(random.choice(self.wl_hash_to_graphs[wl_hash]))
                print("- outputting", count, "motifs of size", pattern_size)
        return cand_patterns_uniq


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
            (self.max_pattern_size + 1 - self.min_pattern_size))):

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

                        if n_embs > 0 and score / n_embs > 0.5:
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
