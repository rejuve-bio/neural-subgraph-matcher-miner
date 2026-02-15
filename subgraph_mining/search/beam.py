import random
from collections import defaultdict

import numpy as np
import torch
import networkx as nx
import scipy.stats as stats
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import utils
from .base import SearchAgent


class BeamSearchAgent(SearchAgent):
    """Beam Search implementation for subgraph pattern mining."""

    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20, beam_width=5, batch_size=64):
        """Initialize the beam search agent."""
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
        self.counts = self.pattern_counts
        self.trials_completed = 0
        self.current_size = self.min_pattern_size
        self.analyze_embs = [] if self.analyze else None

    def is_search_done(self):
        """Check if search is complete."""
        return self.trials_completed >= self.n_trials

    def _compute_pattern_score(self, pattern, anchor=None):
        """Compute score for a pattern using the trained model."""
        if pattern.number_of_edges() == 0:
            return float('inf')

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
                    score += torch.sum(pred[:, 0]).item()

            return score / max(1, n_embs)

    def _sample_seed_node(self):
        """Sample a seed node from the dataset."""
        ps = np.array([len(g) for g in self.dataset], dtype=np.float32)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        graph_idx = graph_dist.rvs()
        graph = self.dataset[graph_idx]

        candidates = []
        for _ in range(min(10, graph.number_of_nodes())):
            node = random.choice(list(graph.nodes))
            subgraph = graph.subgraph(list(nx.ego_graph(graph, node, radius=2)))
            if subgraph.number_of_nodes() >= self.min_pattern_size:
                candidates.append((node, subgraph.number_of_nodes()))

        if not candidates:
            return graph_idx, random.choice(list(graph.nodes))

        node = max(candidates, key=lambda x: x[1])[0]
        return graph_idx, node

    def _grow_patterns(self, beam):
        """Grow patterns in the current beam by one node."""
        new_candidates = []

        for score, pattern, graph_idx, seed_node in beam:
            graph = self.dataset[graph_idx]

            pattern_nodes = set(pattern.nodes)
            frontier = set()
            for node in pattern_nodes:
                frontier.update(n for n in graph.neighbors(node) if n not in pattern_nodes)

            for i in range(0, len(frontier), self.batch_size):
                batch_nodes = list(frontier)[i:i+self.batch_size]

                for node in batch_nodes:
                    new_pattern_nodes = list(pattern_nodes) + [node]
                    new_pattern = graph.subgraph(new_pattern_nodes).copy()

                    if new_pattern.number_of_edges() <= pattern.number_of_edges():
                        continue

                    if self.node_anchored:
                        for v in new_pattern.nodes:
                            new_pattern.nodes[v]["anchor"] = 1 if v == seed_node else 0

                    new_score = self._compute_pattern_score(new_pattern, anchor=seed_node)

                    new_candidates.append((new_score, new_pattern, graph_idx, seed_node))

        return sorted(new_candidates, key=lambda x: x[0])[:self.beam_width]

    def step(self):
        """Execute one step of beam search."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not self.pattern_beams[self.current_size]:
            initial_beam = []
            num_seeds = min(self.beam_width * 2, self.n_trials - self.trials_completed)

            for _ in range(num_seeds):
                graph_idx, seed_node = self._sample_seed_node()
                graph = self.dataset[graph_idx]

                neighbors = list(graph.neighbors(seed_node))
                if not neighbors:
                    continue

                initial_nodes = [seed_node, neighbors[0]]
                pattern = graph.subgraph(initial_nodes).copy()

                if pattern.number_of_edges() == 0:
                    continue

                if self.node_anchored:
                    for v in pattern.nodes:
                        pattern.nodes[v]["anchor"] = 1 if v == seed_node else 0

                current_pattern = pattern
                current_nodes = set(initial_nodes)

                while len(current_nodes) < self.min_pattern_size and neighbors:
                    next_node = neighbors.pop(0)
                    if next_node in current_nodes:
                        continue

                    current_nodes.add(next_node)
                    current_pattern = graph.subgraph(list(current_nodes)).copy()

                    if self.node_anchored:
                        for v in current_pattern.nodes:
                            current_pattern.nodes[v]["anchor"] = 1 if v == seed_node else 0

                if len(current_pattern) < self.min_pattern_size:
                    continue

                score = self._compute_pattern_score(current_pattern, anchor=seed_node)
                initial_beam.append((score, current_pattern, graph_idx, seed_node))

            self.pattern_beams[self.current_size] = sorted(
                initial_beam, key=lambda x: x[0])[:self.beam_width]
            self.trials_completed += len(initial_beam)

        current_beam = self.pattern_beams[self.current_size]
        if current_beam and self.current_size < self.max_pattern_size:
            next_beam = self._grow_patterns(current_beam)

            if next_beam:
                self.pattern_beams[self.current_size + 1] = next_beam

        for score, pattern, graph_idx, seed_node in current_beam:
            self.cand_patterns[len(pattern)].append((score, pattern))
            pattern_hash = utils.wl_hash(pattern, node_anchored=self.node_anchored)
            self.pattern_counts[len(pattern)][pattern_hash].append(pattern)

            if self.analyze:
                with torch.no_grad():
                    anchors = [seed_node] if self.node_anchored else None
                    emb = self.model.emb_model(utils.batch_nx_graphs(
                        [pattern], anchors=anchors)).squeeze(0)
                    self.analyze_embs.append(emb.detach().cpu().numpy())

        self.current_size += 1
        if self.current_size > self.max_pattern_size:
            self.current_size = self.min_pattern_size

    def finish_search(self):
        """Finish search and return identified patterns."""
        if self.analyze:
            print("Saving analysis info in results/analyze.p")
            with open("results/analyze.p", "wb") as f:
                pickle.dump((self.cand_patterns, self.analyze_embs), f)

            analysis_data = np.array(self.analyze_embs)
            if len(analysis_data) > 0 and len(analysis_data[0]) >= 2:
                xs, ys = analysis_data[:, 0], analysis_data[:, 1]
                plt.scatter(xs, ys, color="red", label="motif")
                plt.legend()
                plt.savefig("plots/analyze.png")
                plt.close()

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size + 1):
            pattern_counts = [(h, len(ps)) for h, ps in self.pattern_counts[pattern_size].items()]

            for wl_hash, count in sorted(pattern_counts, key=lambda x: x[1], reverse=True)[:self.out_batch_size]:
                patterns = self.pattern_counts[pattern_size][wl_hash]
                if patterns:
                    cand_patterns_uniq.append(random.choice(patterns))
                    print(f"- outputting {count} motifs of size {pattern_size}")

        return cand_patterns_uniq
