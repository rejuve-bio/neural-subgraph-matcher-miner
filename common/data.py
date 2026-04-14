import os
import pickle
import random

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
from deepsnap.batch import Batch as DSBatch
from deepsnap.dataset import GraphDataset
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.datasets import TUDataset, PPI, QM9
import torch_geometric.utils as pyg_utils
from tqdm import tqdm
import scipy.stats as stats
from common import combined_syn
from common import feature_preprocess
from common import utils

from torch.utils.data import Dataset, DataLoader, DistributedSampler

def load_dataset(name):
    """ Load real-world datasets, available in PyTorch Geometric.

    Used as a helper for DiskDataSource.
    """
    task = "graph"
    if name == "enzymes":
        dataset = TUDataset(root="/tmp/ENZYMES", name="ENZYMES")
    elif name == "proteins":
        dataset = TUDataset(root="/tmp/PROTEINS", name="PROTEINS")
    elif name == "cox2":
        dataset = TUDataset(root="/tmp/cox2", name="COX2")
    elif name == "aids":
        dataset = TUDataset(root="/tmp/AIDS", name="AIDS")
    elif name == "reddit-binary":
        dataset = TUDataset(root="/tmp/REDDIT-BINARY", name="REDDIT-BINARY")
    elif name == "imdb-binary":
        dataset = TUDataset(root="/tmp/IMDB-BINARY", name="IMDB-BINARY")
    elif name == "firstmm_db":
        dataset = TUDataset(root="/tmp/FIRSTMM_DB", name="FIRSTMM_DB")
    elif name == "dblp":
        dataset = TUDataset(root="/tmp/DBLP_v1", name="DBLP_v1")
    elif name == "ppi":
        dataset = PPI(root="/tmp/PPI")
    elif name == "qm9":
        dataset = QM9(root="/tmp/QM9")
    elif name == "atlas":
        dataset = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g)]
    if task == "graph":
        train_len = int(0.8 * len(dataset))
        train, test = [], []
        dataset = list(dataset)
        random.shuffle(dataset)
        has_name = hasattr(dataset[0], "name")
        for i, graph in tqdm(enumerate(dataset)):
            if not type(graph) == nx.Graph:
                if has_name: del graph.name
                graph = pyg_utils.to_networkx(graph).to_undirected()
            if i < train_len:
                train.append(graph)
            else:
                test.append(graph)
    return train, test, task


def sample_subgraph(g_obj, anchors=None, radius=2, hard_neg_idxs=None):
    """
    Sample a subgraph around an anchor node from a graph wrapped in DSGraph.

    Args:
        g_obj (DSGraph): Graph object containing a NetworkX graph (G).
        anchors (list[int], optional): List of anchor nodes per graph index.
        radius (int): Number of hops from the anchor node to include.
        hard_neg_idxs (set, optional): Set of graph indices to apply hard negative logic.

    Returns:
        Tuple[DSGraph, DSGraph]: (Original graph, sampled subgraph)
    """
    G = g_obj.G  # NetworkX graph inside DSGraph
    idx = G.graph.get("idx", 0)  # Unique index for the graph

    # Hard negative logic: return entire graph
    if hard_neg_idxs is not None and idx in hard_neg_idxs:
        subgraph = G.copy()
    else:
        # Choose anchor node
        if anchors is not None:
            anchor = anchors[idx]
        else:
            anchor = random.choice(list(G.nodes))

        # k-hop neighborhood around anchor
        nodes = nx.single_source_shortest_path_length(G, anchor, cutoff=radius).keys()
        subgraph = G.subgraph(nodes).copy()

    # Ensure node_feature exists for all nodes
    for v in subgraph.nodes:
        if "node_feature" not in subgraph.nodes[v]:
            subgraph.nodes[v]["node_feature"] = torch.ones(1)

    return g_obj, DSGraph(subgraph)

class DataSource:
    def gen_batch(batch_target, batch_neg_target, batch_neg_query, train):
        raise NotImplementedError


SEMANTIC_PRESETS = {
    "biology": {
        "node_label_weights": {
            "high": [("TF", 0.60), ("Gene", 0.25), ("Enzyme", 0.15)],
            "mid": [("Gene", 0.45), ("mRNA", 0.35), ("Enzyme", 0.20)],
            "low": [("Metabolite", 0.60), ("mRNA", 0.25), ("Gene", 0.15)],
        },
        "edge_types": ["regulates", "transcribes", "translates", "catalyzes"],
    },
    "ecommerce": {
        "node_label_weights": {
            "high": [("product", 0.55), ("category", 0.30), ("brand", 0.15)],
            "mid": [("customer", 0.45), ("product", 0.35), ("brand", 0.20)],
            "low": [("review", 0.60), ("customer", 0.25), ("brand", 0.15)],
        },
        "edge_types": ["viewed", "purchased", "belongs_to", "made_by", "wrote"],
    },
    "social": {
        "node_label_weights": {
            "high": [("influencer", 0.60), ("community", 0.25), ("user", 0.15)],
            "mid": [("user", 0.55), ("creator", 0.25), ("community", 0.20)],
            "low": [("new_user", 0.60), ("bot", 0.25), ("user", 0.15)],
        },
        "edge_types": ["follows", "mentions", "replies_to", "belongs_to"],
    },
}

UNIVERSAL_NODE_LABEL_WEIGHTS = {
    "high": [
        ("Hub Entity", 0.18), ("Gene", 0.12), ("User", 0.12),
        ("Product", 0.12), ("Organization", 0.10), ("Protein", 0.10),
        ("Category", 0.09), ("Compound", 0.09), ("Topic", 0.08),
    ],
    "mid": [
        ("Entity", 0.16), ("Document", 0.12), ("Disease", 0.10),
        ("Book", 0.10), ("Music", 0.08), ("Pathway", 0.08),
        ("Customer", 0.08), ("Creator", 0.08), ("Anatomy", 0.07),
        ("Location", 0.07), ("Event", 0.06),
    ],
    "low": [
        ("Leaf Entity", 0.16), ("Review", 0.12), ("Side Effect", 0.10),
        ("Symptom", 0.10), ("Metabolite", 0.09), ("Tag", 0.09),
        ("Comment", 0.08), ("Image", 0.07), ("Article", 0.07),
        ("Video", 0.06), ("Record", 0.06),
    ],
}

# R-GCN relation ids are numeric slots. This universal preset deliberately
# exercises the full default 1..63 relation range so a reusable checkpoint does
# not leave most relation transforms untrained.
SEMANTIC_PRESETS["universal"] = {
    "node_label_weights": UNIVERSAL_NODE_LABEL_WEIGHTS,
    "edge_types": ["relation_{:02d}".format(i) for i in range(1, 64)],
}


class OTFSemanticSynDataSource(DataSource):
    """On-the-fly synthetic datasource with semantic labels and label negatives."""
    def __init__(self, max_size=29, min_size=5, node_anchored=False,
                 semantic_preset="biology", label_neg_ratio=0.5,
                 label_noise=0.05, hard_negative_ratio=0.5, seed=42,
                 semantic_mix_presets=None, semantic_mix_weights=None):
        self.max_size = max_size
        self.min_size = min_size
        self.node_anchored = node_anchored
        self.semantic_preset = semantic_preset if semantic_preset in SEMANTIC_PRESETS or semantic_preset == "mixed" else "biology"
        self.semantic_mix_presets = self._resolve_mix_presets(semantic_mix_presets)
        self.semantic_mix_weights = self._resolve_mix_weights(semantic_mix_weights, self.semantic_mix_presets)
        self.label_neg_ratio = float(max(0.0, min(1.0, label_neg_ratio)))
        self.hard_negative_ratio = float(max(0.0, min(1.0, hard_negative_ratio)))
        self.label_noise = float(max(0.0, min(0.5, label_noise)))
        self.generator = combined_syn.get_generator(np.arange(
            self.min_size + 1, self.max_size + 1))
        self.rng = random.Random(seed)

    def _resolve_mix_presets(self, semantic_mix_presets):
        if semantic_mix_presets is None:
            return ["biology", "ecommerce", "social"]
        if isinstance(semantic_mix_presets, str):
            items = [x.strip() for x in semantic_mix_presets.split(",") if x.strip()]
        else:
            items = list(semantic_mix_presets)
        items = [x for x in items if x in SEMANTIC_PRESETS]
        return items if items else ["biology", "ecommerce", "social"]

    def _resolve_mix_weights(self, semantic_mix_weights, presets):
        if semantic_mix_weights is None:
            return [1.0 / len(presets)] * len(presets)
        if isinstance(semantic_mix_weights, str):
            raw = [x.strip() for x in semantic_mix_weights.split(",") if x.strip()]
            try:
                weights = [float(x) for x in raw]
            except ValueError:
                return [1.0 / len(presets)] * len(presets)
        else:
            weights = [float(x) for x in semantic_mix_weights]
        if len(weights) != len(presets) or sum(weights) <= 0:
            return [1.0 / len(presets)] * len(presets)
        s = float(sum(weights))
        return [w / s for w in weights]

    def gen_data_loaders(self, size, batch_size, train=True,
                         use_distributed_sampling=False):
        n_batches = max(1, size // batch_size)
        return [[batch_size] * n_batches for _ in range(3)]

    def _sample_by_weights(self, weighted_values):
        values, weights = zip(*weighted_values)
        return self.rng.choices(values, weights=weights, k=1)[0]

    def _label_bucket(self, degree, q1, q2):
        if degree >= q2:
            return "high"
        if degree <= q1:
            return "low"
        return "mid"

    def _stable_pair_hash(self, src_label, dst_label):
        key = "{}->{}".format(src_label, dst_label)
        return sum((i + 1) * ord(c) for i, c in enumerate(key))

    def _choose_active_preset(self):
        if self.semantic_preset == "mixed":
            return self.rng.choices(self.semantic_mix_presets, weights=self.semantic_mix_weights, k=1)[0]
        return self.semantic_preset

    def _annotate_graph(self, graph):
        g = graph.copy()
        preset_name = self._choose_active_preset()
        preset = SEMANTIC_PRESETS[preset_name]
        edge_types = preset["edge_types"]

        if len(g.nodes()) == 0:
            return g
        degrees = np.array([d for _, d in g.degree()], dtype=float)
        q1 = float(np.percentile(degrees, 35))
        q2 = float(np.percentile(degrees, 75))

        for n in g.nodes():
            bucket = self._label_bucket(g.degree[n], q1, q2)
            label = self._sample_by_weights(preset["node_label_weights"][bucket])
            if self.rng.random() < self.label_noise:
                label = None  # explicit unknown label path
            g.nodes[n]["label"] = label

        for u, v in g.edges():
            src_label = g.nodes[u].get("label")
            dst_label = g.nodes[v].get("label")
            if src_label is None or dst_label is None:
                edge_type = "unknown"
                edge_type_id = 0
            else:
                idx = self._stable_pair_hash(src_label, dst_label) % len(edge_types)
                edge_type = edge_types[idx]
                if self.rng.random() < self.label_noise:
                    edge_type = self.rng.choice(edge_types)
                edge_type_id = idx + 1
            g.edges[u, v]["edge_type"] = edge_type
            # Keep numeric edge type for DeepSNAP compatibility.
            g.edges[u, v]["type"] = float(edge_type_id)
            g.edges[u, v]["type_id"] = int(edge_type_id)
        g.graph["semantic_preset"] = preset_name
        return g

    def _sample_connected_subgraph(self, graph, size, max_tries=12):
        nodes_all = list(graph.nodes())
        if len(nodes_all) <= size:
            return graph.copy()
        for _ in range(max_tries):
            start = self.rng.choice(nodes_all)
            visited = {start}
            frontier = [start]
            while frontier and len(visited) < size:
                curr = frontier.pop(0)
                neighs = [x for x in graph.neighbors(curr) if x not in visited]
                self.rng.shuffle(neighs)
                for nxt in neighs:
                    if len(visited) >= size:
                        break
                    visited.add(nxt)
                    frontier.append(nxt)
            sub = graph.subgraph(list(visited)).copy()
            if sub.number_of_edges() > 0:
                return sub
        # fallback
        sample_nodes = self.rng.sample(nodes_all, min(size, len(nodes_all)))
        return graph.subgraph(sample_nodes).copy()

    def _set_anchor(self, graph, anchor):
        for n in graph.nodes():
            graph.nodes[n]["node_feature"] = torch.tensor(
                [1.0 if n == anchor else 0.0], dtype=torch.float)
        return graph

    def _sample_positive_pair(self):
        for _ in range(30):
            size = self.rng.randint(self.min_size + 1, self.max_size)
            graph = self._annotate_graph(self.generator.generate(size=size))
            size_a = self.rng.randint(self.min_size + 1, min(self.max_size, len(graph)))
            sub_a = self._sample_connected_subgraph(graph, size_a)
            if len(sub_a) <= self.min_size:
                continue
            size_b = self.rng.randint(self.min_size, len(sub_a) - 1)
            sub_b = self._sample_connected_subgraph(sub_a, size_b)
            if sub_a.number_of_edges() == 0 or sub_b.number_of_edges() == 0:
                continue
            if self.node_anchored:
                anchor_a = self.rng.choice(list(sub_a.nodes()))
                anchor_b = anchor_a if anchor_a in sub_b.nodes() else self.rng.choice(list(sub_b.nodes()))
                self._set_anchor(sub_a, anchor_a)
                self._set_anchor(sub_b, anchor_b)
            return sub_a, sub_b
        raise RuntimeError("Failed to generate semantic positive pair")

    def _sample_structural_negative(self):
        for _ in range(30):
            size_a = self.rng.randint(self.min_size + 1, self.max_size)
            size_b = self.rng.randint(self.min_size, self.max_size - 1)
            g1 = self._annotate_graph(self.generator.generate(size=size_a))
            g2 = self._annotate_graph(self.generator.generate(size=size_b + 1))
            sub_a = self._sample_connected_subgraph(g1, min(size_a, len(g1)))
            sub_b = self._sample_connected_subgraph(g2, min(size_b, len(g2)))
            if sub_a.number_of_edges() == 0 or sub_b.number_of_edges() == 0:
                continue
            # Cheap structural mismatch check before expensive isomorphism.
            if (sub_a.number_of_nodes() != sub_b.number_of_nodes() or
                    sub_a.number_of_edges() != sub_b.number_of_edges()):
                if self.node_anchored:
                    self._set_anchor(sub_a, self.rng.choice(list(sub_a.nodes())))
                    self._set_anchor(sub_b, self.rng.choice(list(sub_b.nodes())))
                return sub_a, sub_b
            if not nx.is_isomorphic(sub_a, sub_b):
                if self.node_anchored:
                    self._set_anchor(sub_a, self.rng.choice(list(sub_a.nodes())))
                    self._set_anchor(sub_b, self.rng.choice(list(sub_b.nodes())))
                return sub_a, sub_b
        raise RuntimeError("Failed to generate semantic structural negative")

    def _node_match_semantic(self, a, b):
        return str(a.get("label", "unknown")) == str(b.get("label", "unknown"))

    def _edge_match_semantic(self, a, b):
        return int(a.get("type_id", int(a.get("type", 0)))) == int(
            b.get("type_id", int(b.get("type", 0)))
        )

    def _is_semantic_subgraph(self, target_graph, query_graph):
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            target_graph,
            query_graph,
            node_match=self._node_match_semantic,
            edge_match=self._edge_match_semantic,
        )
        return matcher.subgraph_is_isomorphic()

    def _make_hard_negative_from_positive(self, target_graph, query_graph):
        """
        Create near-miss semantic negatives by small structural perturbation of query.
        """
        for _ in range(8):
            cand = query_graph.copy()
            if cand.number_of_nodes() < 3:
                return None
            # Randomly add or remove one edge.
            if self.rng.random() < 0.5 and cand.number_of_edges() > 1:
                e = self.rng.choice(list(cand.edges()))
                cand.remove_edge(*e)
                if not nx.is_connected(cand):
                    continue
            else:
                non_edges = list(nx.non_edges(cand))
                if not non_edges:
                    continue
                u, v = self.rng.choice(non_edges)
                cand.add_edge(u, v)
                # inherit semantic type from preset deterministically
                cand.edges[u, v]["type"] = 1.0
                cand.edges[u, v]["type_id"] = 1
                cand.edges[u, v]["edge_type"] = "hard_neg"
            if not self._is_semantic_subgraph(target_graph, cand):
                return cand
        return None

    def _corrupt_query_labels(self, query_graph):
        corrupted = query_graph.copy()
        nodes = list(corrupted.nodes())
        if len(nodes) > 1:
            node_labels = [corrupted.nodes[n].get("label") for n in nodes]
            original = list(node_labels)
            self.rng.shuffle(node_labels)
            if node_labels == original:
                node_labels = node_labels[1:] + node_labels[:1]
            for n, lbl in zip(nodes, node_labels):
                corrupted.nodes[n]["label"] = lbl
        edges = list(corrupted.edges())
        if len(edges) > 1:
            edge_types = [corrupted.edges[e].get("type", 0.0) for e in edges]
            original_edge_types = list(edge_types)
            self.rng.shuffle(edge_types)
            if edge_types == original_edge_types:
                edge_types = edge_types[1:] + edge_types[:1]
            for e, t in zip(edges, edge_types):
                corrupted.edges[e]["type"] = float(t)
                corrupted.edges[e]["type_id"] = int(t)
        return corrupted

    def gen_batch(self, a, b, c, train):
        batch_size = int(a)
        n_pos = batch_size // 2
        n_neg = batch_size // 2
        n_label_neg = int(round(n_neg * self.label_neg_ratio))
        n_struct_neg = n_neg - n_label_neg
        n_hard_struct_neg = int(round(n_struct_neg * self.hard_negative_ratio))
        n_easy_struct_neg = n_struct_neg - n_hard_struct_neg

        pos_a, pos_b = [], []
        for _ in range(n_pos):
            g_a, g_b = self._sample_positive_pair()
            pos_a.append(g_a)
            pos_b.append(g_b)

        neg_a, neg_b = [], []
        for _ in range(n_easy_struct_neg):
            g_a, g_b = self._sample_structural_negative()
            neg_a.append(g_a)
            neg_b.append(g_b)

        for _ in range(n_hard_struct_neg):
            idx = self.rng.randrange(len(pos_a))
            g_a = pos_a[idx].copy()
            hard_q = self._make_hard_negative_from_positive(g_a, pos_b[idx])
            if hard_q is None:
                g_a, hard_q = self._sample_structural_negative()
            if self.node_anchored:
                self._set_anchor(g_a, self.rng.choice(list(g_a.nodes())))
                self._set_anchor(hard_q, self.rng.choice(list(hard_q.nodes())))
            neg_a.append(g_a)
            neg_b.append(hard_q)

        for _ in range(n_label_neg):
            idx = self.rng.randrange(len(pos_a))
            g_a = pos_a[idx].copy()
            g_b = self._corrupt_query_labels(pos_b[idx])
            neg_a.append(g_a)
            neg_b.append(g_b)

        pos_a = utils.batch_nx_graphs(pos_a)
        pos_b = utils.batch_nx_graphs(pos_b)
        neg_a = utils.batch_nx_graphs(neg_a)
        neg_b = utils.batch_nx_graphs(neg_b)
        return pos_a, pos_b, neg_a, neg_b


class CustomGraphDataset:
    def __init__(self, graph_pkl_path, node_anchored=False, num_queries=32, subgraph_hops=1, min_size=5, max_size=29):
        self.graph_pkl_path = graph_pkl_path
        self.node_anchored = node_anchored
        self.num_queries = num_queries
        self.subgraph_hops = subgraph_hops
        self.min_size = min_size
        self.max_size = max_size
        self.query_size = 5

        if isinstance(graph_pkl_path, nx.Graph):
            self.full_graph = DSGraph(graph_pkl_path)
            self.graph = self.full_graph.G
        else:
            self.raw_data = self._load_graph()
            self.full_graph = self._build_graph()
            self.graph = self.full_graph.G

    def _load_graph(self):
        with open(self.graph_pkl_path, 'rb') as f:
            return pickle.load(f)

    def _build_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.raw_data['nodes'])

        cleaned_edges = []
        for edge in self.raw_data['edges']:
            if len(edge) == 3:
                u, v, attr = edge
                cleaned_attr = {k: v for k, v in attr.items() if isinstance(v, (int, float))}
                cleaned_edges.append((u, v, cleaned_attr))
            else:
                cleaned_edges.append(edge)
        G.add_edges_from(cleaned_edges)

        for node in G.nodes():
            if 'node_feature' not in G.nodes[node]:
                G.nodes[node]['node_feature'] = torch.tensor([1.0], dtype=torch.float)

        return DSGraph(G)

    def _bfs_sample_subgraph(self, graph, size, max_tries=10):
        """
        Sample a connected subgraph of given size using BFS .
        """
        for _ in range(max_tries):
            start_node = random.choice(list(graph.nodes))
            visited = {start_node}
            queue = [start_node]
            while queue and len(visited) < size:
                current = queue.pop(0)
                neighbors = list(set(graph.neighbors(current)) - visited)
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    if len(visited) >= size:
                        break
                    visited.add(neighbor)
                    queue.append(neighbor)
            subg = graph.subgraph(visited).copy()
           
            if subg.number_of_edges() > 0 and nx.is_connected(subg):
                return subg
        # fallback: largest connected component
        if subg.number_of_edges() == 0 and subg.number_of_nodes() > 1:
            components = list(nx.connected_components(subg))
            if components:
                largest_cc = max(components, key=len)
                subg = subg.subgraph(largest_cc).copy()
        return subg

    def _add_anchor(self, g, anchor=None):
        if anchor is None:
            anchor = random.choice(list(g.nodes))
        for v in g.nodes:
            g.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v else torch.zeros(1))
        return g

    def gen_batch(self, batch_size, *args, train=True, **kwargs):

        """
        Generate a batch of positive and negative graph pairs
        Returns:
            pos_a: Batch of anchor graphs (DSGraph Batch)
            pos_b: Batch of positive graphs (DSGraph Batch)
            neg_a: Batch of negative anchor graphs (DSGraph Batch)
            neg_b: Batch of negative graphs (DSGraph Batch)
        """
        pos_a, pos_b, neg_a, neg_b = [], [], [], []
        tries = 0
        max_tries = 20  # Prevent infinite loops
        
        # Generate positive pairs (nested subgraphs from same graph)
        while len(pos_a) < batch_size // 2 and tries < max_tries:
            size_a = random.randint(self.min_size + 1, self.max_size)
            size_b = random.randint(self.min_size, size_a - 1)
            
            # Sample subgraphs
            sub_a = self._bfs_sample_subgraph(self.graph, size_a)
            sub_b = self._bfs_sample_subgraph(sub_a, size_b)
            
            # Validate subgraphs
            if sub_a.number_of_edges() > 0 and sub_b.number_of_edges() > 0:
                # Handle node anchoring if needed
                if self.node_anchored:
                    anchor = random.choice(list(sub_a.nodes))
                    sub_a = self._add_anchor(sub_a, anchor)
                    sub_b = self._add_anchor(sub_b, anchor if anchor in sub_b.nodes else None)
                
                # Convert to DSGraph before adding to batch
                pos_a.append(DSGraph(sub_a))
                pos_b.append(DSGraph(sub_b))
            tries += 1
        
        tries = 0
        # Generate negative pairs (subgraphs from different parts of graph)
        while len(neg_a) < batch_size // 2 and tries < max_tries:
            size_a = random.randint(self.min_size, self.max_size)
            size_b = random.randint(self.min_size, self.max_size)
            
            # Sample independent subgraphs
            sub_a = self._bfs_sample_subgraph(self.graph, size_a)
            sub_b = self._bfs_sample_subgraph(self.graph, size_b)
            
            # Validate and ensure they're different
            if (sub_a.number_of_edges() > 0 and 
                sub_b.number_of_edges() > 0 and 
                not nx.is_isomorphic(sub_a, sub_b)):
                
                if self.node_anchored:
                    sub_a = self._add_anchor(sub_a)
                    sub_b = self._add_anchor(sub_b)
                
                neg_a.append(DSGraph(sub_a))
                neg_b.append(DSGraph(sub_b))
            tries += 1
        
        # Verify we got enough samples
        if not (pos_a and pos_b and neg_a and neg_b):
            raise RuntimeError(
                f"Failed to generate valid batch after {max_tries} tries. "
                f"Got {len(pos_a)}/{batch_size//2} positive and "
                f"{len(neg_a)}/{batch_size//2} negative pairs."
            )
        
        # Convert to DeepSNAP batches
        def _make_batch(graph_list):
            return Batch.from_data_list(graph_list)
        
        return _make_batch(pos_a), _make_batch(pos_b), _make_batch(neg_a), _make_batch(neg_b)
    def gen_data_loaders(self, size, batch_size, train=True, use_distributed_sampling=False):
        """
        Returns three loaders (lists of batch sizes) for compatibility with the training loop.
        This matches the interface of DiskDataSource and OTFSynDataSource.
        """
        num_batches = size // batch_size
        # Each loader is a list of batch sizes, as expected by the training loop.
        loaders = [[batch_size] * num_batches for _ in range(3)]
        return loaders    
    
class OTFSynDataSource(DataSource):
    """ On-the-fly generated synthetic data for training the subgraph model.

    At every iteration, new batch of graphs (positive and negative) are generated
    with a pre-defined generator (see combined_syn.py).

    DeepSNAP transforms are used to generate the positive and negative examples.
    """
    def __init__(self, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        self.closed = False
        self.max_size = max_size
        self.min_size = min_size
        self.node_anchored = node_anchored
        self.generator = combined_syn.get_generator(np.arange(
            self.min_size + 1, self.max_size + 1))

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = []
        for i in range(2):
            dataset = combined_syn.get_dataset("graph", size // 2,
                np.arange(self.min_size + 1, self.max_size + 1))
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank()) if \
                    use_distributed_sampling else None
            loaders.append(TorchDataLoader(dataset,
                collate_fn=Batch.collate([]), batch_size=batch_size // 2 if i
                == 0 else batch_size // 2,
                sampler=sampler, shuffle=False))
        loaders.append([None]*(size // batch_size))
        return loaders

    def gen_batch(self, batch_target, batch_neg_target, batch_neg_query,
        train):
        def sample_subgraph(graph, offset=0, use_precomp_sizes=False,
            filter_negs=False, supersample_small_graphs=False, neg_target=None,
            hard_neg_idxs=None):
            if neg_target is not None: graph_idx = graph.G.graph["idx"]
            use_hard_neg = (hard_neg_idxs is not None and graph.G.graph["idx"]
                in hard_neg_idxs)
            done = False
            n_tries = 0
            while not done:
                if use_precomp_sizes:
                    size = graph.G.graph["subgraph_size"]
                else:
                    if train and supersample_small_graphs:
                        sizes = np.arange(self.min_size + offset,
                            len(graph.G) + offset)
                        ps = (sizes - self.min_size + 2) ** (-1.1)
                        ps /= ps.sum()
                        size = stats.rv_discrete(values=(sizes, ps)).rvs()
                    else:
                        d = 1 if train else 0
                        size = random.randint(self.min_size + offset - d,
                            len(graph.G) - 1 + offset)
                start_node = random.choice(list(graph.G.nodes))
                neigh = [start_node]
                frontier = list(set(graph.G.neighbors(start_node)) - set(neigh))
                visited = set([start_node])
                while len(neigh) < size:
                    new_node = random.choice(list(frontier))
                    assert new_node not in neigh
                    neigh.append(new_node)
                    visited.add(new_node)
                    frontier += list(graph.G.neighbors(new_node))
                    frontier = [x for x in frontier if x not in visited]
                if self.node_anchored:
                    anchor = neigh[0]
                    for v in graph.G.nodes:
                        graph.G.nodes[v]["node_feature"] = (torch.ones(1) if
                            anchor == v else torch.zeros(1))
                        #print(v, graph.G.nodes[v]["node_feature"])
                neigh = graph.G.subgraph(neigh)
                if use_hard_neg and train:
                    neigh = neigh.copy()
                    if random.random() < 1.0 or not self.node_anchored: # add edges
                        non_edges = list(nx.non_edges(neigh))
                        if len(non_edges) > 0:
                            for u, v in random.sample(non_edges, random.randint(1,
                                min(len(non_edges), 5))):
                                neigh.add_edge(u, v)
                    else:                         # perturb anchor
                        anchor = random.choice(list(neigh.nodes))
                        for v in neigh.nodes:
                            neigh.nodes[v]["node_feature"] = (torch.ones(1) if
                                anchor == v else torch.zeros(1))

                if (filter_negs and train and len(neigh) <= 6 and neg_target is
                    not None):
                    matcher = nx.algorithms.isomorphism.GraphMatcher(
                        neg_target[graph_idx], neigh)
                    if not matcher.subgraph_is_isomorphic(): done = True
                else:
                    done = True

            return graph, DSGraph(neigh)

        augmenter = feature_preprocess.FeatureAugment()

        pos_target = batch_target
        pos_target, pos_query = pos_target.apply_transform_multi(sample_subgraph)
        neg_target = batch_neg_target
        # TODO: use hard negs
        hard_neg_idxs = set(random.sample(range(len(neg_target.G)),
            int(len(neg_target.G) * 1/2)))
        #hard_neg_idxs = set()
        batch_neg_query = Batch.from_data_list(
            [DSGraph(self.generator.generate(size=len(g))
                if i not in hard_neg_idxs else g)
                for i, g in enumerate(neg_target.G)])
        for i, g in enumerate(batch_neg_query.G):
            g.graph["idx"] = i
        _, neg_query = batch_neg_query.apply_transform_multi(sample_subgraph,
            hard_neg_idxs=hard_neg_idxs)
        if self.node_anchored:
            def add_anchor(g, anchors=None):
                if anchors is not None:
                    anchor = anchors[g.G.graph["idx"]]
                else:
                    anchor = random.choice(list(g.G.nodes))
                for v in g.G.nodes:
                    if "node_feature" not in g.G.nodes[v]:
                        g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                            else torch.zeros(1))
                return g
            neg_target = neg_target.apply_transform(add_anchor)
        pos_target = augmenter.augment(pos_target).to(utils.get_device())
        pos_query = augmenter.augment(pos_query).to(utils.get_device())
        neg_target = augmenter.augment(neg_target).to(utils.get_device())
        neg_query = augmenter.augment(neg_query).to(utils.get_device())
        #print(len(pos_target.G[0]), len(pos_query.G[0]))
        return pos_target, pos_query, neg_target, neg_query

class OTFSynImbalancedDataSource(OTFSynDataSource):
    """ Imbalanced on-the-fly synthetic data.

    Unlike the balanced dataset, this data source does not use 1:1 ratio for
    positive and negative examples. Instead, it randomly samples 2 graphs from
    the on-the-fly generator, and records the groundtruth label for the pair (subgraph or not).
    As a result, the data is imbalanced (subgraph relationships are rarer).
    This setting is a challenging model inference scenario.
    """
    def __init__(self, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        super().__init__(max_size=max_size, min_size=min_size,
            n_workers=n_workers, node_anchored=node_anchored)
        self.batch_idx = 0

    def gen_batch(self, graphs_a, graphs_b, _, train):
        def add_anchor(g):
            anchor = random.choice(list(g.G.nodes))
            for v in g.G.nodes:
                g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                    or not self.node_anchored else torch.zeros(1))
            return g
        pos_a, pos_b, neg_a, neg_b = [], [], [], []
        fn = "data/cache/imbalanced-{}-{}".format(str(self.node_anchored),
            self.batch_idx)
        if not os.path.exists(fn):
            graphs_a = graphs_a.apply_transform(add_anchor)
            graphs_b = graphs_b.apply_transform(add_anchor)
            for graph_a, graph_b in tqdm(list(zip(graphs_a.G, graphs_b.G))):
                matcher = nx.algorithms.isomorphism.GraphMatcher(graph_a, graph_b,
                    node_match=(lambda a, b: (a["node_feature"][0] > 0.5) ==
                    (b["node_feature"][0] > 0.5)) if self.node_anchored else None)
                if matcher.subgraph_is_isomorphic():
                    pos_a.append(graph_a)
                    pos_b.append(graph_b)
                else:
                    neg_a.append(graph_a)
                    neg_b.append(graph_b)
            if not os.path.exists("data/cache"):
                os.makedirs("data/cache")
            with open(fn, "wb") as f:
                pickle.dump((pos_a, pos_b, neg_a, neg_b), f)
            print("saved", fn)
        else:
            with open(fn, "rb") as f:
                print("loaded", fn)
                pos_a, pos_b, neg_a, neg_b = pickle.load(f)
        print(len(pos_a), len(neg_a))
        if pos_a:
            pos_a = utils.batch_nx_graphs(pos_a)
            pos_b = utils.batch_nx_graphs(pos_b)
        neg_a = utils.batch_nx_graphs(neg_a)
        neg_b = utils.batch_nx_graphs(neg_b)
        self.batch_idx += 1
        return pos_a, pos_b, neg_a, neg_b

class DiskDataSource(DataSource):
    """ Uses a set of graphs saved in a dataset file to train the subgraph model.

    At every iteration, new batch of graphs (positive and negative) are generated
    by sampling subgraphs from a given dataset.

    See the load_dataset function for supported datasets.
    """
    def __init__(self, dataset_name, node_anchored=False, min_size=5,
        max_size=29):
        self.node_anchored = node_anchored
        self.dataset = load_dataset(dataset_name)
        self.min_size = min_size
        self.max_size = max_size

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = [[batch_size]*(size // batch_size) for i in range(3)]
        return loaders

    def gen_batch(self, a, b, c, train, max_size=15, min_size=5, seed=None,
        filter_negs=False, sample_method="tree-pair"):
        batch_size = a
        train_set, test_set, task = self.dataset
        graphs = train_set if train else test_set
        if seed is not None:
            random.seed(seed)

        pos_a, pos_b = [], []
        pos_a_anchors, pos_b_anchors = [], []
        for i in range(batch_size // 2):
            if sample_method == "tree-pair":
                size = random.randint(min_size+1, max_size)
                graph, a = utils.sample_neigh(graphs, size)
                b = a[:random.randint(min_size, len(a) - 1)]
            elif sample_method == "subgraph-tree":
                graph = None
                while graph is None or len(graph) < min_size + 1:
                    graph = random.choice(graphs)
                a = graph.nodes
                _, b = utils.sample_neigh([graph], random.randint(min_size,
                    len(graph) - 1))
            if self.node_anchored:
                anchor = list(graph.nodes)[0]
                pos_a_anchors.append(anchor)
                pos_b_anchors.append(anchor)
            neigh_a, neigh_b = graph.subgraph(a), graph.subgraph(b)
            pos_a.append(neigh_a)
            pos_b.append(neigh_b)

        neg_a, neg_b = [], []
        neg_a_anchors, neg_b_anchors = [], []
        while len(neg_a) < batch_size // 2:
            if sample_method == "tree-pair":
                size = random.randint(min_size+1, max_size)
                graph_a, a = utils.sample_neigh(graphs, size)
                graph_b, b = utils.sample_neigh(graphs, random.randint(min_size,
                    size - 1))
            elif sample_method == "subgraph-tree":
                graph_a = None
                while graph_a is None or len(graph_a) < min_size + 1:
                    graph_a = random.choice(graphs)
                a = graph_a.nodes
                graph_b, b = utils.sample_neigh(graphs, random.randint(min_size,
                    len(graph_a) - 1))
            if self.node_anchored:
                neg_a_anchors.append(list(graph_a.nodes)[0])
                neg_b_anchors.append(list(graph_b.nodes)[0])
            neigh_a, neigh_b = graph_a.subgraph(a), graph_b.subgraph(b)
            if filter_negs:
                matcher = nx.algorithms.isomorphism.GraphMatcher(neigh_a, neigh_b)
                if matcher.subgraph_is_isomorphic(): # a <= b (b is subgraph of a)
                    continue
            neg_a.append(neigh_a)
            neg_b.append(neigh_b)

        pos_a = utils.batch_nx_graphs(pos_a, anchors=pos_a_anchors if
            self.node_anchored else None)
        pos_b = utils.batch_nx_graphs(pos_b, anchors=pos_b_anchors if
            self.node_anchored else None)
        neg_a = utils.batch_nx_graphs(neg_a, anchors=neg_a_anchors if
            self.node_anchored else None)
        neg_b = utils.batch_nx_graphs(neg_b, anchors=neg_b_anchors if
            self.node_anchored else None)
        return pos_a, pos_b, neg_a, neg_b

class DiskImbalancedDataSource(OTFSynDataSource):
    """ Imbalanced on-the-fly real data.

    Unlike the balanced dataset, this data source does not use 1:1 ratio for
    positive and negative examples. Instead, it randomly samples 2 graphs from
    the on-the-fly generator, and records the groundtruth label for the pair (subgraph or not).
    As a result, the data is imbalanced (subgraph relationships are rarer).
    This setting is a challenging model inference scenario.
    """
    def __init__(self, dataset_name, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        super().__init__(max_size=max_size, min_size=min_size,
            n_workers=n_workers, node_anchored=node_anchored)
        self.batch_idx = 0
        self.dataset = load_dataset(dataset_name)
        self.train_set, self.test_set, _ = self.dataset
        self.dataset_name = dataset_name

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = []
        for i in range(2):
            neighs = []
            for j in range(size // 2):
                graph, neigh = utils.sample_neigh(self.train_set if train else
                    self.test_set, random.randint(self.min_size, self.max_size))
                neighs.append(graph.subgraph(neigh))
            dataset = GraphDataset(neighs)
            loaders.append(TorchDataLoader(dataset,
                collate_fn=Batch.collate([]), batch_size=batch_size // 2 if i
                == 0 else batch_size // 2,
                sampler=None, shuffle=False))
        loaders.append([None]*(size // batch_size))
        return loaders

    def gen_batch(self, graphs_a, graphs_b, _, train):
        def add_anchor(g):
            anchor = random.choice(list(g.G.nodes))
            for v in g.G.nodes:
                g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                    or not self.node_anchored else torch.zeros(1))
            return g
        pos_a, pos_b, neg_a, neg_b = [], [], [], []
        fn = "data/cache/imbalanced-{}-{}-{}".format(self.dataset_name.lower(),
            str(self.node_anchored), self.batch_idx)
        if not os.path.exists(fn):
            graphs_a = graphs_a.apply_transform(add_anchor)
            graphs_b = graphs_b.apply_transform(add_anchor)
            for graph_a, graph_b in tqdm(list(zip(graphs_a.G, graphs_b.G))):
                matcher = nx.algorithms.isomorphism.GraphMatcher(graph_a, graph_b,
                    node_match=(lambda a, b: (a["node_feature"][0] > 0.5) ==
                    (b["node_feature"][0] > 0.5)) if self.node_anchored else None)
                if matcher.subgraph_is_isomorphic():
                    pos_a.append(graph_a)
                    pos_b.append(graph_b)
                else:
                    neg_a.append(graph_a)
                    neg_b.append(graph_b)
            if not os.path.exists("data/cache"):
                os.makedirs("data/cache")
            with open(fn, "wb") as f:
                pickle.dump((pos_a, pos_b, neg_a, neg_b), f)
            print("saved", fn)
        else:
            with open(fn, "rb") as f:
                print("loaded", fn)
                pos_a, pos_b, neg_a, neg_b = pickle.load(f)
        print(len(pos_a), len(neg_a))
        if pos_a:
            pos_a = utils.batch_nx_graphs(pos_a)
            pos_b = utils.batch_nx_graphs(pos_b)
        neg_a = utils.batch_nx_graphs(neg_a)
        neg_b = utils.batch_nx_graphs(neg_b)
        self.batch_idx += 1
        return pos_a, pos_b, neg_a, neg_b
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14})
    for name in ["enzymes", "reddit-binary", "cox2"]:
        data_source = DiskDataSource(name)
        train, test, _ = data_source.dataset
        i = 11
        neighs = [utils.sample_neigh(train, i) for j in range(10000)]
        clustering = [nx.average_clustering(graph.subgraph(nodes)) for graph,
            nodes in neighs]
        path_length = [nx.average_shortest_path_length(graph.subgraph(nodes))
            for graph, nodes in neighs]
        #plt.subplot(1, 2, i-9)
        plt.scatter(clustering, path_length, s=10, label=name)
    plt.legend()
    plt.savefig("plots/clustering-vs-path-length.png")
