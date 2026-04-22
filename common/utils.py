from collections import defaultdict, Counter
import json
import hashlib
import os

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
import torch
import torch.optim as optim
import torch_geometric.utils as pyg_utils
from torch_geometric.data import DataLoader
import networkx as nx
import numpy as np
import random
import scipy.stats as stats
from tqdm import tqdm
import warnings

from common import feature_preprocess
from common import label_encoder

_node_label_vocab = None
_edge_type_vocab = None
_vocab_version = None
MODEL_METADATA_VERSION = "v1"
_semantic_hash_mode = "categorical"
_semantic_hash_label_encoder = None
_semantic_hash_top_k = 4
_semantic_hash_cache = {}


def set_label_vocabs(node_vocab=None, edge_vocab=None, vocab_version=None):
    global _node_label_vocab, _edge_type_vocab, _vocab_version
    _node_label_vocab = node_vocab
    _edge_type_vocab = edge_vocab
    _vocab_version = vocab_version


def configure_semantic_hash(
    semantic_mode="categorical",
    label_encoder_backend="auto",
    label_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    label_encoder_cache_dir=None,
    text_encoder_dim=384,
    text_hash_top_k=4,
):
    global _semantic_hash_mode, _semantic_hash_label_encoder
    global _semantic_hash_top_k, _semantic_hash_cache

    _semantic_hash_mode = semantic_mode or "categorical"
    _semantic_hash_top_k = max(1, int(text_hash_top_k))
    _semantic_hash_cache = {}

    if _semantic_hash_mode == "hybrid_text":
        _semantic_hash_label_encoder = label_encoder.UniversalLabelEncoder(
            model_name=label_encoder_name,
            cache_dir=label_encoder_cache_dir,
            backend=label_encoder_backend,
            embedding_dim=text_encoder_dim,
        )
    else:
        _semantic_hash_label_encoder = None


def build_model_metadata(args):
    keys = [
        "dataset",
        "semantic_preset",
        "semantic_mix_presets",
        "semantic_mix_weights",
        "val_semantic_preset",
        "semantic_mode",
        "use_label_features",
        "label_feature_dim",
        "label_encoder_backend",
        "label_encoder_name",
        "label_encoder_cache_dir",
        "text_encoder_dim",
        "text_label_dim",
        "encoder_type",
        "num_relations",
        "num_bases",
        "rel_reg_lambda",
        "conv_type",
        "hidden_dim",
        "n_layers",
        "skip",
        "dropout",
        "node_anchored",
        "margin",
        "order_threshold_mode",
        "order_margin_factor",
        "seed",
    ]
    meta = {"metadata_version": MODEL_METADATA_VERSION}
    for key in keys:
        if hasattr(args, key):
            meta[key] = getattr(args, key)
    if hasattr(args, "vocab_version"):
        meta["vocab_version"] = getattr(args, "vocab_version")
    return meta


def model_metadata_path(model_path):
    return model_path + ".meta.json"


def save_model_metadata(args, model_path):
    meta_path = model_metadata_path(model_path)
    meta_dir = os.path.dirname(meta_path)
    if meta_dir and not os.path.exists(meta_dir):
        os.makedirs(meta_dir)
    with open(meta_path, "w") as f:
        json.dump(build_model_metadata(args), f, indent=2, sort_keys=True)


def load_model_metadata(model_path, required=False):
    meta_path = model_metadata_path(model_path)
    if not os.path.exists(meta_path):
        if required:
            raise FileNotFoundError("Missing model metadata: {}".format(meta_path))
        return None
    with open(meta_path, "r") as f:
        return json.load(f)


def apply_model_metadata_to_args(args, parser=None, keys=None, strict_conflicts=True):
    if not getattr(args, "model_path", None):
        return None, {}, {}

    metadata = load_model_metadata(args.model_path, required=False)
    if not metadata:
        return None, {}, {}

    candidate_keys = keys or [k for k in metadata.keys() if hasattr(args, k)]
    applied = {}
    conflicts = {}

    for key in candidate_keys:
        if key not in metadata or not hasattr(args, key):
            continue
        meta_value = metadata[key]
        current_value = getattr(args, key)
        default_value = parser.get_default(key) if parser is not None else None

        if current_value == meta_value:
            continue

        if parser is not None and current_value == default_value:
            setattr(args, key, meta_value)
            applied[key] = meta_value
        else:
            conflicts[key] = {
                "runtime": current_value,
                "checkpoint": meta_value,
            }

    if conflicts and strict_conflicts:
        lines = ["Model metadata conflicts with runtime args:"]
        for key, values in sorted(conflicts.items()):
            lines.append(
                "  {}: runtime={} checkpoint={}".format(
                    key, values["runtime"], values["checkpoint"])
            )
        raise ValueError("\n".join(lines))

    return metadata, applied, conflicts


def sample_neigh(graphs, size, graph_type):
    ps = np.array([len(g) for g in graphs], dtype=float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))
    while True:
        idx = dist.rvs()
        #graph = random.choice(graphs)
        graph = graphs[idx]
        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]
        is_directed = graph.is_directed() if hasattr(graph, "is_directed") else False
        if graph_type == "undirected" or not is_directed:
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        elif graph_type == "directed" and is_directed:
            frontier = list((set(graph.successors(start_node)) | set(graph.predecessors(start_node))) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            #new_node = max(sorted(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            if graph_type == "undirected" or not is_directed:
                frontier += list(graph.neighbors(new_node))
            elif graph_type == "directed" and is_directed:
                frontier += list(graph.successors(new_node))
                frontier += list(graph.predecessors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return graph, neigh

cached_masks = None
def vec_hash(v):
    global cached_masks
    if cached_masks is None:
        random.seed(2019)
        cached_masks = [random.getrandbits(32) for i in range(len(v))]
    #v = [hash(tuple(v)) ^ mask for mask in cached_masks]
    v = [hash(v[i]) ^ mask for i, mask in enumerate(cached_masks)]
    #v = [np.sum(v) for mask in cached_masks]
    return v


def node_semantic_label(node_data):
    if node_data is None:
        return None
    if "semantic_label" in node_data:
        return node_data.get("semantic_label")
    return node_data.get("label", None)


def edge_semantic_label(edge_data):
    if edge_data is None:
        return None
    for key in ("type_str", "type", "relation", "edge_type", "label", "input_label"):
        value = edge_data.get(key)
        if value is not None:
            return value
    return None


def _text_bucket_signature(text):
    if _semantic_hash_label_encoder is None:
        return ()
    label_text = label_encoder._safe_label_text(text)
    cached = _semantic_hash_cache.get(label_text)
    if cached is not None:
        return cached

    vec = _semantic_hash_label_encoder.encode_many([text])[0].detach().cpu().numpy()
    if vec.size == 0:
        sig = ()
    else:
        k = min(_semantic_hash_top_k, vec.shape[0])
        if k <= 0:
            sig = ()
        else:
            order = np.argsort(np.abs(vec))[-k:]
            order = sorted(order.tolist(), key=lambda idx: (-abs(float(vec[idx])), int(idx)))
            sig = tuple((int(idx) << 1) | (1 if float(vec[idx]) >= 0 else 0) for idx in order)
    _semantic_hash_cache[label_text] = sig
    return sig


def _text_bucket_id(text):
    if _semantic_hash_mode != "hybrid_text":
        return 0
    sig = _text_bucket_signature(text)
    if not sig:
        return 0
    return _stable_int("textsig::{}".format(",".join(map(str, sig))))


def _node_semantic_mix(node_data):
    label_id = int(node_data.get("label_id", _stable_label_id(node_semantic_label(node_data))))
    base = (label_id << 1)
    if _semantic_hash_mode == "hybrid_text":
        text_bucket_id = int(node_data.get(
            "text_label_bucket_id",
            _text_bucket_id(node_semantic_label(node_data)),
        ))
        base ^= ((text_bucket_id << 3) | (text_bucket_id >> 2))
    return base

def wl_hash(g, dim=64, node_anchored=False):
    g = nx.convert_node_labels_to_integers(g)
    vecs = np.zeros((len(g), dim), dtype=int)
    for v in g.nodes:
        node_data = g.nodes[v]
        base = 0
        if node_anchored and node_data.get("anchor", 0) == 1:
            base ^= 1

        base ^= _node_semantic_mix(node_data)
        vecs[v] = base
    for i in range(len(g)):
        newvecs = np.zeros((len(g), dim), dtype=int)
        for n in g.nodes:
            # Direction-aware neighborhood aggregation for semantic hashing
            neigh_vec = vecs[n].copy()
            if g.is_directed():
                succ_nodes = list(g.successors(n))
                pred_nodes = list(g.predecessors(n))
                if succ_nodes:
                    neigh_vec += np.sum(vecs[succ_nodes], axis=0)
                if pred_nodes:
                    neigh_vec += 3 * np.sum(vecs[pred_nodes], axis=0)
                edge_sem = 0
                for nbr in succ_nodes:
                    edge_sem ^= _edge_semantic_mix(g.get_edge_data(n, nbr), forward=True)
                for nbr in pred_nodes:
                    edge_sem ^= _edge_semantic_mix(g.get_edge_data(n, nbr), forward=False)
                neigh_vec[0] ^= edge_sem
            else:
                neighbors = list(g.neighbors(n))
                if neighbors:
                    neigh_vec += np.sum(vecs[neighbors], axis=0)
                edge_sem = 0
                for nbr in neighbors:
                    edge_sem ^= _edge_semantic_mix(g.get_edge_data(n, nbr), forward=True)
                neigh_vec[0] ^= edge_sem
            newvecs[n] = vec_hash(neigh_vec)
        vecs = newvecs
    return tuple(np.sum(vecs, axis=0))


def _stable_int(text, max_value=2**31 - 1):
    """Deterministic integer id from text (stable across runs/processes)."""
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return value % max_value


def _stable_label_id(label):
    if _node_label_vocab is not None:
        return int(_node_label_vocab.get(str(label), _node_label_vocab.get("UNK", 0)))
    if label is None:
        return 0
    return _stable_int(f"node::{label}") + 1


def _stable_edge_type_id(edge_type):
    if _edge_type_vocab is not None:
        return int(_edge_type_vocab.get(str(edge_type), _edge_type_vocab.get("UNK", 0)))
    if edge_type is None:
        return 0
    return _stable_int(f"edge::{edge_type}") + 1


def _edge_semantic_mix(edge_data, forward):
    edge_data = edge_data or {}
    if "type_id" in edge_data:
        edge_type_id = int(edge_data["type_id"])
    else:
        edge_type = edge_semantic_label(edge_data) or "unknown"
        edge_type_id = _stable_edge_type_id(edge_type)
    if _semantic_hash_mode == "hybrid_text":
        edge_type_id ^= _text_bucket_id(edge_semantic_label(edge_data))
    direction_bias = 11 if forward else 19
    return (edge_type_id * 1315423911) ^ direction_bias

def gen_baseline_queries_rand_esu(queries, targets, node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    max_size = max(sizes.keys())
    all_subgraphs = defaultdict(lambda: defaultdict(list))
    total_n_max_subgraphs, total_n_subgraphs = 0, 0
    for target in tqdm(targets):
        subgraphs = enumerate_subgraph(target, k=max_size,
            progress_bar=len(targets) < 10, node_anchored=node_anchored)
        for (size, k), v in subgraphs.items():
            all_subgraphs[size][k] += v
            if size == max_size: total_n_max_subgraphs += len(v)
            total_n_subgraphs += len(v)
    print(total_n_subgraphs, "subgraphs explored")
    print(total_n_max_subgraphs, "max-size subgraphs explored")
    out = []
    for size, count in sizes.items():
        counts = all_subgraphs[size]
        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
            reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out

def enumerate_subgraph(G, k=3, progress_bar=False, node_anchored=False):
    ps = np.arange(1.0, 0.0, -1.0/(k+1)) ** 1.5
    #ps = [1.0]*(k+1)
    motif_counts = defaultdict(list)
    for node in tqdm(G.nodes) if progress_bar else G.nodes:
        sg = set()
        sg.add(node)
        v_ext = set()
        neighbors = [nbr for nbr in list(G[node].keys()) if nbr > node]
        n_frac = len(neighbors) * ps[1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
            else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            v_ext.add(nbr)
        extend_subgraph(G, k, sg, v_ext, node, motif_counts, ps, node_anchored)
    return motif_counts

def extend_subgraph(G, k, sg, v_ext, node_id, motif_counts, ps, node_anchored):
    # Base case
    sg_G = G.subgraph(sg)
    if node_anchored:
        sg_G = sg_G.copy()
        nx.set_node_attributes(sg_G, 0, name="anchor")
        sg_G.nodes[node_id]["anchor"] = 1

    motif_counts[len(sg), wl_hash(sg_G,
        node_anchored=node_anchored)].append(sg_G)
    if len(sg) == k:
        return
    # Recursive step:
    old_v_ext = v_ext.copy()
    while len(v_ext) > 0:
        w = v_ext.pop()
        new_v_ext = v_ext.copy()
        neighbors = [nbr for nbr in list(G[w].keys()) if nbr > node_id and nbr
            not in sg and nbr not in old_v_ext]
        n_frac = len(neighbors) * ps[len(sg) + 1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
            else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            #if nbr > node_id and nbr not in sg and nbr not in old_v_ext:
            new_v_ext.add(nbr)
        sg.add(w)
        extend_subgraph(G, k, sg, new_v_ext, node_id, motif_counts, ps,
            node_anchored)
        sg.remove(w)

def gen_baseline_queries_mfinder(queries, targets, n_samples=10000,
    node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    #sizes = {}
    #for i in range(5, 17):
    #    sizes[i] = 10
    out = []
    for size, count in tqdm(sizes.items()):
        print(size)
        counts = defaultdict(list)
        for i in tqdm(range(n_samples)):
            graph, neigh = sample_neigh(targets, size, graph_type="undirected")
            v = neigh[0]
            neigh = graph.subgraph(neigh).copy()
            nx.set_node_attributes(neigh, 0, name="anchor")
            neigh.nodes[v]["anchor"] = 1
            neigh.remove_edges_from(nx.selfloop_edges(neigh))
            counts[wl_hash(neigh, node_anchored=node_anchored)].append(neigh)
        #bads, t = 0, 0
        #for ka, nas in counts.items():
        #    for kb, nbs in counts.items():
        #        if ka != kb:
        #            for a in nas:
        #                for b in nbs:
        #                    if nx.is_isomorphic(a, b):
        #                        bads += 1
        #                        print("bad", bads, t)
        #                    t += 1

        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
            reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out

def parse_optimizer(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
            help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
            help='Learning rate decay ratio')
    opt_parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float,
            help='Optimizer weight decay.')

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95,
            weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


def standardize_graph(graph: nx.Graph, anchor: int = None) -> nx.Graph:
    """
    Standardize graph attributes to ensure compatibility with DeepSnap.
    
    Args:
        graph: Input NetworkX graph
        anchor: Optional anchor node index
        
    Returns:
        NetworkX graph with standardized attributes
    """
    if isinstance(graph, nx.DiGraph):
        g = nx.DiGraph()
    else:
        g = nx.Graph()

    g.add_nodes_from((n, dict(attrs)) for n, attrs in graph.nodes(data=True))
    g.add_edges_from((u, v, dict(attrs)) for u, v, attrs in graph.edges(data=True))
   # g = graph.copy()
    
    # Standardize edge attributes
    for u, v in g.edges():
        edge_data = g.edges[u, v]
        edge_type_raw = edge_semantic_label(edge_data) or "unknown"
        edge_type_id_raw = edge_data.get("type_id")

        # Remove invalid keys
        bad_keys = [k for k in list(edge_data.keys()) if not isinstance(k, str) or k.strip() == "" or isinstance(k, dict)]
        for k in bad_keys:
            del edge_data[k]

        # DeepSNAP compatibility: keep only numeric/scalar edge attributes.
        # Any string/object attrs can trigger "Unknown type of key {} in edge attributes."
        for k in list(edge_data.keys()):
            v_attr = edge_data[k]
            if isinstance(v_attr, bool):
                edge_data[k] = float(v_attr)
            elif isinstance(v_attr, (int, float, np.integer, np.floating)):
                edge_data[k] = float(v_attr)
            else:
                del edge_data[k]

        # Clean empty edge attributes if any
        if len(edge_data) == 0:
            edge_data['weight'] = 1.0
        # Ensure weight exists
        if 'weight' not in edge_data:
            edge_data['weight'] = 1.0
        else:
            try:
                edge_data['weight'] = float(edge_data['weight'])
            except (ValueError, TypeError):
                edge_data['weight'] = 1.0
        
        # Deterministic edge-type normalization for semantic mining. Capture the
        # string relation before DeepSNAP-compatible sanitization deletes strings.
        if edge_type_id_raw is not None:
            edge_type_id = int(edge_type_id_raw)
        else:
            edge_type_id = int(_stable_edge_type_id(edge_type_raw))
        edge_data['type_id'] = edge_type_id
        edge_data['type'] = float(edge_type_id)
    
    # Standardize node attributes
    for node in g.nodes():
        node_data = g.nodes[node]
        
        # Initialize node features if needed
        if anchor is not None:
            node_data['node_feature'] = torch.tensor([float(node == anchor)])
        elif 'node_feature' not in node_data:
            # Default feature if no anchor specified
            node_data['node_feature'] = torch.tensor([1.0])
            
        semantic_label = node_semantic_label(node_data)
        if 'label' not in node_data:
            node_data['label'] = str(node)
        node_data['semantic_label'] = semantic_label
        node_data['label_id'] = int(_stable_label_id(semantic_label))
        if _semantic_hash_mode == "hybrid_text":
            node_data['text_label_bucket_id'] = int(_text_bucket_id(semantic_label))
            
        # Ensure id exists
        if 'id' not in node_data:
            node_data['id'] = str(node)
    
    return g




def batch_nx_graphs(graphs, anchors=None):


    # Initialize feature augmenter
    augmenter = feature_preprocess.FeatureAugment()
    
    # Process graphs with proper attribute handling
    processed_graphs = []
    for i, graph in enumerate(graphs):
        anchor = anchors[i] if anchors is not None else None
        try:
            # Standardize graph attributes


            std_graph = standardize_graph(graph, anchor)
            
            # Convert to DeepSnap format
            ds_graph = DSGraph(std_graph)

            processed_graphs.append(ds_graph)
            
        except Exception as e:
            print(f"Warning: Error processing graph {i}: {str(e)}")
            # Create minimal graph with basic features if conversion fails
            minimal_graph = nx.Graph()
            minimal_graph.add_nodes_from(graph.nodes())
            minimal_graph.add_edges_from(graph.edges())
            for node in minimal_graph.nodes():
                minimal_graph.nodes[node]['node_feature'] = torch.tensor([1.0])
            processed_graphs.append(DSGraph(minimal_graph))
    
    # Create batch
    batch = Batch.from_data_list(processed_graphs)
    
    # Suppress the specific warning during augmentation
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unknown type of key*')

        batch = augmenter.augment(batch)
    
    return batch.to(get_device())

def get_device():
    """Get PyTorch device (GPU if available, otherwise CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_gpu_memory():
    """Utility function to clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2 
    return 0
