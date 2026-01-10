from collections import defaultdict, Counter

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
        if graph_type == "undirected":
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        elif graph_type == "directed":
            frontier = list(set(graph.successors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            #new_node = max(sorted(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            if graph_type == "undirected":
                frontier += list(graph.neighbors(new_node))
            elif graph_type == "directed":
                frontier += list(graph.successors(new_node))
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

def wl_hash(g, dim=64, node_anchored=False):
    g = nx.convert_node_labels_to_integers(g)
    vecs = np.zeros((len(g), dim), dtype=int)
    if node_anchored:
        for v in g.nodes:
            if g.nodes[v]["anchor"] == 1:
                vecs[v] = 1
                break
    for i in range(len(g)):
        newvecs = np.zeros((len(g), dim), dtype=int)
        for n in g.nodes:
            newvecs[n] = vec_hash(np.sum(vecs[list(g.neighbors(n)) + [n]],
                axis=0))
        vecs = newvecs
    return tuple(np.sum(vecs, axis=0))

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

    g.add_nodes_from(graph.nodes())
    g.add_edges_from(graph.edges())
   # g = graph.copy()
    
    # Standardize edge attributes
    for u, v in g.edges():
        edge_data = g.edges[u, v]

        # Remove invalid keys
        bad_keys = [k for k in list(edge_data.keys()) if not isinstance(k, str) or k.strip() == "" or isinstance(k, dict)]
        for k in bad_keys:
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
        
        # Handle edge type
        if 'type' in edge_data:
            edge_data['type_str'] = str(edge_data['type'])
            edge_data['type'] = float(hash(str(edge_data['type'])) % 1000)
    
    # Standardize node attributes
    for node in g.nodes():
        node_data = g.nodes[node]
        
        # Initialize node features if needed
        if anchor is not None:
            node_data['node_feature'] = torch.tensor([float(node == anchor)])
        elif 'node_feature' not in node_data:
            # Default feature if no anchor specified
            node_data['node_feature'] = torch.tensor([1.0])
            
        # Ensure label exists
        if 'label' not in node_data:
            node_data['label'] = str(node)
            
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