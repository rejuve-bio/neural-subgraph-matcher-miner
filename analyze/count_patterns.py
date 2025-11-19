import argparse
import csv
import time
import os
import json
import concurrent.futures
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid, KarateClub, QM7b
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm
from tqdm import tqdm
import matplotlib.pyplot as plt

from multiprocessing import Pool
import random
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx
import networkx.algorithms.isomorphism as iso
import pickle
import torch.multiprocessing as mp
from sklearn.decomposition import PCA
from itertools import combinations

MAX_SEARCH_TIME = 1800  
MAX_MATCHES_PER_QUERY = 10000
DEFAULT_SAMPLE_ANCHORS = 1000
CHECKPOINT_INTERVAL = 100  

# Global caches for worker processes
_GLOBAL_QUERIES = None
_GLOBAL_TARGETS = None
_GLOBAL_QUERY_STATS = None
_GLOBAL_TARGET_STATS = None

_GLOBAL_ENGINE = None
_GLOBAL_QUERIES_IG = None
_GLOBAL_TARGETS_IG = None

def _init_worker(queries, targets, query_stats, target_stats,
                 queries_ig, targets_ig, engine):
    """Initializer for worker processes: set global references."""
    global _GLOBAL_QUERIES, _GLOBAL_TARGETS
    global _GLOBAL_QUERY_STATS, _GLOBAL_TARGET_STATS
    global _GLOBAL_ENGINE, _GLOBAL_QUERIES_IG, _GLOBAL_TARGETS_IG

    _GLOBAL_QUERIES = queries
    _GLOBAL_TARGETS = targets
    _GLOBAL_QUERY_STATS = query_stats
    _GLOBAL_TARGET_STATS = target_stats

    _GLOBAL_ENGINE = engine
    _GLOBAL_QUERIES_IG = queries_ig
    _GLOBAL_TARGETS_IG = targets_ig

def compute_graph_stats(G):
    """Compute graph statistics for filtering."""
    stats = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'is_directed': G.is_directed(),
        'degree_seq': sorted([d for _, d in G.degree()], reverse=True),
        'avg_degree': sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1)
    }
    
    if G.is_directed():
        stats['in_degree_seq'] = sorted([d for _, d in G.in_degree()], reverse=True)
        stats['out_degree_seq'] = sorted([d for _, d in G.out_degree()], reverse=True)
    
    try:
        if G.is_directed():
            stats['n_components'] = nx.number_weakly_connected_components(G)
        else:
            stats['n_components'] = nx.number_connected_components(G)
    except:
        stats['n_components'] = 1 
        
    return stats

def can_be_isomorphic(query_stats, target_stats):
    if query_stats['n_nodes'] > target_stats['n_nodes']:
        return False
    if query_stats['n_edges'] > target_stats['n_edges']:
        return False
    
    if query_stats['is_directed'] != target_stats['is_directed']:
        return False
    
    
    if len(query_stats['degree_seq']) > 0 and len(target_stats['degree_seq']) > 0:
        if query_stats['degree_seq'][0] > target_stats['degree_seq'][0]:
            return False
    
    # For directed graphs, check in/out degrees
    if query_stats['is_directed']:
        if (len(query_stats['in_degree_seq']) > 0 and len(target_stats['in_degree_seq']) > 0 and
            query_stats['in_degree_seq'][0] > target_stats['in_degree_seq'][0]):
            return False
        if (len(query_stats['out_degree_seq']) > 0 and len(target_stats['out_degree_seq']) > 0 and
            query_stats['out_degree_seq'][0] > target_stats['out_degree_seq'][0]):
            return False
    
    if query_stats['avg_degree'] > target_stats['avg_degree'] * 1.1:  
        return False
    
    return True

def arg_parse():
    parser = argparse.ArgumentParser(description='count graphlets in a graph')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--queries_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--n_workers', type=int)
    parser.add_argument('--count_method', type=str)
    parser.add_argument('--baseline', type=str)
    parser.add_argument('--node_anchored', action="store_true")
    parser.add_argument('--max_query_size', type=int, default=20, help='Maximum query size to process')
    parser.add_argument('--sample_anchors', type=int, default=DEFAULT_SAMPLE_ANCHORS, help='Number of anchor nodes to sample for large graphs')
    parser.add_argument('--checkpoint_file', type=str, default="checkpoint.json", help='File to save/load progress')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for processing')
    parser.add_argument('--timeout', type=int, default=MAX_SEARCH_TIME, help='Timeout per task in seconds')
    parser.add_argument('--use_sampling', action="store_true", help='Use node sampling for very large graphs')
    parser.add_argument('--graph_type', type=str, default='auto', choices=['directed', 'undirected', 'auto'],
                       help='Graph type: directed, undirected, or auto-detect')
    parser.add_argument(
        '--engine',
        type=str,
        default='nx',
        choices=['nx', 'igraph'],
        help='Matching engine: NetworkX (nx) or igraph backend.'
    )
    parser.set_defaults(dataset="test_100n_200e.pkl",
                       queries_path="patterns_test.pkl",
                       out_path="counts.json",
                       n_workers=4,
                       count_method="bin",
                       baseline="none",
                       node_anchored=True)
    return parser.parse_args()

def load_networkx_graph(filepath, directed=None):
    """Load a Networkx graph from pickle format with proper attributes handling.
    
    Args:
        filepath: Path to the pickle file
        directed: If True, create DiGraph; if False, create Graph; if None, auto-detect
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
        if isinstance(data, (nx.Graph, nx.DiGraph)):
            if directed is None:
                return data
            elif directed and not data.is_directed():
                return data.to_directed()
            elif not directed and data.is_directed():
                return data.to_undirected()
            else:
                return data
        
        if directed is None:
            if isinstance(data, dict):
                directed = data.get('directed', False)
        
        graph = nx.DiGraph() if directed else nx.Graph()
        
        for node in data['nodes']:
            if isinstance(node, tuple):
                node_id, attrs = node
                graph.add_node(node_id, **attrs)
            else:
                graph.add_node(node)
        
        for edge in data['edges']:
            if len(edge) == 3:
                src, dst, attrs = edge
                graph.add_edge(src, dst, **attrs)
            else:
                src, dst = edge[:2]
                graph.add_edge(src, dst)
                
        return graph

def nx_to_igraph(G):
    """Convert a NetworkX graph to an igraph Graph, preserving node/edge attributes.

    This is only used when --engine=igraph is requested. Importing igraph here
    keeps the dependency optional for normal NetworkX runs.
    """
    try:
        import igraph as ig
    except ImportError as e:
        raise ImportError("igraph is required for --engine=igraph but is not installed") from e

    nodes = list(G.nodes())
    nx_to_idx = {n: i for i, n in enumerate(nodes)}
    directed = G.is_directed()

    g = ig.Graph(directed=directed)
    g.add_vertices(len(nodes))

    for n, i in nx_to_idx.items():
        # Preserve original NetworkX node id for later lookup (e.g., anchors)
        g.vs[i]["nx_id"] = n
        for attr, val in G.nodes[n].items():
            g.vs[i][attr] = val

    for u, v, attrs in G.edges(data=True):
        g.add_edge(nx_to_idx[u], nx_to_idx[v])
        eid = g.get_eid(nx_to_idx[u], nx_to_idx[v])
        for attr, val in attrs.items():
            g.es[eid][attr] = val

    return g

def match_task_nx(query, target, method, node_anchored,
                  anchor_or_none, timeout, q_idx, start_time):
    """NetworkX-based matching for a single (query, target, anchor) task."""
    count = 0

    try:
        if method == "freq":
            if query.is_directed():
                ismags = nx.isomorphism.DiGraphMatcher(query, query)
            else:
                ismags = nx.isomorphism.ISMAGS(query, query)
            n_symmetries = len(list(ismags.isomorphisms_iter(symmetry=False)))

        if method == "bin":
            if node_anchored:
                nx.set_node_attributes(target, 0, name="anchor")
                if anchor_or_none in target:
                    target.nodes[anchor_or_none]["anchor"] = 1

                if target.is_directed():
                    matcher = iso.DiGraphMatcher(
                        target, query,
                        node_match=iso.categorical_node_match(["anchor"], [0])
                    )
                else:
                    matcher = iso.GraphMatcher(
                        target, query,
                        node_match=iso.categorical_node_match(["anchor"], [0])
                    )

                if time.time() - start_time > timeout:
                    print(f"Timeout on query {q_idx} before isomorphism check")
                    return 0

                count = int(matcher.subgraph_is_isomorphic())
            else:
                if target.is_directed():
                    matcher = iso.DiGraphMatcher(target, query)
                else:
                    matcher = iso.GraphMatcher(target, query)

                if time.time() - start_time > timeout:
                    print(f"Timeout on query {q_idx} before isomorphism check")
                    return 0

                count = int(matcher.subgraph_is_isomorphic())
        elif method == "freq":
            if target.is_directed():
                matcher = iso.DiGraphMatcher(target, query)
            else:
                matcher = iso.GraphMatcher(target, query)

            count = 0
            for _ in matcher.subgraph_isomorphisms_iter():
                if time.time() - start_time > timeout:
                    print(f"Timeout during isomorphism iteration for query {q_idx}")
                    break
                count += 1
                if count >= MAX_MATCHES_PER_QUERY:
                    break

            if method == "freq" and n_symmetries > 0:
                count = count / n_symmetries

    except TimeoutError as e:
        print(f"Task {q_idx} timed out: {str(e)}")
        count = 0
    except Exception as e:
        print(f"Error processing query {q_idx}: {str(e)}")
        count = 0

    return count


def match_task_igraph(query_ig, target_ig, method, node_anchored,
                      anchor_or_none, timeout, q_idx, start_time):
    """igraph-based matching for a single (query, target, anchor) task.

    Currently supports only method=="bin". For other methods, falls back to
    NotImplementedError so behavior is explicit.
    """
    try:
        import igraph as ig  # noqa: F401  # imported for side-effects / type
    except ImportError as e:
        raise ImportError("igraph is required for --engine=igraph but is not installed") from e

    if method != "bin":
        raise NotImplementedError("igraph backend currently supports only count_method='bin'")

    # Non-anchored case: simple subgraph existence check.
    # Use subisomorphic_vf2 to avoid enumerating all mappings.
    if not node_anchored:
        is_subiso = target_ig.subisomorphic_vf2(query_ig)
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"[igraph] Timeout on query {q_idx} after VF2 ({elapsed:.2f}s)")
            return 0
        return int(bool(is_subiso))

    # Anchored case: enforce that the query's anchor node maps to the given target anchor
    # We assume that query_ig vertices already carry an 'anchor' attribute
    # consistent with the NetworkX version.

    # Find target vertex index corresponding to anchor_or_none using stored nx_id
    anchor_idx = None
    for v in target_ig.vs:
        if "nx_id" in v.attributes() and v["nx_id"] == anchor_or_none:
            anchor_idx = v.index
            break

    if anchor_idx is None:
        # Anchor not present in this target
        return 0

    # Prepare anchor attributes: 1 for anchor, 0 otherwise
    # For target: only the chosen anchor vertex has anchor==1
    anchor_vals_target = [0] * target_ig.vcount()
    anchor_vals_target[anchor_idx] = 1

    # For query: rely on any existing 'anchor' attribute, defaulting to 0
    anchor_vals_query = []
    for v in query_ig.vs:
        val = v["anchor"] if "anchor" in v.attributes() else 0
        anchor_vals_query.append(val)

    # Run VF2 with color constraints based on anchor values.
    # color1/color2 are lists of integers that must match between mapped vertices.
    is_subiso = target_ig.subisomorphic_vf2(
        query_ig,
        color1=anchor_vals_target,
        color2=anchor_vals_query,
    )

    elapsed = time.time() - start_time
    if elapsed > timeout:
        print(f"[igraph] Timeout on query {q_idx} after VF2 ({elapsed:.2f}s)")
        return 0

    # Any valid mapping means at least one anchored match
    return int(bool(is_subiso))


def run_match_task(engine,
                   query, target,
                   query_ig, target_ig,
                   method, node_anchored,
                   anchor_or_none, timeout,
                   q_idx, start_time):
    """Dispatch a single matching task to the selected backend."""
    if engine == "nx":
        return match_task_nx(
            query, target, method, node_anchored,
            anchor_or_none, timeout, q_idx, start_time
        )
    elif engine == "igraph":
        return match_task_igraph(
            query_ig, target_ig, method, node_anchored,
            anchor_or_none, timeout, q_idx, start_time
        )
    else:
        raise ValueError(f"Unknown engine: {engine}")


def count_graphlets_helper(inp):
    q_idx, t_idx, method, node_anchored, anchor_or_none, timeout = inp

    query = _GLOBAL_QUERIES[q_idx]
    target = _GLOBAL_TARGETS[t_idx]
    query_stats = _GLOBAL_QUERY_STATS[q_idx]
    target_stats = _GLOBAL_TARGET_STATS[t_idx]
    engine = _GLOBAL_ENGINE

    query_ig = None
    target_ig = None
    if engine == "igraph":
        query_ig = _GLOBAL_QUERIES_IG[q_idx]
        target_ig = _GLOBAL_TARGETS_IG[t_idx]

    start_time = time.time()

    if not can_be_isomorphic(query_stats, target_stats):
        return q_idx, 0

    query = query.copy()
    query.remove_edges_from(nx.selfloop_edges(query))
    target = target.copy()
    target.remove_edges_from(nx.selfloop_edges(target))

    count = run_match_task(
        engine=engine,
        query=query,
        target=target,
        query_ig=query_ig,
        target_ig=target_ig,
        method=method,
        node_anchored=node_anchored,
        anchor_or_none=anchor_or_none,
        timeout=timeout,
        q_idx=q_idx,
        start_time=start_time,
    )

    processing_time = time.time() - start_time
    if processing_time > 10:
        print(f"Query {q_idx} processed in {processing_time:.2f} seconds with count {count}")

    return q_idx, count

def save_checkpoint(n_matches, checkpoint_file):
    with open(checkpoint_file, 'w') as f:
        json.dump({str(k): v for k, v in n_matches.items()}, f)
    print(f"Checkpoint saved to {checkpoint_file}")

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            try:
                checkpoint = json.load(f)
                return defaultdict(float, {int(k): v for k, v in checkpoint.items()})
            except json.JSONDecodeError:
                print(f"Error loading checkpoint file {checkpoint_file}, starting fresh")
    return defaultdict(float)

def sample_subgraphs(target, n_samples=10, max_size=1000):
    subgraphs = []
    nodes = list(target.nodes())
    
    for _ in range(n_samples):
        start_node = random.choice(nodes)
        subgraph_nodes = {start_node}
        
        if target.is_directed():
            frontier = list(set(target.successors(start_node)) | set(target.predecessors(start_node)))
        else:
            frontier = list(target.neighbors(start_node))
        
        while len(subgraph_nodes) < max_size and frontier:
            next_node = frontier.pop(0)
            if next_node not in subgraph_nodes:
                subgraph_nodes.add(next_node)
                if target.is_directed():
                    new_neighbors = set(target.successors(next_node)) | set(target.predecessors(next_node))
                else:
                    new_neighbors = set(target.neighbors(next_node))
                frontier.extend([n for n in new_neighbors 
                              if n not in subgraph_nodes and n not in frontier])
        
        sg = target.subgraph(subgraph_nodes)
        subgraphs.append(sg)
        
    return subgraphs

def count_graphlets(queries, targets, args):
    print(f"Processing {len(queries)} queries across {len(targets)} targets")
    
    is_directed = any(g.is_directed() for g in queries + targets)
    if is_directed:
        print("Detected directed graphs - using DiGraphMatcher")
    
    n_matches = load_checkpoint(args.checkpoint_file)
    
    problematic_tasks_file = "problematic_tasks.json"
    if os.path.exists(problematic_tasks_file):
        with open(problematic_tasks_file, 'r') as f:
            try:
                problematic_tasks = set(json.load(f))
                print(f"Loaded {len(problematic_tasks)} problematic tasks to skip")
            except:
                problematic_tasks = set()
    else:
        problematic_tasks = set()
    
    if args.use_sampling and any(t.number_of_nodes() > 100000 for t in targets):
        sampled_targets = []
        for target in targets:
            if target.number_of_nodes() > 100000:
                print(f"Sampling subgraphs from large graph with {target.number_of_nodes()} nodes")
                sampled_targets.extend(sample_subgraphs(target, n_samples=20, max_size=10000))
            else:
                sampled_targets.append(target)
        targets = sampled_targets
        print(f"After sampling: {len(targets)} target graphs to process")

    query_stats = [compute_graph_stats(q) for q in queries]
    target_stats = [compute_graph_stats(t) for t in targets]

    engine = args.engine

    queries_ig = None
    targets_ig = None
    if engine == "igraph":
        queries_ig = [nx_to_igraph(q) for q in queries]
        targets_ig = [nx_to_igraph(t) for t in targets]

    inp = []
    for qi, q in enumerate(queries):
        if q.number_of_nodes() > args.max_query_size:
            print(f"Skipping query {qi}: exceeds max size {args.max_query_size}")
            continue
        q_stats = query_stats[qi]

        for ti, t in enumerate(targets):
            t_stats = target_stats[ti]
            if not can_be_isomorphic(q_stats, t_stats):
                continue
            
            task_id = f"{qi}_{ti}"
            
            if task_id in problematic_tasks:
                print(f"Skipping known problematic task {task_id}")
                continue
                
            if task_id in n_matches:
                print(f"Skipping already processed task {task_id}")
                continue
                
            if args.node_anchored:
                if t.number_of_nodes() > args.sample_anchors:
                    anchors = random.sample(list(t.nodes), args.sample_anchors)
                else:
                    anchors = list(t.nodes)
                    
                for anchor in anchors:
                    inp.append((qi, ti, args.count_method, args.node_anchored, anchor, args.timeout))
            else:
                inp.append((qi, ti, args.count_method, args.node_anchored, None, args.timeout))
    
    print(f"Generated {len(inp)} tasks after filtering")
    n_done = 0
    last_checkpoint = time.time()
   
    with Pool(
        processes=args.n_workers,
        initializer=_init_worker,
        initargs=(queries, targets, query_stats, target_stats, queries_ig, targets_ig, engine),
    ) as pool:
        for batch_start in range(0, len(inp), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(inp))
            batch = inp[batch_start:batch_end]

            print(f"Processing batch {batch_start}-{batch_end} out of {len(inp)}")
            batch_start_time = time.time()

            results = pool.imap_unordered(count_graphlets_helper, batch)

            for result in results:
                if time.time() - batch_start_time > 3600:  
                    print(f"Batch {batch_start}-{batch_end} taking too long, marking remaining tasks problematic")
                    for task in batch:
                        i = task[0]
                        task_id = f"{i}_{batch_start}"
                        problematic_tasks.add(task_id)
                    break

                i, n = result
                n_matches[i] += n
                n_done += 1

                if n_done % 10 == 0:
                    print(f"Processed {n_done}/{len(inp)} tasks, queries with matches: {sum(1 for v in n_matches.values() if v > 0)}/{len(n_matches)}", flush=True)

                if time.time() - last_checkpoint > 300:
                    save_checkpoint(n_matches, args.checkpoint_file)
                    with open(problematic_tasks_file, 'w') as f:
                        json.dump(list(problematic_tasks), f)
                    last_checkpoint = time.time()

            save_checkpoint(n_matches, args.checkpoint_file)
            with open(problematic_tasks_file, 'w') as f:
                json.dump(list(problematic_tasks), f)

    print("\nDone counting")
    return [n_matches[i] for i in range(len(queries))]


def generate_one_baseline(args):
    import networkx as nx
    import random

    i, query, targets, method = args

    if len(query) == 0:
        return query

    MAX_ATTEMPTS = 100 
    for attempt in range(MAX_ATTEMPTS):
        try:
            graph = random.choice(targets)
            if graph.number_of_nodes() == 0:
                continue

            if method == "radial":
                node = random.choice(list(graph.nodes))
                if graph.is_directed():
                    neigh = set([node])
                    visited = {node}
                    queue = [(node, 0)]
                    while queue:
                        current, dist = queue.pop(0)
                        if dist < 3:
                            for neighbor in set(graph.successors(current)) | set(graph.predecessors(current)):
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    neigh.add(neighbor)
                                    queue.append((neighbor, dist + 1))
                    neigh = list(neigh)
                else:
                    neigh = list(nx.single_source_shortest_path_length(graph, node, cutoff=3).keys())
                
                subgraph = graph.subgraph(neigh)
                if subgraph.number_of_nodes() == 0:
                    continue
                
                if graph.is_directed():
                    largest_cc = max(nx.weakly_connected_components(subgraph), key=len)
                else:
                    largest_cc = max(nx.connected_components(subgraph), key=len)
                neigh = subgraph.subgraph(largest_cc)
                neigh = nx.convert_node_labels_to_integers(neigh)
                if len(neigh) == len(query):
                    return neigh

            elif method == "tree":
                start_node = random.choice(list(graph.nodes))
                neigh = [start_node]
                if graph.is_directed():
                    frontier = list(set(graph.successors(start_node)) | set(graph.predecessors(start_node)) - set(neigh))
                else:
                    frontier = list(set(graph.neighbors(start_node)) - set(neigh))
                
                while len(neigh) < len(query) and frontier:
                    new_node = random.choice(frontier)
                    neigh.append(new_node)
                    if graph.is_directed():
                        new_neighbors = list(set(graph.successors(new_node)) | set(graph.predecessors(new_node)))
                    else:
                        new_neighbors = list(graph.neighbors(new_node))
                    frontier += new_neighbors
                    frontier = [x for x in frontier if x not in neigh]
                
                if len(neigh) == len(query):
                    sub = graph.subgraph(neigh)
                    return nx.convert_node_labels_to_integers(sub)

        except Exception as e:
            continue 

    print(f"[WARN] Baseline not found for query {i} after {MAX_ATTEMPTS} attempts.")
    return nx.DiGraph() if query.is_directed() else nx.Graph()

def gen_baseline_queries(queries, targets, method="radial", node_anchored=False):
    print(f"Generating {len(queries)} baseline queries in parallel using method: {method}")
    
    args_list = [(i, query, targets, method) for i, query in enumerate(queries)]
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(generate_one_baseline, args_list)
    
    return results

def main():
    global args
    args = arg_parse()
    print("Using {} workers".format(args.n_workers))
    print("Baseline:", args.baseline)
    print(f"Max query size: {args.max_query_size}")
    print(f"Timeout per task: {args.timeout} seconds")
    print(f"Graph type: {args.graph_type}")

    use_directed = (args.graph_type == 'directed')

    if args.dataset.endswith('.pkl'):
        print(f"Loading Networkx graph from {args.dataset}")
        try:
            if args.graph_type == 'auto':
                graph = load_networkx_graph(args.dataset, directed=None)
            else:
                graph = load_networkx_graph(args.dataset, directed=use_directed)
            
            graph_type = "directed" if graph.is_directed() else "undirected"
            print(f"Loaded {graph_type} graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            dataset = [graph]
        except Exception as e:
            print(f"Error loading graph: {str(e)}")
            raise
    elif args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    elif args.dataset == 'cox2':
        dataset = TUDataset(root='/tmp/cox2', name='COX2')
    elif args.dataset == 'reddit-binary':
        dataset = TUDataset(root='/tmp/REDDIT-BINARY', name='REDDIT-BINARY')
    elif args.dataset == 'coil':
        dataset = TUDataset(root='/tmp/coil', name='COIL-DEL')
    elif args.dataset == 'ppi-pathways':
        graph = nx.DiGraph() if use_directed else nx.Graph()
        with open("data/ppi-pathways.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                graph.add_edge(int(row[0]), int(row[1]))
        dataset = [graph]
    elif args.dataset in ['diseasome', 'usroads', 'mn-roads', 'infect']:
        fn = {"diseasome": "bio-diseasome.mtx",
            "usroads": "road-usroads.mtx",
            "mn-roads": "mn-roads.mtx",
            "infect": "infect-dublin.edges"}
        graph = nx.DiGraph() if use_directed else nx.Graph()
        with open("data/{}".format(fn[args.dataset]), "r") as f:
            for line in f:
                if not line.strip(): continue
                a, b = line.strip().split(" ")
                graph.add_edge(int(a), int(b))
        dataset = [graph]
    elif args.dataset.startswith('plant-'):
        size = int(args.dataset.split("-")[-1])
       # dataset = decoder.make_plant_dataset(size)
    elif args.dataset == "analyze":
        with open("results/analyze.p", "rb") as f:
            cand_patterns, _ = pickle.load(f)
            queries = [q for score, q in cand_patterns[10]][:200]
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

    targets = []
    for i, graph in enumerate(dataset):
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            if args.graph_type == 'auto':
                graph = pyg_utils.to_networkx(graph).to_undirected()
            elif use_directed:
                graph = pyg_utils.to_networkx(graph, to_undirected=False)
            else:
                graph = pyg_utils.to_networkx(graph).to_undirected()
            for node in graph.nodes():
                if 'label' not in graph.nodes[node]:
                    graph.nodes[node]['label'] = str(node)
                if 'id' not in graph.nodes[node]:
                    graph.nodes[node]['id'] = str(node)
        else:
            if use_directed and not graph.is_directed():
                graph = graph.to_directed()
            elif not use_directed and graph.is_directed():
                graph = graph.to_undirected()
        targets.append(graph)

    if args.dataset != "analyze":
        with open(args.queries_path, "rb") as f:
            queries = pickle.load(f)
    
    if use_directed:
        queries = [q.to_directed() if not q.is_directed() else q for q in queries]
    else:
        queries = [q.to_undirected() if q.is_directed() else q for q in queries]
            
    query_lens = [len(query) for query in queries]
    print(f"Loaded {len(queries)} query patterns")
    print(f"Query graph type: {'directed' if queries[0].is_directed() else 'undirected'}")
    print(f"Target graph type: {'directed' if targets[0].is_directed() else 'undirected'}")
    # ---- start timing the counting phase ----
    overall_start = time.time()
    
    if args.baseline == "exact":
        print("Using exact counting method")
        n_matches = count_graphlets(queries, targets, args)
    elif args.baseline == "none":
        n_matches = count_graphlets(queries, targets, args)
    else:
        print(f"Generating baseline queries using {args.baseline}")
        baseline_queries = gen_baseline_queries(queries, targets,
            node_anchored=args.node_anchored, method=args.baseline)
        query_lens = [len(q) for q in baseline_queries]
        n_matches = count_graphlets(baseline_queries, targets, args)
    
    total_time = time.time() - overall_start
    print(f"Total counting time: {total_time:.2f} seconds")
    # ---- end timing ----    
    # Save results
    query_lens = [q.number_of_nodes() for q in queries]
    output_file = os.path.join(args.out_path, "counts.json")
    
    # Ensure output directory exists
    os.makedirs(args.out_path, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump({"query_lengths": query_lens, "counts": n_matches, "metadata": {}}, f)
    
    print(f"Results saved to {output_file}")
    print("=== Completed ===")



if __name__ == "__main__":
    main()