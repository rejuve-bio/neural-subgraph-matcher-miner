import argparse
import csv
import collections
from itertools import combinations
import time
import os
import pickle
import sys
import traceback
from pathlib import Path

from deepsnap.batch import Batch
from deepsnap.graph import Graph as DSGraph
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torch_geometric.datasets import TUDataset, PPI
from torch_geometric.datasets import Planetoid, KarateClub, QM7b
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn


from torch_geometric.datasets import tu_dataset as pyg_tu_dataset
tu_url = os.environ.get("TU_DATASET_URL", "https://www.chrsmrrs.com/graphkerneldatasets")
pyg_tu_dataset.url = tu_url
TUDataset.url = tu_url


import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from matplotlib import cm

from common import data
from common import models
from common import utils
from common import combined_syn
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder
import datetime  
import uuid 

# Add root to sys.path for robust imports in various environments (Docker, etc)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from subgraph_mining.search_agents import (
    GreedySearchAgent, MCTSSearchAgent, 
    MemoryEfficientMCTSAgent, MemoryEfficientGreedyAgent, 
    BeamSearchAgent
)

from subgraph_mining.embedding import (
    extract_neighborhood,
    TargetedDataset,
    StreamingNeighborhoodDataset,
    LazyNeighborhoodGraphList,
    collate_fn,
    generate_target_embeddings
)

from subgraph_mining.visualization import (
    VISUALIZER_AVAILABLE,
    visualize_pattern_graph,
    save_instances_to_json,
    update_run_index,
    save_and_visualize_all_instances
)


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
from sklearn.decomposition import PCA
import json 
import logging
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        "plots",
        "plots/cluster",
        "results"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def bfs_chunk(graph, start_node, max_size):
    visited = set([start_node])
    queue = [start_node]
    while queue and len(visited) < max_size:
        node = queue.pop(0)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) >= max_size:
                    break
    return graph.subgraph(visited).copy()


def process_large_graph_in_chunks(graph, chunk_size=10000):
    all_nodes = set(graph.nodes())
    graph_chunks = []
    while all_nodes:
        start_node = next(iter(all_nodes))
        chunk = bfs_chunk(graph, start_node, chunk_size)
        graph_chunks.append(chunk)
        all_nodes -= set(chunk.nodes())
    return graph_chunks


def make_plant_dataset(size):
    generator = combined_syn.get_generator([size])
    random.seed(3001)
    np.random.seed(14853)
    pattern = generator.generate(size=10)
    nx.draw(pattern, with_labels=True)
    plt.savefig("plots/cluster/plant-pattern.png")
    plt.close()
    graphs = []
    for i in range(1000):
        graph = generator.generate()
        n_old = len(graph)
        graph = nx.disjoint_union(graph, pattern)
        for j in range(1, 3):
            u = random.randint(0, n_old - 1)
            v = random.randint(n_old, len(graph) - 1)
            graph.add_edge(u, v)
        graphs.append(graph)
    return graphs


def _process_chunk(args_tuple):
    chunk_dataset, task, args, chunk_index, total_chunks = args_tuple
    start_time = time.time()
    last_print = start_time
    print(f"[{time.strftime('%H:%M:%S')}] Worker PID {os.getpid()} started chunk {chunk_index+1}/{total_chunks}", flush=True)
    try:
        result = None
        while result is None:
            now = time.time()
            if now - last_print >= 10:
                print(f"[{time.strftime('%H:%M:%S')}] Worker PID {os.getpid()} still processing chunk {chunk_index+1}/{total_chunks} ({int(now-start_time)}s elapsed)", flush=True)
                last_print = now
            result = pattern_growth([chunk_dataset], task, args)
        print(f"[{time.strftime('%H:%M:%S')}] Worker PID {os.getpid()} finished chunk {chunk_index+1}/{total_chunks} in {int(time.time()-start_time)}s", flush=True)
        return result
    except Exception as e:
        print(f"Error processing chunk {chunk_index}: {e}", flush=True)
        return []


#  Optimized streaming entry point
def pattern_growth_streaming(dataset, task, args):
    """Entry point for batch processing mode."""
    import gc
    if args.method_type == "end2end":
        model = models.End2EndOrder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    
    model.load_state_dict(torch.load(args.model_path, map_location=utils.get_device()))
    model.eval()
    
    # Batched embedding generation
    global_embs, seed_graphs = generate_target_embeddings(dataset, model, args)
    
    # Release the massive main graph from memory before search
    logger.info("CRITICAL: Cleaning up main graph from RAM to optimize Search Phase...")
    if isinstance(dataset, list):
        dataset.clear()
    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")
    
    # Parallel search
    logger.info("Search phase starting with precomputed embeddings...")
    # Force use_whole_graphs=True because the input 'seed_graphs' are already the extracted neighborhoods
    original_use_whole = args.use_whole_graphs
    args.use_whole_graphs = True
    found_patterns = pattern_growth(seed_graphs, task, args, precomputed_data=global_embs, preloaded_model=model)
    args.use_whole_graphs = original_use_whole
    
    # Global Frequency Validation (Accuracy validator for batch processing)
    logger.info("Performing Global Frequency Validation on discovered patterns...")
    
    if global_embs and len(global_embs) > 0:
        global_matrix = torch.cat(global_embs, dim=0).to(utils.get_device()) # (10000, D)
        
        for pattern in found_patterns:
            pat_anchor = 0 if args.node_anchored else None
            std_pat = utils.standardize_graph(pattern, anchor=pat_anchor)
            ds_pat = DSGraph(std_pat)
            batch_pat = Batch.from_data_list([ds_pat]).to(utils.get_device())
            
            with torch.no_grad():
                pat_emb = model.emb_model(batch_pat) # (1, D)
                        
            diff = pat_emb - global_matrix
            violation = torch.clamp(diff, min=0)
            violation_sq = torch.sum(violation ** 2, dim=1)
            epsilon = 1e-5
            support_count = (violation_sq < epsilon).sum().item()
            
            total_universe_size = global_matrix.shape[0]
            logger.info(f"Pattern Size {len(pattern)}: Corrected Support {pattern.graph.get('support', 0)} -> {support_count} (Universe Size: {total_universe_size})")
            pattern.graph['support'] = support_count
            pattern.graph['frequency'] = support_count / total_universe_size

    return found_patterns


def pattern_growth(dataset, task, args, precomputed_data=None, preloaded_model=None):
    """Main pattern mining function."""
    start_time = time.time()
    
    ensure_directories()
    
    # Load model (or use preloaded)
    if preloaded_model:
        model = preloaded_model
    elif args.method_type == "end2end":
        model = models.End2EndOrder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    
    if not preloaded_model:
        model.load_state_dict(torch.load(args.model_path,
            map_location=utils.get_device()))
    
    model.to(utils.get_device())
    model.eval()

    if task == "graph-labeled":
        dataset, labels = dataset

    neighs_pyg, neighs = [], []
    logger.info(f"{len(dataset)} graphs")
    logger.info(f"Search strategy: {args.search_strategy}")
    logger.info(f"Graph type: {args.graph_type}")
    
    if task == "graph-labeled":
        logger.info("Using label 0")
    
    graphs = []
    for i, graph in enumerate(dataset):
        if task == "graph-labeled" and labels[i] != 0:
            continue
        if task == "graph-truncate" and i >= 1000:
            break
        
        if not type(graph) == nx.Graph and not type(graph) == nx.DiGraph:
            graph = pyg_utils.to_networkx(graph).to_undirected()
            for node in graph.nodes():
                if 'label' not in graph.nodes[node]:
                    graph.nodes[node]['label'] = str(node)
                if 'id' not in graph.nodes[node]:
                    graph.nodes[node]['id'] = str(node)
        graphs.append(graph)
    
    if args.use_whole_graphs:
        neighs = graphs
    else:
        anchors = []
        if args.sample_method == "radial":
            for i, graph in enumerate(graphs):
                logger.info(f"Processing graph {i}")
                for j, node in enumerate(graph.nodes):
                    if len(dataset) <= 10 and j % 100 == 0:
                        logger.debug(f"Graph {i}, node {j}")
                    
                    if args.use_whole_graphs:
                        neigh = graph.nodes
                    else:
                        neigh = list(nx.single_source_shortest_path_length(graph,
                            node, cutoff=args.radius).keys())
                        if args.subgraph_sample_size != 0:
                            neigh = random.sample(neigh, min(len(neigh),
                                args.subgraph_sample_size))
                    
                    if len(neigh) > 1:
                        subgraph = graph.subgraph(neigh)
                        if args.subgraph_sample_size != 0:
                            subgraph = subgraph.subgraph(max(
                                nx.connected_components(subgraph), key=len))
                        
                        orig_attrs = {n: subgraph.nodes[n].copy() for n in subgraph.nodes()}
                        edge_attrs = {(u,v): subgraph.edges[u,v].copy() 
                                    for u,v in subgraph.edges()}
                        
                        mapping = {old: new for new, old in enumerate(subgraph.nodes())}
                        subgraph = nx.relabel_nodes(subgraph, mapping)
                        
                        for old, new in mapping.items():
                            subgraph.nodes[new].update(orig_attrs[old])
                        
                        for (old_u, old_v), attrs in edge_attrs.items():
                            subgraph.edges[mapping[old_u], mapping[old_v]].update(attrs)
                        
                        subgraph.add_edge(0, 0)
                        neighs.append(subgraph)
                        if args.node_anchored:
                            anchors.append(0)
        
        elif args.sample_method == "tree":
            start_time_sample = time.time()
            for j in tqdm(range(args.n_neighborhoods)):
                graph, neigh = utils.sample_neigh(graphs,
                    random.randint(args.min_neighborhood_size,
                        args.max_neighborhood_size), args.graph_type)
                neigh = graph.subgraph(neigh)
                neigh = nx.convert_node_labels_to_integers(neigh)
                neigh.add_edge(0, 0)
                neighs.append(neigh)
                if args.node_anchored:
                    anchors.append(0)

    #  Use precomputed embeddings if available
    if precomputed_data:
        embs = precomputed_data
    else:
        embs = []
        if len(neighs) % args.batch_size != 0:
            logger.warning("Number of graphs not multiple of batch size")
        
        for i in range(len(neighs) // args.batch_size):
            top = (i+1)*args.batch_size
            with torch.no_grad():
                batch = utils.batch_nx_graphs(neighs[i*args.batch_size:top],
                    anchors=anchors if args.node_anchored else None)
                emb = model.emb_model(batch)
                emb = emb.to(torch.device("cpu"))
            embs.append(emb)

    if args.analyze:
        embs_np = torch.stack(embs).numpy()
        plt.scatter(embs_np[:,0], embs_np[:,1], label="node neighborhood")

    if not hasattr(args, 'n_workers'):
        args.n_workers = mp.cpu_count()

    # Initialize search agent
    logger.info(f"Initializing {args.search_strategy} search agent...")
    
    if args.search_strategy == "mcts":
        assert args.method_type == "order"
        if args.memory_efficient:
            agent = MemoryEfficientMCTSAgent(args.min_pattern_size, args.max_pattern_size,
                model, graphs, embs, node_anchored=args.node_anchored,
                analyze=args.analyze, out_batch_size=args.out_batch_size)
        else:
            agent = MCTSSearchAgent(args.min_pattern_size, args.max_pattern_size,
                model, graphs, embs, node_anchored=args.node_anchored,
                analyze=args.analyze, out_batch_size=args.out_batch_size)
    
    elif args.search_strategy == "greedy":
        if args.memory_efficient:
            agent = MemoryEfficientGreedyAgent(args.min_pattern_size, args.max_pattern_size,
                model, graphs, embs, node_anchored=args.node_anchored,
                analyze=args.analyze, model_type=args.method_type,
                out_batch_size=args.out_batch_size)
        else:
            agent = GreedySearchAgent(args.min_pattern_size, args.max_pattern_size,
                model, graphs, embs, node_anchored=args.node_anchored,
                analyze=args.analyze, model_type=args.method_type,
                out_batch_size=args.out_batch_size, n_beams=1,
                n_workers=args.n_workers)
        agent.args = args
    
    elif args.search_strategy == "beam":
        agent = BeamSearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, model_type=args.method_type,
            out_batch_size=args.out_batch_size, beam_width=args.beam_width)
    
    # Run search
    logger.info(f"Running search with {args.n_trials} trials...")
    out_graphs = agent.run_search(args.n_trials)
    
    elapsed = time.time() - start_time
    logger.info(f"Total time: {elapsed:.2f}s ({int(elapsed)//60}m {int(elapsed)%60}s)")

    if hasattr(agent, 'counts') and agent.counts:
        logger.info("\nSaving all pattern instances...")
        pkl_path = save_and_visualize_all_instances(agent, args)
        
        if pkl_path:
            logger.info(f"✓ All instances saved to: {pkl_path}")
        else:
            logger.error("✗ Failed to save all instances")
    else:
        logger.warning("⚠ Agent.counts not found - cannot save all instances")
        logger.warning("  Check that your search agent populates agent.counts")

    count_by_size = defaultdict(int)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    
    successful_visualizations = 0
    
    ensure_directories()
    
    logger.info(f"\nSaving representative patterns to: {args.out_path}")
    with open(args.out_path, "wb") as f:
        pickle.dump(out_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    if os.path.exists(args.out_path):
        file_size = os.path.getsize(args.out_path) / 1024
        logger.info(f"✓ Representatives saved ({file_size:.1f} KB)")
    else:
        logger.error("✗ Failed to save representatives")
    
    json_results = []
    for pattern in out_graphs:
        pattern_data = {
            'nodes': [
                {
                    'id': str(node),
                    'label': pattern.nodes[node].get('label', ''),
                    'anchor': pattern.nodes[node].get('anchor', 0),
                    **{k: v for k, v in pattern.nodes[node].items() 
                       if k not in ['label', 'anchor']}
                }
                for node in pattern.nodes()
            ],
            'edges': [
                {
                    'source': str(u),
                    'target': str(v),
                    'type': data.get('type', ''),
                    **{k: v for k, v in data.items() if k != 'type'}
                }
                for u, v, data in pattern.edges(data=True)
            ],
            'metadata': {
                'num_nodes': len(pattern),
                'num_edges': pattern.number_of_edges(),
                'is_directed': pattern.is_directed()
            }
        }
        json_results.append(pattern_data)
    
    base_path = os.path.splitext(args.out_path)[0]
    if base_path.endswith('.json'):
        base_path = os.path.splitext(base_path)[0]
    
    json_path = base_path + '.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"✓ JSON version saved to: {json_path}")
    
    return out_graphs


def main():
    try:
        ensure_directories()

        parser = argparse.ArgumentParser(description='Decoder arguments')
        parse_encoder(parser)
        parse_decoder(parser)
        
        args = parser.parse_args()

        logger.info(f"Using dataset: {args.dataset}")
        logger.info(f"Graph type: {args.graph_type}")

        if args.dataset.endswith('.pkl'):
            with open(args.dataset, 'rb') as f:
                data = pickle.load(f)
                
                if isinstance(data, (nx.Graph, nx.DiGraph)):
                    graph = data
                    
                    if args.graph_type == "directed" and not graph.is_directed():
                        logger.info("Converting undirected graph to directed...")
                        graph = graph.to_directed()
                    elif args.graph_type == "undirected" and graph.is_directed():
                        logger.info("Converting directed graph to undirected...")
                        graph = graph.to_undirected()
                    
                    graph_type = "directed" if graph.is_directed() else "undirected"
                    logger.info(f"Using NetworkX {graph_type} graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
                    
                    sample_edges = list(graph.edges(data=True))[:3]
                    if sample_edges:
                        logger.info("Sample edge attributes:")
                        for u, v, attrs in sample_edges:
                            direction_info = attrs.get('direction', f"{u} -> {v}" if graph.is_directed() else f"{u} -- {v}")
                            edge_type = attrs.get('type', 'unknown')
                            logger.info(f"  {direction_info} (type: {edge_type})")
                    
                elif isinstance(data, dict) and 'nodes' in data and 'edges' in data:
                    if args.graph_type == "directed":
                        graph = nx.DiGraph()
                    else:
                        graph = nx.Graph()
                    graph.add_nodes_from(data['nodes'])
                    graph.add_edges_from(data['edges'])
                    logger.info(f"Created {args.graph_type} graph from dict format with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
                else:
                    raise ValueError(f"Unknown pickle format. Expected NetworkX graph or dict with 'nodes'/'edges' keys, got {type(data)}")
                    
            dataset = [graph]
            task = 'graph'
        
        elif args.dataset == 'enzymes':
            dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
            task = 'graph'
        elif args.dataset == 'cox2':
            dataset = TUDataset(root='/tmp/cox2', name='COX2')
            task = 'graph'
        elif args.dataset == 'reddit-binary':
            dataset = TUDataset(root='/tmp/REDDIT-BINARY', name='REDDIT-BINARY')
            task = 'graph'
        elif args.dataset == 'dblp':
            dataset = TUDataset(root='/tmp/dblp', name='DBLP_v1')
            task = 'graph-truncate'
        elif args.dataset == 'coil':
            dataset = TUDataset(root='/tmp/coil', name='COIL-DEL')
            task = 'graph'
        elif args.dataset.startswith('roadnet-'):
            graph = nx.Graph() if args.graph_type == "undirected" else nx.DiGraph()
            with open("data/{}.txt".format(args.dataset), "r") as f:
                for row in f:
                    if not row.startswith("#"):
                        a, b = row.split("\t")
                        graph.add_edge(int(a), int(b))
            dataset = [graph]
            task = 'graph'
        elif args.dataset == "ppi":
            dataset = PPI(root="/tmp/PPI")
            task = 'graph'
        elif args.dataset in ['diseasome', 'usroads', 'mn-roads', 'infect']:
            fn = {"diseasome": "bio-diseasome.mtx",
                "usroads": "road-usroads.mtx",
                "mn-roads": "mn-roads.mtx",
                "infect": "infect-dublin.edges"}
            graph = nx.Graph() if args.graph_type == "undirected" else nx.DiGraph()
            with open("data/{}".format(fn[args.dataset]), "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    a, b = line.strip().split(" ")
                    graph.add_edge(int(a), int(b))
            dataset = [graph]
            task = 'graph'
        elif args.dataset.startswith('plant-'):
            size = int(args.dataset.split("-")[-1])
            dataset = make_plant_dataset(size)
            task = 'graph'

        # Adaptive mode selector
        if isinstance(dataset, list) and len(dataset) > 0 and isinstance(dataset[0], (nx.Graph, nx.DiGraph)):
             num_nodes = sum(len(g) for g in dataset)
        else:
             num_nodes = 0 
        
        threshold = getattr(args, 'auto_streaming_threshold', 100000)
        
        # Check if streaming should be used (large graph OR many trials)
        use_streaming = (num_nodes > threshold or args.n_trials > 2000)
        
        logger.info("\nStarting pattern mining...")
        if use_streaming:
            logger.info(f"Adaptive Mode: Enabling Batch Processing for {num_nodes} nodes. 🚀")
            
            # Automatically tune workers for performance vs stability
            total_nodes = num_nodes
            original_workers = args.streaming_workers
            
            if total_nodes > 3500000:
                args.streaming_workers = 0
                reason = "Maximum Stability (Sequential)"
            elif total_nodes > 500000:
                args.streaming_workers = min(original_workers, 2)
                reason = "Balanced Performance (2 workers)"
            else: 
                args.streaming_workers = original_workers
                reason = "Maximum Speed ({} workers)".format(args.streaming_workers)

            if args.streaming_workers != original_workers:
                logger.info(f"⚠ SMART SCALING: Graph size {total_nodes:,} nodes.")
                logger.info(f"  Adjusting streaming_workers: {original_workers} -> {args.streaming_workers} for {reason}.")
            
            # Ensure search phase uses the same optimized worker count
            args.n_workers = args.streaming_workers
            if args.n_workers <= 0:
                logger.info("Sequential Search Mode enabled (n_workers=0)")
            # Pass dataset and then clear local reference
            pattern_growth_streaming(dataset, task, args)
            if isinstance(dataset, list):
                dataset.clear()
            dataset = None
        else:
            logger.info("Adaptive Mode: Standard Sequential Processing.")
            if not hasattr(args, 'n_workers'):
                args.n_workers = 1
            pattern_growth(dataset, task, args)
        
        import gc
        gc.collect()
        logger.info("\n✓ Pattern mining complete!")

    except Exception as e:
        print(f"FATAL ERROR in main: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    # Docker memory fix
    import torch.multiprocessing as mp
    try:
        mp.set_sharing_strategy('file_system')
    except:
        pass
    main()