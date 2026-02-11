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

try:
    from visualizer.visualizer import visualize_pattern_graph_ext, visualize_all_pattern_instances
    VISUALIZER_AVAILABLE = True
except ImportError:
    print("WARNING: Could not import visualizer - visualization will be skipped")
    VISUALIZER_AVAILABLE = False
    visualize_pattern_graph_ext = None
    visualize_all_pattern_instances = None


from subgraph_mining.search_agents import (
    GreedySearchAgent, MCTSSearchAgent, 
    MemoryEfficientMCTSAgent, MemoryEfficientGreedyAgent, 
    BeamSearchAgent
)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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



# Dataset class for parallel processing with on-the-fly sampling
def extract_neighborhood(dataset_graph, seed, args, is_directed):
    """
    Unified neighborhood extraction logic to ensure consistency across all datasets.
    """
    nodes_in_bubble = []
    queue = collections.deque([(seed, 0)]) 
    visited = {seed}
    
    while queue and len(nodes_in_bubble) < args.max_neighborhood_size:
        curr, dist = queue.popleft()
        nodes_in_bubble.append(curr)

        if dist < args.radius:
            neighbors = dataset_graph.successors(curr) if is_directed else dataset_graph.neighbors(curr)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    
    # Induce subgraph
    neigh_graph = dataset_graph.subgraph(nodes_in_bubble).copy()
    
    neigh_graph.graph['anchor_node_original'] = seed
    
    # Standardize mapping
    mapping = {node: i for i, node in enumerate(neigh_graph.nodes())}
    neigh_graph = nx.relabel_nodes(neigh_graph, mapping)
    
    new_anchor_id = mapping[seed]
    neigh_graph.graph['anchor_node'] = new_anchor_id
    
    # Label anchor attribute explicitly
    nx.set_node_attributes(neigh_graph, 0, 'anchor')
    neigh_graph.nodes[new_anchor_id]['anchor'] = 1
    
    if neigh_graph.number_of_edges() == 0:
        neigh_graph.add_edge(new_anchor_id, new_anchor_id)
        
    return neigh_graph, new_anchor_id

class TargetedDataset(Dataset):
    def __init__(self, dataset_graph, selected_seeds, args):
        self.dataset_graph = dataset_graph
        self.selected_seeds = selected_seeds
        self.args = args
        self.is_directed = (args.graph_type == "directed")
        self.radius = args.radius

    def __len__(self):
        return len(self.selected_seeds)

    def __getitem__(self, idx):
        seed = self.selected_seeds[idx]
        neigh_graph, new_anchor_id = extract_neighborhood(self.dataset_graph, seed, self.args, self.is_directed)
        std_g = utils.standardize_graph(neigh_graph, anchor=new_anchor_id)
        return DSGraph(std_g)

#  Streaming Dataset for large graphs
class StreamingNeighborhoodDataset(Dataset):
    """On-the-fly neighborhood sampling for batch processing."""
    def __init__(self, dataset, n_neighborhoods, args):
        self.dataset = dataset
        self.n_neighborhoods = n_neighborhoods
        self.args = args
        self.anchors = [0] * n_neighborhoods if args.node_anchored else None

    def __len__(self):
        return self.n_neighborhoods

    def __getitem__(self, idx):
        graph, neigh = utils.sample_neigh(self.dataset,
            random.randint(self.args.min_neighborhood_size,
                self.args.max_neighborhood_size), self.args.graph_type)
        
        neigh_graph = graph.subgraph(neigh).copy()
        neigh_graph = nx.convert_node_labels_to_integers(neigh_graph)
        
        anchor = 0 if self.args.node_anchored else None
        std_graph = utils.standardize_graph(neigh_graph, anchor)
        return DSGraph(std_graph)


# Extracts neighborhoods only when iterated
class LazyNeighborhoodGraphList:
    def __init__(self, dataset_graph, selected_seeds, args):
        self.dataset_graph = dataset_graph
        self.selected_seeds = selected_seeds
        self.args = args
        self.is_directed = (args.graph_type == "directed")
        self.radius = args.radius

    def __len__(self):
        return len(self.selected_seeds)

    def __getitem__(self, idx):
        seed = self.selected_seeds[idx]
        neigh_graph, _ = extract_neighborhood(self.dataset_graph, seed, self.args, self.is_directed)
        return neigh_graph


def collate_fn(ds_graphs):
    """Batching logic for DeepSnap models."""
    from common import feature_preprocess
    batch = Batch.from_data_list(ds_graphs)
    augmenter = feature_preprocess.FeatureAugment()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unknown type of key*')
        batch = augmenter.augment(batch)
    return batch


#  Embedding generation with DataLoader
def generate_target_embeddings(dataset, model, args):
    """Generate embeddings using Targeted Anchor Streaming."""
    logger.info(f"Setting up Batch Processing Pipeline (Batch Size: {args.batch_size})")

    # Reproducibility
    random.seed(42)
    np.random.seed(42)

    dataset_graph = dataset[0] 
    
    # select seeds from the FULL graph first to ensure we start exactly where intended.
    all_nodes = sorted(list(dataset_graph.nodes()), key=str)
    
    # Filter out "dead seeds" (isolated nodes) that cannot contain patterns
    # This prevents DeepSnap from crashing on 0-edge subgraphs
    is_directed = (args.graph_type == "directed")
    if is_directed:
        all_nodes = [n for n in all_nodes if dataset_graph.out_degree(n) > 0]
    else:
        all_nodes = [n for n in all_nodes if dataset_graph.degree(n) > 0]
        
    if args.sample_method == "radial":
        selected_seeds = all_nodes
        logger.info(f"Radial Method: Targeting {len(selected_seeds)} non-isolated nodes for coverage.")
    else:
        # 100% Uniform Random Seeding
        n_seeds = args.n_neighborhoods
        if len(all_nodes) <= n_seeds:
            selected_seeds = all_nodes
            logger.info(f"Tree Method: Using all {len(selected_seeds)} nodes as seeds.")
        else:
            selected_seeds = np.random.choice(all_nodes, size=n_seeds, replace=False)
            logger.info(f"Tree Method: Sampled {n_seeds} random seeds for graph coverage.")

    targeted_dataset = TargetedDataset(dataset_graph, selected_seeds, args)
    
    num_workers = args.streaming_workers
    safe_batch_size = min(args.batch_size, 64) if num_workers > 0 else args.batch_size
    pin_memory = torch.cuda.is_available()
    
    def create_loader(w, b):
        return DataLoader(targeted_dataset, batch_size=b, 
                          shuffle=False, collate_fn=collate_fn, 
                          num_workers=w, pin_memory=pin_memory)

    dataloader = create_loader(num_workers, safe_batch_size)

    embs = []
    device = utils.get_device()
    model.to(device)
    
    logger.info(f"Generating embeddings for {len(selected_seeds)} targeted neighborhoods (Batch: {safe_batch_size}, Workers: {num_workers})...")
    
    try:
        for batch in tqdm(dataloader):
            with torch.no_grad():
                emb = model.emb_model(batch.to(device))
                embs.append(emb.to(torch.device("cpu")))
    except RuntimeError as e:
        if "unable to write to file" in str(e) or "shared memory" in str(e).lower():
            logger.warning("Docker SHM Limit Hit! Falling back to 100% stable single-process mode...")
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Fallback to single process
            num_workers = 0
            dataloader = create_loader(0, args.batch_size)
            embs = []
            for batch in tqdm(dataloader, desc="Stable Fallback"):
                with torch.no_grad():
                    emb = model.emb_model(batch.to(device))
                    embs.append(emb.to(torch.device("cpu")))
        else:
            raise e
    lazy_graphs = LazyNeighborhoodGraphList(dataset_graph, selected_seeds, args)
    return embs, lazy_graphs


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


def visualize_pattern_graph(pattern, args, count_by_size):
    """Visualize a single pattern representative (original function - kept for compatibility)."""
    try:
        num_nodes = len(pattern)
        num_edges = pattern.number_of_edges()
        edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        base_size = max(12, min(20, num_nodes * 2))
        if edge_density > 0.3:
            figsize = (base_size * 1.2, base_size)
        else:
            figsize = (base_size, base_size * 0.8)
        
        plt.figure(figsize=figsize)

        node_labels = {}
        for n in pattern.nodes():
            node_data = pattern.nodes[n]
            node_id = node_data.get('id', str(n))
            node_label = node_data.get('label', 'unknown')
            
            label_parts = [f"{node_label}:{node_id}"]
            
            other_attrs = {k: v for k, v in node_data.items() 
                          if k not in ['id', 'label', 'anchor'] and v is not None}
            
            if other_attrs:
                for key, value in other_attrs.items():
                    if isinstance(value, str):
                        if edge_density > 0.5 and len(value) > 8:
                            value = value[:5] + "..."
                        elif edge_density > 0.3 and len(value) > 12:
                            value = value[:9] + "..."
                        elif len(value) > 15:
                            value = value[:12] + "..."
                    elif isinstance(value, (int, float)):
                        if isinstance(value, float):
                            value = f"{value:.2f}" if abs(value) < 1000 else f"{value:.1e}"
                    
                    if edge_density > 0.5:
                        label_parts.append(f"{key}:{value}")
                    else:
                        label_parts.append(f"{key}: {value}")
            
            if edge_density > 0.5:
                node_labels[n] = "; ".join(label_parts)
            else:
                node_labels[n] = "\n".join(label_parts)

        if edge_density > 0.3:
            if num_nodes <= 20:
                pos = nx.circular_layout(pattern, scale=3)
            else:
                pos = nx.spring_layout(pattern, k=3.0, seed=42, iterations=100)
        else:
            pos = nx.spring_layout(pattern, k=2.0, seed=42, iterations=50)

        unique_labels = sorted(set(pattern.nodes[n].get('label', 'unknown') for n in pattern.nodes()))
        label_color_map = {label: plt.cm.Set3(i) for i, label in enumerate(unique_labels)}

        unique_edge_types = sorted(set(data.get('type', 'default') for u, v, data in pattern.edges(data=True)))
        edge_color_map = {edge_type: plt.cm.tab20(i % 20) for i, edge_type in enumerate(unique_edge_types)}

        colors = []
        node_sizes = []
        shapes = []
        node_list = list(pattern.nodes())
        
        if edge_density > 0.5:
            base_node_size = 2500
            anchor_node_size = base_node_size * 1.3
        elif edge_density > 0.3:
            base_node_size = 3500
            anchor_node_size = base_node_size * 1.2
        else:
            base_node_size = 5000
            anchor_node_size = base_node_size * 1.2
        
        for i, node in enumerate(node_list):
            node_data = pattern.nodes[node]
            node_label = node_data.get('label', 'unknown')
            is_anchor = node_data.get('anchor', 0) == 1
            
            if is_anchor:
                colors.append('red')
                node_sizes.append(anchor_node_size)
                shapes.append('s')
            else:
                colors.append(label_color_map[node_label])
                node_sizes.append(base_node_size)
                shapes.append('o')

        anchor_nodes = []
        regular_nodes = []
        anchor_colors = []
        regular_colors = []
        anchor_sizes = []
        regular_sizes = []
        
        for i, node in enumerate(node_list):
            if shapes[i] == 's':
                anchor_nodes.append(node)
                anchor_colors.append(colors[i])
                anchor_sizes.append(node_sizes[i])
            else:
                regular_nodes.append(node)
                regular_colors.append(colors[i])
                regular_sizes.append(node_sizes[i])

        if anchor_nodes:
            nx.draw_networkx_nodes(pattern, pos, 
                    nodelist=anchor_nodes,
                    node_color=anchor_colors, 
                    node_size=anchor_sizes, 
                    node_shape='s',
                    edgecolors='black', 
                    linewidths=3,
                    alpha=0.9)

        if regular_nodes:
            nx.draw_networkx_nodes(pattern, pos, 
                    nodelist=regular_nodes,
                    node_color=regular_colors, 
                    node_size=regular_sizes, 
                    node_shape='o',
                    edgecolors='black', 
                    linewidths=2,
                    alpha=0.8)

        if edge_density > 0.5:
            edge_width = 1.5
            edge_alpha = 0.6
        elif edge_density > 0.3:
            edge_width = 2
            edge_alpha = 0.7
        else:
            edge_width = 3
            edge_alpha = 0.8
        
        if pattern.is_directed():
            arrow_size = 30 if edge_density < 0.3 else (20 if edge_density < 0.5 else 15)
            connectionstyle = "arc3,rad=0.1" if edge_density < 0.5 else "arc3,rad=0.15"
            
            for u, v, data in pattern.edges(data=True):
                edge_type = data.get('type', 'default')
                edge_color = edge_color_map[edge_type]
                
                nx.draw_networkx_edges(
                    pattern, pos,
                    edgelist=[(u, v)],
                    width=edge_width,
                    edge_color=[edge_color],
                    alpha=edge_alpha,
                    arrows=True,
                    arrowsize=arrow_size,
                    arrowstyle='-|>',
                    connectionstyle=connectionstyle,
                    node_size=max(node_sizes) * 1.3,
                    min_source_margin=15,
                    min_target_margin=15
                )
        else:
            for u, v, data in pattern.edges(data=True):
                edge_type = data.get('type', 'default')
                edge_color = edge_color_map[edge_type]
                
                nx.draw_networkx_edges(
                    pattern, pos,
                    edgelist=[(u, v)],
                    width=edge_width,
                    edge_color=[edge_color],
                    alpha=edge_alpha,
                    arrows=False
                )

        max_attrs_per_node = max(len([k for k in pattern.nodes[n].keys() 
                                     if k not in ['id', 'label', 'anchor'] and pattern.nodes[n][k] is not None]) 
                                for n in pattern.nodes())
        
        if edge_density > 0.5:
            font_size = max(6, min(9, 150 // (num_nodes + max_attrs_per_node * 5)))
        elif edge_density > 0.3:
            font_size = max(7, min(10, 200 // (num_nodes + max_attrs_per_node * 3)))
        else:
            font_size = max(8, min(12, 250 // (num_nodes + max_attrs_per_node * 2)))
        
        for node, (x, y) in pos.items():
            label = node_labels[node]
            node_data = pattern.nodes[node]
            is_anchor = node_data.get('anchor', 0) == 1
            
            if edge_density > 0.5:
                pad = 0.15
            elif edge_density > 0.3:
                pad = 0.2
            else:
                pad = 0.3
            
            bbox_props = dict(
                facecolor='lightcoral' if is_anchor else (1, 0.8, 0.8, 0.6),
                edgecolor='darkred' if is_anchor else 'gray',
                alpha=0.8,
                boxstyle=f'round,pad={pad}'
            )
            
            plt.text(x, y, label, 
                    fontsize=font_size, 
                    fontweight='bold' if is_anchor else 'normal',
                    color='black',
                    ha='center', va='center',
                    bbox=bbox_props)

        if edge_density < 0.5 and num_edges < 25:
            edge_labels = {}
            for u, v, data in pattern.edges(data=True):
                edge_type = (data.get('type') or 
                           data.get('label') or 
                           data.get('input_label') or
                           data.get('relation') or
                           data.get('edge_type'))
                if edge_type:
                    edge_labels[(u, v)] = str(edge_type)

            if edge_labels:
                edge_font_size = max(5, font_size - 2)
                nx.draw_networkx_edge_labels(pattern, pos, 
                          edge_labels=edge_labels, 
                          font_size=edge_font_size, 
                          font_color='black',
                          bbox=dict(facecolor='white', edgecolor='lightgray', 
                                  alpha=0.8, boxstyle='round,pad=0.1'))

        graph_type = "Directed" if pattern.is_directed() else "Undirected"
        has_anchors = any(pattern.nodes[n].get('anchor', 0) == 1 for n in pattern.nodes())
        anchor_info = " (Red squares = anchor nodes)" if has_anchors else ""
        
        total_node_attrs = sum(len([k for k in pattern.nodes[n].keys() 
                                  if k not in ['id', 'label', 'anchor'] and pattern.nodes[n][k] is not None]) 
                             for n in pattern.nodes())
        attr_info = f", {total_node_attrs} total node attrs" if total_node_attrs > 0 else ""
        
        density_info = f"Density: {edge_density:.2f}"
        if edge_density > 0.5:
            density_info += " (Very Dense)"
        elif edge_density > 0.3:
            density_info += " (Dense)"
        else:
            density_info += " (Sparse)"
        
        title = f"{graph_type} Pattern Graph{anchor_info}\n"
        title += f"(Size: {num_nodes} nodes, {num_edges} edges{attr_info}, {density_info})"
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')

        if unique_edge_types and len(unique_edge_types) > 1:
            x_pos = 1.2
            y_pos = 1.0
            
            edge_legend_elements = [
                plt.Line2D([0], [0], 
                          color=color, 
                          linewidth=3, 
                          label=f'{edge_type}')
                for edge_type, color in edge_color_map.items()
            ]
            
            legend = plt.legend(
                handles=edge_legend_elements,
                loc='upper left',
                bbox_to_anchor=(x_pos, y_pos),
                borderaxespad=0.,
                framealpha=0.9,
                title="Edge Types",
                fontsize=9
            )
            legend.get_title().set_fontsize(10)
            
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            plt.tight_layout()

        pattern_info = [f"{num_nodes}-{count_by_size[num_nodes]}"]

        node_types = sorted(set(pattern.nodes[n].get('label', '') for n in pattern.nodes()))
        if any(node_types):
            pattern_info.append('nodes-' + '-'.join(node_types))

        edge_types = sorted(set(pattern.edges[e].get('type', '') for e in pattern.edges()))
        if any(edge_types):
            pattern_info.append('edges-' + '-'.join(edge_types))

        if has_anchors:
            pattern_info.append('anchored')

        if total_node_attrs > 0:
            pattern_info.append(f'{total_node_attrs}attrs')

        if edge_density > 0.5:
            pattern_info.append('very-dense')
        elif edge_density > 0.3:
            pattern_info.append('dense')
        else:
            pattern_info.append('sparse')

        graph_type_short = "dir" if pattern.is_directed() else "undir"
        filename = f"{graph_type_short}_{('_'.join(pattern_info))}"

        plt.savefig(f"plots/cluster/{filename}.png", bbox_inches='tight', dpi=300)
        plt.savefig(f"plots/cluster/{filename}.pdf", bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"Error visualizing pattern graph: {e}")
        return False

def save_instances_to_json(output_data, args, graph_context=None):  
    json_results = []
    # Add graph context as first item if provided  
    if graph_context:  
        json_results.append({  
            'type': 'graph_context',  
            'data': graph_context  
        })
        print("Added graph context to JSON results")   
    else:  
        print("No graph context provided for JSON results")
    for pattern_key, pattern_info in output_data.items():  
        for instance in pattern_info['instances']:  
            pattern_data = {  
                'nodes': [  
                    {  
                        'id': str(node),  
                        'label': instance.nodes[node].get('label', ''),  
                        'anchor': instance.nodes[node].get('anchor', 0),  
                        **{k: v for k, v in instance.nodes[node].items()   
                           if k not in ['label', 'anchor']}  
                    }  
                    for node in instance.nodes()  
                ],  
                'edges': [  
                    {  
                        'source': str(u),  
                        'target': str(v),  
                        'type': data.get('type', ''),  
                        **{k: v for k, v in data.items() if k != 'type'}  
                    }  
                    for u, v, data in instance.edges(data=True)  
                ],  
                'metadata': {  
                    'pattern_key': pattern_key,  
                    'size': pattern_info['size'],  
                    'rank': pattern_info['rank'],  
                    'num_nodes': len(instance),  
                    'num_edges': instance.number_of_edges(),  
                    'is_directed': instance.is_directed(),  
                    'original_count': pattern_info['count'],  # Use unique count as the "true" count
                    'discovery_frequency': pattern_info['original_count'], # Keep raw hits as extra metadata
                    'duplicates_removed': pattern_info['duplicates_removed'],  
                    'frequency_score': pattern_info['count'] / args.n_trials if args.n_trials > 0 else 0
                }  
            }
         
            json_results.append(pattern_data)  
    base_path = os.path.splitext(args.out_path)[0]  
    json_path = base_path + '_all_instances.json'  
      
    # Ensure directory exists    
    os.makedirs(os.path.dirname(json_path), exist_ok=True)    
        
    with open(json_path, 'w') as f:      
        json.dump(json_results, f, indent=2)      
          
    logger.info(f"JSON saved to: {json_path}")    
        
    return json_path  
def update_run_index(json_path, args):  
    """Update index file with run information"""  
    index_file = "results/run_index.json"  
      
    # Load existing index or create new  
    if os.path.exists(index_file):  
        with open(index_file, 'r') as f:  
            index = json.load(f)  
    else:  
        index = {"runs": []}  
      
    # Add current run  
    run_info = {  
        "timestamp": datetime.datetime.now().isoformat(),  
        "filename": os.path.basename(json_path),  
        "full_path": json_path,  
        "dataset": args.dataset,  
        "n_trials": args.n_trials,  
        "graph_type": args.graph_type,  
        "search_strategy": getattr(args, 'search_strategy', 'unknown')  
    }  
      
    index["runs"].append(run_info)  
      
    # Save updated index  
    with open(index_file, 'w') as f:  
        json.dump(index, f, indent=2)
def save_and_visualize_all_instances(agent, args):
    try:
        logger.info("="*70)
        logger.info("SAVING AND VISUALIZING ALL PATTERN INSTANCES")
        logger.info("="*70)
        graph_context = {} 
        if not hasattr(agent, 'counts'):
            logger.error("Agent has no 'counts' attribute!")
            return None
         
        if hasattr(agent, 'dataset'):  
            logger.info(f"Agent has dataset attribute with {len(agent.dataset)} graphs")  
        else:  
            logger.error("Agent has no 'dataset' attribute!")  
            
        if hasattr(agent, 'dataset') and agent.dataset:    
            total_nodes = sum(g.number_of_nodes() for g in agent.dataset)    
            total_edges = sum(g.number_of_edges() for g in agent.dataset)    
            graph_types = set('directed' if g.is_directed() else 'undirected' for g in agent.dataset)    
            
            graph_context = {    
                'num_graphs': len(agent.dataset),    
                'total_nodes': total_nodes,    
                'total_edges': total_edges,    
                'graph_types': list(graph_types),    
                'sampling_trials': args.n_trials,    
                'neighborhoods_sampled': getattr(args, 'n_neighborhoods', 0),    
                'sample_method': getattr(args, 'sample_method', 'unknown'),    
                'min_pattern_size': args.min_pattern_size,    
                'max_pattern_size': args.max_pattern_size    
            }  
            logger.info(f"Graph context created: {graph_context}")  
        else:  
            logger.warning("Skipping graph_context - agent.dataset is empty or missing")  
        
         
        if not graph_context:  
            graph_context = {  
                'num_graphs': 0,  
                'total_nodes': 0,  
                'total_edges': 0,  
                'graph_types': [],  
                'sampling_trials': args.n_trials,  
                'neighborhoods_sampled': getattr(args, 'n_neighborhoods', 0),  
                'sample_method': getattr(args, 'sample_method', 'unknown'),  
                'min_pattern_size': args.min_pattern_size,  
                'max_pattern_size': args.max_pattern_size,  
                'note': 'Dataset not available on agent'  
            }  
            logger.info("Using fallback graph_context")
        if not agent.counts:
            logger.warning("Agent.counts is empty - no patterns to save")
            return None
        
        logger.info(f"Agent.counts has {len(agent.counts)} size categories")
        
        output_data = {}
        total_instances = 0
        total_unique_instances = 0
        total_visualizations = 0
        
        for size in range(args.min_pattern_size, args.max_pattern_size + 1):
            if size not in agent.counts:
                logger.debug(f"No patterns found for size {size}")
                continue
            
            sorted_patterns = sorted(
                agent.counts[size].items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )
            
            logger.info(f"Size {size}: {len(sorted_patterns)} unique pattern types")
            
            for rank, (wl_hash, instances) in enumerate(sorted_patterns[:args.out_batch_size], 1):
                pattern_key = f"size_{size}_rank_{rank}"
                original_count = len(instances)
                
                logger.debug(f"Processing {pattern_key}: {original_count} raw instances")
                
                unique_instances = []
                seen_signatures = set()
                
                for instance in instances:
                    try:
                        node_ids = frozenset(instance.nodes[n].get('id', n) for n in instance.nodes())
                        
                        edges = []
                        for u, v in instance.edges():
                            u_id = instance.nodes[u].get('id', u)
                            v_id = instance.nodes[v].get('id', v)
                            edge = tuple(sorted([u_id, v_id]))
                            edges.append(edge)
                        edge_ids = frozenset(edges)
                        
                        signature = (node_ids, edge_ids)
                        
                        if signature not in seen_signatures:
                            seen_signatures.add(signature)
                            unique_instances.append(instance)
                    
                    except Exception as e:
                        logger.warning(f"Error processing instance in {pattern_key}: {e}")
                        continue
                
                count = len(unique_instances)
                duplicates = original_count - count
                
                output_data[pattern_key] = {
                    'size': size,
                    'rank': rank,
                    'count': count,  
                    'instances': unique_instances,  
                    
                    'original_count': count,      # Aligned with unique instances for user expectation
                    'discovery_hits': original_count, # Raw discovery frequency
                    'duplicates_removed': duplicates,
                    'duplication_rate': duplicates / original_count if original_count > 0 else 0,
                    
                    'frequency_score': count / args.n_trials if args.n_trials > 0 else 0,
                    'discovery_rate': original_count / count if count > 0 else 0,
                    
                    'mining_trials': args.n_trials,
                    'min_pattern_size': args.min_pattern_size,
                    'max_pattern_size': args.max_pattern_size
                }
                
                total_instances += original_count
                total_unique_instances += count
                
                if duplicates > 0:
                    logger.info(
                        f"  {pattern_key}: {count} unique instances "
                        f"(from {original_count}, removed {duplicates} duplicates)"
                    )
                else:
                    logger.info(f"  {pattern_key}: {count} instances")
                
                if VISUALIZER_AVAILABLE:
                    try:
                        from visualizer.visualizer import visualize_all_pattern_instances, visualize_pattern_graph_ext, clear_visualizations
                        
                        # Use top-level imports already defined to avoid context issues
                        
                        # Cleanup once at the start of the batch if needed (using rank=1 as trigger)
                        if rank == 1 and size == args.min_pattern_size:
                            output_dir = os.path.join("plots", "cluster")
                            if args.visualize_instances:
                                clear_visualizations(output_dir, mode="folder")
                            else:
                                clear_visualizations(output_dir, mode="flat")

                        if args.visualize_instances:
                            # Structured folder mode
                            success = visualize_all_pattern_instances(
                                pattern_instances=unique_instances,
                                pattern_key=pattern_key,
                                count=count,
                                output_dir=os.path.join("plots", "cluster"),
                                visualize_instances=True
                            )
                        else:
                            # Flat descriptive file mode
                            # Use first instance as representative (they are same WL hash)
                            representative = unique_instances[0]
                            success = visualize_pattern_graph_ext(
                                pattern=representative,
                                args=args,
                                count_by_size={size: rank},
                                pattern_key=pattern_key
                            )
                        
                        if success:
                            total_visualizations += (count if args.visualize_instances else 1)
                            logger.info(f"    ✓ Visualized {pattern_key}")
                        else:
                            logger.warning(f"    ✗ Visualization failed for {pattern_key}")
                    except Exception as e:
                        logger.error(f"    ✗ Visualization error: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    logger.warning(f"    ⚠ Skipping visualization (visualizer not available)")
        
        ensure_directories()
        
        base_path = os.path.splitext(args.out_path)[0]
        pkl_path = base_path + '_all_instances.pkl'
        
        logger.info(f"Saving to: {pkl_path}")
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Add unique JSON saving  
        json_path = save_instances_to_json(output_data, args, graph_context)    
        logger.info(f"JSON saved to: {json_path}")    
        if os.path.exists(pkl_path):
            file_size = os.path.getsize(pkl_path) / 1024  # KB
            logger.info(f"✓ PKL file created successfully ({file_size:.1f} KB)")
        else:
            logger.error("✗ PKL file was not created!")
            return None
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared after visualization.")
        
        logger.info("="*70)
        logger.info("✓ COMPLETE")
        logger.info("="*70)
        logger.info(f"PKL file: {pkl_path}")
        logger.info(f"  Pattern types: {len(output_data)}")
        logger.info(f"  Total discoveries: {total_instances}")
        logger.info(f"  Unique instances: {total_unique_instances}")
        logger.info(f"  Duplicates removed: {total_instances - total_unique_instances}")
        
        if total_instances > 0:
            dup_rate = (total_instances - total_unique_instances) / total_instances * 100
            logger.info(f"  Duplication rate: {dup_rate:.1f}%")
        
        if VISUALIZER_AVAILABLE:
            logger.info(f"HTML visualizations: plots/cluster/")
            logger.info(f"  Successfully created: {total_visualizations} files")
        
        logger.info("="*70)
        
        return pkl_path
    
    except Exception as e:
        logger.error(f"FATAL ERROR in save_and_visualize_all_instances: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    
    if VISUALIZER_AVAILABLE and visualize_pattern_graph_ext:
        pass
    else:
        logger.warning("⚠ Skipping representative visualization (visualizer not available)")

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