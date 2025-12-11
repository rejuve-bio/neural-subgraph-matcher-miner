import argparse
import csv
from itertools import combinations
import time
import os
import pickle
import sys
from pathlib import Path

from deepsnap.batch import Batch
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import TUDataset, PPI
from torch_geometric.datasets import Planetoid, KarateClub, QM7b
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm

from common import data
from common import models
from common import utils
from common import combined_syn
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder

# CRITICAL: Import visualizer at top level (not inside functions)
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
import torch.multiprocessing as mp
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


def pattern_growth_streaming(dataset, task, args):
    graph = dataset[0]
    graph_chunks = process_large_graph_in_chunks(graph, chunk_size=args.chunk_size)
    dataset = graph_chunks

    all_discovered_patterns = []

    total_chunks = len(dataset)
    chunk_args = [(chunk_dataset, task, args, idx, total_chunks) for idx, chunk_dataset in enumerate(dataset)]

    with mp.Pool(processes=4) as pool:
        results = pool.map(_process_chunk, chunk_args)

    for chunk_out_graphs in results:
        if chunk_out_graphs:
            all_discovered_patterns.extend(chunk_out_graphs)

    return all_discovered_patterns


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


def save_and_visualize_all_instances(agent, args, representative_patterns=None):
    try:
        logger.info("="*70)
        logger.info("SAVING AND VISUALIZING ALL PATTERN INSTANCES")
        logger.info("="*70)

        if not hasattr(agent, 'counts'):
            logger.error("Agent has no 'counts' attribute!")
            return None

        if not agent.counts:
            logger.warning("Agent.counts is empty - no patterns to save")
            return None

        logger.info(f"Agent.counts has {len(agent.counts)} size categories")

        # Build a mapping from WL hash to representative pattern
        representative_map = {}
        if representative_patterns:
            logger.info(f"Building representative pattern mapping for {len(representative_patterns)} patterns...")
            for rep_pattern in representative_patterns:
                wl = utils.wl_hash(rep_pattern, node_anchored=args.node_anchored)
                representative_map[wl] = rep_pattern
            logger.info(f"  Mapped {len(representative_map)} representative patterns")

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
                    
                    'original_count': original_count,  
                    'duplicates_removed': duplicates,
                    'duplication_rate': duplicates / original_count if original_count > 0 else 0,
                    
                    'frequency_score': original_count / args.n_trials if args.n_trials > 0 else 0,
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
                
                # Check if user wants to visualize instances
                visualize_instances = getattr(args, 'visualize_instances', False)

                if visualize_instances and VISUALIZER_AVAILABLE and visualize_all_pattern_instances:
                    try:
                        # Get the representative pattern for this WL hash
                        representative_pattern = representative_map.get(wl_hash, None)

                        if representative_pattern:
                            logger.info(f"    Using decoder representative pattern for {pattern_key}")
                        else:
                            logger.warning(f"    No decoder representative found for {pattern_key}, will select from instances")

                        logger.info(f"    Mode: Visualizing representative + {count} instances in subdirectory")

                        success = visualize_all_pattern_instances(
                            pattern_instances=unique_instances,
                            pattern_key=pattern_key,
                            count=count,
                            output_dir=os.path.join("plots", "cluster"),
                            representative_pattern=representative_pattern,
                            visualize_instances=True
                        )
                        if success:
                            total_visualizations += count
                            logger.info(f"    ✓ Visualized representative + {count} instances in {pattern_key}/")
                        else:
                            logger.warning(f"    ✗ Visualization failed for {pattern_key}")
                    except Exception as e:
                        logger.error(f"    ✗ Visualization error: {e}")
                elif not visualize_instances:
                    logger.info(f"    Mode: Representatives will be visualized directly in plots/cluster/ (no subdirectories)")
                else:
                    logger.warning(f"    ⚠ Skipping visualization (visualizer not available)")
        
        ensure_directories()
        
        base_path = os.path.splitext(args.out_path)[0]
        pkl_path = base_path + '_all_instances.pkl'
        
        logger.info(f"Saving to: {pkl_path}")
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if os.path.exists(pkl_path):
            file_size = os.path.getsize(pkl_path) / 1024  # KB
            logger.info(f"✓ PKL file created successfully ({file_size:.1f} KB)")
        else:
            logger.error("✗ PKL file was not created!")
            return None
        
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


def pattern_growth(dataset, task, args):
    """Main pattern mining function."""
    start_time = time.time()
    
    ensure_directories()
    agent = None
    # Load model
    if args.method_type == "end2end":
        model = models.End2EndOrder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    
    model.to(utils.get_device())
    model.eval()
    model.load_state_dict(torch.load(args.model_path,
        map_location=utils.get_device()))

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
        pkl_path = save_and_visualize_all_instances(agent, args, out_graphs)

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

    # Only create direct representative visualizations if --visualize_instances is NOT set
    # (When --visualize_instances IS set, representatives are already in subdirectories)
    visualize_instances = getattr(args, 'visualize_instances', False)

    if not visualize_instances and VISUALIZER_AVAILABLE and visualize_pattern_graph_ext:
        logger.info("\nVisualizing representative patterns directly in plots/cluster/...")
        for pattern in out_graphs:
            if visualize_pattern_graph_ext(pattern, args, count_by_size):
                successful_visualizations += 1
            count_by_size[len(pattern)] += 1

        logger.info(f"✓ Visualized {successful_visualizations}/{len(out_graphs)} representative patterns")
    elif visualize_instances:
        logger.info("\nSkipping direct representative visualization (representatives already in subdirectories)")
    else:
        logger.warning("⚠ Skipping representative visualization (visualizer not available)")

    ensure_directories()
    
    logger.info(f"\nSaving representative patterns to: {args.out_path}")
    
    if not os.path.exists("results"):
        os.makedirs("results")
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
        
    return out_graphs


def main():
    ensure_directories()

    parser = argparse.ArgumentParser(description='Decoder arguments')
    parse_encoder(parser)
    parse_decoder(parser)
    
    args = parser.parse_args()

    dataset = None  # Initialize dataset to avoid UnboundLocalError
    task = args.graph_type 

    # PATCH START: Force load custom graph if path is provided 
    if args.graph_pkl_path:
        logger.info(f"Loading custom graph from: {args.graph_pkl_path}")
        try:
            with open(args.graph_pkl_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            # Assume the custom file contains a 'graph' key holding the NetworkX graph
            graph_list = [data_dict['graph']]
            dataset = graph_list
            task = "graph" # Default task for custom graph

            logger.info(f"Successfully loaded custom graph: {dataset[0].number_of_nodes()} nodes, {dataset[0].number_of_edges()} edges.")
            
        except Exception as e:
            logger.error(f"FATAL ERROR loading custom graph from {args.graph_pkl_path}: {e}")
            sys.exit(1)
    #PATCH END

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

    logger.info("\nStarting pattern mining...")
    pattern_growth(dataset, task, args)
    logger.info("\n✓ Pattern mining complete!")


if __name__ == '__main__':
    main()