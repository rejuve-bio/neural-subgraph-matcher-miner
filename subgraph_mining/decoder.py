import argparse
import csv
from itertools import combinations
import time
import os
import pickle

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
from visualizer.visualizer import visualize_pattern_graph_ext
from subgraph_mining.search_agents import GreedySearchAgent, MCTSSearchAgent, MemoryEfficientMCTSAgent, MemoryEfficientGreedyAgent, BeamSearchAgent

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
from sklearn.decomposition import PCA

import warnings 

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
    try:
        num_nodes = len(pattern)
        num_edges = pattern.number_of_edges()
        edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        base_size = max(12, min(20, num_nodes * 2))
        if edge_density > 0.3:  # Dense graph
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
        
        if edge_density > 0.5:  # Very dense
            base_node_size = 2500
            anchor_node_size = base_node_size * 1.3
        elif edge_density > 0.3:  # Dense
            base_node_size = 3500
            anchor_node_size = base_node_size * 1.2
        else:  # Sparse
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
        print(f"Error visualizing pattern graph: {e}")
        return False

def pattern_growth(dataset, task, args):
    start_time = time.time()
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
    print(len(dataset), "graphs")
    print("search strategy:", args.search_strategy)
    print("graph type:", args.graph_type)
    if task == "graph-labeled": print("using label 0")
    
    graphs = []
    for i, graph in enumerate(dataset):
        if task == "graph-labeled" and labels[i] != 0: continue
        if task == "graph-truncate" and i >= 1000: break
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
                print(i)
                for j, node in enumerate(graph.nodes):
                    if len(dataset) <= 10 and j % 100 == 0: print(i, j)
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
            start_time = time.time()
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
        print("WARNING: number of graphs not multiple of batch size")
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
    out_graphs = agent.run_search(args.n_trials)
    
    print(time.time() - start_time, "TOTAL TIME")
    x = int(time.time() - start_time)
    print(x // 60, "mins", x % 60, "secs")

    # Visualize discovered patterns
    count_by_size = defaultdict(int)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    
    successful_visualizations = 0
    for pattern in out_graphs:
        if visualize_pattern_graph_ext(pattern, args, count_by_size):
            successful_visualizations += 1
        count_by_size[len(pattern)] += 1

    print(f"Successfully visualized {successful_visualizations}/{len(out_graphs)} patterns")

    # Save results
    if not os.path.exists("results"):
        os.makedirs("results")
    with open(args.out_path, "wb") as f:
        pickle.dump(out_graphs, f)
    
    return out_graphs

def main():
    if not os.path.exists("plots/cluster"):
        os.makedirs("plots/cluster")

    parser = argparse.ArgumentParser(description='Decoder arguments')
    parse_encoder(parser)
    parse_decoder(parser)
    
    args = parser.parse_args()

    print("Using dataset {}".format(args.dataset))
    print("Graph type: {}".format(args.graph_type))

    # Load dataset based on graph type preference
    if args.dataset.endswith('.pkl'):
        with open(args.dataset, 'rb') as f:
            data = pickle.load(f)
            
            if isinstance(data, (nx.Graph, nx.DiGraph)):
                graph = data
                
                # Convert graph type if needed
                if args.graph_type == "directed" and not graph.is_directed():
                    print("Converting undirected graph to directed...")
                    graph = graph.to_directed()
                elif args.graph_type == "undirected" and graph.is_directed():
                    print("Converting directed graph to undirected...")
                    graph = graph.to_undirected()
                
                graph_type = "directed" if graph.is_directed() else "undirected"
                print(f"Using NetworkX {graph_type} graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
                
                # Show edge direction information if available
                sample_edges = list(graph.edges(data=True))[:3]
                if sample_edges:
                    print("Sample edge attributes:")
                    for u, v, attrs in sample_edges:
                        direction_info = attrs.get('direction', f"{u} -> {v}" if graph.is_directed() else f"{u} -- {v}")
                        edge_type = attrs.get('type', 'unknown')
                        print(f"  {direction_info} (type: {edge_type})")
                
            elif isinstance(data, dict) and 'nodes' in data and 'edges' in data:
                # Create graph based on specified type
                if args.graph_type == "directed":
                    graph = nx.DiGraph()
                else:
                    graph = nx.Graph()
                graph.add_nodes_from(data['nodes'])
                graph.add_edges_from(data['edges'])
                print(f"Created {args.graph_type} graph from dict format with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
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
        # Road networks are typically undirected
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
        # These are typically undirected networks
        graph = nx.Graph() if args.graph_type == "undirected" else nx.DiGraph()
        with open("data/{}".format(fn[args.dataset]), "r") as f:
            for line in f:
                if not line.strip(): continue
                a, b = line.strip().split(" ")
                graph.add_edge(int(a), int(b))
        dataset = [graph]
        task = 'graph'
    elif args.dataset.startswith('plant-'):
        size = int(args.dataset.split("-")[-1])
        dataset = make_plant_dataset(size)
        task = 'graph'

    # Run pattern growth
    pattern_growth(dataset, task, args)

if __name__ == '__main__':
    main()