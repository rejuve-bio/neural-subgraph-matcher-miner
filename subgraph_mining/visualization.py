import os
import json
import pickle
import logging
import datetime

import torch
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import utils

logger = logging.getLogger(__name__)

try:
    from visualizer.visualizer import visualize_pattern_graph_ext, visualize_all_pattern_instances, clear_visualizations
    VISUALIZER_AVAILABLE = True
except ImportError:
    print("WARNING: Could not import visualizer - visualization will be skipped")
    VISUALIZER_AVAILABLE = False
    visualize_pattern_graph_ext = None
    visualize_all_pattern_instances = None
    clear_visualizations = None


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
