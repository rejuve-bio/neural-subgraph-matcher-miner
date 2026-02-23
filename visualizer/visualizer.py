"""
Graph visualization module for neural subgraph matcher.

This module provides functionality to visualize graph patterns as interactive HTML pages.
It extracts graph data, processes templates, and generates browsable visualizations.
"""
import os
import logging
import networkx as nx
import pickle
import torch
import json
from typing import Dict, List, Any, Optional

from .config import DEFAULT_OUTPUT_DIR, DEFAULT_TEMPLATE_NAME
from .extractor import GraphDataExtractor
from .template_processor import HTMLTemplateProcessor
from .pattern_utils import select_representative_pattern, generate_pattern_filename
from .index_generator import IndexHTMLGenerator
from .utils import clear_visualizations, ensure_directory_exists, validate_graph_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_instances_to_json(output_data, args, graph_context=None):
    json_results = []
    if graph_context:
        json_results.append({'type': 'graph_context', 'data': graph_context})
        logger.info("Added graph context to JSON results")
    else:
        logger.info("No graph context provided for JSON results")

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
                    'original_count': pattern_info['count'],
                    'discovery_frequency': pattern_info['original_count'],
                    'duplicates_removed': pattern_info['duplicates_removed'],
                    'frequency_score': pattern_info['count'] / args.n_trials if args.n_trials > 0 else 0
                }
            }

            json_results.append(pattern_data)

    base_path = os.path.splitext(args.out_path)[0]
    json_path = base_path + '_all_instances.json'

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"JSON saved to: {json_path}")
    return json_path


def save_and_visualize_all_instances(agent, args):
    try:
        logger.info("=" * 70)
        logger.info("SAVING AND VISUALIZING ALL PATTERN INSTANCES")
        logger.info("=" * 70)
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

                    'original_count': count,
                    'discovery_hits': original_count,
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

                try:
                    if rank == 1 and size == args.min_pattern_size:
                        output_dir = os.path.join("plots", "cluster")
                        if args.visualize_instances:
                            clear_visualizations(output_dir, mode="folder")
                        else:
                            clear_visualizations(output_dir, mode="flat")

                    if args.visualize_instances:
                        success = visualize_all_pattern_instances(
                            pattern_instances=unique_instances,
                            pattern_key=pattern_key,
                            count=count,
                            output_dir=os.path.join("plots", "cluster"),
                            visualize_instances=True
                        )
                    else:
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

        base_path = os.path.splitext(args.out_path)[0]
        pkl_path = base_path + '_all_instances.pkl'

        logger.info(f"Saving to: {pkl_path}")

        with open(pkl_path, 'wb') as f:
            pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        json_path = save_instances_to_json(output_data, args, graph_context)
        logger.info(f"JSON saved to: {json_path}")

        if os.path.exists(pkl_path):
            file_size = os.path.getsize(pkl_path) / 1024
            logger.info(f"✓ PKL file created successfully ({file_size:.1f} KB)")
        else:
            logger.error("✗ PKL file was not created!")
            return None

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared after visualization.")

        logger.info("=" * 70)
        logger.info("✓ COMPLETE")
        logger.info("=" * 70)
        logger.info(f"PKL file: {pkl_path}")
        logger.info(f"  Pattern types: {len(output_data)}")
        logger.info(f"  Total discoveries: {total_instances}")
        logger.info(f"  Unique instances: {total_unique_instances}")
        logger.info(f"  Duplicates removed: {total_instances - total_unique_instances}")

        if total_instances > 0:
            dup_rate = (total_instances - total_unique_instances) / total_instances * 100
            logger.info(f"  Duplication rate: {dup_rate:.1f}%")

        logger.info(f"HTML visualizations: plots/cluster/")
        logger.info(f"  Successfully created: {total_visualizations} files")

        logger.info("=" * 70)
        return pkl_path

    except Exception as e:
        logger.error(f"FATAL ERROR in save_and_visualize_all_instances: {e}")
        return None

def visualize_pattern_graph_ext(pattern: nx.Graph, args: Any, 
                                count_by_size: Dict[int, int], 
                                pattern_key: Optional[str] = None) -> bool:
    """Main visualizer integration function matching existing API signature."""
    try:
        # Validate input
        if not _validate_pattern_input(pattern):
            return False
        
        # Log graph characteristics
        _log_graph_info(pattern)
        
        # Extract graph data
        graph_data = _extract_pattern_data(pattern, pattern_key)
        if not graph_data:
            return False
        
        # Validate extracted data
        if not validate_graph_data(graph_data):
            logger.error("Extracted graph data failed validation")
            return False
        
        # Generate HTML visualization
        success = _generate_visualization(pattern, graph_data, count_by_size)
        
        if success:
            logger.info("Interactive graph visualization completed successfully")
        
        return success
        
    except KeyboardInterrupt:
        logger.info("Visualization interrupted by user")
        return False
    except MemoryError:
        logger.error("Insufficient memory to process graph visualization")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during graph visualization: {str(e)}")
        logger.debug("Unexpected error details:", exc_info=True)
        return False


def visualize_all_pattern_instances(pattern_instances: List[nx.Graph], 
                                    pattern_key: str, 
                                    count: int,
                                    output_dir: str = DEFAULT_OUTPUT_DIR,
                                    representative_pattern: Optional[nx.Graph] = None,
                                    visualize_instances: bool = False) -> bool:
    """Visualize all instances of a pattern with representative and optional instances."""

    try:
        pattern_dir = os.path.join(output_dir, pattern_key)
        ensure_directory_exists(pattern_dir)
        
        logger.info(f"Visualizing {pattern_key} (visualize_instances={visualize_instances})")
        
        # Cleanup old instance files
        _cleanup_instance_files(pattern_dir)
        
        # Get template path
        template_path = os.path.join(os.path.dirname(__file__), DEFAULT_TEMPLATE_NAME)
        if not os.path.exists(template_path):
            logger.error(f"Template not found: {template_path}")
            return False
        
        processor = HTMLTemplateProcessor(template_path)
        extractor = GraphDataExtractor()
        
        # Create representative visualization
        representative_data, representative_idx = _create_representative_visualization(
            pattern_instances, representative_pattern, extractor, processor, 
            pattern_key, pattern_dir
        )
        
        # Create instance visualizations if requested
        success_count = 0
        if visualize_instances:
            success_count = _create_instance_visualizations(
                pattern_instances, extractor, processor, pattern_key, 
                count, pattern_dir
            )
        else:
            logger.info("  Skipping instance visualizations (visualize_instances=False)")
        
        # Create index.html
        _create_index_html(pattern_key, count, pattern_dir, representative_data, 
                          visualize_instances, representative_idx)
        
        # Log summary
        _log_visualization_summary(visualize_instances, success_count, count, 
                                   pattern_dir, representative_data)
        
        return representative_data is not None
        
    except Exception as e:
        logger.error(f"Failed to visualize pattern: {e}")
        return False


# Private helper functions

def _validate_pattern_input(pattern: nx.Graph) -> bool:
    """Validate pattern input."""
    if pattern is None:
        logger.error("Pattern graph cannot be None")
        return False
        
    if not isinstance(pattern, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        logger.error(f"Pattern must be a NetworkX graph, got {type(pattern)}")
        return False
        
    if len(pattern) == 0:
        logger.error("Pattern graph cannot be empty")
        return False
    
    return True


def _log_graph_info(pattern: nx.Graph) -> None:
    """Log graph characteristics."""
    num_nodes = len(pattern)
    num_edges = pattern.number_of_edges()
    graph_type = "directed" if pattern.is_directed() else "undirected"
    logger.info(f"Processing {graph_type} graph with {num_nodes} nodes and {num_edges} edges")


def _extract_pattern_data(pattern: nx.Graph, pattern_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract graph data from pattern."""
    logger.info("Extracting graph data from NetworkX object...")
    try:
        extractor = GraphDataExtractor()
        graph_data = extractor.extract_graph_data(pattern)
        
        if pattern_key:
            graph_data['metadata']['pattern_key'] = pattern_key
            
        logger.info("Graph data extraction completed successfully")
        return graph_data
    except Exception as e:
        logger.error(f"Graph data extraction failed: {str(e)}")
        logger.debug("Graph data extraction error details:", exc_info=True)
        return None


def _generate_visualization(pattern: nx.Graph, graph_data: Dict[str, Any], 
                           count_by_size: Dict[int, int]) -> bool:
    """Generate HTML visualization file."""
    logger.info("Generating HTML visualization...")
    try:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", DEFAULT_OUTPUT_DIR))
        ensure_directory_exists(output_dir)
        
        # Cleanup structured folders when in flat mode
        clear_visualizations(output_dir, mode="flat")
        
        template_path = os.path.join(os.path.dirname(__file__), DEFAULT_TEMPLATE_NAME)
        processor = HTMLTemplateProcessor(template_path)
        
        # Generate filename based on graph characteristics
        base_filename = generate_pattern_filename(pattern, count_by_size)
        
        # Process template and create HTML file
        output_path = processor.process_template(
            graph_data=graph_data,
            output_filename=base_filename,
            output_dir=output_dir
        )
        
        logger.info(f"HTML visualization created successfully: {output_path}")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Template file not found: {str(e)}")
        logger.info("Make sure template.html exists in the current directory")
        return False
    except Exception as e:
        logger.error(f"HTML generation failed: {str(e)}")
        logger.debug("HTML generation error details:", exc_info=True)
        return False


def _cleanup_instance_files(pattern_dir: str) -> None:
    """Remove stale instance files from pattern directory."""
    if not os.path.exists(pattern_dir):
        return
    
    for item in os.listdir(pattern_dir):
        item_path = os.path.join(pattern_dir, item)
        if os.path.isfile(item_path) and item.startswith("instance_") and item.endswith(".html"):
            try:
                os.remove(item_path)
            except Exception as e:
                logger.warning(f"  Failed to remove stale file {item_path}: {e}")


def _create_representative_visualization(pattern_instances: List[nx.Graph],
                                        representative_pattern: Optional[nx.Graph],
                                        extractor: GraphDataExtractor,
                                        processor: HTMLTemplateProcessor,
                                        pattern_key: str,
                                        pattern_dir: str) -> tuple:
    """Create representative pattern visualization."""
    # Select representative
    if representative_pattern:
        representative = representative_pattern
        logger.info("  Using decoder representative pattern")
    else:
        representative = select_representative_pattern(pattern_instances)
        logger.info("  Selected representative from instances")
    
    representative_data = None
    representative_idx = -1
    
    if representative:
        try:
            representative_data = extractor.extract_graph_data(representative)
            
            # Create descriptive title
            num_nodes = len(representative)
            num_edges = representative.number_of_edges()
            graph_type = "Directed" if representative.is_directed() else "Undirected"
            has_anchors = any(representative.nodes[n].get('anchor', 0) == 1 
                            for n in representative.nodes())
            anchor_info = " with Anchors" if has_anchors else ""
            
            representative_data['metadata']['title'] = \
                f"{graph_type} Pattern ({num_nodes} nodes, {num_edges} edges){anchor_info}"
            representative_data['metadata']['pattern_key'] = pattern_key
            
            processor.process_template(
                graph_data=representative_data,
                output_filename="representative.html",
                output_dir=pattern_dir
            )
            logger.info("  ✓ Created representative visualization")
            
            # Find representative index
            for idx, pattern in enumerate(pattern_instances):
                if pattern is representative:
                    representative_idx = idx
                    break
                    
        except Exception as e:
            logger.error(f"  Failed to create representative: {e}")
            representative_data = None
    
    return representative_data, representative_idx


def _create_instance_visualizations(pattern_instances: List[nx.Graph],
                                    extractor: GraphDataExtractor,
                                    processor: HTMLTemplateProcessor,
                                    pattern_key: str,
                                    count: int,
                                    pattern_dir: str) -> int:
    """Create visualizations for individual pattern instances."""
    success_count = 0
    
    for idx, pattern in enumerate(pattern_instances):
        try:
            graph_data = extractor.extract_graph_data(pattern)
            graph_data['metadata']['title'] = f"{pattern_key} - Instance {idx+1}/{count}"
            graph_data['metadata']['pattern_key'] = pattern_key
            
            filename = f"instance_{idx+1:04d}.html"
            
            processor.process_template(
                graph_data=graph_data,
                output_filename=filename,
                output_dir=pattern_dir
            )
            
            success_count += 1
            
            if (idx + 1) % 10 == 0 or (idx + 1) == count:
                logger.info(f"  Created {idx+1}/{count} instance visualizations")
                
        except Exception as e:
            logger.error(f"  Failed to process instance {idx+1}: {e}")
            continue
    
    return success_count


def _create_index_html(pattern_key: str, count: int, pattern_dir: str,
                      representative_data: Optional[Dict[str, Any]],
                      visualize_instances: bool, representative_idx: int) -> None:
    """Create index.html for pattern browsing."""
    generator = IndexHTMLGenerator()
    generator.create_pattern_index(
        pattern_key=pattern_key,
        count=count,
        pattern_dir=pattern_dir,
        has_representative=(representative_data is not None),
        has_instances=visualize_instances,
        representative_idx=representative_idx
    )


def _log_visualization_summary(visualize_instances: bool, success_count: int,
                               count: int, pattern_dir: str,
                               representative_data: Optional[Dict[str, Any]]) -> None:
    """Log visualization summary."""
    if visualize_instances:
        logger.info(f"✓ Successfully created representative + {success_count}/{count} "
                   f"instance visualizations in {pattern_dir}")
    else:
        logger.info(f"✓ Successfully created representative visualization in {pattern_dir}")


# Convenience functions for backward compatibility

def extract_graph_data(graph: nx.Graph) -> Dict[str, Any]:
    """Convenience function to extract graph data using GraphDataExtractor."""
   
    extractor = GraphDataExtractor()
    return extractor.extract_graph_data(graph)


def process_html_template(graph_data: Dict[str, Any], 
                         template_path: str = DEFAULT_TEMPLATE_NAME,
                         output_filename: Optional[str] = None,
                         output_dir: str = ".") -> str:
    """Convenience function for HTML template processing."""
    processor = HTMLTemplateProcessor(template_path)
    return processor.process_template(graph_data, output_filename, output_dir)