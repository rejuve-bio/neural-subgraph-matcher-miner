import networkx as nx
import json
import traceback
from typing import Dict, List, Any, Tuple, Optional
import math
import random
import time
import os
import re


class GraphDataExtractor:
    """
    Extracts and processes NetworkX graph data for interactive visualization.
    
    This class handles the conversion of NetworkX graph objects into JavaScript-compatible
    data structures that can be embedded in HTML templates for client-side rendering.
    """
    
    def __init__(self):
        """Initialize the graph data extractor."""
        self.color_palette = [
            'rgba(59, 130, 246, 0.7)',   # Blue
            'rgba(34, 197, 94, 0.7)',    # Green  
            'rgba(251, 191, 36, 0.7)',   # Yellow
            'rgba(168, 85, 247, 0.7)',   # Purple
            'rgba(236, 72, 153, 0.7)',   # Pink
            'rgba(156, 163, 175, 0.7)',  # Gray
            'rgba(239, 68, 68, 0.7)',    # Red
            'rgba(20, 184, 166, 0.7)',   # Teal
            'rgba(245, 101, 101, 0.7)',  # Light Red
            'rgba(129, 140, 248, 0.7)',  # Indigo
        ]
        
        self.edge_color_palette = [
            'rgba(59, 130, 246, 0.6)',   # Blue
            'rgba(34, 197, 94, 0.6)',    # Green
            'rgba(251, 191, 36, 0.6)',   # Yellow
            'rgba(168, 85, 247, 0.6)',   # Purple
            'rgba(236, 72, 153, 0.6)',   # Pink
            'rgba(156, 163, 175, 0.6)',  # Gray
            'rgba(239, 68, 68, 0.6)',    # Red
            'rgba(20, 184, 166, 0.6)',   # Teal
        ]
    
    def extract_graph_data(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Extract complete graph data from NetworkX graph.
        
        Args:
            graph: NetworkX graph object to extract data from
            
        Returns:
            Dictionary containing metadata, nodes, edges, and legend data
            
        Raises:
            ValueError: If graph is None or empty
            TypeError: If graph is not a NetworkX graph
        """
        if graph is None:
            raise ValueError("Graph cannot be None")
            
        if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            raise TypeError("Input must be a NetworkX graph object")
            
        if len(graph) == 0:
            raise ValueError("Graph cannot be empty")
        
        try:
            # Extract basic graph information
            metadata = self._extract_metadata(graph)
            
            # Extract nodes with positions and attributes
            nodes = self._extract_nodes(graph)
            
            # Extract edges with attributes
            edges = self._extract_edges(graph)
            
            # Generate legend data
            legend = self._generate_legend(nodes, edges)
            
            return {
                'metadata': metadata,
                'nodes': nodes,
                'edges': edges,
                'legend': legend
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract graph data: {str(e)}") from e
    
    def _extract_metadata(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Extract metadata information from the graph.
        
        Args:
            graph: NetworkX graph object
            
        Returns:
            Dictionary containing graph metadata
        """
        num_nodes = len(graph)
        num_edges = graph.number_of_edges()
        
        # Calculate density
        if num_nodes > 1:
            max_edges = num_nodes * (num_nodes - 1)
            if not graph.is_directed():
                max_edges //= 2
            density = num_edges / max_edges if max_edges > 0 else 0
        else:
            density = 0
        
        # Generate title based on graph characteristics
        graph_type = "Directed" if graph.is_directed() else "Undirected"
        has_anchors = any(graph.nodes[n].get('anchor', 0) == 1 for n in graph.nodes())
        anchor_info = " with Anchors" if has_anchors else ""
        
        title = f"{graph_type} Graph{anchor_info}"
        
        return {
            'title': title,
            'nodeCount': num_nodes,
            'edgeCount': num_edges,
            'isDirected': graph.is_directed(),
            'density': round(density, 3)
        }
    
    def _extract_nodes(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        nodes = []
        pos = self._get_node_positions(graph)
        for node_key in graph.nodes():
            node_data = graph.nodes[node_key]
            node_id = str(node_data['id']) if 'id' in node_data and node_data['id'] is not None else str(node_key)
            x, y = pos.get(node_key, (0, 0))
            is_anchor = node_data.get('anchor', 0) == 1

            # Build display label from all attributes except anchor, x, y, id, label
            display_label_parts = []
            for key, value in node_data.items():
                if key not in {'anchor', 'x', 'y'} and value is not None:
                    display_label_parts.append(f"{key}: {value}")
            print(f"Node {node_id} display label parts: {display_label_parts}")
            display_label = "\\n".join(display_label_parts) if display_label_parts else node_id

            node_dict = dict(node_data)
            node_dict['id'] = node_id
            node_dict['x'] = float(x)
            node_dict['y'] = float(y)
            node_dict['anchor'] = is_anchor
            # Ensure 'label' is the type/category (not a display label)
            if 'label' not in node_dict or node_dict['label'] is None:
                node_dict['label'] = self._get_node_type(node_data)
            node_dict['display_label'] = display_label  # <-- Add this line

            nodes.append(node_dict)
        return nodes
    
    def _extract_edges(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        edges = []
        for source, target, edge_data in graph.edges(data=True):
            source_id = str(graph.nodes[source].get('id', source))
            target_id = str(graph.nodes[target].get('id', target))
            edge_dict = dict(edge_data)
            edge_dict['source'] = source_id
            edge_dict['target'] = target_id
            edge_dict['directed'] = graph.is_directed()
            # Ensure 'label' is the type/category (not a display label)
            if 'label' not in edge_dict or edge_dict['label'] is None:
                edge_dict['label'] = self._get_edge_type(edge_data)
            edges.append(edge_dict)
        return edges
    
    def _get_node_positions(self, graph: nx.Graph) -> Dict[str, Tuple[float, float]]:
        """
        Get or generate node positions for layout.
        
        Args:
            graph: NetworkX graph object
            
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        # Check if positions already exist in node attributes
        has_positions = all('x' in graph.nodes[n] and 'y' in graph.nodes[n] 
                           for n in graph.nodes())
        
        if has_positions:
            return {n: (graph.nodes[n]['x'], graph.nodes[n]['y']) 
                   for n in graph.nodes()}
        
        # Generate layout using spring layout with good parameters
        try:
            pos = nx.spring_layout(graph, k=3.0, iterations=50, seed=42)
            # Scale positions to reasonable canvas coordinates
            scale_factor = 200
            return {n: (pos[n][0] * scale_factor, pos[n][1] * scale_factor) 
                   for n in pos}
        except Exception:
            # Fallback to circular layout
            pos = nx.circular_layout(graph, scale=200)
            return {n: (pos[n][0], pos[n][1]) for n in pos}
    
    def _get_node_type(self, node_data: Dict[str, Any]) -> str:
        """
        Determine node type from node attributes.
        
        Args:
            node_data: Dictionary of node attributes
            
        Returns:
            String representing the node type
        """
        # Priority order for type determination
        type_keys = ['type', 'label', 'category', 'kind', 'class']
        
        for key in type_keys:
            if key in node_data and node_data[key] is not None:
                return str(node_data[key])
        
        return 'default'
    
    def _get_edge_type(self, edge_data: Dict[str, Any]) -> str:
        """
        Determine edge type from edge attributes.

        """
        # Priority order for type determination
        type_keys = ['type', 'label', 'relation', 'category', 'kind']
        
        for key in type_keys:
            if key in edge_data and edge_data[key] is not None:
                return str(edge_data[key])
        
        return 'default'
    
    def _generate_node_label(self, node_id: Any, node_data: Dict[str, Any]) -> str:
        """
        Generate display label for a node.
        """
        # Try to use explicit label first
        if 'label' in node_data and node_data['label'] is not None:
            return str(node_data['label'])
        
        # Use node ID as fallback
        return str(node_id)
    
    def _generate_edge_label(self, edge_data: Dict[str, Any]) -> str:
        """
        Generate display label for an edge.
        """
        # Priority order for label determination
        label_keys = ['label', 'type', 'relation', 'weight']
        
        for key in label_keys:
            if key in edge_data and edge_data[key] is not None:
                value = edge_data[key]
                # Format numeric values nicely
                if isinstance(value, float):
                    return f"{value:.2f}" if abs(value) < 100 else f"{value:.1e}"
                return str(value)
        
        return ''
    
    def _extract_node_metadata(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from node attributes, excluding special keys.
        """
        # Keys to exclude from metadata
        excluded_keys = {'id', 'x', 'y', 'type', 'label', 'anchor'}
        
        metadata = {}
        for key, value in node_data.items():
            if key not in excluded_keys and value is not None:
                # Format values for JSON serialization
                if isinstance(value, (int, float, str, bool, list, dict)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
        
        return metadata
    
    def _extract_edge_metadata(self, edge_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from edge attributes, excluding special keys.
        """
        # Keys to exclude from metadata
        excluded_keys = {'type', 'label', 'source', 'target', 'directed'}
        
        metadata = {}
        for key, value in edge_data.items():
            if key not in excluded_keys and value is not None:
                # Format values for JSON serialization
                if isinstance(value, (int, float, str, bool, list, dict)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
        
        return metadata
    
    def _generate_legend(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        node_types = set(node['label'] for node in nodes)
        edge_types = set(edge['label'] for edge in edges)

        node_legend = []
        for i, node_type in enumerate(sorted(node_types)):
            color = self.color_palette[i % len(self.color_palette)]
            node_legend.append({
                'label': node_type,
                'color': color,
                'description': f"{node_type.title()} nodes"
            })

        edge_legend = []
        for i, edge_type in enumerate(sorted(edge_types)):
            color = self.edge_color_palette[i % len(self.edge_color_palette)]
            edge_legend.append({
                'label': edge_type,
                'color': color,
                'description': f"{edge_type.replace('_', ' ').title()} edges"
            })

        return {
            'nodeTypes': node_legend,
            'edgeTypes': edge_legend
        }


def extract_graph_data(graph: nx.Graph) -> Dict[str, Any]:
    """
    Convenience function to extract graph data using GraphDataExtractor.
    """
    extractor = GraphDataExtractor()
    return extractor.extract_graph_data(graph)

def validate_graph_data(graph_data: Dict[str, Any]) -> bool:
    """
    Validate that extracted graph data has the required structure.
    """
    try:
        # Check required top-level keys
        required_keys = ['metadata', 'nodes', 'edges', 'legend']
        if not all(key in graph_data for key in required_keys):
            return False
        
        # Check metadata structure
        metadata = graph_data['metadata']
        metadata_keys = ['title', 'nodeCount', 'edgeCount', 'isDirected', 'density']
        if not all(key in metadata for key in metadata_keys):
            return False
        
        # Check nodes structure
        nodes = graph_data['nodes']
        if not isinstance(nodes, list) or len(nodes) == 0:
            return False
        
        # Validate first node structure
        node_keys = ['id', 'x', 'y', 'label', 'anchor']
        if not all(key in nodes[0] for key in node_keys):
            return False
        
        # Check edges structure
        edges = graph_data['edges']
        if not isinstance(edges, list):
            return False
        
        # If edges exist, validate structure
        if len(edges) > 0:
            edge_keys = ['source', 'target', 'directed', 'label']
            if not all(key in edges[0] for key in edge_keys):
                return False
        
        # Check legend structure
        legend = graph_data['legend']
        if not isinstance(legend, dict):
            return False
        
        legend_keys = ['nodeTypes', 'edgeTypes']
        if not all(key in legend for key in legend_keys):
            return False
        
        return True
        
    except Exception:
        return False

def handle_extraction_errors(func):
    """
    Decorator to handle common extraction errors with informative messages.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            print(f"Graph validation error: {e}")
            return None
        except TypeError as e:
            print(f"Graph type error: {e}")
            return None
        except RuntimeError as e:
            print(f"Graph extraction error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during graph extraction: {e}")
            traceback.print_exc()
            return None
    
    return wrapper

@handle_extraction_errors
def safe_extract_graph_data(graph: nx.Graph) -> Optional[Dict[str, Any]]:
    """
    Safely extract graph data with comprehensive error handling.
    """
    return extract_graph_data(graph)


class HTMLTemplateProcessor:
    """
    Processes HTML templates and injects graph data for visualization.
    """
    
    def __init__(self, template_path: str = "template.html"):
        """
        Initialize the HTML template processor.
        """
        self.template_path = template_path
        self.template_content = None
        
    def read_template(self) -> str:
        """
        Read template.html file from filesystem.
        """
        try:
            with open(self.template_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            if not content.strip():
                raise ValueError(f"Template file {self.template_path} is empty")
                
            # Validate that it's a proper HTML template with required sections
            if not self._validate_template_structure(content):
                raise ValueError(f"Template file {self.template_path} is missing required structure")
                
            self.template_content = content
            return content
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Template file not found: {self.template_path}")
        except IOError as e:
            raise IOError(f"Failed to read template file {self.template_path}: {str(e)}")
    
    def _validate_template_structure(self, content: str) -> bool:
        """
        Validate that template has required structure for data injection.
        """
        required_elements = [
            '<script>',
            'const GRAPH_DATA',
            '</script>',
            '<canvas id="graph-canvas">',
            '<div id="legend-content">'
        ]
        
        return all(element in content for element in required_elements)
    
    def inject_graph_data(self, template_content: str, graph_data: Dict[str, Any]) -> str:
        """
        Build data injection system to embed graph data into JavaScript section.
        """
        if not template_content or not template_content.strip():
            raise ValueError("Template content cannot be empty")
            
        if not graph_data or not isinstance(graph_data, dict):
            raise ValueError("Graph data must be a non-empty dictionary")
            
        # Validate graph data structure
        if not validate_graph_data(graph_data):
            raise ValueError("Graph data has invalid structure")
        
        try:
            # Convert graph data to JSON with proper formatting
            json_data = json.dumps(graph_data, indent=8, ensure_ascii=False)
            
            # Find the GRAPH_DATA placeholder and replace it
            # Look for the pattern: const GRAPH_DATA = { ... };
            import re
            
            # Pattern to match the GRAPH_DATA assignment
            pattern = r'const GRAPH_DATA\s*=\s*\{[^}]*\}(?:\s*,\s*\{[^}]*\})*\s*;'
            
            # Create replacement with properly formatted JSON
            replacement = f'const GRAPH_DATA = {json_data};'
            
            # Perform the replacement
            if re.search(pattern, template_content, re.DOTALL):
                injected_content = re.sub(pattern, replacement, template_content, flags=re.DOTALL)
            else:
                # Fallback: look for simpler pattern
                simple_pattern = r'const GRAPH_DATA\s*=\s*[^;]+;'
                if re.search(simple_pattern, template_content, re.DOTALL):
                    injected_content = re.sub(simple_pattern, replacement, template_content, flags=re.DOTALL)
                else:
                    raise RuntimeError("Could not find GRAPH_DATA placeholder in template")
            
            # Verify injection was successful
            if 'const GRAPH_DATA' not in injected_content:
                raise RuntimeError("Data injection failed - GRAPH_DATA not found in result")
                
            return injected_content
            
        except json.JSONEncodeError as e:
            raise RuntimeError(f"Failed to serialize graph data to JSON: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Data injection failed: {str(e)}")
    
    def generate_filename(self, graph_data: Dict[str, Any], base_name: str = "pattern") -> str:
        """
        Implement filename generation based on graph characteristics.
        """
        if not graph_data or not isinstance(graph_data, dict):
            raise ValueError("Graph data must be a non-empty dictionary")
            
        if 'metadata' not in graph_data:
            raise ValueError("Graph data must contain metadata section")
        
        metadata = graph_data['metadata']
        
        try:
            # Extract characteristics for filename generation
            node_count = metadata.get('nodeCount', 0)
            edge_count = metadata.get('edgeCount', 0)
            is_directed = metadata.get('isDirected', False)
            density = metadata.get('density', 0)
            
            # Generate descriptive filename components
            components = [base_name]
            
            # Add node count
            components.append(f"{node_count}n")
            
            # Add edge count
            components.append(f"{edge_count}e")
            
            # Add direction indicator
            if is_directed:
                components.append("directed")
            else:
                components.append("undirected")
            
            # Add density category
            if density < 0.1:
                components.append("sparse")
            elif density < 0.5:
                components.append("medium")
            else:
                components.append("dense")
            
            # Join components with underscores
            filename = "_".join(components) + ".html"
            
            # Ensure filename is filesystem-safe
            filename = self._sanitize_filename(filename)
            
            return filename
            
        except Exception as e:
            # Fallback to simple naming scheme
            timestamp = int(time.time()) if 'time' in globals() else random.randint(1000, 9999)
            return f"{base_name}_{timestamp}.html"
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to be filesystem-safe.
        """
        import re
        
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove multiple consecutive underscores
        filename = re.sub(r'_+', '_', filename)
        
        # Ensure reasonable length
        if len(filename) > 100:
            name_part = filename.rsplit('.', 1)[0][:90]
            extension = filename.rsplit('.', 1)[1] if '.' in filename else 'html'
            filename = f"{name_part}.{extension}"
        
        return filename
    
    def write_html_file(self, content: str, filename: str, output_dir: str = ".") -> str:
        """
        Add file writing functionality to create new HTML files.
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
            
        if not filename or not filename.strip():
            raise ValueError("Filename cannot be empty")
        
        # Ensure filename has .html extension
        if not filename.lower().endswith('.html'):
            filename += '.html'
        
        # Create full path
        import os
        full_path = os.path.join(output_dir, filename)
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Write the file
            with open(full_path, 'w', encoding='utf-8') as file:
                file.write(content)
            
            # Verify file was written successfully
            if not os.path.exists(full_path):
                raise IOError(f"File was not created: {full_path}")
                
            # Verify file has content
            if os.path.getsize(full_path) == 0:
                raise IOError(f"File was created but is empty: {full_path}")
            
            return full_path
            
        except IOError as e:
            raise IOError(f"Failed to write HTML file {full_path}: {str(e)}")
        except Exception as e:
            raise IOError(f"Unexpected error writing file {full_path}: {str(e)}")
    
    def process_template(self, graph_data: Dict[str, Any], 
                        output_filename: Optional[str] = None,
                        output_dir: str = ".") -> str:
        """
        Complete template processing workflow: read, inject, and write.
        """
        try:
            # Step 1: Read template file
            template_content = self.read_template()
            
            # Step 2: Inject graph data
            injected_content = self.inject_graph_data(template_content, graph_data)
            
            # Step 3: Generate filename if not provided
            if output_filename is None:
                output_filename = self.generate_filename(graph_data)
            
            # Step 4: Write HTML file
            output_path = self.write_html_file(injected_content, output_filename, output_dir)
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Template processing failed: {str(e)}")
        
def process_html_template(graph_data: Dict[str, Any], 
                         template_path: str = "template.html",
                         output_filename: Optional[str] = None,
                         output_dir: str = ".") -> str:
    """
    Convenience function for HTML template processing.
    """
    processor = HTMLTemplateProcessor(template_path)
    return processor.process_template(graph_data, output_filename, output_dir)

def visualize_pattern_graph_ext(pattern, args, count_by_size, save_dir="plots/cluster" ):
    """
    Main visualizer integration function matching existing API signature.
    """
    import logging
    
    # Configure logging for comprehensive error tracking
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input parameters
        if pattern is None:
            logger.error("Pattern graph cannot be None")
            return False
            
        if not isinstance(pattern, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            logger.error(f"Pattern must be a NetworkX graph, got {type(pattern)}")
            return False
            
        if len(pattern) == 0:
            logger.error("Pattern graph cannot be empty")
            return False
        
        # Log graph characteristics for debugging
        num_nodes = len(pattern)
        num_edges = pattern.number_of_edges()
        graph_type = "directed" if pattern.is_directed() else "undirected"
        logger.info(f"Processing {graph_type} graph with {num_nodes} nodes and {num_edges} edges")
        
        # Step 1: Extract graph data from NetworkX graph
        logger.info("Extracting graph data from NetworkX object...")
        try:
            extractor = GraphDataExtractor()
            graph_data = extractor.extract_graph_data(pattern)
            logger.info("Graph data extraction completed successfully")
        except Exception as e:
            logger.error(f"Graph data extraction failed: {str(e)}")
            logger.debug("Graph data extraction error details:", exc_info=True)
            return False
        
        # Step 2: Validate extracted graph data
        logger.info("Validating extracted graph data...")
        if not validate_graph_data(graph_data):
            logger.error("Extracted graph data failed validation")
            return False
        logger.info("Graph data validation passed")
        
        # Step 3: Generate HTML visualization
        logger.info("Generating HTML visualization...")
        try:
            import os
            # Ensure output directory exists
            output_dir = os.path.abspath(save_dir)
            os.makedirs(output_dir, exist_ok=True)

            template_path = os.path.join(os.path.dirname(__file__), "template.html")
            processor = HTMLTemplateProcessor(template_path)
            
            base_filename = _generate_pattern_filename(pattern, count_by_size)
            
            output_path = processor.process_template(
                graph_data=graph_data,
                output_filename=base_filename,
                output_dir=output_dir
            )
            logger.info(f"HTML visualization created successfully: {output_path}")

        except FileNotFoundError as e:
            logger.error(f"Template file not found: {str(e)}")
            logger.info("Make sure template.html exists in the current directory")
            return False
        except Exception as e:
            logger.error(f"HTML generation failed: {str(e)}")
            logger.debug("HTML generation error details:", exc_info=True)
            return False
        
        # Step 4: Update success counter (for compatibility with existing pipeline)
        if count_by_size is not None and isinstance(count_by_size, dict):
            try:
                # Increment success counter for this graph size
                if num_nodes in count_by_size:
                    # This maintains compatibility with the existing counting system
                    logger.info(f"Graph size {num_nodes} has {count_by_size[num_nodes]} occurrences")
                else:
                    logger.info(f"New graph size {num_nodes} encountered")
            except Exception as e:
                logger.warning(f"Counter update failed (non-critical): {str(e)}")
        
        logger.info("Interactive graph visualization completed successfully")
        return True
        
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
    
def _generate_pattern_filename(pattern: nx.Graph, count_by_size: Dict[int, int]) -> str:
    """
    Generate filename for pattern visualization based on graph characteristics.
    """
    try:
        num_nodes = len(pattern)
        num_edges = pattern.number_of_edges()
        
        # Calculate edge density
        if num_nodes > 1:
            max_edges = num_nodes * (num_nodes - 1)
            if not pattern.is_directed():
                max_edges //= 2
            edge_density = num_edges / max_edges if max_edges > 0 else 0
        else:
            edge_density = 0
        
        # Build filename components
        components = []
        
        # Graph type
        graph_type = "dir" if pattern.is_directed() else "undir"
        components.append(graph_type)
        
        # Node and occurrence count
        occurrence_count = count_by_size.get(num_nodes, 1) if count_by_size else 1
        components.append(f"{num_nodes}-{occurrence_count}")
        
        # Node types
        node_types = sorted(set(
            pattern.nodes[n].get('label', '') 
            for n in pattern.nodes() 
            if pattern.nodes[n].get('label', '')
        ))
        if node_types:
            components.append('nodes-' + '-'.join(node_types))
        
        # Edge types
        edge_types = sorted(set(
            data.get('type', '') 
            for u, v, data in pattern.edges(data=True) 
            if data.get('type', '')
        ))
        if edge_types:
            components.append('edges-' + '-'.join(edge_types))
        
        # Anchor nodes
        has_anchors = any(pattern.nodes[n].get('anchor', 0) == 1 for n in pattern.nodes())
        if has_anchors:
            components.append('anchored')
        
        # Density category
        if edge_density > 0.5:
            components.append('very-dense')
        elif edge_density > 0.3:
            components.append('dense')
        else:
            components.append('sparse')
        
        # Interactive indicator
        components.append('interactive')
        
        # Join components
        filename = '_'.join(components)
        
        # Sanitize filename
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'_+', '_', filename)
        
        return filename
        
    except Exception:
        # Fallback to simple naming
        timestamp = int(time.time())
        return f"pattern_{timestamp}_interactive"