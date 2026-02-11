"""
Graph data extraction and processing.
"""
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional

from .config import (
    NODE_COLOR_PALETTE, 
    EDGE_COLOR_PALETTE,
    SPRING_LAYOUT_K,
    SPRING_LAYOUT_ITERATIONS,
    SPRING_LAYOUT_SEED,
    LAYOUT_SCALE_FACTOR,
    NODE_TYPE_KEYS,
    EDGE_TYPE_KEYS,
    NODE_METADATA_EXCLUDED_KEYS,
    EDGE_METADATA_EXCLUDED_KEYS
)


class GraphDataExtractor:
    """
    Extracts and processes NetworkX graph data for interactive visualization.
    
    This class handles the conversion of NetworkX graph objects into JavaScript-compatible
    data structures that can be embedded in HTML templates for client-side rendering.
    """
    
    def __init__(self):
        """Initialize the graph data extractor."""
        self.color_palette = NODE_COLOR_PALETTE
        self.edge_color_palette = EDGE_COLOR_PALETTE
    
    def extract_graph_data(self, graph: nx.Graph) -> Dict[str, Any]:
        """Extract complete graph data from NetworkX graph."""
        self._validate_graph(graph)
        
        try:
            metadata = self._extract_metadata(graph)
            nodes = self._extract_nodes(graph)
            edges = self._extract_edges(graph)
            legend = self._generate_legend(nodes, edges)
            
            return {
                'metadata': metadata,
                'nodes': nodes,
                'edges': edges,
                'legend': legend
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract graph data: {str(e)}") from e
    
    def _validate_graph(self, graph: nx.Graph) -> None:
        """Validate graph input."""
        if graph is None:
            raise ValueError("Graph cannot be None")
            
        if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            raise TypeError("Input must be a NetworkX graph object")
            
        if len(graph) == 0:
            raise ValueError("Graph cannot be empty")
    
    def _extract_metadata(self, graph: nx.Graph) -> Dict[str, Any]:
        """Extract metadata information from the graph."""
        num_nodes = len(graph)
        num_edges = graph.number_of_edges()
        
        # Calculate density
        density = self._calculate_density(graph, num_nodes, num_edges)
        
        # Generate title based on graph characteristics
        title = self._generate_graph_title(graph)
        
        return {
            'title': title,
            'nodeCount': num_nodes,
            'edgeCount': num_edges,
            'isDirected': graph.is_directed(),
            'density': round(density, 3)
        }
    
    def _calculate_density(self, graph: nx.Graph, num_nodes: int, num_edges: int) -> float:
        """Calculate graph density."""
        if num_nodes > 1:
            max_edges = num_nodes * (num_nodes - 1)
            if not graph.is_directed():
                max_edges //= 2
            return num_edges / max_edges if max_edges > 0 else 0
        return 0
    
    def _generate_graph_title(self, graph: nx.Graph) -> str:
        """Generate descriptive title for the graph."""
        graph_type = "Directed" if graph.is_directed() else "Undirected"
        has_anchors = any(graph.nodes[n].get('anchor', 0) == 1 for n in graph.nodes())
        anchor_info = " with Anchors" if has_anchors else ""
        return f"{graph_type} Graph{anchor_info}"
    
    def _extract_nodes(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Extract node data with positions and attributes."""
        nodes = []
        pos = self._get_node_positions(graph)
        
        for node_key in graph.nodes():
            node_data = graph.nodes[node_key]
            node_dict = self._build_node_dict(node_key, node_data, pos)
            nodes.append(node_dict)
            
        return nodes
    
    def _build_node_dict(self, node_key: Any, node_data: Dict[str, Any], 
                         pos: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Build node dictionary with all required attributes."""
        node_id = str(node_data['id']) if 'id' in node_data and node_data['id'] is not None else str(node_key)
        x, y = pos.get(node_key, (0, 0))
        is_anchor = node_data.get('anchor', 0) == 1
        
        # Build display label from all attributes
        display_label = self._build_display_label(node_data)
        
        node_dict = dict(node_data)
        node_dict.update({
            'id': node_id,
            'x': float(x),
            'y': float(y),
            'anchor': is_anchor,
            'label': node_dict.get('label') or self._get_node_type(node_data),
            'display_label': display_label
        })
        
        return node_dict
    
    def _build_display_label(self, node_data: Dict[str, Any]) -> str:
        """Build display label from node attributes."""
        display_label_parts = []
        for key, value in node_data.items():
            if key not in {'anchor', 'x', 'y'} and value is not None:
                display_label_parts.append(f"{key}: {value}")
        
        return "\\n".join(display_label_parts) if display_label_parts else str(node_data.get('id', ''))
    
    def _extract_edges(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Extract edge data with attributes."""
        edges = []
        for source, target, edge_data in graph.edges(data=True):
            edge_dict = self._build_edge_dict(graph, source, target, edge_data)
            edges.append(edge_dict)
        return edges
    
    def _build_edge_dict(self, graph: nx.Graph, source: Any, target: Any, 
                         edge_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build edge dictionary with all required attributes."""
        source_id = str(graph.nodes[source].get('id', source))
        target_id = str(graph.nodes[target].get('id', target))
        
        edge_dict = dict(edge_data)
        edge_dict.update({
            'source': source_id,
            'target': target_id,
            'directed': graph.is_directed(),
            'label': edge_dict.get('label') or self._get_edge_type(edge_data)
        })
        
        return edge_dict
    
    def _get_node_positions(self, graph: nx.Graph) -> Dict[str, Tuple[float, float]]:
        """Get or generate node positions for layout."""
        # Check if positions already exist in node attributes
        has_positions = all('x' in graph.nodes[n] and 'y' in graph.nodes[n] 
                           for n in graph.nodes())
        
        if has_positions:
            return {n: (graph.nodes[n]['x'], graph.nodes[n]['y']) 
                   for n in graph.nodes()}
        
        # Generate layout using spring layout
        return self._generate_layout(graph)
    
    def _generate_layout(self, graph: nx.Graph) -> Dict[str, Tuple[float, float]]:
        """Generate graph layout using spring algorithm."""
        try:
            pos = nx.spring_layout(
                graph, 
                k=SPRING_LAYOUT_K, 
                iterations=SPRING_LAYOUT_ITERATIONS, 
                seed=SPRING_LAYOUT_SEED
            )
            # Scale positions to reasonable canvas coordinates
            return {n: (pos[n][0] * LAYOUT_SCALE_FACTOR, pos[n][1] * LAYOUT_SCALE_FACTOR) 
                   for n in pos}
        except Exception:
            # Fallback to circular layout
            pos = nx.circular_layout(graph, scale=LAYOUT_SCALE_FACTOR)
            return {n: (pos[n][0], pos[n][1]) for n in pos}
    
    def _get_node_type(self, node_data: Dict[str, Any]) -> str:
        """Determine node type from node attributes."""
        for key in NODE_TYPE_KEYS:
            if key in node_data and node_data[key] is not None:
                return str(node_data[key])
        return 'default'
    
    def _get_edge_type(self, edge_data: Dict[str, Any]) -> str:
        """Determine edge type from edge attributes."""
        for key in EDGE_TYPE_KEYS:
            if key in edge_data and edge_data[key] is not None and edge_data[key] != '':
                return str(edge_data[key])
        
        # If no type found, try to infer from all available attributes
        for key, value in edge_data.items():
            if key not in EDGE_METADATA_EXCLUDED_KEYS and value is not None and value != '':
                return str(value)
        
        return 'default'
    
    def _generate_legend(self, nodes: List[Dict[str, Any]], 
                        edges: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate legend data for node and edge types."""
        node_types = set(node['label'] for node in nodes)
        edge_types = set(edge['label'] for edge in edges)
        
        node_legend = self._create_node_legend(node_types)
        edge_legend = self._create_edge_legend(edge_types)
        
        return {
            'nodeTypes': node_legend,
            'edgeTypes': edge_legend
        }
    
    def _create_node_legend(self, node_types: set) -> List[Dict[str, Any]]:
        """Create legend entries for node types."""
        legend = []
        for i, node_type in enumerate(sorted(node_types)):
            color = self.color_palette[i % len(self.color_palette)]
            legend.append({
                'label': node_type,
                'color': color,
                'description': f"{node_type.title()} nodes"
            })
        return legend
    
    def _create_edge_legend(self, edge_types: set) -> List[Dict[str, Any]]:
        """Create legend entries for edge types."""
        legend = []
        for i, edge_type in enumerate(sorted(edge_types)):
            color = self.edge_color_palette[i % len(self.edge_color_palette)]
            legend.append({
                'label': edge_type,
                'color': color,
                'description': f"{edge_type.replace('_', ' ').title()} edges"
            })
        return legend
