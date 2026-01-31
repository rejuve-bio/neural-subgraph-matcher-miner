"""
Pattern selection and filename generation utilities.
"""
import re
import time
import networkx as nx
from typing import Dict, List, Any, Optional


def select_representative_pattern(pattern_instances: List[nx.Graph]) -> Optional[nx.Graph]:
    """
    Select a representative pattern from a list of instances.
    Uses heuristics like centrality, average degree, etc.
    """
    if not pattern_instances:
        return None

    if len(pattern_instances) == 1:
        return pattern_instances[0]

    # Score each pattern based on various metrics
    scores = []
    for pattern in pattern_instances:
        score = _calculate_pattern_score(pattern)
        scores.append(score)

    # Return pattern with highest score
    max_idx = scores.index(max(scores))
    return pattern_instances[max_idx]


def _calculate_pattern_score(pattern: nx.Graph) -> float:
    """Calculate a score for pattern quality."""
    score = 0.0

    # Prefer patterns with more balanced degree distribution
    if len(pattern) > 1:
        degrees = [pattern.degree(n) for n in pattern.nodes()]
        avg_degree = sum(degrees) / len(degrees)
        degree_variance = sum((d - avg_degree) ** 2 for d in degrees) / len(degrees)
        score -= degree_variance  # Lower variance is better

    # Prefer patterns with anchor nodes (more informative)
    anchor_count = sum(1 for n in pattern.nodes() if pattern.nodes[n].get('anchor', 0) == 1)
    score += anchor_count * 10

    # Prefer patterns with more diverse node attributes
    node_labels = set(pattern.nodes[n].get('label', '') for n in pattern.nodes())
    score += len(node_labels) * 5

    return score


def generate_pattern_filename(pattern: nx.Graph, count_by_size: Dict[int, int]) -> str:
    """Generate filename for pattern visualization based on graph characteristics."""
    try:
        num_nodes = len(pattern)
        num_edges = pattern.number_of_edges()
        
        # Calculate edge density
        edge_density = _calculate_edge_density(pattern, num_nodes, num_edges)
        
        # Build filename components
        components = _build_filename_components(pattern, num_nodes, count_by_size, edge_density)
        
        # Join components and sanitize
        filename = '_'.join(components)
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'_+', '_', filename)
        
        return filename
        
    except Exception:
        # Fallback to simple naming
        timestamp = int(time.time())
        return f"pattern_{timestamp}_interactive"


def _calculate_edge_density(pattern: nx.Graph, num_nodes: int, num_edges: int) -> float:
    """Calculate edge density for the pattern."""
    if num_nodes > 1:
        max_edges = num_nodes * (num_nodes - 1)
        if not pattern.is_directed():
            max_edges //= 2
        return num_edges / max_edges if max_edges > 0 else 0
    return 0


def _build_filename_components(pattern: nx.Graph, num_nodes: int, 
                               count_by_size: Dict[int, int], 
                               edge_density: float) -> List[str]:
    """Build list of filename components."""
    components = []
    
    # 1. Graph type (dir/undir)
    graph_type = "dir" if pattern.is_directed() else "undir"
    components.append(graph_type)
    
    # 2. Size and Rank (size-rank)
    rank = count_by_size.get(num_nodes, 1) if count_by_size else 1
    components.append(f"{num_nodes}-{rank}")
    
    # 3. Node types
    node_types = sorted(set(
        str(pattern.nodes[n].get('label', ''))
        for n in pattern.nodes() 
        if pattern.nodes[n].get('label', '')
    ))
    if node_types:
        components.append('nodes-' + '-'.join(node_types))
    
    # 4. Edge types (simplified for naming)
    edge_types = sorted(set(
        str(data.get('type', ''))
        for u, v, data in pattern.edges(data=True) 
        if data.get('type', '')
    ))
    if edge_types:
        components.append('edges-' + '-'.join(edge_types))
    
    # 5. Anchor nodes
    has_anchors = any(pattern.nodes[n].get('anchor', 0) == 1 for n in pattern.nodes())
    if has_anchors:
        components.append('anchored')
    
    # 6. Density category
    if edge_density > 0.5:
        components.append('very-dense')
    elif edge_density > 0.3:
        components.append('dense')
    else:
        components.append('sparse')
    
    # 7. Interactive indicator
    components.append('interactive')
    
    return components
