"""
Utility functions for the graph visualizer.
"""
import os
import re
import shutil
import logging
from typing import Dict, Any

from .config import MAX_FILENAME_LENGTH

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to be filesystem-safe."""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple consecutive underscores
    filename = re.sub(r'_+', '_', filename)
    
    # Ensure reasonable length
    if len(filename) > MAX_FILENAME_LENGTH:
        name_part = filename.rsplit('.', 1)[0][:90]
        extension = filename.rsplit('.', 1)[1] if '.' in filename else 'html'
        filename = f"{name_part}.{extension}"
    
    return filename


def clear_visualizations(output_dir: str = "plots/cluster", mode: str = "flat") -> None:
    """
    Clears old visualizations from the output directory to ensure consistency.
    
    Args:
        output_dir: The directory containing the visualizations
        mode: "flat" to remove folders (size_X_rank_Y), "folder" to remove flat HTML files
    """
    if not os.path.exists(output_dir):
        return
        
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if mode == "flat":
            # Clear structured folders when switching to flat mode
            if os.path.isdir(item_path) and item.startswith("size_") and "_rank_" in item:
                try:
                    shutil.rmtree(item_path)
                except Exception as e:
                    logger.warning(f"Failed to remove folder {item_path}: {e}")
        elif mode == "folder":
            # Clear flat descriptive files when switching to folder mode
            if os.path.isfile(item_path) and item.endswith("_interactive.html"):
                try:
                    os.remove(item_path)
                except Exception as e:
                    logger.warning(f"Failed to remove file {item_path}: {e}")


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    os.makedirs(directory, exist_ok=True)


def validate_graph_data(graph_data: Dict[str, Any]) -> bool:
    """
    Validate that extracted graph data has the required structure.
    
    Args:
        graph_data: The graph data dictionary to validate
        
    Returns:
        True if valid, False otherwise
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
