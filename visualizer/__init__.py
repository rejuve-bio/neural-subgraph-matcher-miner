"""
Graph Visualizer Package

A modular package for visualizing NetworkX graphs as interactive HTML pages.

Main Functions:
    - visualize_pattern_graph_ext: Visualize a single pattern graph
    - visualize_all_pattern_instances: Visualize pattern with all instances
    - extract_graph_data: Extract graph data for visualization
    - process_html_template: Process HTML template with graph data

Modules:
    - config: Configuration constants
    - extractor: Graph data extraction
    - template_processor: HTML template processing
    - pattern_utils: Pattern selection and filename generation
    - index_generator: Index HTML generation
    - utils: Utility functions
"""

from .visualizer import (
    visualize_pattern_graph_ext,
    visualize_all_pattern_instances,
    extract_graph_data,
    process_html_template,
)

from .extractor import GraphDataExtractor
from .template_processor import HTMLTemplateProcessor
from .pattern_utils import select_representative_pattern, generate_pattern_filename
from .index_generator import IndexHTMLGenerator
from .utils import clear_visualizations, validate_graph_data

__version__ = "2.0.0"

__all__ = [
    # Main API functions
    "visualize_pattern_graph_ext",
    "visualize_all_pattern_instances",
    "extract_graph_data",
    "process_html_template",
    
    # Classes
    "GraphDataExtractor",
    "HTMLTemplateProcessor",
    "IndexHTMLGenerator",
    
    # Utility functions
    "select_representative_pattern",
    "generate_pattern_filename",
    "clear_visualizations",
    "validate_graph_data",
]
