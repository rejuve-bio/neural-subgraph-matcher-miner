# Graph Visualizer Package
# A modular package for visualizing NetworkX graphs as interactive HTML pages.

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
