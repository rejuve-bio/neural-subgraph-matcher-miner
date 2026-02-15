# Configuration constants for the graph visualizer.
import os

ANNOTATION_TOOL_PORT = os.getenv('ANNOTATION_TOOL_PORT', '3000')
CHAT_API_PORT = os.getenv('CHAT_API_PORT', '9002')

# Color palettes for visualization
NODE_COLOR_PALETTE = [
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

EDGE_COLOR_PALETTE = [
    'rgba(59, 130, 246, 0.6)',   # Blue
    'rgba(34, 197, 94, 0.6)',    # Green
    'rgba(251, 191, 36, 0.6)',   # Yellow
    'rgba(168, 85, 247, 0.6)',   # Purple
    'rgba(236, 72, 153, 0.6)',   # Pink
    'rgba(156, 163, 175, 0.6)',  # Gray
    'rgba(239, 68, 68, 0.6)',    # Red
    'rgba(20, 184, 166, 0.6)',   # Teal
]

# Layout configuration
SPRING_LAYOUT_K = 3.0
SPRING_LAYOUT_ITERATIONS = 50
SPRING_LAYOUT_SEED = 42
LAYOUT_SCALE_FACTOR = 200

# File naming
MAX_FILENAME_LENGTH = 100
DEFAULT_OUTPUT_DIR = "plots/cluster"
DEFAULT_TEMPLATE_NAME = "template.html"

# Node and edge type detection priority
NODE_TYPE_KEYS = ['type', 'label', 'category', 'kind', 'class']
EDGE_TYPE_KEYS = ['type', 'label', 'edge_label', 'relation', 'relationship', 
                  'edge_type', 'predicate', 'category', 'kind', 'name']

# Keys to exclude from metadata extraction
NODE_METADATA_EXCLUDED_KEYS = {'id', 'x', 'y', 'type', 'label', 'anchor'}
EDGE_METADATA_EXCLUDED_KEYS = {'type', 'label', 'source', 'target', 'directed', 
                               'source_label', 'target_label', 'weight', 'x', 'y', 'id'}

# Template validation requirements
REQUIRED_TEMPLATE_ELEMENTS = [
    '<script',
    'const GRAPH_DATA',
    '</script>',
]

# Density thresholds for classification
DENSITY_SPARSE_THRESHOLD = 0.1
DENSITY_MEDIUM_THRESHOLD = 0.5
