# HTML template processing and generation.
import os
import json
import re
import time
import random
from typing import Dict, Any, Optional

from .config import (
    DEFAULT_TEMPLATE_NAME, 
    REQUIRED_TEMPLATE_ELEMENTS, 
    DENSITY_SPARSE_THRESHOLD, 
    DENSITY_MEDIUM_THRESHOLD,
    ANNOTATION_TOOL_PORT,
    CHAT_API_PORT
)
from .utils import sanitize_filename, validate_graph_data


class HTMLTemplateProcessor:
    # Processes HTML templates and injects graph data for visualization.
    
    def __init__(self, template_path: str = DEFAULT_TEMPLATE_NAME):
        # Initialize the HTML template processor.
        
        self.template_path = template_path
        self.template_content = None
        
    def read_template(self) -> str:
        # Read template.html file from filesystem.
        
        try:
            with open(self.template_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            if not content.strip():
                raise ValueError(f"Template file {self.template_path} is empty")
                
            if not self._validate_template_structure(content):
                raise ValueError(f"Template file {self.template_path} is missing required structure")
                
            self.template_content = content
            return content
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Template file not found: {self.template_path}")
        except IOError as e:
            raise IOError(f"Failed to read template file {self.template_path}: {str(e)}")
    
    def _validate_template_structure(self, content: str) -> bool:
        # Validate that template has required structure for data injection.
        return all(element in content for element in REQUIRED_TEMPLATE_ELEMENTS)
    
    def inject_graph_data(self, template_content: str, graph_data: Dict[str, Any]) -> str:
        # Inject graph data into JavaScript section of template.
        
        if not template_content or not template_content.strip():
            raise ValueError("Template content cannot be empty")
            
        if not graph_data or not isinstance(graph_data, dict):
            raise ValueError("Graph data must be a non-empty dictionary")
            
        if not validate_graph_data(graph_data):
            raise ValueError("Graph data has invalid structure")
        
        try:
            json_data = json.dumps(graph_data, indent=8, ensure_ascii=False)
            replacement = f'const GRAPH_DATA = {json_data};'
            
            
            patterns = [
                r'const GRAPH_DATA\s*=\s*\{[^}]*\}(?:\s*,\s*\{[^}]*\})*\s*;', # Complex legacy
                r'/\* const GRAPH_DATA\s*=\s*[^;]+\s* \*/',                   # Modular placeholder
                r'const GRAPH_DATA\s*=\s*null\s*;',                          # Default modular
                r'const GRAPH_DATA\s*=\s*[^;]+;'                             # Simple fallback
            ]
            
            injected_content = template_content
            data_injected = False
            
            for pattern in patterns:
                if re.search(pattern, injected_content, re.DOTALL):
                    injected_content = re.sub(pattern, replacement, injected_content, flags=re.DOTALL)
                    data_injected = True
                    break
            
            if not data_injected:
                # If no pattern matched, try a direct injection before the first script
                if '</head>' in injected_content:
                    injected_content = injected_content.replace('</head>', f'<script>{replacement}</script></head>', 1)
                    data_injected = True
                else:
                    raise RuntimeError("Could not find suitable GRAPH_DATA placeholder or target in template")
            
            return injected_content
            
        except json.JSONEncodeError as e:
            raise RuntimeError(f"Failed to serialize graph data to JSON: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Data injection failed: {str(e)}")
    
    def _inject_port_configuration(self, template_content: str) -> str:
        # Inject port configuration into template, replacing hardcoded ports.
        
        # Replace annotation tool port (localhost:3000)
        template_content = re.sub(
            r'localhost:3000',
            f'localhost:{ANNOTATION_TOOL_PORT}',
            template_content
        )
        
        # Replace chat API port (localhost:9002)
        template_content = re.sub(
            r'localhost:9002',
            f'localhost:{CHAT_API_PORT}',
            template_content
        )
        
        return template_content
    
    def generate_filename(self, graph_data: Dict[str, Any], base_name: str = "pattern") -> str:
        # Generate filename based on graph characteristics.
        
        if not graph_data or not isinstance(graph_data, dict):
            raise ValueError("Graph data must be a non-empty dictionary")
            
        if 'metadata' not in graph_data:
            raise ValueError("Graph data must contain metadata section")
        
        metadata = graph_data['metadata']
        
        try:
            node_count = metadata.get('nodeCount', 0)
            edge_count = metadata.get('edgeCount', 0)
            is_directed = metadata.get('isDirected', False)
            density = metadata.get('density', 0)
            
            components = [
                base_name,
                f"{node_count}n",
                f"{edge_count}e",
                "directed" if is_directed else "undirected",
                self._get_density_category(density)
            ]
            
            filename = "_".join(components) + ".html"
            return sanitize_filename(filename)
            
        except Exception:
            # Fallback to simple naming scheme
            timestamp = int(time.time()) if 'time' in globals() else random.randint(1000, 9999)
            return f"{base_name}_{timestamp}.html"
    
    def _get_density_category(self, density: float) -> str:
        """Categorize graph density."""
        if density < DENSITY_SPARSE_THRESHOLD:
            return "sparse"
        elif density < DENSITY_MEDIUM_THRESHOLD:
            return "medium"
        else:
            return "dense"
    
    def write_html_file(self, content: str, filename: str, output_dir: str = ".") -> str:
        # Write HTML content to file.
        
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
            
        if not filename or not filename.strip():
            raise ValueError("Filename cannot be empty")
        
        # Ensure filename has .html extension
        if not filename.lower().endswith('.html'):
            filename += '.html'
        
        full_path = os.path.join(output_dir, filename)
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as file:
                file.write(content)
            
            # Verify file was written successfully
            if not os.path.exists(full_path):
                raise IOError(f"File was not created: {full_path}")
                
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
        # Complete template processing workflow: read, inject, and write.
        try:
            template_content = self.read_template()
            
            # Inject port configuration
            template_content = self._inject_port_configuration(template_content)
            
            # Inject graph data
            injected_content = self.inject_graph_data(template_content, graph_data) 
            
            if output_filename is None:
                output_filename = self.generate_filename(graph_data)
            
            output_path = self.write_html_file(injected_content, output_filename, output_dir)
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Template processing failed: {str(e)}")

