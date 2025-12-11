import pickle
import networkx as nx
import sys
import os
import time


INPUT_TXT_FILE = 'web-NotreDame.txt' 
OUTPUT_PKL_FILE = 'data/web-NotreDame.pkl' 


def prepare_pkl_from_txt():
    """Reads the web-Google.txt file, builds the graph, and saves it as a clean .pkl."""
    
    print(f"Starting conversion from text edge list: {INPUT_TXT_FILE}...")
    start_time = time.time()
    
    # 1. Check if the input file exists
    if not os.path.exists(INPUT_TXT_FILE):
        print(f"Error: Input file {INPUT_TXT_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    # 2. Build the graph from the text edge list
    
    # Initialize an empty directed graph (since it's a web graph)
    full_graph = nx.DiGraph()
    
    try:
        # Open the text file for reading
        with open(INPUT_TXT_FILE, 'r') as f:
            for line in f:
                # Skip comment lines that usually start with '#'
                if line.startswith('#'):
                    continue
                
                # Split the line into source and target nodes (handles spaces or tabs)
                parts = line.strip().split()
                if len(parts) == 2:
                    # Convert node IDs to integers
                    try:
                        u = int(parts[0])
                        v = int(parts[1])
                        full_graph.add_edge(u, v)
                    except ValueError:
                        # Handle cases where non-integer data might be in edge lines
                        print(f"Skipping line with non-integer data: {line.strip()}", file=sys.stderr)
                        continue

    except Exception as e:
        print(f"Error reading or parsing edge list: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Check graph size
    total_nodes = full_graph.number_of_nodes()
    total_edges = full_graph.number_of_edges()
    print(f"Successfully built graph: {total_nodes} nodes, {total_edges} edges.")

    # 4. Create the data dictionary for SPMiner's format
    # Node/Edge labels are set to None as they were not in the original dataset info.
    new_data = {
        'graph': full_graph,
        'node_labels': None,
        'edge_labels': None,
        'nodes': list(full_graph.nodes()) 
    }

    # 5. Save the new dataset
    # Ensure the 'data' directory exists
    os.makedirs(os.path.dirname(OUTPUT_PKL_FILE), exist_ok=True)
    with open(OUTPUT_PKL_FILE, 'wb') as f:
        # Use protocol=4 for compatibility
        pickle.dump(new_data, f, protocol=4) 
    
    end_time = time.time()
    print(f"Successfully saved clean full dataset to {OUTPUT_PKL_FILE}")
    print(f"Total conversion time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    prepare_pkl_from_txt()