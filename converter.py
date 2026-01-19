from neo4j import GraphDatabase
import networkx as nx
import logging
from typing import Optional, Tuple, List, Dict
import argparse
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jToNetworkX:
    def __init__(self, uri: str, username: str, password: str, batch_size: int = 10000):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.batch_size = batch_size
        
    def _get_node_count(self, session) -> int:
        query = "MATCH (n) RETURN count(n) as count"
        result = session.run(query)
        return result.single()["count"]
    
    def _get_edge_count(self, session) -> int:
        query = "MATCH ()-[r]-() RETURN count(r) as count"
        result = session.run(query)
        return result.single()["count"]

    def load_simplified_graph(self) -> nx.Graph:
        """
        Load graph with simplified attributes compatible with the mining code.
        Only preserves essential attributes in a format suitable for processing.
        """
        try:
            G = nx.Graph()
            node_mapping = {}
            current_node_idx = 0
            
            with self.driver.session() as session:
                # Get total counts
                total_nodes = self._get_node_count(session)
                total_edges = self._get_edge_count(session)
                
                # Process nodes
                logger.info("Processing nodes...")
                skip = 0
                while skip < total_nodes:
                    query = """
                    MATCH (n)
                    RETURN id(n) as node_id, 
                           labels(n) as labels,
                           n.id as custom_id,
                           n.label as custom_label
                    SKIP $skip LIMIT $limit
                    """
                    result = session.run(query, skip=skip, limit=self.batch_size)
                    
                    for record in result:
                        node_id = record["node_id"]
                        if node_id not in node_mapping:
                            node_mapping[node_id] = current_node_idx
                            
                            # Use custom label if available, otherwise use first Neo4j label
                            custom_label = record["custom_label"]
                            neo4j_labels = record["labels"]
                            display_label = (custom_label or 
                                          (neo4j_labels[0] if neo4j_labels else "Node"))
                            
                            # Add node with minimal attributes
                            G.add_node(current_node_idx, 
                                     label=str(display_label),  # ensure string type
                                     id=str(record["custom_id"] or node_id))  # ensure string type
                            current_node_idx += 1
                    
                    skip += self.batch_size
                
                # Process edges with minimal attributes
                logger.info("Processing edges...")
                skip = 0
                while skip < total_edges:
                    query = """
                    MATCH (n)-[r]-(m)
                    RETURN id(n) as source, id(m) as target, 
                            type(r) as edge_type
                    SKIP $skip LIMIT $limit
                    """
                    result = session.run(query, skip=skip, limit=self.batch_size)
    
                    for record in result:
                        src = node_mapping[record["source"]]
                        dst = node_mapping[record["target"]]
                        edge_type = record["edge_type"]
                        # Add edge with all attributes at once
                        G.add_edge(src, dst, weight=1.0, type=str(edge_type))  
    
                    skip += self.batch_size
                
                logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                return G
                
        except Exception as e:
            logger.error(f"Error loading graph from Neo4j: {str(e)}")
            raise
        finally:
            self.driver.close()

def main():
    parser = argparse.ArgumentParser(description='Load Neo4j graph with simplified attributes')
    import os
    bolt_port = os.getenv("NEO4J_BOLT_PORT", "7687")
    parser.add_argument('--uri', default=f'bolt://localhost:{bolt_port}')

    parser.add_argument('--username', default='neo4j')
    parser.add_argument('--password', required=True)
    parser.add_argument('--output', default='graph.pkl')
    parser.add_argument('--batch-size', type=int, default=10000)
    
    args = parser.parse_args()
    
    try:
        converter = Neo4jToNetworkX(args.uri, args.username, args.password, args.batch_size)
        graph = converter.load_simplified_graph()
        
        # Save graph data in a version-independent way
        data_to_save = {
            'nodes': list(graph.nodes(data=True)),
            'edges': list(graph.edges(data=True))
        }
        
        with open(args.output, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Graph saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()