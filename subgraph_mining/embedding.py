import collections
import random
import warnings
import logging

import numpy as np
import torch
from tqdm import tqdm
import networkx as nx
from deepsnap.batch import Batch
from deepsnap.graph import Graph as DSGraph
from torch.utils.data import Dataset, DataLoader

from common import utils
from common import feature_preprocess

logger = logging.getLogger(__name__)

# Dataset class for parallel processing with on-the-fly sampling
def extract_neighborhood(dataset_graph, seed, args, is_directed):
    """
    Unified neighborhood extraction logic to ensure consistency across all datasets.
    """
    nodes_in_bubble = []
    queue = collections.deque([(seed, 0)]) 
    visited = {seed}
    
    while queue and len(nodes_in_bubble) < args.max_neighborhood_size:
        curr, dist = queue.popleft()
        nodes_in_bubble.append(curr)

        if dist < args.radius:
            neighbors = dataset_graph.successors(curr) if is_directed else dataset_graph.neighbors(curr)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    
    # Induce subgraph
    neigh_graph = dataset_graph.subgraph(nodes_in_bubble).copy()
    
    neigh_graph.graph['anchor_node_original'] = seed
    
    # Standardize mapping
    mapping = {node: i for i, node in enumerate(neigh_graph.nodes())}
    neigh_graph = nx.relabel_nodes(neigh_graph, mapping)
    
    new_anchor_id = mapping[seed]
    neigh_graph.graph['anchor_node'] = new_anchor_id
    
    # Label anchor attribute explicitly
    nx.set_node_attributes(neigh_graph, 0, 'anchor')
    neigh_graph.nodes[new_anchor_id]['anchor'] = 1
    
    if neigh_graph.number_of_edges() == 0:
        neigh_graph.add_edge(new_anchor_id, new_anchor_id)
        
    return neigh_graph, new_anchor_id

class TargetedDataset(Dataset):
    def __init__(self, dataset_graph, selected_seeds, args):
        self.dataset_graph = dataset_graph
        self.selected_seeds = selected_seeds
        self.args = args
        self.is_directed = (args.graph_type == "directed")
        self.radius = args.radius

    def __len__(self):
        return len(self.selected_seeds)

    def __getitem__(self, idx):
        seed = self.selected_seeds[idx]
        neigh_graph, new_anchor_id = extract_neighborhood(self.dataset_graph, seed, self.args, self.is_directed)
        std_g = utils.standardize_graph(neigh_graph, anchor=new_anchor_id)
        return DSGraph(std_g)

#  Streaming Dataset for large graphs
class StreamingNeighborhoodDataset(Dataset):
    """On-the-fly neighborhood sampling for batch processing."""
    def __init__(self, dataset, n_neighborhoods, args):
        self.dataset = dataset
        self.n_neighborhoods = n_neighborhoods
        self.args = args
        self.anchors = [0] * n_neighborhoods if args.node_anchored else None

    def __len__(self):
        return self.n_neighborhoods

    def __getitem__(self, idx):
        graph, neigh = utils.sample_neigh(self.dataset,
            random.randint(self.args.min_neighborhood_size,
                self.args.max_neighborhood_size), self.args.graph_type)
        
        neigh_graph = graph.subgraph(neigh).copy()
        neigh_graph = nx.convert_node_labels_to_integers(neigh_graph)
        
        anchor = 0 if self.args.node_anchored else None
        std_graph = utils.standardize_graph(neigh_graph, anchor)
        return DSGraph(std_graph)


# Extracts neighborhoods only when iterated
class LazyNeighborhoodGraphList:
    def __init__(self, dataset_graph, selected_seeds, args):
        self.dataset_graph = dataset_graph
        self.selected_seeds = selected_seeds
        self.args = args
        self.is_directed = (args.graph_type == "directed")
        self.radius = args.radius

    def __len__(self):
        return len(self.selected_seeds)

    def __getitem__(self, idx):
        seed = self.selected_seeds[idx]
        neigh_graph, _ = extract_neighborhood(self.dataset_graph, seed, self.args, self.is_directed)
        return neigh_graph


def collate_fn(ds_graphs):
    """Batching logic for DeepSnap models."""
    batch = Batch.from_data_list(ds_graphs)
    augmenter = feature_preprocess.FeatureAugment()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unknown type of key*')
        batch = augmenter.augment(batch)
    return batch


#  Embedding generation with DataLoader
def generate_target_embeddings(dataset, model, args):
    """Generate embeddings using Targeted Anchor Streaming."""
    logger.info(f"Setting up Batch Processing Pipeline (Batch Size: {args.batch_size})")

    # Reproducibility
    random.seed(42)
    np.random.seed(42)

    dataset_graph = dataset[0] 
    
    # select seeds from the FULL graph first to ensure we start exactly where intended.
    all_nodes = sorted(list(dataset_graph.nodes()))
    
    # Filter out "dead seeds" (isolated nodes) that cannot contain patterns
    # This prevents DeepSnap from crashing on 0-edge subgraphs
    is_directed = (args.graph_type == "directed")
    if is_directed:
        all_nodes = [n for n in all_nodes if dataset_graph.out_degree(n) > 0]
    else:
        all_nodes = [n for n in all_nodes if dataset_graph.degree(n) > 0]
        
    if args.sample_method == "radial":
        selected_seeds = all_nodes
        logger.info(f"Radial Method: Targeting {len(selected_seeds)} non-isolated nodes for coverage.")
    else:
        # 100% Uniform Random Seeding
        n_seeds = args.n_neighborhoods
        if len(all_nodes) <= n_seeds:
            selected_seeds = all_nodes
            logger.info(f"Tree Method: Using all {len(selected_seeds)} nodes as seeds.")
        else:
            selected_seeds = np.random.choice(all_nodes, size=n_seeds, replace=False)
            logger.info(f"Tree Method: Sampled {n_seeds} random seeds for graph coverage.")

    targeted_dataset = TargetedDataset(dataset_graph, selected_seeds, args)
    
    num_workers = args.streaming_workers
    safe_batch_size = min(args.batch_size, 64) if num_workers > 0 else args.batch_size
    pin_memory = torch.cuda.is_available()
    
    def create_loader(w, b):
        return DataLoader(targeted_dataset, batch_size=b, 
                          shuffle=False, collate_fn=collate_fn, 
                          num_workers=w, pin_memory=pin_memory)

    dataloader = create_loader(num_workers, safe_batch_size)

    embs = []
    device = utils.get_device()
    model.to(device)
    
    logger.info(f"Generating embeddings for {len(selected_seeds)} targeted neighborhoods (Batch: {safe_batch_size}, Workers: {num_workers})...")
    
    try:
        for batch in tqdm(dataloader):
            with torch.no_grad():
                emb = model.emb_model(batch.to(device))
                embs.append(emb.to(torch.device("cpu")))
    except RuntimeError as e:
        if "unable to write to file" in str(e) or "shared memory" in str(e).lower():
            logger.warning("Docker SHM Limit Hit! Falling back to 100% stable single-process mode...")
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Fallback to single process
            num_workers = 0
            dataloader = create_loader(0, args.batch_size)
            embs = []
            for batch in tqdm(dataloader, desc="Stable Fallback"):
                with torch.no_grad():
                    emb = model.emb_model(batch.to(device))
                    embs.append(emb.to(torch.device("cpu")))
        else:
            raise e
    lazy_graphs = LazyNeighborhoodGraphList(dataset_graph, selected_seeds, args)
    return embs, lazy_graphs