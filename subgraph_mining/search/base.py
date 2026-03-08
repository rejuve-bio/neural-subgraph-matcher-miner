from collections import defaultdict

class SearchAgent:
    """Class for search strategies to identify frequent subgraphs in embedding space."""

    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20):
        """Subgraph pattern search by walking in embedding space."""
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.model = model
        self.dataset = dataset
        self.embs = embs
        self.node_anchored = node_anchored
        self.analyze = analyze
        self.model_type = model_type
        self.out_batch_size = out_batch_size

    def run_search(self, n_trials=1000):
        self.cand_patterns = defaultdict(list)
        self.counts = defaultdict(lambda: defaultdict(list))
        self.n_trials = n_trials

        self.init_search()
        while not self.is_search_done():
            self.step()
        return self.finish_search()

    def init_search(self):
        raise NotImplementedError

    def step(self):
        """Execute a search step (add a node to the pattern)."""
        raise NotImplementedError
