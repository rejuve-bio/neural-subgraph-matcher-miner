
import os
import json
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class LLMService:
    _instance = None
    _patterns_cache = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize Gemini model settings and load patterns."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables.")
        
        self._load_patterns()

    def _load_patterns(self):
        """Load patterns from the JSON file in the submodule root."""
        try:
            # Standard path: submodules/neural-subgraph-matcher-miner/results/patterns_all_instances.json
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            json_path = os.path.join(base_dir, "results", "patterns_all_instances.json")
            
            logger.info(f"Looking for patterns at: {json_path}")
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self._patterns_cache = json.load(f)
                logger.info(f"Successfully loaded {len(self._patterns_cache)} patterns from {json_path}")
            else:
                logger.warning(f"Patterns file not found at {json_path}")
                self._patterns_cache = []
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            self._patterns_cache = []

    def _find_pattern_data(self, pattern_key: str) -> Optional[Dict[str, Any]]:
        """Find the full pattern data object for a specific key, with forced reload."""
        # Force reload from disk to ensure we have the results of the latest mining run
        self._load_patterns()
            
        if not self._patterns_cache:
            return None
            
        start_idx = 1 if self._patterns_cache and self._patterns_cache[0].get('type') == 'graph_context' else 0
        
        for item in self._patterns_cache[start_idx:]:
            if item.get('metadata', {}).get('pattern_key') == pattern_key:
                return item
        return None

    def analyze_motif(self, graph_data: Dict[str, Any], user_query: str, pattern_key: Optional[str] = None, api_key: Optional[str] = None) -> str:
        """
        Analyze a motif using Gemini REST API, integrating graph structure and instance context.
        """
        """
        Analyze a motif using Gemini REST API, integrating graph structure and instance context.
        """
        current_api_key = api_key if api_key else self.api_key

        if not current_api_key:
             return "Error: GEMINI_API_KEY not found. Please provide it in the interface or configure it in the environment."

        context_str = ""
        num_instances = "unknown"
        if pattern_key:
            pattern_data = self._find_pattern_data(pattern_key)
            if pattern_data:
                metadata = pattern_data.get('metadata', {})
                num_instances = metadata.get('original_count', metadata.get('count', 0))
                freq_score = metadata.get('frequency_score', 0)
                
                context_str = f"""
                CONTEXT FROM MINING RESULTS:
                - Pattern Key: {pattern_key}
                - Occurrences: {num_instances} instances found in the dataset.
                - Frequency Score: {freq_score}
                - Size: {metadata.get('size')} nodes
                - Rank: {metadata.get('rank')}
                """
                
                instances = pattern_data.get('instances', [])
                if instances:
                    examples = instances[:3]
                    context_str += "\nINSTANCE EXAMPLES (for context on node/edge attributes):\n"
                    for i, inst in enumerate(examples):
                        nodes_attrs = [n.get('label', 'N/A') for n in inst.get('nodes', [])]
                        context_str += f"  Instance {i+1}: Node Labels: {nodes_attrs}\n"

        prompt = f"""
        You are an expert Graph Theory analyst.
        Your task is to interpret the provided graph motif (subgraph pattern) and answer the user's question.
        
        **CRITICAL FOCUS: NETWORK TOPOLOGY**
        **RESPONSE LENGTH GUIDANCE:**
        - If the user asks a SIMPLE, DIRECT question (e.g., "how many instances?", "what is the count?"), provide a SHORT, DIRECT answer (1-2 sentences maximum).
        - If the user asks for analysis, explanation, or interpretation, provide a detailed structural analysis.

        INSTRUCTION FOR FREQUENCY MENTIONS:
        - If the user ONLY asks about frequency/count: Answer directly with just the number.
        - If the user asks for a general explanation/summary: Include 'This pattern occurred {num_instances} times in the sampled data.' in the middle of your response after describing the structure. Do not mention the rank.
        - If the user asks specific questions about nodes/edges: Skip the frequency statement.

        INSTRUCTION: When providing a detailed analysis (NOT for simple counts):
        Do not just list the data. Analyze the STRUCTURE based on what you see.
        - **Connectivity**: How are nodes connected? chains, stars, cycles, cliques?
        - **Topology**: Describe the topology based on the visual structure.
        - **Flow**: If directed, how does information flow? Source -> Sink?
        - **Roles**: What functions do these topological positions suggest?
        
        GRAPH DATA:
        {json.dumps(graph_data, indent=2)}
        
        {context_str}
        
        USER QUESTION: "{user_query}"
        
        Provide a concise, insightful answer focusing on the structural implications of this pattern.
        """
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={current_api_key}"
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "Error: No response generated from Gemini API."
        except Exception as e:
            logger.error(f"Gemini REST API call failed: {e}")
            return f"Error processing your request: {e}"
