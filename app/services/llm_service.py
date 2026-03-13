import os
import json
import logging
import requests
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
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
        """Initialize model settings and load patterns."""
        # Load environment variables from .env if present
        if load_dotenv:
            load_dotenv()
        else:
            logger.warning("python-dotenv not installed. Skipping .env loading.")
        
        # Preserve Gemini (as requested)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

        # OpenRouter (for specialized biological models replacing BioGPT/BioLLM)
        self.openrouter_api_key = os.getenv("Open_AI")
        # Use BIO_MODEL_NAME as priority, fallback to BIOGPT_MODEL_NAME or a default
        self.bio_model = os.getenv("BIO_MODEL_NAME") or os.getenv("BIOGPT_MODEL_NAME")
        
        # If no valid specialized model is set, default to a high-quality free/available model
        if not self.bio_model or any(x in self.bio_model.lower() for x in ["biogpt", "biollm"]):
             # Recommended Biological models on OpenRouter (if available) or strong general fallbacks
             self.bio_model = "meta-llama/llama-3.1-70b-instruct" 

        if not self.gemini_api_key and not self.openrouter_api_key:
            logger.warning("No API keys (GEMINI_API_KEY or Open_AI/OpenRouter) found in environment variables.")
        
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

    def analyze_motif(self, graph_data: Dict[str, Any], user_query: str, pattern_key: Optional[str] = None, api_key: Optional[str] = None, model_choice: str = "gemini") -> str:
        """
        Analyze a motif using the selected model, integrating graph structure and instance context.
        """
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
                    # Limit instances for specialized models to save tokens (max 3), Gemini can handle more (max 5)
                    is_bio_model = model_choice.lower() in ["biogpt", "openbiollm", "biollm"]
                    max_inst = 3 if is_bio_model else 5
                    examples = instances[:max_inst]
                    context_str += "\nINSTANCE EXAMPLES (for context on node/edge attributes):\n"
                    for i, inst in enumerate(examples):
                        nodes_attrs = [n.get('label', 'N/A') for n in inst.get('nodes', [])]
                        context_str += f"  Instance {i+1}: Node Labels: {nodes_attrs}\n"
                    if len(instances) > max_inst:
                        context_str += f"  (+ {len(instances) - max_inst} more instances)\n"

        if model_choice.lower() in ["biogpt", "openbiollm", "biollm"]:
            # Map choice to the new high-quality OpenBioLLM via OpenRouter
            prompt = f"""
            You are an elite Biological Data Scientist. 
            Interpret this biological network motif (subgraph) discovered from a mining process.
            
            STRUCTURE SUMMARY:
            {self._summarize_graph_for_biological_llm(graph_data)}
            
            CONTEXT:
            - This pattern appeared {num_instances} times in the dataset.
            - Pattern Key: {pattern_key}
            
            TASK:
            Provide a high-fidelity biological interpretation of this motif. 
            Focus on what this structural arrangement (topology) suggests about potential biological regulation, feedback loops, or entity interactions.
            Keep the response scientific, concise, and insightful.
            Don't mention pattern key just say this pattern
            
            USER QUERY: "{user_query}"
            """
            return self._call_openrouter(prompt)
        else:
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
            return self._call_gemini(prompt, api_key)

    def _summarize_graph_for_biological_llm(self, graph_data: Dict[str, Any]) -> str:
        """Convert raw graph JSON into a rich natural language summary for specialized models."""
        try:
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            
            labels = list(set(str(n.get('label', 'unknown')) for n in nodes))
            edge_types = list(set(str(e.get('type', 'unknown')) for e in edges))
            
            summary = f"Biological subgraph with {len(nodes)} entities and {len(edges)} relationships.\n"
            summary += f"Entities involved: {', '.join(labels)}\n"
            summary += f"Relationship types: {', '.join(edge_types)}\n"
            
            if edges:
                summary += "Specific Interactions:\n"
                for e in edges[:8]: # Provide more detail for the richer models
                    src = next((n.get('label', 'Node') for n in nodes if str(n.get('id')) == str(e.get('source'))), "Node")
                    tgt = next((n.get('label', 'Node') for n in nodes if str(n.get('id')) == str(e.get('target'))), "Node")
                    rel = e.get('type', 'interacts with')
                    summary += f"- {src} ({rel}) {tgt}\n"
            
            return summary
        except Exception as e:
            logger.error(f"Error summarizing graph for Biological LLM: {e}")
            return "A biological network motif structure."

    def _call_gemini(self, prompt: str, api_key: Optional[str] = None) -> str:
        """Call Gemini REST API."""
        current_api_key = api_key if api_key else self.gemini_api_key
        if not current_api_key:
            return "Error: GEMINI_API_KEY not found. Please provide it in the interface or configure it in the environment."

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={current_api_key}"
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            # Set WaitMsBeforeAsync to a larger value for network requests
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "Error: No response generated from Gemini API."
        except Exception as e:
            logger.error(f"Gemini REST API call failed: {e}")
            return f"Error processing your request with Gemini: {e}"

    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API to access specialized biological models."""
        if not self.openrouter_api_key:
            return "Error: OpenRouter API key (configured as 'Open_AI' in .env) not found."

        try:
            # We use OpenRouter to access OpenBioLLM or other specialized models
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:3000", # Required by OpenRouter
                "X-Title": "Neural Miner"
            }
            payload = {
                "model": self.bio_model, 
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=45)
            if response.status_code != 200:
                logger.error(f"OpenRouter Error Response: {response.text}")
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return "Error: No response generated from OpenRouter API."
        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            return f"Error processing biological request via OpenRouter: {e}"
