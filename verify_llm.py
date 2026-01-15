from app.services.llm_service import LLMService

def test_biological_qa():
    print("Initializing LLMService...")
    llm = LLMService()
    
    # Sample biological subgraph data
    graph_data = {
        "nodes": [
            {"id": "1", "label": "Protein A"},
            {"id": "2", "label": "Gene B"},
            {"id": "3", "label": "Metabolite C"}
        ],
        "edges": [
            {"source": "1", "target": "2", "type": "regulates"},
            {"source": "2", "target": "3", "type": "synthesizes"}
        ]
    }
    
    query = "What is the biological significance of this motif?"
    
    print("\nTesting 'openbiollm' choice (now routing to OpenBioLLM)...")
    try:
        response = llm.analyze_motif(
            graph_data=graph_data,
            user_query=query,
            model_choice="openbiollm"
        )
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"Error testing openbiollm replacement: {e}")

    print("\nTesting Gemini 2.5 Flash (preserved)...")
    try:
        # Note: This requires GEMINI_API_KEY to be valid
        response = llm.analyze_motif(
            graph_data=graph_data,
            user_query=query,
            model_choice="gemini"
        )
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"Error testing Gemini: {e}")

if __name__ == "__main__":
    test_biological_qa()
