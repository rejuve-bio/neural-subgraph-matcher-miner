
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, Optional
from ..services.llm_service import LLMService

router = APIRouter()
llm_service = LLMService()

class ChatRequest(BaseModel):
    graph_data: Dict[str, Any]
    query: str
    pattern_key: Optional[str] = None
    api_key: Optional[str] = None

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Endpoint to analyze a graph motif using Gemini.
    """
    try:
        response = llm_service.analyze_motif(
            graph_data=request.graph_data,
            user_query=request.query,
            pattern_key=request.pattern_key,
            api_key=request.api_key
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
