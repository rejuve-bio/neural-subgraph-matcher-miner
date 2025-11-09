from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from typing import Dict, Optional, Union
import json
import traceback

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    provider: str
    api_key: str
    query: str
    context: Optional[Union[str, dict]] = None  # accept string or object; optional

# In-memory per-session chat histories
_SESSION_STORES: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _SESSION_STORES:
        _SESSION_STORES[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORES[session_id]

# Shared prompt with memory placeholder
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for interpreting frequent subgraph patterns mined by SNAP SPminer of stanford.\n"
     "Use the provided graph context to reason about structure, roles, motifs, and possible implications if it is as expected or different in detail."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

def build_llm(provider: str, api_key: str):
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    elif provider == "gemini":
        # Use env var for compatibility with langchain_google_genai
        os.environ["GOOGLE_API_KEY"] = api_key
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    else:
        return None

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    session_id = request.headers.get("X-Session-Id", "default-session")
    history = get_session_history(session_id)

    try:
        llm = build_llm(req.provider, req.api_key)
        if llm is None:
            return JSONResponse(status_code=400, content={"reply": "Unrecognized provider. Use 'openai' or 'gemini'."})

        # If context is provided, add it as a system message to history
        if req.context:
            ctx = req.context if isinstance(req.context, str) else json.dumps(req.context)
            history.add_message({
                "role": "system",
                "content":
                    "Frequent Subgraph Pattern Miner Context:\n"
                    "- The following graph pattern was mined using a frequent subgraph miner (SNAP SPminer).\n When considerning graph context, the x and y attributes of nodes are for the ui so don't consider them in interpretation.\n"
                    f"- Graph context:\n{ctx}\n\n"
                    "- Please interpret the biological or structural significance of this pattern, including possible functional roles, motifs, or relevant literature."
            })

        chain = PROMPT | llm
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda sid: get_session_history(sid),
            history_messages_key="history",
            input_messages_key="input",
        )

        user_input = req.query

        result = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        content = getattr(result, "content", str(result))
        return {"reply": content}

    except Exception as e:
        print("Chat error:", e, "\n", traceback.format_exc())
        return JSONResponse(status_code=500, content={"reply": f"Server error: {str(e)}"})