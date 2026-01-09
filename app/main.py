import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router as api_router
from .api.chat_routes import router as chat_router

app = FastAPI(title="Neural Miner API")

# Add CORS middleware to allow requests from the visualizer (even if opened locally)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(chat_router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=9002, reload=True)
