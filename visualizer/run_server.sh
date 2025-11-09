#!/bin/bash
# filepath: /home/dagim/Pictures/neural-subgraph-matcher-miner/visualizer/run_server.sh

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
# Activate your virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then   
        echo "Activating virtual environment..."
        source .venv/bin/activate
        # Install requirements
        pip install -r requirements.txt
    else
        echo "Virtual environment not found. Please create it first."
        exit 1
    fi
fi

# Start the FastAPI server
uvicorn chatbot_server:app --host 0.0.0.0 --port 8000 --reload