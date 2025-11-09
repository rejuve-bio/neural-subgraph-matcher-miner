## Running the Server

Use the provided script to start the FastAPI backend:

```bash
bash run_server.sh
```

The server will start at [http://localhost:8000](http://localhost:8000).

## Usage

1. Open any html file containing mined results in your browser.
2. Enter your API key (OpenAI or Gemini) in the chatbot panel and save.
3. Interact with the graph and ask questions in the chatbot. The context is sent only once per session for efficient memory usage.

- **API Keys:**  
  Enter your OpenAI (`sk-...`) or Gemini (`AIza...`) key in the chatbot UI.
- **Session Memory:**  
  Each browser tab/session maintains its own chat history and context.