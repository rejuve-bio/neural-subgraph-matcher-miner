# Neural Subgraph Matcher Miner

**Neural Subgraph Matcher Miner** is a comprehensive framework for neural subgraph mining, explanation, and visualization. It enables researchers and analysts to efficiently discover, interpret, and explore structural motifs in complex networks across multiple domains from bioinformatics and social network analysis to financial forensics and knowledge graphs.

Central to this framework is a neural subgraph miner inspired by **SPMiner**, integrated with modern **Graph Neural Network (GNN)** techniques for scalable motif discovery. The system is designed for:
- **Efficient Motif Discovery**: Discovering recurring patterns in complex networks at scale.
- **AI-Driven Interpretation**: Bridging the gap between discovery and understanding with explanatory insights.
- **Interactive Visualizations**: Dynamic, web-based tools for seamless exploration of identified patterns.

The library provides powerful tools for analyzing graph structures through two complementary approaches: determining whether specific subgraph patterns exist within larger graphs (matching), and discovering frequently occurring patterns across graph datasets (mining). These capabilities are essential for applications in bioinformatics, social network analysis, knowledge graphs, and molecular chemistry.

### What This Library Does

This library implements two primary tasks that work together to provide comprehensive graph analysis:

### 1. Neural Subgraph Matching (NeuroMatch)

**Purpose**: Determine whether a query subgraph pattern exists within a larger target graph.

<details>
<summary><b>Technical Details & Use Cases</b></summary>

**Problem Setup**: Given a query graph **Q** anchored at node **q**, and a target graph **T** anchored at node **v**, the goal is to predict if there exists an isomorphism mapping a subgraph of **T** to **Q**, such that the isomorphism maps **v** to **q**.

**How It Works**: 
The framework uses Graph Neural Networks to map both query and target graphs into a learned embedding space where structural similarities can be measured. The system employs one of two approaches:

- **MLP/Neural Tensor Network + Cross Entropy Loss**: Uses multi-layer perceptrons or neural tensor networks to compare embeddings and classify matches
- **Order Embedding + Max Margin Loss**: Leverages order embeddings that preserve hierarchical relationships between graphs

The model produces a prediction score that indicates the likelihood of a subgraph match, enabling binary classification based on a learned threshold.

**Use Cases**: 
- Finding specific molecular structures in chemical databases
- Identifying network motifs in social or biological networks
- Querying knowledge graphs for specific relationship patterns
</details>

### 2. Frequent Subgraph Mining (SPMiner)

**Purpose**: Automatically discover recurring subgraph patterns that appear frequently across a graph dataset.

<details>
<summary><b>Technical Details & Use Cases</b></summary>

**How It Works**: 
SPMiner is a GNN-based framework that learns to identify common structural patterns without requiring predefined templates. The pipeline consists of two phases:

1. **Training Phase**: The encoder is trained on synthetically generated graph data to learn meaningful graph representations
2. **Mining Phase**: The trained decoder analyzes the target dataset to extract and rank frequent subgraph patterns

**Use Cases**:
- Discovering common protein interaction patterns in biological networks
- Finding frequent communication patterns in social networks
- Identifying recurring transaction patterns in financial networks
- Extracting common molecular substructures from chemical compound databases
</details>

---

##  Key Features

- **Neural Subgraph Matching**: State-of-the-art NeuroMatch algorithm for subgraph isomorphism prediction
- **Frequent Subgraph Mining**: Efficient SPMiner implementation for pattern discovery
- **Interactive Visualizations**: Custom HTML-based visualization engine for exploring discovered patterns
- **RESTful API**: FastAPI-based service for programmatic access to mining and matching capabilities

---


## Project Structure

```
neural-subgraph-matcher-miner/
├── subgraph_matching/          # Neural subgraph matching module
│   ├── train.py               # Training script for GNN encoder
│   ├── test.py                # Testing and evaluation
│   ├── alignment.py           # Query-target alignment utility
│   ├── config.py              # Configuration parameters
│   └── hyp_search.py          # Hyperparameter search
│
├── subgraph_mining/           # Frequent subgraph mining module
│   ├── decoder.py             # SPMiner decoder implementation
│   ├── search_agents.py       # Search algorithms for pattern discovery
│   └── config.py              # Mining configuration
│
├── common/                    # Shared utilities and models
│   ├── models.py              # GNN model architectures
│   ├── data.py                # Dataset loaders and processors
│   ├── utils.py               # Helper functions
│   ├── combined_syn.py        # Synthetic data generation
│   └── feature_preprocess.py  # Feature preprocessing
│
├── app/                       # FastAPI application
│   ├── main.py                # Application entry point
│   ├── api/
│   │   └── routes.py          # API endpoints
│   ├── services/
│   │   └── mining_service.py  # Mining service logic
│   └── config/                # App configuration
│
├── visualizer/                # Interactive visualization engine
│   ├── visualizer.py          # Graph visualization logic
│   └── template.html          # HTML template for interactive graphs
│
├── analyze/                   # Analysis and evaluation tools
│   ├── count_patterns.py      # Pattern frequency counting
│   ├── analyze_embeddings.py  # Embedding analysis
│   └── analyze_pattern_counts.py  # Pattern statistics
│
├── ckpt/                      # Model checkpoints
│   └── model.pt               # Pre-trained encoder weights
│
├── plots/                     # Output directory for visualizations
│   └── cluster/               # Interactive HTML visualizations
│
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── Makefile                   # Build automation
```

##  Setup-environment

### Prerequisites

- Python 3.7+
- pip package manager
- Docker for containerized deployment

### Step 1: Clone the Repository

```bash
git clone https://github.com/rejuve-bio/neural-subgraph-matcher-miner.git
cd neural-subgraph-matcher-miner
```

### Step 2: Set Up Python Environment

#### Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```


### Step 3: Install Core Dependencies

```bash
# Install base requirements
pip install -r requirements.txt
```

### Step 4: Install PyTorch and PyTorch Geometric

```bash
# Install PyTorch (CPU version)
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch Geometric and extensions
pip install torch-scatter==2.0.2 torch-sparse==0.6.1 torch-cluster==1.5.4 torch-spline-conv==1.2.0 torch-geometric==1.4.3 -f https://data.pyg.org/whl/torch-1.4.0+cpu.html
```

### Step 5: Verify Installation

```bash
# Run a simple test
python test.py
```

### Docker Setup (Alternative)

```bash
# Build Docker image
docker build -t neural-miner .

# Run container with API server
docker run -p 5000:5000 neural-miner

# Access API at http://localhost:5000
```

---

##  Usage Guide

### Production Semantic Mining: Train Once, Mine Many Datasets

The production semantic workflow uses a single universal semantic checkpoint.
Users should not retrain the model for every new graph. Instead:

1. Convert the graph to the miner's typed PKL format.
2. Precompute/cache text-label embeddings for that graph's node and edge labels.
3. Run `subgraph_mining.decoder` with the universal checkpoint.

The semantic stack is:

```text
node/edge labels -> frozen MiniLM embeddings cached as .npy files
                 -> label-id features + text-label features
                 -> R-GCN/order-embedding miner
                 -> typed directed motif JSON/PKL + interactive HTML
```

#### One-Time Setup: Hugging Face Token and MiniLM Cache

MiniLM is downloaded from Hugging Face. Create a read-only token at
`https://huggingface.co/settings/tokens`, then set it on the host:

```bash
export HF_TOKEN='hf_your_read_token_here'
mkdir -p artifacts/hf_cache artifacts/label_encoder_cache_minilm_universal
```

Start the semantic utility container with a persistent HF cache:

```bash
docker run --rm -it \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  -e HF_HOME=/hf_cache \
  -e SENTENCE_TRANSFORMERS_HOME=/hf_cache/sentence_transformers \
  -v "$PWD:/app" \
  -v "$PWD/artifacts/hf_cache:/hf_cache" \
  -w /app \
  --name spminer-semantic \
  --entrypoint /bin/bash \
  spminer-semantic:latest
```

Inside the container, verify MiniLM:

```bash
python - <<'PY'
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("MINILM_OK")
print("dim:", model.get_sentence_embedding_dimension())
PY
```

#### Precompute Label Embeddings

Run this before training the universal hybrid model, and run it again for each
new user dataset. The second command verifies that mining/training can run in
`cache_only` mode without contacting Hugging Face.

```bash
python scripts/precompute_label_cache.py \
  --semantic_preset universal,biology,ecommerce,social \
  --graph_pkl path/to/typed_dataset.pkl \
  --cache_dir artifacts/label_encoder_cache_minilm_universal \
  --backend sentence_transformers \
  --label_encoder_name sentence-transformers/all-MiniLM-L6-v2

python scripts/precompute_label_cache.py \
  --semantic_preset universal,biology,ecommerce,social \
  --graph_pkl path/to/typed_dataset.pkl \
  --cache_dir artifacts/label_encoder_cache_minilm_universal \
  --backend cache_only
```

For a new dataset, the cache must cover that dataset's labels.Should either:

1. recompute/update the cache from the new typed PKL, or
2. extract a compatible precomputed cache that already contains those labels.

#### Train the Universal Semantic Checkpoint

This is a maintainer step. Do it once, then reuse the checkpoint for mining.

```bash
python -m subgraph_matching.train \
  --dataset syn-semantic \
  --semantic_preset mixed \
  --semantic_mix_presets universal,biology,ecommerce,social \
  --semantic_mix_weights 0.55,0.15,0.15,0.15 \
  --semantic_mode hybrid_text \
  --label_encoder_backend cache_only \
  --label_encoder_cache_dir artifacts/label_encoder_cache_minilm_universal \
  --label_encoder_name sentence-transformers/all-MiniLM-L6-v2 \
  --text_encoder_dim 384 \
  --text_label_dim 64 \
  --use_label_features \
  --label_feature_dim 16 \
  --encoder_type rgcn_basis \
  --num_relations 64 \
  --num_bases 8 \
  --rel_reg_lambda 1e-4 \
  --label_neg_ratio 0.5 \
  --hard_negative_ratio 0.5 \
  --label_noise 0.10 \
  --batch_size 16 \
  --n_batches 10000 \
  --eval_interval 500 \
  --val_size 512 \
  --n_workers 1 \
  --order_threshold_mode margin \
  --order_margin_factor 0.5 \
  --model_path ckpt/universal/model_universal_hybrid_minilm_rgcn.pt
```

#### Mine a Real Typed PKL with the Universal Semantic Checkpoint

```bash
python -m subgraph_mining.decoder \
  --dataset path/to/typed_dataset.pkl \
  --graph_type directed \
  --model_path ckpt/universal/model_universal_hybrid_minilm_rgcn.pt \
  --vocab_dir artifacts/vocab_my_dataset \
  --n_trials 1000 \
  --n_neighborhoods 10000 \
  --min_pattern_size 3 \
  --max_pattern_size 8 \
  --min_neighborhood_size 20 \
  --max_neighborhood_size 29 \
  --search_strategy greedy \
  --memory_efficient \
  --out_batch_size 3 \
  --streaming_workers 0 \
  --out_path results/my_dataset_semantic_patterns.pkl
```

The checkpoint metadata automatically restores the semantic model settings
(`hybrid_text`, `cache_only`, R-GCN configuration, and label-cache directory).

#### Structural-Only Mining

Use the structural checkpoint when label semantics should be ignored:

```bash
python -m subgraph_mining.decoder \
  --dataset path/to/graph.pkl \
  --graph_type directed \
  --model_path ckpt/phase5/structural_baseline.pt \
  --out_path results/structural_patterns.pkl
```

### 1. Neural Subgraph Matching

#### Train the Encoder

```bash
python -m subgraph_matching.train --node_anchored
```

**Configuration options** (see `subgraph_matching/config.py`):
- `--dataset`: Dataset to use (default: `syn-balanced`)
- `--node_anchored`: Enable node-anchored matching
- `--method_type`: Embedding method (`order` or `mlp`)

#### Run Alignment

```bash
python -m subgraph_matching.alignment --query_path=path/to/query.pkl --target_path=path/to/target.pkl
```

This generates an alignment matrix with matching scores for all node pairs.

### 2. Frequent Subgraph Mining

#### Mine Patterns

```bash
python -m subgraph_mining.decoder --dataset=enzymes --node_anchored
```

**Configuration options** (see `subgraph_mining/config.py`):
- `--dataset`: Dataset to mine (e.g., `enzymes`, `cox2`, `reddit`)
- `--node_anchored`: Enable node-anchored mining
- `--model_path`: Path to trained encoder checkpoint

#### Count and Analyze Patterns

```bash
# Count pattern frequencies
python -m analyze.count_patterns --dataset=enzymes --out_path=results/counts.json --node_anchored

# Analyze pattern statistics
python -m analyze.analyze_pattern_counts --counts_path=results/
```

### 3. Interactive Visualizations

Visualizations are automatically generated during pattern analysis and saved to `plots/cluster/`.

**To view**:
1. Navigate to `plots/cluster/`
2. Open any `.html` file in a web browser
3. Interact with the graph: zoom, pan, hover over nodes/edges for details

**Manual trigger**:
```bash
python -m analyze.analyze_pattern_counts --counts_path=results/
```
### 4. LLM Motif Analysis
## Running the Interpreter

To use the LLM-powered interpreter:

1. **Configure API Key**:
   The interpreter requires a Google Gemini API key. You can provide it in one of the following ways:
   - **Environment Variable**: `export GEMINI_API_KEY="YOUR_KEY_HERE"`
   - **.env File**: Create a `.env` file in the project root with `GEMINI_API_KEY=your_key`
   - **UI Input**: Enter the key directly in the web interface through the provided field.

2. **Start Backend**:
   ```bash
   python3 -m app.main
   ```

3. **Open Visualizer**:
   Open the generated motif interpreter HTML file in your browser to start the interpreter.

### 5. Using the API

#### Start the API Server

```bash
# Using Python
uvicorn app.main:app --host 0.0.0.0 --port 5000

# Using Docker
docker run -p 5000:5000 neural-miner
```

#### API Endpoints

- `POST /mine`: Submit a mining job
- `GET /status/{job_id}`: Check job status
- `GET /results/{job_id}`: Retrieve mining results

---

## Dependencies

### Core Libraries

- **PyTorch 1.4.0**: Deep learning framework
- **PyTorch Geometric 1.4.3**: Graph neural network library
- **DeepSNAP 0.1.2**: Graph data structure synchronization
- **NetworkX 2.4**: Graph manipulation and analysis

<details>
<summary><b>View detailed library functionality</b></summary>

### Key Functionality

**PyTorch Geometric** provides efficient implementations of message-passing GNNs, enabling the library to scale to large graphs.

**DeepSNAP** facilitates seamless synchronization between NetworkX graph objects and PyTorch Geometric Data objects, allowing graph algorithms (subgraph operations, matching) to be executed during training iterations.

### Additional Dependencies

- **FastAPI**: RESTful API framework
- **Uvicorn**: ASGI server
- **Matplotlib**: Visualization
- **scikit-learn**: Machine learning utilities
- **tqdm**: Progress bars

*Full dependency list available in `requirements.txt`*
</details>

---

## Acknowledgements

This project builds upon foundational research from Stanford's SNAP group:

- [Neural Subgraph Learning GNN (NSL)](https://github.com/snap-stanford/neural-subgraph-learning-GNN) - Original framework
- [NeuroMatch: Neural Subgraph Matching](http://snap.stanford.edu/subgraph-matching/) - Matching algorithm
- [SPMiner: Frequent Subgraph Mining](https://snap.stanford.edu/frequent-subgraph-mining/) - Mining algorithm  
- [DeepSNAP](https://github.com/snap-stanford/deepsnap) - Graph data structures

---
