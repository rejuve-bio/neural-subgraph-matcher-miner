# Use an official Python 3.7 base image (Debian-based)
FROM python:3.7-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Make pip installs on slower networks
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_NO_BUILD_ISOLATION=1

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libigraph-dev python3-igraph \
    pkg-config \
    git \
    libfreetype6-dev \
    libpng-dev \
    libqhull-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency file first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip/setuptools/wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir igraph

# Core numeric stack
RUN pip install --no-cache-dir numpy==1.21.6

# Visualization and ML libs
RUN pip install --no-cache-dir \
    matplotlib==2.1.1 \
    scikit-learn==1.0.2 \
    seaborn==0.9.0

# Torch + PyG ecosystem (CPU-only)
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir \
    torch-scatter==2.0.2 \
    torch-sparse==0.6.1 \
    torch-cluster==1.5.4 \
    torch-spline-conv==1.2.0 \
    torch-geometric==1.4.3 \
    --find-links https://data.pyg.org/whl/torch-1.4.0+cpu.html

# Other utilities
RUN pip install --no-cache-dir \
    deepsnap==0.1.2 \
    networkx==2.4 \
    test-tube==0.7.5 \
    tqdm==4.43.0 \
    requests

# Install FastAPI and related packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart

# Copy the project
COPY . .

# Expose port
EXPOSE 5000

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]

