# ==============================================================================
# Inpainter — CUDA Backend Image
# Target: Linux x86_64 with NVIDIA GPU (e.g. RTX 4070)
# CUDA:   12.4.1 + cuDNN runtime
# Python: 3.11
# ==============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python python3.11 1

# ==============================================================================
# Dependencies layer
# ==============================================================================
FROM base AS deps
WORKDIR /app

COPY pyproject.toml .
COPY .python-version .

# Install PyTorch with CUDA 12.4 pre-built wheels
RUN uv pip install --system torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# Install all other python requirements
RUN uv pip install --system \
    "diffusers>=0.30.0" \
    "transformers>=4.44.0" \
    "accelerate>=0.33.0" \
    "Pillow>=10.0.0" \
    "numpy>=1.26.0" \
    "pyyaml>=6.0" \
    "typer>=0.12.0" \
    "fastapi>=0.112.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.0.0" \
    "pydantic-settings>=2.0.0" \
    "python-multipart>=0.0.9" \
    "aiofiles>=23.0.0" \
    "rich>=13.0.0" \
    "psutil>=5.9.0" \
    "httpx>=0.27.0"

# ==============================================================================
# Application runtime
# ==============================================================================
FROM deps AS app
WORKDIR /app

COPY orchestrator/ ./orchestrator/
COPY cli/ ./cli/
COPY tools/ ./tools/
COPY ui/app.py ./ui/app.py
COPY ui/__init__.py ./ui/__init__.py
COPY config/ ./config/

RUN mkdir -p /app/output /app/loras /app/inputs

ENV PYTHONPATH=/app
ENV INPAINTER_DEVICE=cuda
ENV HF_HOME=/app/.cache/huggingface

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "ui.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
