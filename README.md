# Inpainter

> **N-Pass Multi-Character Scene Generator using Hugging Face `diffusers` & FLUX.1-dev**

Inpainter is a Python-native system for generating complex scenes with **multiple, visually consistent characters**. Each character is placed into a staged, sequential inpainting pass and conditioned by its own prompt, LoRA, and optional ControlNet — eliminating identity bleeding and maintaining pixel-perfect scene integrity without external dependencies like ComfyUI.

---

## ⚡ Hardware Support & Platform Comparison

| Platform | Compute Backend | Precision | Recommended VRAM / RAM | Speed (30 steps, 1024×1024) | Native Support | Docker Support |
|---|---|---|---|---|---|---|
| **Ubuntu Rig (RTX 4070)** | NVIDIA CUDA 12.4 | `bfloat16` | 12 GB VRAM + 32 GB RAM | 🚀 ~30–75s per pass | ✅ Full | ✅ Full (via NVIDIA Container Toolkit) |
| **MacBook Pro (M5 Pro 24GB)** | Apple MPS (Metal) | `float16` | 24 GB Unified Memory | 🐢 ~3–7 min per pass | ✅ Full | ⚠️ CPU only (MPS unsupported in Docker) |
| **CPU Fallback** | CPU | `float32` | 32 GB RAM | 🐌 ~1–2 hours per pass | ✅ Fallback | ✅ Fallback |

> [!IMPORTANT]
> **Apple MPS & Docker Compatibility:**
> Apple Metal Performance Shaders (MPS) requires direct access to macOS kernel drivers. **Docker on macOS cannot access the Apple Silicon GPU** and runs in CPU-only mode.
> - **On MacBook Pro (M5 Pro):** Run **natively** via `uv` for GPU acceleration via MPS.
> - **On Ubuntu (RTX 4070):** Run either **natively** or via **Docker** (the Docker image sets up CUDA, cuDNN, PyTorch, and all dependencies automatically).

---

## 📋 Prerequisites: What to Install Before Running

### 1. Hugging Face Access (Required for FLUX.1-dev)
FLUX.1-dev is a gated checkpoint. Before generating images:
1. Create an account at [Hugging Face](https://huggingface.co).
2. Accept the model terms on the [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) model card.
3. Generate an access token at [Hugging Face Tokens](https://huggingface.co/settings/tokens).
4. Set your token in `.env`: `HF_TOKEN=hf_...`

---

### 2. For MacBook Pro (Apple Silicon M1–M5) — Native MPS Setup
Install **`uv`** (fast Python package manager) and Git:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Ensure Python 3.11 is available
uv python install 3.11
```
*(No manual PyTorch or CUDA installation needed — `uv sync` pulls the MPS-enabled PyTorch build automatically).*

---

### 3. For Ubuntu (RTX 4070 Rig) — Native CUDA Setup
Ensure NVIDIA drivers and CUDA are installed:
```bash
# 1. Verify your GPU is recognized
nvidia-smi

# 2. If drivers are missing, install recommended driver (e.g. 535 or 550)
sudo ubuntu-drivers install
sudo reboot

# 3. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

uv python install 3.11
```

---

### 4. For Ubuntu (RTX 4070 Rig) — Docker Setup (Zero Local Config)
The Docker image configures Ubuntu 22.04, Python 3.11, CUDA 12.4, PyTorch, diffusers, and all tools automatically. You only need:
1. **Docker Engine & Docker Compose v2**
2. **NVIDIA Container Toolkit** (enables GPU passthrough into Docker containers):
```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 Quick Start Guide

### Step 1: Clone and Configure
```bash
git clone https://github.com/your-username/inpainter.git
cd inpainter

# Create your environment file
cp .env.example .env
# Edit .env and enter your HF_TOKEN and optional cache directory
```

---

### Step 2: Run the Dry-Run Health Checker (Zero Model Download)
Before downloading any weights, test your hardware, dependencies, memory, and pipeline sanity:

```bash
# Can be run directly with standard Python (even with zero packages installed!)
python3 tools/dryrun.py
```
Or after installing dependencies:
```bash
uv run inpainter dryrun
```

**What the Dry-Run Checker Verifies:**
- Python version (3.11+)
- `uv` installation
- PyTorch import and backend support (CUDA / MPS / CPU)
- Detected VRAM & RAM limits (verifies 12GB offload suitability or unified memory headroom)
- Free disk space on Hugging Face model cache (warns if < 50 GB)
- Diffusers suite imports (`diffusers`, `transformers`, `accelerate`, `Pillow`, `numpy`)
- Pydantic v2 `JobConfig` schema validation
- Binary mask generation and morphology dilate/feather filters
- Pass scheduler lifecycle simulation
- Output directory writability

**Exit Codes:**
- `0 — HEALTHY`: All checks passed. Hardware acceleration is active and optimal.
- `1 — DEGRADED`: Warnings present (e.g. running on CPU fallback or tight VRAM using CPU offload).
- `2 — BROKEN`: Critical failure (e.g. missing dependencies or unwritable disk). Actionable fix instructions are printed directly in the terminal and saved to `dry_run_report.json`.

---

### Step 3: Run the Pipeline

#### Option A: Native on Mac (M5 Pro) or Ubuntu (RTX 4070)
```bash
# 1. Install dependencies with uv
uv sync

# 2. Check system info
uv run inpainter info

# 3. Start Backend API (FastAPI)
uv run uvicorn ui.app:app --host 0.0.0.0 --port 8000 --reload

# 4. In a separate terminal, launch the React Frontend
cd ui/frontend
npm install
npm run dev
# Open http://localhost:3000 in your browser
```

#### Option B: Docker on Ubuntu (RTX 4070 Rig)
Builds and launches both the CUDA API backend and the React frontend in containers with GPU acceleration:
```bash
docker compose --profile cuda up --build
```
- Frontend UI: `http://localhost:3000`
- API & Docs: `http://localhost:8000/docs`
- Health check inside container:
```bash
docker compose --profile cuda exec api-cuda python -m tools.dryrun
```

#### Option C: Docker CPU Fallback (Testing / Non-GPU)
```bash
docker compose --profile cpu up --build
```

---

## ⌨️ CLI Usage

You can also run generation directly from the terminal without launching the web server.

### Generate via YAML Job Config
```bash
uv run inpainter generate --config examples/job.yaml
```

### Generate via Direct CLI Flags
```bash
uv run inpainter generate \
  --scene "A cyberpunk alley at night, neon lights reflecting on wet ground, cinematic lighting" \
  --character "name=Alice,prompt=young woman in red leather jacket smiling,mask=120:180:320:540,lora=loras/alice.safetensors,lora_strength=0.8" \
  --character "name=Bob,prompt=middle-aged man in grey coat,mask=580:160:340:560" \
  --width 1024 --height 1024 \
  --steps 30 --cfg 3.5 \
  --device auto \
  --output ./output/cyberpunk_scene.png \
  --save-intermediates
```

### Common CLI Options
| Flag | Default | Description |
|---|---|---|
| `--config`, `-c` | `None` | Path to YAML job configuration file |
| `--scene`, `-s` | `None` | Global environment / background prompt (Pass 0) |
| `--character` | `None` | Repeatable character argument (`name=...,prompt=...,mask=x:y:w:h,lora=...`) |
| `--width` / `--height` | `1024` | Image dimensions (must be multiples of 8) |
| `--steps` | `30` | Sampling steps per inpainting pass |
| `--cfg` | `3.5` | Guidance scale |
| `--seed` | `42` | Base random seed (per-pass seeds are derived deterministically) |
| `--device` | `auto` | Execution device: `auto` (detects CUDA -> MPS -> CPU), `cuda`, `mps`, `cpu` |
| `--dtype` | `auto` | Precision: `auto`, `bfloat16`, `float16`, `float32` |
| `--dry-run` | `false` | Run health and validation checks without executing model inference |
| `--output`, `-o` | `./output/final.png` | Destination path for the final composite image |
| `--save-intermediates` | `true` | Save `pass_0.png` through `pass_N.png` to review intermediate stages |

---

## 🖥️ Management Web UI

The web interface is a modern React + Tailwind application mirroring all CLI capabilities:
- **Interactive Character Canvas & Masks**: Define bounding boxes and placement per character.
- **LoRA & ControlNet Swapping**: Assign individual LoRA checkpoints and strengths per pass.
- **Live Pass-by-Pass Stream**: Monitor real-time status and progress logs via Server-Sent Events (SSE).
- **Instant Previews**: Inspect intermediate pass renders and download the final high-resolution composite.

---

## 🔧 Memory & Performance Tuning Guide

### 1. RTX 4070 (12 GB VRAM)
- FLUX.1-dev weights require ~24 GB in `bfloat16`. On a 12 GB card, Inpainter automatically activates:
  - `pipe.enable_model_cpu_offload()` (offloads inactive pipeline components to system RAM during sampling).
  - `pipe.enable_vae_tiling()` (processes the VAE decode in tiles to prevent peak VRAM allocation spikes).
- **Requirement:** Ensure your host has at least **32 GB of system RAM**.

### 2. MacBook Pro (Apple Silicon MPS - 16GB / 24GB / 36GB Unified Memory)
- On Apple Silicon, unified memory is shared between CPU and GPU.
- Inpainter automatically selects `bfloat16` precision for MPS and applies VAE tiling with model offload.
- **Quantization for Low RAM Macs (16GB/24GB)**:
  - Set `quantization: "int8"` in your job config to apply native `torchao` int8 weight-only quantization (`uv add torchao`), reducing transformer memory by ~50% (from 12 GB down to ~6 GB).
  - Alternatively, specify a pre-quantized single-file checkpoint via `transformer_path: "path/to/flux1-dev-fp8.safetensors"` or an HF repo.
  - See `examples/mps_quantized.yaml` for a ready-to-run Mac configuration.

---

## 🔍 Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available() == False` on Ubuntu | Missing NVIDIA driver or PyTorch CPU build | Run `nvidia-smi` to verify driver. Reinstall CUDA PyTorch: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| CUDA Out of Memory (OOM) | Insufficient system RAM or disabled offloading | Ensure at least 32 GB system RAM is available. `enable_model_cpu_offload` is enabled by default for cards with <20 GB VRAM. |
| Slow Generation on Mac | Running under Docker (CPU) or Rosetta | Run **natively** on macOS using `uv sync` and verify MPS: `python3 -c "import torch; print(torch.backends.mps.is_available())"` |
| Model 401 / 403 Forbidden | Missing Hugging Face token for FLUX.1-dev | Accept the FLUX.1-dev license on Hugging Face and set `HF_TOKEN=hf_...` in `.env` |
| Low disk space warning | FLUX.1-dev checkpoint is ~24 GB | Set `HF_HOME=/path/to/large/drive/.cache/huggingface` in `.env` |

---

## 📚 Architecture Details

For the complete pipeline design, LoRA unloading lifecycle, and sequence diagrams, see [ARCHITECTURE.md](./ARCHITECTURE.md).
