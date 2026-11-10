# Inpainter

**Inpainter** is a Python-native image generation system for producing consistent multi-character scenes using local AI models such as **Flux1-Dev**, LoRAs, and ControlNets — all driven directly through the **Hugging Face `diffusers`** library with no ComfyUI dependency.

It exposes:
- A **Management UI** — a web-based control panel to configure scenes, characters, models, and trigger generation jobs.
- A **CLI tool** — a command-line interface for the same pipeline, suitable for scripting and automation.

---

## Overview

Inpainter solves one of the hardest problems in AI image generation: producing a single scene with **multiple, visually consistent characters**. It uses an **N-pass inpainting architecture** — a staged pipeline where each pass replaces a masked placeholder with a fully rendered, LoRA-conditioned character, using the `diffusers` pipeline API directly in-process.

---

## Key Features

| Feature | Description |
|---|---|
| 🐍 Pure Python | Runs entirely via `diffusers` — no ComfyUI, no external server, no REST API |
| 🧑‍🤝‍🧑 Multi-character consistency | N-pass inpainting ensures each character is visually distinct and consistent |
| 🔁 N-pass architecture | 1 scene pass + N character passes; scales linearly with character count |
| 🧬 Local model support | Flux1-Dev, LoRA, ControlNet — all running locally, no cloud dependency |
| 🖥️ Management UI | FastAPI + web frontend to configure scenes, characters, models, and trigger jobs |
| ⌨️ CLI tool | Full CLI access to all generation parameters and pipeline control |
| 🎭 Per-character LoRA | Each character's LoRA is loaded and unloaded once per pass |
| ⚡ VRAM-efficient | Models loaded once; LoRAs swapped per pass via `load_lora_weights` / `unload_lora_weights` |

---

## How It Works — N-Pass Architecture

### Pass 0 — Scene Generation with N Masked Characters

The base scene is generated with **N placeholder masks** representing the spatial position of each character. These are baked into the prompt and/or the image so the model understands composition without committing to any character's appearance.

```
Scene: [ MASK_1 ] [ MASK_2 ] ... [ MASK_N ]
```

### Passes 1 → N — Sequential Character Inpainting

Each subsequent pass:
1. **Removes** one mask placeholder from the scene.
2. **Inpaints** that region with the next character using its prompt and optional **character LoRA**.
3. **Preserves** all previously inpainted characters and the background unchanged.

```
Pass 1: [ CHAR_1 ] [ MASK_2 ] ... [ MASK_N ]   ← LoRA_1 applied
Pass 2: [ CHAR_1 ] [ CHAR_2 ] ... [ MASK_N ]   ← LoRA_2 applied
...
Pass N: [ CHAR_1 ] [ CHAR_2 ] ... [ CHAR_N ]   ← LoRA_N applied
```

### Result

A single high-quality image with **N fully consistent, individually conditioned characters** in a coherent shared scene.

---

## Architecture

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full system and pipeline design.

---

## Tech Stack

- **Image Model**: Flux1-Dev (and compatible checkpoints)
- **Inference Engine**: [Hugging Face `diffusers`](https://github.com/huggingface/diffusers) — `FluxInpaintPipeline`, `FluxControlNetInpaintPipeline`
- **Conditioning**: LoRA (per-character, `load_lora_weights`), ControlNet (pose/depth/edge)
- **Mask Handling**: Pillow + NumPy; optional SAM integration for auto-segmentation
- **Backend**: Python + FastAPI
- **Management UI**: FastAPI + lightweight web frontend
- **CLI**: Python (`click` or `typer`)

---

## Getting Started

> Documentation coming soon. See `ARCHITECTURE.md` for the full pipeline design.

---

## License

TBD
