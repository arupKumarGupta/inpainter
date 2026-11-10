"""
Inpainter CLI — command-line interface for the N-pass inpainting pipeline.

Usage:
    inpainter generate --config job.yaml
    inpainter generate --scene "..." --character "name=Alice,prompt=..."
    inpainter dryrun
    inpainter info
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, Optional

from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from orchestrator.device import resolve_device
from orchestrator.job import (
    CharacterConfig,
    ControlNetConfig,
    JobConfig,
    LoRAConfig,
    MaskConfig,
    ModelConfig,
    OutputConfig,
    ResolutionConfig,
    SamplerConfig,
    load_job_from_yaml,
)

app = typer.Typer(
    name="inpainter",
    help="N-pass multi-character inpainting pipeline using FLUX.1-dev",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _parse_character_arg(arg: str, index: int) -> CharacterConfig:
    """
    Parse a character string in format:
    "name=Alice,prompt=woman in red jacket,lora=path.safetensors,lora_strength=0.8,mask=100:200:300:500"
    """
    parts = arg.split(",")
    data: dict[str, str] = {}
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            data[k.strip().lower()] = v.strip()

    name = data.get("name", f"Character_{index}")
    prompt = data.get("prompt", "a character in the scene")
    char_id = data.get("id", f"char_{index}")

    # LoRA
    lora_cfg = None
    if "lora" in data:
        strength = float(data.get("lora_strength", 0.8))
        lora_cfg = LoRAConfig(path=Path(data["lora"]), strength=strength)

    # ControlNet
    cn_cfg = None
    if "controlnet_image" in data or "controlnet" in data:
        cn_img = data.get("controlnet_image", data.get("controlnet"))
        cn_type = data.get("controlnet_type", "openpose")
        cn_strength = float(data.get("controlnet_strength", 0.65))
        cn_cfg = ControlNetConfig(type=cn_type, image=Path(cn_img), strength=cn_strength)  # type: ignore

    # Mask
    # format: x:y:w:h
    mask_str = data.get("mask", "100:200:300:500")
    coords = [int(c) for c in mask_str.split(":")]
    if len(coords) == 4:
        mask_cfg = MaskConfig(type="bbox", x=coords[0], y=coords[1], width=coords[2], height=coords[3])
    else:
        mask_cfg = MaskConfig(type="bbox", x=100, y=200, width=300, height=500)

    return CharacterConfig(
        id=char_id,
        name=name,
        prompt=prompt,
        lora=lora_cfg,
        controlnet=cn_cfg,
        mask=mask_cfg,
    )


@app.command()
def generate(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to job YAML configuration"),
    ] = None,
    scene: Annotated[
        Optional[str],
        typer.Option("--scene", "-s", help="Global scene description prompt"),
    ] = None,
    characters: Annotated[
        Optional[list[str]],
        typer.Option("--character", help="Character definition string (can be repeated)"),
    ] = None,
    width: Annotated[int, typer.Option("--width", help="Canvas width (multiple of 8)")] = 1024,
    height: Annotated[int, typer.Option("--height", help="Canvas height (multiple of 8)")] = 1024,
    steps: Annotated[int, typer.Option("--steps", help="Inference steps per pass")] = 30,
    cfg: Annotated[float, typer.Option("--cfg", help="Guidance scale")] = 3.5,
    seed: Annotated[int, typer.Option("--seed", help="Base seed")] = 42,
    device: Annotated[str, typer.Option("--device", help="Device: auto | cuda | mps | cpu")] = "auto",
    dtype: Annotated[str, typer.Option("--dtype", help="Dtype: auto | float16 | bfloat16 | float32")] = "auto",
    output: Annotated[Path, typer.Option("--output", "-o", help="Final output path")] = Path("./output/final.png"),
    save_intermediates: Annotated[bool, typer.Option("--save-intermediates/--no-intermediates")] = True,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Verify pipeline health without running models")] = False,
) -> None:
    """Run an N-pass inpainting generation job."""
    if dry_run:
        console.print("[yellow]Running pipeline health dry run...[/yellow]")
        from tools.dryrun import main as run_dryrun
        code = run_dryrun()
        raise typer.Exit(code=code)

    # 1. Resolve configuration
    if config:
        job = load_job_from_yaml(config)
    else:
        if not scene:
            console.print("[red]Error: You must provide either --config or --scene.[/red]")
            raise typer.Exit(code=1)

        char_configs = []
        if characters:
            for idx, c_str in enumerate(characters, start=1):
                char_configs.append(_parse_character_arg(c_str, idx))
        else:
            # Default placeholder character
            char_configs.append(
                CharacterConfig(
                    id="char_1",
                    name="Character 1",
                    prompt="a person standing",
                    mask=MaskConfig(type="bbox", x=256, y=256, width=512, height=512),
                )
            )

        job = JobConfig(
            scene_prompt=scene,
            resolution=ResolutionConfig(width=width, height=height),
            model=ModelConfig(device=device, dtype=dtype),  # type: ignore
            sampler=SamplerConfig(steps=steps, cfg=cfg, seed=seed),
            characters=char_configs,
            output=OutputConfig(path=output, save_intermediates=save_intermediates),
        )

    # 2. Resolve hardware device
    try:
        dev_cfg = resolve_device(job.model.device, job.model.dtype)
    except Exception as e:
        console.print(f"[bold red]Device Resolution Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold green]Starting Inpainter Pipeline[/bold green]\n"
            f"Device: [cyan]{dev_cfg.summary}[/cyan]\n"
            f"Passes: [cyan]{job.total_passes}[/cyan] (Pass 0 Scene + {len(job.characters)} Characters)\n"
            f"Output: [cyan]{job.output.path}[/cyan]",
            title="Inpainter Job",
            border_style="green",
        )
    )

    from orchestrator.pipeline_runner import PipelineRunner
    from orchestrator.scheduler import Scheduler

    runner = PipelineRunner(dev_cfg)
    scheduler = Scheduler(runner, output_dir=job.output.path.parent)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing passes...", total=job.total_passes)

        def on_pass(pass_res):
            if pass_res.status.value == "done":
                desc = "Pass 0 (Scene)" if pass_res.pass_index == 0 else f"Pass {pass_res.pass_index} (Char {pass_res.character_id})"
                progress.console.print(f"[green]✓ Completed {desc}[/green]")
                progress.advance(task)
            elif pass_res.status.value == "failed":
                progress.console.print(f"[red]✗ Failed pass {pass_res.pass_index}: {pass_res.error}[/red]")

        try:
            scheduler.run(job, on_pass_complete=on_pass)
            console.print(f"\n[bold green]✓ Pipeline completed successfully![/bold green] Output: {job.output.path}")
        except Exception as e:
            console.print(f"\n[bold red]Execution failed:[/bold red] {e}")
            raise typer.Exit(code=1)


@app.command()
def dryrun() -> None:
    """Run system dependency and device checks without loading models."""
    from tools.dryrun import main as run_dryrun
    code = run_dryrun()
    raise typer.Exit(code=code)


@app.command()
def info() -> None:
    """Display system hardware, device, and PyTorch diagnostic info."""
    import platform
    import torch

    table = Table(title="Inpainter System Diagnostics", border_style="cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value", style="green")

    table.add_row("OS / Platform", f"{platform.system()} {platform.release()} ({platform.machine()})")
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))

    if torch.cuda.is_available():
        table.add_row("CUDA Device", torch.cuda.get_device_name(0))
        table.add_row("CUDA Version", torch.version.cuda or "N/A")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        table.add_row("VRAM", f"{vram:.2f} GB")

    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    table.add_row("Apple MPS Available", str(mps_avail))

    try:
        cfg = resolve_device("auto", "auto")
        table.add_row("Optimal Device", cfg.summary)
    except Exception as e:
        table.add_row("Optimal Device", f"[red]Error: {e}[/red]")

    console.print(table)


if __name__ == "__main__":
    app()
