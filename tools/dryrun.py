#!/usr/bin/env python3
"""
Inpainter Dry-Run Health Checker
=================================
Checks all system dependencies, device availability, and pipeline structural
integrity WITHOUT loading any AI models or making network requests.

Run:
    python -m tools.dryrun
    inpainter-dryrun
    inpainter dryrun

Exit codes:
    0 — HEALTHY   — all checks passed
    1 — DEGRADED  — warnings only (runs with limitations / CPU fallback)
    2 — BROKEN    — critical failure (will not run)
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Graceful rich fallback
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None  # type: ignore

StatusKind = Literal["PASS", "WARN", "FAIL"]


@dataclass
class CheckResult:
    name: str
    status: StatusKind
    message: str
    fix: str | None = None
    detail: str | None = None


def check_python_version() -> CheckResult:
    v = sys.version_info
    v_str = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 11):
        return CheckResult("python_version", "PASS", f"Python {v_str} meets requirement (>=3.11)")
    if v >= (3, 10):
        return CheckResult(
            "python_version",
            "WARN",
            f"Python {v_str} is usable but 3.11+ is recommended.",
            fix="Install Python 3.11+:\n  uv python install 3.11\n  or brew install python@3.11 (Mac)",
        )
    return CheckResult(
        "python_version",
        "FAIL",
        f"Python {v_str} is unsupported. Requires Python 3.11+.",
        fix="Install Python 3.11+:\n  uv python install 3.11\n  or sudo apt install python3.11 (Ubuntu)",
    )


def check_uv_installed() -> CheckResult:
    uv_path = shutil.which("uv")
    if uv_path:
        return CheckResult("uv_tool", "PASS", f"Found uv package manager at {uv_path}")
    return CheckResult(
        "uv_tool",
        "WARN",
        "uv package manager not found in PATH.",
        fix="Install uv (recommended):\n  curl -LsSf https://astral.sh/uv/install.sh | sh\n  source ~/.local/bin/env",
    )


def check_torch_import() -> CheckResult:
    try:
        import torch

        v = torch.__version__
        return CheckResult("torch_import", "PASS", f"PyTorch {v} imported successfully")
    except ImportError as e:
        return CheckResult(
            "torch_import",
            "FAIL",
            f"PyTorch is not installed ({e})",
            fix=(
                "Install PyTorch via uv:\n"
                "  uv sync\n"
                "Or manually:\n"
                "  Mac (MPS) / CPU: uv pip install torch torchvision\n"
                "  Ubuntu (CUDA 12.4): uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
            ),
        )


def check_device_support() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult("device_support", "FAIL", "Cannot check devices because torch is missing.")

    # 1. CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_ver = torch.version.cuda
        if vram_gb >= 20.0:
            return CheckResult(
                "device_support",
                "PASS",
                f"CUDA GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM, CUDA {cuda_ver}) - Full resident VRAM mode",
            )
        elif vram_gb >= 11.5:
            return CheckResult(
                "device_support",
                "PASS",
                f"CUDA GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM, CUDA {cuda_ver}) - CPU offload enabled",
                detail="12GB VRAM requires cpu_offload and >=32GB system RAM for FLUX.1-dev.",
            )
        else:
            return CheckResult(
                "device_support",
                "WARN",
                f"CUDA GPU detected: {gpu_name} with {vram_gb:.1f} GB VRAM (<12GB).",
                fix=(
                    "FLUX.1-dev in bfloat16 is ~24GB. With <12GB VRAM, sequential offloading or quantized "
                    "checkpoints are required. Ensure at least 32GB system RAM."
                ),
            )

    # 2. Apple MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return CheckResult(
            "device_support",
            "PASS",
            "Apple Metal Performance Shaders (MPS) is available (Apple Silicon).",
            detail="Unified memory will be used with float16 and VAE tiling.",
        )

    # 3. CPU fallback
    is_mac = platform.system() == "Darwin"
    if is_mac:
        return CheckResult(
            "device_support",
            "WARN",
            "Running on macOS but MPS is not available. Falling back to CPU.",
            fix=(
                "Ensure you are running on an Apple Silicon Mac (M1-M5) with macOS 12.3+ "
                "and an arm64 build of Python/PyTorch (not Rosetta x86_64 emulation)."
            ),
        )

    return CheckResult(
        "device_support",
        "WARN",
        "No CUDA GPU detected. System will fall back to CPU inference (very slow for FLUX.1-dev).",
        fix=(
            "For Ubuntu / Linux rigs with NVIDIA GPUs:\n"
            "  1. Verify drivers: nvidia-smi\n"
            "  2. Install drivers: sudo ubuntu-drivers install\n"
            "  3. Install PyTorch with CUDA: uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        ),
    )


def check_system_ram() -> CheckResult:
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb >= 30.0:
            return CheckResult("system_ram", "PASS", f"System RAM: {ram_gb:.1f} GB (optimal for offloading)")
        elif ram_gb >= 15.0:
            return CheckResult(
                "system_ram",
                "WARN",
                f"System RAM: {ram_gb:.1f} GB. Working with 12GB GPUs and offload may be tight.",
                fix="32 GB+ system RAM is recommended when using enable_model_cpu_offload with FLUX.1-dev.",
            )
        else:
            return CheckResult(
                "system_ram",
                "FAIL",
                f"System RAM: {ram_gb:.1f} GB is too low for FLUX models.",
                fix="Ensure machine has at least 16 GB RAM (32 GB recommended).",
            )
    except ImportError:
        return CheckResult("system_ram", "PASS", "psutil not installed (skipped RAM check)")


def check_hf_cache_dir() -> CheckResult:
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    p = Path(hf_home).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(p)
        free_gb = usage.free / (1024**3)
        if free_gb >= 45.0:
            return CheckResult(
                "hf_cache_disk", "PASS", f"HuggingFace cache ({p}) has {free_gb:.1f} GB free disk space"
            )
        return CheckResult(
            "hf_cache_disk",
            "WARN",
            f"HuggingFace cache ({p}) only has {free_gb:.1f} GB free (<50 GB).",
            fix=(
                f"FLUX.1-dev checkpoint requires ~24 GB alone.\n"
                f"Point HF_HOME to a partition with 50+ GB free:\n"
                f"  export HF_HOME=/path/to/large/disk/huggingface\n"
                f"  (or set HF_HOME in your .env file)"
            ),
        )
    except Exception as e:
        return CheckResult(
            "hf_cache_disk",
            "WARN",
            f"Cannot inspect cache directory {p}: {e}",
            fix=f"Create directory: mkdir -p {p} && chmod 755 {p}",
        )


def check_diffusers_suite() -> CheckResult:
    missing = []
    versions = {}
    for pkg in ["diffusers", "transformers", "accelerate", "PIL", "numpy"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "ok")
        except ImportError:
            missing.append(pkg)

    if not missing:
        details = ", ".join(f"{k} {v}" for k, v in versions.items())
        return CheckResult("diffusers_suite", "PASS", f"All core ML libraries available ({details})")
    return CheckResult(
        "diffusers_suite",
        "FAIL",
        f"Missing critical packages: {', '.join(missing)}",
        fix="Install all project dependencies:\n  uv sync",
    )


def check_pydantic_schema() -> CheckResult:
    try:
        from orchestrator.job import CharacterConfig, JobConfig, MaskConfig

        job = JobConfig(
            scene_prompt="A test scene prompt for schema validation",
            characters=[
                CharacterConfig(
                    id="char_1",
                    name="Alice",
                    prompt="woman smiling",
                    mask=MaskConfig(type="bbox", x=100, y=100, width=200, height=200),
                )
            ],
        )
        assert job.total_passes == 2
        return CheckResult("pydantic_schema", "PASS", "JobConfig Pydantic v2 schema validated successfully")
    except ImportError as e:
        return CheckResult(
            "pydantic_schema",
            "FAIL",
            f"Schema dependencies missing: {e}",
            fix="Install project dependencies:\n  uv sync",
        )
    except Exception as e:
        return CheckResult(
            "pydantic_schema",
            "FAIL",
            f"JobConfig schema validation failed: {e}",
            fix="Check orchestrator/job.py for syntax or field validation issues.",
        )


def check_mask_generation() -> CheckResult:
    try:
        from orchestrator.mask_manager import make_bbox_mask

        img = make_bbox_mask(512, 512, 50, 50, 100, 100)
        if img.size == (512, 512) and img.mode == "L":
            return CheckResult("mask_generation", "PASS", "Mask generation pipeline (Pillow/NumPy) works properly")
        return CheckResult("mask_generation", "FAIL", f"Mask output invalid: size={img.size}, mode={img.mode}")
    except Exception as e:
        return CheckResult(
            "mask_generation",
            "FAIL",
            f"Mask generation error: {e}",
            fix="Check orchestrator/mask_manager.py and Pillow installation:\n  uv sync",
        )


def check_scheduler_structure() -> CheckResult:
    try:
        from orchestrator.scheduler import Scheduler

        class MockRunner:
            def run_scene_pass(self, j):
                pass

            def run_character_pass(self, j, c, p, m, idx):
                pass

        sched = Scheduler(MockRunner(), output_dir="./output")
        return CheckResult("scheduler_structure", "PASS", "Pass scheduler logic structure is intact")
    except Exception as e:
        return CheckResult(
            "scheduler_structure",
            "FAIL",
            f"Scheduler instantiation error: {e}",
            fix="Check orchestrator/scheduler.py.",
        )


def check_output_dir_writable() -> CheckResult:
    out_dir = Path("./output")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        test_file = out_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        return CheckResult("output_dir", "PASS", f"Output directory {out_dir.resolve()} is writable")
    except Exception as e:
        return CheckResult(
            "output_dir",
            "FAIL",
            f"Cannot write to output directory: {e}",
            fix="Ensure write permissions: mkdir -p ./output && chmod 777 ./output",
        )


def run_all_checks() -> list[CheckResult]:
    return [
        check_python_version(),
        check_uv_installed(),
        check_torch_import(),
        check_device_support(),
        check_system_ram(),
        check_hf_cache_dir(),
        check_diffusers_suite(),
        check_pydantic_schema(),
        check_mask_generation(),
        check_scheduler_structure(),
        check_output_dir_writable(),
    ]


def main() -> int:
    results = run_all_checks()

    pass_count = sum(1 for r in results if r.status == "PASS")
    warn_count = sum(1 for r in results if r.status == "WARN")
    fail_count = sum(1 for r in results if r.status == "FAIL")

    if HAS_RICH:
        console.print(
            Panel(
                "[bold cyan]Inpainter Pipeline Health Checker (Dry Run)[/bold cyan]\n"
                "Verifies dependencies, hardware acceleration (CUDA/MPS), and pipeline logic without downloading models.",
                border_style="cyan",
            )
        )

        table = Table(title="Health Check Summary", border_style="blue", show_header=True)
        table.add_column("Check", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        for r in results:
            if r.status == "PASS":
                status_text = "[bold green]PASS[/bold green]"
            elif r.status == "WARN":
                status_text = "[bold yellow]WARN[/bold yellow]"
            else:
                status_text = "[bold red]FAIL[/bold red]"

            detail_text = r.message
            if r.detail:
                detail_text += f"\n[dim]{r.detail}[/dim]"
            table.add_row(r.name, status_text, detail_text)

        console.print(table)

        if fail_count > 0:
            console.print("\n[bold red]CRITICAL FAILURES & FIXES:[/bold red]")
            for r in results:
                if r.status == "FAIL" and r.fix:
                    console.print(
                        Panel(
                            f"[bold]{r.name}[/bold]: {r.message}\n\n[bold yellow]Remediation Steps:[/bold yellow]\n{r.fix}",
                            title=f"Fix for {r.name}",
                            border_style="red",
                        )
                    )

        if warn_count > 0:
            console.print("\n[bold yellow]WARNINGS / OPTIMIZATION RECOMMENDATIONS:[/bold yellow]")
            for r in results:
                if r.status == "WARN" and r.fix:
                    console.print(
                        Panel(
                            f"[bold]{r.name}[/bold]: {r.message}\n\n[bold cyan]Recommendation:[/bold cyan]\n{r.fix}",
                            title=f"Warning: {r.name}",
                            border_style="yellow",
                        )
                    )
    else:
        # Plain text rendering
        print("=" * 70)
        print(" INPAINTER PIPELINE HEALTH CHECKER (DRY RUN)")
        print("=" * 70)
        for r in results:
            print(f"[{r.status:4}] {r.name:<24} {r.message}")
            if r.detail:
                print(f"       Detail: {r.detail}")

        if fail_count > 0:
            print("\n" + "!" * 70)
            print(" CRITICAL FAILURES & ACTIONABLE FIXES")
            print("!" * 70)
            for r in results:
                if r.status == "FAIL" and r.fix:
                    print(f"\n[FAIL] {r.name}: {r.message}")
                    print("--- HOW TO FIX ---")
                    for line in r.fix.splitlines():
                        print(f"  {line}")

        if warn_count > 0:
            print("\n" + "-" * 70)
            print(" WARNINGS & RECOMMENDATIONS")
            print("-" * 70)
            for r in results:
                if r.status == "WARN" and r.fix:
                    print(f"\n[WARN] {r.name}: {r.message}")
                    for line in r.fix.splitlines():
                        print(f"  {line}")

    # Save JSON report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": sys.version.split()[0],
        "overall_status": (
            "HEALTHY" if fail_count == 0 and warn_count == 0 else ("DEGRADED" if fail_count == 0 else "BROKEN")
        ),
        "counts": {"pass": pass_count, "warn": warn_count, "fail": fail_count},
        "results": [asdict(r) for r in results],
    }

    report_path = Path("dry_run_report.json")
    try:
        report_path.write_text(json.dumps(report, indent=2))
        msg = f"\nSaved dry run report to {report_path.resolve()}"
        if HAS_RICH:
            console.print(msg)
        else:
            print(msg)
    except Exception as e:
        pass

    if fail_count > 0:
        summary = f"\nOverall Status: BROKEN ({fail_count} failed, {warn_count} warnings)"
        if HAS_RICH:
            console.print(f"[bold red]{summary}[/bold red]")
        else:
            print(summary)
        return 2
    elif warn_count > 0:
        summary = f"\nOverall Status: DEGRADED ({pass_count} passed, {warn_count} warnings)"
        if HAS_RICH:
            console.print(f"[bold yellow]{summary}[/bold yellow]")
        else:
            print(summary)
        return 1
    else:
        summary = f"\nOverall Status: HEALTHY ({pass_count}/{len(results)} passed)"
        if HAS_RICH:
            console.print(f"[bold green]{summary}[/bold green]")
        else:
            print(summary)
        return 0


if __name__ == "__main__":
    sys.exit(main())
