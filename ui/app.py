"""
Inpainter Management UI — FastAPI Backend

Endpoints:
  GET  /health                      — System health & device diagnostic
  POST /jobs                        — Submit a generation job
  GET  /jobs/{job_id}               — Get job status
  GET  /jobs/{job_id}/progress      — SSE stream of pass-by-pass progress
  GET  /jobs/{job_id}/result        — Get result image status
  GET  /images/{filename}           — Serve output images
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from orchestrator.device import resolve_device
from orchestrator.job import JobConfig
from tools.dryrun import run_all_checks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Inpainter API",
    description="Backend service for N-pass multi-character inpainting",
    version="0.1.0",
)

# Enable CORS for the frontend Vite server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output directory exists and mount static file server
output_dir = Path("./output").resolve()
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(output_dir)), name="images")


class JobStatus(str, Enum):
    PENDING = "pending"
    LOADING = "loading"
    RUNNING_PASS = "running_pass"
    DONE = "done"
    FAILED = "failed"


class ProgressEvent(BaseModel):
    pass_num: int
    total_passes: int
    status: str
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.PENDING
    config: JobConfig
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    current_pass: int = 0
    total_passes: int = 1
    error: str | None = None
    result_path: str | None = None
    progress_events: list[ProgressEvent] = Field(default_factory=list)


# In-memory store
_jobs: dict[str, JobRecord] = {}
_runner_instance: Any = None


def _get_or_create_runner(device_str: str = "auto", dtype_str: str = "auto"):
    global _runner_instance
    if _runner_instance is None:
        from orchestrator.pipeline_runner import PipelineRunner
        dev_cfg = resolve_device(device_str, dtype_str)
        _runner_instance = PipelineRunner(dev_cfg)
    return _runner_instance


def _execute_job_sync(job_id: str, config: JobConfig):
    record = _jobs[job_id]
    try:
        record.status = JobStatus.LOADING
        record.progress_events.append(
            ProgressEvent(
                pass_num=0,
                total_passes=config.total_passes,
                status="loading",
                message="Initializing model and hardware...",
            )
        )

        runner = _get_or_create_runner(config.model.device, config.model.dtype)
        if not runner.is_loaded:
            runner.load_model(config.model)

        from orchestrator.scheduler import Scheduler

        scheduler = Scheduler(runner, output_dir=output_dir)

        def on_pass_done(res):
            record.current_pass = res.pass_index
            record.updated_at = datetime.now(timezone.utc).isoformat()
            if res.status.value == "done":
                desc = "Pass 0 (Scene)" if res.pass_index == 0 else f"Pass {res.pass_index} (Character {res.character_id})"
                record.progress_events.append(
                    ProgressEvent(
                        pass_num=res.pass_index,
                        total_passes=config.total_passes,
                        status="pass_done",
                        message=f"Completed {desc}",
                    )
                )

        record.status = JobStatus.RUNNING_PASS
        final_img = scheduler.run(config, on_pass_complete=on_pass_done)

        out_name = f"result_{job_id}.png"
        final_file = output_dir / out_name
        final_img.save(final_file)

        record.status = JobStatus.DONE
        record.result_path = f"/images/{out_name}"
        record.progress_events.append(
            ProgressEvent(
                pass_num=config.total_passes,
                total_passes=config.total_passes,
                status="done",
                message="Generation finished successfully",
            )
        )
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        record.status = JobStatus.FAILED
        record.error = str(e)
        record.progress_events.append(
            ProgressEvent(
                pass_num=record.current_pass,
                total_passes=config.total_passes,
                status="failed",
                message=f"Error: {e}",
            )
        )


@app.get("/health")
def health():
    checks = run_all_checks()
    dev_info = "unknown"
    try:
        cfg = resolve_device("auto", "auto")
        dev_info = cfg.summary
    except Exception as e:
        dev_info = f"Detection failed: {e}"

    fails = [c for c in checks if c.status == "FAIL"]
    status_str = "healthy" if not fails else "degraded"

    return {
        "status": status_str,
        "device": dev_info,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": [
            {"name": c.name, "status": c.status, "message": c.message, "fix": c.fix}
            for c in checks
        ],
    }


@app.post("/jobs")
async def create_job(config: JobConfig, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    record = JobRecord(
        job_id=job_id,
        config=config,
        total_passes=config.total_passes,
    )
    _jobs[job_id] = record
    background_tasks.add_task(_execute_job_sync, job_id, config)

    return {
        "job_id": job_id,
        "status": record.status,
        "total_passes": record.total_passes,
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@app.get("/jobs/{job_id}/progress")
async def get_progress_sse(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        sent_count = 0
        while True:
            job = _jobs.get(job_id)
            if not job:
                break

            events = job.progress_events
            if len(events) > sent_count:
                for ev in events[sent_count:]:
                    payload = ev.model_dump_json()
                    yield f"data: {payload}\n\n"
                sent_count = len(events)

            if job.status in (JobStatus.DONE, JobStatus.FAILED):
                final_payload = json.dumps({"status": job.status, "result_path": job.result_path, "error": job.error})
                yield f"data: {final_payload}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[job_id]
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=400, detail=f"Job not done (status: {job.status})")
    return {"job_id": job_id, "result_path": job.result_path}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ui.app:app", host="0.0.0.0", port=8000, reload=True)
