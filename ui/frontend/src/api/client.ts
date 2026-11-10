import type { HealthResponse, JobConfig, JobStatusResponse } from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`Failed to fetch health: ${res.statusText}`);
  }
  return res.json();
}

export async function submitJob(config: JobConfig): Promise<{ job_id: string; total_passes: number }> {
  const res = await fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Job submission failed: ${err}`);
  }
  return res.json();
}

export async function fetchJob(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch job ${jobId}`);
  }
  return res.json();
}

export function subscribeJobProgress(
  jobId: string,
  onEvent: (event: any) => void,
  onFinish?: () => void,
): () => void {
  const eventSource = new EventSource(`${API_BASE}/jobs/${jobId}/progress`);

  eventSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      onEvent(data);
      if (data.status === 'done' || data.status === 'failed') {
        eventSource.close();
        if (onFinish) onFinish();
      }
    } catch (err) {
      console.error('Error parsing SSE event:', err);
    }
  };

  eventSource.onerror = () => {
    eventSource.close();
    if (onFinish) onFinish();
  };

  return () => {
    eventSource.close();
  };
}
