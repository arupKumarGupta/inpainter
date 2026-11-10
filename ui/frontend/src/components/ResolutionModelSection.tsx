import React from 'react';
import { useJobStore } from '../store/jobStore';
import { Cpu, Sliders } from 'lucide-react';
import type { DeviceKind, DtypeKind } from '../types';

export const ResolutionModelSection: React.FC = () => {
  const { config, setResolution, setModel } = useJobStore();

  return (
    <div className="card-base space-y-4">
      <div className="flex items-center gap-2 text-sky-400 font-semibold text-sm">
        <Sliders className="w-4 h-4" />
        <span>Model & Hardware Resolution</span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="label-base">Width (px)</label>
          <input
            type="number"
            step={8}
            className="input-base"
            value={config.resolution.width}
            onChange={(e) => setResolution({ width: parseInt(e.target.value) || 1024 })}
          />
        </div>
        <div>
          <label className="label-base">Height (px)</label>
          <input
            type="number"
            step={8}
            className="input-base"
            value={config.resolution.height}
            onChange={(e) => setResolution({ height: parseInt(e.target.value) || 1024 })}
          />
        </div>
      </div>

      <div>
        <label className="label-base">Base Model Checkpoint</label>
        <input
          type="text"
          className="input-base font-mono text-xs"
          value={config.model.repo_id}
          onChange={(e) => setModel({ repo_id: e.target.value })}
        />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="label-base">Compute Device</label>
          <select
            className="input-base text-xs"
            value={config.model.device}
            onChange={(e) => setModel({ device: e.target.value as DeviceKind })}
          >
            <option value="auto">Auto (Detect)</option>
            <option value="cuda">CUDA (NVIDIA GPU)</option>
            <option value="mps">MPS (Apple Silicon)</option>
            <option value="cpu">CPU (Slow Fallback)</option>
          </select>
        </div>
        <div>
          <label className="label-base">Data Precision</label>
          <select
            className="input-base text-xs"
            value={config.model.dtype}
            onChange={(e) => setModel({ dtype: e.target.value as DtypeKind })}
          >
            <option value="auto">Auto (Hardware Default)</option>
            <option value="bfloat16">bfloat16 (RTX 30xx/40xx)</option>
            <option value="float16">float16 (MPS / Older GPUs)</option>
            <option value="float32">float32 (CPU)</option>
          </select>
        </div>
      </div>
    </div>
  );
};
