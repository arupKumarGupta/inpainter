import React from 'react';
import { useJobStore } from '../store/jobStore';
import { Dices, Gauge } from 'lucide-react';

export const SamplerSection: React.FC = () => {
  const { config, setSampler } = useJobStore();

  const randomizeSeed = () => {
    const randomSeed = Math.floor(Math.random() * 1000000);
    setSampler({ seed: randomSeed });
  };

  return (
    <div className="card-base space-y-4">
      <div className="flex items-center gap-2 text-sky-400 font-semibold text-sm">
        <Gauge className="w-4 h-4" />
        <span>Sampling Parameters</span>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div>
          <label className="label-base">Steps ({config.sampler.steps})</label>
          <input
            type="range"
            min={1}
            max={60}
            className="w-full accent-sky-500"
            value={config.sampler.steps}
            onChange={(e) => setSampler({ steps: parseInt(e.target.value) || 30 })}
          />
        </div>

        <div>
          <label className="label-base">Guidance CFG ({config.sampler.cfg})</label>
          <input
            type="range"
            min={1.0}
            max={10.0}
            step={0.1}
            className="w-full accent-sky-500"
            value={config.sampler.cfg}
            onChange={(e) => setSampler({ cfg: parseFloat(e.target.value) || 3.5 })}
          />
        </div>

        <div>
          <div className="flex items-center justify-between">
            <label className="label-base">Seed</label>
            <button
              type="button"
              onClick={randomizeSeed}
              className="text-gray-400 hover:text-sky-400 transition-colors p-0.5"
              title="Randomize Seed"
            >
              <Dices className="w-3.5 h-3.5" />
            </button>
          </div>
          <input
            type="number"
            className="input-base text-xs py-1.5"
            value={config.sampler.seed}
            onChange={(e) => setSampler({ seed: parseInt(e.target.value) || 0 })}
          />
        </div>
      </div>
    </div>
  );
};
