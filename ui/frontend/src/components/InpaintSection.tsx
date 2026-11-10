import React from 'react';
import { useJobStore } from '../store/jobStore';
import { Brush } from 'lucide-react';

export const InpaintSection: React.FC = () => {
  const { config, setInpaint } = useJobStore();

  return (
    <div className="card-base space-y-3">
      <div className="flex items-center gap-2 text-sky-400 font-semibold text-sm">
        <Brush className="w-4 h-4" />
        <span>Inpaint Blending Options</span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="label-base">Denoising Strength ({config.inpaint.strength})</label>
          <input
            type="range"
            min={0.5}
            max={1.0}
            step={0.05}
            className="w-full accent-sky-500"
            value={config.inpaint.strength}
            onChange={(e) => setInpaint({ strength: parseFloat(e.target.value) || 0.85 })}
          />
        </div>
        <div>
          <label className="label-base">Mask Feather Blur ({config.inpaint.mask_blur}px)</label>
          <input
            type="range"
            min={0}
            max={32}
            step={2}
            className="w-full accent-sky-500"
            value={config.inpaint.mask_blur}
            onChange={(e) => setInpaint({ mask_blur: parseInt(e.target.value) || 8 })}
          />
        </div>
      </div>
    </div>
  );
};
