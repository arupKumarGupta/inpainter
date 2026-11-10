import React from 'react';
import { useJobStore } from '../store/jobStore';
import { Sparkles } from 'lucide-react';

export const SceneSection: React.FC = () => {
  const { config, setScenePrompt, setNegativePrompt } = useJobStore();

  return (
    <div className="card-base space-y-3">
      <div className="flex items-center gap-2 text-sky-400 font-semibold text-sm">
        <Sparkles className="w-4 h-4" />
        <span>Scene & Background</span>
      </div>

      <div>
        <label className="label-base">Global Scene Prompt (Pass 0)</label>
        <textarea
          rows={3}
          className="input-base"
          placeholder="Describe the environment, lighting, atmosphere..."
          value={config.scene_prompt}
          onChange={(e) => setScenePrompt(e.target.value)}
        />
      </div>

      <div>
        <label className="label-base">Negative Prompt</label>
        <input
          type="text"
          className="input-base"
          placeholder="blurry, low quality, artifacts..."
          value={config.negative_prompt}
          onChange={(e) => setNegativePrompt(e.target.value)}
        />
      </div>
    </div>
  );
};
