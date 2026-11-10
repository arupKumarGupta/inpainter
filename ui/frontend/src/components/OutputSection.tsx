import React from 'react';
import { useJobStore } from '../store/jobStore';
import { FolderCheck } from 'lucide-react';

export const OutputSection: React.FC = () => {
  const { config, setOutput } = useJobStore();

  return (
    <div className="card-base space-y-3">
      <div className="flex items-center gap-2 text-sky-400 font-semibold text-sm">
        <FolderCheck className="w-4 h-4" />
        <span>Output Configuration</span>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex-1">
          <label className="label-base">Destination Path</label>
          <input
            type="text"
            className="input-base text-xs font-mono"
            value={config.output.path}
            onChange={(e) => setOutput({ path: e.target.value })}
          />
        </div>
        <div className="flex items-center gap-2 pt-4">
          <input
            type="checkbox"
            id="intermediates"
            className="rounded accent-sky-500 w-4 h-4 bg-gray-900 border-gray-700"
            checked={config.output.save_intermediates}
            onChange={(e) => setOutput({ save_intermediates: e.target.checked })}
          />
          <label htmlFor="intermediates" className="text-xs text-gray-300 select-none cursor-pointer">
            Save Pass Intermediates
          </label>
        </div>
      </div>
    </div>
  );
};
