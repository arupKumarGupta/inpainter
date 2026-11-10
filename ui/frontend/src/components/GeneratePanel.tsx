import React, { useState } from 'react';
import { useJobStore } from '../store/jobStore';
import { submitJob, subscribeJobProgress } from '../api/client';
import { Play, Loader2, AlertCircle } from 'lucide-react';

export const GeneratePanel: React.FC = () => {
  const {
    config,
    activeStatus,
    setJobStarted,
    addProgressEvent,
    setJobCompleted,
    setJobFailed,
  } = useJobStore();

  const [isLoading, setIsLoading] = useState(false);

  const handleGenerate = async () => {
    setIsLoading(true);
    try {
      const res = await submitJob(config);
      setJobStarted(res.job_id, res.total_passes);

      // Subscribe to Server-Sent Events
      subscribeJobProgress(
        res.job_id,
        (data) => {
          if (data.message) {
            addProgressEvent(data.message, data.pass_num);
          }
          if (data.status === 'done' && data.result_path) {
            setJobCompleted(data.result_path);
            setIsLoading(false);
          }
          if (data.status === 'failed') {
            setJobFailed(data.error || 'Pipeline execution failed.');
            setIsLoading(false);
          }
        },
        () => {
          setIsLoading(false);
        }
      );
    } catch (err: any) {
      setJobFailed(err.message || 'Failed to submit job.');
      setIsLoading(false);
    }
  };

  const isRunning = activeStatus === 'running' || isLoading;

  return (
    <div className="card-base space-y-3">
      <button
        type="button"
        disabled={isRunning || !config.scene_prompt}
        onClick={handleGenerate}
        className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-bold text-sm bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-400 hover:to-blue-500 text-white shadow-lg shadow-sky-950/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
      >
        {isRunning ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Generating Scene...</span>
          </>
        ) : (
          <>
            <Play className="w-4 h-4 fill-current" />
            <span>Start N-Pass Generation ({1 + config.characters.length} Passes)</span>
          </>
        )}
      </button>

      {!config.scene_prompt && (
        <div className="flex items-center gap-1.5 text-xs text-amber-400">
          <AlertCircle className="w-3.5 h-3.5" />
          <span>Scene prompt cannot be empty.</span>
        </div>
      )}
    </div>
  );
};
