import React from 'react';
import { useJobStore } from '../store/jobStore';
import { Terminal, CheckCircle2, AlertTriangle, Loader2 } from 'lucide-react';

export const ProgressPanel: React.FC = () => {
  const { activeStatus, currentPass, totalPasses, progressLogs, errorMessage } = useJobStore();

  if (activeStatus === 'idle' && progressLogs.length === 0) {
    return null;
  }

  const progressPercent = Math.min(
    100,
    Math.round((currentPass / Math.max(1, totalPasses)) * 100)
  );

  return (
    <div className="card-base space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sky-400 font-semibold text-sm">
          <Terminal className="w-4 h-4" />
          <span>Execution Progress</span>
        </div>
        <span className="text-xs font-mono text-gray-400">
          Pass {currentPass} / {totalPasses} ({progressPercent}%)
        </span>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-gray-950 rounded-full h-2 overflow-hidden border border-gray-800">
        <div
          className="bg-sky-500 h-full transition-all duration-300"
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      {/* Logs Console */}
      <div className="bg-gray-950 p-3 rounded-lg border border-gray-800 font-mono text-[11px] text-gray-300 h-36 overflow-y-auto space-y-1">
        {progressLogs.map((log, idx) => (
          <div key={idx} className="flex items-start gap-1.5">
            <span className="text-sky-500">&gt;</span>
            <span>{log}</span>
          </div>
        ))}
        {activeStatus === 'running' && (
          <div className="flex items-center gap-1.5 text-sky-400 animate-pulse pt-1">
            <Loader2 className="w-3 h-3 animate-spin" />
            <span>Processing pass...</span>
          </div>
        )}
      </div>

      {errorMessage && (
        <div className="flex items-start gap-2 p-3 bg-red-950/40 border border-red-800/80 rounded-lg text-xs text-red-300">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-red-400" />
          <div>
            <div className="font-semibold">Pipeline Error</div>
            <div className="font-mono text-[11px] text-red-200 mt-0.5">{errorMessage}</div>
          </div>
        </div>
      )}
    </div>
  );
};
