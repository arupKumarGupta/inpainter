import React, { useEffect, useState } from 'react';
import { fetchHealth } from './api/client';
import type { HealthResponse } from './types';
import { SceneSection } from './components/SceneSection';
import { ResolutionModelSection } from './components/ResolutionModelSection';
import { SamplerSection } from './components/SamplerSection';
import { CharacterListSection } from './components/CharacterListSection';
import { InpaintSection } from './components/InpaintSection';
import { OutputSection } from './components/OutputSection';
import { GeneratePanel } from './components/GeneratePanel';
import { ProgressPanel } from './components/ProgressPanel';
import { ResultPanel } from './components/ResultPanel';
import { Layers, Activity, Cpu } from 'lucide-react';

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((err) => console.warn('Backend offline or health check failed:', err));
  }, []);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="border-b border-gray-800 bg-gray-900/40 backdrop-blur-md px-6 py-3.5 flex items-center justify-between sticky top-0 z-20">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-sky-500/10 border border-sky-500/30 rounded-xl text-sky-400">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
              <span>Inpainter</span>
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-sky-950 text-sky-300 border border-sky-700/50">
                N-Pass Studio
              </span>
            </h1>
            <p className="text-xs text-gray-400">FLUX.1-dev Multi-Character Consistent Inpainting</p>
          </div>
        </div>

        {/* System & Hardware Health Badge */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-900 border border-gray-800 text-xs">
            <Cpu className="w-3.5 h-3.5 text-sky-400" />
            <span className="text-gray-300 font-mono text-[11px]">
              {health?.device || 'Resolving hardware...'}
            </span>
            <span
              className={`w-2 h-2 rounded-full ${
                health?.status === 'healthy'
                  ? 'bg-emerald-400 animate-pulse'
                  : 'bg-amber-400'
              }`}
            />
          </div>
        </div>
      </header>

      {/* Main Studio View */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Configuration Controls (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <SceneSection />
          <CharacterListSection />
          <ResolutionModelSection />
          <SamplerSection />
          <InpaintSection />
          <OutputSection />
        </div>

        {/* Right Column: Execution, Progress & Output (5 cols) */}
        <div className="lg:col-span-5 space-y-4 sticky top-20 self-start">
          <GeneratePanel />
          <ProgressPanel />
          <ResultPanel />
        </div>
      </main>
    </div>
  );
};

export default App;
