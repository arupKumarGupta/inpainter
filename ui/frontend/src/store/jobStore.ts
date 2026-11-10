import { create } from 'zustand';
import type { CharacterConfig, InpaintConfig, JobConfig, ModelConfig, OutputConfig, ResolutionConfig, SamplerConfig } from '../types';

interface JobStore {
  config: JobConfig;
  jobId: string | null;
  activeStatus: string;
  currentPass: number;
  totalPasses: number;
  progressLogs: string[];
  resultImage: string | null;
  errorMessage: string | null;

  setScenePrompt: (prompt: string) => void;
  setNegativePrompt: (neg: string) => void;
  setResolution: (res: Partial<ResolutionConfig>) => void;
  setModel: (model: Partial<ModelConfig>) => void;
  setSampler: (sampler: Partial<SamplerConfig>) => void;
  setInpaint: (inpaint: Partial<InpaintConfig>) => void;
  setOutput: (output: Partial<OutputConfig>) => void;

  addCharacter: () => void;
  removeCharacter: (id: string) => void;
  updateCharacter: (id: string, update: Partial<CharacterConfig>) => void;

  setJobStarted: (jobId: string, totalPasses: number) => void;
  addProgressEvent: (msg: string, passNum?: number) => void;
  setJobCompleted: (resultUrl: string) => void;
  setJobFailed: (error: string) => void;
  resetJobState: () => void;
}

const defaultCharacters: CharacterConfig[] = [
  {
    id: 'char_1',
    name: 'Character 1',
    prompt: 'young woman with red jacket, smiling',
    mask: {
      type: 'bbox',
      x: 150,
      y: 200,
      width: 300,
      height: 550,
    },
  },
];

export const useJobStore = create<JobStore>((set) => ({
  config: {
    scene_prompt: 'A cyberpunk alley at night, neon lights reflecting on wet pavement, cinematic lighting',
    negative_prompt: 'blurry, distorted, low quality, artifacts',
    resolution: { width: 1024, height: 1024 },
    model: { repo_id: 'black-forest-labs/FLUX.1-dev', dtype: 'auto', device: 'auto' },
    sampler: { steps: 30, cfg: 3.5, seed: 42 },
    characters: defaultCharacters,
    inpaint: { strength: 0.85, mask_padding: 24, mask_blur: 8, mask_grow: 12 },
    output: { path: './output/final.png', save_intermediates: true },
  },
  jobId: null,
  activeStatus: 'idle',
  currentPass: 0,
  totalPasses: 1,
  progressLogs: [],
  resultImage: null,
  errorMessage: null,

  setScenePrompt: (scene_prompt) =>
    set((s) => ({ config: { ...s.config, scene_prompt } })),

  setNegativePrompt: (negative_prompt) =>
    set((s) => ({ config: { ...s.config, negative_prompt } })),

  setResolution: (res) =>
    set((s) => ({ config: { ...s.config, resolution: { ...s.config.resolution, ...res } } })),

  setModel: (model) =>
    set((s) => ({ config: { ...s.config, model: { ...s.config.model, ...model } } })),

  setSampler: (sampler) =>
    set((s) => ({ config: { ...s.config, sampler: { ...s.config.sampler, ...sampler } } })),

  setInpaint: (inpaint) =>
    set((s) => ({ config: { ...s.config, inpaint: { ...s.config.inpaint, ...inpaint } } })),

  setOutput: (output) =>
    set((s) => ({ config: { ...s.config, output: { ...s.config.output, ...output } } })),

  addCharacter: () =>
    set((s) => {
      const idx = s.config.characters.length + 1;
      const newChar: CharacterConfig = {
        id: `char_${idx}`,
        name: `Character ${idx}`,
        prompt: 'a character standing in the scene',
        mask: {
          type: 'bbox',
          x: 100 + (idx * 50) % 500,
          y: 200,
          width: 280,
          height: 520,
        },
      };
      return {
        config: { ...s.config, characters: [...s.config.characters, newChar] },
      };
    }),

  removeCharacter: (id) =>
    set((s) => {
      if (s.config.characters.length <= 1) return s;
      return {
        config: {
          ...s.config,
          characters: s.config.characters.filter((c) => c.id !== id),
        },
      };
    }),

  updateCharacter: (id, update) =>
    set((s) => ({
      config: {
        ...s.config,
        characters: s.config.characters.map((c) =>
          c.id === id ? { ...c, ...update } : c
        ),
      },
    })),

  setJobStarted: (jobId, totalPasses) =>
    set({
      jobId,
      activeStatus: 'running',
      currentPass: 0,
      totalPasses,
      progressLogs: [`Job ${jobId} initiated...`],
      resultImage: null,
      errorMessage: null,
    }),

  addProgressEvent: (msg, passNum) =>
    set((s) => ({
      progressLogs: [...s.progressLogs, msg],
      currentPass: passNum !== undefined ? passNum : s.currentPass,
    })),

  setJobCompleted: (resultUrl) =>
    set({
      activeStatus: 'done',
      resultImage: resultUrl,
    }),

  setJobFailed: (error) =>
    set({
      activeStatus: 'failed',
      errorMessage: error,
    }),

  resetJobState: () =>
    set({
      jobId: null,
      activeStatus: 'idle',
      currentPass: 0,
      progressLogs: [],
      resultImage: null,
      errorMessage: null,
    }),
}));
