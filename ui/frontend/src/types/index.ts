export type DeviceKind = 'cuda' | 'mps' | 'cpu' | 'auto';
export type DtypeKind = 'float16' | 'bfloat16' | 'float32' | 'auto';
export type ControlNetType = 'openpose' | 'depth' | 'canny';
export type MaskType = 'bbox' | 'polygon';

export interface MaskConfig {
  type: MaskType;
  x: number;
  y: number;
  width: number;
  height: number;
  points?: [number, number][];
}

export interface LoRAConfig {
  path: string;
  strength: number;
}

export interface ControlNetConfig {
  type: ControlNetType;
  image: string;
  strength: number;
}

export interface CharacterConfig {
  id: string;
  name: string;
  prompt: string;
  lora?: LoRAConfig;
  controlnet?: ControlNetConfig;
  mask: MaskConfig;
}

export interface InpaintConfig {
  strength: number;
  mask_padding: number;
  mask_blur: number;
  mask_grow: number;
}

export interface ResolutionConfig {
  width: number;
  height: number;
}

export interface ModelConfig {
  repo_id: string;
  dtype: DtypeKind;
  device: DeviceKind;
}

export interface SamplerConfig {
  steps: number;
  cfg: number;
  seed: number;
}

export interface OutputConfig {
  path: string;
  save_intermediates: boolean;
}

export interface JobConfig {
  scene_prompt: string;
  negative_prompt: string;
  resolution: ResolutionConfig;
  model: ModelConfig;
  sampler: SamplerConfig;
  characters: CharacterConfig[];
  inpaint: InpaintConfig;
  output: OutputConfig;
}

export interface ProgressEvent {
  pass_num: number;
  total_passes: number;
  status: string;
  message: string;
  timestamp: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: 'pending' | 'loading' | 'running_pass' | 'done' | 'failed';
  current_pass: number;
  total_passes: number;
  error?: string;
  result_path?: string;
  progress_events: ProgressEvent[];
}

export interface HealthCheckItem {
  name: string;
  status: 'PASS' | 'WARN' | 'FAIL';
  message: string;
  fix?: string;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded';
  device: string;
  timestamp: string;
  checks: HealthCheckItem[];
}
