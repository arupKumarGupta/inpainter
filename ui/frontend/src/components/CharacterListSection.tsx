import React from 'react';
import { useJobStore } from '../store/jobStore';
import { Plus, Trash2, User, Layers } from 'lucide-react';
import type { CharacterConfig } from '../types';

export const CharacterListSection: React.FC = () => {
  const { config, addCharacter, removeCharacter, updateCharacter } = useJobStore();

  return (
    <div className="card-base space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sky-400 font-semibold text-sm">
          <Layers className="w-4 h-4" />
          <span>Characters & Inpaint Passes ({config.characters.length})</span>
        </div>
        <button
          type="button"
          onClick={addCharacter}
          className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold bg-sky-600/30 text-sky-300 border border-sky-500/40 rounded-lg hover:bg-sky-600/50 transition-colors"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>Add Character</span>
        </button>
      </div>

      <div className="space-y-3">
        {config.characters.map((char, index) => (
          <div
            key={char.id}
            className="p-3 bg-gray-950/70 border border-gray-800 rounded-lg space-y-3"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-sky-950 border border-sky-700/60 text-sky-400 flex items-center justify-center text-xs font-bold">
                  {index + 1}
                </span>
                <input
                  type="text"
                  className="bg-transparent font-medium text-xs text-gray-200 border-b border-transparent focus:border-sky-500 focus:outline-none"
                  value={char.name}
                  onChange={(e) => updateCharacter(char.id, { name: e.target.value })}
                />
              </div>
              {config.characters.length > 1 && (
                <button
                  type="button"
                  onClick={() => removeCharacter(char.id)}
                  className="text-gray-500 hover:text-red-400 transition-colors p-1"
                  title="Remove Character"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            <div>
              <label className="label-base">Character Prompt</label>
              <textarea
                rows={2}
                className="input-base text-xs"
                placeholder="Appearance, clothing, identity..."
                value={char.prompt}
                onChange={(e) => updateCharacter(char.id, { prompt: e.target.value })}
              />
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <label className="label-base">LoRA Path (Optional)</label>
                <input
                  type="text"
                  placeholder="loras/char.safetensors"
                  className="input-base text-xs font-mono"
                  value={char.lora?.path || ''}
                  onChange={(e) => {
                    const path = e.target.value;
                    updateCharacter(char.id, {
                      lora: path ? { path, strength: char.lora?.strength ?? 0.8 } : undefined,
                    });
                  }}
                />
              </div>
              <div>
                <label className="label-base">LoRA Weight ({char.lora?.strength ?? 0.8})</label>
                <input
                  type="range"
                  min={0.1}
                  max={1.5}
                  step={0.05}
                  disabled={!char.lora}
                  className="w-full accent-sky-500 disabled:opacity-40"
                  value={char.lora?.strength ?? 0.8}
                  onChange={(e) => {
                    if (char.lora) {
                      updateCharacter(char.id, {
                        lora: { ...char.lora, strength: parseFloat(e.target.value) || 0.8 },
                      });
                    }
                  }}
                />
              </div>
            </div>

            <div className="grid grid-cols-4 gap-2 pt-1 border-t border-gray-900">
              <div>
                <label className="label-base">Mask X</label>
                <input
                  type="number"
                  className="input-base text-xs py-1"
                  value={char.mask.x}
                  onChange={(e) =>
                    updateCharacter(char.id, {
                      mask: { ...char.mask, x: parseInt(e.target.value) || 0 },
                    })
                  }
                />
              </div>
              <div>
                <label className="label-base">Mask Y</label>
                <input
                  type="number"
                  className="input-base text-xs py-1"
                  value={char.mask.y}
                  onChange={(e) =>
                    updateCharacter(char.id, {
                      mask: { ...char.mask, y: parseInt(e.target.value) || 0 },
                    })
                  }
                />
              </div>
              <div>
                <label className="label-base">Width</label>
                <input
                  type="number"
                  className="input-base text-xs py-1"
                  value={char.mask.width}
                  onChange={(e) =>
                    updateCharacter(char.id, {
                      mask: { ...char.mask, width: parseInt(e.target.value) || 100 },
                    })
                  }
                />
              </div>
              <div>
                <label className="label-base">Height</label>
                <input
                  type="number"
                  className="input-base text-xs py-1"
                  value={char.mask.height}
                  onChange={(e) =>
                    updateCharacter(char.id, {
                      mask: { ...char.mask, height: parseInt(e.target.value) || 100 },
                    })
                  }
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
