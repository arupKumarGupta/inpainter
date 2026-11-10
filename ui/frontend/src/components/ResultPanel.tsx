import React from 'react';
import { useJobStore } from '../store/jobStore';
import { Image as ImageIcon, Download } from 'lucide-react';

export const ResultPanel: React.FC = () => {
  const { resultImage, activeStatus } = useJobStore();

  if (!resultImage && activeStatus !== 'done') {
    return null;
  }

  const imageUrl = resultImage?.startsWith('http')
    ? resultImage
    : `http://localhost:8000${resultImage}`;

  return (
    <div className="card-base space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-emerald-400 font-semibold text-sm">
          <ImageIcon className="w-4 h-4" />
          <span>Final Output</span>
        </div>
        {resultImage && (
          <a
            href={imageUrl}
            download="inpainter_output.png"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1 text-xs text-sky-400 hover:text-sky-300 font-medium transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download</span>
          </a>
        )}
      </div>

      <div className="bg-gray-950 rounded-lg overflow-hidden border border-gray-800 flex items-center justify-center min-h-[240px]">
        {resultImage ? (
          <img
            src={imageUrl}
            alt="Inpainter Output"
            className="w-full h-auto object-contain max-h-[500px]"
          />
        ) : (
          <span className="text-xs text-gray-500">Image will appear here</span>
        )}
      </div>
    </div>
  );
};
