/**
 * Step 1: Before/After Image Comparison
 * Shows original vs aligned images side-by-side
 */

import React, { useState } from 'react';
import { AlignmentFrame } from '../types/alignment';
import FrameSelector from './FrameSelector';
import '../styles/BeforeAfterView.css';

interface BeforeAfterViewProps {
  frames: AlignmentFrame[];
}

const BeforeAfterView: React.FC<BeforeAfterViewProps> = ({ frames }) => {
  const [selectedFrameIndex, setSelectedFrameIndex] = useState<number>(0);
  const [imageError, setImageError] = useState<{ before: boolean; after: boolean }>({
    before: false,
    after: false,
  });

  if (frames.length === 0) {
    return (
      <div className="before-after-view empty">
        <p>No frames available for comparison</p>
      </div>
    );
  }

  const currentFrame = frames[selectedFrameIndex];

  const handleImageError = (imageType: 'before' | 'after') => {
    setImageError((prev) => ({ ...prev, [imageType]: true }));
  };

  return (
    <div className="before-after-view">
      <div className="view-header">
        <h3>Before / After Alignment Comparison</h3>
        <p className="description">
          Compare the original image (before alignment) with the aligned result. 
          Select different frames to examine alignment quality across channels.
        </p>
      </div>

      <FrameSelector
        frames={frames}
        selectedIndex={selectedFrameIndex}
        onSelectFrame={setSelectedFrameIndex}
      />

      <div className="frame-info">
        <div className="info-badge">
          <strong>Frame:</strong> {currentFrame.frameId}
        </div>
        <div className="info-badge">
          <strong>Channel:</strong> {currentFrame.channelName}
        </div>
        <div className="info-badge">
          <strong>Transform:</strong> {currentFrame.movement.transformType}
        </div>
      </div>

      <div className="image-comparison">
        <div className="image-container before">
          <h4>Before Alignment</h4>
          <div className="image-wrapper">
            {imageError.before ? (
              <div className="image-placeholder error">
                <p>Image not available</p>
                <code>{currentFrame.beforeImageUrl}</code>
              </div>
            ) : (
              <img
                src={currentFrame.beforeImageUrl}
                alt={`Before alignment - ${currentFrame.channelName}`}
                onError={() => handleImageError('before')}
                onContextMenu={(e) => e.preventDefault()}
              />
            )}
          </div>
        </div>

        <div className="comparison-arrow">
          <svg width="40" height="40" viewBox="0 0 40 40">
            <path
              d="M10 20 L30 20 M25 15 L30 20 L25 25"
              stroke="currentColor"
              strokeWidth="2"
              fill="none"
            />
          </svg>
        </div>

        <div className="image-container after">
          <h4>After Alignment</h4>
          <div className="image-wrapper">
            {imageError.after ? (
              <div className="image-placeholder error">
                <p>Image not available</p>
                <code>{currentFrame.afterImageUrl}</code>
              </div>
            ) : (
              <img
                src={currentFrame.afterImageUrl}
                alt={`After alignment - ${currentFrame.channelName}`}
                onError={() => handleImageError('after')}
                onContextMenu={(e) => e.preventDefault()}
              />
            )}
          </div>
        </div>
      </div>

      <div className="quick-stats">
        <div className="stat">
          <span className="stat-label">Horizontal Shift:</span>
          <span className="stat-value">
            {currentFrame.movement.dx.toFixed(2)}px
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Vertical Shift:</span>
          <span className="stat-value">
            {currentFrame.movement.dy.toFixed(2)}px
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Total Movement:</span>
          <span className="stat-value">
            {currentFrame.movement.magnitude.toFixed(2)}px
          </span>
        </div>
      </div>
    </div>
  );
};

export default BeforeAfterView;


