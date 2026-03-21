/**
 * Frame Selector Component
 * Allows users to select which frame to view
 */

import React from 'react';
import { AlignmentFrame } from '../types/alignment';
import '../styles/FrameSelector.css';

interface FrameSelectorProps {
  frames: AlignmentFrame[];
  selectedIndex: number;
  onSelectFrame: (index: number) => void;
}

const FrameSelector: React.FC<FrameSelectorProps> = ({
  frames,
  selectedIndex,
  onSelectFrame,
}) => {
  return (
    <div className="frame-selector">
      <label className="frame-selector-label">Select Frame:</label>
      <div className="frame-selector-buttons">
        {frames.map((frame, index) => (
          <button
            key={`${frame.frameId}-${index}`}
            className={`frame-button ${index === selectedIndex ? 'selected' : ''}`}
            onClick={() => onSelectFrame(index)}
            title={`${frame.frameId} - ${frame.channelName}`}
          >
            <div className="frame-button-content">
              <span className="frame-id">{frame.frameId}</span>
              <span className="channel-name">{frame.channelName}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default FrameSelector;


