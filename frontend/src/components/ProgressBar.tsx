/**
 * Progress Bar Component
 */

import React from 'react';
import '../styles/ProgressBar.css';

interface ProgressBarProps {
  progress: number; // 0.0 to 1.0
  showPercentage?: boolean;
}

const ProgressBar: React.FC<ProgressBarProps> = ({
  progress,
  showPercentage = true,
}) => {
  const percentage = Math.round(progress * 100);

  return (
    <div className="progress-bar-container">
      <div className="progress-bar-background">
        <div
          className="progress-bar-fill"
          style={{ width: `${percentage}%` }}
        />
      </div>
      {showPercentage && (
        <div className="progress-percentage">{percentage}%</div>
      )}
    </div>
  );
};

export default ProgressBar;


