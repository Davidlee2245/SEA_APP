/**
 * Step 2: Movement Quantification
 * Displays alignment movement metrics per frame
 */

import React, { useState } from 'react';
import { AlignmentFrame } from '../types/alignment';
import '../styles/MovementQuantification.css';

interface MovementQuantificationProps {
  frames: AlignmentFrame[];
}

const MovementQuantification: React.FC<MovementQuantificationProps> = ({ frames }) => {
  const [sortBy, setSortBy] = useState<'frameId' | 'magnitude'>('frameId');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');

  const sortedFrames = [...frames].sort((a, b) => {
    let compareValue = 0;

    if (sortBy === 'frameId') {
      compareValue = a.frameId.localeCompare(b.frameId);
    } else if (sortBy === 'magnitude') {
      compareValue = a.movement.magnitude - b.movement.magnitude;
    }

    return sortOrder === 'asc' ? compareValue : -compareValue;
  });

  const maxMagnitude = Math.max(...frames.map((f) => f.movement.magnitude));
  const avgMagnitude = frames.reduce((sum, f) => sum + f.movement.magnitude, 0) / frames.length;

  const handleSort = (column: 'frameId' | 'magnitude') => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortOrder('asc');
    }
  };

  const getBarWidth = (magnitude: number): number => {
    return maxMagnitude > 0 ? (magnitude / maxMagnitude) * 100 : 0;
  };

  const getMagnitudeClass = (magnitude: number): string => {
    if (magnitude < 1) return 'low';
    if (magnitude < 5) return 'medium';
    return 'high';
  };

  return (
    <div className="movement-quantification">
      <div className="view-header">
        <h3>Movement Quantification</h3>
        <p className="description">
          Analysis of alignment corrections applied to each frame. 
          Larger movements indicate greater initial misalignment.
        </p>
      </div>

      <div className="summary-stats">
        <div className="stat-card">
          <h4>Total Frames</h4>
          <div className="stat-value">{frames.length}</div>
        </div>
        <div className="stat-card">
          <h4>Average Movement</h4>
          <div className="stat-value">{avgMagnitude.toFixed(2)} px</div>
        </div>
        <div className="stat-card">
          <h4>Maximum Movement</h4>
          <div className="stat-value">{maxMagnitude.toFixed(2)} px</div>
        </div>
      </div>

      <div className="table-controls">
        <label>
          Sort by:
          <select value={sortBy} onChange={(e) => handleSort(e.target.value as any)}>
            <option value="frameId">Frame ID</option>
            <option value="magnitude">Movement Magnitude</option>
          </select>
        </label>
      </div>

      <div className="movement-table-container">
        <table className="movement-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('frameId')}>
                Frame / Channel {sortBy === 'frameId' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th>Transform Type</th>
              <th>Δx (pixels)</th>
              <th>Δy (pixels)</th>
              <th onClick={() => handleSort('magnitude')}>
                Total Movement {sortBy === 'magnitude' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th>Residual Error</th>
              <th>Matches</th>
            </tr>
          </thead>
          <tbody>
            {sortedFrames.map((frame, index) => (
              <tr key={`${frame.frameId}-${index}`} className={getMagnitudeClass(frame.movement.magnitude)}>
                <td className="frame-cell">
                  <div className="frame-id">{frame.frameId}</div>
                  <div className="channel-name">{frame.channelName}</div>
                </td>
                <td>
                  <span className={`transform-badge ${frame.movement.transformType}`}>
                    {frame.movement.transformType.toUpperCase()}
                  </span>
                </td>
                <td className="numeric">{frame.movement.dx.toFixed(2)}</td>
                <td className="numeric">{frame.movement.dy.toFixed(2)}</td>
                <td className="magnitude-cell">
                  <div className="magnitude-bar-container">
                    <div
                      className={`magnitude-bar ${getMagnitudeClass(frame.movement.magnitude)}`}
                      style={{ width: `${getBarWidth(frame.movement.magnitude)}%` }}
                    />
                    <span className="magnitude-value">
                      {frame.movement.magnitude.toFixed(2)}
                    </span>
                  </div>
                </td>
                <td className="numeric">
                  {frame.movement.residualError !== undefined
                    ? frame.movement.residualError.toFixed(2)
                    : 'N/A'}
                </td>
                <td className="numeric">
                  {frame.movement.numMatches !== undefined
                    ? frame.movement.numMatches
                    : 'N/A'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="legend">
        <h4>Movement Levels:</h4>
        <div className="legend-items">
          <div className="legend-item">
            <span className="legend-color low"></span>
            <span>Low (&lt; 1px)</span>
          </div>
          <div className="legend-item">
            <span className="legend-color medium"></span>
            <span>Medium (1-5px)</span>
          </div>
          <div className="legend-item">
            <span className="legend-color high"></span>
            <span>High (&gt; 5px)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MovementQuantification;

