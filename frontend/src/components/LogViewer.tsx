/**
 * Real-time Log Viewer Component
 */

import React, { useEffect, useRef } from 'react';
import '../styles/LogViewer.css';
import { copyText } from '../lib/clipboard';

interface LogViewerProps {
  logs: string[];
  maxLines?: number;
}

const LogViewer: React.FC<LogViewerProps> = ({ logs, maxLines = 500 }) => {
  const logEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = React.useState(true);

  useEffect(() => {
    if (autoScroll && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  const handleScroll = () => {
    if (containerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
      setAutoScroll(isAtBottom);
    }
  };

  const displayedLogs = logs.slice(-maxLines);

  return (
    <div className="log-viewer">
      <div className="log-viewer-header">
        <h3>📋 Pipeline Logs</h3>
        <div className="log-controls">
          <label>
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
            />
            Auto-scroll
          </label>
          <button onClick={async () => {
            const copied = await copyText(logs.join('\n'));
            if (!copied) {
              console.warn('Failed to copy logs to clipboard');
            }
          }}>
            📋 Copy All
          </button>
        </div>
      </div>
      
      <div
        className="log-container"
        ref={containerRef}
        onScroll={handleScroll}
      >
        {displayedLogs.map((log, index) => (
          <div key={index} className="log-line">
            {log}
          </div>
        ))}
        <div ref={logEndRef} />
      </div>
    </div>
  );
};

export default LogViewer;


