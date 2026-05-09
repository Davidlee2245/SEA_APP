import React from 'react';

interface ShiftVector {
  dx: number;
  dy: number;
  magnitude: number;
  angle_deg?: number;
  type: 'reference' | 'aligned' | 'identity';
  residual_error?: number;
  num_matches?: number;
  note?: string;
}

interface ChannelItem {
  key: string;
  display_label: string;
  channel?: string;
}

interface AlignmentShiftPanelProps {
  shiftVectors: Record<string, ShiftVector>;
  availableItems: ChannelItem[];
  channels: Record<string, { enabled: boolean; color: [number, number, number]; name: string }>;
  onChannelHover?: (channelKey: string | null) => void;
  onChannelClick?: (channelKey: string) => void;
  highlightedChannel?: string | null;
}

const AlignmentShiftPanel: React.FC<AlignmentShiftPanelProps> = ({
  shiftVectors,
  availableItems,
  channels,
  onChannelHover,
  onChannelClick,
  highlightedChannel,
}) => {
  const hasShiftData = Object.keys(shiftVectors).length > 0;

  // Calculate summary statistics
  const summary = hasShiftData ? (() => {
    const maxShift = Math.max(...Object.values(shiftVectors).map(v => v.magnitude));
    const alignedCount = Object.values(shiftVectors).filter(v => v.type !== 'reference' && v.type !== 'identity').length;
    const identityCount = Object.values(shiftVectors).filter(v => v.type === 'identity').length;
    return { maxShift, alignedCount, identityCount };
  })() : null;

  return (
    <div className="alignment-shift-panel" style={{
      padding: '16px',
      background: '#f8f9fa',
      borderRadius: '6px',
      border: '2px solid #007bff',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      height: 'fit-content',
      maxHeight: 'calc(100vh - 200px)',
      overflowY: 'auto'
    }}>
      <h4 style={{ 
        margin: '0 0 16px 0', 
        fontSize: '16px', 
        fontWeight: 'bold', 
        color: '#007bff',
        borderBottom: '2px solid #007bff',
        paddingBottom: '8px'
      }}>
        📊 Alignment Shift Values
      </h4>

      {!hasShiftData ? (
        <div style={{ 
          padding: '24px', 
          background: '#fff', 
          borderRadius: '4px',
          border: '1px dashed #ccc',
          textAlign: 'center',
          color: '#666'
        }}>
          <p style={{ margin: 0, fontSize: '14px', fontWeight: '500' }}>
            Run alignment to view shift values
          </p>
          <p style={{ margin: '8px 0 0 0', fontSize: '12px', color: '#999' }}>
            Shift information will appear here after running alignment
          </p>
        </div>
      ) : (
        <>
          {/* Summary */}
          {summary && (
            <div style={{ 
              marginBottom: '16px', 
              padding: '12px', 
              background: '#e7f3ff', 
              borderRadius: '4px',
              fontSize: '12px',
              border: '1px solid #b3d9ff'
            }}>
              <div style={{ marginBottom: '4px' }}>
                <strong>Max shift:</strong> {summary.maxShift.toFixed(3)} px
              </div>
              <div style={{ marginBottom: '4px' }}>
                <strong>Aligned channels:</strong> {summary.alignedCount}
              </div>
              {summary.identityCount > 0 && (
                <div style={{ color: '#ff8800' }}>
                  <strong>⚠ Skipped:</strong> {summary.identityCount}
                </div>
              )}
            </div>
          )}

          {/* Channel List */}
          <div style={{ 
            display: 'flex',
            flexDirection: 'column',
            gap: '10px',
            fontSize: '12px'
          }}>
            {Object.keys(shiftVectors).map((channelKey) => {
              const shift = shiftVectors[channelKey];
              const channelConfig = channels[channelKey];
              const item = availableItems.find(i => i.key === channelKey);
              const displayLabel = item ? item.display_label : channelKey;
              
              if (!channelConfig) return null;
              
              const [r, g, b] = channelConfig.color;
              const color = `rgb(${r * 255}, ${g * 255}, ${b * 255})`;
              
              let statusText = '';
              let statusColor = '#666';
              
              if (shift.type === 'reference') {
                statusText = 'REF';
                statusColor = '#28a745';
              } else if (shift.type === 'identity') {
                statusText = '⚠ IDENTITY';
                statusColor = '#ff8800';
              } else {
                statusText = 'ALIGNED';
                statusColor = '#007bff';
              }

              const isHighlighted = highlightedChannel === channelKey;
              const isHovered = highlightedChannel === channelKey;
              
              return (
                <div 
                  key={channelKey}
                  onClick={() => onChannelClick?.(channelKey)}
                  onMouseEnter={() => onChannelHover?.(channelKey)}
                  onMouseLeave={() => onChannelHover?.(null)}
                  style={{
                    padding: '12px',
                    background: isHighlighted || isHovered ? '#fff9e6' : '#fff',
                    borderRadius: '4px',
                    border: `2px solid ${color}`,
                    borderStyle: shift.type === 'identity' ? 'dashed' : 'solid',
                    opacity: shift.type === 'identity' ? 0.7 : 1.0,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    boxShadow: isHighlighted || isHovered ? '0 2px 8px rgba(0,0,0,0.15)' : 'none',
                    transform: isHighlighted || isHovered ? 'translateY(-2px)' : 'none'
                  }}
                >
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'space-between',
                    marginBottom: '8px'
                  }}>
                    <div style={{ 
                      fontWeight: 'bold', 
                      color: color,
                      fontSize: '13px'
                    }}>
                      {displayLabel}
                    </div>
                    <div style={{ 
                      fontSize: '11px', 
                      color: statusColor,
                      fontWeight: '600',
                      padding: '2px 8px',
                      background: statusColor === '#28a745' ? '#d4edda' : 
                                  statusColor === '#ff8800' ? '#fff3cd' : '#d1ecf1',
                      borderRadius: '12px'
                    }}>
                      {statusText}
                    </div>
                  </div>

                  {shift.type !== 'reference' && shift.type !== 'identity' && (
                    <div style={{ 
                      display: 'grid', 
                      gridTemplateColumns: '1fr 1fr',
                      gap: '6px',
                      marginTop: '8px',
                      paddingTop: '8px',
                      borderTop: '1px solid #e9ecef'
                    }}>
                      <div>
                        <div style={{ fontSize: '10px', color: '#666', marginBottom: '2px' }}>Δx</div>
                        <div style={{ fontWeight: '600', color: '#333' }}>
                          {shift.dx >= 0 ? '+' : ''}{shift.dx.toFixed(3)} px
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '10px', color: '#666', marginBottom: '2px' }}>Δy</div>
                        <div style={{ fontWeight: '600', color: '#333' }}>
                          {shift.dy >= 0 ? '+' : ''}{shift.dy.toFixed(3)} px
                        </div>
                      </div>
                      <div style={{ gridColumn: '1 / -1' }}>
                        <div style={{ fontSize: '10px', color: '#666', marginBottom: '2px' }}>|Δ| (magnitude)</div>
                        <div style={{ fontWeight: '600', color: '#333' }}>
                          {shift.magnitude.toFixed(3)} px
                        </div>
                      </div>
                      {shift.angle_deg !== undefined && (
                        <>
                          <div>
                            <div style={{ fontSize: '10px', color: '#666', marginBottom: '2px' }}>Rotation</div>
                            <div style={{ fontWeight: '600', color: '#333' }}>
                              {shift.angle_deg >= 0 ? '+' : ''}{shift.angle_deg.toFixed(2)}°
                            </div>
                          </div>
                          <div>
                            <div style={{ fontSize: '10px', color: '#666', marginBottom: '2px' }}>|θ|</div>
                            <div style={{ fontWeight: '600', color: '#333' }}>
                              {Math.abs(shift.angle_deg).toFixed(2)}°
                            </div>
                          </div>
                        </>
                      )}
                      {shift.residual_error !== undefined && (
                        <div style={{ gridColumn: '1 / -1', fontSize: '10px', color: '#666', marginTop: '4px' }}>
                          <strong>Residual:</strong> {shift.residual_error.toFixed(2)} px
                        </div>
                      )}
                      {shift.num_matches !== undefined && shift.num_matches > 0 && (
                        <div style={{ gridColumn: '1 / -1', fontSize: '10px', color: '#666' }}>
                          <strong>Matches:</strong> {shift.num_matches}
                        </div>
                      )}
                    </div>
                  )}

                  {shift.type === 'reference' && (
                    <div style={{ 
                      fontSize: '11px', 
                      color: '#666', 
                      fontStyle: 'italic',
                      marginTop: '4px'
                    }}>
                      Reference channel (no shift)
                    </div>
                  )}

                  {shift.type === 'identity' && (
                    <div style={{ 
                      fontSize: '11px', 
                      color: '#666', 
                      fontStyle: 'italic',
                      marginTop: '4px'
                    }}>
                      No alignment applied
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Legend */}
          <div style={{ 
            marginTop: '16px', 
            padding: '12px', 
            background: '#e9ecef', 
            borderRadius: '4px', 
            fontSize: '11px',
            border: '1px solid #dee2e6'
          }}>
            <strong style={{ display: 'block', marginBottom: '6px' }}>Legend:</strong>
            <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
              <li><span style={{ color: '#28a745', fontWeight: '600' }}>REF</span> = Reference channel</li>
              <li><span style={{ color: '#007bff', fontWeight: '600' }}>ALIGNED</span> = Aligned with transform</li>
              <li><span style={{ color: '#ff8800', fontWeight: '600' }}>⚠ IDENTITY</span> = Skipped (no alignment)</li>
            </ul>
          </div>
        </>
      )}
    </div>
  );
};

export default AlignmentShiftPanel;

