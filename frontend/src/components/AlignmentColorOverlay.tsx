/**
 * Alignment Color Overlay QC Component
 * Multi-channel pseudo-color visualization for alignment verification
 * Enhanced with shift vector visualization
 */

import React, { useState, useEffect, useRef } from 'react';
import '../styles/AlignmentColorOverlay.css';
import { getApiBase } from '../lib/apiBase';

interface ChannelConfig {
  enabled: boolean;
  color: [number, number, number]; // RGB multipliers
  name: string;
}

interface ChannelItem {
  key: string;
  cycle: string;
  channel: string;
  marker: string | null;
  display_label: string;
  tiff_path: string;
  preview_url: string | null;
}

interface ShiftVector {
  dx: number;
  dy: number;
  magnitude: number;
  type: 'affine' | 'tps' | 'identity' | 'reference';
  residual_error?: number;
  num_matches?: number;
  note?: string;
}

export interface CropRect { x: number; y: number; w: number; h: number }

interface AlignmentColorOverlayProps {
  stages: {
    raw?: Record<string, string>;
    contrast_enhance?: Record<string, string>;
    step1?: Record<string, string>;
    step2?: Record<string, string>;
    step3?: Record<string, string>;
    step4?: Record<string, string>;
    aligned?: Record<string, string>;
  };
  availableItems: ChannelItem[];
  isLoaded: boolean;
  // Alignment metadata for shift vector visualization
  sample?: string;
  position?: string;
  refChannel?: string;
  inputStage?: string;
  shiftVectors?: Record<string, ShiftVector>;
  // Callbacks for channel interaction
  onChannelHover?: (channelKey: string | null) => void;
  onChannelClick?: (channelKey: string) => void;
  highlightedChannel?: string | null;
  // Crop tool
  cropRect?: CropRect | null;
  onCropRectChange?: (rect: CropRect | null) => void;
  cropInteractive?: boolean;
}

// Get color for a channel number (1-4)
const getChannelColor = (channelNum: number): { color: [number, number, number]; name: string } => {
  const colors = [
    { color: [1, 0, 0] as [number, number, number], name: 'Red' },      // ch1
    { color: [1, 1, 0] as [number, number, number], name: 'Yellow' }, // ch2
    { color: [0, 1, 0] as [number, number, number], name: 'Green' },   // ch3
    { color: [0, 0, 1] as [number, number, number], name: 'Blue' },   // ch4
  ];
  return colors[(channelNum - 1) % 4];
};

const AlignmentColorOverlay: React.FC<AlignmentColorOverlayProps> = ({
  stages,
  availableItems,
  isLoaded,
  sample,
  position,
  refChannel,
  inputStage,
  shiftVectors: propShiftVectors,
  onChannelHover,
  onChannelClick,
  highlightedChannel,
  cropRect,
  onCropRectChange,
  cropInteractive = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const beforeImagesRef = useRef<Record<string, HTMLImageElement>>({});
  const cropOverlayRef = useRef<HTMLDivElement>(null);

  // Crop drag state (display coords); converted to image space on commit
  const cropDragRef = useRef<{
    dragging: boolean;
    startX: number;
    startY: number;
    currentX: number;
    currentY: number;
  } | null>(null);
  const [cropDragDisplay, setCropDragDisplay] = useState<{
    x: number; y: number; w: number; h: number;
  } | null>(null);
  
  // Auto-select the first available stage (prefer raw, then others)
  const getDefaultStage = (): keyof typeof stages => {
    if (stages.raw && Object.keys(stages.raw).length > 0) return 'raw';
    if (stages.contrast_enhance && Object.keys(stages.contrast_enhance).length > 0) return 'contrast_enhance';
    if (stages.step1 && Object.keys(stages.step1).length > 0) return 'step1';
    if (stages.aligned && Object.keys(stages.aligned).length > 0) return 'aligned';
    return 'raw';
  };
  
  const [stage, setStage] = useState<keyof typeof stages>(getDefaultStage());
  const [globalAlpha, setGlobalAlpha] = useState<number>(0.35);
  const [background, setBackground] = useState<'black' | 'white'>('black');
  const [channels, setChannels] = useState<Record<string, ChannelConfig>>({});
  const [isRendering, setIsRendering] = useState(false);
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 800 });
  
  // NEW: Shift vector visualization state
  const [shiftVectors, setShiftVectors] = useState<Record<string, ShiftVector>>({});
  const [showShiftVectors, setShowShiftVectors] = useState(true);
  const [showBeforeAfter, setShowBeforeAfter] = useState(false);
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [arrowScale, setArrowScale] = useState(5.0); // Scale factor for arrow length (increased default)
  const [arrowBaseLength, setArrowBaseLength] = useState(50); // Base arrow length in pixels
  const [hoveredChannel, setHoveredChannel] = useState<string | null>(null);
  
  // Auto-update stage when new stages become available
  useEffect(() => {
    const availableStages = Object.keys(stages).filter(
      (key) => stages[key as keyof typeof stages] && Object.keys(stages[key as keyof typeof stages]!).length > 0
    );
    
    // If current stage is not available, switch to first available
    if (!availableStages.includes(stage) && availableStages.length > 0) {
      setStage(availableStages[availableStages.length - 1] as keyof typeof stages); // Use latest stage
    }
  }, [stages, stage]);

  // Initialize channel configs
  useEffect(() => {
    const initialChannels: Record<string, ChannelConfig> = {};
    availableItems.forEach((item) => {
      // Extract channel number from item.channel (e.g., "Ch1" -> 1)
      const channelMatch = item.channel.match(/ch(\d+)/i);
      if (channelMatch) {
        const channelNum = parseInt(channelMatch[1], 10);
        const colorInfo = getChannelColor(channelNum);
        initialChannels[item.key] = {
          enabled: true,
          color: colorInfo.color,
          name: colorInfo.name,
        };
      }
    });
    setChannels(initialChannels);
  }, [availableItems]);

  // Use shift vectors from props when viewing aligned stage
  useEffect(() => {
    console.log('[Color Overlay] Stage:', stage, 'propShiftVectors:', propShiftVectors);
    if (stage === 'aligned' && propShiftVectors) {
      setShiftVectors(propShiftVectors);
      console.log('[Color Overlay] Using shift vectors from props:', propShiftVectors);
      console.log('[Color Overlay] Shift vectors keys:', Object.keys(propShiftVectors));
    } else {
      console.log('[Color Overlay] Clearing shift vectors (stage:', stage, 'has props:', !!propShiftVectors, ')');
      setShiftVectors({});
    }
  }, [stage, propShiftVectors]);

  // Render overlay whenever settings change
  useEffect(() => {
    if (!isLoaded || !stages[stage]) return;
    renderOverlay();
  }, [stage, globalAlpha, background, channels, stages, isLoaded, showShiftVectors, showBeforeAfter, shiftVectors, arrowScale, highlightedChannel]);

  const loadImage = (url: string): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = reject;
      // Ensure URL is absolute (add backend server if relative)
      const absoluteUrl = url.startsWith('http') ? url : `${getApiBase()}${url}`;
      try {
        const u = new URL(absoluteUrl);
        // Keep existing token if present; avoid duplicate t=...&t=...
        if (!u.searchParams.has('t')) {
          u.searchParams.set('t', String(Date.now()));
        }
        img.src = u.toString();
      } catch {
        img.src = absoluteUrl;
      }
    });
  };

  // Draw arrow for shift vector (enhanced visibility)
  const drawArrow = (
    ctx: CanvasRenderingContext2D,
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    color: string,
    lineWidth: number = 4,
    arrowheadSize: number = 15
  ) => {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Draw line with shadow for visibility
    ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
    ctx.shadowBlur = 3;
    ctx.shadowOffsetX = 1;
    ctx.shadowOffsetY = 1;
    
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    
    // Draw arrowhead (larger and more visible)
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const arrowAngle = Math.PI / 6; // 30 degrees
    
    ctx.shadowBlur = 0; // No shadow on arrowhead
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(
      x2 - arrowheadSize * Math.cos(angle - arrowAngle),
      y2 - arrowheadSize * Math.sin(angle - arrowAngle)
    );
    ctx.lineTo(
      x2 - arrowheadSize * Math.cos(angle + arrowAngle),
      y2 - arrowheadSize * Math.sin(angle + arrowAngle)
    );
    ctx.closePath();
    ctx.fill();
    
    // Draw starting point circle (more visible)
    ctx.beginPath();
    ctx.arc(x1, y1, 6, 0, 2 * Math.PI);
    ctx.fill();
    
    ctx.restore();
  };

  const renderOverlay = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    setIsRendering(true);

    try {
      const currentStage = stages[stage];
      if (!currentStage) {
        console.log(`[Color Overlay] Stage "${stage}" has no data yet`);
        setIsRendering(false);
        return;
      }

      console.log(`[Color Overlay] Rendering stage "${stage}" with channels:`, Object.keys(currentStage));
      console.log(`[Color Overlay] Stage URLs:`, currentStage);

      // Load all enabled channel images
      const imagePromises: Promise<{ channel: string; img: HTMLImageElement }>[] = [];
      Object.keys(channels).forEach((ch) => {
        const url = currentStage[ch];
        if (channels[ch].enabled && url) {
          console.log(`[Color Overlay] Loading ${ch}: ${url}`);
          imagePromises.push(
            loadImage(url).then((img) => ({ channel: ch, img }))
          );
        } else if (channels[ch].enabled && !url) {
          console.warn(`[Color Overlay] ${ch} is enabled but has no URL in stage "${stage}"`);
        }
      });

      const loadedImages = await Promise.all(imagePromises);

      if (loadedImages.length === 0) {
        console.log('[Color Overlay] No enabled channels to render');
        setIsRendering(false);
        return;
      }

      console.log(`[Color Overlay] Loaded ${loadedImages.length} channel images`);

      // Use first image to determine canvas size
      const firstImg = loadedImages[0].img;
      canvas.width = firstImg.width;
      canvas.height = firstImg.height;
      setCanvasSize({ width: firstImg.width, height: firstImg.height });

      // Fill background
      ctx.fillStyle = background === 'black' ? '#000000' : '#FFFFFF';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Load "before" images if showing before/after
      if (showBeforeAfter && stage === 'aligned' && inputStage && stages[inputStage]) {
        const beforePromises: Promise<{ channel: string; img: HTMLImageElement }>[] = [];
        Object.keys(channels).forEach((ch) => {
          const url = stages[inputStage!]?.[ch];
          if (channels[ch].enabled && url) {
            beforePromises.push(
              loadImage(url).then((img) => {
                beforeImagesRef.current[ch] = img;
                return { channel: ch, img };
              })
            );
          }
        });
        await Promise.all(beforePromises);
      }

      // Create offscreen canvas for compositing
      const offscreen = document.createElement('canvas');
      offscreen.width = canvas.width;
      offscreen.height = canvas.height;
      const offCtx = offscreen.getContext('2d');
      if (!offCtx) return;

      // Draw "before" images as ghost overlay if enabled
      if (showBeforeAfter && stage === 'aligned') {
        for (const { channel, img } of loadedImages) {
          const beforeImg = beforeImagesRef.current[channel];
          if (!beforeImg) continue;
          
          const channelConfig = channels[channel];
          if (!channelConfig) continue;

          // Draw grayscale image to offscreen
          offCtx.clearRect(0, 0, offscreen.width, offscreen.height);
          offCtx.drawImage(beforeImg, 0, 0);

          // Get image data
          const imageData = offCtx.getImageData(0, 0, offscreen.width, offscreen.height);
          const data = imageData.data;

          // Apply color tint with low opacity (ghost effect)
          const [r, g, b] = channelConfig.color;
          for (let i = 0; i < data.length; i += 4) {
            const intensity = data[i]; // Grayscale, so R = G = B
            const alpha = (intensity / 255) * globalAlpha * 0.3; // 30% opacity for ghost

            data[i] = intensity * r; // R
            data[i + 1] = intensity * g; // G
            data[i + 2] = intensity * b; // B
            data[i + 3] = alpha * 255; // A
          }

          offCtx.putImageData(imageData, 0, 0);

          // Composite onto main canvas with additive blending
          ctx.globalCompositeOperation = 'lighter';
          ctx.drawImage(offscreen, 0, 0);
        }
      }

      // Composite each channel (aligned/current)
      for (const { channel, img } of loadedImages) {
        const channelConfig = channels[channel];
        if (!channelConfig) continue;

        // Highlight if this channel is being hovered/selected from shift panel
        const isHighlighted = highlightedChannel === channel;
        const channelAlpha = isHighlighted ? globalAlpha * 1.5 : globalAlpha; // Increase opacity for highlighted channel

        // Draw grayscale image to offscreen
        offCtx.clearRect(0, 0, offscreen.width, offscreen.height);
        offCtx.drawImage(img, 0, 0);

        // Get image data
        const imageData = offCtx.getImageData(0, 0, offscreen.width, offscreen.height);
        const data = imageData.data;

        // Apply color tint with alpha (brighter for highlighted channel)
        const [r, g, b] = channelConfig.color;
        const colorMultiplier = isHighlighted ? 1.3 : 1.0; // Make highlighted channel brighter
        for (let i = 0; i < data.length; i += 4) {
          const intensity = data[i]; // Grayscale, so R = G = B
          const alpha = (intensity / 255) * channelAlpha;

          data[i] = Math.min(255, intensity * r * colorMultiplier); // R
          data[i + 1] = Math.min(255, intensity * g * colorMultiplier); // G
          data[i + 2] = Math.min(255, intensity * b * colorMultiplier); // B
          data[i + 3] = alpha * 255; // A
        }

        offCtx.putImageData(imageData, 0, 0);

        // Composite onto main canvas with additive blending
        ctx.globalCompositeOperation = 'lighter';
        ctx.drawImage(offscreen, 0, 0);
      }

      // Reset composite operation
      ctx.globalCompositeOperation = 'source-over';
      
      // Draw shift vectors if enabled and viewing aligned stage
      if (showShiftVectors && stage === 'aligned' && Object.keys(shiftVectors).length > 0) {
        drawShiftVectors(ctx, canvas.width, canvas.height);
      }
      
      // Draw annotations
      if (showAnnotations && stage === 'aligned' && Object.keys(shiftVectors).length > 0) {
        drawAnnotations(ctx, canvas.width, canvas.height);
      }
      
      // Draw scale reference
      if (showShiftVectors && stage === 'aligned') {
        drawScaleReference(ctx, canvas.width, canvas.height);
      }
      
      console.log(`[Color Overlay] ✓ Rendering complete (${loadedImages.length} channels blended)`);
    } catch (error) {
      console.error('[Color Overlay] Error rendering overlay:', error);
    } finally {
      setIsRendering(false);
    }
  };

  const drawShiftVectors = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    ctx.save();
    
    // Draw multiple arrows across the image for better visibility
    // Use a grid of positions to show vectors at different locations
    const gridPositions = [
      { x: width * 0.25, y: height * 0.25 },
      { x: width * 0.5, y: height * 0.25 },
      { x: width * 0.75, y: height * 0.25 },
      { x: width * 0.25, y: height * 0.5 },
      { x: width * 0.5, y: height * 0.5 }, // Center
      { x: width * 0.75, y: height * 0.5 },
      { x: width * 0.25, y: height * 0.75 },
      { x: width * 0.5, y: height * 0.75 },
      { x: width * 0.75, y: height * 0.75 },
    ];
    
    // Find maximum shift magnitude for scaling
    const maxMagnitude = Math.max(
      ...Object.values(shiftVectors).map(v => v.magnitude),
      0.01 // Minimum to avoid division by zero
    );
    
    // Scale factor: use base length and scale factor
    // Each pixel of shift becomes arrowBaseLength pixels on screen (scaled by arrowScale)
    const pixelsPerShiftUnit = (arrowBaseLength / maxMagnitude) * arrowScale;
    // For very small shifts (< 0.1 px), use much larger scale to make arrows visible
    // For larger shifts, use reasonable scale
    const minScale = maxMagnitude < 0.1 ? 1000.0 : 20.0; // 1000x for sub-pixel shifts
    const scale = Math.max(pixelsPerShiftUnit, minScale);
    
    console.log(`[Shift Vectors] Max magnitude: ${maxMagnitude.toFixed(3)} px, Scale: ${scale.toFixed(1)}x`);
    
    Object.keys(shiftVectors).forEach((channelKey) => {
      const shift = shiftVectors[channelKey];
      const channelConfig = channels[channelKey];
      
      if (!channelConfig || !channelConfig.enabled) return;
      
      // Skip reference channel and identity transforms
      if (shift.type === 'reference' || shift.type === 'identity' || shift.magnitude < 0.01) {
        return;
      }
      
      // Use channel color for arrow (make it brighter/more visible)
      const [r, g, b] = channelConfig.color;
      const color = `rgb(${Math.min(r * 255, 255)}, ${Math.min(g * 255, 255)}, ${Math.min(b * 255, 255)})`;
      
      // Highlight if this channel is being hovered/selected
      const isHighlighted = highlightedChannel === channelKey;
      const arrowThickness = isHighlighted ? 6 : 4;
      const arrowHeadSize = isHighlighted ? 22 : 18;
      const centerArrowThickness = isHighlighted ? 7 : 5;
      const centerArrowHeadSize = isHighlighted ? 25 : 20;
      
      // Draw arrows at multiple grid positions
      gridPositions.forEach((gridPos) => {
        // Calculate arrow endpoints based on shift
        const dx = shift.dx * scale;
        const dy = shift.dy * scale;
        const startX = gridPos.x;
        const startY = gridPos.y;
        const endX = startX + dx;
        const endY = startY + dy;
        
        // Only draw if arrow is within canvas bounds
        if (endX >= 0 && endX <= width && endY >= 0 && endY <= height) {
          // Draw arrow with thicker line for visibility (even thicker if highlighted)
          drawArrow(ctx, startX, startY, endX, endY, color, arrowThickness, arrowHeadSize);
        }
      });
      
      // Also draw a large arrow in the center for reference
      const centerX = width / 2;
      const centerY = height / 2;
      const centerDx = shift.dx * scale;
      const centerDy = shift.dy * scale;
      const centerStartX = centerX - centerDx / 2;
      const centerStartY = centerY - centerDy / 2;
      const centerEndX = centerX + centerDx / 2;
      const centerEndY = centerY + centerDy / 2;
      
      // Draw center arrow with even thicker line (even thicker if highlighted)
      drawArrow(ctx, centerStartX, centerStartY, centerEndX, centerEndY, color, centerArrowThickness, centerArrowHeadSize);
      
      // Add text label near center arrow showing the shift
      ctx.save();
      ctx.fillStyle = color;
      ctx.strokeStyle = background === 'black' ? '#FFFFFF' : '#000000';
      ctx.lineWidth = 3;
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      
      const labelText = `Δx=${shift.dx.toFixed(1)}px, Δy=${shift.dy.toFixed(1)}px`;
      const labelY = centerEndY + 25;
      
      // Draw text with outline for visibility
      ctx.strokeText(labelText, centerEndX, labelY);
      ctx.fillText(labelText, centerEndX, labelY);
      ctx.restore();
    });
    
    ctx.restore();
  };

  const drawAnnotations = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    ctx.save();
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    
    const padding = 10;
    let yOffset = padding;
    
    // Draw title
    ctx.fillStyle = background === 'black' ? '#FFFFFF' : '#000000';
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText('Shift Vectors:', padding, yOffset);
    yOffset += 20;
    
    ctx.font = '12px monospace';
    
    Object.keys(shiftVectors).forEach((channelKey) => {
      const shift = shiftVectors[channelKey];
      const channelConfig = channels[channelKey];
      const item = availableItems.find(i => i.key === channelKey);
      const displayLabel = item ? item.display_label : channelKey;
      
      if (!channelConfig || !channelConfig.enabled) return;
      
      const [r, g, b] = channelConfig.color;
      const color = `rgb(${r * 255}, ${g * 255}, ${b * 255})`;
      
      let text = `${displayLabel}: `;
      
      if (shift.type === 'reference') {
        text += 'REF (no shift)';
        ctx.fillStyle = background === 'black' ? '#888888' : '#666666';
      } else if (shift.type === 'identity') {
        text += '⚠ IDENTITY (skipped)';
        ctx.fillStyle = '#FF8800';
      } else {
        text += `Δx=${shift.dx.toFixed(1)}px, Δy=${shift.dy.toFixed(1)}px, |Δ|=${shift.magnitude.toFixed(1)}px`;
        ctx.fillStyle = color;
      }
      
      // Highlight if this channel is being hovered/selected from shift panel
      if (highlightedChannel === channelKey) {
        ctx.fillStyle = background === 'black' ? '#FFFF00' : '#0000FF';
        ctx.font = 'bold 14px monospace';
      }
      
      ctx.fillText(text, padding, yOffset);
      yOffset += 18;
      
      // Add residual error and matches info for non-identity
      if (shift.type !== 'identity' && shift.type !== 'reference' && shift.residual_error !== undefined) {
        ctx.fillStyle = background === 'black' ? '#AAAAAA' : '#666666';
        ctx.font = '10px monospace';
        ctx.fillText(
          `  residual: ${shift.residual_error.toFixed(2)}px, matches: ${shift.num_matches || 0}`,
          padding + 10,
          yOffset
        );
        yOffset += 16;
        ctx.font = '12px monospace';
      }
    });
    
    ctx.restore();
  };

  const drawScaleReference = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    ctx.save();
    
    // Find maximum shift magnitude
    const maxMagnitude = Math.max(
      ...Object.values(shiftVectors).map(v => v.magnitude),
      0.1
    );
    
    if (maxMagnitude < 0.01) return; // No shifts to show
    
    // Calculate scale same way as in drawShiftVectors
    const pixelsPerShiftUnit = (arrowBaseLength / maxMagnitude) * arrowScale;
    const minScale = 20.0;
    const scale = Math.max(pixelsPerShiftUnit, minScale);
    
    const referenceLength = 5.0; // 5 pixels of shift
    const arrowLength = referenceLength * scale;
    
    // Draw reference arrow in bottom-right corner
    const refX = width - 200;
    const refY = height - 80;
    
    ctx.strokeStyle = background === 'black' ? '#FFFFFF' : '#000000';
    ctx.fillStyle = background === 'black' ? '#FFFFFF' : '#000000';
    ctx.lineWidth = 3;
    
    // Draw reference arrow (larger for visibility)
    drawArrow(ctx, refX, refY, refX + arrowLength, refY, ctx.strokeStyle as string, 3, 15);
    
    // Draw reference text with background for visibility
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    
    // Draw text background
    const text = `Scale: ${referenceLength} px shift = ${arrowLength.toFixed(0)} px arrow`;
    const textMetrics = ctx.measureText(text);
    ctx.fillStyle = background === 'black' ? 'rgba(0, 0, 0, 0.7)' : 'rgba(255, 255, 255, 0.9)';
    ctx.fillRect(refX - 5, refY + 25, textMetrics.width + 10, 20);
    
    // Draw text
    ctx.fillStyle = background === 'black' ? '#FFFFFF' : '#000000';
    ctx.fillText(text, refX, refY + 27);
    
    // Also show max shift magnitude
    ctx.font = '11px sans-serif';
    const maxText = `Max shift: ${maxMagnitude.toFixed(2)} px`;
    const maxTextMetrics = ctx.measureText(maxText);
    ctx.fillStyle = background === 'black' ? 'rgba(0, 0, 0, 0.7)' : 'rgba(255, 255, 255, 0.9)';
    ctx.fillRect(refX - 5, refY + 47, maxTextMetrics.width + 10, 18);
    ctx.fillStyle = background === 'black' ? '#FFFFFF' : '#000000';
    ctx.fillText(maxText, refX, refY + 49);
    
    ctx.restore();
  };

  // ── Crop overlay interaction ─────────────────────────────────────────────────

  const getDisplayScale = (): { scaleX: number; scaleY: number } => {
    const canvas = canvasRef.current;
    if (!canvas || !canvas.offsetWidth) return { scaleX: 1, scaleY: 1 };
    return {
      scaleX: canvas.width / canvas.offsetWidth,
      scaleY: canvas.height / (canvas.offsetHeight || 1),
    };
  };

  const handleCropMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cropInteractive || stage !== 'aligned') return;
    const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    cropDragRef.current = { dragging: true, startX: x, startY: y, currentX: x, currentY: y };
    setCropDragDisplay({ x, y, w: 0, h: 0 });
    e.preventDefault();
  };

  const handleCropMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cropDragRef.current?.dragging) return;
    const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
    const curX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const curY = Math.max(0, Math.min(e.clientY - rect.top, rect.height));
    cropDragRef.current.currentX = curX;
    cropDragRef.current.currentY = curY;
    const { startX, startY } = cropDragRef.current;
    setCropDragDisplay({
      x: Math.min(startX, curX),
      y: Math.min(startY, curY),
      w: Math.abs(curX - startX),
      h: Math.abs(curY - startY),
    });
  };

  const handleCropMouseUp = () => {
    if (!cropDragRef.current?.dragging) return;
    const drag = cropDragRef.current;
    drag.dragging = false;

    const { scaleX, scaleY } = getDisplayScale();
    const x = Math.min(drag.startX, drag.currentX);
    const y = Math.min(drag.startY, drag.currentY);
    const w = Math.abs(drag.currentX - drag.startX);
    const h = Math.abs(drag.currentY - drag.startY);

    if (w > 5 && h > 5 && onCropRectChange) {
      onCropRectChange({
        x: Math.round(x * scaleX),
        y: Math.round(y * scaleY),
        w: Math.round(w * scaleX),
        h: Math.round(h * scaleY),
      });
    }
    setCropDragDisplay(null);
    cropDragRef.current = null;
  };

  // Compute the crop rect in display coords for SVG rendering.
  // This runs during render; canvasSize state ensures re-render when canvas reloads.
  const cropRectDisplay = (() => {
    if (!cropRect || !canvasSize.width) return null;
    const canvas = canvasRef.current;
    if (!canvas || !canvas.offsetWidth) return null;
    const scaleX = canvas.offsetWidth / canvas.width;
    const scaleY = (canvas.offsetHeight || canvas.offsetWidth) / canvas.height;
    return {
      x: cropRect.x * scaleX,
      y: cropRect.y * scaleY,
      w: cropRect.w * scaleX,
      h: cropRect.h * scaleY,
    };
  })();

  const toggleChannel = (ch: string) => {
    setChannels((prev) => ({
      ...prev,
      [ch]: { ...prev[ch], enabled: !prev[ch].enabled },
    }));
  };

  const handleSelectAllChannels = () => {
    setChannels((prev) => {
      const updated: Record<string, ChannelConfig> = {};
      Object.keys(prev).forEach((ch) => {
        updated[ch] = { ...prev[ch], enabled: true };
      });
      return updated;
    });
  };

  const handleDeselectAllChannels = () => {
    setChannels((prev) => {
      const updated: Record<string, ChannelConfig> = {};
      Object.keys(prev).forEach((ch) => {
        updated[ch] = { ...prev[ch], enabled: false };
      });
      return updated;
    });
  };

  // Get available stages
  const availableStages = Object.keys(stages).filter(
    (key) => stages[key as keyof typeof stages] && Object.keys(stages[key as keyof typeof stages]!).length > 0
  );

  const stageLabels: Record<string, string> = {
    raw: 'Raw',
    contrast_enhance: 'Contrast Enhanced',
    step1: 'Bkg Subtracted',
    step2: 'Clipped',
    step3: 'Gaussian Blurred',
    step4: 'Connected',
    aligned: 'Aligned',
  };

  if (!isLoaded) {
    return (
      <div className="alignment-overlay-placeholder">
        <p>Load a position to enable alignment QC overlay</p>
      </div>
    );
  }

  return (
    <div className="alignment-color-overlay">
      <div className="overlay-header">
        <h3 style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          🎨 Alignment QC: Color Overlay
          {cropInteractive && stage === 'aligned' && (
            <span style={{ fontSize: '0.75rem', background: '#e3f2fd', color: '#1565c0',
              padding: '2px 8px', borderRadius: 10, fontWeight: 'normal' }}>
              ✂️ Crop mode — drag to define region
            </span>
          )}
        </h3>
        <div className="overlay-info">
          <span>Multi-channel pseudo-color visualization for alignment verification</span>
        </div>
      </div>

      <div className="overlay-controls">
        <div className="control-row">
          <div className="control-group">
            <label>Stage:</label>
            <select
              value={stage}
              onChange={(e) => setStage(e.target.value as keyof typeof stages)}
              disabled={availableStages.length === 0}
            >
              {availableStages.map((stageKey) => (
                <option key={stageKey} value={stageKey}>
                  {stageLabels[stageKey] || stageKey}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label>Global Alpha: {globalAlpha.toFixed(2)}</label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={globalAlpha}
              onChange={(e) => setGlobalAlpha(parseFloat(e.target.value))}
            />
          </div>

          <div className="control-group">
            <label>Background:</label>
            <div className="button-group-inline">
              <button
                className={background === 'black' ? 'active' : ''}
                onClick={() => setBackground('black')}
              >
                Black
              </button>
              <button
                className={background === 'white' ? 'active' : ''}
                onClick={() => setBackground('white')}
              >
                White
              </button>
            </div>
          </div>
        </div>

        {/* NEW: Shift vector visualization controls */}
        {stage === 'aligned' && (
          <div className="control-row" style={{ marginTop: '12px', padding: '8px', background: '#f0f0f0', borderRadius: '4px' }}>
            <div className="control-group">
              <label>
                <input
                  type="checkbox"
                  checked={showShiftVectors}
                  onChange={(e) => setShowShiftVectors(e.target.checked)}
                />
                <span style={{ marginLeft: '4px' }}>Show Shift Vectors</span>
              </label>
            </div>
            <div className="control-group">
              <label>
                <input
                  type="checkbox"
                  checked={showBeforeAfter}
                  onChange={(e) => setShowBeforeAfter(e.target.checked)}
                />
                <span style={{ marginLeft: '4px' }}>Show Before/After (Ghost)</span>
              </label>
            </div>
            <div className="control-group">
              <label>
                <input
                  type="checkbox"
                  checked={showAnnotations}
                  onChange={(e) => setShowAnnotations(e.target.checked)}
                />
                <span style={{ marginLeft: '4px' }}>Show Annotations</span>
              </label>
            </div>
            {showShiftVectors && (
              <>
                <div className="control-group">
                  <label>Arrow Scale: {arrowScale.toFixed(1)}x</label>
                  <input
                    type="range"
                    min="1.0"
                    max="20.0"
                    step="0.5"
                    value={arrowScale}
                    onChange={(e) => setArrowScale(parseFloat(e.target.value))}
                    style={{ width: '120px' }}
                  />
                </div>
                <div className="control-group">
                  <label>Base Length: {arrowBaseLength}px</label>
                  <input
                    type="range"
                    min="20"
                    max="200"
                    step="10"
                    value={arrowBaseLength}
                    onChange={(e) => setArrowBaseLength(parseInt(e.target.value))}
                    style={{ width: '120px' }}
                  />
                </div>
              </>
            )}
          </div>
        )}

        <div className="control-row">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <label>Channels:</label>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                className="sidebar-btn"
                onClick={handleSelectAllChannels}
                style={{ padding: '4px 12px', fontSize: '0.85rem' }}
              >
                Select All
              </button>
              <button
                className="sidebar-btn"
                onClick={handleDeselectAllChannels}
                style={{ padding: '4px 12px', fontSize: '0.85rem' }}
              >
                Deselect All
              </button>
            </div>
          </div>
          <div className="channel-toggles">
            {Object.keys(channels).map((ch) => {
              const item = availableItems.find(i => i.key === ch);
              const displayLabel = item ? item.display_label : ch;
              const shift = shiftVectors[ch];
              const isIdentity = shift?.type === 'identity';
              const isReference = shift?.type === 'reference';
              
              return (
                <label 
                  key={ch} 
                  className="channel-toggle"
                  onMouseEnter={() => setHoveredChannel(ch)}
                  onMouseLeave={() => setHoveredChannel(null)}
                  style={{
                    opacity: isIdentity ? 0.6 : 1.0,
                    border: isIdentity ? '2px dashed #FF8800' : 'none',
                    padding: isIdentity ? '2px' : '0',
                    borderRadius: isIdentity ? '4px' : '0',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={channels[ch].enabled}
                    onChange={() => toggleChannel(ch)}
                  />
                  <span
                    className="channel-color-indicator"
                    style={{
                      backgroundColor: `rgb(${channels[ch].color[0] * 255}, ${
                        channels[ch].color[1] * 255
                      }, ${channels[ch].color[2] * 255})`,
                    }}
                  />
                  <span>
                    {displayLabel} ({channels[ch].name})
                    {isReference && ' [REF]'}
                    {isIdentity && ' ⚠ IDENTITY'}
                    {shift && !isIdentity && !isReference && showAnnotations && (
                      <span style={{ fontSize: '0.85em', color: '#666' }}>
                        {' '}(Δx={shift.dx.toFixed(1)}, Δy={shift.dy.toFixed(1)})
                      </span>
                    )}
                  </span>
                </label>
              );
            })}
          </div>
        </div>
      </div>

      {/* Canvas Container */}
      <div className="overlay-canvas-container" style={{ marginTop: '12px' }} onContextMenu={(e) => e.preventDefault()}>
        {isRendering && (
          <div className="rendering-indicator">Rendering overlay...</div>
        )}
        {/* Canvas + crop overlay wrapper */}
        <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%' }}>
          <canvas
            ref={canvasRef}
            className="overlay-canvas"
            onContextMenu={(e) => e.preventDefault()}
            style={{
              maxWidth: '100%',
              height: 'auto',
              border: '2px solid #ddd',
              display: 'block',
            }}
          />

          {/* Crop interaction layer — absolutely covers the canvas */}
          <div
            ref={cropOverlayRef}
            style={{
              position: 'absolute',
              inset: 0,
              cursor: cropInteractive && stage === 'aligned' ? 'crosshair' : 'default',
              zIndex: 5,
              pointerEvents: cropInteractive && stage === 'aligned' ? 'auto' : 'none',
            }}
            onMouseDown={handleCropMouseDown}
            onMouseMove={handleCropMouseMove}
            onMouseUp={handleCropMouseUp}
            onMouseLeave={handleCropMouseUp}
          >
            <svg
              width="100%"
              height="100%"
              style={{ position: 'absolute', inset: 0, overflow: 'visible' }}
            >
              {/* Active drag rectangle */}
              {cropDragDisplay && cropDragDisplay.w > 2 && cropDragDisplay.h > 2 && (
                <>
                  <defs>
                    <mask id="crop-drag-mask">
                      <rect width="100%" height="100%" fill="white" />
                      <rect
                        x={cropDragDisplay.x} y={cropDragDisplay.y}
                        width={cropDragDisplay.w} height={cropDragDisplay.h}
                        fill="black"
                      />
                    </mask>
                  </defs>
                  <rect
                    width="100%" height="100%"
                    fill="rgba(0,0,0,0.45)"
                    mask="url(#crop-drag-mask)"
                  />
                  <rect
                    x={cropDragDisplay.x} y={cropDragDisplay.y}
                    width={cropDragDisplay.w} height={cropDragDisplay.h}
                    fill="none"
                    stroke="white"
                    strokeWidth={1.5}
                    strokeDasharray="6 4"
                  />
                </>
              )}

              {/* Committed crop rect (from parent state) */}
              {!cropDragDisplay && cropRectDisplay && cropRectDisplay.w > 0 && cropRectDisplay.h > 0 && (
                <>
                  <defs>
                    <mask id="crop-committed-mask">
                      <rect width="100%" height="100%" fill="white" />
                      <rect
                        x={cropRectDisplay.x} y={cropRectDisplay.y}
                        width={cropRectDisplay.w} height={cropRectDisplay.h}
                        fill="black"
                      />
                    </mask>
                  </defs>
                  <rect
                    width="100%" height="100%"
                    fill="rgba(0,0,0,0.35)"
                    mask="url(#crop-committed-mask)"
                  />
                  <rect
                    x={cropRectDisplay.x} y={cropRectDisplay.y}
                    width={cropRectDisplay.w} height={cropRectDisplay.h}
                    fill="none"
                    stroke="white"
                    strokeWidth={2}
                    strokeDasharray="8 5"
                  />
                  {/* Corner handles */}
                  {[
                    [cropRectDisplay.x, cropRectDisplay.y],
                    [cropRectDisplay.x + cropRectDisplay.w, cropRectDisplay.y],
                    [cropRectDisplay.x, cropRectDisplay.y + cropRectDisplay.h],
                    [cropRectDisplay.x + cropRectDisplay.w, cropRectDisplay.y + cropRectDisplay.h],
                  ].map(([cx, cy], i) => (
                    <circle key={i} cx={cx} cy={cy} r={5} fill="white" stroke="#333" strokeWidth={1} />
                  ))}
                </>
              )}
            </svg>
          </div>
        </div>
        <div className="canvas-dimensions">
          {canvasSize.width} × {canvasSize.height} px
        </div>
      </div>

      <div className="overlay-help">
        <p>
          <strong>💡 Tip:</strong> Misaligned channels will show colored "shadow" or "double"
          edges. Well-aligned channels will have crisp, coincident boundaries.
        </p>
        {stage === 'aligned' && showShiftVectors && (
          <p style={{ marginTop: '8px', fontSize: '0.9em' }}>
            <strong>📊 Shift Vectors:</strong> Arrows show the direction and magnitude of channel movement during alignment.
            Reference channel has no arrow. Identity transforms (skipped channels) are marked with ⚠.
            {Object.keys(shiftVectors).length > 0 && (
              <span style={{ display: 'block', marginTop: '4px', color: '#666' }}>
                <strong>Note:</strong> Very small shifts (&lt; 0.1 px) may not be visible as arrows. Check the shift values panel above for exact measurements.
              </span>
            )}
          </p>
        )}
      </div>
    </div>
  );
};

export default AlignmentColorOverlay;
