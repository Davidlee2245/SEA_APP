/**
 * ManualDiagonalEditor
 *
 * Interactive canvas overlay for drawing and adjusting a rotated bounding box
 * via a two-point diagonal. Used by the "Manual Diagonal" alignment method.
 *
 * Interaction modes:
 *   Draw  — click P1, click P2 → box auto-fits (square, rotated to diagonal)
 *   Edit  — drag body (translate), drag green circle (rotate),
 *            drag corner squares (resize), arrow keys (nudge), Esc (clear)
 *
 * All box coordinates are stored in IMAGE natural-pixel space.
 * The canvas renders in display-pixel space (accounts for CSS scaling).
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';

// ── Public types ──────────────────────────────────────────────────────────────

export interface DiagonalBox {
  cx: number;     // center x, image natural pixels
  cy: number;     // center y, image natural pixels
  width: number;  // box width,  image natural pixels
  height: number; // box height, image natural pixels
  angle: number;  // rotation in radians (CCW positive)
}

interface Props {
  imageUrl: string;
  box: DiagonalBox | null;
  onBoxChange: (box: DiagonalBox | null) => void;
  channelLabel?: string;
}

// ── Geometry helpers ──────────────────────────────────────────────────────────

function boxCorners(b: DiagonalBox): [number, number][] {
  const cos = Math.cos(b.angle), sin = Math.sin(b.angle);
  const hw = b.width / 2, hh = b.height / 2;
  return (
    [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]] as [number, number][]
  ).map(([lx, ly]) => [b.cx + lx * cos - ly * sin, b.cy + lx * sin + ly * cos]);
}

/** Position of the rotation handle: 32px above the top-center of the box */
function rotHandlePos(b: DiagonalBox): [number, number] {
  const OFFSET = 32;
  const cos = Math.cos(b.angle), sin = Math.sin(b.angle);
  return [
    b.cx + 0 * cos - (-b.height / 2 - OFFSET) * sin,
    b.cy + 0 * sin + (-b.height / 2 - OFFSET) * cos,
  ];
}

function pointInBox(px: number, py: number, b: DiagonalBox): boolean {
  const cos = Math.cos(-b.angle), sin = Math.sin(-b.angle);
  const dx = px - b.cx, dy = py - b.cy;
  const lx = dx * cos - dy * sin, ly = dx * sin + dy * cos;
  return Math.abs(lx) <= b.width / 2 + 4 && Math.abs(ly) <= b.height / 2 + 4;
}

function dst(ax: number, ay: number, bx: number, by: number) {
  return Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2);
}

// ── Drag state ────────────────────────────────────────────────────────────────

type Drag =
  | { k: 'none' }
  | { k: 'translate'; six: number; siy: number; ocx: number; ocy: number }
  | { k: 'rotate';    sa: number; oba: number }           // start-angle, orig-box-angle
  | { k: 'resize';    opp: [number, number] };             // opposite corner, image coords

// ── Constants ─────────────────────────────────────────────────────────────────

const CORNER_R = 5;
const ROT_R    = 8;

// ── Component ─────────────────────────────────────────────────────────────────

const ManualDiagonalEditor: React.FC<Props> = ({ imageUrl, box, onBoxChange, channelLabel }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef       = useRef<HTMLImageElement>(null);
  const canvasRef    = useRef<HTMLCanvasElement>(null);

  const [naturalSize, setNaturalSize] = useState<[number, number]>([1, 1]);
  const [displaySize, setDisplaySize] = useState<[number, number]>([1, 1]);
  const [drawP1,      setDrawP1]      = useState<[number, number] | null>(null); // image coords
  const [hoverPos,    setHoverPos]    = useState<[number, number] | null>(null); // display coords

  const dragRef = useRef<Drag>({ k: 'none' });
  const boxRef  = useRef(box);
  boxRef.current = box;

  // Uniform scale factor: natural → display
  const scale = useCallback(
    () => (displaySize[0] > 0 && naturalSize[0] > 0 ? naturalSize[0] / displaySize[0] : 1),
    [naturalSize, displaySize]
  );

  const toImg  = useCallback((dx: number, dy: number): [number, number] => {
    const s = scale(); return [dx * s, dy * s];
  }, [scale]);

  const toDisp = useCallback((ix: number, iy: number): [number, number] => {
    const s = scale(); return [ix / s, iy / s];
  }, [scale]);

  // ── Measure image display size ────────────────────────────────────────────

  const measure = useCallback(() => {
    const img = imgRef.current;
    if (!img || img.offsetWidth === 0) return;
    setDisplaySize([img.offsetWidth, img.offsetHeight]);
    if (img.naturalWidth > 0) setNaturalSize([img.naturalWidth, img.naturalHeight]);
  }, []);

  useEffect(() => {
    const ro = new ResizeObserver(measure);
    if (imgRef.current) ro.observe(imgRef.current);
    return () => ro.disconnect();
  }, [measure]);

  // ── Canvas rendering ──────────────────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const [dw, dh] = displaySize;
    canvas.width  = dw;
    canvas.height = dh;
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, dw, dh);

    // Draw preview line while placing P2
    if (drawP1 && hoverPos) {
      const [p1dx, p1dy] = toDisp(drawP1[0], drawP1[1]);
      ctx.save();
      ctx.strokeStyle = 'cyan'; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(p1dx, p1dy); ctx.lineTo(hoverPos[0], hoverPos[1]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'cyan';
      ctx.beginPath(); ctx.arc(p1dx, p1dy, 5, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }

    if (!box) return;

    // Convert box to display-pixel space
    const s = scale();
    const db: DiagonalBox = { cx: box.cx / s, cy: box.cy / s, width: box.width / s, height: box.height / s, angle: box.angle };
    const corners  = boxCorners(db);
    const rh       = rotHandlePos(db);
    const topMid: [number, number] = [
      (corners[0][0] + corners[1][0]) / 2,
      (corners[0][1] + corners[1][1]) / 2,
    ];

    ctx.save();

    // Box outline — yellow
    ctx.strokeStyle = 'yellow'; ctx.lineWidth = 2; ctx.setLineDash([]);
    ctx.beginPath();
    corners.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.closePath();
    ctx.stroke();

    // Diagonal TL→BR — cyan dashed
    ctx.strokeStyle = 'cyan'; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    ctx.lineTo(corners[2][0], corners[2][1]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Rotation handle stem + circle — green
    ctx.strokeStyle = 'rgba(160,255,160,0.85)'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(topMid[0], topMid[1]); ctx.lineTo(rh[0], rh[1]); ctx.stroke();
    ctx.fillStyle = '#66ee66'; ctx.strokeStyle = '#336633'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(rh[0], rh[1], ROT_R, 0, Math.PI * 2); ctx.fill(); ctx.stroke();

    // Corner handles — white squares
    corners.forEach(([x, y]) => {
      ctx.fillStyle = 'white'; ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
      ctx.fillRect(x - CORNER_R, y - CORNER_R, CORNER_R * 2, CORNER_R * 2);
      ctx.strokeRect(x - CORNER_R, y - CORNER_R, CORNER_R * 2, CORNER_R * 2);
    });

    // Info overlay
    const deg = (box.angle * 180 / Math.PI).toFixed(1);
    const info = `${Math.round(box.cx)},${Math.round(box.cy)}  ${Math.round(box.width)}×${Math.round(box.height)}px  ${deg}°`;
    ctx.font = '12px monospace';
    const tw = ctx.measureText(info).width;
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(4, 4, tw + 10, 20);
    ctx.fillStyle = '#fff';
    ctx.fillText(info, 8, 18);

    ctx.restore();
  }, [box, drawP1, hoverPos, displaySize, naturalSize, scale, toDisp]);

  // ── Hit testing (display coords) ──────────────────────────────────────────

  const hitTest = useCallback((mx: number, my: number): string => {
    if (!box) return 'none';
    const s = scale();
    const db: DiagonalBox = { cx: box.cx / s, cy: box.cy / s, width: box.width / s, height: box.height / s, angle: box.angle };
    const rh = rotHandlePos(db);
    if (dst(mx, my, rh[0], rh[1]) <= ROT_R + 4) return 'rotate';
    const corners = boxCorners(db);
    for (let i = 0; i < 4; i++) {
      if (dst(mx, my, corners[i][0], corners[i][1]) <= CORNER_R + 4) return `corner_${i}`;
    }
    if (pointInBox(mx, my, db)) return 'translate';
    return 'outside';
  }, [box, scale]);

  const cursorFor = (hit: string) =>
    hit === 'rotate'             ? 'grab'       :
    hit === 'translate'          ? 'move'        :
    hit.startsWith('corner_')   ? 'nwse-resize' :
    'crosshair';

  // ── Mouse event helpers ───────────────────────────────────────────────────

  const getMxy = (e: React.MouseEvent): [number, number] => {
    const r = containerRef.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };

  // ── Mouse handlers ────────────────────────────────────────────────────────

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    containerRef.current?.focus();
    const [mx, my] = getMxy(e);
    const [ix, iy] = toImg(mx, my);

    if (!box) {
      // Draw mode
      if (!drawP1) {
        setDrawP1([ix, iy]);
      } else {
        const p1 = drawP1;
        const d = dst(ix, iy, p1[0], p1[1]);
        if (d < 10) { setDrawP1(null); return; }
        const diagAngle = Math.atan2(iy - p1[1], ix - p1[0]);
        const side = d / Math.sqrt(2);
        onBoxChange({
          cx: (ix + p1[0]) / 2, cy: (iy + p1[1]) / 2,
          width: side, height: side,
          angle: diagAngle - Math.PI / 4,
        });
        setDrawP1(null);
      }
      return;
    }

    const hit = hitTest(mx, my);

    if (hit === 'rotate') {
      dragRef.current = { k: 'rotate', sa: Math.atan2(iy - box.cy, ix - box.cx), oba: box.angle };

    } else if (hit === 'translate') {
      dragRef.current = { k: 'translate', six: ix, siy: iy, ocx: box.cx, ocy: box.cy };

    } else if (hit.startsWith('corner_')) {
      const ci = parseInt(hit.split('_')[1], 10);
      const s = scale();
      const db: DiagonalBox = { cx: box.cx / s, cy: box.cy / s, width: box.width / s, height: box.height / s, angle: box.angle };
      const oppDisp = boxCorners(db)[(ci + 2) % 4];
      dragRef.current = { k: 'resize', opp: toImg(oppDisp[0], oppDisp[1]) };

    } else {
      // Click outside box → reset and start new draw
      onBoxChange(null);
      setDrawP1([ix, iy]);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const [mx, my] = getMxy(e);
    setHoverPos([mx, my]);
    const [ix, iy] = toImg(mx, my);
    const drag = dragRef.current;

    if (drag.k === 'translate') {
      onBoxChange({ ...boxRef.current!, cx: drag.ocx + ix - drag.six, cy: drag.ocy + iy - drag.siy });

    } else if (drag.k === 'rotate') {
      const current = Math.atan2(iy - boxRef.current!.cy, ix - boxRef.current!.cx);
      onBoxChange({ ...boxRef.current!, angle: drag.oba + (current - drag.sa) });

    } else if (drag.k === 'resize') {
      const [ox, oy] = drag.opp;
      const d = dst(ix, iy, ox, oy);
      if (d < 5) return;
      // Keep angle, recompute width/height from how the dragged corner lands on the box axes
      const b = boxRef.current!;
      const ncx = (ix + ox) / 2, ncy = (iy + oy) / 2;
      const cos = Math.cos(-b.angle), sin = Math.sin(-b.angle);
      const hdx = ix - ncx, hdy = iy - ncy;
      const hlx = Math.abs(hdx * cos - hdy * sin);
      const hly = Math.abs(hdx * sin + hdy * cos);
      onBoxChange({ ...b, cx: ncx, cy: ncy, width: Math.max(5, hlx * 2), height: Math.max(5, hly * 2) });
    }
  };

  const handleMouseUp = () => { dragRef.current = { k: 'none' }; };

  // ── Keyboard shortcuts ────────────────────────────────────────────────────

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!box) return;
    const step = e.shiftKey ? 10 : 1;
    if      (e.key === 'ArrowLeft')  { e.preventDefault(); onBoxChange({ ...box, cx: box.cx - step }); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); onBoxChange({ ...box, cx: box.cx + step }); }
    else if (e.key === 'ArrowUp')    { e.preventDefault(); onBoxChange({ ...box, cy: box.cy - step }); }
    else if (e.key === 'ArrowDown')  { e.preventDefault(); onBoxChange({ ...box, cy: box.cy + step }); }
    else if (e.key === 'r' || e.key === 'R') {
      e.preventDefault();
      const dir = e.key === 'R' ? -1 : 1;
      onBoxChange({ ...box, angle: box.angle + dir * 5 * Math.PI / 180 });
    }
    else if (e.key === 'Escape') { e.preventDefault(); onBoxChange(null); }
  };

  // ── Cursor ────────────────────────────────────────────────────────────────

  const [hmx, hmy] = hoverPos ?? [-1, -1];
  const hit = hoverPos && box ? hitTest(hmx, hmy) : 'none';
  const cursor = drawP1 || !box ? 'crosshair' : cursorFor(hit);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => { dragRef.current = { k: 'none' }; setHoverPos(null); }}
      onKeyDown={handleKeyDown}
      style={{ position: 'relative', display: 'inline-block', maxWidth: '100%', outline: 'none', cursor }}
    >
      <img
        ref={imgRef}
        src={imageUrl}
        alt={channelLabel || 'channel'}
        onLoad={measure}
        draggable={false}
        style={{ display: 'block', maxWidth: '100%', height: 'auto', userSelect: 'none' }}
      />
      <canvas
        ref={canvasRef}
        style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
      />
      <div style={{
        position: 'absolute', bottom: 6, left: 0, right: 0, textAlign: 'center',
        color: 'rgba(255,255,200,0.9)', fontSize: '0.78rem', pointerEvents: 'none',
        textShadow: '0 1px 3px rgba(0,0,0,1)', lineHeight: 1.4,
      }}>
        {!box && !drawP1 && 'Click to set first diagonal point'}
        {!box &&  drawP1 && 'Click to set second diagonal point'}
        {box && 'Drag body: move · Corners: resize · Green circle: rotate · Arrows: nudge · R: rotate 5° · Esc: clear'}
      </div>
    </div>
  );
};

export default ManualDiagonalEditor;
