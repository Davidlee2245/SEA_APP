#!/usr/bin/env python3
"""
Standalone debug script: overlay ground-truth CSV coordinates on a raw TIFF.

No GUI, frontend, segmentation, or pipeline dependencies beyond numpy/pandas/
tifffile/matplotlib/opencv.

Usage:
  conda activate your-sea-env
  python debug_gt_coordinate_mapping.py [--tiff PATH] [--csv PATH] [--out DIR]

Defaults assume files live alongside this script under test/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _as_2d_grayscale(arr: np.ndarray) -> np.ndarray:
    """Reduce stacks / channels to a single 2D plane for overlay."""
    a = np.asarray(arr)
    a = np.squeeze(a)
    while a.ndim > 2:
        a = a[0]
    return a


def load_tiff(path: Path) -> np.ndarray:
    import tifffile

    raw = tifffile.imread(path)
    return _as_2d_grayscale(raw)


def percentile_stretch_gray(
    gray: np.ndarray, p_low: float = 0.5, p_high: float = 99.5
) -> np.ndarray:
    """Display-only uint8 grayscale via percentile contrast stretching."""
    g = np.asarray(gray, dtype=np.float64)
    lo, hi = np.percentile(g, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-12
    g = np.clip((g - lo) / (hi - lo), 0.0, 1.0)
    return (g * 255.0).astype(np.uint8)


def print_tiff_stats(gray_raw: np.ndarray) -> None:
    g = gray_raw.astype(np.float64)
    print("\n=== TIFF statistics ===")
    print(f"  shape (H, W): {gray_raw.shape}")
    print(f"  dtype (raw): {gray_raw.dtype}")
    print(f"  min / max: {g.min()} / {g.max()}")
    for p in (0.5, 1, 5, 50, 95, 99, 99.5):
        print(f"  percentile {p}%: {np.percentile(g, p)}")
    print(f"  (display uses separate 0.5–99.5 stretch; mapping uses raw geometry only)")


PIXEL_SIZE_UM = 0.21


def micrometers_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Resolve coordinate column names."""
    cols = list(df.columns)
    x_candidates = [c for c in cols if "Center of Mass X" in c]
    y_candidates = [c for c in cols if "Center of Mass Y" in c]
    if not x_candidates or not y_candidates:
        raise ValueError(
            "Could not find 'Center of Mass X' / 'Center of Mass Y' columns.\n"
            f"Got columns: {cols[:20]}..."
        )
    return x_candidates[0], y_candidates[0]


def load_gt_csv(csv_path: Path) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(csv_path)
    x_col, y_col = micrometers_columns(df)
    df = df.copy()
    df["_row_idx"] = np.arange(len(df), dtype=np.int64)
    return df, x_col, y_col


def compute_mappings(
    x_um: np.ndarray,
    y_um: np.ndarray,
    h: int,
    w: int,
    pixel_um: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    um_to_px = 1.0 / pixel_um
    x_px_u = x_um * um_to_px
    y_px_u = y_um * um_to_px

    mappings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    mappings["standard"] = (x_px_u.copy(), y_px_u.copy())
    mappings["xy_swapped"] = (y_px_u.copy(), x_px_u.copy())
    mappings["y_flipped"] = (x_px_u.copy(), h - y_px_u)
    mappings["xy_swapped_y_flipped"] = (y_px_u.copy(), h - x_px_u)
    mappings["no_unit_conversion"] = (x_um.astype(np.float64), y_um.astype(np.float64))

    # Clip for safety when drawing only (arrays stay full float)
    return mappings


def in_bounds(xx: np.ndarray, yy: np.ndarray, h: int, w: int) -> np.ndarray:
    return (
        (xx >= 0)
        & (xx < w)
        & (yy >= 0)
        & (yy < h)
    )


def overlay_matplotlib_save(
    display_u8: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    inside: np.ndarray,
    indices: np.ndarray,
    png_path: Path,
    title: str,
    label_every: int = 50,
    point_radius_px: float = 5,
) -> None:
    """Draw circles + sparse index labels on stretched grayscale."""
    hh, ww = display_u8.shape[:2]
    fig, ax = plt.subplots(figsize=(min(16, ww / 80), min(16, hh / 80)))
    ax.imshow(display_u8, cmap="gray", vmin=0, vmax=255, origin="upper", aspect="equal")
    # Valid points cyan; out-of-bounds red (still attempt draw clipped by ax limit)
    for i in range(len(xs)):
        cx, cy = float(xs[i]), float(ys[i])
        color = "cyan" if inside[i] else "red"
        circ = plt.Circle(
            (cx, cy),
            radius=point_radius_px,
            fill=False,
            edgecolor=color,
            linewidth=1.2,
        )
        ax.add_patch(circ)
        ax.plot([cx - 8, cx + 8], [cy, cy], color=color, linewidth=0.8)
        ax.plot([cx, cx], [cy - 8, cy + 8], color=color, linewidth=0.8)

    if label_every > 0:
        for i in range(len(xs)):
            if i % label_every != 0:
                continue
            cx, cy = float(xs[i]), float(ys[i])
            ax.text(
                cx + point_radius_px + 1,
                cy,
                str(int(indices[i])),
                color="yellow",
                fontsize=6,
                ha="left",
                va="center",
                clip_on=True,
            )

    ax.set_xlim(0, ww)
    ax.set_ylim(hh, 0)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def crop_region(center_x: float, center_y: float, half_size: int, h: int, w: int) -> tuple[int, int, int, int]:
    cx = int(round(center_x))
    cy = int(round(center_y))
    x0 = max(0, cx - half_size)
    x1 = min(w, cx + half_size)
    y0 = max(0, cy - half_size)
    y1 = min(h, cy + half_size)
    return y0, y1, x0, x1


def densest_cell_xy(
    xs: np.ndarray, ys: np.ndarray, inside: np.ndarray, h: int, w: int, grid_n: int = 10
) -> tuple[float, float]:
    gx = max(1, w // grid_n)
    gy = max(1, h // grid_n)
    best_i, best_j = 0, 0
    best_c = -1
    mask = inside
    xs_v = xs[mask]
    ys_v = ys[mask]
    for j in range(grid_n):
        y_lo, y_hi = j * gy, min(h, (j + 1) * gy)
        for i in range(grid_n):
            x_lo, x_hi = i * gx, min(w, (i + 1) * gx)
            c = np.sum((xs_v >= x_lo) & (xs_v < x_hi) & (ys_v >= y_lo) & (ys_v < y_hi))
            if c > best_c:
                best_c = c
                best_i, best_j = i, j
    cx = ((best_i + 0.5) * gx) if gx else w / 2
    cy = ((best_j + 0.5) * gy) if gy else h / 2
    return float(cx), float(cy)


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Ground-truth coordinate mapping debug overlays.")
    ap.add_argument(
        "--tiff",
        type=Path,
        default=here / "A2780Cis_P1_B1B_C0.tif",
        help="Path to TIFF (default: sibling of this script)",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=here / "Stack_Crop_HyperStack_A2780Cis10_P1_PanEV_PanEV.csv",
        help="Ground-truth CSV path",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=here / "gt_mapping_debug_outputs",
        help="Directory for PNG/CSV outputs",
    )
    ap.add_argument("--pixel-size-um", type=float, default=PIXEL_SIZE_UM, help="µm per pixel")
    ap.add_argument("--label-every", type=int, default=50, help="Label every N-th point index (0=off)")
    ap.add_argument("--zoom-half", type=int, default=220, help="Half-size in pixels for zoom crops")
    args = ap.parse_args()

    pixel_um = float(args.pixel_size_um)

    if not args.tiff.is_file():
        print(f"ERROR: TIFF not found: {args.tiff}", file=sys.stderr)
        return 1
    if not args.csv.is_file():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TIFF...")
    gray_raw = load_tiff(args.tiff)
    print_tiff_stats(gray_raw)

    display_u8 = percentile_stretch_gray(gray_raw, 0.5, 99.5)
    h, w = int(gray_raw.shape[0]), int(gray_raw.shape[1])

    print("\nLoading CSV...")
    df, x_col, y_col = load_gt_csv(args.csv)
    print(f"  Column names (first 15): {list(df.columns[:15])}")
    print(f"  Using X column: {x_col!r}")
    print(f"  Using Y column: {y_col!r}")

    x_um = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float64)
    y_um = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float64)
    valid_xy = np.isfinite(x_um) & np.isfinite(y_um)
    n_nan = int(np.sum(~valid_xy))
    if n_nan:
        print(f"  Warning: dropping {n_nan} rows with non-finite X/Y")
    df = df.loc[valid_xy].reset_index(drop=True)
    x_um = x_um[valid_xy]
    y_um = y_um[valid_xy]
    row_indices = df["_row_idx"].to_numpy(dtype=np.int64)

    n_gt = len(x_um)
    print(f"\n  Number of ground-truth points (finite): {n_gt}")
    print(f"  X_um min/max: {x_um.min()} / {x_um.max()}")
    print(f"  Y_um min/max: {y_um.min()} / {y_um.max()}")

    x_px_std = x_um / pixel_um
    y_px_std = y_um / pixel_um
    print(f"\n  After µm→px (÷{pixel_um}):")
    print(f"  x_px min/max: {x_px_std.min()} / {x_px_std.max()}")
    print(f"  y_px min/max: {y_px_std.min()} / {y_px_std.max()}")

    mappings = compute_mappings(x_um, y_um, h, w, pixel_um)

    names_order = [
        ("standard", "overlay_standard_um_to_px.png"),
        ("xy_swapped", "overlay_xy_swapped_um_to_px.png"),
        ("y_flipped", "overlay_y_flipped_um_to_px.png"),
        ("xy_swapped_y_flipped", "overlay_xy_swapped_y_flipped_um_to_px.png"),
        ("no_unit_conversion", "overlay_no_unit_conversion.png"),
    ]

    in_counts: dict[str, int] = {}
    mapping_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, _fn in names_order:
        xs, ys = mappings[key]
        mapping_arrays[key] = (xs, ys)
        ins = in_bounds(xs, ys, h, w)
        in_counts[key] = int(np.sum(ins))

    print("\n  In-bounds counts (before saving overlays):")
    for key, _fn in names_order:
        print(f"    {key}: {in_counts[key]} / {n_gt}")

    label_every = int(args.label_every)

    for key, fname in names_order:
        xs, ys = mapping_arrays[key]
        ins = in_bounds(xs, ys, h, w)
        outp = out_dir / fname
        title = f"{key} | in-bounds {in_counts[key]}/{n_gt}"
        overlay_matplotlib_save(
            display_u8,
            xs,
            ys,
            ins,
            row_indices,
            outp,
            title=title,
            label_every=label_every,
        )
        print(f"  Wrote {outp}")

    # Zoomed panels
    zm_half = int(args.zoom_half)
    regions_spec = [
        ("zoom_tl", w * 0.25, h * 0.25),
        ("zoom_center", w * 0.5, h * 0.5),
        ("zoom_br", w * 0.75, h * 0.75),
    ]
    sx, sy = mapping_arrays["standard"]
    swapped_x, swapped_y = mapping_arrays["xy_swapped"]
    ins_std = in_bounds(sx, sy, h, w)
    ins_sw = in_bounds(swapped_x, swapped_y, h, w)
    bx, by = densest_cell_xy(sx, sy, ins_std, h, w)
    regions_spec.append(("zoom_dense", bx, by))

    for tag, cxx, cyy in regions_spec:
        for variant_name, vx, vy, vins in [
            ("standard", sx, sy, ins_std),
            ("xy_swapped", swapped_x, swapped_y, ins_sw),
        ]:
            y0, y1, x0, x1 = crop_region(cxx, cyy, zm_half, h, w)
            crop_disp = display_u8[y0:y1, x0:x1]
            xs_sub = vx - x0
            ys_sub = vy - y0
            ins_sub = in_bounds(xs_sub, ys_sub, y1 - y0, x1 - x0)
            outp = out_dir / f"{tag}_{variant_name}.png"
            overlay_matplotlib_save(
                crop_disp,
                xs_sub,
                ys_sub,
                ins_sub,
                row_indices,
                outp,
                title=f"{tag} ({variant_name}) crop [{y0}:{y1},{x0}:{x1}]",
                label_every=min(label_every, 20),
            )
            print(f"  Wrote {outp}")

    # Coordinate debug CSV
    debug_cols = {
        "csv_row_idx": row_indices,
        "X_um": x_um,
        "Y_um": y_um,
        "standard_x_px": mapping_arrays["standard"][0],
        "standard_y_px": mapping_arrays["standard"][1],
        "swapped_x_px": mapping_arrays["xy_swapped"][0],
        "swapped_y_px": mapping_arrays["xy_swapped"][1],
        "y_flipped_x_px": mapping_arrays["y_flipped"][0],
        "y_flipped_y_px": mapping_arrays["y_flipped"][1],
        "xy_swapped_y_flipped_x_px": mapping_arrays["xy_swapped_y_flipped"][0],
        "xy_swapped_y_flipped_y_px": mapping_arrays["xy_swapped_y_flipped"][1],
        "no_conversion_x_px": mapping_arrays["no_unit_conversion"][0],
        "no_conversion_y_px": mapping_arrays["no_unit_conversion"][1],
        "in_bounds_standard": in_bounds(*mapping_arrays["standard"], h, w),
        "in_bounds_swapped": in_bounds(*mapping_arrays["xy_swapped"], h, w),
        "in_bounds_y_flipped": in_bounds(*mapping_arrays["y_flipped"], h, w),
        "in_bounds_xy_swapped_y_flipped": in_bounds(*mapping_arrays["xy_swapped_y_flipped"], h, w),
        "in_bounds_no_conversion": in_bounds(*mapping_arrays["no_unit_conversion"], h, w),
    }
    debug_df = pd.DataFrame(debug_cols)
    debug_csv = out_dir / "coordinate_debug.csv"
    debug_df.to_csv(debug_csv, index=False)
    print(f"\n  Wrote {debug_csv}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"TIFF dimensions (width x height): {w} x {h}")
    print(f"Ground-truth points (finite X/Y): {n_gt}")
    print(f"\nCoordinate ranges (µm): X [{x_um.min():.4f}, {x_um.max():.4f}], Y [{y_um.min():.4f}, {y_um.max():.4f}]")
    print(
        f"After µm→px (÷{pixel_um}): x_px [{x_px_std.min():.4f}, {x_px_std.max():.4f}], "
        f"y_px [{y_px_std.min():.4f}, {y_px_std.max():.4f}]"
    )
    print("\nIn-bounds counts per mapping variant:")
    for key, _fn in names_order:
        print(f"  {key}: {in_counts[key]}")
    print("\nOutput files:")
    for _, fn in names_order:
        print(f"  {out_dir / fn}")
    print(f"  {debug_csv}")
    for tag, _, _ in regions_spec:
        for variant_name in ("standard", "xy_swapped"):
            print(f"  {out_dir / f'{tag}_{variant_name}.png'}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
