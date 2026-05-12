"""Stage 3 preprocessing: QC filtering, normalization, scaling, thresholding."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.filters import threshold_otsu

from stage1_loader import ID_COL, marker_base_columns, score_column, valid_marker_names, load_cygnus_object


sns.set_theme(style="whitegrid")
_LOG = logging.getLogger(__name__)


def _get_input_df(ao: Dict[str, Any], use_filtered: bool = False) -> pd.DataFrame:
    if use_filtered:
        if ao.get("filtered_data") is None:
            raise ValueError("use_filtered=True but ao['filtered_data'] is None. Run apply_qc_filters first.")
        return ao["filtered_data"]
    return ao["cleaned_data"]


def apply_qc_filters(
    ao: dict,
    remove_zero_area: bool = True,
    remove_zero_perimeter: bool = True,
    remove_nan_circularity: bool = True,
    area_range: Optional[Tuple[float, float]] = None,
    circularity_range: Optional[Tuple[float, float]] = None,
    min_panev: Optional[float] = None,
    min_marker: Optional[Dict[str, float]] = None,
) -> dict:
    """Apply additive QC filters without modifying cleaned_data/raw_data."""
    df = ao["cleaned_data"]
    mask = pd.Series(True, index=df.index)
    applied_filters = []

    markers = marker_base_columns(ao)
    score = score_column(ao)

    if remove_zero_area:
        mask &= df["area"] > 0
        applied_filters.append("remove_zero_area")

    if remove_zero_perimeter:
        mask &= df["perimeter"] > 0
        applied_filters.append("remove_zero_perimeter")

    if remove_nan_circularity:
        mask &= df["circularity"].notna()
        applied_filters.append("remove_nan_circularity")

    if area_range is not None:
        lo, hi = area_range
        mask &= df["area"].between(lo, hi, inclusive="both")
        applied_filters.append(f"area_range={area_range}")

    if circularity_range is not None:
        lo, hi = circularity_range
        mask &= df["circularity"].between(lo, hi, inclusive="both")
        applied_filters.append(f"circularity_range={circularity_range}")

    if min_panev is not None:
        mask &= df[score] >= min_panev
        applied_filters.append(f"min_pan_score({score})={min_panev}")

    if min_marker:
        for marker, min_val in min_marker.items():
            if marker not in markers:
                raise ValueError(f"min_marker contains invalid marker '{marker}'. Expected one of {markers}.")
            mask &= df[marker] >= min_val
            applied_filters.append(f"min_marker[{marker}]={min_val}")

    before = len(df)
    any_filter_requested = (
        remove_zero_area
        or remove_zero_perimeter
        or remove_nan_circularity
        or area_range is not None
        or circularity_range is not None
        or min_panev is not None
        or bool(min_marker)
    )

    if any_filter_requested:
        ao["filtered_data"] = df.loc[mask].copy()
    else:
        ao["filtered_data"] = None

    after = len(ao["filtered_data"]) if ao["filtered_data"] is not None else before
    removed = before - after
    removed_pct = (removed / before * 100.0) if before else 0.0

    print("=== QC Filter Summary ===")
    print(f"Before filtering : {before} objects")
    print(f"After filtering  : {after} objects")
    print(f"Removed          : {removed} objects ({removed_pct:.1f}%)")
    print(f"Filters applied  : {', '.join(applied_filters) if applied_filters else 'none'}")
    print("=========================")
    return ao


def normalize_by_panev(
    ao: dict,
    use_filtered: bool = False,
    epsilon: float = 1e-6,
    include_panev: bool = False,
) -> dict:
    """Normalize per-marker intensities by a Pan-EV reference column + epsilon.

    When ``ao['bg_subtracted_matrix']`` is present (Results Viewer **BG Subtracted** merge),
    numerators use that matrix and ``bg__Pan-EV`` from the input dataframe as denominator.
    Otherwise ``{marker}_raw_tiff_mean`` … (unchanged). Only markers in ``ao['valid_marker_names']``
    are included in the output matrix. Sets ``ao['panev_intensity_col']`` to the denominator column name used.
    """
    df = _get_input_df(ao, use_filtered=use_filtered)
    markers_all = [m for m in marker_base_columns(ao) if m in set(valid_marker_names(ao))]
    if not markers_all:
        raise ValueError("No valid markers to normalize (ao['valid_marker_names'] is empty).")

    bg_mat = ao.get("bg_subtracted_matrix")
    use_rv_bg = (
        isinstance(bg_mat, pd.DataFrame)
        and not bg_mat.empty
        and "bg__Pan-EV" in df.columns
    )
    if use_rv_bg:
        num_cols = [m for m in markers_all if m in bg_mat.columns]
        use_rv_bg = bool(num_cols)
    raw_present = [m for m in markers_all if f"{m}_raw_tiff_mean" in df.columns]
    use_raw = bool(raw_present) and not use_rv_bg
    bg_present = [m for m in markers_all if f"{m}_intensity_bg_subtracted" in df.columns]
    use_bg = bool(bg_present) and not use_raw and not use_rv_bg

    if not use_raw and not use_bg and not use_rv_bg:
        _LOG.warning(
            "Raw TIFF mean columns not found; falling back to *_positive / score normalization",
        )

    def _pick_denominator_column() -> str:
        for cand in ("Pan-EV_raw_tiff_mean", "Pan-EV"):
            if cand in df.columns:
                return cand
        sc = score_column(ao)
        if sc is None or sc not in df.columns:
            raise ValueError(
                "No Pan-EV intensity column (Pan-EV_raw_tiff_mean / Pan-EV) and no usable *_score column for normalization.",
            )
        return sc

    if use_rv_bg:
        den_col = "bg__Pan-EV"
        oid_vals = df[ID_COL].to_numpy()
        numer = pd.DataFrame(
            {
                m: pd.to_numeric(bg_mat.reindex(oid_vals)[m].to_numpy(), errors="coerce")
                for m in num_cols
            },
            index=df.index,
        )
        _LOG.info("Using Results Viewer BG Subtracted matrix for normalization (denominator bg__Pan-EV).")
    elif use_raw:
        den_col = _pick_denominator_column()
        num_cols = raw_present
        numer = pd.concat(
            [pd.to_numeric(df[f"{m}_raw_tiff_mean"], errors="coerce") for m in num_cols],
            axis=1,
        )
        numer.columns = num_cols
    elif use_bg:
        den_col = _pick_denominator_column()
        num_cols = bg_present
        numer = pd.concat(
            [pd.to_numeric(df[f"{m}_intensity_bg_subtracted"], errors="coerce") for m in num_cols],
            axis=1,
        )
        numer.columns = num_cols
        _LOG.info("Using intensity_bg_subtracted columns for normalization")
    else:
        den_col = score_column(ao)
        if den_col is None or den_col not in df.columns:
            raise ValueError("score_column(ao) is missing or not in dataframe for legacy normalization.")
        num_cols = markers_all
        numer = df[num_cols].apply(pd.to_numeric, errors="coerce")

    panev = pd.to_numeric(df[den_col], errors="coerce")
    low_panev_flag = panev < epsilon
    ao.setdefault("ev_meta", {})
    ao["ev_meta"]["low_panev_flag"] = low_panev_flag
    ao["panev_intensity_col"] = den_col

    denom = panev + epsilon
    norm = numer.div(denom, axis=0)

    if include_panev:
        ratio_col = den_col if den_col not in norm.columns else f"{den_col}_div_denom"
        norm[ratio_col] = panev.div(denom)
        norm = norm[list(num_cols) + [ratio_col]]

    norm.index = df[ID_COL].values
    norm.index.name = ID_COL

    ao["matrices"]["normalized_exp_mat"] = norm
    return ao


def scale_expression_matrix(
    ao: dict,
    method: str = "zscore",
    input_matrix: str = "normalized",
) -> dict:
    """Scale selected expression matrix with zscore/maxscale/robust method."""
    source_map = {
        "raw": ao["matrices"]["Raw_Score"],
        "normalized": ao["matrices"]["normalized_exp_mat"],
    }
    if input_matrix not in source_map:
        raise ValueError("input_matrix must be one of {'raw', 'normalized'}.")

    mat = source_map[input_matrix]
    if mat is None:
        raise ValueError(f"Input matrix '{input_matrix}' is None.")

    scaled = mat.copy()
    for col in scaled.columns:
        x = pd.to_numeric(scaled[col], errors="coerce")
        if method == "maxscale":
            max_val = x.max()
            scaled[col] = x / max_val if pd.notna(max_val) and max_val != 0 else 0.0
        elif method == "zscore":
            mean = x.mean()
            std = x.std(ddof=0)
            scaled[col] = (x - mean) / std if pd.notna(std) and std != 0 else 0.0
        elif method == "robust":
            med = x.median()
            q75 = x.quantile(0.75)
            q25 = x.quantile(0.25)
            iqr = q75 - q25
            scaled[col] = (x - med) / iqr if pd.notna(iqr) and iqr != 0 else 0.0
        else:
            raise ValueError("method must be one of {'zscore', 'maxscale', 'robust'}.")

    ao["matrices"]["scaled_exp_matrix"] = scaled
    return ao


def _resolve_thresholds(
    data: pd.DataFrame,
    method: str,
    markers: List[str],
    thresholds: Optional[Dict[str, float]] = None,
    global_threshold: Optional[float] = None,
    percentile: float = 95.0,
) -> Dict[str, float]:
    thr: Dict[str, float] = {}
    cols = [c for c in markers if c in data.columns]

    if method == "manual":
        if not thresholds:
            raise ValueError("method='manual' requires thresholds dict.")
        for col in cols:
            if col not in thresholds:
                raise ValueError(f"Missing manual threshold for marker '{col}'.")
            thr[col] = float(thresholds[col])
    elif method == "global":
        if global_threshold is None:
            raise ValueError("method='global' requires global_threshold.")
        for col in cols:
            thr[col] = float(global_threshold)
    elif method == "percentile":
        for col in cols:
            values = pd.to_numeric(data[col], errors="coerce").dropna()
            if len(values) == 0:
                print(f"[WARNING] Marker '{col}' is all-NaN. Skipping threshold.")
                thr[col] = np.nan
                continue
            thr[col] = float(np.percentile(values, percentile))
    elif method == "otsu":
        for col in cols:
            series_vals = pd.to_numeric(data[col], errors="coerce").dropna()
            vals = series_vals.to_numpy()
            if vals.size == 0:
                thr[col] = 0.0
            elif series_vals.nunique() <= 1:
                print(f"[WARNING] Marker '{col}' has zero variance. Using median as fallback threshold.")
                thr[col] = float(series_vals.median())
            else:
                thr[col] = float(threshold_otsu(vals))
    else:
        raise ValueError("method must be one of {'manual', 'global', 'percentile', 'otsu'}.")
    return thr


def binarize_markers(
    ao: dict,
    method: str = "percentile",
    thresholds: Optional[Dict[str, float]] = None,
    global_threshold: Optional[float] = None,
    percentile: float = 95.0,
    input_matrix: str = "normalized",
) -> dict:
    """Binarize relevant marker expression matrix and store threshold table."""
    source_map = {
        "raw": ao["matrices"]["Raw_Score"],
        "normalized": ao["matrices"]["normalized_exp_mat"],
        "scaled": ao["matrices"]["scaled_exp_matrix"],
    }
    if input_matrix not in source_map:
        raise ValueError("input_matrix must be one of {'raw', 'normalized', 'scaled'}.")
    mat = source_map[input_matrix]
    if mat is None:
        raise ValueError(f"Input matrix '{input_matrix}' is None.")

    markers = marker_base_columns(ao)

    if ao.get("markers_source_is_binary"):
        raw = ao["matrices"]["Raw_Score"]
        binary = pd.DataFrame(index=raw.index)
        for marker in raw.columns:
            binary[marker] = (
                pd.to_numeric(raw[marker], errors="coerce").fillna(0.0) >= 0.5
            ).astype("int8")
        binary = binary.reindex(mat.index)
        binary.index.name = ID_COL
        ao["matrices"]["binary_exp_matrix"] = binary
        ao["threshold_table"] = pd.DataFrame(
            [{"marker": m, "threshold_value": 0.5, "method": "source_positive"} for m in markers if m in raw.columns],
            columns=["marker", "threshold_value", "method"],
        )
        return ao

    thr = _resolve_thresholds(
        mat,
        method=method,
        markers=markers,
        thresholds=thresholds,
        global_threshold=global_threshold,
        percentile=percentile,
    )

    binary = pd.DataFrame(index=mat.index)
    for marker in markers:
        if marker in mat.columns:
            binary[marker] = (pd.to_numeric(mat[marker], errors="coerce") > thr[marker]).astype("int8")
    binary.index.name = ID_COL

    ao["matrices"]["binary_exp_matrix"] = binary
    ao["threshold_table"] = pd.DataFrame(
        [{"marker": marker, "threshold_value": value, "method": method} for marker, value in thr.items()],
        columns=["marker", "threshold_value", "method"],
    )
    return ao


def plot_threshold_preview(ao: dict, marker: str, save_path: Optional[str] = None):
    """Preview threshold against marker histogram."""
    threshold_table = ao.get("threshold_table")
    if threshold_table is None or threshold_table.empty:
        raise ValueError("threshold_table is empty. Run binarize_markers first.")
    allowed = marker_base_columns(ao)
    if marker not in allowed:
        raise ValueError(f"marker must be one of {allowed}.")

    row = threshold_table.loc[threshold_table["marker"] == marker]
    if row.empty:
        raise ValueError(f"No threshold found for marker '{marker}'.")
    threshold_value = float(row["threshold_value"].iloc[0])
    method = str(row["method"].iloc[0])

    matrix = ao["matrices"]["normalized_exp_mat"]
    if matrix is None or marker not in matrix.columns:
        raise ValueError("normalized_exp_mat is missing or marker not present.")

    values = pd.to_numeric(matrix[marker], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(values, bins=50, kde=True, ax=ax, color="#4C72B0")
    ax.axvline(threshold_value, color="red", linestyle="--", linewidth=1.8, label=f"Threshold: {threshold_value:.4g}")
    ax.axvspan(threshold_value, values.max() if not values.empty else threshold_value, color="red", alpha=0.15)
    ax.set_xlabel(f"{marker} intensity")
    ax.set_ylabel("Count")
    ax.set_title(f"{marker} — threshold preview ({method})")
    ax.legend(loc="best")

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ao = load_cygnus_object("all_cells.csv")

    ao = apply_qc_filters(
        ao,
        remove_zero_area=True,
        remove_zero_perimeter=True,
        remove_nan_circularity=True,
    )
    ao = normalize_by_panev(ao, use_filtered=False, epsilon=1e-6, include_panev=False)
    ao = scale_expression_matrix(ao, method="zscore", input_matrix="normalized")
    ao = binarize_markers(
        ao,
        method="percentile",
        percentile=95.0,
        input_matrix="normalized",
    )

    print("\n=== Threshold Table ===")
    print(ao["threshold_table"].to_string(index=False))

    out_dir = Path("./output/stage3")
    out_dir.mkdir(parents=True, exist_ok=True)
    for marker_name in valid_marker_names(ao):
        plot_threshold_preview(ao, marker_name, save_path=str(out_dir / f"threshold_preview_{marker_name}.png"))
    print(f"\nSaved threshold previews to: {out_dir}")
