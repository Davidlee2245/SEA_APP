"""Stage 1 loader for Cygnus-like single EV analysis object initialization."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


_LOG = logging.getLogger(__name__)


ID_COL = "object_id"

COORD_COLS = ["centroid_x", "centroid_y"]

MORPHOLOGY_COLS = ["area", "perimeter", "circularity"]

POSITION_COL = "position"
SAMPLE_COL = "sample"

# Deprecated: populated only after load; use analysis_object["marker_names"] / ["score_col"].
RELEVANT_MARKERS: List[str] = []
PANEV_MARKER = ""  # use ao["score_col"] after load


def _infer_score_col(columns: pd.Index) -> str:
    candidates = [str(c) for c in columns if str(c).endswith("_score")]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one column ending in '_score'; found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _infer_marker_positive_cols(columns: pd.Index) -> List[str]:
    return sorted([str(c) for c in columns if str(c).endswith("_positive")])


def _infer_raw_tiff_mean_cols(columns: pd.Index) -> List[str]:
    return sorted([str(c) for c in columns if str(c).endswith("_raw_tiff_mean")])


def _infer_intensity_bg_subtracted_cols(columns: pd.Index) -> List[str]:
    return sorted([str(c) for c in columns if str(c).endswith("_intensity_bg_subtracted")])


def _positive_col_to_base(col: str) -> str:
    if col.endswith("_positive"):
        return col[: -len("_positive")]
    return col


def _raw_tiff_mean_col_to_base(col: str) -> str:
    suf = "_raw_tiff_mean"
    if str(col).endswith(suf):
        return str(col)[: -len(suf)]
    return str(col)


def _intensity_bg_subtracted_col_to_base(col: str) -> str:
    suf = "_intensity_bg_subtracted"
    if str(col).endswith(suf):
        return str(col)[: -len(suf)]
    return str(col)


def _nan_rate_source_column(base: str, mpc: str, df: pd.DataFrame) -> str:
    """Prefer ``raw__{base}`` (Results Viewer Raw Intensity sheet) for NaN-rate filtering when present."""
    rcol = f"raw__{base}"
    if rcol in df.columns:
        return rcol
    return _source_col_for_nan_rate(base, mpc, df)


def _source_col_for_nan_rate(base: str, mpc: str, df: pd.DataFrame) -> str:
    """Same intensity priority as short ``cleaned_data[base]``: raw TIFF > ibg > *_positive."""
    rtc = f"{base}_raw_tiff_mean"
    ibg = f"{base}_intensity_bg_subtracted"
    if rtc in df.columns:
        return rtc
    if ibg in df.columns:
        return ibg
    return mpc


def _infer_valid_marker_cols(
    cleaned_data: pd.DataFrame,
    marker_names: List[str],
    marker_positive_cols: List[str],
    nan_threshold: float,
) -> Tuple[List[str], List[str]]:
    """Return (valid_marker_names, excluded_marker_names) using per-marker NaN rate on intensity columns."""
    valid: List[str] = []
    excluded: List[str] = []
    for base, mpc in zip(marker_names, marker_positive_cols):
        col = _nan_rate_source_column(base, mpc, cleaned_data)
        ser = pd.to_numeric(cleaned_data[col], errors="coerce")
        n = int(len(ser))
        frac = float(ser.isna().sum() / n) if n else 0.0
        if frac > nan_threshold:
            excluded.append(base)
        else:
            valid.append(base)
    return valid, excluded


def _marker_nan_fraction_pairs(
    cleaned_data: pd.DataFrame,
    marker_names: List[str],
    marker_positive_cols: List[str],
) -> List[Tuple[str, float]]:
    """Per-marker NaN fraction on the same intensity column used for filtering."""
    pairs: List[Tuple[str, float]] = []
    for base, mpc in zip(marker_names, marker_positive_cols):
        col = _nan_rate_source_column(base, mpc, cleaned_data)
        ser = pd.to_numeric(cleaned_data[col], errors="coerce")
        n = int(len(ser))
        frac = float(ser.isna().sum() / n) if n else 0.0
        pairs.append((base, frac))
    return pairs


def _build_markers_meta(marker_positive_cols: List[str], score_col: str) -> pd.DataFrame:
    rows = []
    for mpc in marker_positive_cols:
        rows.append(
            {
                "marker_name": _positive_col_to_base(mpc),
                "positive_column": mpc,
                "is_panev": False,
                "is_pan_score_column": False,
            }
        )
    rows.append(
        {
            "marker_name": score_col,
            "positive_column": "",
            "is_panev": False,
            "is_pan_score_column": True,
        }
    )
    return pd.DataFrame(rows)


def load_cygnus_object(filepath: str, *, valid_marker_nan_threshold: float = 0.5) -> Dict[str, Any]:
    """Load CSV and initialize the stage-1 analysis object.

    Parameters
    ----------
    valid_marker_nan_threshold
        Exclude a marker from ``valid_marker_names`` when its NaN fraction on the
        rate column exceeds this value (default ``0.5`` = more than 50% NaN).
        The rate column is ``raw__{marker}`` when present (Results Viewer **Raw Intensity**
        merge); otherwise ``*_raw_tiff_mean`` > ``*_intensity_bg_subtracted`` > ``*_positive``.
        If **no** markers pass, the loader logs a warning and sets ``valid_marker_names``
        to marker(s) tied for the **lowest** NaN fraction (so the pipeline never stops solely
        because every column failed the threshold).
    """
    raw_data = pd.read_csv(filepath)

    required_fixed = [
        ID_COL,
        SAMPLE_COL,
        POSITION_COL,
        *COORD_COLS,
        *MORPHOLOGY_COLS,
    ]
    missing = [c for c in required_fixed if c not in raw_data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    score_col = _infer_score_col(raw_data.columns)
    marker_positive_cols = _infer_marker_positive_cols(raw_data.columns)
    overlap = set(marker_positive_cols) & {score_col}
    if overlap:
        raise ValueError(f"score column must not also be a _positive column: {overlap}")
    if not marker_positive_cols:
        raise ValueError("No columns ending with '_positive' found.")

    marker_names = [_positive_col_to_base(mpc) for mpc in marker_positive_cols]
    if len(set(marker_names)) != len(marker_names):
        raise ValueError(f"Duplicate marker base names after stripping '_positive': {marker_positive_cols}")

    raw_tiff_mean_cols = _infer_raw_tiff_mean_cols(raw_data.columns)
    dim_red_raw_tiff_mean_columns: List[str] = []
    dim_red_uses_raw_tiff_mean = False
    if raw_tiff_mean_cols:
        bases_from_raw = sorted({_raw_tiff_mean_col_to_base(c) for c in raw_tiff_mean_cols})
        if bases_from_raw != sorted(marker_names):
            raise ValueError(
                f"*_raw_tiff_mean marker bases {bases_from_raw} must exactly match bases from *_positive "
                f"{sorted(marker_names)} (same markers, no extras or omissions)."
            )
        if len(raw_tiff_mean_cols) != len(marker_names):
            raise ValueError(
                f"Expected {len(marker_names)} *_raw_tiff_mean columns matching markers; found {len(raw_tiff_mean_cols)}."
            )
        overlap_rm = set(raw_tiff_mean_cols) & {score_col}
        if overlap_rm:
            raise ValueError(f"score column must not also be a *_raw_tiff_mean column: {overlap_rm}")
        overlap_pos = set(raw_tiff_mean_cols) & set(marker_positive_cols)
        if overlap_pos:
            raise ValueError(f"*_positive columns must not duplicate *_raw_tiff_mean names: {overlap_pos}")
        dim_red_raw_tiff_mean_columns = [f"{mn}_raw_tiff_mean" for mn in marker_names]
        dim_red_uses_raw_tiff_mean = all(c in raw_data.columns for c in dim_red_raw_tiff_mean_columns)
        if not dim_red_uses_raw_tiff_mean:
            raise ValueError("Internal error: raw TIFF mean column alignment failed.")

    ibg_mean_cols = _infer_intensity_bg_subtracted_cols(raw_data.columns)
    dim_red_intensity_bg_columns: List[str] = []
    dim_red_uses_intensity_bg_subtracted = False
    if not dim_red_uses_raw_tiff_mean and ibg_mean_cols:
        bases_from_ibg = sorted({_intensity_bg_subtracted_col_to_base(c) for c in ibg_mean_cols})
        if bases_from_ibg != sorted(marker_names):
            raise ValueError(
                f"*_intensity_bg_subtracted marker bases {bases_from_ibg} must exactly match bases from *_positive "
                f"{sorted(marker_names)} (same markers, no extras or omissions)."
            )
        if len(ibg_mean_cols) != len(marker_names):
            raise ValueError(
                f"Expected {len(marker_names)} *_intensity_bg_subtracted columns matching markers; found {len(ibg_mean_cols)}."
            )
        overlap_ibg_score = set(ibg_mean_cols) & {score_col}
        if overlap_ibg_score:
            raise ValueError(f"score column must not also be a *_intensity_bg_subtracted column: {overlap_ibg_score}")
        overlap_ibg_pos = set(ibg_mean_cols) & set(marker_positive_cols)
        if overlap_ibg_pos:
            raise ValueError(
                f"*_positive columns must not duplicate *_intensity_bg_subtracted names: {overlap_ibg_pos}"
            )
        dim_red_intensity_bg_columns = [f"{mn}_intensity_bg_subtracted" for mn in marker_names]
        dim_red_uses_intensity_bg_subtracted = all(c in raw_data.columns for c in dim_red_intensity_bg_columns)
        if not dim_red_uses_intensity_bg_subtracted:
            raise ValueError("Internal error: intensity bg subtracted column alignment failed.")

    cleaned_data = raw_data.copy()

    cleaned_data[ID_COL] = pd.to_numeric(cleaned_data[ID_COL], errors="raise").astype("int64")

    float_required = list(COORD_COLS) + list(MORPHOLOGY_COLS) + [score_col]
    for col in float_required:
        cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors="raise").astype("float64")

    for mpc in marker_positive_cols:
        cleaned_data[mpc] = pd.to_numeric(cleaned_data[mpc], errors="coerce")

    for rtc in dim_red_raw_tiff_mean_columns:
        cleaned_data[rtc] = pd.to_numeric(cleaned_data[rtc], errors="coerce").astype("float64")

    for ibc in dim_red_intensity_bg_columns:
        cleaned_data[ibc] = pd.to_numeric(cleaned_data[ibc], errors="coerce").astype("float64")

    for c in list(cleaned_data.columns):
        if isinstance(c, str) and (c.startswith("raw__") or c.startswith("bg__")):
            cleaned_data[c] = pd.to_numeric(cleaned_data[c], errors="coerce").astype("float64")

    if not (0.0 <= valid_marker_nan_threshold < 1.0):
        raise ValueError("valid_marker_nan_threshold must satisfy 0.0 <= valid_marker_nan_threshold < 1.0.")
    valid_list, excluded_high_nan = _infer_valid_marker_cols(
        cleaned_data,
        marker_names,
        marker_positive_cols,
        valid_marker_nan_threshold,
    )
    if excluded_high_nan:
        _LOG.warning(
            "Excluding markers with >%g%% NaN: %s",
            valid_marker_nan_threshold * 100.0,
            excluded_high_nan,
        )
    if not valid_list:
        nan_pairs = _marker_nan_fraction_pairs(cleaned_data, marker_names, marker_positive_cols)
        min_frac = min(frac for _, frac in nan_pairs)
        valid_list = [base for base, frac in nan_pairs if frac == min_frac]
        dropped = [base for base, frac in nan_pairs if frac != min_frac]
        _LOG.warning(
            "No markers pass the NaN-rate filter (valid_marker_nan_threshold=%s); "
            "all candidates had NaN fraction > %s. Continuing with %s marker(s) tied for "
            "lowest NaN fraction (nan_frac=%.6f): %s. Other markers excluded from "
            "valid_marker_names for downstream analysis: %s",
            valid_marker_nan_threshold,
            valid_marker_nan_threshold,
            len(valid_list),
            min_frac,
            valid_list,
            dropped or excluded_high_nan,
        )

    for mpc in marker_positive_cols:
        cleaned_data[mpc] = cleaned_data[mpc].fillna(0.0).astype("float64")

    # Short base-name columns: raw TIFF mean > intensity bg subtracted > *_positive.
    for base, mpc in zip(marker_names, marker_positive_cols):
        rtc = f"{base}_raw_tiff_mean"
        ibg = f"{base}_intensity_bg_subtracted"
        if rtc in cleaned_data.columns:
            cleaned_data[base] = cleaned_data[rtc]
        elif ibg in cleaned_data.columns:
            cleaned_data[base] = cleaned_data[ibg]
        else:
            cleaned_data[base] = cleaned_data[mpc]

    cleaned_data[POSITION_COL] = cleaned_data[POSITION_COL].astype("string").astype("category")
    cleaned_data[SAMPLE_COL] = cleaned_data[SAMPLE_COL].astype("string").astype("category")

    def _matrix_from_prefixed(prefix: str) -> pd.DataFrame | None:
        cols = [c for c in cleaned_data.columns if isinstance(c, str) and c.startswith(prefix)]
        if not cols:
            return None
        rename_map = {c: c[len(prefix) :] for c in cols}
        part = cleaned_data[[ID_COL] + cols].copy()
        part = part.rename(columns=rename_map)
        part = part.set_index(ID_COL)
        for c in part.columns:
            part[c] = pd.to_numeric(part[c], errors="coerce")
        return part

    raw_intensity_matrix = _matrix_from_prefixed("raw__")
    bg_subtracted_matrix = _matrix_from_prefixed("bg__")

    raw_score = pd.DataFrame(
        {base: cleaned_data[base].astype("float64") for base in marker_names},
    )
    raw_score.index = cleaned_data[ID_COL].values
    raw_score.index.name = ID_COL

    analysis_object: Dict[str, Any] = {
        "raw_data": raw_data,
        "cleaned_data": cleaned_data,
        "filtered_data": None,
        "score_col": score_col,
        "marker_positive_cols": list(marker_positive_cols),
        "marker_names": list(marker_names),
        "marker_raw_tiff_mean_cols": list(dim_red_raw_tiff_mean_columns),
        "dim_red_uses_raw_tiff_mean": dim_red_uses_raw_tiff_mean,
        "marker_intensity_bg_subtracted_cols": list(dim_red_intensity_bg_columns),
        "dim_red_uses_intensity_bg_subtracted": dim_red_uses_intensity_bg_subtracted,
        "valid_marker_names": list(valid_list),
        "valid_marker_nan_threshold": float(valid_marker_nan_threshold),
        "raw_intensity_matrix": raw_intensity_matrix,
        "bg_subtracted_matrix": bg_subtracted_matrix,
        "markers_source_is_binary": True,
        "matrices": {
            "Raw_Score": raw_score,
            "normalized_exp_mat": None,
            "scaled_exp_matrix": None,
            "binary_exp_matrix": None,
        },
        "coordinates": {
            "centroid_x": cleaned_data["centroid_x"],
            "centroid_y": cleaned_data["centroid_y"],
        },
        "morphology": {
            "area": cleaned_data["area"],
            "perimeter": cleaned_data["perimeter"],
            "circularity": cleaned_data["circularity"],
        },
        "ev_meta": {
            "object_id": cleaned_data[ID_COL],
            "position": cleaned_data[POSITION_COL],
            "sample": cleaned_data[SAMPLE_COL],
            "cluster_labels": {},
        },
        "markers_meta": _build_markers_meta(marker_positive_cols, score_col),
        "marker_analysis": {},
        "dim_red": {
            "pca": None,
            "tsne": None,
            "umap": None,
        },
        "clustering": {},
        "colocalization": {},
        "threshold_table": None,
    }
    # Always set after construction so ao['score_col'] exists for score_column() and downstream.
    analysis_object["score_col"] = str(score_col)
    return analysis_object


def marker_base_columns(ao: Dict[str, Any]) -> List[str]:
    """Per-marker matrix column names (strip _positive) — same as normalized / Raw_Score columns."""
    return list(ao.get("marker_names") or [])


def valid_marker_names(ao: Dict[str, Any]) -> List[str]:
    """Markers that pass the load-time NaN-rate filter; use for analysis stages."""
    v = ao.get("valid_marker_names")
    if isinstance(v, list):
        return list(v)
    return marker_base_columns(ao)


def score_column(ao: Dict[str, Any]) -> Optional[str]:
    """Pan-marker score column name in cleaned_data (ends with _score)."""
    sc = ao.get("score_col")
    if sc is not None and str(sc).strip() != "":
        return str(sc)
    cleaned = ao.get("cleaned_data")
    if isinstance(cleaned, pd.DataFrame):
        for c in cleaned.columns:
            if str(c).endswith("_score"):
                return str(c)
    return None


def _print_summary(analysis_object: Dict[str, Any]) -> None:
    cleaned_data = analysis_object["cleaned_data"]
    mn = analysis_object["marker_names"]
    sc = score_column(analysis_object)

    missing_counts = cleaned_data.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    print("=== Cygnus Analysis Object Initialized ===")
    print(f"Total objects     : {len(cleaned_data)}")
    print(f"Total columns     : {cleaned_data.shape[1]}")
    print(f"Samples           : {list(cleaned_data[SAMPLE_COL].cat.categories)}")
    print(f"Positions         : {list(cleaned_data[POSITION_COL].cat.categories)}")
    print(f"Score column      : {sc}")
    print(f"Marker (+) cols   : {', '.join(analysis_object['marker_positive_cols'])}")
    print(f"Marker bases      : {', '.join(mn)}")
    vm = analysis_object.get("valid_marker_names") or []
    if vm and set(vm) != set(mn):
        print(f"Valid markers     : {', '.join(vm)} (after NaN-rate filter)")
    if analysis_object.get("dim_red_uses_raw_tiff_mean"):
        print(
            "Raw TIFF mean cols: "
            + ", ".join(analysis_object.get("marker_raw_tiff_mean_cols") or [])
        )
    if analysis_object.get("dim_red_uses_intensity_bg_subtracted"):
        print(
            "Intensity bg cols: "
            + ", ".join(analysis_object.get("marker_intensity_bg_subtracted_cols") or [])
        )
    print(f"Morphology cols   : {', '.join(MORPHOLOGY_COLS)}")
    if missing_counts.empty:
        print("Missing values    : None")
    else:
        print(f"Missing values    : {missing_counts.to_dict()}")
    print("==========================================")


if __name__ == "__main__":
    analysis_object = load_cygnus_object("all_cells.csv")
    _print_summary(analysis_object)
