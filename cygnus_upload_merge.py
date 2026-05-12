"""Merge multiple Cygnus-style uploads (CSV / Excel) for /api/cygnus/run."""

from __future__ import annotations

from io import BytesIO
from typing import Any, List, Tuple

import pandas as pd

from stage1_loader import (
    COORD_COLS,
    ID_COL,
    MORPHOLOGY_COLS,
    POSITION_COL,
    SAMPLE_COL,
    _infer_marker_positive_cols,
    _infer_raw_tiff_mean_cols,
    _infer_score_col,
    _positive_col_to_base,
    _raw_tiff_mean_col_to_base,
)

_FIXED = [ID_COL, SAMPLE_COL, POSITION_COL, *COORD_COLS, *MORPHOLOGY_COLS]

CYGNUS_UPLOAD_SHEET_HINT = (
    "Please upload the Full sheet from the Results Viewer Excel export, or export as CSV from the Full sheet."
)


def _find_sheet_ci(sheet_names: List[str], want: str) -> str | None:
    wl = want.strip().lower()
    for n in sheet_names:
        if str(n).strip().lower() == wl:
            return str(n)
    return None


def _read_excel_results_viewer_three_sheets(
    bio: BytesIO, fn: str, *, engine: str | None
) -> pd.DataFrame | None:
    """
    If the workbook has ``Full``, ``Raw Intensity``, and ``BG Subtracted`` (case-insensitive),
    read all three, validate ``Full`` with the Cygnus schema, prefix non-fixed columns on
    Raw/BG sheets with ``raw__`` / ``bg__``, and left-join onto ``Full`` on ``object_id``.

    Returns ``None`` if any sheet is missing or ``Full`` does not match the marker schema
    (caller falls back to single-sheet logic).
    """
    bio.seek(0)
    xl = pd.ExcelFile(bio, engine=engine) if engine else pd.ExcelFile(bio)
    names = list(xl.sheet_names)
    n_full = _find_sheet_ci(names, "full")
    n_raw = _find_sheet_ci(names, "raw intensity")
    n_bg = _find_sheet_ci(names, "bg subtracted")
    if not n_full or not n_raw or not n_bg:
        return None

    def read_one(sheet: str) -> pd.DataFrame:
        bio.seek(0)
        kw: dict[str, Any] = {"sheet_name": sheet}
        if engine:
            kw["engine"] = engine
        return pd.read_excel(bio, **kw)

    df_full = read_one(n_full)
    if not _dataframe_has_cygnus_marker_schema(df_full):
        return None

    df_raw = read_one(n_raw)
    df_bg = read_one(n_bg)
    missing_id = [s for s, d in (("Raw Intensity", df_raw), ("BG Subtracted", df_bg)) if ID_COL not in d.columns]
    if missing_id:
        raise ValueError(f"{fn}: {', '.join(missing_id)} sheet(s) missing required column {ID_COL!r}.")

    extra_raw = [c for c in df_raw.columns if c not in _FIXED]
    df_r = df_raw[[ID_COL] + extra_raw].copy()
    df_r = df_r.rename(columns={c: f"raw__{c}" for c in extra_raw})

    extra_bg = [c for c in df_bg.columns if c not in _FIXED]
    df_b = df_bg[[ID_COL] + extra_bg].copy()
    df_b = df_b.rename(columns={c: f"bg__{c}" for c in extra_bg})

    out = df_full.merge(df_r, on=ID_COL, how="left")
    out = out.merge(df_b, on=ID_COL, how="left")
    return out


def _dataframe_has_cygnus_marker_schema(df: pd.DataFrame) -> bool:
    """True if columns look like the Results Viewer Full export (fixed cols + *_positive + one *_score)."""
    if df is None or getattr(df, "empty", False):
        return False
    if any(c not in df.columns for c in _FIXED):
        return False
    if not _infer_marker_positive_cols(df.columns):
        return False
    try:
        _infer_score_col(df.columns)
    except ValueError:
        return False
    return True


def _read_excel_cygnus_sheet(bio: BytesIO, fn: str, *, engine: str | None) -> pd.DataFrame:
    """
    Prefer a sheet named ``Full`` (case-insensitive). Otherwise try the first sheet, then
    any remaining sheet until one matches the Cygnus marker schema.

    Raises ValueError with CYGNUS_UPLOAD_SHEET_HINT if no suitable sheet exists.
    """
    bio.seek(0)
    xl = pd.ExcelFile(bio, engine=engine) if engine else pd.ExcelFile(bio)
    names = list(xl.sheet_names)
    if not names:
        raise ValueError(f"{fn}: Excel workbook has no sheets. {CYGNUS_UPLOAD_SHEET_HINT}")

    full_name = next((n for n in names if str(n).strip().lower() == "full"), None)
    tried: set[str] = set()

    def read_one(sheet: Any) -> pd.DataFrame:
        bio.seek(0)
        kw: dict[str, Any] = {"sheet_name": sheet}
        if engine:
            kw["engine"] = engine
        return pd.read_excel(bio, **kw)

    def mark(sheet: Any) -> None:
        tried.add(str(sheet))

    # 1) Prefer Full when present
    if full_name is not None:
        df_full = read_one(full_name)
        mark(full_name)
        if _dataframe_has_cygnus_marker_schema(df_full):
            return df_full

    # 2) First sheet (backward compatibility; skip if already read as Full)
    first = names[0]
    if str(first) not in tried:
        df0 = read_one(first)
        mark(first)
        if _dataframe_has_cygnus_marker_schema(df0):
            return df0

    # 3) Remaining sheets
    for s in names[1:]:
        if str(s) in tried:
            continue
        dfi = read_one(s)
        mark(s)
        if _dataframe_has_cygnus_marker_schema(dfi):
            return dfi

    raise ValueError(f"{fn}: {CYGNUS_UPLOAD_SHEET_HINT}")


def _validate_optional_raw_tiff_mean(df: pd.DataFrame, marker_bases: List[str], label: str) -> None:
    """If any *_raw_tiff_mean columns exist, require one per marker matching *_positive bases."""
    raw_cols = _infer_raw_tiff_mean_cols(df.columns)
    if not raw_cols:
        return
    bases_exp = sorted(marker_bases)
    rb = sorted({_raw_tiff_mean_col_to_base(c) for c in raw_cols})
    if rb != bases_exp:
        raise ValueError(
            f"{label}: *_raw_tiff_mean marker bases {rb} must exactly match *_positive bases {bases_exp}"
        )
    if len(raw_cols) != len(bases_exp):
        raise ValueError(
            f"{label}: expected {len(bases_exp)} *_raw_tiff_mean columns when present; found {len(raw_cols)}"
        )


def _validate_frame(df: pd.DataFrame, label: str) -> Tuple[str, List[str], List[str]]:
    missing = [c for c in _FIXED if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns {missing}")
    score_col = _infer_score_col(df.columns)
    positives = _infer_marker_positive_cols(df.columns)
    overlap = set(positives) & {score_col}
    if overlap:
        raise ValueError(f"{label}: score column must not also be a _positive column: {overlap}")
    bases = sorted(_positive_col_to_base(p) for p in positives)
    if len(bases) != len(set(bases)):
        raise ValueError(f"{label}: duplicate marker base names from _positive columns")
    _validate_optional_raw_tiff_mean(df, bases, label)
    return score_col, positives, bases


def read_uploaded_table(file_storage: Any, openpyxl_available: bool) -> Tuple[pd.DataFrame, str]:
    fn = (getattr(file_storage, "filename", None) or "upload").strip() or "upload"
    raw = file_storage.read()
    bio = BytesIO(raw)
    low = fn.lower()
    if low.endswith(".csv"):
        df = pd.read_csv(bio)
    elif low.endswith(".xlsx"):
        if not openpyxl_available:
            raise ValueError(f"{fn}: .xlsx requires openpyxl; install openpyxl or use CSV")
        bio.seek(0)
        df_three = _read_excel_results_viewer_three_sheets(bio, fn, engine="openpyxl")
        if df_three is not None:
            df = df_three
        else:
            bio.seek(0)
            df = _read_excel_cygnus_sheet(bio, fn, engine="openpyxl")
    elif low.endswith(".xls"):
        bio.seek(0)
        df_three = _read_excel_results_viewer_three_sheets(bio, fn, engine=None)
        if df_three is not None:
            df = df_three
        else:
            bio.seek(0)
            df = _read_excel_cygnus_sheet(bio, fn, engine=None)
    else:
        raise ValueError(f"{fn}: unsupported type (use .csv, .xlsx, or .xls)")
    return df, fn


def merge_cygnus_upload_files(file_storages: List[Any], openpyxl_available: bool) -> pd.DataFrame:
    """
    Load each upload, validate fixed + marker columns, ensure marker signatures match,
    then concatenate rows with ignore_index=True.
    """
    if not file_storages:
        raise ValueError("No files provided")

    parsed: List[Tuple[str, pd.DataFrame, str, List[str], List[str]]] = []
    for fs in file_storages:
        df, label = read_uploaded_table(fs, openpyxl_available)
        score_col, _pos_cols, bases = _validate_frame(df, label)
        parsed.append((label, df, score_col, _pos_cols, bases))

    ref_label, ref_df, ref_score, _rp, ref_bases = parsed[0]
    ref_colset = set(ref_df.columns)

    for other_label, other_df, o_score, _op, o_bases in parsed[1:]:
        if ref_bases != o_bases:
            raise ValueError(
                f"Column mismatch: file {ref_label} has markers {ref_bases} but file {other_label} has markers {o_bases}"
            )
        if ref_score != o_score:
            raise ValueError(
                f"Column mismatch: file {ref_label} uses score column {ref_score!r} but file {other_label} uses {o_score!r}"
            )
        if ref_colset != set(other_df.columns):
            only_ref = sorted(ref_colset - set(other_df.columns))
            only_other = sorted(set(other_df.columns) - ref_colset)
            raise ValueError(
                f"Column mismatch: file {ref_label} and file {other_label} have different columns "
                f"(only in {ref_label}: {only_ref}, only in {other_label}: {only_other})"
            )

    col_order = list(ref_df.columns)
    frames_ordered = [df.loc[:, col_order] for _, df, _, _, _ in parsed]
    return pd.concat(frames_ordered, ignore_index=True)
