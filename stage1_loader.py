"""Stage 1 loader for Cygnus-like single EV analysis object initialization."""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd


ID_COL = "object_id"

COORD_COLS = ["center_x_um", "center_y_um"]

MORPHOLOGY_COLS = ["area_um2", "perimeter_um", "circularity"]

MARKER_COLS = ["PanEV", "EpCAM", "MET", "SDC1", "EGFR", "ADAM10", "CTSH", "PDL1", "HER2"]

PANEV_MARKER = "PanEV"

RELEVANT_MARKERS = ["EpCAM", "MET", "SDC1", "EGFR", "ADAM10", "CTSH", "PDL1", "HER2"]

POSITION_COL = "position"
SAMPLE_COL = "sample"


COLUMN_MAPPING = {
    "Unnamed: 0": "object_id",
    "Center.of.Mass.X...m.": "center_x_um",
    "Center.of.Mass.Y...m.": "center_y_um",
    "Area...m2.": "area_um2",
    "Perimeter...m.": "perimeter_um",
    "Circularity": "circularity",
    "PanEV": "PanEV",
    "EpCAM": "EpCAM",
    "MET": "MET",
    "SDC1": "SDC1",
    "EGFR": "EGFR",
    "ADAM10": "ADAM10",
    "CTSH": "CTSH",
    "PDL1": "PDL1",
    "HER2": "HER2",
    "image": "position",
    "celltype": "sample",
}


def _build_markers_meta() -> pd.DataFrame:
    rows = []
    for marker in MARKER_COLS:
        rows.append(
            {
                "marker_name": marker,
                "is_panev": marker == PANEV_MARKER,
                "relevant": marker in RELEVANT_MARKERS,
            }
        )
    return pd.DataFrame(rows, columns=["marker_name", "is_panev", "relevant"])


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [raw_col for raw_col in COLUMN_MAPPING if raw_col not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")


def load_cygnus_object(filepath: str) -> Dict[str, Any]:
    """Load CSV and initialize the stage-1 analysis object."""
    raw_data = pd.read_csv(filepath)
    _validate_required_columns(raw_data)

    cleaned_data = raw_data.rename(columns=COLUMN_MAPPING).copy()

    cleaned_data[ID_COL] = pd.to_numeric(cleaned_data[ID_COL], errors="raise").astype("int64")

    float_cols = COORD_COLS + MORPHOLOGY_COLS + MARKER_COLS
    for col in float_cols:
        cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors="raise").astype("float64")

    cleaned_data[POSITION_COL] = cleaned_data[POSITION_COL].astype("string").astype("category")
    cleaned_data[SAMPLE_COL] = cleaned_data[SAMPLE_COL].astype("string").astype("category")

    raw_score = cleaned_data[[ID_COL] + RELEVANT_MARKERS].set_index(ID_COL)

    analysis_object = {
        "raw_data": raw_data,
        "cleaned_data": cleaned_data,
        "filtered_data": None,
        "matrices": {
            "Raw_Score": raw_score,
            "normalized_exp_mat": None,
            "scaled_exp_matrix": None,
            "binary_exp_matrix": None,
        },
        "coordinates": {
            "center_x_um": cleaned_data["center_x_um"],
            "center_y_um": cleaned_data["center_y_um"],
        },
        "morphology": {
            "area_um2": cleaned_data["area_um2"],
            "perimeter_um": cleaned_data["perimeter_um"],
            "circularity": cleaned_data["circularity"],
        },
        "ev_meta": {
            "object_id": cleaned_data["object_id"],
            "position": cleaned_data["position"],
            "sample": cleaned_data["sample"],
            "cluster_labels": {},
        },
        "markers_meta": _build_markers_meta(),
        "marker_analysis": {},  # populated in Stage 2+; dict so nested assignment / setdefault works
        "dim_red": {
            "pca": None,
            "tsne": None,
            "umap": None,
        },
        "clustering": {},  # populated in Stage 5; must be dict (not None) for nested assignment
        "colocalization": {},  # filled in Stage 4; must be dict (not None) for nested assignment
        "threshold_table": None,
    }
    return analysis_object


def _print_summary(analysis_object: Dict[str, Any]) -> None:
    cleaned_data = analysis_object["cleaned_data"]

    missing_counts = cleaned_data.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    print("=== Cygnus Analysis Object Initialized ===")
    print(f"Total objects     : {len(cleaned_data)}")
    print(f"Total columns     : {cleaned_data.shape[1]}")
    print(f"Samples           : {list(cleaned_data[SAMPLE_COL].cat.categories)}")
    print(f"Positions         : {list(cleaned_data[POSITION_COL].cat.categories)}")
    print(f"Marker columns    : {', '.join(MARKER_COLS)}")
    print(f"Relevant markers  : {', '.join(RELEVANT_MARKERS)}")
    print(f"Morphology cols   : {', '.join(MORPHOLOGY_COLS)}")
    if missing_counts.empty:
        print("Missing values    : None")
    else:
        print(f"Missing values    : {missing_counts.to_dict()}")
    print("==========================================")


if __name__ == "__main__":
    analysis_object = load_cygnus_object("all_cells.csv")
    _print_summary(analysis_object)
