"""Stage 2 visualization module for Cygnus-like single EV analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from stage1_loader import (
    COORD_COLS,
    ID_COL,
    MORPHOLOGY_COLS,
    POSITION_COL,
    SAMPLE_COL,
    load_cygnus_object,
    marker_base_columns,
    score_column,
    valid_marker_names,
)


sns.set_theme(style="whitegrid")


def _ensure_dir(save_dir: Optional[str]) -> Optional[Path]:
    if save_dir is None:
        return None
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_figure(fig: Any, save_path: Optional[Path]) -> None:
    if save_path is None:
        return
    if hasattr(fig, "write_image"):
        fig.write_image(str(save_path))
    else:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")


def _normalize_sizes(series: pd.Series, min_size: float = 20.0, max_size: float = 180.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.nunique(dropna=True) <= 1:
        return pd.Series([min_size] * len(s), index=s.index, dtype="float64")
    scaled = (s - s.min()) / (s.max() - s.min())
    return min_size + scaled * (max_size - min_size)


def plot_marker_distributions(
    df: pd.DataFrame,
    marker: str,
    group_col: Optional[str] = None,
    plot_type: str = "all",
    save_dir: Optional[str] = None,
    valid_markers: Optional[Sequence[str]] = None,
) -> Dict[str, plt.Figure]:
    """Plot marker distributions globally or grouped by sample/position."""
    if marker not in df.columns:
        raise ValueError(f"Column '{marker}' is not in dataframe.")
    if valid_markers is not None and marker not in valid_markers:
        raise ValueError(f"Unknown marker '{marker}'. Expected one of {list(valid_markers)}.")

    valid_groups = {None, SAMPLE_COL, POSITION_COL}
    if group_col not in valid_groups:
        raise ValueError(f"group_col must be one of {valid_groups}.")

    valid_plot_types = {"all", "histogram", "density", "violin", "boxplot"}
    if plot_type not in valid_plot_types:
        raise ValueError(f"plot_type must be one of {valid_plot_types}.")

    out_dir = _ensure_dir(save_dir)
    figs: Dict[str, plt.Figure] = {}

    def should_make(name: str) -> bool:
        return plot_type in ("all", name)

    if group_col is None:
        if should_make("histogram"):
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(data=df, x=marker, kde=True, ax=ax, color="#4C72B0")
            ax.set_xlabel(f"{marker} intensity")
            ax.set_title(f"{marker} - histogram by global")
            figs["histogram"] = fig
            if out_dir is not None:
                _save_figure(fig, out_dir / f"{marker}_histogram_global.png")

        if should_make("density"):
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.kdeplot(data=df, x=marker, fill=True, ax=ax, color="#55A868")
            ax.set_xlabel(f"{marker} intensity")
            ax.set_title(f"{marker} - density by global")
            figs["density"] = fig
            if out_dir is not None:
                _save_figure(fig, out_dir / f"{marker}_density_global.png")
    else:
        if should_make("violin"):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.violinplot(data=df, x=group_col, y=marker, ax=ax, inner="quartile")
            ax.set_xlabel(group_col)
            ax.set_ylabel(f"{marker} intensity")
            ax.set_title(f"{marker} - violin by {group_col}")
            ax.tick_params(axis="x", rotation=45)
            figs["violin"] = fig
            if out_dir is not None:
                _save_figure(fig, out_dir / f"{marker}_violin_{group_col}.png")

        if should_make("boxplot"):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.boxplot(data=df, x=group_col, y=marker, ax=ax)
            ax.set_xlabel(group_col)
            ax.set_ylabel(f"{marker} intensity")
            ax.set_title(f"{marker} - boxplot by {group_col}")
            ax.tick_params(axis="x", rotation=45)
            figs["boxplot"] = fig
            if out_dir is not None:
                _save_figure(fig, out_dir / f"{marker}_boxplot_{group_col}.png")

    return figs


def plot_all_markers(
    df: pd.DataFrame,
    marker_cols: Sequence[str],
    score_col: str,
    save_dir: Optional[str] = None,
) -> Dict[str, Dict[str, plt.Figure]]:
    """Generate distribution plots for pan score and each per-marker column."""
    all_plots: Dict[str, Dict[str, plt.Figure]] = {}
    valid = list(marker_cols)
    for col in [score_col] + valid:
        if col not in df.columns:
            continue
        marker_plots: Dict[str, plt.Figure] = {}
        marker_plots.update(
            plot_marker_distributions(
                df, col, group_col=None, plot_type="all", save_dir=save_dir, valid_markers=None
            )
        )
        sample_plots = plot_marker_distributions(
            df, col, group_col=SAMPLE_COL, plot_type="all", save_dir=save_dir, valid_markers=None
        )
        marker_plots.update({f"{k}_by_sample": v for k, v in sample_plots.items()})
        position_plots = plot_marker_distributions(
            df, col, group_col=POSITION_COL, plot_type="all", save_dir=save_dir, valid_markers=None
        )
        marker_plots.update({f"{k}_by_position": v for k, v in position_plots.items()})
        all_plots[col] = marker_plots
    return all_plots


def plot_spatial(
    df: pd.DataFrame,
    color_by: str,
    size_by: Optional[str] = None,
    marker_size: float = 4,
    save_path: Optional[str] = None,
    interactive: bool = False,
):
    """Plot spatial scatter of objects colored by marker or metadata."""
    if color_by not in df.columns:
        raise ValueError(f"'{color_by}' is not in dataframe columns.")
    if size_by is not None and size_by not in {"area", "circularity"}:
        raise ValueError("size_by must be one of {'area', 'circularity', None}.")

    x_col, y_col = COORD_COLS[0], COORD_COLS[1]
    is_categorical = str(df[color_by].dtype) in ("category", "string") or df[color_by].dtype == "object"
    size_vals = _normalize_sizes(df[size_by]) if size_by else None

    if interactive:
        hover_fields: Dict[str, Any] = {
            ID_COL: True,
            SAMPLE_COL: True,
            POSITION_COL: True,
            x_col: ":.3f",
            y_col: ":.3f",
            "area": ":.3f",
            "circularity": ":.3f",
        }
        if pd.api.types.is_numeric_dtype(df[color_by]):
            hover_fields[color_by] = ":.3f"

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_by,
            size=size_by,
            hover_data=hover_fields,
            color_continuous_scale="viridis",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title=f"Spatial map - colored by {color_by}",
        )
        fig.update_layout(xaxis_title="X position (µm)", yaxis_title="Y position (µm)")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        if is_categorical:
            sns.scatterplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=color_by,
                size=size_vals if size_by else None,
                sizes=(20, 180) if size_by else None,
                palette="tab10",
                linewidth=0,
                alpha=0.8,
                ax=ax,
            )
            ax.legend(loc="best", frameon=True, title=color_by)
        else:
            scatter = ax.scatter(
                df[x_col],
                df[y_col],
                c=df[color_by],
                cmap="viridis",
                s=size_vals if size_by else marker_size**2,
                alpha=0.85,
                linewidth=0,
            )
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(color_by)
        ax.set_xlabel("X position (µm)")
        ax.set_ylabel("Y position (µm)")
        ax.set_title(f"Spatial map - colored by {color_by}")
        ax.set_aspect("equal", adjustable="box")

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, out_path)
    return fig


def plot_morphology_distributions(df: pd.DataFrame, save_dir: Optional[str] = None) -> Dict[str, Dict[str, plt.Figure]]:
    """Plot histogram and density for morphology columns."""
    out_dir = _ensure_dir(save_dir)
    results: Dict[str, Dict[str, plt.Figure]] = {}
    for col in MORPHOLOGY_COLS:
        fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
        sns.histplot(data=df, x=col, kde=True, ax=ax_hist, color="#8172B2")
        ax_hist.set_xlabel(col)
        ax_hist.set_title(f"{col} - histogram by global")

        fig_density, ax_density = plt.subplots(figsize=(7, 4))
        sns.kdeplot(data=df, x=col, fill=True, ax=ax_density, color="#CCB974")
        ax_density.set_xlabel(col)
        ax_density.set_title(f"{col} - density by global")

        if out_dir is not None:
            _save_figure(fig_hist, out_dir / f"{col}_histogram_global.png")
            _save_figure(fig_density, out_dir / f"{col}_density_global.png")

        results[col] = {"histogram": fig_hist, "density": fig_density}
    return results


def plot_morphology_by_group(
    df: pd.DataFrame,
    morphology_col: str,
    group_col: str,
    save_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """Plot morphology distributions grouped by sample or position."""
    if morphology_col not in MORPHOLOGY_COLS:
        raise ValueError(f"morphology_col must be one of {MORPHOLOGY_COLS}.")
    if group_col not in {SAMPLE_COL, POSITION_COL}:
        raise ValueError(f"group_col must be one of {{'{SAMPLE_COL}', '{POSITION_COL}'}}.")

    out_dir = _ensure_dir(save_dir)

    fig_violin, ax_violin = plt.subplots(figsize=(8, 4.5))
    sns.violinplot(data=df, x=group_col, y=morphology_col, inner="quartile", ax=ax_violin)
    ax_violin.set_xlabel(group_col)
    ax_violin.set_ylabel(morphology_col)
    ax_violin.set_title(f"{morphology_col} - violin by {group_col}")
    ax_violin.tick_params(axis="x", rotation=45)

    fig_box, ax_box = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x=group_col, y=morphology_col, ax=ax_box)
    ax_box.set_xlabel(group_col)
    ax_box.set_ylabel(morphology_col)
    ax_box.set_title(f"{morphology_col} - boxplot by {group_col}")
    ax_box.tick_params(axis="x", rotation=45)

    if out_dir is not None:
        _save_figure(fig_violin, out_dir / f"{morphology_col}_violin_{group_col}.png")
        _save_figure(fig_box, out_dir / f"{morphology_col}_boxplot_{group_col}.png")

    return {"violin": fig_violin, "boxplot": fig_box}


def plot_morphology_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_by: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Create morphology scatter for selected axes with optional color."""
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("x_col and y_col must exist in dataframe.")
    if color_by is not None and color_by not in df.columns:
        raise ValueError("color_by must exist in dataframe when provided.")

    out_dir = _ensure_dir(save_dir)

    fig, ax = plt.subplots(figsize=(7, 5))
    if color_by is None:
        ax.scatter(df[x_col], df[y_col], s=14, alpha=0.7, linewidth=0, color="#4C72B0")
    else:
        is_cat = str(df[color_by].dtype) in ("category", "string") or df[color_by].dtype == "object"
        if is_cat:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=color_by, palette="tab10", s=16, alpha=0.8, ax=ax)
        else:
            points = ax.scatter(df[x_col], df[y_col], c=df[color_by], s=16, alpha=0.8, cmap="plasma", linewidth=0)
            cbar = fig.colorbar(points, ax=ax)
            cbar.set_label(color_by)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}" + (f" colored by {color_by}" if color_by else ""))

    if out_dir is not None:
        color_tag = color_by if color_by else "none"
        _save_figure(fig, out_dir / f"scatter_{x_col}_vs_{y_col}_color_{color_tag}.png")
    return fig


def run_stage2_visualizations(analysis_object: Dict[str, Any], save_dir: Optional[str] = None) -> Dict[str, Any]:
    """Generate all required stage-2 plots and store them in analysis_object."""
    df = analysis_object["cleaned_data"]
    mn = valid_marker_names(analysis_object)
    sc = score_column(analysis_object)
    base_dir = _ensure_dir(save_dir)

    dist_dir = str(base_dir / "distributions") if base_dir else None
    spatial_dir = str(base_dir / "spatial") if base_dir else None
    morph_dir = str(base_dir / "morphology") if base_dir else None

    distribution_plots = plot_all_markers(df, marker_cols=mn, score_col=sc, save_dir=dist_dir)

    spatial_plots: Dict[str, Any] = {
        "sample": plot_spatial(df, color_by=SAMPLE_COL, save_path=f"{spatial_dir}/spatial_sample.png" if spatial_dir else None),
        "position": plot_spatial(df, color_by=POSITION_COL, save_path=f"{spatial_dir}/spatial_position.png" if spatial_dir else None),
    }
    if sc in df.columns:
        pan_key = sc[:-len("_score")] if sc.endswith("_score") else sc
        spatial_plots[pan_key] = plot_spatial(
            df, color_by=sc, save_path=f"{spatial_dir}/spatial_{pan_key}.png" if spatial_dir else None
        )
    for m in mn:
        spatial_plots[m] = plot_spatial(
            df, color_by=m, save_path=f"{spatial_dir}/spatial_{m}.png" if spatial_dir else None
        )

    if mn and "area" in df.columns:
        m0 = mn[0]
        spatial_plots[f"{m0}_size_area"] = plot_spatial(
            df,
            color_by=m0,
            size_by="area",
            save_path=f"{spatial_dir}/spatial_{m0}_size_area.png" if spatial_dir else None,
        )

    morph_distributions = plot_morphology_distributions(df, save_dir=morph_dir)

    morph_by_group: Dict[str, Dict[str, plt.Figure]] = {}
    for morph_col in MORPHOLOGY_COLS:
        for grp in (SAMPLE_COL, POSITION_COL):
            key = f"{morph_col}_by_{grp}"
            morph_by_group[key] = plot_morphology_by_group(df, morph_col, grp, save_dir=morph_dir)

    morph_scatter: Dict[str, plt.Figure] = {}
    morph_scatter["area_vs_circularity_by_sample"] = plot_morphology_scatter(
        df,
        x_col="area",
        y_col="circularity",
        color_by=SAMPLE_COL,
        save_dir=morph_dir,
    )
    for marker in mn:
        morph_scatter[f"area_vs_{marker}"] = plot_morphology_scatter(
            df,
            x_col="area",
            y_col=marker,
            color_by=None,
            save_dir=morph_dir,
        )
        morph_scatter[f"circularity_vs_{marker}"] = plot_morphology_scatter(
            df,
            x_col="circularity",
            y_col=marker,
            color_by=None,
            save_dir=morph_dir,
        )

    analysis_object["marker_analysis"] = {
        "distribution_plots": distribution_plots,
        "spatial_plots": spatial_plots,
        "morphology_plots": {
            "distributions": morph_distributions,
            "by_group": morph_by_group,
            "scatter": morph_scatter,
        },
    }
    return analysis_object


if __name__ == "__main__":
    ao = load_cygnus_object("all_cells.csv")
    ao = run_stage2_visualizations(ao, save_dir="./output/stage2/")
    print("Stage 2 visualizations completed and stored in analysis_object['marker_analysis'].")
    print(f"Score column: {score_column(ao)} | Marker bases: {', '.join(marker_base_columns(ao))}")
    print(f"Valid markers (analysis): {', '.join(valid_marker_names(ao))}")
