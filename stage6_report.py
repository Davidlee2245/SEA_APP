"""Stage 6: sample comparison, position QC, export, and HTML reporting."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Template
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from stage1_loader import ID_COL, MORPHOLOGY_COLS, POSITION_COL, SAMPLE_COL, load_cygnus_object, marker_base_columns, score_column, valid_marker_names
from stage2_visualization import (
    plot_marker_distributions,
    plot_morphology_by_group,
    run_stage2_visualizations,
)
from stage3_preprocessing import (
    apply_qc_filters,
    binarize_markers,
    normalize_by_panev,
    plot_threshold_preview,
    scale_expression_matrix,
)
from stage4_colocalization import (
    REFERENCE_COLOC_COLUMNS,
    compute_colocalization,
    compute_reference_colocalization,
    plot_expression_heatmap,
    plot_upset,
    plot_volcano,
)
from stage5_dimred_clustering import (
    plot_clusters,
    plot_dim_red,
    plot_pca_scree,
    run_dim_reduction,
    run_hdbscan,
    run_kmeans,
    run_leiden,
)


sns.set_theme(style="whitegrid")

_LOG = logging.getLogger(__name__)

# Columns written by compare_samples for pairwise_tests (used when table is empty).
PAIRWISE_TEST_COLUMNS = [
    "marker",
    "sample_1",
    "sample_2",
    "U_statistic",
    "pvalue",
    "effect_size_rbc",
    "adj_pvalue",
    "significance",
    "sample_pair",
]


def _sig_label(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


def _get_matrix(ao: Dict[str, Any], input_matrix: str) -> pd.DataFrame:
    matrix_map = {
        "raw": ao["matrices"]["Raw_Score"],
        "normalized": ao["matrices"]["normalized_exp_mat"],
        "scaled": ao["matrices"]["scaled_exp_matrix"],
    }
    if input_matrix not in matrix_map:
        raise ValueError("input_matrix must be one of {'raw', 'normalized', 'scaled'}.")
    mat = matrix_map[input_matrix]
    if mat is None:
        raise ValueError(f"Input matrix '{input_matrix}' is None.")
    return mat


def compare_samples(
    ao: dict,
    input_matrix: str = "normalized",
    markers: Optional[List[str]] = None,
    correction_method: str = "fdr_bh",
) -> dict:
    """Compare marker expression across sample groups."""
    if correction_method not in {"bonferroni", "fdr_bh"}:
        raise ValueError("correction_method must be 'bonferroni' or 'fdr_bh'.")

    mat = _get_matrix(ao, input_matrix=input_matrix)
    if markers is not None:
        marker_list = markers
    else:
        marker_list = [m for m in valid_marker_names(ao) if m in mat.columns]
    meta = ao["cleaned_data"][[ID_COL, SAMPLE_COL]].copy()
    data = meta.merge(mat.reset_index(), on=ID_COL, how="inner")
    samples = [str(x) for x in sorted(data[SAMPLE_COL].dropna().unique())]

    summary_rows = []
    kw_rows = []
    pair_rows = []

    for marker in marker_list:
        grouped = data[[SAMPLE_COL, marker]].dropna().groupby(SAMPLE_COL)[marker]
        desc = grouped.describe(percentiles=[0.25, 0.75]).reset_index()
        for _, row in desc.iterrows():
            summary_rows.append(
                {
                    "marker": marker,
                    "sample": row[SAMPLE_COL],
                    "count": row["count"],
                    "mean": row["mean"],
                    "median": row["50%"],
                    "std": row["std"],
                    "min": row["min"],
                    "q25": row["25%"],
                    "q75": row["75%"],
                    "max": row["max"],
                }
            )

        values_by_sample = [data.loc[data[SAMPLE_COL] == s, marker].dropna().values for s in samples]
        valid_groups = [vals for vals in values_by_sample if len(vals) > 0]
        if len(valid_groups) >= 2:
            h_stat, pval = kruskal(*valid_groups)
        else:
            h_stat, pval = np.nan, np.nan
        kw_rows.append({"marker": marker, "H_statistic": h_stat, "pvalue": pval})

        local_pair_rows = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                s1, s2 = samples[i], samples[j]
                x = data.loc[data[SAMPLE_COL] == s1, marker].dropna().values
                y = data.loc[data[SAMPLE_COL] == s2, marker].dropna().values
                if len(x) == 0 or len(y) == 0:
                    u_stat, p_pair = np.nan, np.nan
                else:
                    u_stat, p_pair = mannwhitneyu(x, y, alternative="two-sided")
                # rank-biserial from U
                if len(x) > 0 and len(y) > 0 and not np.isnan(u_stat):
                    effect = 2 * (u_stat / (len(x) * len(y))) - 1
                else:
                    effect = np.nan
                local_pair_rows.append(
                    {
                        "marker": marker,
                        "sample_1": s1,
                        "sample_2": s2,
                        "U_statistic": u_stat,
                        "pvalue": p_pair,
                        "effect_size_rbc": effect,
                    }
                )
        if local_pair_rows:
            pvals = [row["pvalue"] if pd.notna(row["pvalue"]) else 1.0 for row in local_pair_rows]
            _, adj_p, _, _ = multipletests(pvals, method=correction_method)
            for row, ap in zip(local_pair_rows, adj_p):
                row["adj_pvalue"] = float(ap)
                row["significance"] = _sig_label(float(ap))
                row["sample_pair"] = f"{row['sample_1']} vs {row['sample_2']}"
            pair_rows.extend(local_pair_rows)

    summary_df = pd.DataFrame(summary_rows)
    kw_df = pd.DataFrame(kw_rows)
    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        pair_df = pd.DataFrame(columns=PAIRWISE_TEST_COLUMNS)

    ao.setdefault("marker_analysis", {})
    ao["marker_analysis"]["sample_comparison"] = {
        "summary_table": summary_df,
        "kruskal_wallis": kw_df,
        "pairwise_tests": pair_df,
    }
    return ao


def _placeholder_sample_comparison_fig(title: str, message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10, transform=ax.transAxes, wrap=True)
    return fig


def plot_sample_comparison(ao: dict, marker: str, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """Plot violin/box and pairwise significance matrix for one marker."""
    sc = ao.get("marker_analysis", {}).get("sample_comparison")
    if sc is None:
        raise ValueError("sample_comparison not found. Run compare_samples first.")

    mat = ao["matrices"]["normalized_exp_mat"]
    if mat is None or marker not in mat.columns:
        raise ValueError(f"Marker '{marker}' not found in normalized matrix.")

    df = ao["cleaned_data"][[ID_COL, SAMPLE_COL]].merge(mat[[marker]].reset_index(), on=ID_COL, how="inner")
    out = Path(save_dir) if save_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    fig_violin, ax_v = plt.subplots(figsize=(8, 4.5))
    sns.violinplot(data=df, x=SAMPLE_COL, y=marker, inner="quartile", ax=ax_v)
    if len(df) < 500:
        sns.stripplot(data=df, x=SAMPLE_COL, y=marker, color="black", alpha=0.25, size=2, ax=ax_v)
    ax_v.set_title(f"{marker} by sample (violin)")
    ax_v.tick_params(axis="x", rotation=45)

    fig_box, ax_b = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x=SAMPLE_COL, y=marker, ax=ax_b)
    ax_b.set_title(f"{marker} by sample (boxplot)")
    ax_b.tick_params(axis="x", rotation=45)

    samples = sorted(df[SAMPLE_COL].astype(str).unique())
    pair = sc.get("pairwise_tests")
    if not isinstance(pair, pd.DataFrame):
        pair = pd.DataFrame(columns=PAIRWISE_TEST_COLUMNS)
    pairwise_ok = not pair.empty and "marker" in pair.columns
    if not pairwise_ok:
        _LOG.warning(
            "plot_sample_comparison(%r): pairwise_tests empty or missing 'marker' column "
            "(common cause: only one sample level, so no sample–sample pairs). Using placeholder p-value heatmap.",
            marker,
        )
        marker_pair = pd.DataFrame(columns=PAIRWISE_TEST_COLUMNS)
        fig_hm = _placeholder_sample_comparison_fig(
            f"{marker} pairwise adjusted p-values",
            "No pairwise sample comparisons (need at least two samples with data).",
        )
    else:
        marker_pair = pair[pair["marker"] == marker].copy()
        pmat = pd.DataFrame(np.nan, index=samples, columns=samples)
        np.fill_diagonal(pmat.values, 0.0)
        for _, row in marker_pair.iterrows():
            s1, s2 = str(row["sample_1"]), str(row["sample_2"])
            pmat.loc[s1, s2] = row["adj_pvalue"]
            pmat.loc[s2, s1] = row["adj_pvalue"]

        fig_hm, ax_hm = plt.subplots(figsize=(6, 5))
        sns.heatmap(pmat, cmap="viridis_r", annot=True, fmt=".2g", cbar_kws={"label": "adjusted p-value"}, ax=ax_hm)
        ax_hm.set_title(f"{marker} pairwise adjusted p-values")

    # Add simple bracket labels for significant pairs.
    if marker_pair.empty or "significance" not in marker_pair.columns:
        sig_pairs = pd.DataFrame()
    else:
        sig_pairs = marker_pair[marker_pair["significance"] != "ns"]
    if not sig_pairs.empty:
        ymax = float(df[marker].max())
        y_offset = (float(df[marker].max()) - float(df[marker].min()) + 1e-9) * 0.06
        sample_to_x = {s: i for i, s in enumerate(samples)}
        for idx, (_, row) in enumerate(sig_pairs.iterrows()):
            x1 = sample_to_x[str(row["sample_1"])]
            x2 = sample_to_x[str(row["sample_2"])]
            y = ymax + y_offset * (idx + 1)
            ax_b.plot([x1, x1, x2, x2], [y, y + y_offset * 0.2, y + y_offset * 0.2, y], color="black", linewidth=0.8)
            ax_b.text((x1 + x2) / 2, y + y_offset * 0.25, row["significance"], ha="center", va="bottom", fontsize=8)

    if out:
        fig_violin.savefig(out / f"{marker}_sample_violin.png", dpi=200, bbox_inches="tight")
        fig_box.savefig(out / f"{marker}_sample_boxplot.png", dpi=200, bbox_inches="tight")
        fig_hm.savefig(out / f"{marker}_pairwise_pvalue_heatmap.png", dpi=200, bbox_inches="tight")
    return {"violin": fig_violin, "boxplot": fig_box, "pairwise_heatmap": fig_hm}


def plot_pairwise_significance_heatmap(
    pairwise_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Plot marker x sample-pair significance matrix using -log10(adj p-value)."""
    if pairwise_df is None or pairwise_df.empty:
        raise ValueError("pairwise_df is empty.")
    pivot = pairwise_df.pivot_table(
        index="marker",
        columns="sample_pair",
        values="adj_pvalue",
        aggfunc="first",
    )
    log_pivot = -np.log10(pivot.clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(max(6, len(log_pivot.columns)), max(4, len(log_pivot.index))))
    sns.heatmap(
        log_pivot,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "-log10(adj p-value)"},
    )
    ax.set_title("Pairwise Comparison Significance (-log10 adj p-value)")
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def run_position_qc(ao: dict, save_dir: Optional[str] = None) -> dict:
    """Run position-level QC summaries and generate plots."""
    df = ao["cleaned_data"]
    out = Path(save_dir) if save_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    count_by_position = df.groupby(POSITION_COL).size().rename("object_count").reset_index()
    count_sample_position = df.groupby([SAMPLE_COL, POSITION_COL]).size().rename("object_count").reset_index()

    fig_count, ax_count = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=count_by_position, x=POSITION_COL, y="object_count", ax=ax_count)
    ax_count.set_title("Object count by position")
    ax_count.tick_params(axis="x", rotation=45)

    pivot_sp = count_sample_position.pivot(index=POSITION_COL, columns=SAMPLE_COL, values="object_count").fillna(0)
    fig_sp, ax_sp = plt.subplots(figsize=(10, 5))
    pivot_sp.plot(kind="bar", stacked=True, ax=ax_sp, colormap="tab20")
    ax_sp.set_title("Object count by sample and position")
    ax_sp.set_ylabel("Count")
    ax_sp.tick_params(axis="x", rotation=45)

    marker_list = valid_marker_names(ao)
    marker_means = df.groupby(POSITION_COL)[marker_list].mean()
    morph_means = df.groupby(POSITION_COL)[MORPHOLOGY_COLS].mean()

    if out:
        fig_count.savefig(out / "object_count_by_position.png", dpi=200, bbox_inches="tight")
        fig_sp.savefig(out / "object_count_by_sample_position.png", dpi=200, bbox_inches="tight")

    marker_violin_by_position: Dict[str, plt.Figure] = {}
    marker_box_by_position: Dict[str, plt.Figure] = {}
    # Per-marker distribution plots by position
    for marker in marker_list:
        violin_fig = plot_marker_distributions(
            df,
            marker=marker,
            group_col=POSITION_COL,
            plot_type="violin",
            save_dir=str(out) if out else None,
            valid_markers=marker_list,
        )
        box_fig = plot_marker_distributions(
            df,
            marker=marker,
            group_col=POSITION_COL,
            plot_type="boxplot",
            save_dir=str(out) if out else None,
            valid_markers=marker_list,
        )
        if "violin" in violin_fig:
            marker_violin_by_position[marker] = violin_fig["violin"]
        if "boxplot" in box_fig:
            marker_box_by_position[marker] = box_fig["boxplot"]

    morphology_by_position_figs: Dict[str, Dict[str, plt.Figure]] = {}
    for morph_col in MORPHOLOGY_COLS:
        morphology_by_position_figs[morph_col] = plot_morphology_by_group(
            df, morphology_col=morph_col, group_col=POSITION_COL, save_dir=str(out) if out else None
        )

    fig_hm, ax_hm = plt.subplots(figsize=(8, 5))
    sns.heatmap(marker_means, cmap="viridis", linewidths=0.2, ax=ax_hm)
    ax_hm.set_title("Average marker expression by position")
    if out:
        fig_hm.savefig(out / "avg_marker_expression_by_position.png", dpi=200, bbox_inches="tight")

    ao.setdefault("marker_analysis", {})
    dimred_by_position: Dict[str, Any] = {}
    for method in ["pca", "tsne", "umap"]:
        if ao["dim_red"].get(method) is not None:
            dimred_by_position[method] = plot_dim_red(
                ao,
                method=method,
                color_by=POSITION_COL,
                interactive=False,
                save_path=str(out / f"{method}_by_position.png") if out else None,
            )

    ao["marker_analysis"]["position_qc"] = {
        "object_counts": count_sample_position,
        "marker_by_position": marker_means.reset_index(),
        "morphology_by_position": morph_means.reset_index(),
        "object_count_by_position_fig": fig_count,
        "object_count_by_sample_position_fig": fig_sp,
        "avg_marker_expression_by_position_fig": fig_hm,
        "marker_violin_by_position": marker_violin_by_position,
        "marker_box_by_position": marker_box_by_position,
        "morphology_by_position_figs": morphology_by_position_figs,
        "dimred_by_position": dimred_by_position,
    }
    return ao


def export_all(ao: dict, output_dir: str = "./output/exports/") -> Dict[str, str]:
    """Export all major data artifacts and return path map."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    export_paths: Dict[str, str] = {}

    def save_table(label: str, stem: str, df: Optional[pd.DataFrame]) -> None:
        if df is None:
            return
        csv_path = os.path.join(str(out), f"{stem}.csv")
        xlsx_path = os.path.join(str(out), f"{stem}.xlsx")
        df.to_csv(csv_path, index=False)
        df.to_excel(xlsx_path, index=False, engine="openpyxl")
        export_paths[f"{label} (CSV)"] = csv_path
        export_paths[f"{label} (XLSX)"] = xlsx_path

    save_table("Cleaned data", "cleaned_data", ao.get("cleaned_data"))
    save_table("Filtered data", "filtered_data", ao.get("filtered_data"))
    save_table(
        "Raw marker matrix",
        "raw_score_matrix",
        ao["matrices"].get("Raw_Score").reset_index() if ao["matrices"].get("Raw_Score") is not None else None,
    )
    save_table(
        "Normalized marker matrix",
        "normalized_exp_mat",
        ao["matrices"].get("normalized_exp_mat").reset_index() if ao["matrices"].get("normalized_exp_mat") is not None else None,
    )
    save_table(
        "Scaled marker matrix",
        "scaled_exp_matrix",
        ao["matrices"].get("scaled_exp_matrix").reset_index() if ao["matrices"].get("scaled_exp_matrix") is not None else None,
    )
    save_table(
        "Binary marker matrix",
        "binary_exp_matrix",
        ao["matrices"].get("binary_exp_matrix").reset_index() if ao["matrices"].get("binary_exp_matrix") is not None else None,
    )
    save_table("Morphology table", "morphology_table", ao["cleaned_data"][[ID_COL] + MORPHOLOGY_COLS])
    save_table("Metadata table", "metadata_table", ao["cleaned_data"][[ID_COL, SAMPLE_COL, POSITION_COL]])
    save_table("Threshold table", "threshold_table", ao.get("threshold_table"))

    col = ao.get("colocalization") or {}
    save_table("Colocalization results", "colocalization_all", col.get("all_combinations"))
    for ref, ref_df in col.get("reference_centered", {}).items():
        save_table(f"Reference-centered colocalization ({ref})", f"colocalization_reference_{ref}", ref_df)

    save_table("PCA coordinates", "dim_red_pca", ao["dim_red"].get("pca"))
    save_table("t-SNE coordinates", "dim_red_tsne", ao["dim_red"].get("tsne"))
    save_table("UMAP coordinates", "dim_red_umap", ao["dim_red"].get("umap"))

    labels = pd.DataFrame({ID_COL: ao["cleaned_data"][ID_COL]})
    for name in ("kmeans", "hdbscan", "leiden"):
        s = ao.get("ev_meta", {}).get("cluster_labels", {}).get(name)
        if s is not None:
            labels[name] = labels[ID_COL].map(s)
    save_table("Clustering results", "clustering_labels", labels)

    sc = ao.get("marker_analysis", {}).get("sample_comparison", {})
    save_table("Sample comparison summary", "sample_comparison_summary", sc.get("summary_table"))
    save_table("Sample pairwise tests", "sample_pairwise_tests", sc.get("pairwise_tests"))
    save_table("Position QC summary", "position_qc_summary", ao.get("marker_analysis", {}).get("position_qc", {}).get("object_counts"))

    normalized = ao["matrices"].get("normalized_exp_mat")
    if normalized is not None:
        meta = ao["cleaned_data"][[ID_COL, SAMPLE_COL, POSITION_COL]]
        merged = meta.merge(normalized.reset_index(), on=ID_COL, how="inner")
        mn = valid_marker_names(ao)
        marker_cols = [c for c in normalized.columns if c in mn]
        avg_sample = merged.groupby(SAMPLE_COL)[marker_cols].mean().reset_index()
        avg_position = merged.groupby(POSITION_COL)[marker_cols].mean().reset_index()
        save_table("Average expression by sample", "avg_expression_by_sample", avg_sample)
        save_table("Average expression by position", "avg_expression_by_position", avg_position)

    pkl_path = out / "analysis_object.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(ao, f)
    export_paths["Pickle analysis object"] = str(pkl_path)

    summary_json = {
        "total_objects": int(len(ao["cleaned_data"])),
        "samples": sorted([str(x) for x in ao["cleaned_data"][SAMPLE_COL].astype(str).unique()]),
        "positions": sorted([str(x) for x in ao["cleaned_data"][POSITION_COL].astype(str).unique()]),
        "markers": list(marker_base_columns(ao)),
        "valid_markers": list(valid_marker_names(ao)),
        "score_col": score_column(ao),
        "threshold_table": ao["threshold_table"].to_dict(orient="records") if ao.get("threshold_table") is not None else [],
        "export_paths": export_paths,
    }
    json_path = out / "analysis_summary.json"
    json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    export_paths["JSON metadata summary"] = str(json_path)

    return export_paths


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _plotly_to_html(fig, include_plotlyjs: bool) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if include_plotlyjs else False)


def _df_to_html(df: Optional[pd.DataFrame], max_rows: int = 30) -> str:
    if df is None:
        return "<p>Not available.</p>"
    return df.head(max_rows).to_html(index=False, classes="table table-sm")


def _wrap_plot_grid(parts: List[str]) -> str:
    """Wrap sequential plot HTML fragments in a two-column grid; skip if empty."""
    if not parts:
        return ""
    return '<div class="plot-grid">' + "".join(parts) + "</div>"


def generate_report(
    ao: dict,
    export_paths: Dict[str, str],
    output_path: str = "./output/cygnus_report.html",
):
    """Generate offline HTML report with embedded assets."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = ao["cleaned_data"]
    marker_analysis = ao.get("marker_analysis", {})
    norm = ao["matrices"].get("normalized_exp_mat")
    plotly_used = False

    def embed_fig(fig):
        nonlocal plotly_used
        if fig is None:
            return "<p>Not available.</p>"
        if hasattr(fig, "to_html"):
            html = _plotly_to_html(fig, include_plotlyjs=not plotly_used)
            plotly_used = True
            return html
        return f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" class="img"/>'

    # Ensure key figures exist.
    morph_dist = marker_analysis.get("morphology_plots", {}).get("distributions", {})
    distribution_plots = marker_analysis.get("distribution_plots", {})
    spatial_plots = marker_analysis.get("spatial_plots", {})
    col_plots = marker_analysis.get("colocalization_plots", {})
    dimred_plots = marker_analysis.get("dimred_plots", {})
    cluster_plots = marker_analysis.get("cluster_plots", {})

    low_flag = ao.get("ev_meta", {}).get("low_panev_flag")
    low_frac = float(np.mean(low_flag)) if low_flag is not None and len(low_flag) > 0 else np.nan

    top_coloc = (ao.get("colocalization") or {}).get("all_combinations")
    top_coloc = top_coloc.sort_values("fdr_pvalue").head(20) if isinstance(top_coloc, pd.DataFrame) and not top_coloc.empty else None
    ref_centered = (ao.get("colocalization") or {}).get("reference_centered") or {}
    ref_first_key = next(iter(ref_centered.keys()), None)
    ref_centered_df = ref_centered.get(ref_first_key) if ref_first_key else None

    sc = marker_analysis.get("sample_comparison", {})
    sample_kw = sc.get("kruskal_wallis")
    pairwise_tests = sc.get("pairwise_tests")
    position_qc = marker_analysis.get("position_qc", {})

    scol = score_column(ao)
    mnames = valid_marker_names(ao)
    panev_stats = {}
    if scol in df.columns:
        panev_stats = {
            "min": float(df[scol].min()),
            "max": float(df[scol].max()),
            "mean": float(df[scol].mean()),
        }

    template = Template(
        """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Single EV Analysis Report</title>
  <style>
    /* All rules scoped under .cygnus-report — standalone file uses body.cygnus-report; SPA mounts a div.cygnus-report */
    .cygnus-report {
      font-family: Arial, sans-serif;
      margin: 0;
      display: flex;
      box-sizing: border-box;
    }
    .cygnus-report *,
    .cygnus-report *::before,
    .cygnus-report *::after {
      box-sizing: inherit;
    }
    .cygnus-report nav {
      width: 260px;
      position: sticky;
      top: 0;
      align-self: flex-start;
      max-height: 100vh;
      overflow-y: auto;
      background: #f6f6f8;
      padding: 16px;
      border-right: 1px solid #ddd;
      flex-shrink: 0;
    }
    .cygnus-report nav a {
      display: block;
      margin: 6px 0;
      color: #333;
      text-decoration: none;
    }
    .cygnus-report main {
      flex: 1;
      padding: 20px;
      min-width: 0;
    }
    .cygnus-report section {
      margin-bottom: 36px;
    }
    .cygnus-report h1,
    .cygnus-report h2 {
      margin: 0 0 10px 0;
    }
    .cygnus-report .img {
      max-width: 100%;
      margin: 8px 0;
      border: 1px solid #ddd;
    }
    .cygnus-report table {
      border-collapse: collapse;
      width: 100%;
      margin: 8px 0;
      font-size: 12px;
    }
    .cygnus-report th,
    .cygnus-report td {
      border: 1px solid #ddd;
      padding: 6px;
      text-align: left;
    }
    .cygnus-report th {
      background: #f1f1f1;
    }
    .cygnus-report .muted {
      color: #666;
      font-size: 12px;
    }
    /* Plot images — ~half width */
    .cygnus-report img.img {
      width: 100% !important;
      max-width: 100% !important;
      height: auto !important;
      display: block;
    }
    .cygnus-report .plotly-graph-div {
      width: 100% !important;
      max-width: 100% !important;
    }
    .cygnus-report .plot-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      align-items: start;
      margin-bottom: 24px;
    }
    .cygnus-report .plot-grid > * {
      min-width: 0;
    }
  </style>
</head>
<body class="cygnus-report">
  <nav>
    <h3>Sections</h3>
    {% for sec in sections %}<a href="#{{ sec.id }}">{{ sec.title }}</a>{% endfor %}
  </nav>
  <main>
    <section id="header">
      <h1>Single EV Analysis Report</h1>
      <p class="muted">Generated: {{ generated_at }} | Input file: {{ input_file }}</p>
    </section>
    <section id="data-summary"><h2>Data Summary</h2>
      <p>Total objects: {{ total_objects }}</p>
      <p>Samples: {{ samples }}</p><p>Positions: {{ positions }}</p>
      <p>Markers: {{ markers }}</p><p>Morphology columns: {{ morphology }}</p>
      <p>Applied QC filters: {{ qc_filters }}</p>
    </section>
    <section id="morphology-qc"><h2>Morphology QC</h2>{{ morphology_qc|safe }}</section>
    <section id="marker-distributions"><h2>Marker Intensity Distributions</h2>{{ marker_dist|safe }}</section>
    <section id="spatial"><h2>Spatial Visualization</h2>{{ spatial|safe }}</section>
    <section id="panev"><h2>PanEV Normalization</h2>
      <p>PanEV min/max/mean: {{ panev_stats }}</p>
      <p>Low PanEV flagged fraction: {{ low_frac }}</p>
      {{ panev_before_after|safe }}
    </section>
    <section id="thresholding"><h2>Thresholding Summary</h2>{{ thresholding|safe }}</section>
    <section id="colocalization"><h2>Colocalization</h2>{{ colocalization|safe }}</section>
    <section id="ref-colocalization"><h2>Reference-Centered Colocalization</h2>{{ ref_colocalization|safe }}</section>
    <section id="heatmaps"><h2>Average Expression Heatmaps</h2>{{ heatmaps|safe }}</section>
    <section id="dim-red"><h2>Dimensionality Reduction</h2>{{ dimred|safe }}</section>
    <section id="clustering"><h2>Clustering</h2>{{ clustering|safe }}</section>
    <section id="sample-comparison"><h2>Sample-Level Comparison</h2>{{ sample_comparison|safe }}</section>
    <section id="position-qc"><h2>Position-Level QC</h2>{{ position_qc|safe }}</section>
    <section id="exports"><h2>Exported Files</h2>{{ exported_files|safe }}</section>
    <footer><p class="muted">Generated by SEA Analysis Pipeline</p></footer>
  </main>
</body>
</html>"""
    )

    morphology_html = []
    for col in MORPHOLOGY_COLS:
        block = morph_dist.get(col, {})
        for key in ("histogram", "density"):
            if key in block:
                morphology_html.append(embed_fig(block[key]))
        # Add morphology by sample and by position breakdowns for report completeness.
        sample_group = plot_morphology_by_group(df, morphology_col=col, group_col=SAMPLE_COL)
        position_group = plot_morphology_by_group(df, morphology_col=col, group_col=POSITION_COL)
        for fig in sample_group.values():
            morphology_html.append(embed_fig(fig))
        for fig in position_group.values():
            morphology_html.append(embed_fig(fig))

    marker_dist_html = []
    for marker in [scol] + mnames:
        mplots = distribution_plots.get(marker, {})
        for key in ("histogram", "density", "violin_by_sample", "violin_by_position"):
            if key in mplots:
                marker_dist_html.append(embed_fig(mplots[key]))

    spatial_html = []
    for k in spatial_plots:
        spatial_html.append(embed_fig(spatial_plots[k]))

    panev_html = ""
    demo_marker = mnames[0] if mnames else None
    if norm is not None and demo_marker is not None and demo_marker in norm.columns and demo_marker in df.columns:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df[demo_marker], kde=True, ax=ax[0])
        ax[0].set_title(f"{demo_marker} before normalization")
        sns.histplot(norm[demo_marker], kde=True, ax=ax[1])
        ax[1].set_title(f"{demo_marker} after normalization")
        panev_html = embed_fig(fig)

    thr_table = _df_to_html(ao.get("threshold_table"))
    thr_plots: List[str] = []
    for marker in mnames:
        try:
            thr_plots.append(embed_fig(plot_threshold_preview(ao, marker)))
        except Exception:
            continue

    coloc_plots_only: List[str] = []
    if "upset" in col_plots:
        coloc_plots_only.append(embed_fig(col_plots["upset"]))
    if "volcano" in col_plots:
        coloc_plots_only.append(embed_fig(col_plots["volcano"]))
    colocalization_block = _wrap_plot_grid(coloc_plots_only) + _df_to_html(top_coloc, max_rows=20)

    ref_coloc_html = _df_to_html(ref_centered_df, max_rows=20)

    heatmap_html = []
    if "heatmap_by_sample" in col_plots:
        heatmap_html.append(embed_fig(col_plots["heatmap_by_sample"]))
    if "heatmap_by_position" in col_plots:
        heatmap_html.append(embed_fig(col_plots["heatmap_by_position"]))

    dimred_html = []
    if "scree" in dimred_plots:
        dimred_html.append(embed_fig(dimred_plots["scree"]))
    for key in ("pca_sample", "pca_position", "umap_sample", "umap_position", "tsne_sample", "tsne_position"):
        if key in dimred_plots:
            dimred_html.append(embed_fig(dimred_plots[key]))
    extra_umap = sorted(k for k in dimred_plots if k.startswith("umap_") and k not in ("umap_sample", "umap_position"))
    for key in extra_umap:
        dimred_html.append(embed_fig(dimred_plots[key]))

    cluster_html = []
    for method in ("kmeans", "hdbscan", "leiden"):
        block = cluster_plots.get(method, {})
        for key in ("cluster_embedding", "composition_sample", "composition_position", "cluster_expression_heatmap"):
            if key in block:
                cluster_html.append(embed_fig(block[key]))

    sc_html = []
    top_markers = mnames[:4]
    for marker in top_markers:
        comp_block = marker_analysis.get("sample_comparison_plots", {}).get(marker, {})
        for key in ("violin", "pairwise_heatmap"):
            if key in comp_block:
                sc_html.append(embed_fig(comp_block[key]))
    if isinstance(pairwise_tests, pd.DataFrame) and not pairwise_tests.empty:
        sc_html.append(embed_fig(plot_pairwise_significance_heatmap(pairwise_tests)))
    sc_html.append(_df_to_html(sample_kw))

    pos_plots: List[str] = []
    position_qc_plots = marker_analysis.get("position_qc", {})
    for key in ("object_count_by_position_fig", "object_count_by_sample_position_fig", "avg_marker_expression_by_position_fig"):
        fig = position_qc_plots.get(key)
        if fig is not None:
            pos_plots.append(embed_fig(fig))
    for marker in mnames[:3]:
        fig = position_qc_plots.get("marker_violin_by_position", {}).get(marker)
        if fig is not None:
            pos_plots.append(embed_fig(fig))
    for morph_col in MORPHOLOGY_COLS:
        block = position_qc_plots.get("morphology_by_position_figs", {}).get(morph_col, {})
        if "violin" in block:
            pos_plots.append(embed_fig(block["violin"]))
    for method in ("pca", "tsne", "umap"):
        fig = position_qc_plots.get("dimred_by_position", {}).get(method)
        if fig is not None:
            pos_plots.append(embed_fig(fig))
    pos_table = _df_to_html(position_qc.get("marker_by_position"), max_rows=20)
    position_qc_block = _wrap_plot_grid(pos_plots) + pos_table

    before = len(ao["cleaned_data"])
    after = len(ao["filtered_data"]) if ao.get("filtered_data") is not None else before
    removed = before - after
    removed_pct = (removed / before * 100.0) if before else 0.0
    qc_description = (
        f"Objects before filtering: {before} | After: {after} | "
        f"Removed: {removed} ({removed_pct:.1f}%)"
    )

    exported_table = pd.DataFrame({"label": list(export_paths.keys()), "path": list(export_paths.values())})

    html = template.render(
        sections=[
            {"id": "header", "title": "Header"},
            {"id": "data-summary", "title": "Data Summary"},
            {"id": "morphology-qc", "title": "Morphology QC"},
            {"id": "marker-distributions", "title": "Marker Intensity Distributions"},
            {"id": "spatial", "title": "Spatial Visualization"},
            {"id": "panev", "title": "PanEV Normalization"},
            {"id": "thresholding", "title": "Thresholding Summary"},
            {"id": "colocalization", "title": "Colocalization"},
            {"id": "ref-colocalization", "title": "Reference-Centered Colocalization"},
            {"id": "heatmaps", "title": "Average Expression Heatmaps"},
            {"id": "dim-red", "title": "Dimensionality Reduction"},
            {"id": "clustering", "title": "Clustering"},
            {"id": "sample-comparison", "title": "Sample-Level Comparison"},
            {"id": "position-qc", "title": "Position-Level QC"},
            {"id": "exports", "title": "Exported Files"},
        ],
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        input_file="all_cells.csv",
        total_objects=len(df),
        samples=", ".join(sorted(df[SAMPLE_COL].astype(str).unique())),
        positions=", ".join(sorted(df[POSITION_COL].astype(str).unique())),
        markers=", ".join([scol] + mnames),
        morphology=", ".join(MORPHOLOGY_COLS),
        qc_filters=qc_description,
        morphology_qc=_wrap_plot_grid(morphology_html),
        marker_dist=_wrap_plot_grid(marker_dist_html),
        spatial=_wrap_plot_grid(spatial_html),
        panev_stats=panev_stats,
        low_frac=low_frac,
        panev_before_after=_wrap_plot_grid([panev_html]) if panev_html else "",
        thresholding=thr_table + _wrap_plot_grid(thr_plots),
        colocalization=colocalization_block,
        ref_colocalization=ref_coloc_html,
        heatmaps=_wrap_plot_grid(heatmap_html),
        dimred=_wrap_plot_grid(dimred_html),
        clustering=_wrap_plot_grid(cluster_html),
        sample_comparison=_wrap_plot_grid(sc_html[:-1]) + (sc_html[-1] if sc_html else ""),
        position_qc=position_qc_block,
        exported_files=_df_to_html(exported_table, max_rows=300),
    )
    out.write_text(html, encoding="utf-8")


def _run_stage5_defaults(ao: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    mn = valid_marker_names(ao)
    run_dim_reduction(ao, method="pca", input_matrix="scaled", markers=mn, n_components=2)
    run_dim_reduction(ao, method="tsne", input_matrix="scaled", markers=mn, n_components=2)
    run_dim_reduction(ao, method="umap", input_matrix="scaled", markers=mn, n_components=2)
    run_kmeans(ao, n_clusters=5, input_space="umap")
    run_hdbscan(ao, min_cluster_size=50, min_samples=5, input_space="umap")
    run_leiden(ao, resolution=1.0, n_neighbors=15, input_space="umap")

    ao.setdefault("marker_analysis", {})
    sc = score_column(ao)
    dim_plots = {
        "pca_sample": plot_dim_red(ao, "pca", SAMPLE_COL, interactive=True),
        "pca_position": plot_dim_red(ao, "pca", POSITION_COL, interactive=True),
        "umap_sample": plot_dim_red(ao, "umap", SAMPLE_COL, interactive=True),
        "umap_position": plot_dim_red(ao, "umap", POSITION_COL, interactive=True),
        "tsne_sample": plot_dim_red(ao, "tsne", SAMPLE_COL, interactive=True),
        "tsne_position": plot_dim_red(ao, "tsne", POSITION_COL, interactive=True),
        "scree": plot_pca_scree(ao),
    }
    for col in [sc] + mn[: min(3, len(mn))]:
        if col not in ao["cleaned_data"].columns:
            continue
        safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(col))
        dim_plots[f"umap_{safe}"] = plot_dim_red(ao, "umap", col, interactive=True)

    ao["marker_analysis"]["dimred_plots"] = dim_plots
    ao["marker_analysis"]["cluster_plots"] = {
        "kmeans": plot_clusters(ao, "kmeans", dim_red="umap", interactive=True, save_dir=str(out_dir / "kmeans")),
        "hdbscan": plot_clusters(ao, "hdbscan", dim_red="umap", interactive=True, save_dir=str(out_dir / "hdbscan")),
        "leiden": plot_clusters(ao, "leiden", dim_red="umap", interactive=True, save_dir=str(out_dir / "leiden")),
    }
    return ao


if __name__ == "__main__":
    output_root = Path("./output")
    output_root.mkdir(parents=True, exist_ok=True)

    ao = load_cygnus_object("all_cells.csv")
    ao = run_stage2_visualizations(ao, save_dir=str(output_root / "stage2"))
    ao = apply_qc_filters(ao, remove_zero_area=True, remove_zero_perimeter=True, remove_nan_circularity=True)
    ao = normalize_by_panev(ao, use_filtered=False, epsilon=1e-6, include_panev=False)
    ao = scale_expression_matrix(ao, method="zscore", input_matrix="normalized")
    ao = binarize_markers(ao, method="percentile", percentile=95.0, input_matrix="normalized")

    ao = compute_colocalization(ao)
    vm4 = valid_marker_names(ao)
    if len(vm4) >= 2:
        compute_reference_colocalization(ao, reference_marker=vm4[0])
    elif len(vm4) == 1:
        if ao.get("colocalization") is None:
            ao["colocalization"] = {}
        ao["colocalization"].setdefault("reference_centered", {})
        ao["colocalization"]["reference_centered"][vm4[0]] = pd.DataFrame(columns=REFERENCE_COLOC_COLUMNS)
    else:
        if ao.get("colocalization") is None:
            ao["colocalization"] = {}
        ao["colocalization"].setdefault("reference_centered", {})
    ao.setdefault("marker_analysis", {})
    ao["marker_analysis"]["colocalization_plots"] = {
        "upset": plot_upset(ao),
        "volcano": plot_volcano(ao, interactive=False),
        "heatmap_by_sample": plot_expression_heatmap(ao, group_by="sample", input_matrix="normalized"),
        "heatmap_by_position": plot_expression_heatmap(ao, group_by="position", input_matrix="normalized"),
    }

    ao = _run_stage5_defaults(ao, output_root / "stage5")
    ao = compare_samples(ao, input_matrix="normalized")
    ao["marker_analysis"]["sample_comparison_plots"] = {
        m: plot_sample_comparison(ao, m, save_dir=str(output_root / "stage6" / "sample_comparison")) for m in valid_marker_names(ao)
    }

    ao = run_position_qc(ao, save_dir=str(output_root / "stage6" / "position_qc"))

    export_paths = export_all(ao, output_dir=str(output_root / "exports"))
    generate_report(ao, export_paths, output_path=str(output_root / "cygnus_report.html"))
    print("Report saved to ./output/cygnus_report.html")
