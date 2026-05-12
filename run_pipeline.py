"""Full Cygnus pipeline orchestration script."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from stage1_loader import load_cygnus_object, score_column, valid_marker_names
from stage2_visualization import run_stage2_visualizations
from stage3_preprocessing import apply_qc_filters, binarize_markers, normalize_by_panev, scale_expression_matrix
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
from stage6_report import compare_samples, export_all, generate_report, plot_sample_comparison, run_position_qc

_LOGGER = logging.getLogger(__name__)


def cygnus_pipeline_log(stage: str, t0: float, detail: str = "") -> None:
    """Emit Cygnus / data-analysis pipeline progress (stdout + logger)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    elapsed = time.perf_counter() - t0
    msg = f"[cygnus] stage={stage!r} elapsed_s={elapsed:.3f} ts={ts}"
    if detail:
        msg += f" | {detail}"
    print(msg, flush=True)
    _LOGGER.info(msg)


def run_full_pipeline(
    filepath: str = "all_cells.csv",
    output_dir: str = "./output/",
    *,
    pipeline_t0: Optional[float] = None,
):
    t0 = pipeline_t0 if pipeline_t0 is not None else time.perf_counter()
    cygnus_pipeline_log("pipeline_enter", t0, detail=f"filepath={filepath!r} output_dir={output_dir!r}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Stage 1
    cygnus_pipeline_log("stage1_load_start", t0)
    ao = load_cygnus_object(filepath)
    n_rows = len(ao.get("cleaned_data", []))
    cygnus_pipeline_log("stage1_load_done", t0, detail=f"rows={n_rows} markers={len(valid_marker_names(ao))}")

    # Stage 2
    cygnus_pipeline_log("stage2_visualization_start", t0)
    ao = run_stage2_visualizations(ao, save_dir=str(output_root / "stage2"))
    cygnus_pipeline_log("stage2_visualization_done", t0)

    # Stage 3
    cygnus_pipeline_log("stage3_preprocess_start", t0)
    ao = apply_qc_filters(ao)
    ao = normalize_by_panev(ao)
    ao = scale_expression_matrix(ao, method="zscore", input_matrix="normalized")
    ao = binarize_markers(ao, method="percentile", percentile=95.0, input_matrix="normalized")
    cygnus_pipeline_log("stage3_preprocess_done", t0, detail=f"rows={len(ao.get('cleaned_data', []))}")

    # Stage 4
    cygnus_pipeline_log("stage4_colocalization_start", t0)
    ao = compute_colocalization(ao)
    vm4 = valid_marker_names(ao)
    if len(vm4) >= 2:
        compute_reference_colocalization(ao, reference_marker=vm4[0])
    elif len(vm4) == 1:
        _LOGGER.warning(
            "Skipping reference-centered colocalization: need at least 2 valid markers; found 1 (%r).",
            vm4[0],
        )
        if ao.get("colocalization") is None:
            ao["colocalization"] = {}
        ao["colocalization"].setdefault("reference_centered", {})
        ao["colocalization"]["reference_centered"][vm4[0]] = pd.DataFrame(columns=REFERENCE_COLOC_COLUMNS)
    else:
        _LOGGER.warning("Skipping reference-centered colocalization: no valid markers.")
        if ao.get("colocalization") is None:
            ao["colocalization"] = {}
        ao.setdefault("colocalization", {}).setdefault("reference_centered", {})
    ao.setdefault("marker_analysis", {})
    ao["marker_analysis"]["colocalization_plots"] = {
        "upset": plot_upset(ao, save_path=str(output_root / "stage4" / "upset_plot.png")),
        "volcano": plot_volcano(ao, save_path=str(output_root / "stage4" / "volcano_plot.png"), interactive=False),
        "heatmap_by_sample": plot_expression_heatmap(
            ao, group_by="sample", input_matrix="normalized", save_path=str(output_root / "stage4" / "heatmap_by_sample.png")
        ),
        "heatmap_by_position": plot_expression_heatmap(
            ao, group_by="position", input_matrix="normalized", save_path=str(output_root / "stage4" / "heatmap_by_position.png")
        ),
    }
    cygnus_pipeline_log("stage4_colocalization_done", t0)

    # Stage 5 — dimensionality reduction
    cygnus_pipeline_log("dimred_pca_start", t0)
    run_dim_reduction(ao, method="pca")
    cygnus_pipeline_log("dimred_pca_done", t0)

    cygnus_pipeline_log("dimred_tsne_start", t0)
    run_dim_reduction(ao, method="tsne")
    cygnus_pipeline_log("dimred_tsne_done", t0)

    cygnus_pipeline_log("dimred_umap_start", t0)
    run_dim_reduction(ao, method="umap")
    cygnus_pipeline_log("dimred_umap_done", t0)

    cygnus_pipeline_log("cluster_kmeans_start", t0)
    ao = run_kmeans(ao, n_clusters=5)
    cygnus_pipeline_log("cluster_kmeans_done", t0)

    cygnus_pipeline_log("cluster_hdbscan_start", t0)
    ao = run_hdbscan(ao)
    cygnus_pipeline_log("cluster_hdbscan_done", t0)

    cygnus_pipeline_log("cluster_leiden_start", t0)
    ao = run_leiden(ao)
    cygnus_pipeline_log("cluster_leiden_done", t0)

    mn = valid_marker_names(ao)
    sc = score_column(ao)
    cygnus_pipeline_log("plots_dimred_cluster_start", t0, detail="interactive dimred + cluster figures")
    dimred_plots = {
        "pca_sample": plot_dim_red(ao, "pca", "sample", interactive=True, save_path=str(output_root / "stage5" / "pca_sample.html")),
        "pca_position": plot_dim_red(ao, "pca", "position", interactive=True, save_path=str(output_root / "stage5" / "pca_position.html")),
        "umap_sample": plot_dim_red(ao, "umap", "sample", interactive=True, save_path=str(output_root / "stage5" / "umap_sample.html")),
        "umap_position": plot_dim_red(ao, "umap", "position", interactive=True, save_path=str(output_root / "stage5" / "umap_position.html")),
        "tsne_sample": plot_dim_red(ao, "tsne", "sample", interactive=True, save_path=str(output_root / "stage5" / "tsne_sample.html")),
        "tsne_position": plot_dim_red(ao, "tsne", "position", interactive=True, save_path=str(output_root / "stage5" / "tsne_position.html")),
        "scree": plot_pca_scree(ao, save_path=str(output_root / "stage5" / "pca_scree.png")),
    }
    for col in [sc] + mn[: min(3, len(mn))]:
        if col not in ao["cleaned_data"].columns:
            continue
        safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(col))
        key = f"umap_{safe}"
        dimred_plots[key] = plot_dim_red(
            ao, "umap", col, interactive=True, save_path=str(output_root / "stage5" / f"{key}.html")
        )
    ao["marker_analysis"]["dimred_plots"] = dimred_plots
    ao["marker_analysis"]["cluster_plots"] = {
        "kmeans": plot_clusters(ao, method="kmeans", dim_red="umap", interactive=True, save_dir=str(output_root / "stage5" / "kmeans")),
        "hdbscan": plot_clusters(ao, method="hdbscan", dim_red="umap", interactive=True, save_dir=str(output_root / "stage5" / "hdbscan")),
        "leiden": plot_clusters(ao, method="leiden", dim_red="umap", interactive=True, save_dir=str(output_root / "stage5" / "leiden")),
    }
    cygnus_pipeline_log("plots_dimred_cluster_done", t0)

    # Stage 6
    cygnus_pipeline_log("stage6_compare_export_start", t0)
    ao = compare_samples(ao)
    ao["marker_analysis"]["sample_comparison_plots"] = {
        marker: plot_sample_comparison(ao, marker, save_dir=str(output_root / "stage6" / "sample_comparison"))
        for marker in valid_marker_names(ao)
    }
    ao = run_position_qc(ao, save_dir=str(output_root / "stage6" / "position_qc"))
    export_paths = export_all(ao, output_dir=str(output_root / "exports/"))
    cygnus_pipeline_log("stage6_export_done", t0, detail=f"export_keys={len(export_paths)}")

    cygnus_pipeline_log("report_generate_start", t0)
    generate_report(ao, export_paths, output_path=str(output_root / "cygnus_report.html"))
    cygnus_pipeline_log("report_generate_done", t0, detail=f"path={output_root / 'cygnus_report.html'}")

    cygnus_pipeline_log("pipeline_complete", t0, detail=f"report={output_root / 'cygnus_report.html'}")
    return ao


if __name__ == "__main__":
    run_full_pipeline()
