"""Full Cygnus pipeline orchestration script."""

from __future__ import annotations

from pathlib import Path

from stage1_loader import RELEVANT_MARKERS, load_cygnus_object
from stage2_visualization import run_stage2_visualizations
from stage3_preprocessing import apply_qc_filters, binarize_markers, normalize_by_panev, scale_expression_matrix
from stage4_colocalization import (
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


def run_full_pipeline(filepath: str = "all_cells.csv", output_dir: str = "./output/"):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Stage 1
    ao = load_cygnus_object(filepath)

    # Stage 2
    ao = run_stage2_visualizations(ao, save_dir=str(output_root / "stage2"))

    # Stage 3
    ao = apply_qc_filters(ao)
    ao = normalize_by_panev(ao)
    ao = scale_expression_matrix(ao, method="zscore", input_matrix="normalized")
    ao = binarize_markers(ao, method="percentile", percentile=95.0, input_matrix="normalized")

    # Stage 4
    ao = compute_colocalization(ao)
    compute_reference_colocalization(ao, reference_marker="EpCAM")
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

    # Stage 5
    run_dim_reduction(ao, method="pca")
    run_dim_reduction(ao, method="tsne")
    run_dim_reduction(ao, method="umap")
    ao = run_kmeans(ao, n_clusters=5)
    ao = run_hdbscan(ao)
    ao = run_leiden(ao)

    ao["marker_analysis"]["dimred_plots"] = {
        "pca_sample": plot_dim_red(ao, "pca", "sample", interactive=True, save_path=str(output_root / "stage5" / "pca_sample.html")),
        "pca_position": plot_dim_red(ao, "pca", "position", interactive=True, save_path=str(output_root / "stage5" / "pca_position.html")),
        "umap_sample": plot_dim_red(ao, "umap", "sample", interactive=True, save_path=str(output_root / "stage5" / "umap_sample.html")),
        "umap_position": plot_dim_red(ao, "umap", "position", interactive=True, save_path=str(output_root / "stage5" / "umap_position.html")),
        "tsne_sample": plot_dim_red(ao, "tsne", "sample", interactive=True, save_path=str(output_root / "stage5" / "tsne_sample.html")),
        "tsne_position": plot_dim_red(ao, "tsne", "position", interactive=True, save_path=str(output_root / "stage5" / "tsne_position.html")),
        "umap_EpCAM": plot_dim_red(ao, "umap", "EpCAM", interactive=True, save_path=str(output_root / "stage5" / "umap_EpCAM.html")),
        "umap_MET": plot_dim_red(ao, "umap", "MET", interactive=True, save_path=str(output_root / "stage5" / "umap_MET.html")),
        "scree": plot_pca_scree(ao, save_path=str(output_root / "stage5" / "pca_scree.png")),
    }
    ao["marker_analysis"]["cluster_plots"] = {
        "kmeans": plot_clusters(ao, method="kmeans", dim_red="umap", interactive=True, save_dir=str(output_root / "stage5" / "kmeans")),
        "hdbscan": plot_clusters(ao, method="hdbscan", dim_red="umap", interactive=True, save_dir=str(output_root / "stage5" / "hdbscan")),
        "leiden": plot_clusters(ao, method="leiden", dim_red="umap", interactive=True, save_dir=str(output_root / "stage5" / "leiden")),
    }

    # Stage 6
    ao = compare_samples(ao)
    ao["marker_analysis"]["sample_comparison_plots"] = {
        marker: plot_sample_comparison(ao, marker, save_dir=str(output_root / "stage6" / "sample_comparison"))
        for marker in RELEVANT_MARKERS
    }
    ao = run_position_qc(ao, save_dir=str(output_root / "stage6" / "position_qc"))
    export_paths = export_all(ao, output_dir=str(output_root / "exports/"))
    generate_report(ao, export_paths, output_path=str(output_root / "cygnus_report.html"))

    print(f"Pipeline complete. Report: {output_root}/cygnus_report.html")
    return ao


if __name__ == "__main__":
    run_full_pipeline()
