"""Stage 4: colocalization analysis, UpSet, volcano, and heatmaps."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from upsetplot import UpSet, from_memberships

from stage1_loader import ID_COL, PANEV_MARKER, POSITION_COL, RELEVANT_MARKERS, SAMPLE_COL, load_cygnus_object
from stage3_preprocessing import apply_qc_filters, binarize_markers, normalize_by_panev, scale_expression_matrix


sns.set_theme(style="whitegrid")


def _sig_label(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


def _get_binary_matrix(ao: Dict[str, Any]) -> pd.DataFrame:
    mat = ao["matrices"].get("binary_exp_matrix")
    if mat is None:
        raise ValueError("ao['matrices']['binary_exp_matrix'] is None. Run stage 3 binarization first.")
    return mat


def _get_matrix(ao: Dict[str, Any], input_matrix: str) -> pd.DataFrame:
    src = {
        "raw": ao["matrices"]["Raw_Score"],
        "normalized": ao["matrices"]["normalized_exp_mat"],
        "scaled": ao["matrices"]["scaled_exp_matrix"],
    }
    if input_matrix not in src:
        raise ValueError("input_matrix must be one of {'raw', 'normalized', 'scaled'}.")
    mat = src[input_matrix]
    if mat is None:
        raise ValueError(f"Input matrix '{input_matrix}' is None.")
    return mat


def compute_colocalization(
    ao: dict,
    n_permutations: int = 1000,
    min_degree: int = 2,
    max_degree: Optional[int] = None,
    random_seed: int = 42,
) -> dict:
    """Compute observed/expected co-expression and permutation p-values."""
    binary_mat = _get_binary_matrix(ao)
    markers = [m for m in RELEVANT_MARKERS if m in binary_mat.columns]
    if max_degree is None:
        max_degree = len(markers)

    n_objects = len(binary_mat)
    if n_objects == 0:
        raise ValueError("Binary matrix has zero rows.")

    rng = np.random.default_rng(random_seed)
    base = binary_mat[markers].astype(np.int8).to_numpy()
    freq = {m: float(binary_mat[m].mean()) for m in markers}

    combo_records: List[Dict[str, Any]] = []
    for degree in range(min_degree, max_degree + 1):
        for combo in combinations(markers, degree):
            combo_idx = [markers.index(m) for m in combo]
            observed_bool = np.all(base[:, combo_idx] == 1, axis=1)
            observed_count = int(observed_bool.sum())
            observed_freq = observed_count / n_objects
            expected_count = float(n_objects * np.prod([freq[m] for m in combo]))
            deviation = float(observed_count - expected_count)

            ge_count = 0
            for _ in range(n_permutations):
                shuffled = np.empty_like(base)
                for i in range(base.shape[1]):
                    shuffled[:, i] = rng.permutation(base[:, i])
                perm_count = int(np.all(shuffled[:, combo_idx] == 1, axis=1).sum())
                if perm_count >= observed_count:
                    ge_count += 1
            empirical_p = (ge_count + 1) / (n_permutations + 1)

            combo_records.append(
                {
                    "combination": "+".join(combo),
                    "markers": list(combo),
                    "degree": degree,
                    "observed_count": observed_count,
                    "observed_frequency": observed_freq,
                    "expected_count": expected_count,
                    "deviation": deviation,
                    "empirical_pvalue": empirical_p,
                }
            )

    result_df = pd.DataFrame(combo_records)
    if result_df.empty:
        raise ValueError("No combinations were generated. Check min_degree/max_degree.")

    n_tests = len(result_df)
    result_df["bonferroni_pvalue"] = np.minimum(result_df["empirical_pvalue"] * n_tests, 1.0)
    _, fdr_vals, _, _ = multipletests(result_df["empirical_pvalue"].values, method="fdr_bh")
    result_df["fdr_pvalue"] = fdr_vals
    result_df["significance"] = result_df["fdr_pvalue"].map(_sig_label)
    result_df = result_df.sort_values(["fdr_pvalue", "observed_count"], ascending=[True, False]).reset_index(drop=True)

    if ao.get("colocalization") is None:
        ao["colocalization"] = {}
    ao["colocalization"].setdefault("reference_centered", {})
    ao["colocalization"]["all_combinations"] = result_df
    return ao


def compute_reference_colocalization(ao: dict, reference_marker: str) -> pd.DataFrame:
    """Compute reference-centered double-positive enrichment and Fisher exact p-values."""
    binary_mat = _get_binary_matrix(ao)
    if reference_marker not in binary_mat.columns:
        raise ValueError(f"reference_marker '{reference_marker}' not found in binary matrix.")

    total_objects = len(binary_mat)
    ref_pos_mask = binary_mat[reference_marker].astype(int) == 1
    ref_positive_count = int(ref_pos_mask.sum())
    if ref_positive_count == 0:
        print(
            f"[WARNING] Reference marker '{reference_marker}' has 0 positive objects. "
            "Skipping reference-centered colocalization."
        )
        ref_df = pd.DataFrame(
            columns=[
                "reference_marker",
                "target_marker",
                "reference_positive_count",
                "double_positive_count",
                "fraction_among_reference",
                "background_rate",
                "enrichment_vs_all",
                "pvalue",
                "fdr_pvalue",
                "significance",
            ]
        )
        if ao.get("colocalization") is None:
            ao["colocalization"] = {}
        ao["colocalization"].setdefault("reference_centered", {})
        ao["colocalization"]["reference_centered"][reference_marker] = ref_df
        return ref_df

    rows = []
    for target in [m for m in RELEVANT_MARKERS if m != reference_marker and m in binary_mat.columns]:
        target_pos_mask = binary_mat[target].astype(int) == 1
        double_pos = int((ref_pos_mask & target_pos_mask).sum())
        frac_among_ref = double_pos / ref_positive_count
        background_rate = float(target_pos_mask.mean())
        enrichment = frac_among_ref / background_rate if background_rate > 0 else np.nan

        a = double_pos
        b = ref_positive_count - double_pos
        c = int(target_pos_mask.sum()) - double_pos
        d = total_objects - (a + b + c)
        _, pval = fisher_exact([[a, b], [c, d]], alternative="two-sided")

        rows.append(
            {
                "reference_marker": reference_marker,
                "target_marker": target,
                "reference_positive_count": ref_positive_count,
                "double_positive_count": double_pos,
                "fraction_among_reference": frac_among_ref,
                "background_rate": background_rate,
                "enrichment_vs_all": enrichment,
                "pvalue": pval,
            }
        )

    ref_df = pd.DataFrame(rows)
    if not ref_df.empty:
        _, fdr_vals, _, _ = multipletests(ref_df["pvalue"].values, method="fdr_bh")
        ref_df["fdr_pvalue"] = fdr_vals
        ref_df["significance"] = ref_df["fdr_pvalue"].map(_sig_label)
        ref_df = ref_df.sort_values(["fdr_pvalue", "double_positive_count"], ascending=[True, False]).reset_index(drop=True)
    else:
        ref_df["fdr_pvalue"] = []
        ref_df["significance"] = []

    if ao.get("colocalization") is None:
        ao["colocalization"] = {}
    ao["colocalization"].setdefault("reference_centered", {})
    ao["colocalization"]["reference_centered"][reference_marker] = ref_df
    return ref_df


def plot_upset(
    ao: dict,
    min_count: int = 5,
    min_degree: int = 2,
    color_by_deviation: bool = True,
    highlight_significant: bool = True,
    include_panev: bool = False,
    save_path: Optional[str] = None,
):
    """Plot UpSet intersections from combination table."""
    col_df = (ao.get("colocalization") or {}).get("all_combinations")
    if col_df is None or col_df.empty:
        raise ValueError("No colocalization combinations found. Run compute_colocalization first.")

    df_plot = col_df.copy()
    if not include_panev:
        df_plot = df_plot[~df_plot["markers"].apply(lambda ms: PANEV_MARKER in ms)]
    df_plot = df_plot[(df_plot["observed_count"] >= min_count) & (df_plot["degree"] >= min_degree)]
    if df_plot.empty:
        raise ValueError("No intersections remain after min_count/min_degree filtering.")

    memberships = [tuple(m) for m in df_plot["markers"]]
    series = from_memberships(memberships, data=df_plot["observed_count"].values)

    upset = UpSet(series, sort_by="cardinality", show_counts=True)
    axes = upset.plot()
    fig = plt.gcf()
    fig.suptitle("Marker intersection UpSet plot", y=1.02)

    membership_to_dev = {tuple(markers): float(dev) for markers, dev in zip(df_plot["markers"], df_plot["deviation"])}
    membership_to_sig = {tuple(markers): str(sig) for markers, sig in zip(df_plot["markers"], df_plot["significance"])}
    marker_levels = list(upset.intersections.index.names)
    plotted_memberships = [
        tuple(marker for marker, present in zip(marker_levels, idx_vals) if bool(present))
        for idx_vals in upset.intersections.index
    ]
    inter_ax = axes["intersections"]
    bars = inter_ax.patches

    if color_by_deviation and bars:
        dev_vals = np.array([membership_to_dev.get(m, 0.0) for m in plotted_memberships], dtype=float)
        vmax = np.nanmax(np.abs(dev_vals)) if np.any(np.isfinite(dev_vals)) else 1.0
        vmax = 1.0 if vmax == 0 else vmax
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)
        cmap = plt.get_cmap("coolwarm")
        for patch, dev in zip(bars, dev_vals):
            patch.set_facecolor(cmap(norm(dev)))

    if highlight_significant and bars:
        for patch, membership in zip(bars, plotted_memberships):
            sig = membership_to_sig.get(membership, "ns")
            if sig != "ns":
                patch.set_edgecolor("black")
                patch.set_linewidth(1.5)

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def plot_volcano(
    ao: dict,
    x_metric: str = "deviation",
    p_col: str = "fdr_pvalue",
    min_count: int = 5,
    label_significant: bool = True,
    save_path: Optional[str] = None,
    interactive: bool = False,
):
    """Plot volcano chart for combination significance."""
    col_df = (ao.get("colocalization") or {}).get("all_combinations")
    if col_df is None or col_df.empty:
        raise ValueError("No colocalization combinations found. Run compute_colocalization first.")
    if p_col not in col_df.columns:
        raise ValueError(f"p_col '{p_col}' not present.")

    df_plot = col_df[col_df["observed_count"] >= min_count].copy()
    if df_plot.empty:
        raise ValueError("No combinations remain after min_count filter.")

    if x_metric == "deviation":
        df_plot["x_val"] = df_plot["deviation"]
        x_label = "Deviation (observed - expected)"
    elif x_metric == "log2_enrichment":
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (df_plot["observed_count"] + 1e-9) / (df_plot["expected_count"] + 1e-9)
            df_plot["x_val"] = np.log2(ratio)
        x_label = "log2 enrichment"
    else:
        raise ValueError("x_metric must be one of {'deviation', 'log2_enrichment'}.")

    df_plot["neglog10_p"] = -np.log10(np.clip(df_plot[p_col].astype(float), 1e-300, 1.0))
    color_map = {"***": "red", "**": "orange", "*": "yellow", "ns": "grey"}

    if interactive:
        fig = px.scatter(
            df_plot,
            x="x_val",
            y="neglog10_p",
            color="significance",
            size="observed_count",
            color_discrete_map=color_map,
            hover_data=["combination", "observed_count", "deviation", p_col],
            title="Colocalization volcano plot",
        )
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="black")
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.update_layout(xaxis_title=x_label, yaxis_title=f"-log10({p_col})")
        if save_path:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out.with_suffix(".html")))
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        for sig, sub in df_plot.groupby("significance"):
            ax.scatter(
                sub["x_val"],
                sub["neglog10_p"],
                s=20 + sub["observed_count"].values * 2,
                c=color_map.get(sig, "grey"),
                alpha=0.75,
                linewidth=0.2,
                edgecolor="black" if sig != "ns" else "none",
                label=sig,
            )
        ax.axhline(-np.log10(0.05), linestyle="--", color="black", linewidth=1)
        ax.axvline(0, linestyle="--", color="black", linewidth=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"-log10({p_col})")
        ax.set_title("Colocalization volcano plot")
        ax.legend(title="significance")

        if label_significant:
            to_label = df_plot[(df_plot["significance"] != "ns") & (df_plot["degree"] <= 3)]
            for _, row in to_label.iterrows():
                ax.text(row["x_val"], row["neglog10_p"], row["combination"], fontsize=8, alpha=0.85)

        if save_path:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def plot_expression_heatmap(
    ao: dict,
    group_by: str = "sample",
    input_matrix: str = "normalized",
    include_panev: bool = False,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    save_path: Optional[str] = None,
):
    """Plot average marker expression by sample/position/cluster label."""
    matrix = _get_matrix(ao, input_matrix=input_matrix).copy()
    markers = [m for m in matrix.columns if include_panev or m != PANEV_MARKER]

    base_df = ao["cleaned_data"][[ID_COL, SAMPLE_COL, POSITION_COL]].copy()
    if group_by == "sample":
        group_series = base_df.set_index(ID_COL)[SAMPLE_COL]
    elif group_by == "position":
        group_series = base_df.set_index(ID_COL)[POSITION_COL]
    elif group_by in {"cluster_kmeans", "cluster_hdbscan", "cluster_leiden"}:
        key = group_by.replace("cluster_", "")
        labels = ao.get("ev_meta", {}).get("cluster_labels", {}).get(key)
        if labels is None:
            raise ValueError(f"Cluster labels for '{group_by}' are not available.")
        group_series = pd.Series(labels, index=matrix.index, name=group_by)
    else:
        raise ValueError("group_by must be one of {'sample', 'position', 'cluster_kmeans', 'cluster_hdbscan', 'cluster_leiden'}.")

    aligned = matrix[markers].copy()
    aligned["group"] = group_series.reindex(aligned.index).astype(str)
    grouped_mean = aligned.groupby("group")[markers].mean()
    group_sizes = aligned.groupby("group").size()
    grouped_mean.index = [f"{grp} (n={int(group_sizes.loc[grp])})" for grp in grouped_mean.index]

    cmap = "RdBu_r" if input_matrix == "scaled" else "viridis"

    if cluster_rows or cluster_cols:
        fig = sns.clustermap(
            grouped_mean,
            cmap=cmap,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            linewidths=0.3,
            figsize=(10, 7),
        )
        fig.fig.suptitle(f"Average expression heatmap by {group_by}", y=1.02)
        out_fig = fig.fig
    else:
        out_fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(grouped_mean, cmap=cmap, linewidths=0.3, ax=ax)
        ax.set_title(f"Average expression heatmap by {group_by}")
        ax.set_xlabel("Markers")
        ax.set_ylabel("Groups")

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out_fig.savefig(out, dpi=200, bbox_inches="tight")
    return out_fig


def run_stage4(ao: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Execute full stage-4 workflow and attach outputs to analysis object."""
    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    ao = compute_colocalization(ao)
    compute_reference_colocalization(ao, reference_marker="EpCAM")

    upset_fig = plot_upset(ao, save_path=str(out_dir / "upset_plot.png") if out_dir else None)
    volcano_fig = plot_volcano(ao, save_path=str(out_dir / "volcano_plot.png") if out_dir else None)
    heat_sample = plot_expression_heatmap(
        ao,
        group_by="sample",
        input_matrix="normalized",
        save_path=str(out_dir / "heatmap_by_sample.png") if out_dir else None,
    )
    heat_position = plot_expression_heatmap(
        ao,
        group_by="position",
        input_matrix="normalized",
        save_path=str(out_dir / "heatmap_by_position.png") if out_dir else None,
    )

    ao.setdefault("marker_analysis", {})
    ao["marker_analysis"]["colocalization_plots"] = {
        "upset": upset_fig,
        "volcano": volcano_fig,
        "heatmap_by_sample": heat_sample,
        "heatmap_by_position": heat_position,
    }

    if out_dir:
        ao["colocalization"]["all_combinations"].to_csv(out_dir / "colocalization_all_combinations.csv", index=False)
        ao["colocalization"]["reference_centered"]["EpCAM"].to_csv(
            out_dir / "colocalization_reference_EpCAM.csv", index=False
        )

    return ao


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
    ao = binarize_markers(ao, method="percentile", percentile=95.0, input_matrix="normalized")
    ao = run_stage4(ao, output_dir="./output/stage4/")
    print("Stage 4 complete.")
    print("Saved tables and plots to ./output/stage4/")
