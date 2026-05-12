"""Stage 5: dimensionality reduction and clustering."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
except ImportError:  # pragma: no cover
    hdbscan = None

try:
    import igraph as ig
    import leidenalg
except ImportError:  # pragma: no cover
    ig = None
    leidenalg = None

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None

from stage1_loader import ID_COL, marker_base_columns, score_column, POSITION_COL, SAMPLE_COL, valid_marker_names, load_cygnus_object
from stage3_preprocessing import apply_qc_filters, binarize_markers, normalize_by_panev, scale_expression_matrix


sns.set_theme(style="whitegrid")
_LOG = logging.getLogger(__name__)


def _get_input_matrix(ao: Dict[str, Any], input_matrix: str) -> pd.DataFrame:
    matrix_map = {
        "raw": ao["matrices"]["Raw_Score"],
        "normalized": ao["matrices"]["normalized_exp_mat"],
        "scaled": ao["matrices"]["scaled_exp_matrix"],
    }
    if input_matrix not in matrix_map:
        raise ValueError("input_matrix must be one of {'raw', 'normalized', 'scaled'}.")
    mat = matrix_map[input_matrix]
    if mat is None:
        raise ValueError(f"Selected input matrix '{input_matrix}' is None.")
    return mat


def _is_categorical(series: pd.Series) -> bool:
    return str(series.dtype) in ("category", "string") or series.dtype == "object"


def _normalize_size(series: pd.Series, min_size: float = 8.0, max_size: float = 28.0) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.nunique(dropna=True) <= 1:
        return pd.Series(np.full(len(x), min_size), index=x.index)
    z = (x - x.min()) / (x.max() - x.min())
    return min_size + z * (max_size - min_size)


def _space_array(ao: Dict[str, Any], input_space: str) -> Tuple[np.ndarray, pd.Index]:
    markers = [m for m in valid_marker_names(ao)]
    if input_space == "scaled":
        mat = ao["matrices"]["scaled_exp_matrix"]
        if mat is None:
            raise ValueError("scaled_exp_matrix is None.")
        cols = [m for m in markers if m in mat.columns]
        if not cols:
            raise ValueError("scaled_exp_matrix has no columns for valid markers.")
        return mat[cols].to_numpy(), mat.index
    if input_space == "pca":
        df = ao["dim_red"]["pca"]
        if df is None:
            raise ValueError("PCA not available. Run run_dim_reduction(method='pca').")
        cols = [c for c in df.columns if c.startswith("PC")]
        return df[cols].to_numpy(), df[ID_COL]
    if input_space == "umap":
        df = ao["dim_red"]["umap"]
        if df is None:
            raise ValueError("UMAP not available. Run run_dim_reduction(method='umap').")
        cols = [c for c in df.columns if c.startswith("UMAP")]
        return df[cols].to_numpy(), df[ID_COL]
    if input_space == "tsne":
        df = ao["dim_red"]["tsne"]
        if df is None:
            raise ValueError("t-SNE not available. Run run_dim_reduction(method='tsne').")
        cols = [c for c in df.columns if c.startswith("tSNE")]
        return df[cols].to_numpy(), df[ID_COL]
    raise ValueError("input_space must be one of {'scaled', 'pca', 'umap', 'tsne'}.")


def run_dim_reduction(
    ao: dict,
    method: str,
    input_matrix: str = "scaled",
    markers: Optional[Sequence[str]] = None,
    n_components: int = 2,
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 1000,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run PCA/t-SNE/UMAP and return metadata-joined coordinates."""
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3.")
    vset = set(valid_marker_names(ao))
    if markers is not None:
        sel_markers = [m for m in markers if m in vset]
    else:
        sel_markers = [m for m in marker_base_columns(ao) if m in vset]
    if not sel_markers:
        raise ValueError("No valid markers for dimensionality reduction after NaN-rate filtering.")
    missing = [m for m in sel_markers if m not in marker_base_columns(ao)]
    if missing:
        raise ValueError(f"Unknown markers for this dataset: {missing}")

    use_raw_tiff = bool(ao.get("dim_red_uses_raw_tiff_mean"))
    raw_cols_all: list = list(ao.get("marker_raw_tiff_mean_cols") or [])
    raw_cols = [f"{m}_raw_tiff_mean" for m in sel_markers]
    use_ibg = bool(ao.get("dim_red_uses_intensity_bg_subtracted")) and not use_raw_tiff
    ibg_cols_all: list = list(ao.get("marker_intensity_bg_subtracted_cols") or [])
    ibg_cols = [f"{m}_intensity_bg_subtracted" for m in sel_markers]

    rim = ao.get("raw_intensity_matrix")
    used_rim = False
    if isinstance(rim, pd.DataFrame) and not rim.empty:
        rim_cols = [m for m in sel_markers if m in rim.columns]
        if rim_cols:
            mat_ref = _get_input_matrix(ao, input_matrix=input_matrix)
            ids = mat_ref.index
            sub = rim.reindex(ids)[rim_cols].apply(pd.to_numeric, errors="coerce")
            sub = sub.fillna(sub.median(numeric_only=True))
            x = StandardScaler().fit_transform(sub.to_numpy(dtype=np.float64))
            ids_arr = ids.to_numpy()
            used_rim = True
            _LOG.info(
                "Dimensionality reduction (%s) using Results Viewer Raw Intensity matrix (z-scored per marker).",
                method,
            )
    if not used_rim and use_raw_tiff and raw_cols_all and all(c in ao["cleaned_data"].columns for c in raw_cols):
        mat_ref = _get_input_matrix(ao, input_matrix=input_matrix)
        ids = mat_ref.index
        work = ao["cleaned_data"].set_index(ID_COL)
        try:
            sub = work.reindex(ids)[raw_cols].apply(pd.to_numeric, errors="coerce")
        except KeyError as exc:
            raise ValueError(f"Missing *_raw_tiff_mean columns for dim reduction: {exc}") from exc
        sub = sub.fillna(sub.median(numeric_only=True))
        x = StandardScaler().fit_transform(sub.to_numpy(dtype=np.float64))
        ids_arr = ids.to_numpy()
        _LOG.info(
            "Dimensionality reduction (%s) using *_raw_tiff_mean columns (z-scored per marker).",
            method,
        )
    elif not used_rim and use_ibg and ibg_cols_all and all(c in ao["cleaned_data"].columns for c in ibg_cols):
        mat_ref = _get_input_matrix(ao, input_matrix=input_matrix)
        ids = mat_ref.index
        work = ao["cleaned_data"].set_index(ID_COL)
        try:
            sub = work.reindex(ids)[ibg_cols].apply(pd.to_numeric, errors="coerce")
        except KeyError as exc:
            raise ValueError(f"Missing *_intensity_bg_subtracted columns for dim reduction: {exc}") from exc
        sub = sub.fillna(sub.median(numeric_only=True))
        x = StandardScaler().fit_transform(sub.to_numpy(dtype=np.float64))
        ids_arr = ids.to_numpy()
        _LOG.info(
            "Dimensionality reduction (%s) using *_intensity_bg_subtracted columns (z-scored per marker).",
            method,
        )
    elif not used_rim:
        ibg_wanted = bool(ao.get("dim_red_uses_intensity_bg_subtracted")) and not use_raw_tiff
        ibg_complete = bool(ibg_cols_all) and all(c in ao["cleaned_data"].columns for c in ibg_cols)
        if ibg_wanted and not ibg_complete:
            if not ao.get("_dim_red_logged_ibg_incomplete_fallback"):
                _LOG.warning(
                    "Using input_matrix=%r for %s: dim_red_uses_intensity_bg_subtracted is set but "
                    "*_intensity_bg_subtracted columns are incomplete in cleaned_data.",
                    input_matrix,
                    method,
                )
                ao["_dim_red_logged_ibg_incomplete_fallback"] = True
        elif not use_raw_tiff and not ao.get("_dim_red_logged_marker_matrix_fallback"):
            _LOG.warning(
                "Using input_matrix=%r (*_positive-derived expression matrix) for %s: "
                "no complete *_raw_tiff_mean columns for dimensionality reduction.",
                input_matrix,
                method,
            )
            ao["_dim_red_logged_marker_matrix_fallback"] = True
        mat = _get_input_matrix(ao, input_matrix=input_matrix)
        missing_m = [m for m in sel_markers if m not in mat.columns]
        if missing_m:
            raise ValueError(f"Markers not found in input matrix: {missing_m}")
        x = mat[sel_markers].to_numpy()
        ids_arr = mat.index.to_numpy()

    if method == "pca":
        model = PCA(n_components=n_components, random_state=random_seed)
        coords = model.fit_transform(x)
        comp_cols = [f"PC{i+1}" for i in range(n_components)]
        ao["dim_red"]["pca_explained_variance"] = model.explained_variance_ratio_
    elif method == "tsne":
        model = TSNE(
            n_components=n_components,
            perplexity=tsne_perplexity,
            max_iter=tsne_n_iter,
            random_state=random_seed,
            init="pca",
            learning_rate="auto",
        )
        coords = model.fit_transform(x)
        comp_cols = [f"tSNE{i+1}" for i in range(n_components)]
    elif method == "umap":
        if umap is None:
            raise ImportError("umap-learn is required for UMAP. Install it in your conda environment.")
        model = umap.UMAP(
            n_components=n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=random_seed,
        )
        coords = model.fit_transform(x)
        comp_cols = [f"UMAP{i+1}" for i in range(n_components)]
    else:
        raise ValueError("method must be one of {'pca', 'tsne', 'umap'}.")

    coord_df = pd.DataFrame(coords, columns=comp_cols)
    coord_df[ID_COL] = ids_arr

    cleaned = ao["cleaned_data"]
    sc = score_column(ao)
    meta_cols = [ID_COL, SAMPLE_COL, POSITION_COL, "centroid_x", "centroid_y", "area", "circularity", sc]
    marker_cols = [m for m in valid_marker_names(ao) if m in cleaned.columns]
    meta_cols = [c for c in meta_cols if c in cleaned.columns]
    merged = coord_df.merge(cleaned[meta_cols + marker_cols], on=ID_COL, how="left")

    ao["dim_red"][method] = merged
    return merged


def plot_dim_red(
    ao: dict,
    method: str,
    color_by: str,
    components: Tuple[int, int] = (1, 2),
    size_by: Optional[str] = None,
    interactive: bool = True,
    save_path: Optional[str] = None,
):
    """Plot PCA/t-SNE/UMAP coordinates with metadata/marker coloring."""
    df = ao["dim_red"].get(method)
    if df is None:
        raise ValueError(f"Dimensionality reduction '{method}' not found.")

    prefix_map = {"pca": "PC", "tsne": "tSNE", "umap": "UMAP"}
    prefix = prefix_map.get(method)
    if prefix is None:
        raise ValueError("method must be one of {'pca', 'tsne', 'umap'}.")
    x_col = f"{prefix}{components[0]}"
    y_col = f"{prefix}{components[1]}"
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Requested components not present: {x_col}, {y_col}.")
    if color_by not in df.columns:
        raise ValueError(f"color_by '{color_by}' not found in {method} dataframe.")
    if size_by is not None and size_by not in df.columns:
        raise ValueError(f"size_by '{size_by}' not found in dataframe.")

    title = f"{method.upper()} colored by {color_by}"
    size_series = _normalize_size(df[size_by]) if size_by else None

    hover_cols = [ID_COL, SAMPLE_COL, POSITION_COL, "centroid_x", "centroid_y", "area", "circularity", score_column(ao)]
    hover_cols += [m for m in valid_marker_names(ao) if m in df.columns]
    hover_cols = [c for c in hover_cols if c in df.columns]

    if interactive:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_by,
            size=size_by,
            color_continuous_scale="viridis",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data=hover_cols,
            title=title,
        )
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
        if save_path:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out.with_suffix(".html")))
        return fig

    fig, ax = plt.subplots(figsize=(7, 6))
    if _is_categorical(df[color_by]):
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue=color_by,
            size=size_series if size_by else None,
            sizes=(10, 50) if size_by else None,
            palette="tab10",
            linewidth=0,
            alpha=0.85,
            ax=ax,
        )
    else:
        sc = ax.scatter(df[x_col], df[y_col], c=df[color_by], cmap="viridis", s=size_series if size_by else 15, alpha=0.85, linewidth=0)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_by)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def plot_pca_scree(ao: Dict[str, Any], save_path: Optional[str] = None):
    """Plot explained variance ratios for PCA."""
    evr = ao["dim_red"].get("pca_explained_variance")
    if evr is None:
        raise ValueError("pca_explained_variance not found. Run PCA first.")
    x = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, evr, color="#4C72B0")
    ax.plot(x, np.cumsum(evr), marker="o", color="black", linewidth=1)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA scree plot")
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def run_kmeans(
    ao: dict,
    n_clusters: int = 5,
    input_space: str = "umap",
    random_seed: int = 42,
) -> dict:
    """Run K-means clustering and store outputs."""
    x, ids = _space_array(ao, input_space=input_space)
    model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    labels = pd.Series(model.fit_predict(x), index=ids, name="kmeans_cluster")

    ao["ev_meta"]["cluster_labels"]["kmeans"] = labels
    if ao.get("clustering") is None:
        ao["clustering"] = {}
    ao["clustering"]["kmeans"] = {"n_clusters": n_clusters, "labels": labels, "model": model}

    if input_space in {"pca", "tsne", "umap"} and ao["dim_red"][input_space] is not None:
        ao["dim_red"][input_space]["kmeans_cluster"] = ao["dim_red"][input_space][ID_COL].map(labels)
    return ao


def run_hdbscan(
    ao: dict,
    min_cluster_size: int = 50,
    min_samples: int = 5,
    input_space: str = "umap",
) -> dict:
    """Run HDBSCAN clustering and store outputs."""
    if hdbscan is None:
        raise ImportError("hdbscan is required for run_hdbscan. Install it in your conda environment.")
    x, ids = _space_array(ao, input_space=input_space)
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = pd.Series(model.fit_predict(x), index=ids, name="hdbscan_cluster")

    ao["ev_meta"]["cluster_labels"]["hdbscan"] = labels
    if ao.get("clustering") is None:
        ao["clustering"] = {}
    ao["clustering"]["hdbscan"] = {"labels": labels, "model": model}

    if input_space in {"pca", "tsne", "umap"} and ao["dim_red"][input_space] is not None:
        ao["dim_red"][input_space]["hdbscan_cluster"] = ao["dim_red"][input_space][ID_COL].map(labels)
    return ao


def run_leiden(
    ao: dict,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    input_space: str = "umap",
) -> dict:
    """Run Leiden clustering on a kNN graph from selected space."""
    if ig is None or leidenalg is None:
        raise ImportError("leidenalg and igraph are required for run_leiden. Install them in your conda environment.")

    x, ids = _space_array(ao, input_space=input_space)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(x)
    neigh_idx = nn.kneighbors(return_distance=False)[:, 1:]
    edges = {(int(i), int(j)) for i in range(len(x)) for j in neigh_idx[i] if i != j}

    graph = ig.Graph(n=len(x), edges=list(edges), directed=False)
    part = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
    )
    labels = pd.Series(np.array(part.membership, dtype=int), index=ids, name="leiden_cluster")

    ao["ev_meta"]["cluster_labels"]["leiden"] = labels
    if ao.get("clustering") is None:
        ao["clustering"] = {}
    ao["clustering"]["leiden"] = {"labels": labels, "resolution": resolution}

    if input_space in {"pca", "tsne", "umap"} and ao["dim_red"][input_space] is not None:
        ao["dim_red"][input_space]["leiden_cluster"] = ao["dim_red"][input_space][ID_COL].map(labels)
    return ao


def plot_clusters(
    ao: dict,
    method: str,
    dim_red: str = "umap",
    interactive: bool = True,
    save_dir: Optional[str] = None,
):
    """Plot cluster embeddings, compositions, and mean marker heatmap."""
    if method not in {"kmeans", "hdbscan", "leiden"}:
        raise ValueError("method must be one of {'kmeans', 'hdbscan', 'leiden'}.")
    if dim_red not in {"pca", "tsne", "umap"}:
        raise ValueError("dim_red must be one of {'pca', 'tsne', 'umap'}.")

    out = Path(save_dir) if save_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    labels = ao["ev_meta"]["cluster_labels"].get(method)
    if labels is None:
        raise ValueError(f"Cluster labels for '{method}' not found.")
    emb = ao["dim_red"].get(dim_red)
    if emb is None:
        raise ValueError(f"Dimensionality reduction '{dim_red}' not found.")

    label_col = f"{method}_cluster"
    emb_plot = emb.copy()
    emb_plot[label_col] = emb_plot[ID_COL].map(labels).astype("Int64").astype(str)

    cluster_fig = plot_dim_red(
        {"dim_red": {dim_red: emb_plot}},
        method=dim_red,
        color_by=label_col,
        interactive=interactive,
        save_path=str(out / f"{method}_{dim_red}_clusters.png") if out and not interactive else str(out / f"{method}_{dim_red}_clusters.html") if out else None,
    )

    markers = [m for m in valid_marker_names(ao) if m in ao["cleaned_data"].columns]
    merged = ao["cleaned_data"][[ID_COL, SAMPLE_COL, POSITION_COL] + markers].copy()
    merged[label_col] = merged[ID_COL].map(labels).astype("Int64")
    merged = merged.dropna(subset=[label_col]).copy()
    merged[label_col] = merged[label_col].astype(int).astype(str)

    sample_ct = pd.crosstab(merged[label_col], merged[SAMPLE_COL])
    sample_prop = sample_ct.div(sample_ct.sum(axis=1), axis=0)
    position_ct = pd.crosstab(merged[label_col], merged[POSITION_COL])
    position_prop = position_ct.div(position_ct.sum(axis=1), axis=0)

    fig_sample, axes_sample = plt.subplots(1, 2, figsize=(14, 5))
    sample_ct.plot(kind="bar", stacked=True, ax=axes_sample[0], colormap="tab20")
    axes_sample[0].set_title(f"{method} cluster composition by sample (counts)")
    axes_sample[0].set_xlabel("Cluster")
    axes_sample[0].set_ylabel("Count")
    sample_prop.plot(kind="bar", stacked=True, ax=axes_sample[1], colormap="tab20")
    axes_sample[1].set_title(f"{method} cluster composition by sample (proportion)")
    axes_sample[1].set_xlabel("Cluster")
    axes_sample[1].set_ylabel("Proportion")
    plt.tight_layout()
    if out:
        fig_sample.savefig(out / f"{method}_composition_sample.png", dpi=200, bbox_inches="tight")

    fig_position, axes_position = plt.subplots(1, 2, figsize=(14, 5))
    position_ct.plot(kind="bar", stacked=True, ax=axes_position[0], colormap="tab20")
    axes_position[0].set_title(f"{method} cluster composition by position (counts)")
    axes_position[0].set_xlabel("Cluster")
    axes_position[0].set_ylabel("Count")
    position_prop.plot(kind="bar", stacked=True, ax=axes_position[1], colormap="tab20")
    axes_position[1].set_title(f"{method} cluster composition by position (proportion)")
    axes_position[1].set_xlabel("Cluster")
    axes_position[1].set_ylabel("Proportion")
    plt.tight_layout()
    if out:
        fig_position.savefig(out / f"{method}_composition_position.png", dpi=200, bbox_inches="tight")

    cluster_mean = merged.groupby(label_col)[markers].mean()
    fig_heat, ax_heat = plt.subplots(figsize=(8, 5))
    sns.heatmap(cluster_mean, cmap="viridis", linewidths=0.3, ax=ax_heat)
    ax_heat.set_title(f"{method} average marker expression per cluster")
    ax_heat.set_xlabel("Marker")
    ax_heat.set_ylabel("Cluster")
    if out:
        fig_heat.savefig(out / f"{method}_cluster_expression_heatmap.png", dpi=200, bbox_inches="tight")

    return {
        "cluster_embedding": cluster_fig,
        "composition_sample": fig_sample,
        "composition_position": fig_position,
        "cluster_expression_heatmap": fig_heat,
    }


if __name__ == "__main__":
    ao = load_cygnus_object("all_cells.csv")
    ao = apply_qc_filters(ao, remove_zero_area=True, remove_zero_perimeter=True, remove_nan_circularity=True)
    ao = normalize_by_panev(ao, use_filtered=False, epsilon=1e-6, include_panev=False)
    ao = scale_expression_matrix(ao, method="zscore", input_matrix="normalized")
    ao = binarize_markers(ao, method="percentile", percentile=95.0, input_matrix="normalized")

    out_dir = Path("./output/stage5")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dim_reduction(ao, method="pca", input_matrix="scaled", markers=marker_base_columns(ao), n_components=2)
    run_dim_reduction(ao, method="tsne", input_matrix="scaled", markers=marker_base_columns(ao), n_components=2)
    run_dim_reduction(ao, method="umap", input_matrix="scaled", markers=marker_base_columns(ao), n_components=2)

    mn = marker_base_columns(ao)
    default_colors = [SAMPLE_COL, POSITION_COL, score_column(ao)] + mn[: min(3, len(mn))]
    for method_name in ("pca", "tsne", "umap"):
        for color in default_colors:
            plot_dim_red(
                ao,
                method=method_name,
                color_by=color,
                interactive=True,
                save_path=str(out_dir / f"{method_name}_by_{color}.html"),
            )

    plot_pca_scree(ao, save_path=str(out_dir / "pca_scree.png"))

    run_kmeans(ao, n_clusters=5, input_space="umap", random_seed=42)
    run_hdbscan(ao, min_cluster_size=50, min_samples=5, input_space="umap")
    run_leiden(ao, resolution=1.0, n_neighbors=15, input_space="umap")

    for cluster_method in ("kmeans", "hdbscan", "leiden"):
        plot_clusters(ao, method=cluster_method, dim_red="umap", interactive=True, save_dir=str(out_dir / cluster_method))

    print("Stage 5 complete.")
    print(f"Saved dimensionality reduction and clustering outputs to: {out_dir}")
