"""
clustering.py – Shared clustering logic used by both /faces/cluster and /faces/cluster-group

Algorithm comparison (choose for your use-case):
  connected_components  Best for person grouping with a known similarity level.
                        Every pair above `similarity_threshold` ends up in the same group.
                        No noise, very interpretable.  O(n²) — great up to ~5 000 faces.

  hdbscan               Best when you have many faces and expect noise / outliers.
                        Fully automatic group count.  Slow on CPU for n > 10 000.

  agglomerative         Hierarchical merging. Fast.  Use when you want to control
                        "tightness" via distance_threshold and have few noise faces.
"""

import numpy as np
import logging
from typing import Optional

log = logging.getLogger("vision-api.clustering")


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import normalize
    return normalize(vectors.astype(np.float32))


def run_dbscan(vectors_norm: np.ndarray, min_similarity: float = 0.60) -> list[int]:
    """
    DBSCAN on L2-normalised embeddings using cosine distance.
    eps = 1 - min_similarity  (cosine distance threshold)
    min_samples=1  → every point gets a cluster, no noise by default
    """
    from sklearn.cluster import DBSCAN
    eps = max(0.001, 1.0 - min_similarity)
    db = DBSCAN(eps=eps, min_samples=1, metric="cosine")
    return db.fit_predict(vectors_norm).tolist()


def run_connected_components(
    vectors_norm: np.ndarray,
    similarity_threshold: float = 0.65,
) -> list[int]:
    """
    Graph-based grouping: build a similarity graph where an edge exists between
    face i and face j iff  cosine_similarity(i, j) >= similarity_threshold.
    Returns the connected-component labels (0-based, no noise / -1 labels).

    Recommended for person grouping with a known similarity level.
    Lower threshold → fewer, larger groups.  Higher → more, tighter groups.

    Typical ranges for ArcFace:
      0.60 – loose (same person, different lighting/angle OK)
      0.68 – medium (same model default verification threshold)
      0.75 – strict (very similar appearance required)

    Complexity: O(n²) in time and memory → suitable for up to ~5 000 faces.
    """
    n = len(vectors_norm)
    if n == 0:
        return []

    # Full cosine-similarity matrix (n×n) — fast numpy matmul on normalised vectors
    sim = vectors_norm @ vectors_norm.T   # shape (n, n)

    # Union-Find with path compression + union by rank
    parent = list(range(n))
    rank   = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path halving
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    # Vectorised edge enumeration — avoid Python loop for the inner dimension
    rows, cols = np.where(np.triu(sim >= similarity_threshold, k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        union(i, j)

    # Map roots → contiguous 0-based labels
    root_to_label: dict[int, int] = {}
    labels: list[int] = []
    for i in range(n):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels.append(root_to_label[root])

    return labels


def run_hdbscan(
    vectors_norm: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: float = 0.0,
) -> list[int]:
    """
    HDBSCAN on L2-normalised embeddings.
    - min_cluster_size: minimum group size (≥2).  Lower → more, smaller groups.
    - min_samples: core-point threshold.  Defaults to min_cluster_size.
                   Lower → less noise, more faces assigned to a group.
    - cluster_selection_epsilon: merge sub-clusters whose centroids are closer
                   than this L2 distance (~cosine dist on unit sphere).
                   0.0 = disabled.  0.1–0.2 prevents over-fragmentation.
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("hdbscan is not installed. Add it to requirements.txt.")
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=max(min_cluster_size, 2),
        min_samples=min_samples,                     # None → same as min_cluster_size
        metric="euclidean",                          # L2 on normalised ≈ cosine
        cluster_selection_method="eom",
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    return hdb.fit_predict(vectors_norm).tolist()


def run_agglomerative(
    vectors_norm: np.ndarray,
    n_groups: Optional[int],
    distance_threshold: float = 0.4,
) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering
    if n_groups is not None:
        agg = AgglomerativeClustering(
            n_clusters=n_groups,
            distance_threshold=None,
            metric="cosine",
            linkage="average",
        )
    else:
        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
    return agg.fit_predict(vectors_norm).tolist()


def run_kmeans(vectors_norm: np.ndarray, n_groups: int) -> list[int]:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_groups, random_state=42, n_init="auto")
    return km.fit_predict(vectors_norm).tolist()


def build_groups(items: list, labels: list[int]) -> list[dict]:
    """
    Given parallel lists of items and their integer cluster labels,
    return a list of group dicts sorted by group_id.
    Each item must already be a dict.
    """
    group_map: dict[int, list] = {}
    for item, label in zip(items, labels):
        group_map.setdefault(int(label), []).append(item)

    return [
        {
            "group_id": gid,
            "cluster_count": len(members),
            "members": members,
        }
        for gid, members in sorted(group_map.items())
    ]


def compute_centroid(embeddings: list[list[float]]) -> list[float]:
    """Mean-pooled centroid of a list of embedding vectors."""
    arr = np.array(embeddings, dtype=np.float32)
    mean = arr.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean /= norm
    return mean.tolist()
