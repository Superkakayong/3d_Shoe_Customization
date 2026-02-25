import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from typing import Tuple

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# --- Geometric (bounding-box) trim along any axis (x/y/z)
_AXIS2IDX = {"x": 0, "y": 1, "z": 2}

def trim_by_axis_range(
    pcd: o3d.geometry.PointCloud,
    axis: str = "x",
    min_ratio: float = 0.0,
    max_ratio: float = 1.0,
    return_indices: bool = True,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Geometric (bounding-box) trim along a chosen axis.

    This uses the axis extent of the point cloud's bounding box (not percentiles).
    Example:
        axis="x", min_ratio=0.10, max_ratio=0.90
        -> remove 10% on left + 10% on right along X (keep the middle 80%)

    Args:
        pcd: Open3D PointCloud (already PCA-aligned)
        axis: "x" | "y" | "z"
        min_ratio: float in [0, 1]. 0.10 means start at 10% of axis-range from axis_min.
        max_ratio: float in [0, 1]. 0.90 means up to 90% of axis-range from axis_min.
        return_indices: whether to return indices into input pcd

    Returns:
        trimmed_pcd: filtered point cloud
        idx: indices into input pcd (empty if return_indices=False)
    """
    ax = axis.lower().strip()
    if ax not in _AXIS2IDX:
        raise ValueError("axis must be one of: 'x', 'y', 'z'")
    if not (0.0 <= min_ratio <= max_ratio <= 1.0):
        raise ValueError("Ratios must satisfy 0 ≤ min_ratio ≤ max_ratio ≤ 1")

    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return o3d.geometry.PointCloud(), np.array([], dtype=int)

    j = _AXIS2IDX[ax]
    v = pts[:, j]
    v_min = float(v.min())
    v_max = float(v.max())

    # Degenerate extent: keep everything
    if v_max <= v_min + 1e-12:
        idx = np.arange(len(v), dtype=int)
        trimmed_pcd = pcd.select_by_index(idx)
        return (trimmed_pcd, idx) if return_indices else (trimmed_pcd, np.array([], dtype=int))

    v_lo = v_min + min_ratio * (v_max - v_min)
    v_hi = v_min + max_ratio * (v_max - v_min)

    mask = (v >= v_lo) & (v <= v_hi)
    idx = np.where(mask)[0].astype(int)

    trimmed_pcd = pcd.select_by_index(idx)

    if return_indices:
        return trimmed_pcd, idx
    return trimmed_pcd, np.array([], dtype=int)