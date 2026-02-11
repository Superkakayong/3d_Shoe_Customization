# align.py
import numpy as np
import open3d as o3d
from typing import Dict, Tuple, Optional, List

from scipy.interpolate import interp1d


# ============================================================
# Overall dynamic alignment pipeline:
# 1) Remove unwanted regions from point clouds (e.g., artifacts).
# 2) Compute vertical displacement profile dz(y) by sweeping thin cutters along Y.
# 3) Build an interpolation function f(y) from sampled dz(y).
# 4) Warp the foot point cloud by applying z_new = z_old - f(y).
# ============================================================


# ============================================================
# Helpers: index / selection
# ============================================================

def remove_region_from_pcd(
    pcd: o3d.geometry.PointCloud,
    region_idx: np.ndarray,
) -> o3d.geometry.PointCloud:
    """
    Remove points whose indices are in region_idx (indices into pcd).
    Returns a new pcd.
    """
    # Convert region indices to integer numpy array for indexing
    region_idx = np.asarray(region_idx, dtype=int)

    # Create a boolean mask of all points to keep (True)
    keep_mask = np.ones(len(pcd.points), dtype=bool)
    # Set points in region_idx to False (to remove them)
    keep_mask[region_idx] = False

    # Convert boolean mask to explicit indices of points to keep
    keep_idx = np.where(keep_mask)[0].astype(int)

    # Select and return a new point cloud with only the kept points
    return pcd.select_by_index(keep_idx)


def build_y_cutter_mask(y: np.ndarray, y0: float, cutter_width: float) -> np.ndarray:
    """
    It returns a boolean mask: for each point, True/False = “is this point inside the current Y-cutter?”
    cutter center is at y0
    """
    # Compute half cutter width for easier comparison
    half = 0.5 * float(cutter_width)

    # Return boolean mask selecting points within cutter around y0
    return (y >= (y0 - half)) & (y <= (y0 + half))


# ============================================================
# Step A: sample dz(y) by sweeping cutters
# ============================================================

def compute_dz_profile(
    foot_pcd: o3d.geometry.PointCloud,
    insole_pcd: o3d.geometry.PointCloud,
    cutter_width: float = 1.0,
    y_step: float = 1.0,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    # robust statistics
    foot_z_stat: str = "p05",     # "min", "p05", "p10", "median"
    insole_z_stat: str = "median",# "mean", "median", "p90"
    min_points_per_cutter: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Sweep a thin XZ cutter along +Y. For each y0, compute dz(y0).

    dz convention:
        dz = foot_z - insole_z
    If dz > 0 => foot is above insole at that slice (needs to move DOWN by dz)
    We'll later apply: z_new = z_old - f(y)

    Returns dict:
        {
          "y_samples": (M,),
          "dz_samples": (M,),
          "counts_foot": (M,),
          "counts_insole": (M,)
        }
    """
    # Convert point clouds to numpy arrays of points for easier processing
    foot_pts = np.asarray(foot_pcd.points)
    insole_pts = np.asarray(insole_pcd.points)

    # Early exit if either point cloud is empty (no data)
    if len(foot_pts) == 0 or len(insole_pts) == 0:
        return {"y_samples": np.array([]), "dz_samples": np.array([]),
                "counts_foot": np.array([]), "counts_insole": np.array([])}

    # Extract Y coordinates for foot and insole points
    fy = foot_pts[:, 1] # all foot Y values
    iy = insole_pts[:, 1] # all insole Y values

    # Determine overlapping Y range between foot and insole
    # Use max of mins to ensure cutter is within both clouds
    if y_min is None:
        y_min = max(float(fy.min()), float(iy.min()))
    # Use min of maxs for the same reason
    if y_max is None:
        y_max = min(float(fy.max()), float(iy.max()))

    # If no overlap in Y, return empty result
    if y_max <= y_min:
        return {"y_samples": np.array([]), "dz_samples": np.array([]),
                "counts_foot": np.array([]), "counts_insole": np.array([])}

    # Initialize lists to accumulate sampled data
    y0_list: List[float] = []
    dz_list: List[float] = []
    cf_list: List[int] = []
    ci_list: List[int] = []

    # Sweep y0 from y_min to y_max in steps of y_step
    y0 = float(y_min)
    while y0 <= float(y_max):
        # Build boolean masks selecting points within cutter around y0 for foot and insole
        fmask = build_y_cutter_mask(fy, y0, cutter_width)
        imask = build_y_cutter_mask(iy, y0, cutter_width)

        # Extract Z coordinates for points inside current cutter
        fz = foot_pts[fmask, 2]
        iz = insole_pts[imask, 2]

        # Count number of points in cutter for foot and insole
        cf = int(fz.size)
        ci = int(iz.size)

        # Only proceed if both cutters have enough points for robust statistics
        if cf >= min_points_per_cutter and ci >= min_points_per_cutter:
            # Compute robust vertical position statistics for foot and insole
            foot_z = compute_z_stat(fz, foot_z_stat) # take the 5th percentile of foot Z inside this cutter (min can be noisy/outlier)
            insole_z = compute_z_stat(iz, insole_z_stat)

            # Compute vertical displacement dz = foot_z - insole_z
            dz = float(foot_z - insole_z)

            # Append results to lists
            y0_list.append(y0)
            dz_list.append(dz)
            cf_list.append(cf)
            ci_list.append(ci)

        # Increment y0 by step size (i.e. move cutter forward)
        y0 += float(y_step)

    # Convert accumulated lists to numpy arrays and return
    return {
        "y_samples": np.asarray(y0_list, dtype=float),
        "dz_samples": np.asarray(dz_list, dtype=float),
        "counts_foot": np.asarray(cf_list, dtype=int),
        "counts_insole": np.asarray(ci_list, dtype=int),
    }


def compute_z_stat(z: np.ndarray, mode: str) -> float:
    # Convert input z array to float for numeric computations
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        return float("nan")

    m = mode.lower()
    # Return minimum z value
    if m == "min":
        return float(np.min(z))
    # Return mean (average) z value
    if m in ("mean", "avg"):
        return float(np.mean(z))
    # Return median (50th percentile) z value
    if m in ("median", "p50"):
        return float(np.median(z))
    # Return 5th percentile z value (robust low-end statistic)
    if m == "p05":
        return float(np.percentile(z, 5))
    # Return 10th percentile z value (robust low-end statistic)
    if m == "p10":
        return float(np.percentile(z, 10))
    # Return 90th percentile z value (robust high-end statistic)
    if m == "p90":
        return float(np.percentile(z, 90))

    # Raise error if unknown mode requested
    raise ValueError(f"Unknown z_stat mode: {mode}")


# ============================================================
# Step B: build interpolation function f(y)
# ============================================================

def build_dz_interpolator(
    y_samples: np.ndarray,
    dz_samples: np.ndarray,
    kind: str = "linear",       # "linear" is safest; "cubic" can overshoot
    fill_mode: str = "clamp",   # "clamp" or "extrapolate"
) -> interp1d:
    """
    Build f(y) from sampled points.

    fill_mode:
      - "clamp": outside sampled y-range -> use boundary dz
      - "extrapolate": allow extrapolation (risk)
    """
    # Convert inputs to float numpy arrays for consistency
    y_samples = np.asarray(y_samples, dtype=float)
    dz_samples = np.asarray(dz_samples, dtype=float)

    # Require at least two samples to build interpolation
    if y_samples.size < 2:
        raise ValueError("Need at least 2 samples to build interpolator.")

    # Sort samples by y to ensure monotonic increasing order for interp1d
    order = np.argsort(y_samples)
    y = y_samples[order]
    dz = dz_samples[order]

    if fill_mode == "clamp":
        # Clamp fill_value uses boundary dz values outside interpolation range
        f = interp1d(
            y, dz,
            kind=kind,
            bounds_error=False,
            fill_value=(dz[0], dz[-1]),  # left and right boundary values
            assume_sorted=True,
        )
    elif fill_mode == "extrapolate":
        # Extrapolate allows linear extrapolation beyond sampled y range
        f = interp1d(
            y, dz,
            kind=kind,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
    else:
        raise ValueError("fill_mode must be 'clamp' or 'extrapolate'.")

    return f


# ============================================================
# Step C: apply warp to the foot
# ============================================================

def warp_foot_by_dz_function(
    foot_pcd: o3d.geometry.PointCloud,
    f_dz,  # callable y -> dz
    y_smooth_window: float = 0.0,  # optional additional smoothing by local average in Y
    return_displacements: bool = False,
) -> Tuple[o3d.geometry.PointCloud, Optional[np.ndarray]]:
    """
    Apply z_new = z_old - f_dz(y) to each foot point.

    If y_smooth_window > 0:
      for each point, use average dz over [y-w/2, y+w/2] sampled at a few points (cheap smoothing).
    """
    # Convert foot points to numpy array for processing
    pts = np.asarray(foot_pcd.points)

    # Handle empty point cloud gracefully by returning empty output
    if len(pts) == 0:
        out = o3d.geometry.PointCloud()
        if return_displacements:
            return out, np.zeros((0,), dtype=float)
        return out, None

    # Extract Y coordinates for all points
    y = pts[:, 1]

    if y_smooth_window > 0:
        # Smooth dz values by averaging over a local Y window around each point
        dz = smooth_dz_per_point(y, f_dz, window=float(y_smooth_window))
    else:
        # Directly evaluate dz function at each point's Y coordinate
        dz = np.asarray(f_dz(y), dtype=float)

    # Create new points array to hold warped coordinates
    new_pts = pts.copy()
    # Apply vertical displacement: move foot down by dz
    new_pts[:, 2] = new_pts[:, 2] - dz

    # Create new point cloud and assign warped points
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(new_pts)

    # Preserve colors if present in the original point cloud
    if foot_pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(foot_pcd.colors))
    # Preserve normals if present
    if foot_pcd.has_normals():
        out.normals = o3d.utility.Vector3dVector(np.asarray(foot_pcd.normals))

    # Return warped point cloud and optionally the displacement array
    if return_displacements:
        return out, dz
    return out, None


def smooth_dz_per_point(y: np.ndarray, f_dz, window: float, n_samples: int = 5) -> np.ndarray:
    """
    For each y_i, sample n_samples points within window and average f_dz.
    """
    y = np.asarray(y, dtype=float)

    # Half window width for symmetric sampling around each y_i
    half = 0.5 * window

    # Create offsets evenly spaced within [-half, half]
    offsets = np.linspace(-half, half, n_samples)

    # Initialize accumulator for dz values
    dz_acc = np.zeros_like(y, dtype=float)

    # Accumulate dz values sampled at offsets around each y_i
    for off in offsets:
        dz_acc += np.asarray(f_dz(y + off), dtype=float)

    # Average accumulated dz values to smooth
    return dz_acc / float(n_samples)