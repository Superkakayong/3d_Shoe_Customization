# generate_mesh.py
import numpy as np
import open3d as o3d
import copy

from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from matplotlib.path import Path


def transform_pre_cut_region(
    top_insole: o3d.geometry.PointCloud,
    arch: o3d.geometry.PointCloud,
    polygon_xy: np.ndarray,
    method: str = "linear",
    max_xy_dist: float = 8.0,         # mm: safety gate so we don't update far-away points
    return_region_pcd: bool = True,
):
    """
    Update ONLY the Z values of the ORIGINAL insole top points (insole_top_pcd),
    for points inside polygon_xy, so they match the arch_bottom_pcd surface.

    Returns:
        updated_top_insole: same point count / same XY as insole_top_pcd, but with updated Z in the pre-cut region
        updated_pre_cut_region: (optional) only the updated points, for visualization (i.e. a subset of the top_insole)
        update_mask: boolean mask over top_insole saying which points changed (True where Z was replaced)
    """
    if len(top_insole.points) == 0:
        empty = o3d.geometry.PointCloud()
        return empty, empty if return_region_pcd else None, np.zeros((0,), dtype=bool)

    if len(arch.points) == 0:
        # Nothing to project onto: return original as-is
        out = o3d.geometry.PointCloud(top_insole)
        mask = np.zeros((len(out.points),), dtype=bool)
        return out, o3d.geometry.PointCloud() if return_region_pcd else None, mask

    P = np.asarray(top_insole.points, dtype=float)   # (N,3)
    xy = P[:, :2]

    A = np.asarray(arch.points, dtype=float)  # (M,3)
    axy = A[:, :2]
    az = A[:, 2]

    # --- 1) inside polygon test in XY ---
    poly = Path(np.asarray(polygon_xy, dtype=float))
    inside = poly.contains_points(xy)   # (N,) bool

    # If no points are inside, return the original
    if not np.any(inside):
        out = o3d.geometry.PointCloud(top_insole)
        mask = np.zeros((len(P),), dtype=bool)
        return out, o3d.geometry.PointCloud() if return_region_pcd else None, mask

    # --- 2) distance gate: only update points that are close (in XY) to the arch data ---
    tree = cKDTree(axy)
    xy_inside = xy[inside]
    dists, _ = tree.query(xy_inside, k=1)
    near_inside = dists <= float(max_xy_dist)

    # indices in original point cloud that we will update
    inside_idx = np.where(inside)[0]
    update_idx = inside_idx[near_inside]

    update_mask = np.zeros((len(P),), dtype=bool)
    update_mask[update_idx] = True

    # If nothing passes the distance gate, return original
    if update_idx.size == 0:
        out = o3d.geometry.PointCloud(top_insole)
        return out, o3d.geometry.PointCloud() if return_region_pcd else None, update_mask

    # --- 3) interpolate arch Z at those XY locations ---
    xy_cand = xy[update_idx]

    # linear (or cubic) first
    z_interp = griddata(axy, az, xy_cand, method=method)

    # fill holes with the nearest
    nan_mask = np.isnan(z_interp)
    if np.any(nan_mask):
        z_near = griddata(axy, az, xy_cand[nan_mask], method="nearest")
        z_interp[nan_mask] = z_near

    # --- 4) apply Z update ---
    P_new = P.copy()
    P_new[update_idx, 2] = z_interp

    updated_top_insole = o3d.geometry.PointCloud()
    updated_top_insole.points = o3d.utility.Vector3dVector(P_new)

    # Preserve colors / normals if present (optional but nice)
    if top_insole.has_colors():
        updated_top_insole.colors = o3d.utility.Vector3dVector(np.asarray(top_insole.colors))
    if top_insole.has_normals():
        updated_top_insole.normals = o3d.utility.Vector3dVector(np.asarray(top_insole.normals))

    updated_pre_cut_region = None
    if return_region_pcd:
        region_pts = P_new[update_idx]
        updated_pre_cut_region = o3d.geometry.PointCloud()
        updated_pre_cut_region.points = o3d.utility.Vector3dVector(region_pts)

    return updated_top_insole, updated_pre_cut_region, update_mask

def poisson_reconstruction(input_mesh, depth: object = 8, scale: object = 1.1, linear_fit: object = False, n_threads: object = 4):
    mesh = copy.deepcopy(input_mesh)
    pcd = mesh.sample_points_poisson_disk(number_of_points=100000)

    dists = pcd.compute_nearest_neighbor_distance()
    avg_spacing = np.mean(dists)
    radius = 5 #5 * avg_spacing  # Start with 5× spacing # 0.005
    max_nn = 30
    k = 50

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=k)

    # Perform Poisson surface reconstruction
    print("starting mesh generation!")
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=scale, linear_fit=linear_fit, n_threads=n_threads)

    radii = [2.0, 4.0]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    # Remove low-density vertices (optional)
    # vertices_to_remove = densities < np.quantile(densities, 0.01) # Compute the 1st percentile
    # mesh.remove_vertices_by_mask(vertices_to_remove)

    # Clean up mesh (optional)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    print("finished mesh generation!")

    return mesh

def bpa_reconstruction(pcd):
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=50)

    print("Running BPA...")
    radii = [2.0, 4.0]  # tune based on spacing
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    return mesh