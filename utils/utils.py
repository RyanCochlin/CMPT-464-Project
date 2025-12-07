import mesh_to_sdf
import numpy as np
import trimesh
from skimage.measure import marching_cubes

def compute_sdf(mesh, num_points):
    points = np.random.uniform(-1.0, 1.0, size=(num_points, 3))
    values = mesh_to_sdf.mesh_to_sdf(mesh, points)
    return points, values


def get_grid_points(resolution):
    ix, iy, iz = np.indices((resolution, resolution, resolution))
    pts = (np.stack([ix, iy, iz], axis=-1).reshape(-1, 3) + 0.5) / resolution
    pts = pts * 2.0 - 1.0
    return pts

def compute_sdf_grid(mesh, resolution):
    points = get_grid_points(resolution)
    values = mesh_to_sdf.mesh_to_sdf(mesh, points)
    return points, values

def sample_pcd(mesh, num_points):
    pcd, _ = trimesh.sample.sample_surface(mesh, num_points)
    return pcd

def sdf_grid_to_mesh(sdf_grid):
    R = sdf_grid.shape[0]
    verts, faces, normals, _ = marching_cubes(
        volume=sdf_grid,
        level=0.0,
        spacing=(1.0, 1.0, 1.0)
    )
    scale = 1 / (R - 1)
    verts_normalized = verts * scale
    verts_world = verts_normalized * 2.0 - 1.0
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, vertex_normals=normals)
    return mesh