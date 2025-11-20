import numpy as np
import os
import trimesh
import mcubes
from scipy.interpolate import griddata

# Global flag to enable/disable visualization
# this is so we don't have to remove visualization calls throughout the code
g_enable_vis = True

def extract_mesh_from_sdf(sdf_values, sdf_points, resolution=256):
	"""Extract mesh from SDF using PyMCubes marching cubes.
	
	Args:
		sdf_values (torch.tensor): SDF values
		resolution (int): Grid resolution
		bounds (tuple): Grid bounds
		
	Returns:
		trimesh.Trimesh: Extracted mesh
	"""
	# Reshape SDF values to 3D grid
	xmin, ymin, zmin = sdf_points.min(axis=0)
	xmax, ymax, zmax = sdf_points.max(axis=0)
	x_lin = np.linspace(xmin, xmax, resolution)
	y_lin = np.linspace(ymin, ymax, resolution)
	z_lin = np.linspace(zmin, zmax, resolution)
	X, Y, Z = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
	grid = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
	sdf_grid = griddata(sdf_points, sdf_values, grid, method='linear', fill_value=1.0)
	sdf_grid = sdf_grid.reshape((resolution, resolution, resolution))
	
	# Use PyMCubes marching cubes to extract mesh
	vertices, faces = mcubes.marching_cubes(sdf_grid, 0.0)
	
	mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
	mesh.invert()
	return mesh

def visualize_mesh(mesh, scene=None):
	if not g_enable_vis:
		return
	
	ret_scene = True
	if scene is None:
		scene = trimesh.Scene()
		ret_scene = False
	if mesh is not None:
		scene.add_geometry(mesh)
	
	if(ret_scene):
		return scene
	else:
		scene.show()

def visualize_ocu(ocu_values, scene=None):
	if not g_enable_vis:
		return
	
	ret_scene = True
	if scene is None:
		scene = trimesh.Scene()
		ret_scene = False

	if ocu_values is not None:
		scene.add_geometry(ocu_values)
	
	if(ret_scene):
		return scene
	else:
		scene.show()

def visualize_sdf(sdf_values, sdf_points, resolution=128, scene=None):
	if not g_enable_vis:
		return
	
	sdf_model = extract_mesh_from_sdf(sdf_values, sdf_points, resolution)

	ret_scene = True
	if scene is None:
		scene = trimesh.Scene()
		ret_scene = False

	if sdf_values is not None:
		scene.add_geometry(sdf_model)
	
	if(ret_scene):
		return scene
	else:
		scene.show()

def visualize_pcd(points, color=(0, 0, 255, 255), scene=None):
	if not g_enable_vis:
		return
	
	colors = np.array([color] * len(points.vertices))
	points.visual.vertex_colors = colors

	ret_scene = True
	if scene is None:
		scene = trimesh.Scene()
		ret_scene = False

	if points is not None:
		scene.add_geometry(points)
	
	if(ret_scene):
		return scene
	else:
		scene.show()

def visualize_pointcloud(points, colors=None, scene=None):
	if not g_enable_vis:
		return
	
	if colors is None:
		colors = np.array([[0, 0, 255, 255]] * len(points))
	point_cloud = trimesh.points.PointCloud(points, colors=colors)

	ret_scene = True
	if scene is None:
		scene = trimesh.Scene()
		ret_scene = False

	if point_cloud is not None:
		scene.add_geometry(point_cloud)
	
	if(ret_scene):
		return scene
	else:
		scene.show()

def visualize_sphere_by_params(center, radius, color=(255, 0, 0, 100), scene=None):
	if not g_enable_vis:
		return
	
	sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
	sphere.apply_translation(center)
	colors = np.array([color] * len(sphere.vertices))
	sphere.visual.vertex_colors = colors

	ret_scene = True
	if scene is None:
		scene = trimesh.Scene()
		ret_scene = False

	if sphere is not None:
		scene.add_geometry(sphere)
	
	if(ret_scene):
		return scene
	else:
		scene.show()

def visualize_model(dataset_name, vis_type):
	"""Visualize provided data models by data type.

	Args:
		dataset_name (str): Name of the dataset
		vis_type (str): Type of data to visualize ("mesh", "sdf", "pcd", "vox")
	"""
	match vis_type:
		case "mesh":
			dataset_path = os.path.join("./data", dataset_name)
			mesh_path = os.path.join(dataset_path, f"{dataset_name}.obj")
			mesh = trimesh.load(mesh_path)
			visualize_mesh(mesh)
		case "ocu":
			dataset_path = os.path.join("./data", dataset_name)
			path = os.path.join(dataset_path, "sdf_visualization.ply")
			sdf_values = trimesh.load(path)
			visualize_ocu(sdf_values)
		case "pcd":
			dataset_path = os.path.join("./data", dataset_name)
			path = os.path.join(dataset_path, "surface_points.ply")
			points = trimesh.load(path)
			visualize_pcd(points)
		case "sdf":
			dataset_path = os.path.join("./data", dataset_name)
			path = os.path.join(dataset_path, "voxel_and_sdf.npz")
			data = np.load(path)
			sdf_values = data["sdf_values"]
			sdf_points = data["sdf_points"]
			visualize_sdf(sdf_values, sdf_points)
		case "vox":
			print("Voxel visualization not implemented yet.")
		case _:
			raise ValueError(f"Visualization type {vis_type} not supported.")