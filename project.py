import numpy as np
import torch.nn as nn
import os
import trimesh
from scipy.spatial import KDTree
import mcubes
import vis_utils as vu

def load_dataset(dataset_name, data_type):
	"""Load dataset based on the specified data type.

	Args:
		dataset_name (str): Name of the dataset
		data_type (str): Type of data to load ("mesh", "sdf", "pcd", "vox")

	Returns:
		Loaded data object
	"""
	match data_type:
		case "mesh":
			dataset_path = os.path.join("./data", dataset_name)
			mesh_path = os.path.join(dataset_path, f"{dataset_name}.obj")
			mesh = trimesh.load(mesh_path)
			return mesh
		case "sdf":
			dataset_path = os.path.join("./data", dataset_name)
			path = os.path.join(dataset_path, "voxel_and_sdf.npz")
			data = np.load(path)
			sdf_values = data["sdf_values"]
			sdf_points = data["sdf_points"]
			return sdf_values, sdf_points
		case "pcd":
			dataset_path = os.path.join("./data", dataset_name)
			path = os.path.join(dataset_path, "surface_points.ply")
			points = trimesh.load(path)
			return points

def compute_centroid(points):
	"""Compute the centroid of the given points.

	Args:
		points (np.Array): Point cloud data

	Returns:
		centroid (np.ndarray): Centroid of the points
	"""
	centroid = np.mean(points, axis=0)
	return centroid

def tetrahedron_volume(a, b, c, d):
	"""Compute the volume of a tetrahedron defined by four points.
	Args:
		a, b, c, d (np.ndarray): Four points defining the tetrahedron
	Returns:
		volume (float): Volume of the tetrahedron
	"""
	return np.abs(np.linalg.det(np.vstack([b-a, c-a, d-a]))) / 6.0

def compute_furthest_points(points, centroid, num_points=4, offset=0):
	"""Compute the furthest points from the centroid.

	Args:
		points (np.Array): Point cloud data
		centroid (np.ndarray): Centroid of the points
		num_points (int): Number of furthest points to return
		offset (int): Offset for selecting furthest points

	Returns:
		furthest_points (np.ndarray): Furthest points from the centroid
	"""
	diff = points - centroid
	dist = np.linalg.norm(diff, axis=1)
	ind = num_points + offset
	furthest_indices = np.argsort(dist)[-ind:-offset if offset != 0 else None]
	furthest_points = points[furthest_indices]
	return furthest_points

def sphere_residuals(sphere_params, points):
	"""Compute the residuals of the sphere parameters with respect to the points.

	Args:
		sphere_params (torch.Tensor): Sphere parameters (center and radius)
		points (torch.Tensor): Point cloud data

	Returns:
		residuals (torch.Tensor): Residuals of the sphere parameters
	"""
	center = sphere_params[:3]
	radius = sphere_params[3:]

	diff = points - center
	dist = np.linalg.norm(diff, axis=1)

	residuals = dist - radius
	return residuals
		
def sphere_jacobian(sphere_params, points):
	"""Compute the Jacobian of the sphere parameters with respect to the points.

	Args:
		sphere_params (torch.Tensor): Sphere parameters (center and radius)
		points (torch.Tensor): Point cloud data

	Returns:
		jacobian (torch.Tensor): Jacobian matrix
	"""
	center = sphere_params[:3]
	radius = sphere_params[3:]

	diff = points - center
	dist = np.linalg.norm(diff, axis=1, keepdims=True)

	jacobian_center = -diff / dist
	jacobian_radius = -np.ones_like(dist)

	jacobian = np.concatenate([jacobian_center, jacobian_radius], axis=1)
	return jacobian
		
def get_enclosing_sphere_data(points):
	"""Get center and radius of the enclosing sphere for the given points.

	Args:
		points (np.Array): Point cloud data

	Returns:
		center (np.ndarray): Center of the enclosing sphere
		radius (float): Radius of the enclosing sphere
	"""
	center = np.mean(points, axis=0)
	radius = np.max(np.linalg.norm(points - center, axis=1))
	return center, radius

def build_occupancy_from_sdf(sdf_points, sdf_values):
	"""Build occupancy grid from SDF values.

	Args:
		sdf_points (np.Array): Points corresponding to SDF values
		sdf_values (np.Array): SDF values

	Returns:
		occupancy (np.Array): Occupancy grid
	"""
	occupancy = np.zeros_like(sdf_values)
	occupancy[sdf_values <= 0] = 1
	return occupancy

def fit_sphere_ransac(sampled_points, all_points, occupancy):
	"""Fit a sphere to the given points using RANSAC.

	Args:
		sampled_points (np.Array): Sampled points for sphere fitting
		all_points (np.Array): All points in the point cloud
		occupancy (np.Array): Occupancy values for the points

	Returns:
		center (np.ndarray): Center of the fitted sphere
		radius (float): Radius of the fitted sphere
		found (bool): Whether a valid sphere was found
	"""
	# Initial guess
	params = np.array([0.0, 0.0, 0.0, 0.1])  # x, y, z, r
	point_occupied = all_points[occupancy == 1]
	found = True

	for i in range(100):
		residuals = sphere_residuals(params, sampled_points)
		jacobian = sphere_jacobian(params, sampled_points)

		lam = 1e-8
		JTJ = jacobian.T @ jacobian
		JTJ = JTJ + lam * np.eye(JTJ.shape[0]) # Levenberg-Marquardt damping
		JTr = jacobian.T @ residuals
		delta = -np.linalg.solve(JTJ, JTr).flatten()

		new_params = params + delta

		# reject sphere with occupied points inside
		if point_occupied.size > 0:
			diff = point_occupied - new_params[:3]
			dist = np.linalg.norm(diff, axis=1)
			if np.any(dist < new_params[3]):
				found = False
				break
		
		params = new_params

		if np.linalg.norm(delta) < 1e-4:
			break
	
	return params, found

def attamet_2():
	"""Idea:
	1. Randomly sample 2 points from unoccupied points
	2. Use these 2 points to define a sphere radius and center
	3. Check how many inliers (points inside sphere) we have
	4. Discard sphere if any inlier is occupied
	5. Store sphere parameters if valid and remove inliers from point cloud
	6. Repeat until desired number of spheres is found
	"""
	dataset_name = "dog"

	sdf_values, sdf_points = load_dataset(dataset_name, "sdf")
	occupancy = build_occupancy_from_sdf(sdf_points, sdf_values)
	selection_points = np.copy(sdf_points)
	selection_occupancy = np.copy(occupancy)
	spheres = 1000
	sphere_params = np.zeros((spheres, 4))  # x, y, z, r

	found_count = 0
	missed_count = 0
	while found_count < spheres:
		# early stop if too many misses
		if missed_count >= 100000:
			break

		# 2 random points
		non_occupied_indices = np.where(selection_occupancy == 0.0)[0]
		rand_indices = np.random.choice(non_occupied_indices, size=2, replace=False)
		point1 = selection_points[rand_indices[0]]
		point2 = selection_points[rand_indices[1]]

		# compute sphere params
		center = (point1 + point2) / 2.0
		radius = np.linalg.norm(point1 - point2) / 2.0 + 0.01  # small offset
		params = np.array([center[0], center[1], center[2], radius])

		# check inliers
		diff = selection_points - center
		dist = np.linalg.norm(diff, axis=1)
		inliers = dist < radius

		point_occupied = selection_points[selection_occupancy == 1]
		if point_occupied.size > 0:
			diff_occ = point_occupied - center
			dist_occ = np.linalg.norm(diff_occ, axis=1)
			if np.any(dist_occ < radius):
				missed_count += 1
				continue  # discard sphere

		sphere_params[found_count] = params
		found_count += 1
		print(f"Found sphere {found_count}: Center = {params[:3]}, Radius = {params[3]}")

		# remove inlier points
		test = selection_points[inliers]
		selection_points = selection_points[~inliers]
		selection_occupancy = selection_occupancy[~inliers]

	# create subtractvie CSG model
	center, radius = get_enclosing_sphere_data(sdf_points)
	model_params = np.array([center[0], center[1], center[2], radius])
	enclosing_sphere_sdf = sphere_residuals(model_params, sdf_points)

	for i in range(spheres):
		if sphere_params[i,3] <= 0 or sphere_params[i,3] >= radius:
			continue
		sphere_sdf = sphere_residuals(sphere_params[i], sdf_points)
		enclosing_sphere_sdf = np.maximum(enclosing_sphere_sdf, -sphere_sdf)

	final_model = enclosing_sphere_sdf

	vu.visualize_sdf(final_model, sdf_points)

def main():
	dataset_name = "dog"

	# test visualization functions
	#vu.visualize_model(dataset_name, "mesh")
	#vu.visualize_model(dataset_name, "sdf")
	#vu.visualize_sphere_by_points(center=[0,0,0], radius=1.0)

	sdf_values, sdf_points = load_dataset(dataset_name, "sdf")

	occupancy = build_occupancy_from_sdf(sdf_points, sdf_values)

	spheres = 3000
	k_val = 40
	inlier_threshold = 0.01
	sphere_params = np.zeros((spheres, 4))  # x, y, z, r
	selection_points = np.copy(sdf_points)
	selection_occupancy = np.copy(occupancy)
	tree = KDTree(selection_points)

	found_count = 0
	missed_count = 0
	while found_count < spheres:
		# 1 random point and it's 4 nearest neighbors
		non_occupied_indices = np.where(selection_occupancy == 0.0)[0]
		rand_indices = np.random.choice(non_occupied_indices, size=1, replace=False)
		query_point = selection_points[rand_indices[0]]
		distance, indices = tree.query(query_point, k=k_val)
		sampled_points = selection_points[indices]

		# This doesn't quite work yet, but the idea is to avoid coplanar points by selecting the furthest points from the centroid
		# This should give better results from the Jacobian-based optimization but theres a problem with the points still being clustered
		# The fix should be to select furthest points on each side of the centroid rather than just overall furthest points

		# get centeroid and find 4 furthest points from sampled points	
		# centroid = compute_centroid(sampled_points)
		# offset = 0
		# while True:
		# 	sphere_points = compute_furthest_points(sampled_points, centroid, num_points=40, offset=offset)
		# 	# check if points are not coplanar
		# 	vol = tetrahedron_volume(sphere_points[0], sphere_points[1], sphere_points[2], sphere_points[3])
		# 	offset += 1
		# 	if vol > 1e-6:
		# 		break

		# visualize the selected points for testing
		# cloud = trimesh.points.PointCloud(sphere_points, colors=[255, 255, 0, 255])
		# selection_points_cloud = trimesh.points.PointCloud(selection_points, colors=[255, 0, 0, 10])
		# sampled_points_cloud = trimesh.points.PointCloud(sampled_points, colors=[0, 0, 255, 100])
		# scene = trimesh.Scene()
		# scene.add_geometry(cloud)
		# scene.add_geometry(selection_points_cloud)
		# scene.add_geometry(sampled_points_cloud)
		# scene.show()

		#params, found = fit_sphere_ransac(sphere_points, sdf_points, occupancy)
		params, found = fit_sphere_ransac(sampled_points, sdf_points, occupancy)
		if not found:
			missed_count += 1
			continue
		
		# shrink nearest neighbor count if too many misses
		if(missed_count >= 500):
			k_val = max(k_val - 5, 5)
		
		missed_count = 0
		sphere_params[found_count] = params
		found_count += 1
		print(f"Found sphere {found_count}: Center = {params[:3]}, Radius = {params[3]}")

		# remove inlier points
		residuals = sphere_residuals(params, selection_points)
		inliers = np.abs(residuals) < inlier_threshold
		selection_points = selection_points[~inliers]
		selection_occupancy = selection_occupancy[~inliers]

		tree = KDTree(selection_points)
		if selection_points.shape[0] < 10:
			break

	# create subtractvie CSG model
	center, radius = get_enclosing_sphere_data(sdf_points)
	model_params = np.array([center[0], center[1], center[2], radius])
	enclosing_sphere_sdf = sphere_residuals(model_params, sdf_points)

	# had an idea to also do this sphere rejection by radius inside the RANSAC loop instead of here to ensure we get a full count of spheres
	# still need to test if this improves results though
	for i in range(spheres):
		if sphere_params[i,3] <= 0 or sphere_params[i,3] >= radius:
			continue
		sphere_sdf = sphere_residuals(sphere_params[i], sdf_points)
		enclosing_sphere_sdf = np.maximum(enclosing_sphere_sdf, -sphere_sdf)

	final_model = enclosing_sphere_sdf

	vu.visualize_sdf(final_model, sdf_points)


if __name__ == "__main__":
	method = "attamet_2"

	match method:
		case "attamet_2":
			attamet_2()
		case "main":
			main()