import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh
from scipy.spatial import KDTree
from utils.vis_utils import *
from utils.utils import compute_sdf

def k_means_clustering(points, k, max_iters=100):
	idx = np.random.choice(points.shape[0], size=k, replace=False)
	cluster_centers = points[idx]

	for iteration in range(max_iters):
		diff = points[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :]
		dist = np.linalg.norm(diff, axis=2)
		labels = np.argmin(dist, axis=1)

		for i in range(k):
			cluster_points = points[labels == i]
			if len(cluster_points) > 0:
				new_center = np.mean(cluster_points, axis=0)

				if np.allclose(new_center, cluster_centers[i]):
					return cluster_centers, labels
				
				cluster_centers[i] = new_center
		
	return cluster_centers, labels
	

def sphere_residuals(sphere_params, points):
	center = sphere_params[:3]
	radius = sphere_params[3:]
	diff = points - center
	dist = np.linalg.norm(diff, axis=1)
	residuals = dist - radius
	return residuals


def sphere_jacobian(sphere_params, points):
	center = sphere_params[:3]
	e = 1e-8
	diff = points - center
	dist = np.linalg.norm(diff, axis=1, keepdims=True)
	dist = np.maximum(dist, e)

	jacobian_center = -diff / dist
	jacobian_radius = -np.ones_like(dist)
	jacobian = np.concatenate([jacobian_center, jacobian_radius], axis=1)
	return jacobian


def get_enclosing_sphere_data(points):
	center = np.mean(points, axis=0)
	radius = np.max(np.linalg.norm(points - center, axis=1))
	return center, radius


def build_occupancy_from_sdf(sdf_points, sdf_values):
	occupancy = np.zeros_like(sdf_values)
	occupancy[sdf_values <= 0] = 1
	return occupancy

def fit_sphere_levenberg_marquardt(sampled_points, all_points, occupancy, max_iters=100):
	params = np.array([0.0, 0.0, 0.0, 0.1])
	point_occupied = all_points[occupancy == 1]
	found = True

	for i in range(max_iters):
		residuals = sphere_residuals(params, sampled_points)
		jacobian = sphere_jacobian(params, sampled_points)

		lam = 1e-8
		JTJ = jacobian.T @ jacobian
		JTJ = JTJ + lam * np.eye(JTJ.shape[0])
		JTr = jacobian.T @ residuals
		delta = -np.linalg.solve(JTJ, JTr)

		new_params = params + delta

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


def fit_sphere_ransac(sampled_points, all_points, occupancy, num_iterations=1000):
	best_params = None
	best_inlier_count = 0

	point_occupied = all_points[occupancy == 1]
	for i in range(num_iterations):
		rand_indices = np.random.choice(sampled_points.shape[0], size=2, replace=False)
		point1 = sampled_points[rand_indices[0]]
		point2 = sampled_points[rand_indices[1]]

		center = (point1 + point2) / 2.0
		radius = np.linalg.norm(point1 - point2) / 2.0
		params = np.array([center[0], center[1], center[2], radius])

		diff = sampled_points - center
		dist = np.linalg.norm(diff, axis=1)
		inliers = dist < radius
		inlier_count = np.sum(inliers)

		if inlier_count > best_inlier_count:
			if point_occupied.size > 0:
				diff_occ = point_occupied - center
				dist_occ = np.linalg.norm(diff_occ, axis=1)
				if np.any(dist_occ < radius):
					continue

			best_inlier_count = inlier_count
			best_params = params

		if best_inlier_count > sampled_points.shape[0] * 0.9:
			break

	found = best_params is not None
	return best_params, found

def normalize_mesh(mesh):
    #normalize to [-0.5,0.5]
    bbox = mesh.bounding_box.extents
    max_dim = np.max(bbox)
    centroid = mesh.bounding_box.centroid
    mesh.apply_translation(-centroid)
    mesh.apply_scale(1.0 / max_dim)
    return centroid, max_dim

def unnormalize_mesh(mesh, centroid, max_dim):
    mesh.apply_scale(max_dim)
    mesh.apply_translation(centroid)
    return mesh

def reconstruct_with_kmeans_ransac(
	mesh,
	num_spheres=2000,
	k_clusters=50,
	ransac_iterations=1000,
	inlier_threshold=0.01,
	min_cluster_size=4,
	verbose=True,
	sdf_n_points=100000
):
	centroid, max_dim = normalize_mesh(mesh)
	sdf_points, sdf_values = compute_sdf(mesh, num_points=sdf_n_points)
	occupancy = build_occupancy_from_sdf(sdf_points, sdf_values)
	sphere_params = np.zeros((num_spheres, 4))
	selection_points = np.copy(sdf_points)
	selection_occupancy = np.copy(occupancy)

	non_occupied_indices = np.where(selection_occupancy == 0.0)[0]
	selection_points = selection_points[non_occupied_indices]
	selection_occupancy = selection_occupancy[non_occupied_indices]

	found_count = 0
	while found_count < num_spheres:
		k = min(k_clusters, selection_points.shape[0])
		cluster_centers, point_labels = k_means_clustering(selection_points, k)

		start_found_count = found_count
		missed_count = 0
		
		for i in range(k):
			cluster_points = selection_points[point_labels == i]

			if cluster_points.shape[0] < min_cluster_size:
				missed_count += 1
				continue

			params, found = fit_sphere_ransac(cluster_points, sdf_points, occupancy, ransac_iterations)
			if not found:
				missed_count += 1
				if verbose:
					print(f"Sphere fitting failed, skipping with {missed_count} misses.")
				continue
			
			if verbose:
				print(f"Found sphere {found_count}: Center = {params[:3]}, Radius = {params[3]}")


			sphere_params[found_count] = params
			found_count += 1
			if found_count >= num_spheres:
				break

		for i in range(start_found_count, found_count):
			params = sphere_params[i]
			residuals = sphere_residuals(params, selection_points)
			inliers = residuals < inlier_threshold
			selection_points = selection_points[~inliers]
			selection_occupancy = selection_occupancy[~inliers]

		if found_count >= num_spheres or missed_count >= k or selection_points.shape[0] < 10:
			break

	center, radius = get_enclosing_sphere_data(sdf_points)
	model_params = np.array([center[0], center[1], center[2], radius])
	enclosing_sphere_sdf = sphere_residuals(model_params, sdf_points)

	for i in range(found_count):
		if sphere_params[i, 3] <= 0 or sphere_params[i, 3] >= radius:
			continue
		sphere_sdf = sphere_residuals(sphere_params[i], sdf_points)
		enclosing_sphere_sdf = np.maximum(enclosing_sphere_sdf, -sphere_sdf)

	mesh = extract_mesh_from_sdf(enclosing_sphere_sdf, sdf_points, resolution=64)

	mesh = unnormalize_mesh(mesh, centroid, max_dim)

	return mesh


def main():
	dataset_name = "dog"
	dataset_path = os.path.join("./data", dataset_name)
	mesh_path = os.path.join(dataset_path, f"{dataset_name}.obj")
	mesh_model = trimesh.load(mesh_path)	
	reconstructed_mesh = reconstruct_with_kmeans_ransac(
		mesh_model,
		num_spheres=512,
		k_clusters=40,
		verbose=True
	)
	reconstructed_mesh.show()


if __name__ == "__main__":
	main()