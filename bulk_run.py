import argparse
import os
import sys
import numpy as np
import trimesh

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gd_fitting import reconstruct_mesh_from_spheres
from kmeans_ransac import reconstruct_with_kmeans_ransac
from superquadric import reconstruct_mesh_with_superquadrics
from neural_spheres import neural_sphere_reconstruction



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='data', help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Path to the output directory')
    parser.add_argument('--method', type=str, choices=['gd', 'kmeans_ransac', 'knn_ransac', 'superquadrics', 'neural_spheres'],nargs='+', required=True, help='Reconstruction method to use')
    parser.add_argument('--num_spheres', type=int, default=512, help='Number of spheres for GD method')
    parser.add_argument('--num_primitives', type=int, default=20, help='Number of primitives for superquadrics')
    parser.add_argument('--resolution', type=int, default=50, help='Grid resolution for GD method')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for GD method')
    parser.add_argument('--steps', type=int, default=3000, help='Number of steps for superquadrics')
    args = parser.parse_args()

    models = []
    for model_name in os.listdir(args.dataset_dir):
        model_dir = os.path.join(args.dataset_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
            
        obj_path = os.path.join(model_dir, f"{model_name}.obj")
        voxel_sdf_path = os.path.join(model_dir, "voxel_and_sdf.npz")
        surface_points_path = os.path.join(model_dir, "surface_points.ply")
        
        if os.path.exists(obj_path):
            models.append({
                'name': model_name,
                'obj_path': obj_path,
                'voxel_sdf_path': voxel_sdf_path,
                'surface_points_path': surface_points_path
            })

    
    
    print(f"Found {len(models)} models to process")
    print(f"Using method: {args.method}")

    if "gd" in args.method:
        os.makedirs(os.path.join(args.output_dir, "gd"), exist_ok=True)
        for i, model in enumerate(models):
            print(f"\n[{i+1}/{len(models)}] Processing {model['name']} with gradient descent method...")
            mesh = trimesh.load(model['obj_path'])
            reconstructed_mesh, sphere_params = reconstruct_mesh_from_spheres(
                mesh,
                num_spheres=args.num_spheres,
                resolution=args.resolution,
                num_epochs=args.epochs,
                verbose=True
            )
            
            output_path = os.path.join(args.output_dir,"gd", f"{model['name']}.obj")
            reconstructed_mesh.export(output_path)
            print(f"Saved to {output_path}")

    if "kmeans_ransac" in args.method:
        os.makedirs(os.path.join(args.output_dir, "kmeans_ransac"), exist_ok=True)
        for i, model in enumerate(models):
            print(f"\n[{i+1}/{len(models)}] Processing {model['name']} with k-means RANSAC method...")
            mesh = trimesh.load(model['obj_path'])
            
            mesh = reconstruct_with_kmeans_ransac(
                mesh,
                num_spheres=args.num_spheres,
                k_clusters=40,
                # grid_resolution=args.resolution,
                verbose=True
            )
            
            output_path = os.path.join(args.output_dir, "kmeans_ransac", f"{model['name']}.obj")
            mesh.export(output_path)
            print(f"Saved to {output_path}")

    if "neural_spheres" in args.method:
        os.makedirs(os.path.join(args.output_dir, "neural_spheres"), exist_ok=True)
        for i, model in enumerate(models):
            print(f"\n[{i+1}/{len(models)}] Processing {model['name']} with neural spheres method...")
            mesh = trimesh.load(model['obj_path'])
            
            reconstructed_mesh, sphere_params = neural_sphere_reconstruction(
                mesh,
                resolution=args.resolution,
            )
            
            output_path = os.path.join(args.output_dir, "neural_spheres", f"{model['name']}.obj")
            reconstructed_mesh.export(output_path)
            print(f"Saved to {output_path}")

    if "superquadrics" in args.method:
        os.makedirs(os.path.join(args.output_dir, "superquadrics"), exist_ok=True)
        for i, model in enumerate(models):
            print(f"\n[{i+1}/{len(models)}] Processing {model['name']} with superquadrics method...")

            mesh = trimesh.load(model['obj_path'])

            
            reconstructed_mesh, sq_model = reconstruct_mesh_with_superquadrics(
                mesh,
                num_primitives=args.num_primitives,
                num_steps=args.steps,
                grid_resolution=args.resolution,
                verbose=True
            )
            
            output_path = os.path.join(args.output_dir, "superquadrics", f"{model['name']}.obj")
            reconstructed_mesh.export(output_path)
            print(f"Saved to {output_path}")
    
    print("\nProcessing complete!")
