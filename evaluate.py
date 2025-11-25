import trimesh
import argparse
import os
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import linear_sum_assignment
import json


def sample_points_from_mesh(mesh, num_points=10000):
    """Sample points uniformly from mesh surface."""
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def chamfer_distance(points1, points2):
    """
    Compute Chamfer Distance between two point clouds.
    
    Args:
        points1: Nx3 array of points
        points2: Mx3 array of points
    
    Returns:
        Chamfer distance (average of bidirectional distances)
    """
    # For each point in points1, find nearest point in points2
    from scipy.spatial import cKDTree
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    # Distance from points1 to points2
    dist1, _ = tree2.query(points1)
    # Distance from points2 to points1
    dist2, _ = tree1.query(points2)
    
    # Chamfer distance is the mean of both directions
    cd = (np.mean(dist1) + np.mean(dist2)) / 2.0
    return cd


def earth_movers_distance(points1, points2, num_samples=1000):
    """
    Compute approximation of Earth Mover's Distance (Wasserstein distance).
    Uses Hungarian algorithm on subsampled point sets for efficiency.
    
    Args:
        points1: Nx3 array of points
        points2: Mx3 array of points
        num_samples: Number of points to subsample for efficiency
    
    Returns:
        Approximate EMD
    """
    # Subsample if needed
    if len(points1) > num_samples:
        idx = np.random.choice(len(points1), num_samples, replace=False)
        points1 = points1[idx]
    if len(points2) > num_samples:
        idx = np.random.choice(len(points2), num_samples, replace=False)
        points2 = points2[idx]
    
    # Ensure same number of points
    min_len = min(len(points1), len(points2))
    points1 = points1[:min_len]
    points2 = points2[:min_len]
    
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    cost_matrix = cdist(points1, points2)
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    emd = cost_matrix[row_ind, col_ind].mean()
    
    return emd


def hausdorff_distance(points1, points2):
    """
    Compute Hausdorff Distance between two point clouds.
    
    Args:
        points1: Nx3 array of points
        points2: Mx3 array of points
    
    Returns:
        Hausdorff distance (maximum of directed Hausdorff distances)
    """
    hd1 = directed_hausdorff(points1, points2)[0]
    hd2 = directed_hausdorff(points2, points1)[0]
    return max(hd1, hd2)


def evaluate_reconstruction(gt_mesh, recon_mesh, num_samples=10000, compute_metrics=None):
    """
    Evaluate reconstruction quality using multiple metrics.
    
    Args:
        gt_mesh: Ground truth trimesh object
        recon_mesh: Reconstructed trimesh object
        num_samples: Number of points to sample from each mesh
        compute_metrics: Dict indicating which metrics to compute
    
    Returns:
        Dictionary of metric values
    """
    if compute_metrics is None:
        compute_metrics = {'cd': True, 'emd': True, 'hd': True}
    
    # Sample points from both meshes
    gt_points = sample_points_from_mesh(gt_mesh, num_samples)
    recon_points = sample_points_from_mesh(recon_mesh, num_samples)
    
    results = {}
    
    if compute_metrics.get('cd', False):
        results['chamfer_distance'] = chamfer_distance(gt_points, recon_points)
    
    if compute_metrics.get('emd', False):
        results['earth_movers_distance'] = earth_movers_distance(gt_points, recon_points, num_samples=1000)
    
    if compute_metrics.get('hd', False):
        results['hausdorff_distance'] = hausdorff_distance(gt_points, recon_points)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate 3D reconstruction quality')
    parser.add_argument('--dataset_dir', type=str, default='data', help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Path to the output directory')
    parser.add_argument('--method', type=str, default='all', 
                        choices=['all', 'gd', 'kmeans_ransac', 'superquadrics'],
                        help='Method to evaluate (default: all methods)')
    parser.add_argument('--cd', action='store_true', help='Compute Chamfer Distance')
    parser.add_argument('--emd', action='store_true', help='Compute Earth Mover\'s Distance')
    parser.add_argument('--hd', action='store_true', help='Compute Hausdorff Distance')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of points to sample from meshes')
    args = parser.parse_args()
    
    # If no metrics specified, compute all
    if not (args.cd or args.emd or args.hd):
        args.cd = args.emd = args.hd = True
    
    compute_metrics = {
        'cd': args.cd,
        'emd': args.emd,
        'hd': args.hd
    }
    
    # Get list of methods to evaluate
    if args.method == 'all':
        methods = ['gd', 'kmeans_ransac', 'superquadrics']
    else:
        methods = [args.method]
    
    # Filter methods that have results
    available_methods = []
    for method in methods:
        method_dir = os.path.join(args.output_dir, method)
        if os.path.exists(method_dir) and len(os.listdir(method_dir)) > 0:
            available_methods.append(method)
        else:
            print(f"Warning: No results found for method '{method}' in {method_dir}")
    
    if not available_methods:
        print("Error: No reconstruction results found to evaluate.")
        exit(1)
    
    # Get list of models
    models = []
    for model_name in sorted(os.listdir(args.dataset_dir)):
        model_path = os.path.join(args.dataset_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        
        # Look for ground truth mesh in the data directory
        gt_obj_path = None
        for ext in ['.obj', '.stl', '.ply']:
            potential_path = os.path.join(model_path, f"{model_name}{ext}")
            if os.path.exists(potential_path):
                gt_obj_path = potential_path
                break
        
        if gt_obj_path:
            models.append({
                'name': model_name,
                'gt_path': gt_obj_path
            })
    
    if not models:
        print("Error: No models found in dataset directory.")
        exit(1)
    
    print(f"Evaluating {len(models)} models using {len(available_methods)} method(s)")
    print(f"Metrics: {', '.join([k for k, v in compute_metrics.items() if v])}")
    print("=" * 80)
    
    # Store all results
    all_results = {}
    
    for method in available_methods:
        print(f"\nEvaluating method: {method}")
        print("-" * 80)
        
        method_results = {}
        
        for model in models:
            model_name = model['name']
            gt_path = model['gt_path']
            recon_path = os.path.join(args.output_dir, method, f"{model_name}.obj")
            
            if not os.path.exists(recon_path):
                print(f"  {model_name}: Reconstruction not found, skipping")
                continue
            
            try:
                # Load meshes
                gt_mesh = trimesh.load(gt_path, force='mesh')
                recon_mesh = trimesh.load(recon_path, force='mesh')
                
                # Evaluate
                results = evaluate_reconstruction(gt_mesh, recon_mesh, args.num_samples, compute_metrics)
                method_results[model_name] = results
                
                # Print results
                print(f"  {model_name}:")
                for metric, value in results.items():
                    print(f"    {metric}: {value:.6f}")
            
            except Exception as e:
                print(f"  {model_name}: Error - {str(e)}")
                continue
        
        all_results[method] = method_results
    
    # Compute and display summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    summary_statistics = {}
    
    for method in available_methods:
        if method not in all_results or not all_results[method]:
            continue
        
        print(f"\n{method.upper()}:")
        method_results = all_results[method]
        
        # Aggregate metrics
        metric_names = set()
        for model_results in method_results.values():
            metric_names.update(model_results.keys())
        
        method_summary = {}
        for metric in sorted(metric_names):
            values = [r[metric] for r in method_results.values() if metric in r]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                print(f"  {metric}:")
                print(f"    mean: {mean_val:.6f} ± {std_val:.6f}")
                print(f"    min:  {min_val:.6f}")
                print(f"    max:  {max_val:.6f}")
                
                # Store summary statistics
                method_summary[metric] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'min': float(min_val),
                    'max': float(max_val)
                }
        
        summary_statistics[method] = method_summary
    
    # Prepare output data with both individual results and summary
    output_data = {
        'individual_results': all_results,
        'summary_statistics': summary_statistics
    }
    
    # Save results to JSON
    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")