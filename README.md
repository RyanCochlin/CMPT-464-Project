# CMPT 464 Project - 3D Shape Reconstruction

This project implements three different methods for 3D shape reconstruction using primitive-based representations.

## Methods

### 1. Gradient Descent Sphere Fitting (GD)
Uses gradient descent optimization to fit a collection of spheres to a mesh using signed distance functions.

**Features:**
- Differentiable sphere parameters (centers and radii)
- Smooth minimum approximation using log-sum-exp
- MSE loss with radius regularization
- Outputs reconstructed mesh in original coordinate frame

### 2. K-Means RANSAC Sphere Fitting
Uses clustering and RANSAC to fit negative spheres for subtractive CSG modeling.

**Two variants:**
- **K-Means RANSAC**: Clusters unoccupied points and fits spheres to each cluster
- **K-NN RANSAC**: Uses k-nearest neighbors with Levenberg-Marquardt optimization

**Features:**
- Iterative sphere fitting with inlier removal
- Occupancy-based validation
- Subtractive CSG model construction

### 3. Superquadrics
Fits superquadric primitives using k-means initialization and gradient descent optimization.

**Features:**
- K-means + PCA for initialization
- Learnable translation, rotation (quaternion), scale, and shape parameters
- Union of primitives via minimum SDF
- Outputs reconstructed mesh

## Installation

```bash
conda create -n cmpt464 python=3.10
conda activate cmpt464
pip install -r requirements.txt
```

## Usage

### Computing The Reconstructions

Process multiple models with a single command:

```bash
python bulk_run.py --method gd --num_spheres 512 --resolution 50 --epochs 500
python bulk_run.py --method kmeans_ransac
python bulk_run.py --method superquadrics --num_primitives 20 --steps 3000
```

**Arguments:**
- `--dataset_dir`: Path to dataset directory (default: `data`)
- `--output_dir`: Path to output directory (default: `results`)
- `--method`: Reconstruction method (`gd`, `kmeans_ransac`, `knn_ransac`, `superquadrics`)
- `--num_spheres`: Number of spheres for GD method (default: 512)
- `--num_primitives`: Number of primitives for superquadrics (default: 20)
- `--resolution`: Grid resolution for GD method (default: 50)
- `--epochs`: Number of epochs for GD method (default: 1000)
- `--steps`: Number of optimization steps for superquadrics (default: 3000)

### Evaluation

Evaluate reconstruction quality using multiple distance metrics:

```bash
# Evaluate all methods with all metrics
python evaluate.py --cd --emd --hd

# Evaluate specific method
python evaluate.py --method gd --cd --emd --hd

# Evaluate with custom sampling
python evaluate.py --num_samples 20000 --cd --emd
```

**Arguments:**
- `--dataset_dir`: Path to dataset directory (default: `data`)
- `--output_dir`: Path to output directory (default: `results`)
- `--method`: Method to evaluate - `all`, `gd`, `kmeans_ransac`, or `superquadrics` (default: `all`)
- `--cd`: Compute Chamfer Distance
- `--emd`: Compute Earth Mover's Distance (approximate)
- `--hd`: Compute Hausdorff Distance
- `--num_samples`: Number of points to sample from meshes (default: 10000)

**Metrics:**
- **Chamfer Distance (CD)**: Average bidirectional nearest-neighbor distance between point clouds
- **Earth Mover's Distance (EMD)**: Approximated using Hungarian algorithm on subsampled points
- **Hausdorff Distance (HD)**: Maximum of directed Hausdorff distances in both directions

**Output:**
Results are saved to `results/evaluation_results.json` containing:
- Individual results for each model and method
- Summary statistics (mean, std, min, max) for each metric per method
