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
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Usage

### Computing The Reconstructions

Process multiple models with a single command:

```bash
python bulk_run.py --method gd --num_spheres 512 --resolution 50 --epochs 500
python bulk_run.py --method kmeans_ransac
python bulk_run.py --method knn_ransac
python bulk_run.py --method superquadrics --num_primitives 20 --steps 3000
```

**Arguments:**
- `--dataset_dir`: Path to dataset directory (default: `data`)
- `--output_dir`: Path to output directory (default: `results`)
- `--method`: Reconstruction method (`gd`, `kmeans_ransac`, `knn_ransac`, `superquadrics`)
- `--num_spheres`: Number of spheres for GD method (default: 512)
- `--num_primitives`: Number of primitives for superquadrics (default: 20)
- `--resolution`: Grid resolution for GD method (default: 50)
- `--epochs`: Number of epochs for GD method (default: 500)
- `--steps`: Number of optimization steps for superquadrics (default: 3000)

### Evaluation
