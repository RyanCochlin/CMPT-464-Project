# CMPT 464 Project - 3D Shape Reconstruction

This project implements three different methods for 3D shape reconstruction using primitive-based representations.



## Installation

```bash
conda create -n cmpt464 python=3.10
conda activate cmpt464
pip install -r requirements.txt
```

## Usage

### Computing The Reconstructions

Process a single mesh:
```bash
python src/subtractive_spheres.py --input_obj path/to/input.obj --output_obj path/to/output.obj
```

**Arguments:**
- `--input_obj`: Path to input OBJ file (required)
- `--output_obj`: Path to output OBJ file (required)
- `--num_spheres`: Number of spheres (default: 512)
- `--resolution`: SDF grid resolution (default: 50)
- `--epochs`: Number of optimization epochs (default: 1000)


Process multiple models with a single command:

```bash
python bulk_run.py --method subtractive --num_spheres 512 --resolution 50 --epochs 500
python bulk_run.py --method kmeans_ransac
python bulk_run.py --method superquadrics --num_primitives 20 --steps 3000
```

**Arguments:**
- `--dataset_dir`: Path to dataset directory (default: `data`)
- `--output_dir`: Path to output directory (default: `results`)
- `--method`: Reconstruction method (`subtractive`,`neural_spheres`, `kmeans_ransac`, `superquadrics`)
- `--num_spheres`: Number of spheres for subtractive spheres method (default: 512)
- `--num_primitives`: Number of primitives for superquadrics (default: 20)
- `--resolution`: Grid resolution for subtractive spheres method (default: 50)
- `--epochs`: Number of epochs for subtractive spheres method (default: 1000)
- `--steps`: Number of optimization steps for superquadrics (default: 3000)

### Evaluation

Evaluate reconstruction quality using multiple distance metrics:

```bash
# Evaluate all methods with all metrics
python evaluate.py 

# Evaluate specific method
python evaluate.py --method subtractive --cd --emd --hd

# Evaluate with custom sampling
python evaluate.py --num_samples 20000 --cd --emd
```

**Arguments:**
- `--dataset_dir`: Path to dataset directory (default: `data`)
- `--output_dir`: Path to output directory (default: `results`)
- `--method`: Method to evaluate - `all`, `subtractive`, `kmeans_ransac`, or `superquadrics` (default: `all`)
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
