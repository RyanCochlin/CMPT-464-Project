""" pip install
torch
numpy
trimesh
mcubes
scikit-learn
tqdm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import trimesh
import mcubes
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm


DATASET_NAME = "dog"
NUM_PRIMITIVES = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
NUM_STEPS = 3000
CLIP_GRAD_NORM = 1.0
INPUT_PC_PATH = f"./data/{DATASET_NAME}/surface_points.ply"
INPUT_SDF_PATH = f"./data/{DATASET_NAME}/voxel_and_sdf.npz"
OUTPUT_DIR = f"./output/"
OUTPUT_MESH_PATH = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_final.obj")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def quat_to_rot_matrix(quat):
    """ Differentiable quaternion [w, x, y, z] to 3x3 rot matrix. """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    n = torch.dot(quat, quat)
    if n == 0:
        return torch.eye(3, device=device)
    
    q = quat / torch.sqrt(n)
    w, x, y, z = q[0], q[1], q[2], q[3]

    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return torch.stack([
        torch.stack([1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)]),
        torch.stack([2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)]),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)])
    ])


def rot_matrix_to_quat(m):
    """ Differentiable 3x3 rot matrix to quaternion [w, x, y, z]. """
    m00, m01, m02 = m[0, 0], m[0, 1], m[0, 2]
    m10, m11, m12 = m[1, 0], m[1, 1], m[1, 2]
    m20, m21, m22 = m[2, 0], m[2, 1], m[2, 2]

    tr = m00 + m11 + m22
    
    w_cond = tr > 0
    x_cond = (m00 > m11) & (m00 > m22)
    y_cond = m11 > m22

    S_w = torch.sqrt(tr + 1.0) * 2
    w_A = 0.25 * S_w
    x_A = (m21 - m12) / S_w
    y_A = (m02 - m20) / S_w
    z_A = (m10 - m01) / S_w

    S_x = torch.sqrt(1.0 + m00 - m11 - m22) * 2
    w_B = (m21 - m12) / S_x
    x_B = 0.25 * S_x
    y_B = (m01 + m10) / S_x
    z_B = (m02 + m20) / S_x

    S_y = torch.sqrt(1.0 + m11 - m00 - m22) * 2
    w_C = (m02 - m20) / S_y
    x_C = (m01 + m10) / S_y
    y_C = 0.25 * S_y
    z_C = (m12 + m21) / S_y

    S_z = torch.sqrt(1.0 + m22 - m00 - m11) * 2
    w_D = (m10 - m01) / S_z
    x_D = (m02 + m20) / S_z
    y_D = (m12 + m21) / S_z
    z_D = 0.25 * S_z

    w = torch.where(w_cond, w_A, torch.where(x_cond, w_B, torch.where(y_cond, w_C, w_D)))
    x = torch.where(w_cond, x_A, torch.where(x_cond, x_B, torch.where(y_cond, x_C, x_D)))
    y = torch.where(w_cond, y_A, torch.where(x_cond, y_B, torch.where(y_cond, y_C, y_D)))
    z = torch.where(w_cond, z_A, torch.where(x_cond, z_B, torch.where(y_cond, z_C, z_D)))
    
    return torch.stack([w, x, y, z])


def superquadric_implicit(points, scale, shape):
    eps = 1e-5
    e1 = torch.clamp(shape[0], 0.2, 2.0)
    e2 = torch.clamp(shape[1], 0.2, 2.0)
    scale = torch.clamp(torch.abs(scale), min=1e-4)
    
    p = points / scale
    p_x, p_y, p_z = p[:, 0], p[:, 1], p[:, 2]
    
    term1 = (torch.abs(p_x)**(2.0 / e2) + torch.abs(p_y)**(2.0 / e2) + eps)**(e2 / e1)
    term2 = torch.abs(p_z)**(2.0 / e1)
    
    implicit_val = term1 + term2 - 1
    return implicit_val


class PrimitiveModel(nn.Module):
    def __init__(self, init_params):
        super(PrimitiveModel, self).__init__()
        
        # Unpack initial parameters
        init_trans = init_params['translations']
        init_rots = init_params['rotations']
        init_scales = init_params['scales']
        init_shapes = init_params['shapes']
        
        self.num_primitives = init_trans.shape[0]

        # Convert initial rotation matrices to quaternions
        init_quats = torch.stack([rot_matrix_to_quat(m) for m in init_rots])

        # Create nn.Parameters from these smart initial values
        self.translations = nn.Parameter(init_trans)
        self.quaternions = nn.Parameter(init_quats)
        self.scales = nn.Parameter(init_scales)
        self.shapes = nn.Parameter(init_shapes)

    def forward(self, points):
        """ Calculates the SDF of the union of all primitives. """
        self.quaternions.data = F.normalize(self.quaternions.data, p=2, dim=1)
        
        all_sdfs = []
        for i in range(self.num_primitives):
            trans = self.translations[i]
            rot_matrix = quat_to_rot_matrix(self.quaternions[i])
            scale = self.scales[i]
            shape = self.shapes[i]
            
            local_points = (points - trans) @ rot_matrix.T
            sdf_i = superquadric_implicit(local_points, scale, shape)
            all_sdfs.append(sdf_i)
        
        sdf_stack = torch.stack(all_sdfs, dim=1)
        union_sdf, _ = torch.min(sdf_stack, dim=1)
        return union_sdf


def load_data(path):
    print(f"Loading data from {path}.")
    data = trimesh.load(path)
    
    if isinstance(data, trimesh.PointCloud):
        points = data.vertices
    
    centroid = np.mean(points, axis=0)
    points_normalized = points - centroid
    scale = np.max(np.linalg.norm(points_normalized, axis=1))
    points_normalized = points_normalized / scale
    
    unnormalize_info = (centroid, scale)
    return points_normalized, unnormalize_info


def get_initialization(points, num_primitives):
    """
    Uses K-Means and PCA to get a good starting guess for primitive parameters.
    """
    print(f"Running k-Means and PCA for initialization.")
    
    kmeans = KMeans(n_clusters=num_primitives, random_state=0, n_init=10)
    labels = kmeans.fit_predict(points)
    
    init_trans = []
    init_rots = []
    init_scales = []
    init_shapes = []

    for i in range(num_primitives):
        cluster_points = points[labels == i]
        if cluster_points.shape[0] < 3:
            continue
            
        pca = PCA(n_components=3)
        pca.fit(cluster_points)
        
        # Translation
        center = pca.mean_
        init_trans.append(torch.tensor(center, dtype=torch.float32))
        
        # Rotation
        rot_matrix = pca.components_
        if np.linalg.det(rot_matrix) < 0:
            rot_matrix[2, :] *= -1
        init_rots.append(torch.tensor(rot_matrix, dtype=torch.float32))

        # Scale
        scales = 2.0 * np.sqrt(pca.explained_variance_)
        scales = np.maximum(scales, 1e-3)
        init_scales.append(torch.tensor(scales, dtype=torch.float32))
        
        # Shape
        init_shapes.append(torch.tensor([1.0, 1.0], dtype=torch.float32))

    init_params = {
        'translations': torch.stack(init_trans).to(device),
        'rotations': torch.stack(init_rots).to(device),
        'scales': torch.stack(init_scales).to(device),
        'shapes': torch.stack(init_shapes).to(device),
    }
    
    print(f"Initialized {len(init_trans)} valid primitives.")
    return init_params


def calculate_iou(model, sdf_gt_path):
    data = np.load(sdf_gt_path)
    sdf_points_gt = data["sdf_points"]
    sdf_values_gt = data["sdf_values"]

    gt_points_tensor = torch.tensor(sdf_points_gt, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        sdf_values_pred = model(gt_points_tensor)
    
    voxels_gt = (sdf_values_gt <= 0)
    voxels_pred = (sdf_values_pred.cpu().numpy() <= 0)
    
    intersection = np.sum(voxels_gt & voxels_pred)
    union = np.sum(voxels_gt | voxels_pred)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def export_mesh(model, unnormalize_info, sdf_gt_path, path):
    model.eval()
    data = np.load(sdf_gt_path)

    grid_res = None
    grid_points = None
    
    if "voxels" in data and data["voxels"].ndim == 3:
        grid_shape = data["voxels"].shape
        if grid_shape[0] == grid_shape[1] == grid_shape[2]:
            grid_res = grid_shape[0]

    elif "sdf_points" in data:
        N = data["sdf_points"].shape[0]
        res_guess = int(round(N**(1/3.0)))
        if res_guess**3 == N:
            grid_res = res_guess
            grid_points = torch.tensor(data["sdf_points"], dtype=torch.float32).to(device)

    t = torch.linspace(-1, 1, grid_res)
    x, y, z = torch.meshgrid(t, t, t, indexing='ij')
    if grid_points is None:
        grid_points = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=1).to(device)

    sdf_grid = []
    for p in torch.split(grid_points, 100000):
        with torch.no_grad():
            sdf_grid.append(model(p).cpu())
    
    sdf_grid = torch.cat(sdf_grid, dim=0).reshape(grid_res, grid_res, grid_res).numpy()
    
    vertices, triangles = mcubes.marching_cubes(sdf_grid, 0.0)

    centroid, scale = unnormalize_info
    vertices_normalized = vertices / (grid_res - 1) * 2.0 - 1.0
    vertices_world = vertices_normalized * scale + centroid
    
    mesh = trimesh.Trimesh(vertices=vertices_world, faces=triangles)
    mesh.export(path)
    print(f"Exported mesh to {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    points_gt_np, unnormalize_info = load_data(INPUT_PC_PATH)
    points_gt_torch = torch.tensor(points_gt_np, dtype=torch.float32).to(device)
    
    # 2. Get Initialization (K-Means + PCA)
    init_params = get_initialization(points_gt_np, NUM_PRIMITIVES)
    
    # 3. Initialize Model and Optimizer
    model = PrimitiveModel(init_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 4. --- OPTIMIZATION LOOP ---
    pbar = tqdm(range(NUM_STEPS), desc="Optimizing", ncols=150)
    for _ in pbar:
        model.train()
        optimizer.zero_grad()
        
        sdf_pred = model(points_gt_torch)
        loss = torch.abs(sdf_pred).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
        optimizer.step()
        
        pbar.set_postfix(loss=f"{loss.item():.6f}")
    
    # 5. --- FINAL EVALUATION ---
    model.eval()
    with torch.no_grad():
        sdf_final = model(points_gt_torch)
        loss_final = torch.abs(sdf_final).mean().item()
    
    iou_final = calculate_iou(model, INPUT_SDF_PATH)
    
    print(f"Loss: {loss_final:<8.6f}")
    print(f"IoU: {iou_final:<8.4f}")

    # 6. --- EXPORT FINAL MESH ---
    export_mesh(model, unnormalize_info, INPUT_SDF_PATH, OUTPUT_MESH_PATH)


if __name__ == "__main__":
    main()
