import torch
import numpy as np
import torch.nn as nn
import os
import trimesh
from scipy.spatial import KDTree
import mcubes
from torch.nn import functional as F
from skimage.measure import marching_cubes

import matplotlib.pyplot as plt
from mesh_to_sdf import mesh_to_voxels, mesh_to_sdf

# os.environ["QT_QPA_PLATFORM"] = "xcb"



def sdf_grid_to_mesh(sdf_grid, bounds_min, bounds_max):
    """
    Convert a uniform 3D SDF grid back into a mesh using marching cubes.
    
    Args:
        sdf_grid: (R,R,R) numpy array of SDF values
        bounds_min: (3,) lower corner of the bounding box
        bounds_max: (3,) upper corner of the bounding box

    Returns:
        trimesh.Trimesh object
    """
    R = sdf_grid.shape[0]

    # Run marching cubes in *voxel coordinates*
    verts, faces, normals, values = marching_cubes(
        volume=sdf_grid,
        level=0.0,      # SDF zero-surface
        spacing=(1.0, 1.0, 1.0)
    )

    # # Convert voxel coords → world coords
    # bounds_min = np.asarray(bounds_min)
    # bounds_max = np.asarray(bounds_max)

    # # size of the cube
    # extents = bounds_max - bounds_min

    # voxel scaling
    scale = 1 / (R - 1)

    # voxel index -> world coordinate
    verts_world = verts * scale

    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, vertex_normals=normals)
    return mesh


def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix


def determine_sphere_sdf(query_points, sphere_params):
    """Query sphere sdf for a set of points.

    Args:
        query_points (torch.tensor): Nx3 tensor of query points.
        sphere_params (torch.tensor): Kx4 tensor of sphere parameters (center and radius).

    Returns:
        torch.tensor: Signed distance field of each sphere primitive with respect to each query point. NxK tensor.
    """

    # Determine the SDF value of each query point with respect to each sphere ###
    sphere_sdf = []
    for sphere in sphere_params:
        center = sphere[:3]
        radius = sphere[3]
        sdf_values = torch.norm(query_points - center, dim=1) - radius
        sphere_sdf.append(sdf_values.unsqueeze(1)) 
    sphere_sdf = torch.cat(sphere_sdf, dim=1) 
    return sphere_sdf


class SphereSet(nn.Module):
    """
    A PyTorch module that contains a set of learnable spheres.
    No input is required; the forward() method just returns the sphere parameters.
    """

    def __init__(self, num_spheres, 
                 init_radius_min=0.01, 
                 init_radius_max=0.1):
        super().__init__()

        self.num_spheres = num_spheres

        # ----- Random centers: uniform in [-0.5, 0.5] -----
        # These lie safely inside the box even after radius growth.
        center0 = torch.rand(num_spheres, 3) * 1.0 - 0.5  # → [-0.5, 0.5]

        # Convert to raw center so tanh(center_raw) = center0
        center_raw = torch.atanh(torch.clamp(center0, -0.999, 0.999))

        self.center_raw = nn.Parameter(center_raw)


        # ----- Random radii: uniform small range -----
        radius0 = torch.rand(num_spheres) * (init_radius_max - init_radius_min) + init_radius_min

        # Convert radius to raw (softplus inverse)
        radius_raw = torch.log(torch.exp(radius0) - 1.0)

        self.radius_raw = nn.Parameter(radius_raw)

    def forward(self, query_points):
        # Positive radius
        radius = F.softplus(self.radius_raw)  # (num_spheres,)

        # Constrain center so sphere stays inside [-1,1]
        # center ∈ [-(1 - r), 1 - r] in each dimension
        scale = 1.0 - radius.unsqueeze(1)  # (num_spheres, 1)
        center = torch.tanh(self.center_raw) * scale  # (num_spheres, 3)

        # Sphere params for downstream SDF computation
        sphere_params = torch.cat(
            [center, radius.unsqueeze(1)], dim=1
        )  # (num_spheres, 4)

        # Compute SDF for all query points
        sphere_sdf = determine_sphere_sdf(
            query_points=query_points,
            sphere_params=sphere_params,
        )  # (N, num_spheres)

        return sphere_sdf, sphere_params

def visualise_spheres(sphere_params, reference_model=None, save_path=None):
    sphere_params = sphere_params.cpu().detach().numpy()
    sphere_centers = sphere_params[..., :3]
    sphere_radii = np.abs(sphere_params[..., 3])
    scene = trimesh.Scene()
    if reference_model is not None:
        scene.add_geometry(reference_model)
    for center, radius in zip(sphere_centers, sphere_radii):
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        sphere.apply_translation(center)
        scene.add_geometry(sphere)
    if save_path is not None:
        scene.export(save_path)
    scene.show()


def visualise_sdf(points, values):
    #convert to point cloud
    #only show negative sdf values
    mask = values < 0
    points = points[mask]
    values = values[mask]
    print(points.min(), points.max())
    pcd = trimesh.points.PointCloud(points, colors=plt.get_cmap('jet')((values - values.min()) / (values.max() - values.min()))[:, :3]*255)
    scene = trimesh.Scene()
    scene.add_geometry(pcd)
    scene.show()

    return pcd

def sample_points_on_sphere(center, radius, num_points):
    """Sample points uniformly on the surface of a sphere."""
    points = []
    for _ in range(num_points):
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        u = np.random.uniform(0, 1)

        theta = np.arccos(costheta)
        r = radius * (u ** (1/3))

        x = r * np.sin(theta) * np.cos(phi) + center[0]
        y = r * np.sin(theta) * np.sin(phi) + center[1]
        z = r * np.cos(theta) + center[2]

        points.append([x, y, z])
    return np.array(points)

def chamfer_distance(points1, points2):
    """Compute the Chamfer Distance between two point clouds."""
    tree1 = KDTree(points1)
    tree2 = KDTree(points2)

    dist1, _ = tree1.query(points2)
    dist2, _ = tree2.query(points1)

    chamfer_dist = np.mean(dist1**2) + np.mean(dist2**2)
    return chamfer_dist

def voxels_to_points(voxels):
    R = voxels.shape[0]

    # integer index grid
    ix, iy, iz = np.indices((R, R, R))  # each in [0, R-1]

    # normalize coordinates to [-1, 1]
    # using voxel centers: (i + 0.5) / R ∈ (0,1)
    pts = (np.stack([ix, iy, iz], axis=-1).reshape(-1, 3) + 0.5) / R
    pts = pts * 2.0 - 1.0   # → [-1, 1]

    # flatten voxel values
    vals = voxels.reshape(-1).astype(float)

    return pts, vals

def get_grid_points(R):
    # integer index grid
    ix, iy, iz = np.indices((R, R, R))  # each in [0, R-1]

    # normalize coordinates to [-1, 1]
    # using voxel centers: (i + 0.5) / R ∈ (0,1)
    pts = (np.stack([ix, iy, iz], axis=-1).reshape(-1, 3) + 0.5) / R
    pts = pts * 2.0 - 1.0   # → [-1, 1]

    return pts

def normalize_mesh(mesh, scale=0.9):
    """Normalize mesh to fit inside a unit cube centered at the origin."""
    vertices = mesh.vertices
    centroid = vertices.mean(axis=0)
    vertices -= centroid
    bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
    max_dim = bbox_size.max()
    vertices /= max_dim
    vertices *= scale
    mesh.vertices = vertices
    return mesh

def plot_losses(losses):
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss over Iterations")
    plt.show()

def main():

    mesh_model = trimesh.load("data/dog/dog.obj")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    normalize_mesh(mesh_model, scale=1.6)

    res = 50

    points = get_grid_points(res)

    values = mesh_to_sdf(
        mesh_model,
        points)
    
    

    # voxels = mesh_to_voxels(
    #     mesh_model, 50, pad=False)
    
    # print(voxels.shape)

    # #pad voxels to 30x30x30
    # padded_voxels = np.zeros((55, 55, 55), dtype=voxels.dtype)
    # start = (55 - voxels.shape[0]) // 2
    # padded_voxels[start:start+voxels.shape[0], start:start+voxels.shape[1], start:start+voxels.shape[2]] = voxels
    # voxels = padded_voxels
    # print(voxels.shape)




    # res = voxels.shape
    # points, values = voxels_to_points(
    #     voxels
    # )

    points = torch.from_numpy(points).float().to(device)
    values = torch.from_numpy(values).float().to(device)

    
    # visualise_sdf(points.cpu().detach().numpy(), values.cpu().detach().numpy())
    # exit()


    model = SphereSet(num_spheres=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    losses = []
    # cds = []

    #invert values
    values = -values

    best_sdf = None

    # print(points.min(), points.max())
    # exit()

    num_epochs = 500
    for i in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        sphere_sdf, sphere_params = model(
             points
        )
        sphere_sdf = bsmin(sphere_sdf, dim=-1).squeeze()
        # sphere_sdf = torch.min(sphere_sdf, dim=-1).values.to(device)

        clamped_sphere_sdf = torch.clamp(sphere_sdf, 0, 0.1)
        values_clamped = torch.clamp(values, 0, 0.1)
        mseloss = nn.MSELoss()(clamped_sphere_sdf, values_clamped)
        # mseloss = nn.MSELoss()(sphere_sdf, values)

        best_sdf = -sphere_sdf

        reg = 0.001 * torch.mean(sphere_params[:,3])


        loss = mseloss + reg
        loss.backward()
        optimizer.step()
        

        losses.append(loss.item())
        print(f"Iteration {i}, Loss: {loss.item()}")

        
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    visualise_sdf(points.cpu().detach().numpy(), (-torch.abs(clamped_sphere_sdf)).cpu().detach().numpy())
    

    # visualise_spheres(sphere_params, reference_model=None, save_path=f"spheres.obj")
    # best_sdf = -best_sdf


    grid = best_sdf.cpu().detach().numpy().reshape(res, res, res)
    np.save(f"dog_sdf_grid.npy", grid)
    # grid = np.load(f"dog_sdf_grid.npy")

    grid[0,:,:] = 0.1
    grid[-1,:,:] = 0.1
    grid[:,0,:] = 0.1
    grid[:,-1,:] = 0.1
    grid[:,:,0] = 0.1
    grid[:,:,-1] = 0.1
    mesh = sdf_grid_to_mesh(
        grid,
        mesh_model.bounds[0] - 0.1,
        mesh_model.bounds[1] + 0.1
    )
    mesh.show()



if __name__ == "__main__":
    main()
