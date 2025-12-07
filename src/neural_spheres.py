"""Taken From CMPT464 Assignment Task 3"""
# Part of the code in adopted from DualSDF repository.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import torch.nn as nn
import trimesh
from utils.dgcnn import DGCNNFeat
from scipy.spatial import KDTree
from utils.utils import compute_sdf_grid, get_grid_points, sdf_grid_to_mesh


import matplotlib.pyplot as plt


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

    ### Your code here ###
    # Determine the SDF value of each query point with respect to each sphere ###
    sphere_sdf = []
    for sphere in sphere_params:
        center = sphere[:3]
        radius = sphere[3]
        sdf_values = torch.norm(query_points - center, dim=1) - radius
        sphere_sdf.append(sdf_values.unsqueeze(1)) 
    sphere_sdf = torch.cat(sphere_sdf, dim=1) 
    ### End of your code ###
    return sphere_sdf


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        in_ch = 256
        out_ch = 1024
        feat_ch = 512

        self.net1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
            nn.ReLU(inplace=True),
        )

        self.net2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, out_ch),
        )
        num_params = sum(p.numel() for p in self.parameters())
        print("[num parameters: {}]".format(num_params))

    def forward(self, z):
        in1 = z
        out1 = self.net1(in1)
        in2 = torch.cat([out1, in1], dim=-1)
        out2 = self.net2(in2)
        return out2


class SphereNet(nn.Module):
    def __init__(self, num_spheres=256):
        super(SphereNet, self).__init__()
        self.num_spheres = num_spheres
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder()

    def forward(self, surface_points, query_points):
        features = self.encoder(surface_points)
        sphere_params = self.decoder(features) 
        
        ### Comment on the following 4 lines, why do we have to do it?###
        sphere_params = torch.sigmoid(sphere_params.view(-1, 4))
        sphere_adder = torch.tensor([-0.5, -0.5, -0.5, 0.1]).to(sphere_params.device)
        sphere_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.4]).to(sphere_params.device)
        sphere_params = sphere_params * sphere_multiplier + sphere_adder

        sphere_sdf = determine_sphere_sdf(query_points, sphere_params)
        return sphere_sdf, sphere_params


def visualise_spheres(sphere_params, reference_model, save_path=None):
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
    """Visualise the SDF values as a point cloud."""
    # Use trimesh to create a point cloud from the SDF values
    inside_points = points[values < 0]
    outside_points = points[values > 0]
    inside_points = trimesh.points.PointCloud(inside_points)
    outside_points = trimesh.points.PointCloud(outside_points)
    inside_points.colors = [0, 0, 1, 1]  # Blue color for inside points
    outside_points.colors = [1, 0, 0, 1]  # Red color for outside points
    scene = trimesh.Scene()
    scene.add_geometry([inside_points, outside_points])
    scene.show()

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


def normalize_mesh(mesh, scale=1):
    vertices = mesh.vertices
    centroid = vertices.mean(axis=0)
    vertices -= centroid
    bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
    max_dim = bbox_size.max()
    vertices /= max_dim
    vertices *= scale
    mesh.vertices = vertices
    return centroid, max_dim


def unnormalize_mesh(mesh, centroid, max_dim, scale=1):
    vertices = mesh.vertices
    vertices /= scale
    vertices *= max_dim
    vertices += centroid
    mesh.vertices = vertices
    return mesh



def neural_sphere_reconstruction(
    mesh,
    resolution=50,
):


    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    centroid, max_dim = normalize_mesh(mesh)
    
    points, values = compute_sdf_grid(mesh, resolution=resolution)

    

    points = torch.from_numpy(points).float().to(device)
    values = torch.from_numpy(values).float().to(device)
    surface_pointcloud = trimesh.sample.sample_surface(mesh, 2048)[0]
    surface_pointcloud = torch.from_numpy(surface_pointcloud).float().to(device)

    model = SphereNet(num_spheres=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    losses = []
    # cds = []

    num_epochs = 500
    for i in range(num_epochs):
        optimizer.zero_grad()
        sphere_sdf, sphere_params = model(
            surface_pointcloud.unsqueeze(0).transpose(2, 1), points
        )
        ### Explain why the following line is necessary and what does it do###
        sphere_sdf = bsmin(sphere_sdf, dim=-1).to(device)

        ### Your code here ###
        ### Determine the loss function to train the model, i.e. the mean squared error between gt sdf field and predicted sdf field. ###
        ### Bonus: Design additional losses that helps to achieve a better result. ###
        mseloss = nn.MSELoss()(sphere_sdf.squeeze(), values)

        ### End of your code ###

        loss = mseloss
        loss.backward()
        optimizer.step()
        

        losses.append(loss.item())
        print(f"Iteration {i}, Loss: {loss.item()}")

    sphere_sdf = sphere_sdf.squeeze().reshape((resolution, resolution, resolution)).cpu().detach().numpy()
    reconstructed_mesh = sdf_grid_to_mesh(sphere_sdf)
    reconstructed_mesh = unnormalize_mesh(reconstructed_mesh, centroid, max_dim)
    return reconstructed_mesh, sphere_params

    


if __name__ == "__main__":
    mesh = trimesh.load("./data/dog/dog.obj")
    reconstructed_mesh, sphere_params = neural_sphere_reconstruction(
        mesh,
        resolution=50,
    )
    reconstructed_mesh.show()