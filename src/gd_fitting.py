import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.nn as nn
import trimesh
from torch.nn import functional as F
from skimage.measure import marching_cubes
from mesh_to_sdf import mesh_to_sdf
from utils.utils import compute_sdf_grid, get_grid_points


def sdf_grid_to_mesh(sdf_grid):
    R = sdf_grid.shape[0]
    verts, faces, normals, _ = marching_cubes(
        volume=sdf_grid,
        level=0.0,
        spacing=(1.0, 1.0, 1.0)
    )
    scale = 1 / (R - 1)
    verts_normalized = verts * scale
    verts_world = verts_normalized * 2.0 - 1.0
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, vertex_normals=normals)
    return mesh


def bsmin(a, dim, k=22.0, keepdim=False):
    return -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k


def compute_sphere_sdf(query_points, sphere_params):
    sphere_sdf = []
    for sphere in sphere_params:
        center = sphere[:3]
        radius = sphere[3]
        sdf_values = torch.norm(query_points - center, dim=1) - radius
        sphere_sdf.append(sdf_values.unsqueeze(1))
    return torch.cat(sphere_sdf, dim=1)


class SphereSet(nn.Module):
    def __init__(self, num_spheres, init_radius_min=0.01, init_radius_max=0.1):
        super().__init__()
        self.num_spheres = num_spheres
        
        center0 = torch.rand(num_spheres, 3) * 1.0 - 0.5
        center_raw = torch.atanh(torch.clamp(center0, -0.999, 0.999))
        self.center_raw = nn.Parameter(center_raw)
        
        radius0 = torch.rand(num_spheres) * (init_radius_max - init_radius_min) + init_radius_min
        radius_raw = torch.log(torch.exp(radius0) - 1.0)
        self.radius_raw = nn.Parameter(radius_raw)

    def forward(self, query_points):
        radius = F.softplus(self.radius_raw) + 0.01
        scale = 1.0 - radius.unsqueeze(1)
        center = torch.tanh(self.center_raw) * scale
        
        sphere_params = torch.cat([center, radius.unsqueeze(1)], dim=1)
        sphere_sdf = compute_sphere_sdf(query_points, sphere_params)
        
        return sphere_sdf, sphere_params





def normalize_mesh(mesh, scale=0.9):
    vertices = mesh.vertices
    centroid = vertices.mean(axis=0)
    vertices -= centroid
    bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
    max_dim = bbox_size.max()
    vertices /= max_dim
    vertices *= scale
    mesh.vertices = vertices
    return centroid, max_dim


def unnormalize_mesh(mesh, centroid, max_dim, scale=0.9):
    vertices = mesh.vertices
    vertices /= scale
    vertices *= max_dim
    vertices += centroid
    mesh.vertices = vertices
    return mesh


def reconstruct_mesh_from_spheres(
    mesh,
    num_spheres=512,
    resolution=50,
    num_epochs=1000,
    learning_rate=0.05,
    regularization=0.001,
    mesh_scale=1.6,
    clamp_value=0.1,
    device=None,
    verbose=True,
    output_resolution=100
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Using device: {device}")
    
    centroid, max_dim = normalize_mesh(mesh, scale=mesh_scale)
    
    points, values = compute_sdf_grid(mesh, resolution=resolution)
    
    points = torch.from_numpy(points).float().to(device)
    values = torch.from_numpy(values).float().to(device)
    values = -values
    
    model = SphereSet(num_spheres=num_spheres).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    band_width_nb = 0.04 #0.05
    global_weight = 0.05
    
    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        
        sphere_sdf, sphere_params = model(points)
        sphere_sdf = bsmin(sphere_sdf, dim=-1).squeeze()
        
        # Original MSE loss with clamping. I kept it here for reference.
        #clamped_sphere_sdf = torch.clamp(sphere_sdf, 0, clamp_value)
        #values_clamped = torch.clamp(values, 0, clamp_value)
        #mse_loss = nn.MSELoss()(clamped_sphere_sdf, values_clamped)

        # narrow band loss based on method outlined here https://microsites.arinex.com.au/EMBC/pdf/full-paper_167.pdf
        # Combined with weak global loss to encourage overall shape matching
        near_mask = values.abs() < band_width_nb
        loss_near = F.mse_loss(sphere_sdf[near_mask], values[near_mask])
        loss_global = global_weight * F.mse_loss(sphere_sdf, values)
        loss_sign = 0.001 * torch.mean((torch.sign(sphere_sdf) - torch.sign(values))**2)
        
        reg_loss = regularization * torch.mean(sphere_params[:, 3])
        loss = loss_near + loss_global + loss_sign + reg_loss
        
        loss.backward()
        optimizer.step()
        
        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}")
    
    best_sdf = -sphere_sdf

    grid_points = get_grid_points(output_resolution)
    sdf = compute_sphere_sdf(torch.from_numpy(grid_points).float(), sphere_params.to('cpu'))
    best_sdf = -bsmin(sdf, dim=-1).detach().numpy().reshape(output_resolution, output_resolution, output_resolution)
    best_sdf[0, :, :] = clamp_value
    best_sdf[-1, :, :] = clamp_value
    best_sdf[:, 0, :] = clamp_value
    best_sdf[:, -1, :] = clamp_value
    best_sdf[:, :, 0] = clamp_value
    best_sdf[:, :, -1] = clamp_value    
    reconstructed_mesh = sdf_grid_to_mesh(best_sdf)
    reconstructed_mesh = unnormalize_mesh(reconstructed_mesh, centroid, max_dim, scale=mesh_scale)
    return reconstructed_mesh, sphere_params

    # grid = best_sdf.cpu().detach().numpy().reshape(resolution, resolution, resolution)
    
    # grid[0, :, :] = clamp_value
    # grid[-1, :, :] = clamp_value
    # grid[:, 0, :] = clamp_value
    # grid[:, -1, :] = clamp_value
    # grid[:, :, 0] = clamp_value
    # grid[:, :, -1] = clamp_value
    
    # reconstructed_mesh = sdf_grid_to_mesh(grid)
    # reconstructed_mesh = unnormalize_mesh(reconstructed_mesh, centroid, max_dim, scale=mesh_scale)
    
    # return reconstructed_mesh, sphere_params


def main():
    mesh_model = trimesh.load("data/dog/dog.obj")
    
    reconstructed_mesh, sphere_params = reconstruct_mesh_from_spheres(
        mesh_model,
        num_spheres=512,
        resolution=50,
        num_epochs=1000,
        verbose=True
    )
    
    reconstructed_mesh.show()


if __name__ == "__main__":
    main()
