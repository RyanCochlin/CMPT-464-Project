import numpy as np
import torch.nn as nn
import os
import trimesh
import mcubes
import vis_utils as vu

def visualize_pcd(points):
	scene = trimesh.Scene()
	if points is not None:
		scene.add_geometry(points)
	scene.show()

def main():
	dataset_name = "hand"

	#vu.visualize_model(dataset_name, "mesh")
	#vu.visualize_model(dataset_name, "sdf")
	vu.visualize_sphere_mesh(center=[0,0,0], radius=1.0)

if __name__ == "__main__":
	main()