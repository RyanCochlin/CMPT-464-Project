import trimesh
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='data', help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Path to the output directory')
    parser.add_argument('--cd', action='store_true', help='Compute Chamfer Distance')
    parser.add_argument('--emd', action='store_true', help='Compute Earth Mover\'s Distance')
    args = parser.parse_args()

    models = []
    for model_name in os.listdir(args.dataset_dir):
        obj_path = os.path.join(args.dataset_dir, model_name, f"{model_name}.obj")
        voxel_sdf_path = os.path.join(args.dataset_dir, model_name, "voxel_and_sdf.npz")
        pcd_path = os.path.join(args.dataset_dir, model_name, f"{model_name}_pcd.npy")
        models.append({
            'name': model_name,
            'obj_path': obj_path,
            'voxel_sdf_path': voxel_sdf_path,
            'pcd_path': pcd_path
        })


    
        
    