import pickle as pkl
import os
import torch
import numpy as np
import sys
from tqdm import tqdm

from pytorch3d.structures import Meshes
from pytorch3d.io import IO
from pytorch3d.structures.meshes import join_meshes_as_scene
import argparse
import trimesh


if True:
    HOOD_PROJECT = os.path.dirname(__file__)
    saved_folder = os.path.join(HOOD_PROJECT, 'exp/example_simulation')
    # motion_path = '/is/cluster/fast/sbian/data/bedlam_motion_for_blender/male_31_us_1190_0022_300.npz'      
    lower_garment_obj = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v5/0/lower_garmentcode/valid_garment_target_long_pants_wb/valid_garment_target_long_pants_wb/valid_garment_target_long_pants_wb_sim.obj'
    upper_garment_obj = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v5/0/upper_garmentcode/valid_garment_longshirt/valid_garment_longshirt/valid_garment_longshirt_sim.obj'
    rest_smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params.pkl'
    obj_path = os.path.join(saved_folder, 'combined_garment.obj')
    npz_path = 'exp/example_simulation/output_anypose.npz'
    
    saved_folder_final = os.path.join(saved_folder, 'motion_0')

    mesh_saved_path = os.path.join(saved_folder_final, 'meshes')
    if not os.path.exists(mesh_saved_path):
        os.makedirs(mesh_saved_path)

    print('mesh_saved_path', mesh_saved_path)
    traj_dict = np.load(npz_path)
    
    len_mesh_upper_vertices = traj_dict['start_vertex_indices']
    len_mesh_upper_faces = traj_dict['start_face_indices']


start_index = 2

pos = traj_dict['pred']
cloth_faces = traj_dict['cloth_faces']

if len(cloth_faces.shape) == 3:
    cloth_faces = cloth_faces[0]

pos = torch.from_numpy(pos).float()
cloth_faces = torch.from_numpy(cloth_faces).long()

seq_len = pos.shape[0]

all_verts = []
frame_indices = [i for i in range(start_index, seq_len, 30)]

for i in frame_indices:
    upper_verts = pos[i][:len_mesh_upper_vertices]
    upper_faces = cloth_faces[:len_mesh_upper_faces]

    mesh_cloth = Meshes([upper_verts], [upper_faces])

    IO().save_mesh(
        mesh_cloth, 
        os.path.join(mesh_saved_path, f"upper_{i-start_index}.obj")
    )

    lower_verts = pos[i][len_mesh_upper_vertices:]
    lower_faces = cloth_faces[len_mesh_upper_faces:] - len_mesh_upper_vertices

    mesh_cloth = Meshes([lower_verts], [lower_faces])

    IO().save_mesh(
        mesh_cloth, 
        os.path.join(mesh_saved_path, f"lower_{i-start_index}.obj")
    )


