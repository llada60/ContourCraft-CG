import os
import sys

HOOD_PROJECT = os.path.dirname(__file__)
HOOD_DATA = "/is/cluster/fast/sbian/github/HOOD/hood_data"

os.environ["HOOD_PROJECT"] = HOOD_PROJECT
os.environ["HOOD_DATA"] = HOOD_DATA

sys.path.insert(1, HOOD_PROJECT)

from utils.mesh_creation import obj2template
from utils.common import pickle_dump
from utils.defaults import DEFAULTS
from pathlib import Path
import random

from utils.arguments import load_params, create_modules
from utils.common import move2device
from utils.io import pickle_dump
from utils.defaults import DEFAULTS
from pathlib import Path
import torch
from utils.arguments import create_runner
import pickle as pkl
from utils.mesh_creation import add_pinned_verts_single_template
import argparse
import yaml
import trimesh
import numpy as np
from utils.anypose_utils import get_rest_garemnt_info_easy, calculate_smpl_based_transform_warp_easy, calculate_pinned_v_dense
from runners.smplx.body_models import SMPL, SMPLXLayer
from pytorch3d.structures import Meshes
from pytorch3d.ops.knn import knn_points, knn_gather
from pytorch3d.io import load_obj, IO, load_ply
from pytorch3d.structures.meshes import join_meshes_as_scene
from utils.datasets import make_fromanypose_dataloader
from utils.validation import apply_material_params, apply_material2_params
from utils.common import random_between_log

device = 'cuda'

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_random_params():
    lame_mu_min = 15909
    lame_mu_max = 63636
    lame_lambda_min = 3535.414406069427
    lame_lambda_max = 93333.73508005822
    bending_coeff_min = 6.370782056371576e-08
    bending_coeff_max = 0.0013139737991266374
    density_min = 4.34e-2
    density_max = 7e-1

    config_dict = dict()
    size = [1]
    config_dict['density'] = random_between_log(density_min, density_max, size).item()
    config_dict['lame_mu'] = random_between_log(lame_mu_min, lame_mu_max, size).item()
    config_dict['lame_lambda'] = random_between_log(lame_lambda_min, lame_lambda_max, size).item()
    config_dict['bending_coeff'] = random_between_log(bending_coeff_min, bending_coeff_max, size).item()
    config_dict['smpl_model'] = None

    return config_dict



smplx_layer = SMPLXLayer(
    '/home/ids/liliu/data/body_models/models/smplx/SMPLX_NEUTRAL.pkl',
    ext='pkl',
    num_betas=300
).to(device)


def get_pinned_verts(garment_rest_verts, rest_smplx_params, device, lower_garment=False):
    garment_rest_verts = torch.from_numpy(garment_rest_verts).to(device).float().unsqueeze(0) * 0.01

    ##### calculate the smpl-based vertices
    rest_A, rest_smpl_V, smpl_based_lbs_weight = get_rest_garemnt_info_easy(
        rest_smplx_params, smplx_layer, garment_rest_verts
    )

    joints_rest = torch.from_numpy(rest_smplx_params['joints']).float().reshape(-1, 3).to(device)
    if 'smplx_vertices' in rest_smplx_params:
        rest_smpl_V = torch.from_numpy(rest_smplx_params['smplx_vertices']).float().reshape(-1, 3).to(device)
    else:
        rest_smpl_V = torch.from_numpy(rest_smplx_params['vertices']).float().reshape(-1, 3).to(device)
    
    pinned_vertices, closest_idx_on_smpl = calculate_pinned_v_dense(
        garment_rest_verts.reshape(-1, 3), None, joints_rest, rest_smpl_V, lower_garment=lower_garment
    )

    return pinned_vertices, closest_idx_on_smpl


if __name__ == "__main__":
    setup_seed(0)

    saved_folder = os.path.join(HOOD_PROJECT, 'exp/example_simulation')
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
        
    motion_path = '../assets/eval/male_31_us_1190_0022_300.npz'
    lower_garment_obj = '../assets/eval/valid_garment_target_long_pants_wb_sim.obj'
    upper_garment_obj = '../assets/eval/valid_garment_longshirt_sim.obj'
    rest_smplx_params_path = '../assets/eval/aaa_mesh_registrarion/registered_params.pkl'
    obj_path = os.path.join(saved_folder, 'combined_garment.obj')

    if True:
        with open(rest_smplx_params_path, 'rb') as f:
            rest_smplx_params = pkl.load(f)

        mesh_upper = IO().load_mesh(upper_garment_obj, include_textures=False)
        mesh_lower = IO().load_mesh(lower_garment_obj, include_textures=False)

        mesh_upper  = trimesh.Trimesh(vertices=mesh_upper.verts_packed().numpy(), faces=mesh_upper.faces_packed().numpy(), process=True)
        mesh_lower  = trimesh.Trimesh(vertices=mesh_lower.verts_packed().numpy(), faces=mesh_lower.faces_packed().numpy(), process=True)

        mesh_upper_vertices = mesh_upper.vertices
        mesh_lower_vertices = mesh_lower.vertices

        pinned_verts_up, closest_idx_up = get_pinned_verts(mesh_upper_vertices, rest_smplx_params, 'cuda')
        pinned_verts_low, closest_idx_low = get_pinned_verts(mesh_lower_vertices, rest_smplx_params, 'cuda', lower_garment=True)

        mesh_upper_faces = mesh_upper.faces
        mesh_lower_faces = mesh_lower.faces + len(mesh_upper_vertices)

        combined_vertices = np.concatenate([mesh_upper_vertices, mesh_lower_vertices], axis=0)
        combined_faces = np.concatenate([mesh_upper_faces, mesh_lower_faces], axis=0)
        
        combined_mesh = trimesh.Trimesh(vertices=combined_vertices * 0.01, faces=combined_faces)
        combined_mesh.export(obj_path)

        pinned_verts_low = [item + len(mesh_upper_vertices) for item in pinned_verts_low]
        pinned_verts_final = pinned_verts_up + pinned_verts_low
        closest_idx_final = closest_idx_up + closest_idx_low

    out_template_path = obj_path.replace('.obj', '.pkl')

    print('obj_path', obj_path)
    template_dict = obj2template(obj_path, verbose=True, approximate_center=True)
    pickle_dump(template_dict, out_template_path)
    
    
    faces_new = template_dict['faces']
    verts_new = template_dict['rest_pos'].reshape(1, -1, 3)

    verts_up_old = mesh_upper_vertices.reshape(1, -1, 3) * 0.01
    _, idx, _ = knn_points(
        torch.from_numpy(verts_up_old).to(device).float(), 
        torch.from_numpy(verts_new).to(device).float(), K=1
    )
    idx_up_max = idx.squeeze(0).cpu().numpy().max().item() + 1
    print('idx_up', idx_up_max, len(mesh_upper_vertices))

    assert idx_up_max == len(mesh_upper_vertices)


    pinned_verts_old = combined_vertices[pinned_verts_final] * 0.01
    _, idx_pinned, _ = knn_points(
        torch.from_numpy(pinned_verts_old).to(device).float().reshape(1, -1, 3), 
        torch.from_numpy(verts_new).to(device).float(), K=1
    )
    idx_pinned = idx_pinned.reshape(-1).cpu().numpy().tolist()
    add_pinned_verts_single_template(out_template_path, pinned_verts_final)
    
    
    ############################# RUN SIMULATION
    config_dict_upper = get_random_params()
    config_dict_lower = get_random_params()
    modules, config = load_params('aux/from_any_pose')
    checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'contourcraft.pth'

    config = apply_material_params(config, config_dict_upper)
    config = apply_material2_params(config, config_dict_lower)
    runner_name = list(config.runner.keys())[0]
    config.runner[runner_name].material2.use_meterial2 = True

    if config.runner[runner_name].material2.use_meterial2:
        config.runner[runner_name].material2.start_face_indices = len(mesh_upper_faces)
        config.runner[runner_name].material2.start_vertex_indices = idx_up_max
    else:
        config.runner[runner_name].material2.start_face_indices = -1
        config.runner[runner_name].material2.start_vertex_indices = -1

    runner_module, runner, aux_modules = create_runner(modules, config)

    state_dict =  torch.load(checkpoint_path)
    runner.load_state_dict(state_dict['training_module'])
    runner = runner.to(device)

    motion_dict = np.load(motion_path)
    pose_sequence_path = os.path.join(saved_folder, 'new_obstacle.pkl')
    smplx_out_final = smplx_layer.forward_simple(
        betas=torch.from_numpy(motion_dict['betas']).to(device).float().expand(len(motion_dict['poses']), -1),
        full_pose=torch.from_numpy(motion_dict['poses']).to(device).float(),
        transl=torch.from_numpy(motion_dict['trans']).to(device).float(), # zero translation
        pose2rot=True
    )

    smplx_verts = smplx_out_final.vertices.cpu().numpy()
    smplx_faces = smplx_layer.faces_tensor.cpu().numpy()

    saved_dict = {
        'verts': smplx_verts,
        'faces': smplx_faces
    }
    with open(pose_sequence_path, 'wb') as f:
        pkl.dump(saved_dict, f)

    pose_sequence_type = 'mesh'
    garment_template_path = out_template_path

    garment_dict2 = {'vertices': verts_new.reshape(-1, 3)}
    dataloader = make_fromanypose_dataloader(pose_sequence_type=pose_sequence_type, 
                        pose_sequence_path=pose_sequence_path, 
                        garment_template_path=garment_template_path,
                        garment_dict2=garment_dict2)

    sample = next(iter(dataloader))

    trajectories_dict = runner.valid_rollout(sample)

    # Save the sequence to disk
    out_path = os.path.join(saved_folder, 'output_anypose.pkl')

    smplx_verts = smplx_out_final.vertices.cpu()
    smplx_faces = smplx_layer.faces_tensor.cpu().numpy()
    pred_pos = trajectories_dict['pred']
    cloth_faces = trajectories_dict['cloth_faces']
    print('cloth_faces', cloth_faces.shape)

    pred_pos = torch.tensor(pred_pos).float()
    cloth_faces = torch.tensor(cloth_faces).long()

    metrics = trajectories_dict['metrics']
    print('metrics', list(metrics.keys()))

    saved_mesh_dir = os.path.join(saved_folder, 'mesh_tmp')
    if not os.path.exists(saved_mesh_dir):
        os.makedirs(saved_mesh_dir)

    seq_len = len(pred_pos) - 2
    sampled_indices = np.arange(0, seq_len, 25)
    for frame_number in sampled_indices:
        smplx_mesh = Meshes([smplx_verts[frame_number]], [torch.tensor(smplx_faces).long()])
        garment_mesh = Meshes([pred_pos[frame_number+2]], [cloth_faces])

        combined_mesh = join_meshes_as_scene([smplx_mesh, garment_mesh])

        IO().save_mesh(
            combined_mesh, os.path.join(saved_mesh_dir, f'{frame_number}_0.obj')
        )

        print(os.path.join(saved_mesh_dir, f'{frame_number}_0.obj'))

    pred_pos = trajectories_dict['pred']
    trajectories_dict_saved = dict(
        pred=pred_pos,
        cloth_faces=trajectories_dict['cloth_faces'],
        pinned_vertices=pinned_verts_final,
        closest_idx_on_smpl=closest_idx_final,
        metrics=metrics,
        config_dict_lower=config_dict_lower,
        config_dict_upper=config_dict_upper,
        start_face_indices=config.runner[runner_name].material2.start_face_indices,
        start_vertex_indices=config.runner[runner_name].material2.start_vertex_indices,

        idx_up_max=idx_up_max,
        idx_pinned=idx_pinned
    )
    np.savez(out_path.replace('.pkl', '.npz'), **trajectories_dict_saved)




