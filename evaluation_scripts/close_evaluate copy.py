import os
import sys
import numpy as np
import torch
import pickle
import argparse
from tqdm import tqdm

from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene
from pytorch3d.io import IO, save_obj, load_ply
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
import copy

sys.path.insert(0, '/is/cluster/fast/sbian/github/ContourCraft/')

from utils.close_utils import get_seged_points
from utils.smplx_garment_conversion import deform_garments
from runners.smplx.body_models import SMPLXLayer
import subprocess

argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str, required=True, help='Path to the folder containing the results')
argparser.add_argument('--method', type=str, default='llava', help='Path to the folder containing the results')
argparser.add_argument('--is_Apose', type=int, default=0, help='Path to the folder containing the results')
argparser.add_argument('--is_fscore', type=int, default=0, help='Path to the folder containing the results')
argparser.add_argument('--dataset', type=str, default='args.dataset', help='Path to the close dataset')
args = argparser.parse_args()


smplx_layer = SMPLXLayer(
    '../../Pose_estimation/TokenHMR/data/body_models/smpl/SMPL_NEUTRAL.pkl',
    ext='pkl',
    num_betas=300
).cuda()

pkl_path = '/is/cluster/fast/scratch/gbecherini/siyuan/240930/smplxn_params.pkl'
with open(pkl_path, 'rb') as f:
    smplx_data = pickle.load(f)


def rotate_pose(pose, angle, which_axis='x'):
    # pose: (n, 72)
    
    from scipy.spatial.transform import Rotation as R
    is_tensor = torch.is_tensor(pose)
    if is_tensor:
        pose = pose.cpu().detach().numpy()
    
    swap_rotation = R.from_euler(which_axis, [angle/np.pi*180], degrees=True)
    root_rot = R.from_rotvec(pose[:, :3])
    pose[:, :3] = (swap_rotation * root_rot).as_rotvec()

    if is_tensor:
        pose = torch.FloatTensor(pose)

    return pose


# runs/try_lr1e_4_wholebody_pose_v2_detailT2_upd_possibleDebug_v3_garmentcontrol_addFTdata_onlyimg_CloSE_eva_crop
def get_meshes_llava(path):
    # runs/try_v16_13b_lr1e_4_v3_garmentcontrol_4h100_openai_imgs_cropped_crop/vis_new/valid_garment_00170__Inner__Take1/valid_garment_lower/valid_garment_lower/valid_garment_lower_sim.obj
    llava_parent_folder = '/is/cluster/fast/sbian/github/LLaVA/'
    path = os.path.join(llava_parent_folder, path)
    args.path = path
    all_folders = os.listdir(os.path.join(path, 'vis_new'))
    all_folders = [item for item in all_folders if os.path.isdir(os.path.join(path, 'vis_new', item))]
    mesh_dict = {}
    path_dict = {}
    for folder in tqdm(all_folders, dynamic_ncols=True):
        folder_name = folder[len('valid_garment_'):]
        mesh_dict[folder_name] = {}
        path_dict[folder_name] = {}
        img_result_dir = os.path.join(path, 'vis_new', folder)
        subfolders = os.listdir(img_result_dir)
        subfolders = [item for item in subfolders if os.path.isdir(os.path.join(img_result_dir, item))]
        for subfolder in subfolders:
            garment_path = os.path.join(img_result_dir, subfolder, subfolder, f'{subfolder}_sim.obj')
            if not os.path.exists(garment_path):
                continue
            mesh = IO().load_mesh(garment_path, load_textures=False)
            mesh_dict[folder_name][subfolder] = mesh
            path_dict[folder_name][subfolder] = garment_path
        
        if len(mesh_dict[folder_name]) == 0:
            mesh_dict.pop(folder_name)
            path_dict.pop(folder_name)
            continue

        meshes_all = list(mesh_dict[folder_name].values())
        garment_combined = join_meshes_as_scene(meshes_all)
        # if garment_combined.verts_padded().max() > 1e3:
        #     continue
        mesh_dict[folder_name]['combined'] = garment_combined.cuda()
        mesh_dict[folder_name]['folder'] = img_result_dir

        # print(garment_combined.verts_packed().shape)

    smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params.pkl'
    with open(smplx_params_path, 'rb') as f:
        smplx_params = pickle.load(f)
    
    smplx_dict = {
        'betas': torch.tensor(smplx_params['pred_shape'], dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['pred_pose'], dtype=torch.float32).reshape(1, 165).cuda(),
        'transl': torch.tensor(smplx_params['pred_transl'], dtype=torch.float32).reshape(1, 3).cuda(),
    }
    return mesh_dict, path_dict, smplx_dict


# runs/try_lr1e_4_wholebody_pose_v2_detailT2_upd_possibleDebug_v3_garmentcontrol_addFTdata_2step_CLose_eva
def get_meshes_sewformer(path):
    # Sewformer/outputs/Dress4D/Detr2d-V6-final-dif-ce-focal-schd-agp/00187__Inner__Take2/bottom_pattern/bottom_pattern_sim.obj
    all_folders = os.listdir(path)
    all_folders = [item for item in all_folders if os.path.isdir(os.path.join(path, item))]
    mesh_dict = {}
    path_dict = {}
    for folder in tqdm(all_folders, dynamic_ncols=True):
        if 'tmp' in folder:
            continue
        folder_name = folder
        mesh_dict[folder_name] = {}
        path_dict[folder_name] = {}
        folder_path = os.path.join(path, folder)
        subfolders = os.listdir(folder_path)
        if 'bottom_pattern' in subfolders:
            garment_path = os.path.join(folder_path, 'bottom_pattern', 'bottom_pattern_sim.obj')
            if os.path.exists(garment_path):
                mesh = IO().load_mesh(garment_path)
                mesh_dict[folder_name]['bottom'] = mesh
                path_dict[folder_name] = garment_path
        
        if 'top_pattern' in subfolders:
            garment_path = os.path.join(folder_path, 'top_pattern', 'top_pattern_sim.obj')
            if os.path.exists(garment_path):
                mesh = IO().load_mesh(garment_path)
                mesh_dict[folder_name]['top'] = mesh
                path_dict[folder_name] = garment_path
        
        meshes_all = list(mesh_dict[folder_name].values())
        if len(meshes_all) == 0:
            continue
        garment_combined = join_meshes_as_scene(meshes_all)
        print(garment_combined.verts_padded().max(), garment_combined.verts_padded().min())
        mesh_dict[folder_name]['combined'] = garment_combined
        mesh_dict[folder_name]['folder'] = folder_path
    
    smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params_garmentgenerator.pkl'
    with open(smplx_params_path, 'rb') as f:
        smplx_params = pickle.load(f)
    
    smplx_dict = {
        'betas': torch.tensor(smplx_params['pred_shape'], dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['pred_pose'], dtype=torch.float32).reshape(1, 165).cuda(),
        'transl': torch.tensor(smplx_params['pred_transl'], dtype=torch.float32).reshape(1, 3).cuda(),
    }
    return mesh_dict, path_dict, smplx_dict



def get_meshes_dresscode(path):
    # outputs/241106-17-09-43/005/pred_0/garment/garment_sim.obj
    pkl_path = '/is/cluster/fast/scratch/sbian/CloSE_eva/gpt_labels/garment_name_dict.pkl'
    img_path = '/is/cluster/fast/scratch/sbian/CloSE_eva/imgs'
    saved_folder = '/is/cluster/fast/scratch/sbian/CloSE_eva/dresscode_results'
    all_imgs = os.listdir(img_path)
    all_imgs = [item for item in all_imgs if item.endswith('.jpg') or item.endswith('.png')]
    all_imgs = sorted(all_imgs)

    with open(pkl_path, 'rb') as f:
        garment_name_dict = pickle.load(f)

    all_folders = os.listdir(path)
    print('all_folders', all_folders)
    all_folders = [item for item in all_folders if os.path.isdir(os.path.join(path, item))]
    mesh_dict = {}
    path_dict = {}
    for folder in tqdm(all_folders, dynamic_ncols=True):
        if 'tmp' in folder:
            continue
        
        index = int(folder)
        garment_name = garment_name_dict[index].split('.')[0]
        if garment_name not in mesh_dict:
            mesh_dict[garment_name] = {}
            path_dict[garment_name] = {}
        
        mesh_path = os.path.join(path, folder, 'pred_0', 'garment', 'garment_sim.obj')
        if os.path.exists(mesh_path):
            mesh = IO().load_mesh(mesh_path)
            mesh_dict[garment_name][folder] = mesh
            path_dict[garment_name][folder] = mesh_path
    
    ks = list(mesh_dict.keys())
    # for garment_name, garment_dict in mesh_dict.items():
    for garment_name in ks:
        garment_dict = mesh_dict[garment_name]
        meshes_all = list(garment_dict.values())
        if len(meshes_all) == 0:
            mesh_dict.pop(garment_name)
            continue
        garment_combined = join_meshes_as_scene(meshes_all)
        mesh_dict[garment_name]['combined'] = garment_combined
        mesh_dict[garment_name]['folder'] = os.path.join(saved_folder, garment_name)
        if not os.path.exists(mesh_dict[garment_name]['folder']):
            os.makedirs(mesh_dict[garment_name]['folder'])

    smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params_garmentgenerator.pkl'
    with open(smplx_params_path, 'rb') as f:
        smplx_params = pickle.load(f)
    
    smplx_dict = {
        'betas': torch.tensor(smplx_params['pred_shape'], dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['pred_pose'], dtype=torch.float32).reshape(1, 165).cuda(),
        'transl': torch.tensor(smplx_params['pred_transl'], dtype=torch.float32).reshape(1, 3).cuda(),
    }
    return mesh_dict, path_dict, smplx_dict



def get_meshes_gpt4o(path):
    args.path = path
    all_folders = os.listdir(path)
    all_folders = [item for item in all_folders if os.path.isdir(os.path.join(path, item))]
    mesh_dict = {}
    path_dict = {}
    for folder in tqdm(all_folders, dynamic_ncols=True):
        if 'tmp' in folder:
            continue

        folder_name = folder
        mesh_dict[folder_name] = {}
        path_dict[folder_name] = {}
        img_result_dir = os.path.join(path, folder)
        subfolders = os.listdir(img_result_dir)
        subfolders = [item for item in subfolders if os.path.isdir(os.path.join(img_result_dir, item))]
        for subfolder in subfolders:
            subsubfolders = os.listdir(os.path.join(img_result_dir, subfolder))
            subsubfolders = [item for item in subsubfolders if os.path.isdir(os.path.join(img_result_dir, subfolder, item))]
            assert len(subsubfolders) <= 1
            if len(subsubfolders) == 0:
                continue
            garment_path = os.path.join(img_result_dir, subfolder, subsubfolders[0], f'{subsubfolders[0]}_sim.obj')
            if os.path.exists(garment_path):
                mesh = IO().load_mesh(garment_path)
                mesh_dict[folder_name][subfolder] = mesh
                path_dict[folder_name][subfolder] = garment_path
        
        meshes_all = list(mesh_dict[folder_name].values())
        garment_combined = join_meshes_as_scene(meshes_all)
        mesh_dict[folder_name]['combined'] = garment_combined.cuda()
        # print('garment_combined', garment_combined.verts_padded().max(), garment_combined.verts_padded().min())
        mesh_dict[folder_name]['folder'] = img_result_dir

    smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params_garmentgenerator.pkl'
    with open(smplx_params_path, 'rb') as f:
        smplx_params = pickle.load(f)
    
    smplx_dict = {
        'betas': torch.tensor(smplx_params['pred_shape'], dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['pred_pose'], dtype=torch.float32).reshape(1, 165).cuda(),
        'transl': torch.tensor(smplx_params['pred_transl'], dtype=torch.float32).reshape(1, 3).cuda(),
    }
    return mesh_dict, path_dict, smplx_dict



def get_meshes_garmentrecovery_rest(path):
    all_possible_fnames = ['result_shirt', 'result_jacket', 'result_trousers', 'result_skirt']
    mesh_dict = {}
    path_dict = {}
    for fname in tqdm(all_possible_fnames, dynamic_ncols=True):
        folder_path = os.path.join(path, fname)
        all_subfolders = os.listdir(folder_path)
        all_subfolders = [item for item in all_subfolders if os.path.isdir(os.path.join(folder_path, item))]
        for subfolder in all_subfolders:
            folder_name = subfolder
            if folder_name not in mesh_dict:
                mesh_dict[folder_name] = {}
                path_dict[folder_name] = {}
            # fitting-data/garment/result_shirt/00185__Inner__Take4/mesh_atlas_sewing.obj
            garment_path = os.path.join(folder_path, subfolder, f'mesh_atlas_sewing.obj')
            if os.path.exists(garment_path):
                mesh = IO().load_mesh(garment_path)
                mesh_dict[folder_name][fname] = mesh
                path_dict[folder_name][fname] = garment_path

    mesh_dict_cp = copy.deepcopy(mesh_dict)
    for garment_name, garment_dict in mesh_dict_cp.items():
        meshes_all = list(garment_dict.values())
        if len(meshes_all) < 2:
            mesh_dict.pop(garment_name)
            path_dict.pop(garment_name)
            continue
        garment_combined = join_meshes_as_scene(meshes_all)
        garment_combined_verts = garment_combined.verts_padded() * 100
        garment_combined = Meshes(verts=garment_combined_verts, faces=garment_combined.faces_padded())
        mesh_dict[garment_name]['combined'] = garment_combined
        mesh_dict[garment_name]['folder'] = os.path.join(path, 'result_combined', garment_name)
        if not os.path.exists(mesh_dict[garment_name]['folder']):
            os.makedirs(mesh_dict[garment_name]['folder'])

    smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params_garmentgenerator_cloth3d.pkl'
    with open(smplx_params_path, 'rb') as f:
        smplx_params = pickle.load(f)
    
    smplx_dict = {
        'betas': torch.tensor(smplx_params['pred_shape'], dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['pred_pose'], dtype=torch.float32).reshape(1, 165).cuda(),
        'transl': torch.tensor(smplx_params['pred_transl'], dtype=torch.float32).reshape(1, 3).cuda(),
    }
    return mesh_dict, path_dict, smplx_dict



def get_meshes_garmentrecovery_pose(path):
    all_possible_fnames = ['result_shirt', 'result_jacket', 'result_trousers', 'result_skirt']
    mesh_dict = {}
    path_dict = {}
    for fname in tqdm(all_possible_fnames, dynamic_ncols=True):
        folder_path = os.path.join(path, fname)
        all_subfolders = os.listdir(folder_path)
        all_subfolders = [item for item in all_subfolders if os.path.isdir(os.path.join(folder_path, item))]
        for subfolder in all_subfolders:
            folder_name = subfolder
            if folder_name not in mesh_dict:
                mesh_dict[folder_name] = {}
                path_dict[folder_name] = {}
            # fitting-data/garment/result_shirt/00185__Inner__Take4/mesh_atlas_sewing.obj
            garment_path = os.path.join(folder_path, subfolder, f'mesh_verts_opt.obj')
            # garment_path = os.path.join(folder_path, subfolder, f'mesh_atlas_sewing.obj')

            if os.path.exists(garment_path):
                mesh = IO().load_mesh(garment_path)
                mesh_dict[folder_name][fname] = mesh
                path_dict[folder_name][fname] = garment_path
                path_dict[folder_name]['body'] = os.path.join(folder_path, subfolder, f'body.obj')

    mesh_dict_cp = copy.deepcopy(mesh_dict)
    for garment_name, garment_dict in mesh_dict_cp.items():
        meshes_all = list(garment_dict.values())
        if len(meshes_all) < 2:
            mesh_dict.pop(garment_name)
            path_dict.pop(garment_name)
            continue
        garment_combined = join_meshes_as_scene(meshes_all)
        garment_combined_verts = garment_combined.verts_padded() * 100
        garment_combined = Meshes(verts=garment_combined_verts, faces=garment_combined.faces_padded())
        mesh_dict[garment_name]['combined'] = garment_combined
        mesh_dict[garment_name]['folder'] = os.path.join(path, 'result_combined', garment_name)
        if not os.path.exists(mesh_dict[garment_name]['folder']):
            os.makedirs(mesh_dict[garment_name]['folder'])
        
        smpl_np = np.load(
            f'/is/cluster/fast/sbian/github/GarmentRecovery/fitting-data/garment/processed/econ/obj/{garment_name}_smpl_00.npy', 
        allow_pickle=True)[()]

        shape_params = smpl_np['betas'].detach().cpu()
        expression_params = smpl_np['expression'].detach().cpu()
        body_pose = smpl_np['body_pose'].detach().cpu()
        global_pose = smpl_np['global_orient'].detach().cpu()
        jaw_pose = smpl_np['jaw_pose'].detach().cpu()
        left_hand_pose = smpl_np['left_hand_pose'].detach().cpu()
        right_hand_pose = smpl_np['right_hand_pose'].detach().cpu()

        global_pose = rotate_pose(global_pose.view(-1, 3), np.pi, which_axis='x').reshape(-1)
        np_shape = torch.zeros((1, 300))
        np_shape[0, :200] = shape_params

        full_pose = torch.cat(
            [global_pose.reshape(-1, 3),
             body_pose.reshape(-1, 3),
             jaw_pose.reshape(-1, 3),
             torch.zeros((2, 3)),
             left_hand_pose.reshape(-1, 3),
             right_hand_pose.reshape(-1,3)],
            dim=0)

        body_path = path_dict[garment_name]['body']
        body_mesh = IO().load_mesh(body_path)

        scale = smpl_np['scale'].detach().cpu()
        mesh_dict[garment_name]['smplx'] = {
            'betas': torch.tensor(np_shape, dtype=torch.float32).reshape(1, 300).cuda(),
            'poses': torch.tensor(full_pose, dtype=torch.float32).reshape(1, 165).cuda(),
            'body_vs': body_mesh.verts_padded().cuda(),
            'scale': torch.tensor(scale, dtype=torch.float32).reshape(1, 1).cuda(),
        }

    smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params_garmentgenerator_cloth3d.pkl'
    with open(smplx_params_path, 'rb') as f:
        smplx_params = pickle.load(f)

    smplx_dict = {
        'betas': torch.tensor(smplx_params['pred_shape'], dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['pred_pose'], dtype=torch.float32).reshape(1, 165).cuda(),
        'transl': torch.tensor(smplx_params['pred_transl'], dtype=torch.float32).reshape(1, 3).cuda(),
    }
    return mesh_dict, path_dict, smplx_dict




def convert_garments(pred_garment_mesh, img_name, smplx_params_raw, saved_folder=''):
    print('Start converting garments', img_name)
    img_name = img_name.split('.')[0]
    garnment_id = img_name
    target_npz_path = os.path.join(
        'args.dataset', f'{garnment_id}.npz'
    )
    target_npz = np.load(target_npz_path)

    smplx_params = smplx_data[garnment_id]
    gt_points_upper, gt_points_lower, gt_points_wholebody, gt_points = get_seged_points(target_npz)
    gt_points = gt_points / target_npz['scale']
    print('scale', target_npz['scale'])
    # gt_points_wholebody = torch.from_numpy(gt_points_wholebody).float().cuda()
    # print('gt_points_wholebody', gt_points_wholebody.shape)
    gt_points = torch.from_numpy(gt_points)[::10].unsqueeze(0).float().cuda()
    print('gt_points', gt_points.shape)

    betas = np.zeros(300)
    betas[:16] = smplx_params['betas']

    # print('smplx_params', list(smplx_params.keys()))

    smplx_params_new = {
        'betas': torch.tensor(betas, dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['poses'], dtype=torch.float32).reshape(1, 55, 3).cuda(),
        'transl': torch.tensor(smplx_params['trans'], dtype=torch.float32).reshape(1, 3).cuda(),
    }

    deformed_garment_verts = deform_garments(
        smplx_layer, smplx_params_raw, smplx_params_new, pred_garment_mesh, smplx_layer.lbs_weights
    )

    deformed_garment_mesh = Meshes(verts=[deformed_garment_verts], faces=[pred_garment_mesh.faces_packed()])
    pred_points = sample_points_from_meshes(deformed_garment_mesh, len(gt_points[0]))
    # gt_points = gt_points_wholebody.unsqueeze(0)

    print('pred_points', pred_points.shape, pred_points.mean(dim=1))
    print('gt_points', gt_points.shape, gt_points.mean(dim=1))

    chamfer_dist = chamfer_distance(pred_points, gt_points)
    print(chamfer_dist)

    IO().save_mesh(deformed_garment_mesh, os.path.join(saved_folder, f'{img_name}_converted.obj'))
    pointcloud = Pointclouds(points=[gt_points[0]])
    IO().save_pointcloud(pointcloud, os.path.join(saved_folder, f'{img_name}_converted.ply'))
    print('saved_folder', saved_folder)

    return chamfer_dist[0] * 1e3



def fscore_func(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2


def calculate_fscore(pred_garment_mesh, img_name, smplx_params_raw, saved_folder=''):
    # print('Start converting garments', img_name)
    # img_name = img_name.split('.')[0]
    # garnment_id = img_name
    # target_npz_path = os.path.join(
    #     'args.dataset', f'{garnment_id}.npz'
    # )
    # target_npz = np.load(target_npz_path)

    # smplx_params = smplx_data[garnment_id]
    # gt_points_upper, gt_points_lower, gt_points_wholebody, gt_points = get_seged_points(target_npz)
    # # gt_points_wholebody = torch.from_numpy(gt_points_wholebody).float().cuda()
    # # print('gt_points_wholebody', gt_points_wholebody.shape)
    # gt_points = torch.from_numpy(gt_points)[::10].unsqueeze(0).float().cuda()
    # print('gt_points', gt_points.shape)

    # betas = np.zeros(300)
    # betas[:16] = smplx_params['betas']

    # smplx_params_new = {
    #     'betas': torch.tensor(betas, dtype=torch.float32).reshape(1, 300).cuda(),
    #     'poses': torch.tensor(smplx_params['poses'], dtype=torch.float32).reshape(1, 55, 3).cuda(),
    #     'transl': torch.tensor(smplx_params['trans'], dtype=torch.float32).reshape(1, 3).cuda(),
    # }

    # deformed_garment_verts = deform_garments(
    #     smplx_layer, smplx_params_raw, smplx_params_new, pred_garment_mesh, smplx_layer.lbs_weights
    # )

    # deformed_garment_mesh = Meshes(verts=[deformed_garment_verts], faces=[pred_garment_mesh.faces_packed()])
    # print('Start converting garments', img_name)
    img_name = img_name.split('.')[0]
    garnment_id = img_name
    target_npz_path = os.path.join(
        'args.dataset', f'{garnment_id}.npz'
    )
    target_npz = np.load(target_npz_path)

    gt_points_upper, gt_points_lower, gt_points_wholebody, gt_points = get_seged_points(target_npz)
    gt_points = gt_points / target_npz['scale']
    gt_points = torch.from_numpy(gt_points)[::10].unsqueeze(0).float().cuda()

    deformed_garment_mesh = IO().load_mesh(os.path.join(saved_folder, f'{img_name}_converted.obj'))
    pred_points = sample_points_from_meshes(deformed_garment_mesh.cuda(), len(gt_points[0]))

    # print('pred_points', pred_points.shape, pred_points.max())
    # print('gt_points', gt_points.shape, gt_points.max())

    chamfer_x, chamfer_y = chamfer_distance(pred_points, gt_points, batch_reduction=None, point_reduction=None)[0]
    # print(chamfer_x.mean(), chamfer_y.mean())
    # chamfer_x = torch.sqrt(chamfer_x)
    # chamfer_y = torch.sqrt(chamfer_y)
    # print('chamfer_x', chamfer_x.mean(), chamfer_y.mean())
    fscore = fscore_func(chamfer_x, chamfer_y)[0]
    print('fscore', fscore)

    return fscore


def run_python(garmentpath, garmentpath2=None, saved_folder=''):
    if garmentpath2 is None:
        process = subprocess.Popen(
            ["/is/cluster/fast/sbian/data/blender-3.6.14-linux-x64/blender", 
            "--background", "--python", "blender_rendering_eva.py", 
            "--", "--garmentpath", garmentpath, "--savedfolder", saved_folder
            ], stdout=subprocess.PIPE
        )
    else:
        process = subprocess.Popen(
            ["/is/cluster/fast/sbian/data/blender-3.6.14-linux-x64/blender", 
            "--background", "--python", "blender_rendering_eva.py", 
            "--", "--garmentpath", garmentpath, "--garmentpath2", garmentpath2, "--savedfolder", saved_folder
            ], stdout=subprocess.PIPE
        )

    process.wait()
    print('finished', garmentpath, process.returncode)
    return


def convert_garments_Apose(pred_garment_mesh, img_name, smplx_params_raw, inp_path):
    print('Start converting garments', img_name)
    smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params.pkl'
    with open(smplx_params_path, 'rb') as f:
        smplx_params = pickle.load(f)
    
    smplx_params_new = {
        'betas': torch.tensor(smplx_params['pred_shape'], dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['pred_pose'], dtype=torch.float32).reshape(1, 165).cuda(),
        'transl': torch.tensor(smplx_params['pred_transl'], dtype=torch.float32).reshape(1, 3).cuda(),
    }

    deformed_garment_verts = deform_garments(
        smplx_layer, smplx_params_raw, smplx_params_new, pred_garment_mesh, smplx_layer.lbs_weights
    )

    deformed_garment_mesh = Meshes(verts=[deformed_garment_verts], faces=[pred_garment_mesh.faces_packed()])
    saved_path = inp_path.replace('.obj', '_converted_Apose.obj')
    IO().save_mesh(deformed_garment_mesh, saved_path)
    print('saved_path', saved_path)

    return saved_path

if __name__ == '__main__':
    chamfer_dist_all = []
    if args.method == 'llava':
        mesh_dict, path_dict, smplx_dict = get_meshes_llava(args.path)
    elif args.method == 'sewformer':
        mesh_dict, path_dict, smplx_dict = get_meshes_sewformer(args.path)
    elif args.method == 'dresscode':
        mesh_dict, path_dict, smplx_dict = get_meshes_dresscode(args.path)
    elif args.method == 'gpt4o':
        mesh_dict, path_dict, smplx_dict = get_meshes_gpt4o(args.path)
    elif args.method == 'garmentrecovery_rest':
        mesh_dict, path_dict, smplx_dict = get_meshes_garmentrecovery_rest(args.path)
    elif args.method == 'garmentrecovery':
        mesh_dict, path_dict, smplx_dict = get_meshes_garmentrecovery_pose(args.path)

    if not args.is_Apose and not args.is_fscore:
        summary_dict = {}
        print(len(mesh_dict))
        # assert False
        for img_name, pred_garment_mesh_dict in mesh_dict.items():
            if 'smplx' in pred_garment_mesh_dict:
                smplx_dict = pred_garment_mesh_dict['smplx']
            chamfer_dist = convert_garments(
                pred_garment_mesh_dict['combined'].cuda(), img_name, smplx_dict, saved_folder=pred_garment_mesh_dict['folder'])

            if chamfer_dist > 200:
                continue

            summary_dict[img_name] = chamfer_dist
            chamfer_dist_all.append(chamfer_dist)
        
        print('chamfer_dist_all', torch.tensor(chamfer_dist_all).mean())
        with open(os.path.join(args.path, 'summary_dict.pkl'), 'wb') as f:
            pickle.dump(summary_dict, f)
        
        with open(os.path.join(args.path, 'summary_dict.txt'), 'w') as f:
            f.write(str(torch.tensor(chamfer_dist_all).mean()))
    
    elif args.is_fscore:
        with open(os.path.join(args.path, 'summary_dict.pkl'), 'rb') as f:
            summary_dict = pickle.load(f)

        fscore_dict = {}
        for img_name, pred_garment_mesh_dict in mesh_dict.items():
            if img_name not in summary_dict:
                continue
            
            if args.method == 'llava' and summary_dict[img_name] > 200:
                # a bug
                continue
            
            fscore0 = calculate_fscore(
                pred_garment_mesh_dict['combined'].cuda(), img_name, smplx_dict, saved_folder=pred_garment_mesh_dict['folder'])

            fscore_dict[img_name] = fscore0
            chamfer_dist_all.append(fscore0)
        
        print('fscore_all', torch.tensor(chamfer_dist_all).mean())
        with open(os.path.join(args.path, 'fscore_dict.pkl'), 'wb') as f:
            pickle.dump(fscore_dict, f)
    
    else:
        saved_all = []
        for img_name, pred_garment_mesh_dict in mesh_dict.items():
            saved_paths = []
            for submesh_name, submesh in pred_garment_mesh_dict.items():
                if submesh_name == 'combined' or submesh_name == 'folder':
                    continue

                saved_path = convert_garments_Apose(
                    pred_garment_mesh_dict[submesh_name].cuda(), img_name, smplx_dict, inp_path=path_dict[img_name][submesh_name])

                saved_paths.append(saved_path)
            
            if len(saved_paths) == 0:
                continue

            elif len(saved_paths) == 1:
                run_python(saved_paths[0], saved_folder=pred_garment_mesh_dict['folder'])
            
            else:
                run_python(saved_paths[0], saved_paths[1], saved_folder=pred_garment_mesh_dict['folder'])

            # run_python(saved_path)
            # saved_all.append(saved_path)
        
        with open(os.path.join(args.path, 'saved_all_Apose_meshes.txt'), 'w') as f:
            for item in saved_all:
                f.write(f'"{item}"' + '\n')
