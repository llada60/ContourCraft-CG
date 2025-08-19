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

sys.path.insert(0, '/home/ids/liliu/projects/ChatGarment/ContourCraft-CG/')

from utils.close_utils import get_seged_points
from utils.smplx_garment_conversion import deform_garments
from runners.smplx.body_models import SMPLXLayer
import subprocess

argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str, required=True, help='Path to the folder containing the results')
argparser.add_argument('--method', type=str, default='llava', help='Path to the folder containing the results')
argparser.add_argument('--is_Apose', type=int, default=0, help='Path to the folder containing the results')
argparser.add_argument('--is_fscore', type=int, default=0, help='Path to the folder containing the results')
argparser.add_argument('--dataset', type=str, default='/home/ids/liliu/data/close/CloSe-Di', help='Path to the close dataset')
args = argparser.parse_args()


smplx_layer = SMPLXLayer(
    '/home/ids/liliu/data/body_models/models/smplx/SMPLX_NEUTRAL.pkl',
    ext='pkl',
    num_betas=300
).cuda()

pkl_path = '/home/ids/liliu/projects/ChatGarment/assets/eval/smplxn_params.pkl'
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
    # llava_parent_folder = '/is/cluster/fast/sbian/github/LLaVA/'
    # path = os.path.join(llava_parent_folder, path)
    args.path = path
    all_folders = os.listdir(os.path.join(path, 'vis_new'))
    all_folders = [item for item in all_folders if os.path.isdir(os.path.join(path, 'vis_new', item))]
    mesh_dict = {}
    path_dict = {}
    for folder in tqdm(all_folders, dynamic_ncols=True): # get all prediction
        folder_name = folder[len('valid_garment_'):]
        mesh_dict[folder_name] = {}
        path_dict[folder_name] = {}
        img_result_dir = os.path.join(path, 'vis_new', folder)
        subfolders = os.listdir(img_result_dir)
        subfolders = [item for item in subfolders if os.path.isdir(os.path.join(img_result_dir, item))]
        for subfolder in subfolders:
            garment_path = os.path.join(img_result_dir, subfolder, subfolder, f'{subfolder}_sim.obj') # mesh_file ended in _sim
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

    smplx_params_path = '/home/ids/liliu/projects/ChatGarment/assets/eval/aaa_mesh_registrarion/registered_params.pkl'
    with open(smplx_params_path, 'rb') as f:
        smplx_params = pickle.load(f)
    
    smplx_dict = {
        'betas': torch.tensor(smplx_params['pred_shape'], dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['pred_pose'], dtype=torch.float32).reshape(1, 165).cuda(),
        'transl': torch.tensor(smplx_params['pred_transl'], dtype=torch.float32).reshape(1, 3).cuda(),
    }
    return mesh_dict, path_dict, smplx_dict


def convert_garments(pred_garment_mesh, img_name, smplx_params_raw, saved_folder=''):
    # print('Start converting garments', img_name)
    img_name = img_name.split('.')[0]
    garnment_id = img_name
    target_npz_path = os.path.join(
        args.dataset, f'{garnment_id}.npz'
    )
    target_npz = np.load(target_npz_path)
    smplx_params = smplx_data[garnment_id]
    gt_points_upper, gt_points_lower, gt_points_wholebody, gt_points = get_seged_points(target_npz)
    gt_points = gt_points / target_npz['scale']
    # print('scale', target_npz['scale'])
    # gt_points_wholebody = torch.from_numpy(gt_points_wholebody).float().cuda()
    # print('gt_points_wholebody', gt_points_wholebody.shape)
    gt_points = torch.from_numpy(gt_points)[::10].unsqueeze(0).float().cuda()
    pointcloud = Pointclouds(points=[gt_points[0]])
    IO().save_pointcloud(pointcloud, os.path.join(saved_folder, f'{img_name}_gt.ply'))
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

    # print('pred_points', pred_points.shape, pred_points.meaRn(dim=1))
    # print('gt_points', gt_points.shape, gt_points.mean(dim=1))

    chamfer_dist = chamfer_distance(pred_points, gt_points)
    # print(chamfer_dist)

    IO().save_mesh(deformed_garment_mesh, os.path.join(saved_folder, f'{img_name}_converted.obj'))
    pointcloud = Pointclouds(points=[pred_points[0]])
    IO().save_pointcloud(pointcloud, os.path.join(saved_folder, f'{img_name}_converted.ply'))
    # print('saved_folder', saved_folder)

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
    #     args.dataset, f'{garnment_id}.npz'
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
        args.dataset, f'{garnment_id}.npz'
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
    # print('fscore', fscore)

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
    # print('finished', garmentpath, process.returncode)
    return


def convert_garments_Apose(pred_garment_mesh, img_name, smplx_params_raw, inp_path):
    # print('Start converting garments', img_name)
    smplx_params_path = '/home/ids/liliu/projects/ChatGarment/assets/eval/aaa_mesh_registrarion/registered_params.pkl'
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
    # print('saved_path', saved_path)

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
    # mesh_dict, path_dict, smplx_dict = {}, {}, {}
    if not args.is_Apose and not args.is_fscore:
        # compute chamfer distance
        summary_dict = {}
        print(len(mesh_dict))
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
        # compute fscore
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
        # convert garments to Apose
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
