import sys
# Set this to the SLP github repo
sys.path.append('/home/patrick/bed/SLP-Dataset-and-Code')
import numpy as np
import open3d as o3d
import os
from data.SLP_RD import SLP_RD
import json
import utils.utils as ut    # SLP utils
import torch
from tqdm import tqdm
import pickle
import smplx
import trimesh
from prox.misc_utils import text_3d
import matplotlib.cm as cm
from prox.camera import PerspectiveCamera
import pprint


SLP_PATH = '/home/patrick/datasets/SLP/danaLab'
FITS_PATH = '/home/patrick/bed/prox/slp_fits'


def get_all_smpl(pkl_data, json_data):
    gender = json_data['people'][0]['gender_gt']
    all_meshes = []

    trans = np.array([4, 0, 0])

    for i, result in enumerate(pkl_data['all_results']):
        t = trans + [0, i * 3, 0]
        betas = torch.Tensor(result['betas']).unsqueeze(0)
        pose = torch.Tensor(result['body_pose']).unsqueeze(0)
        transl = torch.Tensor(result['transl']).unsqueeze(0)
        global_orient = torch.Tensor(result['global_orient']).unsqueeze(0)

        model = smplx.create('models', model_type='smpl', gender=gender)
        output = model(betas=betas, body_pose=pose, transl=transl, global_orient=global_orient, return_verts=True)
        smpl_vertices = output.vertices.detach().cpu().numpy().squeeze()

        smpl_o3d = o3d.TriangleMesh()
        smpl_o3d.triangles = o3d.Vector3iVector(model.faces)
        smpl_o3d.vertices = o3d.Vector3dVector(smpl_vertices)
        smpl_o3d.compute_vertex_normals()
        smpl_o3d.translate(t)

        for idx, key in enumerate(result['loss_dict'].keys()):
            lbl = '{} {:.2f}'.format(key, float(result['loss_dict'][key]))
            all_meshes.append(text_3d(lbl, t + [1, idx * 0.2 - 1, 2], direction=(0.01, 0, -1), degree=-90, font_size=150, density=0.2))

        all_meshes.append(smpl_o3d)

    return all_meshes

def get_smpl(pkl_data, json_data):
    gender = json_data['people'][0]['gender_gt']
    print('Target height {}, weight {}'.format(json_data['people'][0]['height'], json_data['people'][0]['weight']))

    betas = torch.Tensor(pkl_data['betas']).unsqueeze(0)
    pose = torch.Tensor(pkl_data['body_pose']).unsqueeze(0)
    transl = torch.Tensor(pkl_data['transl']).unsqueeze(0)
    global_orient = torch.Tensor(pkl_data['global_orient']).unsqueeze(0)

    model = smplx.create('models', model_type='smpl', gender=gender)
    output = model(betas=betas, body_pose=pose, transl=transl, global_orient=global_orient, return_verts=True)
    smpl_vertices = output.vertices.detach().cpu().numpy().squeeze()
    smpl_joints = output.joints.detach().cpu().numpy().squeeze()

    output_unposed = model(betas=betas, body_pose=pose * 0, transl=transl, global_orient=global_orient, return_verts=True)
    smpl_vertices_unposed = output_unposed.vertices.detach().cpu().numpy().squeeze()

    for i, lbl in enumerate(['Wingspan', 'Height', 'Thickness']):
        print('Actual', lbl, smpl_vertices_unposed[:, i].max() - smpl_vertices_unposed[:, i].min(), end=' ')
    print()

    smpl_trimesh = trimesh.Trimesh(vertices=np.asarray(smpl_vertices_unposed), faces=model.faces)
    print('Est weight from volume', smpl_trimesh.volume * 1.03 * 1000)
    # print('Pose embedding', pkl_data['pose_embedding'])
    # print('Body pose', np.array2string(pkl_data['body_pose'], separator=', '))

    smpl_o3d = o3d.TriangleMesh()
    smpl_o3d.triangles = o3d.Vector3iVector(model.faces)
    smpl_o3d.vertices = o3d.Vector3dVector(smpl_vertices)
    smpl_o3d.compute_vertex_normals()
    # smpl_o3d.paint_uniform_color([0.3, 0.3, 0.3])

    smpl_o3d_2 = o3d.TriangleMesh()
    smpl_o3d_2.triangles = o3d.Vector3iVector(model.faces)
    smpl_o3d_2.vertices = o3d.Vector3dVector(smpl_vertices + np.array([1.5, 0, 0]))
    smpl_o3d_2.compute_vertex_normals()
    smpl_o3d_2.paint_uniform_color([0.7, 0.3, 0.3])

    # Visualize SMPL joints - Patrick

    camera = PerspectiveCamera(rotation=torch.tensor(pkl_data['camera_rotation']).unsqueeze(0),
                               translation=torch.tensor(pkl_data['camera_translation']).unsqueeze(0),
                               center=torch.tensor(pkl_data['camera_center']),
                               focal_length_x=torch.tensor(pkl_data['camera_focal_length_x']),
                               focal_length_y=torch.tensor(pkl_data['camera_focal_length_y']))

    gt_pos_3d = camera.inverse_camera_tform(torch.tensor(pkl_data['gt_joints']).unsqueeze(0), 1.8).detach().squeeze(0).cpu().numpy()

    all_markers = []
    for i in range(25):
        color = cm.jet(i / 25.0)[:3]
        # smpl_marker = get_o3d_sphere(color=color, pos=smpl_joints[i, :])
        # all_markers.append(smpl_marker)

        pred_marker = get_o3d_sphere(color=color, pos=gt_pos_3d[i, :], radius=0.03)
        all_markers.append(pred_marker)

    return smpl_vertices, model.faces, smpl_o3d, smpl_o3d_2, all_markers


def get_depth(idx):
    depth, jt, bb = SLP_dataset.get_array_joints(idx_smpl=idx, mod='depthRaw', if_sq_bb=False)
    bb = bb.round().astype(int)

    pointcloud = ut.get_ptc(depth, SLP_dataset.f_d, SLP_dataset.c_d, bb) / 1000.0

    valid_pcd = np.logical_and(pointcloud[:, 2] > 1.5, pointcloud[:, 2] < 2.5)  # Cut out any outliers above the bed
    pointcloud = pointcloud[valid_pcd, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    return pcd


def get_rgb(sample):
    # Load RGB image
    rgb_path = os.path.join(FITS_PATH, '{}_{:05d}'.format(sample[1], sample[0]), 'images', 'image_{:06d}'.format(sample[2]), '000', 'output.png')
    rgb_image = o3d.io.read_image(rgb_path)
    rgb_raw = np.asarray(rgb_image)
    depth_raw = np.ones((rgb_raw.shape[0], rgb_raw.shape[1]), dtype=np.float32) * 2.15
    depth_image = o3d.geometry.Image(depth_raw)
    rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(rgb_image, depth_image, depth_scale=1)
    rgbd_ptc = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault))

    return rgbd_ptc


def get_o3d_sphere(color=[0.3, 1.0, 0.3], pos=[0, 0, 0], radius=0.06):
        mesh_sphere = o3d.geometry.create_mesh_sphere(radius=radius, resolution=5)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color(color)
        mean = np.asarray(mesh_sphere.vertices).mean(axis=0)
        diff = np.asarray(pos) - mean
        mesh_sphere.translate(diff)
        return mesh_sphere


def view_fit(sample, idx):
    # if sample[0] < 5 or sample[2] < 32:
    #     return
    pkl_path = os.path.join(FITS_PATH, '{}_{:05d}'.format(sample[1], sample[0]), 'results', 'image_{:06d}'.format(sample[2]), '000.pkl')
    if not os.path.exists(pkl_path):
        return

    print('Reading', pkl_path)
    pkl_np = pickle.load(open(pkl_path, 'rb'))

    json_path = os.path.join(FITS_PATH, 'keypoints', '{}_{:05d}'.format(sample[1], sample[0]), 'image_{:06d}_keypoints.json'.format(sample[2]))
    with open(json_path) as keypoint_file:
        json_data = json.load(keypoint_file)

    smpl_vertices, smpl_faces, smpl_mesh, smpl_mesh_calc, joint_markers = get_smpl(pkl_np, json_data)
    pcd = get_depth(idx)
    rgbd_ptc = get_rgb(sample)

    vis = o3d.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(smpl_mesh)
    vis.add_geometry(smpl_mesh_calc)
    vis.add_geometry(rgbd_ptc)
    lbl = 'Participant {} sample {}'.format(sample[0], sample[2])
    vis.add_geometry(text_3d(lbl, (-0.5, -1.5, 2), direction=(0.01, 0, -1), degree=-90, font_size=200, density=0.2))

    for j in joint_markers:
        vis.add_geometry(j)

    all_smpl = get_all_smpl(pkl_np, json_data)
    for o in all_smpl:
        vis.add_geometry(o)

    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("view_fits_camera.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)
    vis.run()
    vis.destroy_window()
    print('\n')


def make_dataset():
    all_samples = SLP_dataset.pthDesc_li

    for idx, sample in enumerate(tqdm(all_samples)):
        view_fit(sample, idx)

        # if idx > 5:
        #     break


if __name__ == "__main__":
    class PseudoOpts:
        SLP_fd = SLP_PATH
        sz_pch = [256, 256]
        fc_depth = 50
        cov_li = ['uncover']  # give the cover class you want here
    SLP_dataset = SLP_RD(PseudoOpts, phase='all')  # all test result

    make_dataset()
