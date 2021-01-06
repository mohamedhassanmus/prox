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


SLP_PATH = '/home/patrick/datasets/SLP/danaLab'
FITS_PATH = '/home/patrick/bed/prox/slp_fits'


def get_smpl(pkl_data, json_data):
    gender = json_data['people'][0]['gender_gt']
    print('Target height {}, weight {}'.format(json_data['people'][0]['height'], json_data['people'][0]['weight']))
    pkl_data['transl'][0, 0] += 2

    model = smplx.create('models', model_type='smpl', gender=gender)
    output = model(betas=torch.Tensor(pkl_data['betas']), body_pose=torch.Tensor(pkl_data['body_pose']), transl=torch.Tensor(pkl_data['transl']),
                   global_orient=torch.Tensor(pkl_data['global_orient']), return_verts=True)
    smpl_vertices = output.vertices.detach().cpu().numpy().squeeze()
    # smpl_joints = output.joints.detach().cpu().numpy().squeeze()

    output_unposed = model(betas=torch.Tensor(pkl_data['betas']), body_pose=torch.Tensor(pkl_data['body_pose'] * 0), transl=torch.Tensor(pkl_data['transl']),
                           global_orient=torch.Tensor(pkl_data['global_orient']), return_verts=True)
    smpl_vertices_unposed = output_unposed.vertices.detach().cpu().numpy().squeeze()

    for i, lbl in enumerate(['Wingspan', 'Height', 'Thickness']):
        print('Actual', lbl, smpl_vertices_unposed[:, i].max() - smpl_vertices_unposed[:, i].min())

    smpl_trimesh = trimesh.Trimesh(vertices=np.asarray(smpl_vertices_unposed), faces=model.faces)
    print('Est weight from volume', smpl_trimesh.volume * 1.03 * 1000)

    smpl_o3d = o3d.TriangleMesh()
    smpl_o3d.triangles = o3d.Vector3iVector(model.faces)
    smpl_o3d.vertices = o3d.Vector3dVector(smpl_vertices)
    smpl_o3d.compute_vertex_normals()
    smpl_o3d.paint_uniform_color([1.0, 0.0, 0.0])

    return smpl_vertices, model.faces, smpl_o3d


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
    ply_path = os.path.join(FITS_PATH, '{}_{:05d}'.format(sample[1], sample[0]), 'meshes', 'image_{:06d}'.format(sample[2]), '000.ply')
    print('Reading', ply_path)
    if not os.path.exists(ply_path):
        return

    json_path = os.path.join(FITS_PATH, 'keypoints', '{}_{:05d}'.format(sample[1], sample[0]), 'image_{:06d}_keypoints.json'.format(sample[2]))
    with open(json_path) as keypoint_file:
        json_data = json.load(keypoint_file)

    pkl_path = os.path.join(FITS_PATH, '{}_{:05d}'.format(sample[1], sample[0]), 'results', 'image_{:06d}'.format(sample[2]), '000.pkl')
    pkl_np = pickle.load(open(pkl_path, 'rb'))

    smpl_vertices, smpl_faces, smpl_mesh_calc = get_smpl(pkl_np, json_data)
    pcd = get_depth(idx)
    rgbd_ptc = get_rgb(sample)

    smpl_mesh = o3d.io.read_triangle_mesh(ply_path)
    smpl_mesh.compute_vertex_normals()

    vis = o3d.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(smpl_mesh)
    vis.add_geometry(smpl_mesh_calc)
    vis.add_geometry(rgbd_ptc)
    lbl = 'Participant {} sample {}'.format(sample[0], sample[2])
    vis.add_geometry(text_3d(lbl, (0, -1.5, 2), direction=(0.01, 0, -1), degree=-90, font_size=200, density=0.2))

    # vis.add_geometry(get_o3d_sphere(pos=smpl_vertices[336, :]))

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
