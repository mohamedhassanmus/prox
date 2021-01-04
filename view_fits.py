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


SLP_PATH = '/home/patrick/datasets/SLP/danaLab'
FITS_PATH = '/home/patrick/bed/prox/slp_fits'


def view_fit(sample, idx):
    ply_path = os.path.join(FITS_PATH, '{}_{:05d}'.format(sample[1], sample[0]), 'meshes', 'image_{:06d}'.format(sample[2]), '000.ply')
    print('Reading', ply_path)
    if not os.path.exists(ply_path):
        # print('Couldnt find', ply_path)
        return

    json_path = os.path.join(FITS_PATH, 'keypoints', '{}_{:05d}'.format(sample[1], sample[0]), 'image_{:06d}_keypoints.json'.format(sample[2]))
    with open(json_path) as keypoint_file:
        json_data = json.load(keypoint_file)
    gender = json_data['people'][0]['gender_gt']
    print('Target height {}, weight {}'.format(json_data['people'][0]['height'], json_data['people'][0]['weight']))

    pkl_path = os.path.join(FITS_PATH, '{}_{:05d}'.format(sample[1], sample[0]), 'results', 'image_{:06d}'.format(sample[2]), '000.pkl')
    pkl_np = pickle.load(open(pkl_path, 'rb'))

    # pkl_np['body_pose'] *= 0
    # pkl_np['global_orient'] *= 0
    pkl_np['transl'][0, 0] += 2

    model = smplx.create('models', model_type='smpl', gender=gender)
    output = model(betas=torch.Tensor(pkl_np['betas']), body_pose=torch.Tensor(pkl_np['body_pose']), transl=torch.Tensor(pkl_np['transl']),
                   global_orient=torch.Tensor(pkl_np['global_orient']), return_verts=True)
    smpl_vertices = output.vertices.detach().cpu().numpy().squeeze()
    # smpl_joints = output.joints.detach().cpu().numpy().squeeze()

    for i, lbl in enumerate(['Wingspan', 'Height', 'Thickness']):
        print('Actual', lbl, smpl_vertices[:, i].max() - smpl_vertices[:, i].min())

    depth, jt, bb = SLP_dataset.get_array_joints(idx_smpl=idx, mod='depthRaw', if_sq_bb=False)
    bb = bb.round().astype(int)

    pointcloud = ut.get_ptc(depth, SLP_dataset.f_d, SLP_dataset.c_d, bb) / 1000.0

    valid_pcd = np.logical_and(pointcloud[:, 2] > 1.5, pointcloud[:, 2] < 2.5)  # Cut out any outliers above the bed
    pointcloud = pointcloud[valid_pcd, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)

    smpl_mesh = o3d.io.read_triangle_mesh(ply_path)
    smpl_mesh.compute_vertex_normals()

    # smpl_trimesh = trimesh.Trimesh(vertices=smpl_vertices, faces=model.faces)
    smpl_trimesh = trimesh.Trimesh(vertices=np.asarray(smpl_mesh.vertices), faces=model.faces)
    smpl_volume = smpl_trimesh.volume
    print('Est weight volume', smpl_volume * 1.03 * 1000)

    # print('PKL betas', pkl_np['betas'])
    # print('Volume', smpl_volume)

    smpl_mesh_calc = o3d.TriangleMesh()
    smpl_mesh_calc.triangles = o3d.Vector3iVector(model.faces)
    smpl_mesh_calc.vertices = o3d.Vector3dVector(smpl_vertices)
    smpl_mesh_calc.compute_vertex_normals()
    smpl_mesh_calc.paint_uniform_color([1.0, 0.0, 0.0])

    # Load RGB image
    rgb_path = os.path.join(FITS_PATH, '{}_{:05d}'.format(sample[1], sample[0]), 'images', 'image_{:06d}'.format(sample[2]), '000', 'output.png')
    rgb_image = o3d.io.read_image(rgb_path)
    rgb_raw = np.asarray(rgb_image)
    depth_raw = np.ones((rgb_raw.shape[0], rgb_raw.shape[1]), dtype=np.float32) * 2
    depth_image = o3d.geometry.Image(depth_raw)
    rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(rgb_image, depth_image, depth_scale=1)
    rgbd_ptc = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault))
    print('Pointcloud', np.asarray(rgbd_ptc.points))

    # o3d.visualization.draw_geometries([smpl_mesh])
    vis = o3d.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(smpl_mesh)
    vis.add_geometry(smpl_mesh_calc)
    vis.add_geometry(rgbd_ptc)
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("view_fits_camera.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)
    vis.run()
    vis.destroy_window()
    print('\n')

    # o3d.visualization.draw_geometries([pcd, smpl_mesh])


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
