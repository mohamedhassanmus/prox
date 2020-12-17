import sys
sys.path.append('/home/patrick/bed/SLP-Dataset-and-Code')
import numpy as np
import open3d as o3d
import os
from glob import glob
from shutil import copyfile
from data.SLP_RD import SLP_RD
import json
import cv2
import utils.utils as ut    # SLP utils
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from prox.camera import PerspectiveCamera
import torch
from tqdm import tqdm


SLP_PATH = '/home/patrick/datasets/SLP/danaLab'
FITS_PATH = '/home/patrick/bed/prox/slp_fits'


def view_fit(sample, idx):
    ply_path = os.path.join(FITS_PATH, '{}_{:05d}'.format(sample[1], sample[0]), 'meshes', 'image_{:06d}'.format(sample[2]), '000.ply')
    if not os.path.exists(ply_path):
        print('Couldnt find', ply_path)
        return

    depth, jt, bb = SLP_dataset.get_array_joints(idx_smpl=idx, mod='depthRaw', if_sq_bb=False)
    bb = bb.round().astype(int)

    pointcloud = ut.get_ptc(depth, SLP_dataset.f_d, SLP_dataset.c_d, bb) / 1000.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)

    smpl_mesh = o3d.io.read_triangle_mesh(ply_path)
    smpl_mesh.compute_vertex_normals()
    print(np.asarray(smpl_mesh.vertices))

    # o3d.visualization.draw_geometries([smpl_mesh])
    o3d.visualization.draw_geometries([pcd, smpl_mesh])


def make_dataset():
    all_samples = SLP_dataset.pthDesc_li

    for idx, sample in enumerate(tqdm(all_samples)):
        view_fit(sample, idx)

        # if idx > 5:
        #     break


if __name__ == "__main__":
    class PseudoOpts:
        SLP_fd = '/home/patrick/datasets/SLP/danaLab'  # give your dataset folder here
        sz_pch = [256, 256]
        fc_depth = 50
        cov_li = ['uncover']  # give the cover class you want here
    SLP_dataset = SLP_RD(PseudoOpts, phase='all')  # all test result

    print(SLP_dataset.pthDesc_li)

    make_dataset()
