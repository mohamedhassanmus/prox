# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import os.path as osp
import cv2
import numpy as np
import json
import open3d as o3d
import argparse
from prox.projection_utils import Projection

def main(args):
    recording_name = osp.basename(args.recording_dir)
    scene_name = recording_name.split("_")[0]
    base_dir = os.path.abspath(osp.join(args.recording_dir, os.pardir, os.pardir))
    cam2world_dir = osp.join(base_dir, 'cam2world')
    scene_dir = osp.join(base_dir, 'scenes')
    calib_dir = osp.join(base_dir, 'calibration')

    projection = Projection(calib_dir)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    color_dir = os.path.join(args.recording_dir, 'Color')
    depth_dir = os.path.join(args.recording_dir,'Depth')
    bodyIndex_dir = os.path.join(args.recording_dir,'BodyIndex')
    bodyIndexColor_dir = os.path.join(args.recording_dir, 'BodyIndexColor')

    vis = o3d.Visualizer()
    vis.create_window()

    trans = np.eye(4)
    if args.show_scene:
        scene = o3d.io.read_triangle_mesh(osp.join(scene_dir, scene_name + '.ply'))
        with open(os.path.join(cam2world_dir,scene_name + '.json'), 'r') as f:
            trans = np.array(json.load(f))
        vis.add_geometry(scene)

    scan = o3d.PointCloud()
    vis.add_geometry(scan)

    count = 0
    for img_name in sorted(os.listdir(color_dir))[args.start::args.step]:
        img_name = osp.splitext(img_name)[0]


        color_img = cv2.imread(os.path.join(color_dir, img_name + '.jpg'))
        color_img = cv2.flip(color_img, 1)

        if args.show_scan:
            depth_img = cv2.imread(os.path.join(depth_dir, img_name + '.png'), -1).astype(float)
            depth_img /= 8.0
            depth_img /= 1000.0

            if args.show_body_only:
                if args.mask_on_color:
                    mask = cv2.imread(os.path.join(bodyIndexColor_dir, img_name + '.png'), cv2.IMREAD_GRAYSCALE)
                else:
                    mask = cv2.imread(os.path.join(bodyIndex_dir, img_name + '.png'), cv2.IMREAD_GRAYSCALE)
                # the result of this is a mask, where 255 indicate a no-body and 0 indicate a body
                mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)[1]
            else:
                mask = np.zeros(depth_img.shape[:2])

            depth_img = cv2.flip(depth_img, 1)
            mask = cv2.flip(mask, 1)

            if args.show_color:
                scan_dict = projection.create_scan(mask, depth_img, mask_on_color=args.mask_on_color, color_im=color_img[:, :, ::-1])
            else:
                scan_dict = projection.create_scan(mask, depth_img, mask_on_color=args.mask_on_color)

            scan.points = o3d.Vector3dVector(scan_dict.get('points'))
            scan.colors = o3d.Vector3dVector(scan_dict.get('colors'))

            if np.asarray(scan.points).size == 0:
                continue
            scan.transform(trans)
        vis.update_geometry()


        print('viz frame {}, print Esc to continue'.format(img_name))
        while True:
            cv2.imshow('frame', color_img)
            vis.poll_events()
            vis.update_renderer()
            key = cv2.waitKey(30)
            if key == 27:
                break

        count += 1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('recording_dir', type=str, default=os.getcwd(),
                        help='recording dir')
    parser.add_argument('--show_scene', default=True, type=lambda arg: arg.lower() in ['true', '1'],help='')
    parser.add_argument('--show_scan',default=True, type=lambda arg: arg.lower() in ['true', '1'],help='')
    parser.add_argument('--show_body_only', default=False, type=lambda arg: arg.lower() in ['true', '1'],help='')
    parser.add_argument('--mask_on_color',default=True, type=lambda arg: arg.lower() in ['true', '1'],help='')
    parser.add_argument('--show_color', default=False, type=lambda arg: arg.lower() in ['true', '1'],help='')
    parser.add_argument('--start', type=int, default=0,help='id of the starting frame')
    parser.add_argument('--step', type=int, default=1, help='id of the starting frame')
    parser.add_argument('--coord', default='color', type=str, choices=['color', 'depth'],help='')

    args = parser.parse_args()
    main(args)
