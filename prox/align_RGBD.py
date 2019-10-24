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
import argparse
from prox.projection_utils import Projection

def main(args):
    recording_name = osp.basename(args.recording_dir)
    color_dir = os.path.join(args.recording_dir, 'Color')
    depth_dir = os.path.join(args.recording_dir,'Depth')
    scene_name = recording_name.split("_")[0]
    base_dir = os.path.abspath(osp.join(args.recording_dir, os.pardir, os.pardir))
    calib_dir = osp.join(base_dir, 'calibration')

    projection = Projection(calib_dir)

    if args.mode == 'color2depth':
        color_aligned_dir = osp.join(args.recording_dir, 'Color_aligned')
        if not osp.exists(color_aligned_dir):
            os.mkdir(color_aligned_dir)
    else:
        depth_aligned_dir = osp.join(args.recording_dir, 'Depth_aligned')
        if not osp.exists(depth_aligned_dir):
            os.mkdir(depth_aligned_dir)

    for img_name in sorted(os.listdir(color_dir)):
        img_name = osp.splitext(img_name)[0]
        print('aligning frame {}'.format(img_name))

        color_img = cv2.imread(os.path.join(color_dir, img_name + '.jpg'))

        depth_img = cv2.imread(os.path.join(depth_dir, img_name + '.png'), -1).astype(float)
        depth_raw = depth_img.copy()
        depth_img /= 8.0
        depth_img /= 1000.0

        color_img = cv2.flip(color_img, 1)
        depth_img = cv2.flip(depth_img, 1)
        depth_raw = cv2.flip(depth_raw, 1)

        if args.mode == 'color2depth':
            color_aligned = projection.align_color2depth(depth_img, color_img)
            cv2.imwrite(osp.join(color_aligned_dir, img_name + '.jpg'), color_aligned)

        else:
            depth_aligned = projection.align_depth2color(depth_img, depth_raw)
            cv2.imwrite(osp.join(depth_aligned_dir, img_name + '.png'), depth_aligned)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('recording_dir', type=str, default=os.getcwd(),
                        help='path to recording')
    parser.add_argument('--mode', default='color2depth', type=str,
                        choices=['color2depth', 'depth2color'],
                        help='')


    args = parser.parse_args()
    main(args)
