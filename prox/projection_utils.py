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

import os.path as osp
import cv2
import numpy as np
import json

class Projection():
    def __init__(self, calib_dir):
        with open(osp.join(calib_dir, 'IR.json'), 'r') as f:
            self.depth_cam = json.load(f)
        with open(osp.join(calib_dir, 'Color.json'), 'r') as f:
            self.color_cam = json.load(f)

    def row(self, A):
        return A.reshape((1, -1))
    def col(self, A):
        return A.reshape((-1, 1))

    def unproject_depth_image(self, depth_image, cam):
        us = np.arange(depth_image.size) % depth_image.shape[1]
        vs = np.arange(depth_image.size) // depth_image.shape[1]
        ds = depth_image.ravel()
        uvd = np.array(np.vstack((us.ravel(), vs.ravel(), ds.ravel())).T)
        #unproject
        xy_undistorted_camspace = cv2.undistortPoints(np.asarray(uvd[:, :2].reshape((1, -1, 2)).copy()),
                                                      np.asarray(cam['camera_mtx']), np.asarray(cam['k']))
        xyz_camera_space = np.hstack((xy_undistorted_camspace.squeeze(), self.col(uvd[:, 2])))
        xyz_camera_space[:, :2] *= self.col(xyz_camera_space[:, 2])  # scale x,y by z
        other_answer = xyz_camera_space - self.row(np.asarray(cam['view_mtx'])[:, 3])  # translate
        xyz = other_answer.dot(np.asarray(cam['view_mtx'])[:, :3])  # rotate

        return xyz.reshape((depth_image.shape[0], depth_image.shape[1], -1))

    def projectPoints(self, v, cam):
        v = v.reshape((-1,3)).copy()
        return cv2.projectPoints(v, np.asarray(cam['R']), np.asarray(cam['T']), np.asarray(cam['camera_mtx']), np.asarray(cam['k']))[0].squeeze()

    def create_scan(self, mask, depth_im, color_im=None, mask_on_color=False, coord='color', TH=1e-2, default_color=[1.00, 0.75, 0.80]):
        if not mask_on_color:
            depth_im[mask != 0] = 0
        if depth_im.size == 0:
            return {'v': []}

        points = self.unproject_depth_image(depth_im, self.depth_cam).reshape(-1, 3)
        colors = np.tile(default_color, [points.shape[0], 1])

        uvs = self.projectPoints(points, self.color_cam)
        uvs = np.round(uvs).astype(int)
        valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < 1080)
        valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < 1920)
        valid_idx = np.logical_and(valid_x, valid_y)
        if mask_on_color:
            valid_mask_idx = valid_idx.copy()
            valid_mask_idx[valid_mask_idx == True] = mask[uvs[valid_idx == True][:, 1], uvs[valid_idx == True][:,
                                                                                        0]] == 0
            uvs = uvs[valid_mask_idx == True]
            points = points[valid_mask_idx]
            colors = np.tile(default_color, [points.shape[0], 1])
            # colors = colors[valid_mask_idx]
            valid_idx = valid_mask_idx
            if color_im is not None:
                colors[:, :3] = color_im[uvs[:, 1], uvs[:, 0]] / 255.0
        else:
            uvs = uvs[valid_idx == True]
            if color_im is not None:
                colors[valid_idx == True,:3] = color_im[uvs[:, 1], uvs[:, 0]]/255.0

        if coord == 'color':
            # Transform to color camera coord
            T = np.concatenate([np.asarray(self.color_cam['view_mtx']), np.array([0, 0, 0, 1]).reshape(1, -1)])
            stacked = np.column_stack((points, np.ones(len(points)) ))
            points = np.dot(T, stacked.T).T[:, :3]
            points = np.ascontiguousarray(points)
        ind = points[:, 2] > TH
        return {'points':points[ind], 'colors':colors[ind]}


    def align_color2depth(self, depth_im, color_im, interpolate=True):
        (w_d, h_d) = (512, 424)
        if interpolate:
            # fill depth holes to avoid black spots in aligned rgb image
            zero_mask = np.array(depth_im == 0.).ravel()
            depth_im_flat = depth_im.ravel()
            depth_im_flat[zero_mask] = np.interp(np.flatnonzero(zero_mask), np.flatnonzero(~zero_mask),
                                                 depth_im_flat[~zero_mask])
            depth_im = depth_im_flat.reshape(depth_im.shape)

        points = self.unproject_depth_image(depth_im, self.depth_cam).reshape(-1, 3)
        uvs = self.projectPoints(points, self.color_cam)
        uvs = np.round(uvs).astype(int)
        valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < 1080)
        valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < 1920)
        valid_idx = np.logical_and(valid_x, valid_y)
        uvs = uvs[valid_idx == True]
        aligned_color = np.zeros((h_d, w_d, 3)).astype(color_im.dtype)
        aligned_color[valid_idx.reshape(h_d, w_d)] = color_im[uvs[:, 1], uvs[:, 0]]

        return aligned_color

    def align_depth2color(self, depth_im, depth_raw):
        (w_rgb, h_rgb) = (1920, 1080)
        (w_d, h_d) = (512, 424)
        points = self.unproject_depth_image(depth_im, self.depth_cam).reshape(-1, 3)
        uvs = self.projectPoints(points, self.color_cam)
        uvs = np.round(uvs).astype(int)
        valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < 1080)
        valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < 1920)
        valid_idx = np.logical_and(valid_x, valid_y)
        uvs = uvs[valid_idx == True]

        aligned_depth = np.zeros((h_rgb, w_rgb)).astype('uint16')
        aligned_depth[uvs[:, 1], uvs[:, 0]] = depth_raw[valid_idx.reshape(h_d, w_d)]

        return aligned_depth
