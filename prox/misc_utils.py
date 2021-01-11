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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import numpy as np

import torch
import torch.nn as nn

import pyrender

from pyquaternion import Quaternion
from PIL import Image, ImageFont, ImageDraw
import open3d as o3d


def get_data_from_batched_dict(in_dict, batch_idx, batch_n):
    out_dict = dict()
    out_dict['batch_idx'] = batch_idx

    for key, value in in_dict.items():
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()

        if isinstance(value, np.ndarray) and len(value.shape) >= 1 and value.shape[0] == batch_n:
            out_dict[key] = value[batch_idx, ...]
        elif isinstance(value, dict):
            out_dict[key] = get_data_from_batched_dict(value, batch_idx, batch_n)
        else:
            out_dict[key] = value

    return out_dict


def batched_index_select(t, dim, inds):
    """
    Helper function to extract batch-varying indicies along array
    :param t: array to select from
    :param dim: dimension to select along
    :param inds: batch-vary indicies
    :return:
    """
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out


def text_3d(text, pos, direction=None, degree=-90.0, density=10, font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', font_size=10):
    """
    Generate a Open3D text point cloud used for visualization.
    https://github.com/intel-isl/Open3D/issues/2
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    # font_obj = ImageFont.truetype(font, font_size)
    font_obj = ImageFont.truetype(font, int(font_size * density))
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    # pcd.points = o3d.utility.Vector3dVector(indices / 100.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 1000 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def to_tensor(tensor, dtype=torch.float32):
    if torch.Tensor == type(tensor):
        return tensor.clone().detach()
    else:
        return torch.tensor(tensor, dtype)


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


def max_grad_change(grad_arr):
    return grad_arr.abs().max()


class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist

class GMoF_unscaled(nn.Module):
    def __init__(self, rho=1):
        super(GMoF_unscaled, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return dist

def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))
