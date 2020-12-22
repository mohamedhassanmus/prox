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

import open3d as o3d
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from psbody.mesh.visibility import visibility_compute
from psbody.mesh import Mesh

import misc_utils as utils
import dist_chamfer as ext
distChamfer = ext.chamferDist()

import matplotlib.pyplot as plt
import matplotlib.cm as cm

@torch.no_grad()
def guess_init(model,
               joints_2d,
               edge_idxs,
               focal_length=5000,
               pose_embedding=None,
               vposer=None,
               use_vposer=True,
               dtype=torch.float32,
               model_type='smpl',
               **kwargs):
    ''' Initializes the camera translation vector

        Parameters
        ----------
        model: nn.Module
            The PyTorch module of the body
        joints_2d: torch.tensor 1xJx2
            The 2D tensor of the joints
        edge_idxs: list of lists
            A list of pairs, each of which represents a limb used to estimate
            the camera translation
        focal_length: float, optional (default = 5000)
            The focal length of the camera
        pose_embedding: torch.tensor 1x32
            The tensor that contains the embedding of V-Poser that is used to
            generate the pose of the model
        dtype: torch.dtype, optional (torch.float32)
            The floating point type used
        vposer: nn.Module, optional (None)
            The PyTorch module that implements the V-Poser decoder
        Returns
        -------
        init_t: torch.tensor 1x3, dtype = torch.float32
            The vector with the estimated camera location

    '''

    body_pose = vposer.decode(
        pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
    if use_vposer and model_type == 'smpl':
        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
        body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    output = model(body_pose=body_pose, return_verts=False,
                   return_full_pose=False)
    joints_3d = output.joints
    joints_2d = joints_2d.to(device=joints_3d.device)

    diff3d = []
    diff2d = []
    for edge in edge_idxs:
        diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
        diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])

    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)

    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()

    height2d = length_2d.mean(dim=1)
    height3d = length_3d.mean(dim=1)

    est_d = focal_length * (height3d / height2d)

    # just set the z value
    batch_size = joints_3d.shape[0]
    x_coord = torch.zeros([batch_size], device=joints_3d.device,
                          dtype=dtype)
    y_coord = x_coord.clone()
    init_t = torch.stack([x_coord, y_coord, est_d], dim=1)
    return init_t


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 viz_mode='mv',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

        self.visualize = visualize
        self.viz_mode = viz_mode

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            if self.viz_mode == 'o3d':
                self.vis_o3d = o3d.Visualizer()
                self.vis_o3d.create_window()
                self.body_o3d = o3d.TriangleMesh()
                self.scan = o3d.PointCloud()

                self.joints_opt = []
                self.joints_gt = []
                for i in range(25):
                    color = cm.jet(i / 25.0)[:3]
                    self.joints_opt.append(self.get_o3d_sphere(color=color))
                    self.joints_gt.append(self.get_o3d_sphere(color=color, radius=0.03))
            else:
                self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            if self.viz_mode == 'o3d':
                self.vis_o3d.close()
            else:
                self.mv.close_viewer()

    def get_o3d_sphere(self, color=[0.3, 1.0, 0.3], pos=[0, 0, 0], radius=0.06):
        mesh_sphere = o3d.geometry.create_mesh_sphere(radius=radius, resolution=5)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color(color)
        mean = np.asarray(mesh_sphere.vertices).mean(axis=0)
        diff = np.asarray(pos) - mean
        print(diff, type(diff))
        mesh_sphere.translate(diff)
        return mesh_sphere

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break
            prev_loss = loss.item()

        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_joints=None, loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               scan_tensor=None,
                               create_graph=False,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(
                pose_embedding, output_type='aa').view(
                    1, -1) if use_vposer else None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)
            total_loss = loss(body_model_output, camera=camera,
                              gt_joints=gt_joints,
                              body_model=body_model,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              scan_tensor=scan_tensor,
                              visualize=self.visualize,
                              **kwargs)

            if backward:
                total_loss.backward(create_graph=create_graph)


            if self.visualize:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()
                joints = model_output.joints.detach().cpu().numpy().squeeze()

                if self.steps == 0 and self.viz_mode == 'o3d':

                    self.body_o3d.vertices = o3d.Vector3dVector(vertices.squeeze())
                    self.body_o3d.triangles = o3d.Vector3iVector(body_model.faces)
                    self.body_o3d.vertex_normals = o3d.Vector3dVector([])
                    self.body_o3d.triangle_normals = o3d.Vector3dVector([])
                    self.body_o3d.compute_vertex_normals()
                    self.vis_o3d.add_geometry(self.body_o3d)

                    joints_gt_3d = camera.inverse_camera_tform(gt_joints, 1.7)

                    # Visualize SMPL joints - Patrick
                    for i in range(25):
                        mean = np.asarray(self.joints_opt[i].vertices).mean(axis=0)
                        self.joints_opt[i].translate(joints[i, :] - mean)
                        self.vis_o3d.add_geometry(self.joints_opt[i])

                        j = joints_gt_3d[0, i, :].detach().cpu().numpy()
                        self.joints_gt[i].translate(j)
                        self.vis_o3d.add_geometry(self.joints_gt[i])

                    # Visualize camera
                    camera_origin = camera.translation.detach().cpu().numpy().squeeze()
                    self.vis_o3d.add_geometry(self.get_o3d_sphere([1.0, 0.0, 0.0], pos=camera_origin))

                    if scan_tensor is not None:
                        self.scan.points = o3d.Vector3dVector(scan_tensor.detach().cpu().numpy().squeeze())
                        N = np.asarray(self.scan.points).shape[0]
                        self.scan.colors = o3d.Vector3dVector(np.tile([1.00, 0.75, 0.80], [N, 1]))
                        self.vis_o3d.add_geometry(self.scan)

                    self.vis_o3d.update_geometry()
                    self.vis_o3d.poll_events()
                    self.vis_o3d.update_renderer()
                elif self.steps % self.summary_steps == 0:
                    if self.viz_mode == 'o3d':
                        self.body_o3d.vertices = o3d.Vector3dVector(vertices.squeeze())
                        self.body_o3d.triangles = o3d.Vector3iVector(body_model.faces)
                        self.body_o3d.vertex_normals = o3d.Vector3dVector([])
                        self.body_o3d.triangle_normals = o3d.Vector3dVector([])
                        self.body_o3d.compute_vertex_normals()

                        # Visualize SMPL joints - Patrick
                        for i in range(25):
                            mean = np.asarray(self.joints_opt[i].vertices).mean(axis=0)
                            self.joints_opt[i].translate(joints[i, :] - mean)

                        self.vis_o3d.update_geometry()
                        self.vis_o3d.poll_events()
                        self.vis_o3d.update_renderer()
                    else:
                        self.mv.update_mesh(vertices.squeeze(),
                                        body_model.faces)
            self.steps += 1

            return total_loss

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


class SMPLifyLoss(nn.Module):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 expr_prior=None,
                 angle_prior=None,
                 jaw_prior=None,
                 use_joints_conf=True,
                 use_face=True, use_hands=True,
                 left_hand_prior=None, right_hand_prior=None,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0, jaw_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 s2m=False,
                 m2s=False,
                 rho_s2m=1,
                 rho_m2s=1,
                 s2m_weight=0.0,
                 m2s_weight=0.0,
                 head_mask=None,
                 body_mask=None,
                 sdf_penetration=False,
                 voxel_size=None,
                 grid_min=None,
                 grid_max=None,
                 sdf=None,
                 sdf_normals=None,
                 sdf_penetration_weight=0.0,
                 R=None,
                 t=None,
                 contact=False,
                 contact_loss_weight=0.0,
                 contact_verts_ids=None,
                 rho_contact=0.0,
                 contact_angle=0.0,
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.s2m = s2m
        self.m2s = m2s
        self.s2m_robustifier = utils.GMoF(rho=rho_s2m)
        self.m2s_robustifier = utils.GMoF(rho=rho_m2s)

        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior

        self.body_mask = body_mask
        self.head_mask = head_mask

        self.R = R
        self.t = t

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance


        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                                 torch.tensor(hand_prior_weight, dtype=dtype))
        if self.use_face:
            self.register_buffer('expr_prior_weight',
                                 torch.tensor(expr_prior_weight, dtype=dtype))
            self.register_buffer('jaw_prior_weight',
                                 torch.tensor(jaw_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))

        self.register_buffer('s2m_weight',
                             torch.tensor(s2m_weight, dtype=dtype))
        self.register_buffer('m2s_weight',
                             torch.tensor(m2s_weight, dtype=dtype))

        self.sdf_penetration = sdf_penetration
        if self.sdf_penetration:
            self.sdf = sdf
            self.sdf_normals = sdf_normals
            self.voxel_size = voxel_size
            self.grid_min = grid_min
            self.grid_max = grid_max
            self.register_buffer('sdf_penetration_weight',
                                 torch.tensor(sdf_penetration_weight, dtype=dtype))
        self.contact = contact
        if self.contact:
            self.contact_verts_ids = contact_verts_ids
            self.rho_contact = rho_contact
            self.contact_angle = contact_angle
            self.register_buffer('contact_loss_weight',
                                 torch.tensor(contact_loss_weight, dtype=dtype))
            self.contact_robustifier = utils.GMoF_unscaled(rho=self.rho_contact)

    def reset_loss_weights(self, loss_weight_dict):
            for key in loss_weight_dict:
                if hasattr(self, key):
                    weight_tensor = getattr(self, key)
                    if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                        weight_tensor = loss_weight_dict[key].clone().detach()
                    else:
                        weight_tensor = torch.tensor(loss_weight_dict[key],
                                                     dtype=weight_tensor.dtype,
                                                     device=weight_tensor.device)
                    setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, joints_conf,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embedding=None,
                scan_tensor=None, visualize=False,
                scene_v=None, scene_vn=None, scene_f=None,ftov=None,
                **kwargs):
        projected_joints = camera(body_model_output.joints)
        # Calculate the weights for each joints
        weights = (joint_weights * joints_conf
                   if self.use_joints_conf else
                   joint_weights).unsqueeze(dim=-1)

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        joint_err = gt_joints - projected_joints
        joint_diff = self.robustifier(gt_joints - projected_joints)
        joint_loss = (torch.sum(weights ** 2 * joint_diff) *
                      self.data_weight ** 2)

        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2

        shape_loss = torch.sum(self.shape_prior(
            body_model_output.betas)) * self.shape_weight ** 2
        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight ** 2

        # Apply the prior on the pose space of the hand
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = torch.sum(
                self.left_hand_prior(
                    body_model_output.left_hand_pose)) * \
                self.hand_prior_weight ** 2

        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = torch.sum(
                self.right_hand_prior(
                    body_model_output.right_hand_pose)) * \
                self.hand_prior_weight ** 2

        expression_loss = 0.0
        jaw_prior_loss = 0.0
        if self.use_face:
            expression_loss = torch.sum(self.expr_prior(
                body_model_output.expression)) * \
                self.expr_prior_weight ** 2

            if hasattr(self, 'jaw_prior'):
                jaw_prior_loss = torch.sum(
                    self.jaw_prior(
                        body_model_output.jaw_pose.mul(
                            self.jaw_prior_weight)))

        pen_loss = 0.0
        # Calculate the loss due to interpenetration
        if (self.interpenetration and self.coll_loss_weight.item() > 0):
            batch_size = projected_joints.shape[0]
            triangles = torch.index_select(
                body_model_output.vertices, 1,
                body_model_faces).view(batch_size, -1, 3, 3)

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)

            # Remove unwanted collisions
            if self.tri_filtering_module is not None:
                collision_idxs = self.tri_filtering_module(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(
                    self.coll_loss_weight *
                    self.pen_distance(triangles, collision_idxs))

        s2m_dist = 0.0
        m2s_dist = 0.0
        # calculate the scan2mesh and mesh2scan loss from the sparse point cloud
        if (self.s2m or self.m2s) and (
                self.s2m_weight > 0 or self.m2s_weight > 0) and scan_tensor is not None:
            vertices_np = body_model_output.vertices.detach().cpu().numpy().squeeze()
            body_faces_np = body_model_faces.detach().cpu().numpy().reshape(-1, 3)
            m = Mesh(v=vertices_np, f=body_faces_np)

            (vis, n_dot) = visibility_compute(v=m.v, f=m.f, cams=np.array([[0.0, 0.0, 0.0]]))
            vis = vis.squeeze()

            if self.s2m and self.s2m_weight > 0 and vis.sum() > 0:
                s2m_dist, _, _, _ = distChamfer(scan_tensor,
                                                body_model_output.vertices[:, np.where(vis > 0)[0], :])
                s2m_dist = self.s2m_robustifier(s2m_dist.sqrt())
                s2m_dist = self.s2m_weight * s2m_dist.sum()
            if self.m2s and self.m2s_weight > 0 and vis.sum() > 0:
                _, m2s_dist, _, _ = distChamfer(scan_tensor,
                                                body_model_output.vertices[:, np.where(np.logical_and(vis > 0, self.body_mask))[0], :])

                m2s_dist = self.m2s_robustifier(m2s_dist.sqrt())
                m2s_dist = self.m2s_weight * m2s_dist.sum()

        # Transform vertices to world coordinates
        if self.R is not None and self.t is not None:
            vertices = body_model_output.vertices
            nv = vertices.shape[1]
            vertices.squeeze_()
            vertices = self.R.mm(vertices.t()).t() + self.t.repeat([nv, 1])
            vertices.unsqueeze_(0)

        # Compute scene penetration using signed distance field (SDF)
        sdf_penetration_loss = 0.0
        if self.sdf_penetration and self.sdf_penetration_weight > 0:
            grid_dim = self.sdf.shape[0]
            sdf_ids = torch.round(
               (vertices.squeeze() - self.grid_min) / self.voxel_size).to(dtype=torch.long)
            sdf_ids.clamp_(min=0, max=grid_dim-1)

            norm_vertices = (vertices - self.grid_min) / (self.grid_max - self.grid_min) * 2 - 1
            body_sdf = F.grid_sample(self.sdf.view(1, 1, grid_dim, grid_dim, grid_dim),
                                     norm_vertices[:, :, [2, 1, 0]].view(1, nv, 1, 1, 3),
                                     padding_mode='border')
            sdf_normals = self.sdf_normals[sdf_ids[:,0], sdf_ids[:,1], sdf_ids[:,2]]
            # if there are no penetrating vertices then set sdf_penetration_loss = 0
            if body_sdf.lt(0).sum().item() < 1:
                sdf_penetration_loss = torch.tensor(0.0, dtype=joint_loss.dtype, device=joint_loss.device)
            else:
              if sdf_normals is None:
                sdf_penetration_loss = self.sdf_penetration_weight * (body_sdf[body_sdf < 0].unsqueeze(dim=-1).abs()).pow(2).sum(dim=-1).sqrt().sum()
              else:
                sdf_penetration_loss = self.sdf_penetration_weight * (body_sdf[body_sdf < 0].unsqueeze(dim=-1).abs() * sdf_normals[body_sdf.view(-1) < 0, :]).pow(2).sum(dim=-1).sqrt().sum()

        # Compute the contact loss
        contact_loss = 0.0
        if self.contact and self.contact_loss_weight >0:
            # select contact vertices
            contact_body_vertices = vertices[:, self.contact_verts_ids, :]
            contact_dist, _, idx1, _ = distChamfer(
                contact_body_vertices.contiguous(), scene_v)

            body_triangles = torch.index_select(
                vertices, 1,
                body_model_faces).view(1, -1, 3, 3)
            # Calculate the edges of the triangles
            # Size: BxFx3
            edge0 = body_triangles[:, :, 1] - body_triangles[:, :, 0]
            edge1 = body_triangles[:, :, 2] - body_triangles[:, :, 0]
            # Compute the cross product of the edges to find the normal vector of
            # the triangle
            body_normals = torch.cross(edge0, edge1, dim=2)
            # Normalize the result to get a unit vector
            body_normals = body_normals / \
                torch.norm(body_normals, 2, dim=2, keepdim=True)
            # compute the vertex normals
            body_v_normals = torch.mm(ftov, body_normals.squeeze())
            body_v_normals = body_v_normals / \
                torch.norm(body_v_normals, 2, dim=1, keepdim=True)

            # vertix normals of contact vertices
            contact_body_verts_normals = body_v_normals[self.contact_verts_ids, :]
            # scene normals of the closest points on the scene surface to the contact vertices
            contact_scene_normals = scene_vn[:, idx1.squeeze().to(
                dtype=torch.long), :].squeeze()

            # compute the angle between contact_verts normals and scene normals
            angles = torch.asin(
                torch.norm(torch.cross(contact_body_verts_normals, contact_scene_normals), 2, dim=1, keepdim=True)) *180 / np.pi

            # consider only the vertices which their normals match
            valid_contact_mask = (angles.le(self.contact_angle) + angles.ge(180 - self.contact_angle)).ge(1)
            valid_contact_ids = valid_contact_mask.squeeze().nonzero().squeeze()

            contact_dist = self.contact_robustifier(contact_dist[:, valid_contact_ids].sqrt())
            contact_loss = self.contact_loss_weight * contact_dist.mean()

        total_loss = (joint_loss + pprior_loss + shape_loss +
                      angle_prior_loss + pen_loss +
                      jaw_prior_loss + expression_loss +
                      left_hand_prior_loss + right_hand_prior_loss + m2s_dist + s2m_dist
                      + sdf_penetration_loss + contact_loss)
        if visualize:
            # print('total:{:.2f}, joint_loss:{:0.2f},  s2m:{:0.2f}, m2s:{:0.2f}, penetration:{:0.2f}, contact:{:0.2f}'.
            #       format(total_loss.item(), joint_loss.item() ,torch.tensor(s2m_dist).item(),
            #              torch.tensor(m2s_dist).item() ,torch.tensor(sdf_penetration_loss).item(), torch.tensor(contact_loss).item()))
            # print('pprior:{:.2f}, shape:{:.2f}, angle_pri:{:.2f}, pen:{:.2f}, jaw:{:.2f}, expres:{:.2f}'.format(
            #     pprior_loss.item(), shape_loss.item(), angle_prior_loss.item(),
            #     torch.tensor(pen_loss).item(), torch.tensor(jaw_prior_loss).item(), torch.tensor(expression_loss).item()))

            print('tot:{:.2f}, j_loss:{:0.2f}, s2m:{:0.2f}, m2s:{:0.2f}, pprior:{:.2f}, shape:{:.2f}, ang_pri:{:.2f}, pen:{:.2f}'.
                  format(total_loss.item(), joint_loss.item() ,torch.tensor(s2m_dist).item(),
                         torch.tensor(m2s_dist).item(), pprior_loss.item(), shape_loss.item(),
                         angle_prior_loss.item(), torch.tensor(pen_loss).item()))
        return total_loss


class SMPLifyCameraInitLoss(nn.Module):

    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2,
                 camera_mode='moving',
                 dtype=torch.float32,
                 **kwargs):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype
        self.camera_mode = camera_mode

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, body_model,
                **kwargs):

        projected_joints = camera(body_model_output.joints)

        joint_err = gt_joints - projected_joints
        joint_err_sel = torch.index_select(joint_err, 1, self.init_joints_idxs).cpu().detach().numpy()

        # plt.scatter(gt_joints[0, :, 0].cpu().detach().numpy(), gt_joints[0, :, 1].cpu().detach().numpy())
        # plt.scatter(projected_joints[0, :, 0].cpu().detach().numpy(), projected_joints[0, :, 1].cpu().detach().numpy())
        # plt.show()

        joint_error = torch.pow(
            torch.index_select(gt_joints, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2

        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
                None):
            if self.camera_mode == 'moving':
                depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                                                                             camera.translation[:,
                                                                             2] - self.trans_estimation[:, 2]).pow(2))
            elif self.camera_mode == 'fixed':
                depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                    body_model.transl[:, 2] - self.trans_estimation[:, 2]).pow(2))



        return joint_loss + depth_loss
