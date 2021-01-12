import numpy as np
import open3d as o3d
import torch
import smplx
import trimesh
from prox.misc_utils import text_3d

# Joint reference here https://github.com/gulvarol/smplpytorch
# https://github.com/Healthcare-Robotics/bodies-at-rest/blob/master/lib_py/mesh_depth_lib_br.py#L70
axang_limits_patrick = np.array(  # In degrees
    [
     # [-140, 10], [-60, 120], [-40, 60],  # Hip L 0
     # [-140, 10], [-120, 60], [-60, 40],  # Hip R
     [-120., 10.], [-60., 120.], [-60., 60.],   # Hip L henry
     [-120., 10.], [-120., 60.], [-60., 60.],   # Hip R henry
     [-30., 110.], [-8., 8.], [-8., 8.],  # Lower back
     [-1.3, 139.9], [-0.6, 0.6], [-0.6, 0.6],  # Knee L
     [-1.3, 139.9], [-0.6, 0.6], [-0.6, 0.6],  # Knee R
     [-20., 20.], [-8., 8.], [-8., 8.],  # Mid back 5
     [-20., 50.], [-20., 20.], [-20., 20.],  # Ankle L
     [-20., 50.], [-20., 20.], [-20., 20.],  # Ankle R
     [-20., 20.], [-8., 8.], [-8., 8.],  # Upper back
     [-0.6, 0.6], [-0.6, 0.6], [-0.6, 0.6],  # Foot L?
     [-0.6, 0.6], [-0.6, 0.6], [-0.6, 0.6],  # Foot R? 10
     [-45., 60.], [-15., 15.], [-20., 20.],  # Lower neck
     # [-40., 50], [-50, 20], [-40., 50],  # Inner shoulder L
     # [-40., 50], [-20, 50], [-50, 40.],  # Inner shoulder R
     [-40., 40.], [-50., 20.], [-40., 40.],  # Inner shoulder L
     [-40., 40.], [-20., 50.], [-40., 40.],  # Inner shoulder R
     [-30., 60.], [-5., 5.], [-5., 5.],  # Upper neck
     # [-70., 70], [-90, 35], [-90., 50],  # Outer shoulder L 15
     # [-70., 70], [-35, 90], [-50, 90.],  # Outer shoulder R
     [-70., 70.], [-90., 35.], [-90., 60.],  # Outer shoulder L 15
     [-70., 70.], [-30., 90.], [-60., 90.],  # Outer shoulder R
     [-0.6, 0.6], [-160, 2.7], [-0.6, 0.6],  # Elbow L
     [-0.6, 0.6], [-2.7, 160], [-0.6, 0.6],  # Elbow R
     [-30., 30.], [-15., 15.], [-30., 30.],  # Wrist L
     [-30., 30.], [-15., 15.], [-30., 30.],  # Wrist R 20
     [-0.6, 0.6], [-0.6, 0.6], [-0.6, 0.6],  # Fingers L?
     [-0.6, 0.6], [-0.6, 0.6], [-0.6, 0.6]]  # Fingers R?
)


def get_smpl(joints, axes, amounts, translation=(0, 0, 0)):
    model = smplx.create('models', model_type='smpl', gender='male')

    body_pose = torch.zeros([1, 69])
    for i in range(len(joints)):
        pose_index = int(joints[i] * 3 + axes[i])
        body_pose[0, pose_index] = axang_mean[pose_index] + axang_var[pose_index] * (amounts[i] * 2 - 1)

    output = model(body_pose=torch.Tensor(body_pose), return_verts=True)
    smpl_vertices = output.vertices.detach().cpu().numpy().squeeze()

    smpl_o3d = o3d.TriangleMesh()
    smpl_o3d.triangles = o3d.Vector3iVector(model.faces)
    smpl_o3d.vertices = o3d.Vector3dVector(smpl_vertices + np.array(translation))
    smpl_o3d.compute_vertex_normals()
    smpl_o3d.paint_uniform_color([amounts[0]/2 + 0.5, 0.3, 0.3])

    return smpl_o3d


def view_fit(joint, translation=(0, 0, 0)):
    translation = np.array(translation)
    vis = o3d.Visualizer()
    vis.create_window()
    vis.add_geometry(get_smpl([joint], [0], [0.5], translation=translation))

    for i in range(6):
        trans = translation + (int(i/2)*1.5 + 1, 0, 0)
        vis.add_geometry(get_smpl([joint], [i / 2], [i % 2], translation=trans))

    lbl = 'Joint {} dark min red max'.format(joint)
    vis.add_geometry(text_3d(lbl, translation + (0, 1, 0), font_size=200, density=0.2))

    vis.run()
    vis.destroy_window()


def view_multi(joints, axes, amounts):
    vis = o3d.Visualizer()
    vis.create_window()
    vis.add_geometry(get_smpl(joints, axes, amounts))

    lbl = 'Joints {} {} {} dark min red max'.format(joints, axes, amounts)
    vis.add_geometry(text_3d(lbl, (0, 1, 0), font_size=200, density=0.2))

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    axang_limits = torch.tensor(axang_limits_patrick / 180 * np.pi)
    axang_mean = axang_limits.detach().mean(1)
    axang_var = torch.abs(axang_limits[:, 1] - axang_mean)

    # view_multi([2, 5, 8], [0, 0, 0], [0, 0, 0]) # Back
    # view_multi([2, 5, 8], [0, 0, 0], [1, 1, 1])
    # view_multi([2, 5, 8], [1, 1, 1], [0, 0, 0]) # Back
    # view_multi([2, 5, 8], [1, 1, 1], [1, 1, 1])
    # view_multi([2, 5, 8], [2, 2, 2], [0, 0, 0]) # Back
    # view_multi([2, 5, 8], [2, 2, 2], [1, 1, 1])

    # view_multi([11, 14], [0, 0], [0, 0]) # Neck
    # view_multi([11, 14], [0, 0], [1, 1]) # Neck
    # view_multi([11, 14], [1, 1], [0, 0]) # Neck
    # view_multi([11, 14], [1, 1], [1, 1]) # Neck
    # view_multi([11, 14], [2, 2], [0, 0]) # Neck
    # view_multi([11, 14], [2, 2], [1, 1]) # Neck

    # view_multi([12, 15, 17], [0, 0, 1], [0, 0, 0.5]) # Shoulder
    # view_multi([12, 15, 17], [0, 0, 1], [1, 1, 0.5]) # Shoulder
    # view_multi([12, 15, 17], [1, 1, 1], [0, 0, 0.5]) # Shoulder
    # view_multi([12, 15, 17], [1, 1, 1], [1, 1, 0.5]) # Shoulder
    # view_multi([12, 15, 17], [2, 2, 1], [0, 0, 0.5]) # Shoulder
    # view_multi([12, 15, 17], [2, 2, 1], [1, 1, 0.5]) # Shoulder


    for i in range(0, 23):
        view_fit(i)
