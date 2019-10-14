import os
import os.path as osp
import cv2
import numpy as np
import json
import open3d as o3d
import argparse

import torch
import pickle
import smplx

def main(args):
    fitting_dir = args.fitting_dir
    recording_name = os.path.abspath(fitting_dir).split("/")[-1]
    fitting_dir = osp.join(fitting_dir, 'results')
    scene_name = recording_name.split("_")[0]
    base_dir = args.base_dir
    scene_dir = osp.join(base_dir, 'scenes')
    recording_dir = osp.join(base_dir, 'recordings', recording_name)
    color_dir = os.path.join(recording_dir, 'Color')

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    vis = o3d.Visualizer()
    vis.create_window()

    scene = o3d.io.read_triangle_mesh(osp.join(scene_dir, scene_name + '.ply'))
    with open(os.path.join(base_dir, 'vicon2scene.json'), 'r') as f:
        trans = np.array(json.load(f))
    vis.add_geometry(scene)


    model = smplx.create(args.model_folder, model_type='smplx',
                         gender=args.gender, ext='npz',
                         num_pca_comps=args.num_pca_comps,
                         create_global_orient=True,
                         create_body_pose=True,
                         create_betas=True,
                         create_left_hand_pose=True,
                         create_right_hand_pose=True,
                         create_expression=True,
                         create_jaw_pose=True,
                         create_leye_pose=True,
                         create_reye_pose=True,
                         create_transl=True
                         )

    count = 0
    for img_name in sorted(os.listdir(fitting_dir))[args.start::args.step]:
        print('viz frame {}'.format(img_name))

        with open(osp.join(fitting_dir, img_name, '000.pkl'), 'rb') as f:
            param = pickle.load(f, encoding='latin1')

        torch_param = {}
        for key in param.keys():
            torch_param[key] = torch.tensor(param[key])
        output = model(return_verts=True, **torch_param)
        vertices = output.vertices.detach().cpu().numpy().squeeze()

        if count == 0:
            body = o3d.TriangleMesh()
            vis.add_geometry(body)
        body.vertices = o3d.Vector3dVector(vertices)
        body.triangles = o3d.Vector3iVector(model.faces)
        body.vertex_normals = o3d.Vector3dVector([])
        body.triangle_normals = o3d.Vector3dVector([])
        body.compute_vertex_normals()
        body.transform(trans)


        color_img = cv2.imread(os.path.join(color_dir, img_name + '.jpg'))
        color_img = cv2.flip(color_img, 1)

        vis.update_geometry()
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
    parser.add_argument('fitting_dir', type=str, default=os.getcwd(),
                        help='recording dir')
    parser.add_argument('--base_dir', type=str, default=os.getcwd(),
                        help='recording dir')
    parser.add_argument('--start', type=int, default=0, help='id of the starting frame')
    parser.add_argument('--step', type=int, default=1, help='id of the starting frame')
    parser.add_argument('--model_folder', default='~/models', type=str, help='')
    parser.add_argument('--num_pca_comps', type=int, default=12, help='')
    parser.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'],
                        help='Use gender neutral or gender specific SMPL' +
                             'model')
    args = parser.parse_args()
    main(args)

