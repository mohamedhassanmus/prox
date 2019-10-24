import os
import os.path as osp
import cv2
import numpy as np
import json
import trimesh
import argparse
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import PIL.Image as pil_img
import pickle
import smplx
import torch

def main(args):
    fitting_dir = args.fitting_dir
    recording_name = os.path.abspath(fitting_dir).split("/")[-1]
    female_subjects_ids = [162, 3452, 159, 3403]
    subject_id = int(recording_name.split('_')[1])
    if subject_id in female_subjects_ids:
        gender = 'female'
    else:
        gender = 'male'
    pkl_files_dir = osp.join(fitting_dir, 'results')
    scene_name = recording_name.split("_")[0]
    base_dir = args.base_dir
    cam2world_dir = osp.join(base_dir, 'cam2world')
    scene_dir = osp.join(base_dir, 'scenes')
    recording_dir = osp.join(base_dir, 'recordings', recording_name)
    color_dir = os.path.join(recording_dir, 'Color')
    meshes_dir = os.path.join(fitting_dir, 'meshes')
    rendering_dir = os.path.join(fitting_dir, 'images')

    body_model = smplx.create(args.model_folder, model_type='smplx',
                         gender=gender, ext='npz',
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

    if args.rendering_mode == '3d' or args.rendering_mode == 'both':
        static_scene = trimesh.load(osp.join(scene_dir, scene_name + '.ply'))
        with open(os.path.join(cam2world_dir,scene_name + '.json'), 'r') as f:
            trans = np.array(json.load(f))
        trans = np.linalg.inv(trans)
        static_scene.apply_transform(trans)

        body_scene_rendering_dir = os.path.join(fitting_dir, 'renderings')
        if not osp.exists(body_scene_rendering_dir):
            os.mkdir(body_scene_rendering_dir)

    #common
    H, W = 1080, 1920
    camera_center = np.array([951.30, 536.77])
    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
    camera = pyrender.camera.IntrinsicsCamera(
        fx=1060.53, fy=1060.38,
        cx=camera_center[0], cy=camera_center[1])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

    for img_name in sorted(os.listdir(pkl_files_dir)):
        print('viz frame {}'.format(img_name))
        with open(osp.join(pkl_files_dir, img_name, '000.pkl'), 'rb') as f:
            param = pickle.load(f)
        torch_param = {}
        for key in param.keys():
            if key in ['pose_embedding', 'camera_rotation', 'camera_translation']:
                continue
            else:
                torch_param[key] = torch.tensor(param[key])

        output = body_model(return_verts=True, **torch_param)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        body = trimesh.Trimesh(vertices, body_model.faces, process=False)
        if args.save_meshes:
            body.export(osp.join(meshes_dir,img_name, '000.ply'))

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        body_mesh = pyrender.Mesh.from_trimesh(
            body, material=material)

        if args.rendering_mode == 'body' or args.rendering_mode == 'both':
            img = cv2.imread(os.path.join(color_dir, img_name + '.jpg'))[:, :, ::-1] / 255.0
            H, W, _ = img.shape
            img = cv2.flip(img, 1)

            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                   ambient_light=(0.3, 0.3, 0.3))
            scene.add(camera, pose=camera_pose)
            scene.add(light, pose=camera_pose)

            scene.add(body_mesh, 'mesh')

            r = pyrender.OffscreenRenderer(viewport_width=W,
                                           viewport_height=H,
                                           point_size=1.0)
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0

            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img
            output_img = (color[:, :, :-1] * valid_mask +
                          (1 - valid_mask) * input_img)

            img = pil_img.fromarray((output_img * 255).astype(np.uint8))
            img.save(os.path.join(rendering_dir, img_name, '000',  'output.png'))

        if args.rendering_mode == '3d' or args.rendering_mode == 'both':
            static_scene_mesh = pyrender.Mesh.from_trimesh(
                static_scene)

            scene = pyrender.Scene()
            scene.add(camera, pose=camera_pose)
            scene.add(light, pose=camera_pose)

            scene.add(static_scene_mesh, 'mesh')
            body_mesh = pyrender.Mesh.from_trimesh(
                body, material=material)
            scene.add(body_mesh, 'mesh')

            r = pyrender.OffscreenRenderer(viewport_width=W,
                                           viewport_height=H)
            color, _ = r.render(scene)
            color = color.astype(np.float32) / 255.0
            img = pil_img.fromarray((color * 255).astype(np.uint8))
            img.save(os.path.join(body_scene_rendering_dir, img_name + '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fitting_dir', type=str, default=os.getcwd(),
                        help='recording dir')
    parser.add_argument('--base_dir', type=str, default=os.getcwd(),
                        help='recording dir')
    parser.add_argument('--start', type=int, default=0, help='id of the starting frame')
    parser.add_argument('--step', type=int, default=1, help='id of the starting frame')
    parser.add_argument('--model_folder', default='models', type=str, help='')
    parser.add_argument('--num_pca_comps', type=int, default=12,help='')
    parser.add_argument('--save_meshes', type=lambda arg: arg.lower() in ['true', '1'],
                 default=True, help='')
    parser.add_argument('--rendering_mode', default='both', type=str,
                choices=['overlay', '3d', 'both'],
                help='')

    args = parser.parse_args()
    main(args)
