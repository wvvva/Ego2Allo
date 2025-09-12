'''
Scene Abstraction:
- Predict depth
- Model: Depth Pro (https://github.com/apple/ml-depth-pro)
'''

import os
import cv2
import sys
sys.path.append("..")
import yaml
from typing import List
import numpy as np
from PIL import Image
import torch
from box import Box
import open3d as o3d
from scipy.stats import mode

# Import modules
import depth_pro
from depth_pro.depth_pro import DepthProConfig
from .src.omni3d.cubercnn import util as cube_utils
from .src.omni3d.cubercnn import vis as cube_vis

class DepthModule:
    def __init__(self, config, device="cuda"):
        self.device = device
        self.config = config

        # Load Depth Pro
        self.depth_model, self.depth_transform = depth_pro.create_model_and_transforms(
            config=DepthProConfig(
                patch_encoder_preset="dinov2l16_384",
                image_encoder_preset="dinov2l16_384",
                checkpoint_uri=config.depth.ckpt_path,
                decoder_features=256,
                use_fov_head=True,
                fov_encoder_preset="dinov2l16_384",
            ),
            device=torch.device(self.device),
            precision=torch.float16
        )
        self.depth_model.eval()
        print("* [INFO] Loaded depth model")

    def depth_process_image(
        self,
        image: Image.Image,
        auto_rotate: bool = True, 
        remove_alpha: bool = True
    ):
        '''
        Transform PIL image to depth_pro format
        (modified from depth_pro/utils.py)
        '''
        img_exif = depth_pro.utils.extract_exif(image)
        icc_profile = image.info.get("icc_profile", None)

        # Rotate the image.
        if auto_rotate:
            exif_orientation = img_exif.get("Orientation", 1)
            if exif_orientation == 3:
                image = image.transpose(Image.ROTATE_180)
            elif exif_orientation == 6:
                image = image.transpose(Image.ROTATE_270)
            elif exif_orientation == 8:
                image = image.transpose(Image.ROTATE_90)
            elif exif_orientation != 1:
                print(f"Ignoring image orientation {exif_orientation}.")

        # Convert to numpy array
        image_npy = np.array(image)

        # Convert to RGB if single channel.
        if image_npy.ndim < 3 or image_npy.shape[2] == 1:
            image_npy = np.dstack((image_npy, image_npy, image_npy))

        if remove_alpha:
            image_npy = image_npy[:, :, :3]

        # Extract the focal length from exif data.
        f_35mm = img_exif.get(
            "FocalLengthIn35mmFilm",
            img_exif.get(
                "FocalLenIn35mmFilm", img_exif.get("FocalLengthIn35mmFormat", None)
            ),
        )
        if f_35mm is not None and f_35mm > 0:
            f_px = depth_pro.utils.fpx_from_f35(image_npy.shape[1], image_npy.shape[0], f_35mm)
        else:
            f_px = None
        
        return image_npy, icc_profile, f_px

    def make_bbox_corners(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        dx: float,
        dy: float,
        dz: float,
    ):
        '''
        Make 8 corners of a bounding box
        '''
        corners_flag = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]

        corners = []
        for flag in corners_flag:
            c = np.array([x_min, y_min, z_min]) + \
                np.array(flag) * \
                np.array([dx, dy, dz])
            corners.append(c)

        return np.array(corners)
    
    def box3D_get_dims(self, bbox3D):
        # Modified from https://github.com/UVA-Computer-Vision-Lab/ovmono3d/blob/main/tools/ovmono3d_geo.py
        x = np.sqrt(np.sum((bbox3D[0] - bbox3D[1]) * (bbox3D[0] - bbox3D[1])))
        y = np.sqrt(np.sum((bbox3D[0] - bbox3D[3]) * (bbox3D[0] - bbox3D[3])))
        z = np.sqrt(np.sum((bbox3D[0] - bbox3D[4]) * (bbox3D[0] - bbox3D[4])))
        
        return np.array([z, y, x])

    def box3D_get_pose(self, bbox3d_a, bbox3d_b):
        # Modified from https://github.com/UVA-Computer-Vision-Lab/ovmono3d/blob/main/tools/ovmono3d_geo.py
        center = np.mean(bbox3d_a, axis=0)
        dim_a = self.box3D_get_dims(bbox3d_a)
        dim_b = self.box3D_get_dims(bbox3d_b)
        bbox3d_a -= center
        bbox3d_b -= center
        U, _, Vt = np.linalg.svd(bbox3d_a.T @ bbox3d_b, full_matrices=True)
        R = U @ Vt

        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R
    
    def run_depth_estimation(
        self,
        image: Image.Image,
    ):
        '''
        Run depth estimation
        '''
        # Process image
        image, _, f_px = self.depth_process_image(image)
        image = self.depth_transform(image)

        # Run Depth Pro
        depth_pred = self.depth_model.infer(image, f_px=f_px)
        depth = depth_pred["depth"]
        depth_npy = depth.cpu().numpy().astype(np.float32)

        return depth_npy
    
    def unproject_to_3D(
        self,
        image: Image.Image,
        depth: np.array,
        segment_mask: np.array,
        return_box3D: bool = False,    # whether to return 3D bbox (for visualization)
    ):
        '''
        Using detection + segmentation + depth results, unproject each object to 3D points.
        
        NOTE: Modified from 3D bbox extraction pipeline in Ovmono3D [Yao et al., 2024]
        (https://github.com/UVA-Computer-Vision-Lab/ovmono3d/blob/main/tools/ovmono3d_geo.py)
        '''

        # Set camera intrinsic matrix (K)
        image_w, image_h = image.size
        focal_len_ndc = 4.0
        focal_len = focal_len_ndc * image_w
        px, py = image_w / 2, image_h / 2
        
        K = np.array([
            [focal_len, 0.0, px],
            [0.0, focal_len, py],
            [0.0, 0.0, 1.0]
        ])

        # Convert to Open3D format
        depth_o3d = o3d.geometry.Image(depth)
        image_o3d = o3d.geometry.Image(np.array(image).astype(np.uint8))

        # Unproject (Open3D)
        depth_unproj = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=image_o3d,
            depth=depth_o3d,
            depth_scale=1.0,
            depth_trunc=1000.0,
            convert_rgb_to_intensity=False,
        ).depth
        depth_unproj = np.array(depth_unproj)
        
        # Filter out points outside the mask
        ys, xs = np.where(segment_mask > 0.5)
        depth_values = []
        for y, x in zip(ys, xs):
            z = depth_unproj[y, x]
            depth_values.append(z)

        # Find the most frequent depth
        depth_values = np.array(depth_values)
        mode_depth, _ = mode(depth_values, keepdims=True)   # scipy mode function

        # Set a depth threshold (e.g., keep points within ±10% of the mode)
        threshold = 0.1 * mode_depth[0]
        depth_values_filtered = (
            depth_values >= mode_depth[0] - threshold
        ) & (
            depth_values <= mode_depth[0] + threshold
        )

        # Unproject only filtered points
        points_filtered = []
        for i, (y, x) in enumerate(zip(ys, xs)):
            z = depth_unproj[y, x]
            if depth_values_filtered[i]:
                x_3D = z * (x - K[0, 2]) / K[0, 0]
                y_3D = z * (y - K[1, 2]) / K[1, 1]
                points_filtered.append([x_3D, -y_3D, -z])  # flip
        
        # Compute final 3D coordinates as the median of filtered points
        points_filtered = np.array(points_filtered)
        med_x, med_y, med_z = np.median(points_filtered, axis=0)

        # Make a dummy bounding box
        box3D_size = 0.02 * mode_depth[0]

        box3D_pseudo = self.make_bbox_corners(
            med_x - box3D_size/2, med_y - box3D_size/2, med_z - box3D_size/2,
            box3D_size, box3D_size, box3D_size
        )
        flip_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        box3D_pseudo = box3D_pseudo.dot(flip_matrix)

        # Define bbox vertice, face, and pose
        box3D_dims = torch.from_numpy(self.box3D_get_dims(box3D_pseudo)).unsqueeze(0)
        bbox3D = torch.from_numpy(np.mean(box3D_pseudo, axis=0)).unsqueeze(0)
        box3D_pose = torch.eye(3).unsqueeze(0)

        # Obtain 3D bbox vertices (CubeRCNN, Omni3D)
        box3D_ = cube_utils.get_cuboid_verts_faces(
            torch.cat((bbox3D, box3D_dims), dim=1),
            box3D_pose,
        )[0]
        box3D_ = box3D_.squeeze().numpy()
        box3D_pose_ = self.box3D_get_pose(box3D_pseudo, box3D_)

        # Obtain bbox mesh (PyTorch3D)
        box3D_mesh = cube_utils.mesh_cuboid(
            torch.cat((bbox3D, box3D_dims), dim=1),
            box3D_pose_,
            color=[1, 0, 0],
        )
        box3D_mesh.verts_padded()[0] = box3D_mesh.verts_padded()[0] + 1e-6 # avoid division by zero 

        # Get 3D position (centroid)
        verts = box3D_mesh.verts_padded()[0]
        verts = verts.cpu().numpy()
        pos3D = np.mean(verts, axis=0)

        if return_box3D:
            return pos3D, box3D_mesh # for visualization
        else:
            return pos3D, None
        
    def visualize_scene_abstraction(
        self,
        image: Image.Image,
        box3D_meshes,
        box3D_obj_names,
        **apc_args,
    ):
        '''
        Visualize the box3D meshes on top of the image
        using CubeRCNN visualization function
        '''
        # Set camera intrinsic matrix (K)
        image_w, image_h = image.size
        focal_len_ndc = 4.0
        focal_len = focal_len_ndc * image_w
        px, py = image_w / 2, image_h / 2
        
        K = np.array([
            [focal_len, 0.0, px],
            [0.0, focal_len, py],
            [0.0, 0.0, 1.0]
        ])

        # Convert to numpy array
        image_npy = np.array(image)
        image_npy = cv2.cvtColor(image_npy, cv2.COLOR_RGB2BGR)

        try:
            # Draw the box3D meshes
            image_with_box3D, image_topdown, _ = cube_vis.draw_scene_view(
                image_npy,
                K,
                box3D_meshes,
                text=box3D_obj_names,
                scale=image_npy.shape[0],
                blend_weight=0.5,
                blend_weight_overlay=0.60,
            )
            image_concat = np.concatenate(
                (image_with_box3D, image_topdown), axis=1
            )
            image_concat = Image.fromarray(
                image_concat.astype(np.uint8)[..., ::-1]
            )
            # Save the image
            image_concat.save(os.path.join(apc_args['trace_save_dir'], "scene_abstraction.png"))
            print(f"* [INFO] Saved the scene abstraction image to {os.path.join(apc_args['trace_save_dir'], 'scene_abstraction.png')}")
        except:
            raise Exception("Error in saving the scene abstraction image")