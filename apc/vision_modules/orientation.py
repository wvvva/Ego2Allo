'''
Scene Abstraction:
- Extract each object's frontal orientation
- Model: OrientAnything (https://github.com/SpatialVision/Orient-Anything)
'''
from typing import List
import os
import sys
import math
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from transformers import AutoImageProcessor
from huggingface_hub import hf_hub_download

# Import from Orient-Anything
from orient_anything.render import render, Model
from orient_anything.vision_tower import DINOv2_MLP

class OrientationModule:
    def __init__(self, config, device="cuda"):
        self.device = device
        self.config = config

        # Load Orient-Anything
        self.orientation_model = DINOv2_MLP(
            dino_mode='large',
            in_dim=1024,
            out_dim=360+180+180+2,
            evaluate=True,
            mask_dino=False,
            frozen_back=False
        ).eval()
        
        # TODO: add checkpoint path to config
        orientation_ckpt_path = hf_hub_download(
            repo_id="Viglong/Orient-Anything", 
            filename="croplargeEX2/dino_weight.pt", 
            repo_type="model", 
            cache_dir=config.orientation.ckpt_path, 
            resume_download=True
        )
        self.orientation_model.load_state_dict(
            torch.load(orientation_ckpt_path, map_location='cpu')
        )
        self.orientation_model.to(self.device)

        self.orientation_processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-large", 
            cache_dir=config.orientation.ckpt_path,
        )
        print(f"* [INFO] Loaded Orient-Anything!")

    def orientation_to_direction(self, azimuth_deg, polar_deg):
        '''
        Helper function: Convert orientation to direction vector
        '''
        direction_vector = np.array([
            -math.sin(np.radians(azimuth_deg)),
            math.sin(np.radians(polar_deg)) * math.cos(np.radians(azimuth_deg)),
            -math.cos(np.radians(polar_deg)) * math.cos(np.radians(azimuth_deg))
        ])
        return direction_vector
    
    def render_3D_axis(
        self,
        phi,
        theta,
        gamma,
        radius: int = 240,
        height: int = 512,
        width: int = 512,
        axis_obj_path: str = "/ocean/projects/cis250208p/shared/models/vision_modules/orient_anything/assets/axis.obj",
        axis_texture_path: str = "/ocean/projects/cis250208p/shared/models/vision_modules/orient_anything/assets/axis.png",
        **apc_args,
    ):
        '''
        Render the 3D axis
        (Modified from https://github.com/SpatialVision/Orient-Anything/blob/main/utils.py)
        '''
        # 
        camera_location = [
            -1*radius * math.cos(phi),
            -1*radius * math.tan(theta),
            radius * math.sin(phi)
        ]
        axis_model = Model(axis_obj_path, texture_filename=axis_texture_path)

        rendered_image = render(
            axis_model,
            height=height,
            width=width,
            filename=os.path.join(apc_args["trace_save_dir"], "orientation.png"),   # temporary file
            cam_loc=camera_location,
        )
        rendered_image = rendered_image.rotate(gamma)
        return rendered_image
    
    def get_3angle(self, image):
        '''
        Get the 3D orientation of the object
        (Modified from https://github.com/SpatialVision/Orient-Anything/blob/main/inference.py)
        '''
        # Preprocess the image
        image_inputs = self.orientation_processor(images=image)
        image_inputs['pixel_values'] = torch.from_numpy(np.array(image_inputs['pixel_values'])).to(self.device)
        
        # Run the model
        with torch.no_grad():
            dino_pred = self.orientation_model(image_inputs)

        # Get the angles
        gaus_ax_pred = torch.argmax(dino_pred[:, 0:360], dim=-1)
        gaus_pl_pred = torch.argmax(dino_pred[:, 360:360+180], dim=-1)
        gaus_ro_pred = torch.argmax(dino_pred[:, 360+180:360+180+180], dim=-1)
        
        angles = torch.zeros(4)
        angles[0] = gaus_ax_pred
        angles[1] = gaus_pl_pred - 90
        angles[2] = gaus_ro_pred - 90
        
        return angles
    
    def overlay_images_with_scaling(
        self,
        center_image: Image.Image,
        background_image,
        target_size=(512, 512),
    ):
        '''
        Draw the axis on top of the image
        (From https://github.com/SpatialVision/Orient-Anything/blob/main/utils.py)
        '''
        if center_image.mode != "RGBA":
            center_image = center_image.convert("RGBA")
        if background_image.mode != "RGBA":
            background_image = background_image.convert("RGBA")
        
        # Resize image
        center_image = center_image.resize(target_size)
        bg_width, bg_height = background_image.size
        scale = target_size[0] / max(bg_width, bg_height)
        new_width = int(bg_width * scale)
        new_height = int(bg_height * scale)
        resized_background = background_image.resize((new_width, new_height))
        
        # Calculate padding
        pad_width = target_size[0] - new_width
        pad_height = target_size[0] - new_height
        left = pad_width // 2
        right = pad_width - left
        top = pad_height // 2
        bottom = pad_height - top
        resized_background = ImageOps.expand(resized_background, border=(left, top, right, bottom), fill=(255,255,255,255))
        
        # Overlay the center image on the background
        result = resized_background.copy()
        result.paste(center_image, (0, 0), mask=center_image)
        
        return result

    def run_orientation_estimation(
        self,
        image: Image.Image,
        category: str = None,              
        bbox: List[int] = None,        # optional
        **apc_args,
    ):
        '''
        Run orientation estimation of a single image
        Returns the direction to which the object is facing
        '''
        # Crop the object region
        if bbox is not None:
            image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        # Resize the image - keep the ratio, but make the longer side 512
        if image.size[0] > image.size[1]:
            new_size = (512, int(512 * image.size[1] / image.size[0]))
        else:
            new_size = (int(512 * image.size[0] / image.size[1]), 512)
        image = image.resize(new_size)
        
        # Run Orient-Anything
        pred_angles = self.get_3angle(image)

        # Obtain each angle
        azimuth_deg = float(pred_angles[0])
        polar_deg = float(pred_angles[1])
        rotation = float(pred_angles[2])

        # Convert to direction vector
        direction_vector = self.orientation_to_direction(azimuth_deg, polar_deg)

        # Save the image with the direction vector
        if apc_args["trace_save_dir"] is not None and apc_args["visualize_trace"]:
            azimuth_rad = np.radians(azimuth_deg)
            polar_rad = np.radians(polar_deg)
            render_axis = self.render_3D_axis(
                azimuth_rad,
                polar_rad,
                rotation,
                **apc_args
            )
            # Draw axis on top of the (cropped) image
            image_with_axis = self.overlay_images_with_scaling(render_axis, image)
            
            if category is not None:
                image_with_axis.save(os.path.join(apc_args["trace_save_dir"], f"orientation_{category}.png"))
            else:
                # Make bbox -> string
                bbox_str = f"bbox_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                image_with_axis.save(os.path.join(apc_args["trace_save_dir"], f"orientation_{bbox_str}.png"))

        return direction_vector