'''
Scene Abstraction:
- Detect and segment each object of interest
- Models:
    - Grounding DINO (https://github.com/IDEA-Research/GroundingDINO)
    - SAM (https://github.com/facebookresearch/segment-anything)
'''
import os
import io
import sys
import cv2
import matplotlib.pyplot as plt
sys.path.append("..")
import yaml
from typing import List
import numpy as np
from PIL import Image
import torch
from box import Box
import open3d as o3d
from .vision_utils import *
from ..utils import add_message

# Import modules
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as GT
import groundingdino.config.GroundingDINO_SwinT_OGC as groundingdino_config
from segment_anything import SamPredictor, sam_model_registry


class DetectionModule:
    def __init__(
            self, 
            config, 
            device="cuda",
        ):
        self.device = device
        self.config = config
        # Detection parameters
        self.box_threshold = config.detection.box_threshold
        self.text_threshold = config.detection.text_threshold
        self.num_candidates = config.detection.num_candidates

        # Load Grounding DINO
        config_path = groundingdino_config.__file__

        # Load model
        self.detection_model = load_model(config_path, config.detection.ckpt_path)
        self.detection_model.to(self.device)
        print("* [INFO] Loaded GroundingDINO!")

        # Load SAM
        self.segmentation_model = sam_model_registry["default"](
            checkpoint=config.segmentation.ckpt_path).to(device=self.device)
        self.segmentation_predictor = SamPredictor(self.segmentation_model)
        print("* [INFO] Loaded SAM!")

    def detection_process_image(
        self, 
        image: Image.Image,
    ):
        '''
        Image processing function for GroundingDINO
        (modified from https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/util/inference.py)
        '''
        transform = GT.Compose(
            [
                # GT.RandomResize([800], max_size=1333),
                GT.ToTensor(),
                GT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_npy = np.asarray(image)
        image_transformed, _ = transform(image, None)

        return image_npy, image_transformed
    
    def run_detection(
        self, 
        image: torch.Tensor,            # after detection_process_image()
        category: str,                # category to detect
    ):
        '''
        Run detection model (GroundingDINO)
        '''
        print(f"* [INFO] Running detection for {category}...")

        boxes, scores, _ = predict(
            model=self.detection_model,
            image=image,
            caption=category,
            box_threshold=self.config.detection.box_threshold,
            text_threshold=self.config.detection.text_threshold,
        )
        # Get detections with highest scores
        boxes_sorted = sort_by_scores(scores, [boxes])[0]
        boxes_output = boxes_sorted[:self.config.detection.num_candidates]

        return boxes_output
    
    def run_detection_refinement(
        self,
        vlm_model,          # VLM model (defined in apc_pipeline.py)
        image: Image.Image,
        category: str,
        boxes_output: List[List[int]],
        **apc_args,
    ):
        '''
        Run detection refinement with VLM
        (Choose the best detection from top-N detections)
        '''
        print(f"* [INFO] Running detection refinement for {category}...")

        W, H = image.size
        
        # Convert relative boxes to absolute boxes
        boxesAbs = [
            [
                int(cxcywh_to_xyxy(box)[0] * W),  # xmin
                int(cxcywh_to_xyxy(box)[1] * H),  # ymin
                int(cxcywh_to_xyxy(box)[2] * W),  # xmax
                int(cxcywh_to_xyxy(box)[3] * H),  # ymax
            ]
            for box in boxes_output
        ]

        # Get list of cropped images
        cropped_images = [
            image.crop((box[0], box[1], box[2], box[3]))
            for box in boxesAbs
        ]
        cropped_images = [
            np.asarray(crop) for crop in cropped_images
        ]

        # Make a grid of cropped images (for input to VLM)
        fig = visualize_crops(cropped_images)
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        candidate_image = Image.open(buf).convert("RGB")
        plt.close()
        cW, cH = candidate_image.size
        candidate_image = candidate_image.resize((cW // 3, cH // 3))

        # Save for visualization
        if apc_args['trace_save_dir'] is not None:
            candidate_image.save(
                os.path.join(apc_args['trace_save_dir'], f"detection_candidates_{category}.png")
            )

        # Query VLM for the best detection
        prompt_refinement = f"""
        Select the image that best fits the description: '{category}'.
        
        Please return its index.
        """

        # Make conversation
        messages = add_message(
            [], role="user", 
            text=prompt_refinement, image=candidate_image
        )

        # Query VLM
        response = vlm_model.process_messages(messages, max_new_tokens=512)
        print(f"* [INFO] Response for VLM detection refinement: {response}")
        
        # Get the selected index
        selected_idx = 0
        for i in range(min(self.num_candidates, len(boxesAbs))):
            if str(i) in response:
                selected_idx = i
                break

        # Return the selected detection for category
        return boxesAbs[selected_idx]

    def run_segmentation(
        self, 
        image: Image.Image,
        box2d: List[int]       # detected 2D bbox
    ):
        '''
        Run segmentation model (SAM)
        '''
        # Convert to BGR numpy array
        image_npy = np.array(image)
        image_bgr = cv2.cvtColor(image_npy, cv2.COLOR_RGB2BGR)

        self.segmentation_predictor.set_image(image_bgr, image_format="BGR")
        box2d = np.array(box2d)
        masks, _, _ = self.segmentation_predictor.predict(box=box2d)
        
        return masks