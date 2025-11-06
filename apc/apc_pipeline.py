'''
Main pipeline for APC
'''
import re
import sys
import logging
import os
from typing import List, Dict, Tuple, Any, Optional
import importlib
from PIL import Image
from .utils import *
from .vlms import VLM_MODELS

# Add external module paths
sys.path.append("/ocean/projects/cis250208p/shared/models/vision_modules/omni3d")
sys.path.append("/ocean/projects/cis250208p/shared/models/vision_modules/orient_anything")
sys.path.append("/ocean/projects/cis250208p/shared/models/vision_modules/GroundingDINO")

# Import vision modules
from .vision_modules.vision_utils import cxcywh_to_xyxy, transform_src_to_tgt
from .vision_modules import DetectionModule, DepthModule, OrientationModule
from .renderer import RenderModule
from .prompts import PromptParser

# Initialize logger
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_vlm_model_class(vlm_type: str):
    """
    Dynamically import and return the VLM model class corresponding to `vlm_type`.
    Raises KeyError if `vlm_type` is unknown.
    """
    try:
        module_name, class_name = VLM_MODELS[vlm_type]
    except KeyError:
        raise NotImplementedError(f"Unknown VLM type: {vlm_type}")

    full_module_path = f"apc.vlms.{module_name}"
    module = importlib.import_module(full_module_path)
    
    return getattr(module, class_name)


# Define the APC pipeline
class APC:
    def __init__(
        self,
        config,
        device_vlm: str = 'cuda',
        device_vision: str = 'cuda',
    ):
        self.config = config
        self.device_vlm = device_vlm
        self.device_vision = device_vision

        # Set logger
        self.logger = logging.getLogger(__name__)
        
        # Import VLM model class
        VLMModel = get_vlm_model_class(config.vlm.type)

        # Initialize VLM model
        self.vlm_model = VLMModel(config.vlm, device_vlm)
        self.logger.info(f"[APC] Initialized VLM model: {config.vlm.type}")

        # Initialize vision modules
        self.detection_module = DetectionModule(config, device_vision)
        self.depth_module = DepthModule(config, device_vision)
        self.orientation_module = OrientationModule(config, device_vision)
        self.logger.info(f"[APC] Initialized vision modules")

        # Load instruction parser
        self.prompt_parser = PromptParser(config)
        self.logger.info(f"[APC] Initialized prompt parser")

        # Load renderer
        self.render_module = RenderModule(device=device_vision)
        
        # Set color dictionary (for visual prompts)
        self.color_dict = [
            ['blue', [0, 0, 1]],
            ['red', [1, 0, 0]],
            ['green', [0, 1, 0]],
            ['yellow', [1, 1, 0]],
            ['purple', [1, 0, 1]],
            ['orange', [1, 0.5, 0]],
            ['brown', [0.5, 0.25, 0]],
            ['gray', [0.5, 0.5, 0.5]],
        ]
    
    def get_objects_of_interest(
        self,
        image: Image.Image,
        prompt: str,
        num_tries: int = 2,
        conv_history: list = None,
        conv_save_path: str = None,
    ):
        '''
        From the image and prompt, extract the objects of interest

        Input:
            image: PIL.Image.Image
            prompt: str

        Output:
            objects_of_interest: list of objects to extract
        '''

        # Number of tries
        match_found = False
        response_objs = None

        for try_idx in range(num_tries):
            if try_idx == 0:
                prompt_objs = self.prompt_parser.get_prompt_by_type("get_objects_of_interest")
                prompt_objs = prompt_objs.format(question=prompt)
            else:
                prompt_objs = self.prompt_parser.get_prompt_by_type("get_objects_of_interest_aux")
                prompt_objs = prompt_objs.format(question=prompt, response=response_objs)

            # Add prompt to messages
            messages = add_message(
                [],
                role="user",
                text=prompt_objs,
                image=image.resize((400, 400)),     # resize image
            )

            # Query VLM
            response_objs = self.vlm_model.process_messages(messages)
            if conv_save_path is not None:
                store_conv(messages, response_objs, conv_save_path, "get_objects_of_interest")

            if response_objs.lower().startswith("[detect]"):
                response_objs = response_objs.lower().replace("[detect]", "").strip()

            # Parse the response
            pattern_objs = self.prompt_parser.get_prompt_by_type("pattern_get_objects_of_interest")

            # Regex pattern: Matches strings that start with '[' and end with ']'
            match_objs = re.findall(pattern_objs, response_objs)

            if len(match_objs) > 0:
                match_found = True
                break
        
        # If no match found, return error
        if not match_found:
            self.logger.error(f"[Scene Abstraction] No match found for objects of interest")
            return None
        
        # Parse the response
        match_objs = match_objs[-1]
        objs_of_interest = match_objs.strip().replace("[", "").replace("]", "").split(",")
        objs_of_interest = [obj.strip().lower().replace("'", "") for obj in objs_of_interest]

        # Update conv_history if needed
        if conv_history is not None:
            conv_history += [
                {'text': prompt_objs, 'image': image.resize((400, 400))},
                {'text': response_objs, 'image': None},
            ]
        
        return objs_of_interest, conv_history
    
    def get_reference_viewer(
        self,
        prompt: str,
        objects_of_interest: List[str],
        abstract_scene_dict_camera: Dict,
        conv_history: list = None,
        conv_save_path: str = None,
    ):
        '''
        From the prompt, extract the object of the reference viewer

        Input:
            prompt: str

        Output:
            ref_viewer: str
        '''
        prompt_ref_viewer = self.prompt_parser.get_prompt_by_type("get_reference_viewer")
        
        obj_str = ', '.join(objects_of_interest)
        prompt_ref_viewer = prompt_ref_viewer.format(question=prompt, obj_str=obj_str)

        # Add prompt to messages
        messages = add_message(
            [],
            role="user",
            text=prompt_ref_viewer,
            image=None,
        )
        # Query VLM
        response_ref_viewer = self.vlm_model.process_messages(messages)
        if conv_save_path is not None:
            store_conv(messages, response_ref_viewer, conv_save_path, "get_reference_viewer")

        # Parse the response
        pattern_ref_viewer = self.prompt_parser.get_prompt_by_type("pattern_reference_viewer")
        match_ref_viewer = re.findall(pattern_ref_viewer, response_ref_viewer)

        if len(match_ref_viewer) > 0:
            ref_viewer = match_ref_viewer[-1]
        else:
            ref_viewer = response_ref_viewer
        ref_viewer = ref_viewer.strip().lower().replace("'", "")

        if ref_viewer not in abstract_scene_dict_camera:
            self.logger.error(f"[Perspective Change] Reference viewer {ref_viewer} not found in abstract scene dict")
            ref_viewer = "camera" # set to camera by default

        self.logger.info(f"[Perspective Change] Reference viewer: {ref_viewer}")

        # Update conv_history if needed
        if conv_history is not None:
            conv_history += [
                {'text': prompt_ref_viewer, 'image': None},
                {'text': response_ref_viewer, 'image': None},
            ]

        return ref_viewer, conv_history
    
    def prompt_real_to_abstract(
        self,
        prompt: str,
        abstract_scene_dict_ref: Dict,
        ref_viewer: str,
    ):
        '''
        Convert the real (original) prompt to an abstract prompt
        '''
        prompt_abstract = prompt
        color_obj_map = ""

        for obj_name, obj_dict in abstract_scene_dict_ref.items():
            # Remove the reference viewer and the camera
            if obj_name == ref_viewer or obj_name == "camera":
                continue

            color = obj_dict['color'][0]
            # Replace the object name with the color cube
            prompt_abstract = re.sub(r' ' + re.escape(obj_name), f' {color} cube', prompt_abstract, flags=re.IGNORECASE)
            # TODO: add option to replace using VLM

            # Make the color-object map
            color_obj_map += f"- {color} cube -> {obj_name}\n"

        return prompt_abstract, color_obj_map
    
    # @apc_stage
    def do_scene_abstraction(
        self, 
        image: Image.Image, 
        objects_of_interest: List[str],
        conv_save_path: str = None,
        **apc_args,
    ):
        '''
        Run scene abstraction

        Input:
            image: PIL.Image.Image
            objects_of_interest: list of objects to extract

        Output:
            abstract_scene_dict: dict of abstract scene
        '''
        self.logger.info("[Scene Abstraction] Running scene abstraction...")

        abstract_scene_dict = {
            "camera": {
                "position": np.array([0, 0, 0]),
                "orientation": np.array([0, 0, 1]),
            }
        }
        box3D_meshes = []
        box3D_obj_names = []
        _, image_processed = self.detection_module.detection_process_image(image)
        image_h, image_w = image_processed.shape[1], image_processed.shape[2]

        for obj_name in objects_of_interest:
            if obj_name == "camera":
                continue

            self.logger.info(f"[Scene Abstraction] Running abstraction for {obj_name}")

            # ------------------------------------------------------------ #
            # Run detection
            # ------------------------------------------------------------ #
            boxes = self.detection_module.run_detection(image_processed, obj_name)
            if len(boxes) == 0:
                self.logger.error(f"[Scene Abstraction] No detection for {obj_name}")
                continue

            # Detection refinement using VLM
            selected_idx = 0
            if self.config.detection.use_vlm_refinement:
                box2D = self.detection_module.run_detection_refinement(
                    self.vlm_model,
                    image,
                    obj_name,
                    boxes_output=boxes,
                    conv_save_path=conv_save_path,
                    **apc_args,
                )
            else:
                box2D = boxes[selected_idx]

            # ------------------------------------------------------------ #
            # Run depth estimation
            # ------------------------------------------------------------ #
            depth_npy = self.depth_module.run_depth_estimation(image)

            # ------------------------------------------------------------ #
            # Run segmentation
            # ------------------------------------------------------------ #
            segment_mask = self.detection_module.run_segmentation(image, box2D)
            segment_mask = segment_mask[2, :, :]  # largest mask

            # ------------------------------------------------------------ #
            # Unproject to 3D
            # ------------------------------------------------------------ #
            position, box3D_mesh = self.depth_module.unproject_to_3D(
                image=image,
                depth=depth_npy,
                segment_mask=segment_mask,
                return_box3D=apc_args['visualize_trace'],
            )
            box3D_meshes.append(box3D_mesh)
            box3D_obj_names.append(obj_name)

            # ------------------------------------------------------------ #
            # Run orientation estimation
            # ------------------------------------------------------------ #
            orientation = self.orientation_module.run_orientation_estimation(
                image=image,
                category=obj_name,
                bbox=box2D,
                **apc_args,
            )

            # ------------------------------------------------------------ #
            # Save the results
            # ------------------------------------------------------------ #
            abstract_scene_dict[obj_name] = {
                'position': position,
                'orientation': orientation,
            }

        # Render the box3D meshes
        if apc_args['visualize_scene_abstraction'] and len(box3D_meshes) > 0:
            self.depth_module.visualize_scene_abstraction(
                image,
                box3D_meshes,
                box3D_obj_names,
                **apc_args,
            )

        return abstract_scene_dict

    # @apc_stage
    def do_perspective_change(
        self,
        abstract_scene_src: Dict,
        ref_viewer: str,
    ):
        '''
        Transform the abstract scene to align with the reference (target) perspective

        Input:
            abstract_scene_src: dict of abstract scene from the source perspective
            ref_viewer: target perspective

        Output:
            abstract_scene_tgt: dict of abstract scene from the target perspective
        '''
        self.logger.info(f"[Perspective Change] Running perspective change from {ref_viewer}...")

        if ref_viewer == "camera":
            return abstract_scene_src
        
        abstract_scene_tgt = {
            ref_viewer: {   # ref_viewer should be at the origin, facing towards +z
                'position': np.array([0, 0, 0]),
                'orientation': np.array([0, 0, 1]),
            }
        }
        # Set the target origin and z-axis (ref_viewer)
        origin_tgt = abstract_scene_src[ref_viewer]['position']
        z_axis_tgt = abstract_scene_src[ref_viewer]['orientation']

        for obj_name, obj_dict in abstract_scene_src.items():
            if obj_name == ref_viewer:
                continue
            # Get position and orientation in source perspective
            position_src = obj_dict['position']
            orientation_src = obj_dict['orientation']

            # Transform to target perspective
            position_tgt, orientation_tgt = transform_src_to_tgt(
                position_src, 
                orientation_src, 
                origin_tgt,
                z_axis_tgt,
            )
            # Store the results
            abstract_scene_tgt[obj_name] = {
                'position': position_tgt,
                'orientation': orientation_tgt,
            }

        return abstract_scene_tgt
    
    # @apc_stage
    def do_perspective_prompting_visual(
        self,
        ref_viewer: str,
        abstract_scene_dict_ref: Dict,
        prompt: str,
        conv_history: list = None,
        conv_save_path: str = None,
        **apc_args,
    ):
        '''
        Generate visual perspective prompts and query VLM
        '''
        self.logger.info("[Perspective Prompting] Generate visual perspective prompts...")

        # Convert convention (PyTorch3D -> TriMesh)
        abstract_scene_dict_ref_ = {}

        for obj_idx, (obj_name, obj_dict) in enumerate(abstract_scene_dict_ref.items()):
            position = obj_dict['position']
            orientation = obj_dict['orientation']
            position[1:] = -position[1:]
            orientation[1:] = -orientation[1:]

            if obj_name == ref_viewer:
                position = np.array([0, 0, 0])
                orientation = np.array([0, 0, 1])

            # Assign color (for face color)
            color = self.color_dict[obj_idx] # [name, color RGB]

            # Store results
            abstract_scene_dict_ref_[obj_name] = {
                'position': position,
                'orientation': orientation,
                'color': color,
            }
        
        # Make the visual prompt
        visual_prompt = self.render_module.render_visual_prompt(
            abstract_scene_dict_ref_,
            ref_viewer,
            **apc_args,
        )

        # print(visual_prompt.mode, visual_prompt.size)
        # print(np.min(visual_prompt), np.max(visual_prompt))
        # visual_prompt.show()

        # visual_prompt.save(f"outputs/visual_prompt_{ref_viewer}.png")

        visual_prompt = visual_prompt.resize((512, 512))
        self.logger.info("[Perspective Prompting] Rendered the visual prompt!")

        # Make the text prompt for visual prompting
        prompt_abstract, color_obj_map = self.prompt_real_to_abstract(
            prompt, 
            abstract_scene_dict_ref_,
            ref_viewer,
        )
        # Remove perspective description (convert to egocentric)
        prompt_convert_to_ego = self.prompt_parser.get_prompt_by_type("convert_to_ego")
        prompt_convert_to_ego = prompt_convert_to_ego.format(question=prompt_abstract, ref_viewer=ref_viewer)

        # Query VLM
        messages = add_message(
            [],
            role='user',
            text=prompt_convert_to_ego,
        )
        
        prompt_ego = self.vlm_model.process_messages(messages)
        if conv_save_path is not None:
            store_conv(messages, prompt_ego, conv_save_path, "convert_to_ego")
        
        self.logger.info(f"[Perspective Prompting] Converted to egocentric: {prompt_ego}")

        # Query VLM with the final visual prompt
        prompt_perspective_visual = self.prompt_parser.get_prompt_by_type("perspective_visual")
        prompt_perspective_visual = prompt_perspective_visual.format(question=prompt_ego)

        messages = add_message(
            [],
            role='user',
            text=prompt_perspective_visual,
            image=visual_prompt,
        )
        response_perspective_visual_abstract = self.vlm_model.process_messages(messages)
        if conv_save_path is not None:
            store_conv(messages, response_perspective_visual_abstract, conv_save_path, "perspective_visual")

        messages = add_message(
            messages,
            role='system',
            text=response_perspective_visual_abstract,
        )
        self.logger.info(f"[Perspective Prompting] Obtained abstract response: {response_perspective_visual_abstract}")

        # Translate the 'abstract' response to the real world
        prompt_abstract_to_real = self.prompt_parser.get_prompt_by_type("abstract_to_real")
        prompt_abstract_to_real = prompt_abstract_to_real.format(
            color_obj_map=color_obj_map,
            abstract_response=response_perspective_visual_abstract,
        )

        # Query VLM
        messages = add_message(
            messages,
            role='user',
            text=prompt_abstract_to_real,
        )
        response_perspective_visual = self.vlm_model.process_messages(messages)
        if conv_save_path is not None:
            store_conv(messages, response_perspective_visual, conv_save_path, "abstract_to_real")

        messages = add_message(messages, role='system', text=response_perspective_visual,)
        self.logger.info(f"[Perspective Prompting] Translate abstract -> real: {response_perspective_visual}")

        # Update conv_history if needed
        if conv_history is not None:
            conv_history += [
                {'text': prompt_perspective_visual, 'image': visual_prompt},
                {'text': response_perspective_visual_abstract, 'image': None},
                {'text': prompt_abstract_to_real, 'image': None},
                {'text': response_perspective_visual, 'image': None},
            ]

        # Choose options if needed
        options = apc_args['options']
        if options is not None:
            prompt_choose_options = self.prompt_parser.get_prompt_by_type("choose_options")
            prompt_choose_options = prompt_choose_options.format(
                question=prompt,
                response=response_perspective_visual,
                options=options,
            )
            # Add prompt to messages
            messages = add_message(
                messages,
                role="user",
                text=prompt_choose_options,
            )
            response_perspective_visual = self.vlm_model.process_messages(messages)
            if conv_save_path is not None:
                store_conv(messages, response_perspective_visual, conv_save_path, "choose_options")

            self.logger.info(f"[Perspective Prompting] Chose options: {response_perspective_visual}")

            # Update conv_history if needed
            if conv_history is not None:
                conv_history += [
                    {'text': prompt_choose_options, 'image': None},
                    {'text': response_perspective_visual, 'image': None},
                ]

        return response_perspective_visual, conv_history

    def run_apc(
        self, 
        image: Image.Image,
        prompt: str,
        # auxiliary arguments
        trace_save_dir: str = 'outputs',                # save directory for intermediate results
        perspective_prompt_type: str = "visual",        # perspective prompting type: visual, numerical
        visualize_trace: bool = True,                   # visualize all intermediate results
        visualize_scene_abstraction: bool = True,       # visualize the 3D scene abstraction
        render_whole_scene: bool = False,               # render the whole scene regardless of the reference viewer
        return_conv_history: bool = False,              # store all conversations and images for visualization
        options: str = None,                            # for evaluation, choose one of the given options
        logging: bool = True,
        conv_save_path: str = None,
    ):
        if logging:
            self.logger.disabled = False
            self.logger.info("[APC] Starting APC pipeline...")
        else:
            # print("Disabling logging...")
            self.logger.disabled = True
        # ------------------------------------------------------------ #
        # Define auxiliary arguments
        # ------------------------------------------------------------ #
        apc_args = {
            'trace_save_dir': trace_save_dir,
            'visualize_trace': visualize_trace,
            'visualize_scene_abstraction': visualize_scene_abstraction,
            'render_whole_scene': render_whole_scene,
            'options': options,
        }

        assert apc_args['trace_save_dir'] is not None, "trace_save_dir is required"
        
        # Initialize conversation storage if requested
        conv_history = [] if return_conv_history else None

        # ------------------------------------------------------------ #
        # [1] Scene Abstraction
        # ------------------------------------------------------------ #
        # Extract objects of interest
        objs_of_interest, conv_history = self.get_objects_of_interest(image, prompt, conv_history=conv_history, conv_save_path=conv_save_path)
        self.logger.info(f"[Scene Abstraction] Objects of interest: {objs_of_interest}")

        # Run 3D scene abstraction (from camera's perspective)
        abstract_scene_dict = {}
        abstract_scene_dict['camera'] = self.do_scene_abstraction(
            image, 
            objs_of_interest,
            conv_save_path=conv_save_path,
            **apc_args,
        )
        self.logger.info(f"[Scene Abstraction] Abstracted scene (camera's perspective): {abstract_scene_dict['camera']}")

        # ------------------------------------------------------------ #
        # [2] Perspective Change
        # ------------------------------------------------------------ #
        # Extract the reference (target) viewer
        ref_viewer, conv_history = self.get_reference_viewer(
            prompt, 
            objs_of_interest, 
            abstract_scene_dict['camera'],
            conv_history=conv_history,
            conv_save_path=conv_save_path,
        )

        # Transform the abstract scene to align with the target perspective
        abstract_scene_dict[ref_viewer] = self.do_perspective_change(
            abstract_scene_dict['camera'],
            ref_viewer=ref_viewer,
        )
        self.logger.info(f"[Perspective Change] Abstracted scene ({ref_viewer}'s perspective): {abstract_scene_dict[ref_viewer]}")

        # ------------------------------------------------------------ #
        # [3] Perspective Prompting
        # ------------------------------------------------------------ #
        if perspective_prompt_type == "visual":
            response_apc, conv_history = self.do_perspective_prompting_visual(
                ref_viewer,
                abstract_scene_dict[ref_viewer],
                prompt,
                conv_history=conv_history,
                conv_save_path=conv_save_path,
                **apc_args,
            )
        elif perspective_prompt_type == "numerical":
            # TODO: add numerical perspective prompting
            raise NotImplementedError("Numerical perspective prompting is not implemented yet!")
        else:
            raise ValueError(f"Invalid perspective prompt type: {perspective_prompt_type}")

        # Store final response if conversation tracking is enabled
        if conv_history is not None:
            # store the original question and response
            conv_history += [
                {'text': "[Question]\n\n" + prompt, 'image': image.resize((400, 400))},
                {'text': "[Response (APC)]\n\n" + response_apc, 'image': None},
            ]

        return response_apc, conv_history