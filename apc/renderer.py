'''
Renderer (for Visual Prompt)
'''
import os
import io
import sys
from typing import List, Dict, Tuple, Any
import cv2
import numpy as np
import math
import torch
from PIL import Image
import trimesh
# import trimesh.transformations as tf
from pytorch3d.renderer import look_at_view_transform
from apc.utils import perspective_grid, unit_cube, duplicate_verts

from trimesh.exchange import gltf
import PIL.Image

class RenderModule:
    def __init__(self, device="cuda"):
        self.device = device

    def draw_grid(
        self, 
        scene,
        grid_size: int = 10,
        step: int = 10,
        max_depth: int = 1000,
    ):
        '''
        Draw a tunnel-like grid to give a sense of depth
        '''
        # Make grid
        lines = perspective_grid(grid_size, step, max_depth)
        
        # Draw grid
        if len(lines) > 0:
            grid_path = trimesh.load_path(lines)
            grid_path.colors = [(100, 100, 100, 25)] * len(grid_path.entities)
            grid_path.merge_vertices()
            scene.add_geometry(grid_path)
    
    def make_cube(
        self,
        pos: np.ndarray,
        ori: np.ndarray,
        box_size: float = 2,
        color: List[int] = None,
    ):
        '''
        Make a cube mesh for visual prompt
        '''
        # Get vertices and faces
        verts, faces = unit_cube(
            -box_size, -box_size, -box_size,
            +box_size, +box_size, +box_size,
            is_torch=False
        )

        cube_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        eye = torch.tensor([0, 0, 0], dtype=torch.float32)
        at = torch.tensor(ori, dtype=torch.float32)
        up = torch.tensor(np.array([0, 1, 0]), dtype=torch.float32)

        R, T = look_at_view_transform(
            eye=eye[None, :],
            at=at[None, :],
            up=up[None, :],
            device=self.device,
        )

        # Set the transformation matrix
        M = torch.eye(4, device=R.device)
        M = M.unsqueeze(0).repeat(R.shape[0], 1, 1)
        M[:, :3, :3] = R
        M[:, :3, 3] = torch.tensor(
            [pos], device=self.device, dtype=torch.float32
        )
        M = M[0].cpu().numpy()

        # Rotate, translate the cube
        cube_mesh.apply_transform(M)

        trimesh.repair.fix_normals(cube_mesh)
        verts = cube_mesh.vertices
        faces = cube_mesh.faces

        # Set face colors
        cube_mesh = duplicate_verts(cube_mesh)
        cube_mesh.visual.face_colors = color

        return cube_mesh
    
    # def recenter_camera(self, scene_dict):
    #     objs = [v for k, v in scene_dict.items() if k != 'camera']
    #     obj_positions = np.array([o['position'] for o in objs])
        
    #     # Average position of all objects
    #     center = np.mean(obj_positions, axis=0)
        
    #     # Compute camera distance (so objects stay in frame)
    #     max_dist = np.linalg.norm(obj_positions - center, axis=1).max()
    #     vertical_spread = np.max(np.abs(obj_positions[:, 1] - center[1]))
    #     depth_spread = np.max(np.abs(obj_positions[:, 2] - center[2]))
    #     distance = max(max_dist, vertical_spread * 1.5, depth_spread * 1.5) * 2.5

    #     # Adjust camera position (move backward along its orientation)
    #     new_dir = center - scene_dict['camera']['position']
    #     scene_dict['camera']['orientation'] = new_dir / np.linalg.norm(new_dir)

    #     cam_dir = scene_dict['camera']['orientation'] / np.linalg.norm(scene_dict['camera']['orientation'])
    #     scene_dict['camera']['position'] = center - cam_dir * distance
        
    #     return scene_dict

    
    def add_wireframe(
        self,
        scene,              # Original trimesh scene
        visual_prompt,     # Rendered image (without wireframe)
        cube_meshes,        # List of cube meshes
        trace_save_dir: str,
    ):
        '''
        Draw a wireframe to each cube (for better visualization)
        '''

        os.makedirs(trace_save_dir, exist_ok=True)
        # Save the rendered image (temporary)
        temp_path = os.path.join(trace_save_dir, 'visual_prompt_temp.png')

        # Ensure we always have PNG bytes
        import io
        from PIL import Image
        import numpy as np

        if isinstance(visual_prompt, Image.Image):
            buf = io.BytesIO()
            visual_prompt.save(buf, format="PNG")
            visual_prompt_bytes = buf.getvalue()
        elif isinstance(visual_prompt, bytes):
            visual_prompt_bytes = visual_prompt
        else:
            raise TypeError(f"Unexpected visual_prompt type: {type(visual_prompt)}")

        with open(temp_path, "wb") as f:
            f.write(visual_prompt_bytes)
        
        # Read the image
        image = cv2.imread(temp_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Add wireframe to each cube
        for cube_mesh in cube_meshes:
            verts = cube_mesh.vertices

            # TODO: Add comments
            verts_per_face = verts.reshape(-1, 2, 3, 3)
            center_per_face = verts_per_face.mean(axis=-2).mean(axis=-2)
            
            # TODO: Add comments
            center_dist_per_face = np.linalg.norm(center_per_face, axis=-1)
            face_sorted_idx = np.argsort(center_dist_per_face, axis=0).tolist()
            front_face_idx = [
                (face_idx * 2, face_idx * 2 + 1)
                for face_idx in face_sorted_idx
            ][:3]

            for face_idx in front_face_idx:
                # Init new trimesh scene
                wire_scene = trimesh.Scene()
                wire_scene.camera_transform = scene.camera_transform
                wire_scene.camera.fov = scene.camera.fov
                
                # Set face colors
                face_colors = np.array([
                    [0, 0, 0, 255] if i in face_idx else [0, 0, 0, 0]
                    for i in range(12)
                ])
                cube_mesh.visual.face_colors = face_colors
                wire_scene.add_geometry(cube_mesh)
                
                # Save mask
                wire_png = wire_scene.save_image(resolution=(512, 512))
                wire_image = Image.open(io.BytesIO(wire_png)).convert('RGB')
                wire_image = np.array(wire_image)[:, :, ::-1]

                # Temporarily save the mask
                wire_mask_path = os.path.join(trace_save_dir, 'wire_mask.png')
                with open(wire_mask_path, 'wb') as f:
                    f.write(wire_png)
                    f.close()

                # Read the mask 
                wire_mask = cv2.imread(wire_mask_path, cv2.IMREAD_GRAYSCALE)
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(wire_mask, kernel, iterations=1)
                outline = cv2.subtract(dilated, wire_mask)

                outline_color = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)

                # Draw outline on the original image
                image[np.where((outline_color == [255, 255, 255]).all(axis=2))] = [0, 0, 0]

        # Get the final image with wireframes
        visual_prompt_with_wireframes = Image.fromarray(image)
        return visual_prompt_with_wireframes

    
    # def render_visual_prompt(
    #     self,
    #     scene_abs_dict: Dict[str, Any],         # scene abstraction dictionary
    #     ref_viewer: str,                           # reference object (whose perspective is used)
    #     **apc_args,
    # ):
    #     '''
    #     Render a visual prompt
    #     '''
        
    #     # Set fixed parameters
    #     cam_fov_deg = 120.0
    #     box_alpha = 0.50
    #     box_size = 2
    #     grid_size = 10
    #     min_depth = 1.5 * grid_size
    #     max_depth = 3 * grid_size
    #     trace_save_dir = apc_args['trace_save_dir']
    #     assert trace_save_dir is not None, "trace_save_dir is required"

    #     # Get 3D positions and orientations
    #     objects_to_render = [
    #         k for k in scene_abs_dict.keys() if k not in [ref_viewer, "camera"]
    #     ]
    #     positions = []
    #     orientations = []
    #     colors = []
    #     for obj in objects_to_render:
    #         positions.append(scene_abs_dict[obj]["position"])
    #         orientations.append(scene_abs_dict[obj]["orientation"])
    #         colors.append(scene_abs_dict[obj]["color"])
        
    #     # Init scene
    #     scene = trimesh.Scene()
    #     cam_matrix = np.identity(4)
    #     scene.camera_transform = cam_matrix
    #     scene.camera.fov = [cam_fov_deg, cam_fov_deg]

    #     # Draw a tunnel-like grid to give a sense of depth
    #     self.draw_grid(scene, grid_size=grid_size)

    #     # Set the object positions to be rendered
    #     if apc_args['render_whole_scene']:
    #         z_list = [pos[2] for pos in positions]
    #         max_z_obj_idx = np.argmax(z_list)
    #         max_z_obj_pos = positions[max_z_obj_idx]
    #         max_z = max_z_obj_pos[2]

    #         if max_z >= 0:
    #             dist_xy = np.linalg.norm(max_z_obj_pos[:2])
    #             z_trans = max_z + dist_xy * math.atan(math.radians(cam_fov_deg / 2.0))
    #             z_trans = z_trans + box_size    # How much to push camera to back

    #         # Translate camera to back
    #         for i, pos in enumerate(positions):
    #             x, y, z = pos
    #             z = z - z_trans
    #             positions[i] = [x, y, z]
    #         positions_ori_idx = list(range(len(positions)))
    #     else:
    #         # Filter out objects with z >= 0 (behind the reference viewer)
    #         positions_ori_idx = [i for i, pos in enumerate(positions) if pos[2] < 0]    # Keep original indices
    #         positions = [pos for pos in positions if pos[2] < 0]

    #     # Normalize positions
    #     if len(positions) > 0:
    #         # Get max x, y
    #         max_x = max([abs(pos[0]) for pos in positions])
    #         max_y = max([abs(pos[1]) for pos in positions])
    #         max_xy = max(max_x, max_y)

    #         # Compute scale factor for xy plane (place the outermost object at grid_size)
    #         xy_scale = grid_size / max_xy

    #         # Get min, max z (NOTE: z is negative!)
    #         z_list = [pos[2] for pos in positions]
    #         min_z = min(z_list)
    #         max_z = max(z_list)
    #         z_offset = max_z

    #         # Normalize
    #         if len(positions) == 1:
    #             # If there's only one object, set z_scale to 1
    #             z_scale = 1
    #         else:
    #             z_scale = (max_depth - min_depth) / (abs(min_z - max_z) + 1e-6)
    #             z_scale = z_scale if z_scale > 0 else 1
            
    #         # Scale each position
    #         for i, pos in enumerate(positions):
    #             positions[i][0] = pos[0] * xy_scale
    #             positions[i][1] = pos[1] * xy_scale
    #             positions[i][2] = (pos[2] - z_offset) * z_scale - min_depth
            
    #         print("* [INFO] Scaled all positions!")
        
    #     # Set abstract cubes for each object & render
    #     cube_meshes = []

    #     for idx, (pos, ori) in enumerate(zip(positions, orientations)):
    #         # Get color
    #         # cube_color = CUBE_COLOR_MAP[colors[idx]]
    #         cube_color = colors[idx][1]
            
    #         # Make a cube mesh
    #         face_color = [
    #             cube_color + [box_alpha]
    #             for _ in range(12)
    #         ]
    #         cube_mesh = self.make_cube(
    #             pos, ori, 
    #             box_size=box_size, 
    #             color=face_color,
    #         )
    #         cube_meshes.append(cube_mesh)

    #         # Add the cube to the scene
    #         scene.add_geometry(cube_mesh)
        
    #     # Render
    #     png = scene.save_image(resolution=(512, 512), visible=False)
    #     if isinstance(png, bytes):
    #         visual_prompt = PIL.Image.open(trimesh.util.wrap_as_stream(png))
    #     else:
    #         buf = io.BytesIO()
    #         png.save(buf, format="PNG")
    #         visual_prompt = PIL.Image.open(io.BytesIO(buf.getvalue()))
    #     # visual_prompt = scene.save_image()

    #     # Add wireframe to each cube
    #     visual_prompt_with_wireframes = self.add_wireframe(
    #         scene,
    #         visual_prompt,
    #         cube_meshes,
    #         trace_save_dir,
    #     )

    #     # Save the visual prompt
    #     if apc_args['visualize_trace']:
    #         visual_prompt_with_wireframes.save(
    #             os.path.join(trace_save_dir, 'visual_prompt.png')
    #         )

    #     return visual_prompt_with_wireframes
    

    def make_arrow(self, origin, direction, length=0.5, color=(1.0, 0.0, 0.0), shift = 2):
        """
        Create a thin red arrow (shaft + head) pointing along 'direction' from 'origin'.
        """
        import trimesh
        import numpy as np

        # Normalize direction
        direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return trimesh.Trimesh()  # skip invalid arrows
        direction /= norm

        # Dimensions (thinner)
        shaft_length = length * 0.85
        shaft_radius = length * 0.02     # thinner
        cone_length = length * 0.15
        cone_radius = shaft_radius * 4.0

        # Cylinder (shaft)
        shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_length, sections=24)
        shaft.apply_translation([0, 0, shaft_length / 2.0])

        # Cone (head)
        head = trimesh.creation.cone(radius=cone_radius, height=cone_length, sections=24)
        head.apply_translation([0, 0, shaft_length + cone_length / 2.0])

        # Combine
        arrow = trimesh.util.concatenate([shaft, head])

        # Align Z-axis with direction
        z_axis = np.array([0, 0, 1])
        rotation, _ = trimesh.geometry.align_vectors(z_axis, direction, return_angle=True)
        arrow.apply_transform(rotation)
        arrow.apply_translation(origin + direction * shift)

        # Color (solid red by default)
        rgba = np.array([color[0]*255, color[1]*255, color[2]*255, 255])
        arrow.visual.face_colors = rgba
        return arrow



    def render_visual_prompt(
        self,
        scene_abs_dict: Dict[str, Any],  # scene abstraction dictionary
        ref_viewer: str,                 # reference object (whose perspective is used)
        **apc_args,
    ):
        """
        Render a visual prompt safely (works even in headless environments).
        """
        import io
        import math
        import numpy as np
        import PIL
        from PIL import Image
        import trimesh
        from pyglet.canvas.xlib import NoSuchDisplayException

        # Fixed parameters
        cam_fov_deg = 120.0
        box_alpha = 0.50
        box_size = 2
        grid_size = 10
        min_depth = 1.5 * grid_size
        max_depth = 3 * grid_size
        trace_save_dir = apc_args['trace_save_dir']
        assert trace_save_dir is not None, "trace_save_dir is required"

        # Collect object data
        objects_to_render = [k for k in scene_abs_dict.keys() if k not in [ref_viewer, "camera"]]
        positions, orientations, colors = [], [], []
        for obj in objects_to_render:
            positions.append(scene_abs_dict[obj]["position"])
            orientations.append(scene_abs_dict[obj]["orientation"])
            colors.append(scene_abs_dict[obj]["color"])

        # Init scene
        scene = trimesh.Scene()
        cam_matrix = np.identity(4)
        scene.camera_transform = cam_matrix
        scene.camera.fov = [cam_fov_deg, cam_fov_deg]
        self.draw_grid(scene, grid_size=grid_size)

        # Determine which objects to render
        if apc_args["render_whole_scene"]:
            z_list = [pos[2] for pos in positions]
            max_z_obj_idx = np.argmax(z_list)
            max_z_obj_pos = positions[max_z_obj_idx]
            max_z = max_z_obj_pos[2]
            
            # Always define z_trans
            z_trans = 0.0

            # Compute translation adaptively
            dist_xy = np.linalg.norm(max_z_obj_pos[:2])
            z_trans = abs(max_z) + dist_xy * math.tan(math.radians(cam_fov_deg / 2.0))
            z_trans += box_size * 2.0  # small buffer so all objects stay visible


            for i, pos in enumerate(positions):
                x, y, z = pos
                z = z - z_trans
                positions[i] = [x, y, z]
            positions_ori_idx = list(range(len(positions)))
        else:
            positions_ori_idx = [i for i, pos in enumerate(positions) if pos[2] < 0]
            positions = [pos for pos in positions if pos[2] < 0]

        # Normalize positions
        if len(positions) > 0:
            max_x = max([abs(pos[0]) for pos in positions])
            max_y = max([abs(pos[1]) for pos in positions])
            max_xy = max(max_x, max_y)
            xy_scale = grid_size / max_xy if max_xy > 0 else 1.0
            z_list = [pos[2] for pos in positions]
            min_z = min(z_list)
            max_z = max(z_list)
            z_offset = max_z
            z_scale = (max_depth - min_depth) / (abs(min_z - max_z) + 1e-6)
            z_scale = z_scale if z_scale > 0 else 1.0

            for i, pos in enumerate(positions):
                positions[i][0] = pos[0] * xy_scale
                positions[i][1] = pos[1] * xy_scale
                positions[i][2] = (pos[2] - z_offset) * z_scale - min_depth

            # print("* [INFO] Scaled all positions!")

        # Add cubes to scene
        cube_meshes = []
        for idx, (pos, ori) in enumerate(zip(positions, orientations)):
            cube_color = colors[idx]
            face_color = [cube_color + [box_alpha] for _ in range(12)]

            # --- Fix for vertical degeneracy ---
            if abs(ori[1]) > 0.9999:
                ori[2] = 1e-10
            # -----------------------------------

            cube_mesh = self.make_cube(pos, ori, box_size=box_size, color=face_color)
            cube_meshes.append(cube_mesh)
            scene.add_geometry(cube_mesh)
            
            arrow = self.make_arrow(pos, ori, length=10, shift = box_size)
            scene.add_geometry(arrow)

            # arrow = self.make_arrow(pos, ori, length=box_size * 0.8, color=(0.2, 0.8, 0.2))


        # --- Safe rendering ---
        png = scene.save_image(resolution=(512, 512))

        # Convert to PIL.Image (compatible with downstream code)
        if isinstance(png, bytes):
            visual_prompt = PIL.Image.open(trimesh.util.wrap_as_stream(png))
        else:
            buf = io.BytesIO()
            png.save(buf, format="PNG")
            visual_prompt = PIL.Image.open(io.BytesIO(buf.getvalue()))

        # Add wireframe overlay
        visual_prompt_with_wireframes = self.add_wireframe(
            scene, visual_prompt, cube_meshes, trace_save_dir
        )

        os.makedirs(trace_save_dir, exist_ok=True)
        # Save the image if requested
        if apc_args.get("visualize_trace", False):
            visual_prompt_with_wireframes.save(
                os.path.join(trace_save_dir, "visual_prompt.png")
            )

        return visual_prompt_with_wireframes
