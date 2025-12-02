import os
import json
from tqdm import tqdm
import apc.renderer

import numpy as np
from IPython.display import display
# RenderModule (local renderer)
from apc.renderer import RenderModule

os.environ['PYGLET_HEADLESS'] = '1'

import OpenGL
print("OpenGL version:", OpenGL.__version__)
from OpenGL import GL
print("GL import ok")
import pyglet
print("pyglet version info:", pyglet.version)  # Use pyglet.version instead of __version__
# Force software rendering
pyglet.options['shadow_window'] = False
pyglet.options['debug_gl'] = False

color_dict = {
    'blue':   ['blue', [0, 0, 1]],
    'green':  ['green', [0, 1, 0]],
    'yellow': ['yellow', [1, 1, 0]],
    'purple': ['purple', [1, 0, 1]],
    'orange': ['orange', [1, 0.5, 0]],
    'brown':  ['brown', [0.5, 0.25, 0]],
    'gray':   ['gray', [0.5, 0.5, 0.5]],
}

def to_trimesh_scene(scene: dict, ref_viewer: str = 'camera') -> dict:
    out = {}
    color_idx = 0  # Separate counter for non-camera objects
    
    for name, entry in scene.items():
        # print(name)
        if name == ref_viewer or not name.startswith("object"):
            continue  # Skip camera object

        pos = np.array(entry['position']).copy()
        ori = np.array(entry['orientation']).copy()
        
        # Normalize orientation before conversion
        # ori = ori / np.linalg.norm(ori)

        color = entry['color']
        
        # # Copy colors with proper indexing
        # print(color)
        color = color_dict[str(color)]
        # color_idx += 1
        
        out[name] = {
            'position': pos,
            'orientation': ori,
            'color': color,
            'original_pos': np.array(entry['position']),
            'original_ori': ori
        }
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="orien_1")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1000)
    args = parser.parse_args()

    start_index = args.start_index
    end_index = args.end_index
    file_name = args.file_name
    file = f"/jet/home/ydinga/Ego2Allo/{file_name}.json"
    
    with open(file, "r") as f:
        scene_list = json.load(f)

    for i, scene_data in tqdm(enumerate(scene_list[start_index:end_index])):
        
        index = i + start_index
        
        # Convert lists to numpy arrays
        abstract_scene_dict = {
            key: {
                k: np.array(v)
                for k, v in value.items()
            }
            for key, value in scene_data.items() if key == 'camera' or key.startswith("object")
        }

        # print(f"\n=== Rendering Scene {i} ===")
        # print(abstract_scene_dict)

        scene_for_renderer = to_trimesh_scene(abstract_scene_dict, ref_viewer='camera')

        # print(scene_for_renderer)

        # Example: render the scene
        renderer = RenderModule(device='cuda')
        img = renderer.render_visual_prompt(
            scene_for_renderer,
            ref_viewer='camera',
            # trace_save_dir=f"/ocean/projects/cis250208p/shared/datasets/synthetic/{file_name}/scene_{index}",
            render_whole_scene=True,
            visualize_trace=True,
        )

        
        scene_list[index]["img"] =  f"/ocean/projects/cis250208p/shared/datasets/synthetic/{file_name}/scene_{index}/visual_prompt.png"

    
    with open(file, 'w') as f:
        json.dump(scene_list, f, indent = 2)

        # img.show()