'''
Simple script to run APC

Example:
python run_APC.py \
    --config apc/configs/qwenvl2_5_7b_instruct.yaml \
    --device_vlm cuda:0 \
    --device_vision cuda:0 \
    --image_path demo/sample_image_man.jpg \
    --prompt "If I stand at the person’s position facing where it is facing, is the table on the left or on the right of me?" \
    --save_dir outputs/demo/man_table \
    --visualize_trace \
    --return_conv_history
'''

import os
import sys
os.environ["DISPLAY"] =':1'
sys.path.append("apc/vision_modules")
sys.path.append(os.path.join(os.path.dirname(__file__), "GroundingDINO"))
import yaml
import argparse
from box import Box
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# import APC pipeline
from apc.apc_pipeline import APC
from apc.utils import visualize_conversation, create_image_with_text

def main(args):
    # load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config)

    # load APC pipeline
    apc = APC(config, device_vlm=args.device_vlm, device_vision=args.device_vision)
    print(f"* [INFO] Loaded APC pipeline!")

    # load input image
    image = Image.open(args.image_path).convert("RGB")

    # set prompt
    prompt = args.prompt
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # visualize question
    image_with_text = create_image_with_text(image, "[Q] " + prompt, fontsize=20)
    image_with_text.save(os.path.join(save_dir, "question.png"))

    # run APC
    # NOTE: for faster inference, set visualize_trace=False
    response, conv_history = apc.run_apc(
        image,
        prompt,
        trace_save_dir=save_dir,
        perspective_prompt_type="visual",
        visualize_trace=args.visualize_trace,
        visualize_scene_abstraction=True,
        return_conv_history=args.return_conv_history,
    )

    # visualize conversation
    if args.return_conv_history:
        conv_viz = visualize_conversation(
            conv_history,
            width=900,
            row_gap=0,
            font_size=13,
            image_max_width=180,
            output_path=os.path.join(save_dir, "conversation_viz.png")
        )
    
    print(f"* [INFO] Response: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="apc/configs/qwenvl2_5_7b_instruct.yaml")
    parser.add_argument("--device_vlm", type=str, default="cuda:0")
    parser.add_argument("--device_vision", type=str, default="cuda:0")
    parser.add_argument("--image_path", type=str, default="demo/sample_image_man.jpg")
    parser.add_argument("--prompt", type=str, default="If I stand at the person’s position facing where it is facing, is the table on the left or on the right of me?")
    parser.add_argument("--save_dir", type=str, default="outputs/demo/man_table")
    parser.add_argument("--visualize_trace", action="store_true", help="Visualize the trace")
    parser.add_argument("--return_conv_history", action="store_true", help="Return the conversation history")
    args = parser.parse_args()

    main(args)