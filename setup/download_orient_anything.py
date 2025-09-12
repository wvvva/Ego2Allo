'''
Download Orient-Anything checkpoint
https://github.com/SpatialVision/Orient-Anything
'''
import argparse
from huggingface_hub import hf_hub_download

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--cache_dir", type=str, default="apc/vision_modules/src/checkpoints")
args = parser.parse_args()

# Download Orient-Anything checkpoint
orientation_ckpt_path = hf_hub_download(
    repo_id="Viglong/Orient-Anything", 
    filename="croplargeEX2/dino_weight.pt", 
    repo_type="model", 
    cache_dir=args.cache_dir, 
    resume_download=True
)