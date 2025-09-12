VISION_MODULES_DIR="apc/vision_modules/src"
CHECKPOINT_DIR="$VISION_MODULES_DIR/checkpoints"

# ================================
# install vision module dependencies
# ================================
mkdir -p $VISION_MODULES_DIR
cd $VISION_MODULES_DIR

# Grounding DINO
git clone git@github.com:IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
echo "* [INFO] Installed Grounding DINO"

# Depth Pro
git clone git@github.com:apple/ml-depth-pro.git
cd ml-depth-pro
pip install -e .
cd ..
echo "* [INFO] Installed Depth Pro"

# Orient-Anything
git clone git@github.com:SpatialVision/Orient-Anything.git
mv Orient-Anything orient_anything
echo "* [INFO] Cloned Orient-Anything"

# Omni3D
git clone git@github.com:facebookresearch/omni3d.git
echo "* [INFO] Cloned Omni3D"

# return to root directory
cd ../../../

# ================================
# download vision modules checkpoints
# ================================

# make sure the checkpoint folder exists
mkdir -p $CHECKPOINT_DIR

# SAM (https://github.com/facebookresearch/segment-anything#model-checkpoints)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P $CHECKPOINT_DIR
echo "* [INFO] Downloaded SAM checkpoint"

# Grounding DINO (https://github.com/IDEA-Research/GroundingDINO)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P $CHECKPOINT_DIR
echo "* [INFO] Downloaded Grounding DINO checkpoint"

# Orient-Anything (https://github.com/SpatialVision/Orient-Anything)
python setup/download_orient_anything.py --cache_dir $CHECKPOINT_DIR
echo "* [INFO] Downloaded Orient-Anything checkpoint"

# Depth Pro (https://github.com/apple/ml-depth-pro/blob/main/get_pretrained_models.sh)
wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P $CHECKPOINT_DIR
echo "* [INFO] Downloaded Depth Pro checkpoint"

# ================================