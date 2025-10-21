<div align="center">
  <img src="assets/apc_logo.png" alt="APC Logo" width="100"/>
</div>

# Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation

<div align="left">
  <img src="assets/teaser_apc.jpg" alt="Teaser" width="95%"/>
</div><br>

[![arXiv](https://img.shields.io/badge/arXiv-2504.17207-B31B1B?logo=arXiv&logoColor=white)](https://arxiv.org/abs/2504.17207)
[![arXiv](https://img.shields.io/badge/Project-APC-green)](https://arxiv.org/abs/2504.17207)

[Phillip Y. Lee](https://phillipinseoul.github.io/)<sup>1</sup>, 
[Jihyeon Je](https://jihyeonje.com/)<sup>2</sup>, 
[Chanho Park](https://charlieppark.kr/)<sup>1</sup>, 
[Mikaela Angelina Uy](https://mikacuy.github.io/)<sup>3</sup>, 
[Leonidas Guibas](https://geometry.stanford.edu/?member=guibas)<sup>2</sup>,
[Minhyuk Sung](https://mhsung.github.io/)<sup>1</sup>

<sup>1</sup>KAIST, <sup>2</sup>Stanford University, <sup>3</sup>NVIDIA  

ICCV 2025

## 💬 Introduction

This repository contains the official implementation of **Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation (ICCV 2025)**.

We propose a framework that enables Vision-Language Models to perform spatial reasoning in arbitrary perspectives.

## 🔧 Get Started
We have tested on Python 3.10, CUDA 12.4, and PyTorch 2.4.1. Please follow the below scripts for setting up the environment.

```bash
# create conda env
conda create -n apc_vlm python=3.10 -y
conda activate apc_vlm
# install torch
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -c pytorch -y 
# install vision module dependencies & download checkpoints
bash setup/setup_vision_modules.sh
# install other dependencies
pip install -r setup/requirements.txt
```

## ✏️ How to Run
We provide an easy-to-use notebook, [run_APC.ipynb](./run_APC.ipynb), for quickly testing our APC framework.

Alternatively, you can run inference directly with `run_APC.py`. For example:

```bash
python run_APC.py \
    --config apc/configs/qwenvl2_5_7b_instruct.yaml \
    --device_vlm cuda:0 \
    --device_vision cuda:0 \
    --image_path demo/sample_image_man.jpg \
    --prompt "If I stand at the person’s position facing where it is facing, is the table on the left or on the right of me?" \
    --save_dir outputs/demo/man_table \
    --visualize_trace \
    --return_conv_history
```

An example of the saved conversation history from APC is as follows:
<div align="left">
  <img src="assets/inference_example.png" alt="Inference example" width="90%"/>
</div><br>

## 🙌 Acknowledgements
Our implementation is built upon amazing projects including [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), [Depth Pro](https://github.com/apple/ml-depth-pro), [SAM](https://github.com/facebookresearch/segment-anything), [Orient Anything](https://github.com/facebookresearch/segment-anything), [Omni3D](https://github.com/facebookresearch/omni3d), [Ovmono3D](https://github.com/UVA-Computer-Vision-Lab/ovmono3d), and [trimesh](https://github.com/mikedh/trimesh). We greatly thank all authors and contributors for open-sourcing their code and model checkpoints.

## 🔖 Citation
If you find our work useful, please consider citing:
```
@inproceedings{lee2025perspective,
  title={Perspective-aware reasoning in vision-language models via mental imagery simulation},
  author={Lee, Phillip Y and Je, Jihyeon and Park, Chanho and Uy, Mikaela Angelina and Guibas, Leonidas and Sung, Minhyuk},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```


### new readme

1. Load anaconda3, gcc/13.3.1

Add .toml file
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
```