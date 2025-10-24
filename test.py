from datasets import load_dataset, load_from_disk
import torch, gc
gc.collect()
torch.cuda.empty_cache()
import os
import sys
os.environ["DISPLAY"] =':1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # set GPU device
sys.path.append("apc/vision_modules")
import yaml
import re
import requests
from box import Box
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from io import BytesIO
from PIL import Image
# import APC pipeline
from apc.apc_pipeline import APC
from apc.utils import visualize_conversation, create_image_with_text

# set device
device_vlm = "cuda:0"
device_vision = "cuda:0"

# load config
config_path = "apc/configs/qwenvl2_2.5_2b.yaml"
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = Box(config)

# load APC pipeline
apc = APC(config, device_vlm=device_vlm, device_vision=device_vision)

def download_image(image_url):
    response = requests.get(image_url, timeout=20)
    image = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))
    return image

ds = load_from_disk("test_dataset_150")

def run_single(i, example):
    image_url = example["image_url"]
    question = example["question"]
    options = [example["A"], example["B"], example["C"], example["D"]]
    correct = example["answer"]
    category = example["category"]

    # Download image
    image = download_image(image_url)

    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Index {i} — Category: {example['category']}")
    plt.show()
    
    # Build prompt
    valid_options = [(chr(65 + j), opt) for j, opt in enumerate(options) if opt and str(opt).strip()]
    options_text = " or ".join([f"{label}. {opt}" for label, opt in valid_options])
    prompt = f"From the camera's point of view, {question.strip()} {options_text}, give the letter of the correct answer."

    # print("Prompt:")
    # print(prompt)

    # Directory for saving intermediate results
    save_dir = f"outputs/benchmark/{category}_{i}"
    os.makedirs(save_dir, exist_ok=True)

    image_with_text = create_image_with_text(image, "[Q] " + prompt, fontsize=20)
    image_with_text.save(f"outputs/benchmark/3DSRBench_{i}_prompt.png")

    # Run APC pipeline
    response_text, conv_history = apc.run_apc(
        image,
        prompt,
        trace_save_dir=save_dir,
        perspective_prompt_type="visual",
        visualize_trace=False,
        visualize_scene_abstraction=False,
        return_conv_history=True,
        logging=False,
    )

    # print("Response", response_text)
    # Extract predicted answer (search for 'A', 'B', 'C', 'D')
    match = re.search(r"\b([ABCD])\b", response_text.upper())
    pred_letter = match.group(1) if match else None

    # print(len(conv_history))
    for i in range(len(conv_history)):
        print(conv_history[i])

    conv_viz = visualize_conversation(
        conv_history,
        width=900,
        row_gap=0,
        font_size=13,
        image_max_width=180,
        output_path=os.path.join(save_dir, "conversation_viz.png")
    )

    return {
        "index": example["index"],
        "category": category,
        "question": question,
        "prediction": pred_letter,
        "answer": correct,
        "is_correct": pred_letter == correct,
        "response_text": response_text,
    }

def run():
    results = []

    for i, example in enumerate(tqdm(ds[:1], desc="Evaluating 3DSRBench")):
        results.append(run_single(i, example))
    
    return results



if __name__ == "__main__":
    print("--------------------------------")
    print(ds)
    results = run_single(0, ds[0])
    print(results)