import os
os.environ['PYGLET_HEADLESS'] = '1'

import torch, gc
gc.collect()
torch.cuda.empty_cache()

import os
import sys
import json
os.environ["DISPLAY"] =':1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # set GPU device
sys.path.append("apc/vision_modules")
import yaml
from box import Box
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# import APC pipeline
from apc.apc_pipeline import APC
from apc.utils import visualize_conversation, create_image_with_text, split_response
import requests
from io import BytesIO
from tqdm import tqdm
import re
# set device
device_vlm = "cuda:0"
device_vision = "cuda:0"

class APCRunner:
    def __init__(self, config_path):
        # load config
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config)

        # load APC pipeline
        self.apc = APC(config, device_vlm=device_vlm, device_vision=device_vision)

    def download_image(self, image_url):
        response = requests.get(image_url, timeout=20)
        image = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))
        return image

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB").resize((512, 512))
        return image

    def run_single(self, i, example, verbose=False, prompt_type="visual", datasource="3DSRBench", conv_save_path=None):
        question = example["question"]
        category = example["category"]
        if (datasource == "3DSRBench"):
            image_url = example["image_url"]
            options = [example["A"], example["B"], example["C"], example["D"]]
            correct = example["answer"]
        elif (datasource == "SAT"):
            image_path = example["image"]
            options = example["answer_choices"]
            correct = example["correct_answer"]

        # Download image
        if (datasource == "3DSRBench"):
            image = self.download_image(image_url)
        elif (datasource == "SAT"):
            image = self.load_image(image_path)

        if conv_save_path is not None:
            os.makedirs(conv_save_path, exist_ok=True)
            conv_save_path = os.path.join(conv_save_path, f"conversation_{i}.jsonl")

        if verbose:
            plt.imshow(image)
            plt.axis("off")
            plt.title(f"Index {i} — Category: {example['category']}")
            plt.show()
        
        # Build prompt
        valid_options = [(chr(65 + j), opt) for j, opt in enumerate(options) if opt and str(opt).strip()]
        options_text = " or ".join([f"{label}. {opt}" for label, opt in valid_options])
        prompt = f"From the camera's point of view, {question.strip()} {options_text}, give the letter of the correct answer."

        if verbose:
            print("Prompt:")
            print(prompt)

        # Directory for saving intermediate results
        save_dir = f"outputs/benchmark/{category}_{i}"
        os.makedirs(save_dir, exist_ok=True)

        image_with_text = create_image_with_text(image, "[Q] " + prompt, fontsize=20)
        if verbose:
            image_with_text.save(f"outputs/benchmark/3DSRBench_{i}_prompt.png")

        # Run APC pipeline
        response_text, conv_history = self.apc.run_apc(
            image,
            prompt,
            trace_save_dir=save_dir,
            perspective_prompt_type=prompt_type,
            visualize_trace=False,
            visualize_scene_abstraction=False,
            return_conv_history=True if verbose else False,
            logging=False,
            conv_save_path=conv_save_path,
        )

        if verbose:
            print("Response", response_text)

        reasoning, answer = split_response(response_text)

        # Extract predicted answer (search for 'A', 'B', 'C', 'D')
        match = re.search(r"\b([ABCD])\b", answer.upper())
        pred_letter = match.group(1) if match else None

        if verbose:
            print(len(conv_history))
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
            "index": i,
            "category": category,
            "question": question,
            "reasoning": reasoning,
            "prediction": pred_letter,
            "answer": correct,
            "is_correct": pred_letter == correct,
            "response_text": response_text,
        }

    def run(self, ds, verbose=False, prompt_type="visual", datasource="3DSRBench", conv_save_path=None):
        results = []

        for i, example in enumerate(tqdm(ds, desc=f"Evaluating {datasource}")):
            try:
                results.append(self.run_single(i, ds[i], verbose, prompt_type, datasource, conv_save_path))
            except Exception as e:
                print(f"Error running example {i}: {e}")
                results.append({
                    "index": example["index"],
                    "category": example["category"],
                    "question": example["question"],
                    "prediction": "ERROR",
                    "reasoning": "",
                    "answer": example["answer"],
                    "is_correct": False,
                    "response_text": str(e),
                })
                continue
        
        return results