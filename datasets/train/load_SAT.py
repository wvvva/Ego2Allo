from datasets import load_dataset
import io
from PIL import Image
import json
import random
import os
from tqdm import tqdm

cache_dir = "/ocean/projects/cis250208p/shared/datasets"
output_dir = cache_dir + "/SAT"
os.makedirs(output_dir, exist_ok=True)

# seed for reproducibility
random.seed(42)

split = "static"
dataset = load_dataset(
    cache_dir,
    data_files={
        "static": "SAT_static.parquet",
    },
    cache_dir=cache_dir,
    batch_size=128,
)

print(dataset)

# filter out the examples that have multiple images
filtered_dataset = [example for example in tqdm(dataset[split]) if len(example['image_bytes']) == 1]

# shuffle the dataset
random.shuffle(filtered_dataset)

filtered_dataset = filtered_dataset[:8000]

text_data = []

for index, example in tqdm(enumerate(filtered_dataset)):
    try:
        image = Image.open(io.BytesIO(example['image_bytes'][0]))
        image.save(output_dir + f"/SAT_{index}.png")
        text_data.append({
            "image": output_dir + f"/SAT_{index}.png",
            "question": example['question'],
            "answer_choices": example['answers'],
            "correct_answer": example['correct_answer']
        })
    except Exception as e:
        print(f"Error processing example {example['index']}: {e}")
        continue


with open(output_dir + "/SAT_data.jsonl", "w") as f:
    for item in text_data:
        json.dump(item, f, ensure_ascii=False, indent=4)
        f.write("\n")


# Qwen-VL2,5 7B used to filter out examples that have similar question types as 3DSRBench benchmark.
# Use GPT-4 to create training data to further finetune a smaller VLM model on SAT dataset.