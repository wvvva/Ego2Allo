import os
import pandas as pd
from APC_runner import APCRunner
from datasets import load_from_disk, load_dataset
import torch, gc
import argparse

args = argparse.ArgumentParser()
args.add_argument("--prompt-type", type=str, default="visual", help="Type of prompt to use: none, numerical, visual")

# Set environment variables
os.environ['PYGLET_HEADLESS'] = '1'
cache_root = os.environ.get("HF_HOME") or "/ocean/projects/cis250208p/shared/hf_cache"
os.makedirs(cache_root, exist_ok=True)
os.environ["HF_HOME"] = cache_root
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_root
os.environ["TRANSFORMERS_CACHE"] = cache_root

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["HF_HUB_DISABLE_BNB"] = "1"
os.environ["TRANSFORMERS_NO_BITSANDBYTES"] = "1"

# Load model and dataset
# model_name = "qwenvl2_5_7b_instruct" # Change this
# model_name = "qwenvl2_5_3b_instruct" # Change this
model_name = "grpo_trained" # Change this
# model_name = "qwenvl2_5_7b_instruct" # Change this
# model_name = "qwen3vl_4b_instruct"
config_path = f"apc/configs/{model_name}.yaml"

# Initialize APC runner
gc.collect()
torch.cuda.empty_cache()
apc_runner = APCRunner(config_path)

# Load dataset
# ds = load_from_disk("sample_dataset")
ds = load_from_disk("test_dataset_150")

# Run APC
# results = apc_runner.run_single(2, ds[2], verbose=True, prompt_type="numerical_react")
# print(results)

# prompt_type = "numerical"
prompt_type = args.parse_args().prompt_type

# print(ds)

results = apc_runner.run(ds, verbose=False, prompt_type=prompt_type)
df = pd.DataFrame(results)
df.to_csv(f"3DSRBench_raw_predictions_{model_name}_{prompt_type}_4b.csv", index=False) 

# # Run APC with numerical prompt
# prompt_type = "numerical"
# results = apc_runner.run(ds,verbose=True, prompt_type=prompt_type)
# df = pd.DataFrame(results)
# df.to_csv(f"3DSRBench_raw_predictions_{model_name}_{prompt_type}.csv", index=False)

# # numerical 7b 36:22<00:00, 14.55s/it
# # numerical 3b 29:27<00:00, 11.78s/it

# # Run APC with visual prompt
# prompt_type = "visual"
# results = apc_runner.run(ds,verbose=True, prompt_type=prompt_type)
# df = pd.DataFrame(results)
# df.to_csv(f"3DSRBench_raw_predictions_{model_name}_{prompt_type}.csv", index=False)

# # visual 7b 44:52<00:00, 17.95s/it
# # visual 3b 35:09<00:00, 14.06s/it

# numerical react 3b 1:02:12<00:00, 24.88s/it
