from APC_runner import APCRunner
import pandas as pd
from datasets import load_from_disk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompt-type", type=str, required=False, default=None)
args = parser.parse_args()

print("Modules imported.")

# model_name = "qwen3vl_4b_instruct_sft" # Change this
# model_name = "qwen3vl_4b_instruct" # Change this
model_name = "qwenvl2_5_3b_instruct"
# model_name = "qwenvl2_5_7b_instruct"
config_path = f"apc/configs/{model_name}.yaml"

apc_runner = APCRunner(config_path)

print("APC Runner initialized.")

# load the dataset
ds = load_from_disk("test_dataset_150")

# run the model on the dataset
# results = apc_runner.run_single(3, ds[3], verbose=True, datasource="3DSRBench")

results = apc_runner.run(ds, verbose=False, datasource="3DSRBench", prompt_type=args.prompt)
df = pd.DataFrame(results)

df.to_csv(f"3DSRBench_raw_predictions_{model_name}_8_8_2.csv", index=False)