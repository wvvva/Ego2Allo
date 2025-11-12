from APC_runner import APCRunner
import pandas as pd
import json
import os
from datasets import Dataset
import argparse

args = argparse.ArgumentParser()
args.add_argument("--start_index", type=int, default=800)
args.add_argument("--end_index", type=int, default=1500)
args = args.parse_args()

start_index = args.start_index
end_index = args.end_index

model_name = "gemini_2.5_flash" # Change this
# model_name = "qwenvl2_5_7b_instruct"
config_path = f"apc/configs/{model_name}.yaml"

apc_runner = APCRunner(config_path)

# load the dataset
SAT_DATASET_FOLDER = "/ocean/projects/cis250208p/shared/datasets/SAT"
data = []

with open(os.path.join(SAT_DATASET_FOLDER, "SAT_labeled.jsonl"), "r") as f:
    for line in f:
        data.append(json.loads(line))

ds = Dataset.from_list(data)

print(ds)

ds = ds.select(range(start_index, end_index))

SAT_TRAIN_GENERATED_FOLDER = "/ocean/projects/cis250208p/shared/datasets/SAT/train_generated_2"

# run the model on the dataset
#results = apc_runner.run_single(1, ds[1], verbose=False, datasource="SAT", conv_save_path=SAT_TRAIN_GENERATED_FOLDER)

# results = apc_runner.run(ds, verbose=False, datasource="SAT", conv_save_path=SAT_TRAIN_GENERATED_FOLDER)
results = apc_runner.run(ds, verbose=False, datasource="SAT", index_offset=start_index, conv_save_path=SAT_TRAIN_GENERATED_FOLDER)
# df = pd.DataFrame(results)

# df.to_csv(f"SAT_raw_predictions_{model_name}_2.csv", index=False)