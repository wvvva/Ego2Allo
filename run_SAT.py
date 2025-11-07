from APC_runner import APCRunner
import pandas as pd
import json
import os
from datasets import Dataset

model_name = "gemini_2.5_flash" # Change this
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

ds = ds.select(range(25))

SAT_TRAIN_GENERATED_FOLDER = "/ocean/projects/cis250208p/shared/datasets/SAT/train_generated"

# run the model on the dataset
results = apc_runner.run_single(1, ds[1], verbose=False, datasource="SAT", conv_save_path=SAT_TRAIN_GENERATED_FOLDER)
results = apc_runner.run(ds, verbose=False, datasource="SAT", conv_save_path=SAT_TRAIN_GENERATED_FOLDER)
df = pd.DataFrame(results)
df.to_csv(f"SAT_raw_predictions_{model_name}.csv", index=False)