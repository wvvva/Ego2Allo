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

# run the model on the dataset
results = apc_runner.run_single(2, ds[2], verbose=True, datasource="SAT")
# results = apc_runner.run(ds, verbose=False, datasource="SAT")
# df = pd.DataFrame(results)
# df.to_csv(f"SAT_raw_predictions_{model_name}.csv", index=False)