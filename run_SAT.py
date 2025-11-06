from datasets import load_dataset
from APC_runner import APCRunner
import pandas as pd

model_name = "gemini_2.5_flash" # Change this
config_path = f"apc/configs/{model_name}.yaml"

apc_runner = APCRunner(config_path)

ds = load_dataset("path_to_SAT")

results = apc_runner.run_single(2, ds[2], verbose=True)
# results = apc_runner.run(ds, verbose=False)
# df = pd.DataFrame(results)
# df.to_csv(f"SAT_raw_predictions_{model_name}.csv", index=False)