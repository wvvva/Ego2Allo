from datasets import load_dataset

cache_dir = "/ocean/projects/cis250208p/shared/datasets"

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("ccvl/3DSRBench", cache_dir=cache_dir)
