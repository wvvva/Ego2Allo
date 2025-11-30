import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Disable scaling — REQUIRED for fp16 training with Unsloth
os.environ["ACCELERATE_DISABLE_FP16_GRAD_SCALER"] = "true"

# Disable FA2 + BF16 (your GPU unsupported)
os.environ["UNSLOTH_DISABLE_BF16"] = "true"
os.environ["UNSLOTH_DISABLE_FLASH_ATTENTION_2"] = "true"

from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
PatchDPOTrainer()

import torch
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

from datasets import Dataset
from PIL import Image
import json

dtype = torch.float16

# -----------------------------------------------
# 1. Load your 3DSRBench JSON and convert to DPO
# -----------------------------------------------

REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

def load_3dsrbench_for_dpo(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    dpo_rows = []

    for ex in data:
        # ---- Prompt ----
        prompt = (
            f"{ex['question']} Options: "
            + ", ".join(f"{k}: {v}" for k, v in ex["options"].items())
            + ".\n\nProvide reasoning and final answer."
        )

        # ---- Chosen answer (correct) ----
        correct_letter = ex["answer"].strip()

        chosen = (
            f"{REASONING_START} The correct answer is {correct_letter}. "
            f"Because {ex['options'][correct_letter]} is higher. "
            f"{REASONING_END}"
            f"{SOLUTION_START}{correct_letter}{SOLUTION_END}"
        )

        # ---- Rejected (random wrong option) ----
        wrong = [k for k in ex["options"] if k != correct_letter]
        wrong_letter = wrong[0]   # simplest deterministic choice

        rejected = (
            f"{REASONING_START} The answer is {wrong_letter}. "
            f"{REASONING_END}"
            f"{SOLUTION_START}{wrong_letter}{SOLUTION_END}"
        )

        dpo_rows.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    return Dataset.from_list(dpo_rows)


# -----------------------------------------------
# 2. Load model (minimal changes)
# -----------------------------------------------
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=1024,
    dtype="float16",          # REQUIRED (not torch.float16)
    load_in_4bit=False,       # or True if you want massive VRAM savings
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
)

# -----------------------------------------------
# 3. Prepare your dataset for DPO
# -----------------------------------------------
json_path = "/jet/home/vwei/Ego2Allo/rl_data/orien.json"
train_dataset = load_3dsrbench_for_dpo(json_path)

# -----------------------------------------------
# 4. DPO Trainer (minimal editing)
# -----------------------------------------------
dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,      # reference model is auto-cloned inside trainer
    args = DPOConfig(   output_dir="/ocean/projects/cis250208p/vwei/3b_DPO",
                        bf16 = False,
                        fp16 = True,
                        gradient_accumulation_steps=1,
                        fp16_full_eval=False,
                        do_train=True,
                        do_eval=False,
                        gradient_checkpointing=False,
                        optim="adamw_torch",
                        no_cuda=False,
                        fp16_opt_level="O0"),
    beta = 0.1,
    train_dataset = train_dataset,
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
    
)

dpo_trainer.train()

model = model.merge_and_unload()
model.save_pretrained("/ocean/projects/cis250208p/vwei/qwen_3b_dpo_merged")
tokenizer.save_pretrained("/ocean/projects/cis250208p/vwei/qwen_3b_dpo_merged")


