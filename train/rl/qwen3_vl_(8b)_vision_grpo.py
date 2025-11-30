import os

os.environ["ACCELERATE_DISABLE_FP16_GRAD_SCALER"] = "true"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import re
from typing import List, Optional

import torch
from datasets import Dataset
from PIL import Image

from trl import GRPOConfig, GRPOTrainer
from unsloth import FastVisionModel
from transformers import TrainerCallback

MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
JSON_PATH = "/jet/home/vwei/Ego2Allo/rl_data/orien.json"
MAX_SEQ_LENGTH = 4096
IMAGE_RESOLUTION = (512, 512)
USE_WANDB = os.getenv("USE_WANDB", "1").lower() not in {"0", "false"}
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ego2allo-grpo")
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", "qwen3vl_grpo")

REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.float16,
    load_in_4bit=True,
    fast_inference=False,
    gpu_memory_utilization=0.8,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    use_gradient_checkpointing="unsloth",
)


def _load_orien_json(path: str) -> Dataset:
    with open(path, "r") as f:
        raw = json.load(f)

    rows = []
    for ex in raw:
        options_text = ", ".join(f"{k}: {v}" for k, v in ex["options"].items())
        user_text = (
            f"{ex['question']} Options: {options_text}. "
            "Think carefully before answering. "
            f"First describe your reasoning between {REASONING_START} and {REASONING_END}, "
            f"then give the final answer letter between {SOLUTION_START} and {SOLUTION_END}."
        )
        rows.append(
            {
                "image_path": ex["img"],
                "prompt_template": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": user_text},
                        ],
                    }
                ],
                "answer": ex["answer"].strip().upper(),
            }
        )

    dataset = Dataset.from_list(rows)

    def add_image_and_prompt(example):
        image = Image.open(example["image_path"]).convert("RGB")
        image = image.resize(IMAGE_RESOLUTION)
        prompt = tokenizer.apply_chat_template(
            example["prompt_template"],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"image": image, "prompt": prompt}

    dataset = dataset.map(add_image_and_prompt, remove_columns=None)
    dataset = dataset.remove_columns(["prompt_template", "image_path"])
    return dataset


train_dataset = _load_orien_json(JSON_PATH)


def formatting_reward_func(completions: List[str], **_) -> List[float]:
    thinking_pattern = f"{REASONING_START}(.*?){REASONING_END}"
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"
    rewards = []
    for completion in completions:
        reward = 0.0
        if len(re.findall(thinking_pattern, completion, re.DOTALL)) == 1:
            reward += 1.0
        if len(re.findall(answer_pattern, completion, re.DOTALL)) == 1:
            reward += 1.0
        if completion:
            removal = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion) - len(removal)) / len(completion) >= 0.5:
                reward -= 2.0
        rewards.append(reward)
    return rewards


def correctness_reward_func(prompts, completions, answer, **_) -> List[float]:
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"
    rewards = []
    for completion, gold in zip(completions, answer):
        match = re.search(answer_pattern, completion, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue
        predicted = match.group(1).strip().upper()
        reward = 2.0 if predicted == gold else 0.0
        rewards.append(reward)
    return rewards


def reasoning_length_penalty(completions: List[str], **_) -> List[float]:
    thinking_pattern = f"{REASONING_START}(.*?){REASONING_END}"
    limit_tokens = 160
    rewards = []
    for completion in completions:
        match = re.search(thinking_pattern, completion, re.DOTALL)
        if not match:
            rewards.append(-1.0)
            continue
        reasoning = match.group(1)
        length = len(reasoning.split())
        excess = max(0, length - limit_tokens)
        rewards.append(-excess / limit_tokens)
    return rewards

report_to_target = "wandb" if USE_WANDB else "none"

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    log_completions=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=2,
    max_prompt_length=1024,
    max_completion_length=512,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to=report_to_target,
    output_dir="/ocean/projects/cis250208p/vwei",
    bf16=False,
    fp16=False,
    importance_sampling_level="sequence",
    mask_truncated_completions=False,
    loss_type="dr_grpo",
)


class SaveLoraCallback(TrainerCallback):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return
        if model is None:
            return
        step_dir = os.path.join(self.base_dir, f"step-{state.global_step}")
        os.makedirs(step_dir, exist_ok=True)
        model.save_pretrained(step_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(step_dir)


def _latest_checkpoint(path: str) -> Optional[str]:
    if not os.path.isdir(path):
        return None
    checkpoints = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(path, d))
    ]
    if not checkpoints:
        return None
    return sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]


resume_from_checkpoint = os.getenv("GRPO_RESUME_FROM") or _latest_checkpoint(training_args.output_dir)
lora_checkpoint_dir = os.path.join(training_args.output_dir, "lora_checkpoints")

wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None
if USE_WANDB:
    import wandb

    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model_name": MODEL_NAME,
            "json_path": JSON_PATH,
            "max_seq_length": MAX_SEQ_LENGTH,
            "image_resolution": IMAGE_RESOLUTION,
            "learning_rate": training_args.learning_rate,
            "num_generations": training_args.num_generations,
        },
    )

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    reward_funcs=[
        formatting_reward_func,
        correctness_reward_func,
        reasoning_length_penalty,
    ],
    train_dataset=train_dataset,
    callbacks=[SaveLoraCallback(lora_checkpoint_dir)],
)

if resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
else:
    trainer.train()

if wandb_run is not None:
    wandb_run.finish()

model.save_pretrained("/ocean/projects/cis250208p/vwei/qwen_3b_dpo_merged")
tokenizer.save_pretrained("/ocean/projects/cis250208p/vwei/qwen_3b_dpo_merged")
