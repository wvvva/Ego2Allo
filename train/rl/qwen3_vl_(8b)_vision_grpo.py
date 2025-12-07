import logging
import os

os.environ["ACCELERATE_DISABLE_FP16_GRAD_SCALER"] = "true"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import re
from typing import List, Optional

import torch
from datasets import Dataset, concatenate_datasets
from PIL import Image
from glob import glob

from trl import GRPOConfig, GRPOTrainer
from unsloth import FastVisionModel
from transformers import TrainerCallback

MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
JSON_PATH = "/ocean/projects/cis250208p/vwei/Ego2Allo/rl_data/*"
MAX_SEQ_LENGTH = 2048
IMAGE_RESOLUTION = (168, 168)
USE_WANDB = os.getenv("USE_WANDB", "1").lower() not in {"0", "false"}
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ego2allo-grpo")
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", "qwen3vl_grpo")


logger = logging.getLogger(__name__)


def _env_number(name: str, default, cast=int):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return cast(raw_value)
    except ValueError:
        logger.warning("Invalid value %r supplied for %s; falling back to %s.", raw_value, name, default)
        return default


PER_DEVICE_BATCH_SIZE = _env_number("GRPO_PER_DEVICE_BATCH_SIZE", 1, int)
GRAD_ACCUM_STEPS = _env_number("GRPO_GRAD_ACCUMULATION_STEPS", 2, int)
NUM_GENERATIONS = _env_number("GRPO_NUM_GENERATIONS",2, int)
MAX_PROMPT_LEN = _env_number("GRPO_MAX_PROMPT_LENGTH", 512, int)
MAX_COMPLETION_LEN = _env_number("GRPO_MAX_COMPLETION_LENGTH", 268, int)
GEN_TEMPERATURE = float(os.getenv("GRPO_TEMPERATURE", "0.3"))
GEN_TOP_P = float(os.getenv("GRPO_TOP_P", "0.8"))
GEN_REPEAT_PENALTY = float(os.getenv("GRPO_REPETITION_PENALTY", "1.05"))
GEN_NUM_BEAMS = _env_number("GRPO_NUM_BEAMS", 1, int)
GEN_DO_SAMPLE = os.getenv("GRPO_DO_SAMPLE", "1").lower() not in {"0", "false"}

REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


def ensure_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = []
        for part in value:
            text = ensure_text(part).strip()
            if text:
                parts.append(text)
        return " ".join(parts)
    return str(value)

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.float16,
    load_in_4bit=False,
    fast_inference=False,
    gpu_memory_utilization=0.8,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=False,
    r=4,
    lora_alpha=4,
    lora_dropout=0.0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    use_gradient_checkpointing=False,
)
FastVisionModel.for_training(model)
# FastVisionModel.for_inference(model, cache_vision=True)

if hasattr(model, "print_trainable_parameters"):
    model.print_trainable_parameters()
model.generation_config.temperature = GEN_TEMPERATURE
model.generation_config.top_p = GEN_TOP_P
model.generation_config.repetition_penalty = GEN_REPEAT_PENALTY
model.generation_config.num_beams = GEN_NUM_BEAMS
model.generation_config.do_sample = GEN_DO_SAMPLE


def _load_orien_json(path: str) -> Dataset:
    with open(path, "r") as f:
        raw = json.load(f)

    rows = []
    for ex in raw:
        options_text = ", ".join(f"{k}: {ensure_text(v)}" for k, v in ex["options"].items())
        user_text = (
            f"{ensure_text(ex['question'])} Options: {options_text}. "
            "Think carefully before answering. "
            f"First describe your reasoning between {REASONING_START} and {REASONING_END}, "
            f"then give the final answer letter between {SOLUTION_START} and {SOLUTION_END}."
        )
        rows.append(
            {
                "image_path": ensure_text(ex["img"]),
                "prompt_template": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": user_text},
                        ],
                    }
                ],
                "answer": ensure_text(ex["answer"]).strip().upper(),
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


def load_training_dataset(json_path: str) -> Dataset:
    """
    Accept either a single JSON file or a glob-style pattern pointing to multiple files.
    """
    if os.path.isfile(json_path):
        return _load_orien_json(json_path)

    matches = sorted(glob(json_path))
    if not matches:
        raise FileNotFoundError(f"No JSON files matched pattern: {json_path}")

    datasets = [_load_orien_json(match) for match in matches]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


train_dataset = load_training_dataset(JSON_PATH)


def unified_reward_func(prompts, completions, **kwargs):
    """
    TRL passes dataset columns via keyword arguments. We rely on the `answer`
    column but fall back to zeros if it's absent so the reward func never
    crashes when the trainer only provides prompts/completions.
    """
    answers = kwargs.get("answer") or kwargs.get("answers")
    if answers is None:
        answers = [""] * len(completions)
    elif isinstance(answers, torch.Tensor):
        answers = answers.tolist()
    elif isinstance(answers, tuple):
        answers = list(answers)
    elif not isinstance(answers, list):
        answers = [answers] * len(completions)
    if len(answers) != len(completions):
        if len(answers) == 1:
            answers = answers * len(completions)
        else:
            answers = (answers + [""] * len(completions))[: len(completions)]

    thinking_pattern = f"{REASONING_START}(.*?){REASONING_END}"
    answer_pattern   = f"{SOLUTION_START}(.*?){SOLUTION_END}"

    limit_tokens = 160
    rewards = []

    for prompt, completion, gold in zip(prompts, completions, answers):
        total_reward = 0.0

        # ---------------------------
        # Formatting reward component
        # ---------------------------
        # Reward exact 1 reasoning block
        thinking_blocks = re.findall(thinking_pattern, completion, re.DOTALL)
        if thinking_blocks:
            if len(thinking_blocks) == 1:
                total_reward += 1.0
            else:
                total_reward -= 0.1

        # Reward exact 1 answer block
        answer_blocks = re.findall(answer_pattern, completion, re.DOTALL)
        if answer_blocks:
            if len(answer_blocks) == 1:
                total_reward += 1.0
            else:
                total_reward -= 0.5

        # Penalize spammy newlines
        newline_count = completion.count("\n")
        total_chars = max(len(completion), 1)
        newline_ratio = newline_count / total_chars
        if newline_ratio > 0.5:
            total_reward -= 2.0

        # ---------------------------
        # Correctness reward component
        # ---------------------------
        match = re.search(answer_pattern, completion, re.DOTALL)
        if match:
            predicted = match.group(1).strip().upper()
            gold_clean = str(gold).strip().upper()
            if predicted == gold_clean:
                total_reward += 2.0

        # ---------------------------
        # Reasoning length penalty
        # ---------------------------
        match = re.search(thinking_pattern, completion, re.DOTALL)
        if not match:
            total_reward -= 0.5
        else:
            reasoning = match.group(1)
            length = len(reasoning.split())
            if length > limit_tokens:
                excess = length - limit_tokens
                total_reward += -min(excess / limit_tokens, 1.0)

        rewards.append(total_reward)

    return rewards

report_to_target = "wandb" if USE_WANDB else "none"

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    log_completions=False,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_generations=NUM_GENERATIONS,
    max_prompt_length=MAX_PROMPT_LEN,
    max_completion_length=MAX_COMPLETION_LEN,
    num_train_epochs=1,
    save_steps=10,
    max_grad_norm=0.1,
    report_to=report_to_target,
    output_dir="/ocean/projects/cis250208p/vwei/4-4",
    bf16=False,
    fp16=True,
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
    reward_funcs=unified_reward_func,
    train_dataset=train_dataset,
    callbacks=[SaveLoraCallback(lora_checkpoint_dir)],
)

if resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
else:
    trainer.train()

if wandb_run is not None:
    wandb_run.finish()

merged = model.merge_and_unload()
merged.save_pretrained("/ocean/projects/cis250208p/shared/qwen_4b_grpo_merged_4_4")
tokenizer.save_pretrained("/ocean/projects/cis250208p/shared/qwen_4b_grpo_merged_4_4")