from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
import json
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

import os
import wandb

# Resume from checkpoint
# run = wandb.init()
# artifact = run.use_artifact('<username>/<Wandb-project-name>/<run-id>', type='model')
# artifact_dir = artifact.download()
# trainer.train(resume_from_checkpoint=artifact_dir)

os.environ["WANDB_PROJECT"] = "Ego2Allo-VLM-SFT"
os.environ["WANDB_LOG_MODEL"] = "try1"

print("Loading model")

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-4B-Instruct",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

print("Model loaded")
print("Getting PEFT model")

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 8,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0.1,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

print("Training data loaded")
print("Loading training data")

with open("/ocean/projects/cis250208p/shared/datasets/train_data.json", "r") as f:
    train_data = json.load(f)

with open("/ocean/projects/cis250208p/shared/datasets/test_data.json", "r") as f:
    test_data = json.load(f)

print("Training data loaded")
print("Enabling training")

FastVisionModel.for_training(model) # Enable for training!


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = train_data,
    eval_dataset = test_data,
    args = SFTConfig(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        # max_steps = 1000, # Set to None for full training runs
        num_train_epochs = 1, # 1 - 3
        learning_rate = 1e-4,
        optim = "adamw_8bit",
        weight_decay = 0.1,
        lr_scheduler_type = "linear",
        seed = 3407,

        report_to = "wandb",     # For Weights and Biases
        logging_steps = 1,
        save_strategy = "steps",
        save_steps = 50,
        run_name = "try1",
        output_dir = "training_checkpoints",
        
        eval_strategy = "steps",             # evaluate every N steps
        eval_steps = 25,                     # how many steps until we do evaluation
        load_best_model_at_end = True,       # MUST USE for early stopping
        metric_for_best_model = "eval_loss", # metric we want to early stop on
        greater_is_better = False,           # the lower the eval loss, the better

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()
    
print("Training complete")
print("Saving model")

model.save_pretrained("4b_lora_model")  # Local saving
tokenizer.save_pretrained("4b_lora_model")

print("Model saved")