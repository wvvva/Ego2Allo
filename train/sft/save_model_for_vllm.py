from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "/jet/home/ydinga/idl_project/shared/qwen_4b_grpo_merged_4_8_2_unmerged", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = False, # Set to False for 16bit LoRA
)
FastVisionModel.for_inference(model) # Enable for inference!

SAVE_PATH = "/ocean/projects/cis250208p/shared/models/rl/Qwen3-VL-4B-Instruct-SFT-RL-4_8_2"

model.save_pretrained_merged(SAVE_PATH, tokenizer)