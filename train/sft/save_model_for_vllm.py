from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "/jet/home/ydinga/idl_project/ydinga/checkpoint-1201", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = False, # Set to False for 16bit LoRA
)
FastVisionModel.for_inference(model) # Enable for inference!

SAVE_PATH = "/jet/home/ydinga/idl_project/shared/models/rl/Qwen3-VL-4B-Instruct-SFT-RL-8_8_2"

model.save_pretrained_merged(SAVE_PATH, tokenizer)