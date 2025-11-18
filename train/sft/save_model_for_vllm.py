from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "4b_lora_model_4_4_2", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = False, # Set to False for 16bit LoRA
)
FastVisionModel.for_inference(model) # Enable for inference!

SAVE_PATH = "/ocean/projects/cis250208p/shared/models/sft/Qwen3-VL-4B-Instruct-SFT_4_4_2"

model.save_pretrained_merged(SAVE_PATH, tokenizer)