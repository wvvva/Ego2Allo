# filename: qwenvl_server.py
# run with: uvicorn qwenvl_server:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

app = FastAPI(title="Qwen2.5-VL-7B API")

# ---------- Load model ----------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model... This may take a while.")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("Model loaded successfully!")

# ---------- Request schema ----------
class GenerationRequest(BaseModel):
    prompt: str
    # image_url: str | None = None
    max_new_tokens: int = 512

# ---------- Inference endpoint ----------
@app.post("/generate")
async def generate(req: GenerationRequest):
    # multimodal chat format
    # if req.image_url:
    #     msgs = [{"role": "user", "content": [
    #         {"type": "image", "image_url": req.image_url},
    #         {"type": "text", "text": req.prompt}
    #     ]}]
    # else:
    msgs = [{"role": "user", "content": [{"type": "text", "text": req.prompt}]}]

    # Apply chat template (no need to manually embed <|im_start|> tokens)
    text_in = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )

    # Collect image inputs properly
    images = []
    for m in msgs:
        for c in m["content"]:
            if c["type"] == "image":
                images.append(c["image_url"])

    # Prepare tensors safely
    inputs = processor(
        text=[text_in],
        images=images if images else None,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=req.max_new_tokens)

    response_text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return {"response": response_text}

@app.get("/")
async def root():
    return {"message": "Qwen2.5-VL-7B FastAPI server is running!"}
