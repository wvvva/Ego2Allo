from .vlm_base import VLMBase

# QwenVL2.5
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info

class ModelQwenVL2_5(VLMBase):
    '''
    VLM class for QwenVL2.5
    '''
    def __init__(
            self, 
            config, 
            device: str = "cuda"
        ):
        super().__init__(device)
        self.device = device

        # Initialize model
        self.load_model(config, device=device)

    def load_model(self, config, device="cuda"):
        cfg = AutoConfig.from_pretrained(config.pretrained_model)
        print(f"[INFO] Loaded config {config.pretrained_model}")
        print(f"[INFO] Loaded config hidden_size={cfg.hidden_size}, model_type={cfg.model_type}")

        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.pretrained_model, 
            dtype="auto", 
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(config.pretrained_model)
        print(f"[INFO] Loaded model type: {self.vlm_model.config.model_type}")
        print(f"[INFO] Hidden size: {self.vlm_model.config.hidden_size}")

    def generate_response(
        self,
        prompt,
        image=None,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
    ):
        """
        Run a single VQA
        """
        # set messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ],
            }
        ]
        # process the conversation
        response = self.process_messages(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        return response

    def process_messages(
        self,
        messages,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
    ):
        """
        Process full conversation
        """
        # process messages
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # check if image is present
        has_image = False
        for message in messages:
            if "image" in message['content'][0]:
                has_image = True
                break

        if has_image:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs, video_inputs = None, None
        
        # process
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Run the model
        generated_ids = self.vlm_model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response