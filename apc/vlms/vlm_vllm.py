from .vlm_base import VLMBase
from vllm import LLM

class ModelvLLM(VLMBase):
    '''
    VLM class for vLLM
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

        self.llm = LLM(
            model=config.pretrained_model,
            enforce_eager=True,
            max_model_len=4096,
            max_num_seqs=1,
            limit_mm_per_prompt={"image": 1},
            dtype="auto",
            device_map=device,
            seed=3407,
        )

        self.sampling_params = SamplingParams(
            top_p=0.8,
            top_k=20,
            temperature=0.7,
            repetition_penalty=1.0,
            presence_penalty=1.5,
        )

    def messages_to_vllm_prompt(messages):
        """
        Convert OpenAI-style messages to Qwen2.5-Omni ChatML for vLLM.
        Supports multi-turn and multiple images per message.
        """
        chatml = ""

        for msg in messages:
            role = msg["role"]
            content_list = msg["content"]

            chatml += f"<|im_start|>{role}\n"

            for item in content_list:
                if item["type"] == "image":
                    # Insert vision placeholder
                    chatml += "<|vision_bos|><|IMAGE|><|vision_eos|>"
                elif item["type"] == "text":
                    chatml += item["text"]

            chatml += "<|im_end|>\n"

        # final turn for model to speak
        chatml += "<|im_start|>assistant\n"
        return chatml

    def collect_images(messages):
        """Extracts all images (in order) from messages."""
        imgs = []
        for msg in messages:
            for item in msg["content"]:
                if item["type"] == "image":
                    imgs.append(item["image"])
        return imgs

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

        prompt = messages_to_vllm_prompt(messages)
        images = collect_images(messages)
        iamges = [convert_image_mode(image, "RGB") for image in images]

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": images
            }
        }

        output = self.llm.generate(
            inputs,
            sampling_params=self.sampling_params,
        )

        response = outputs[0].outputs[0].text
        return response