from .vlm_base import VLMBase
from openai import OpenAI
from typing import List, Dict, Any
from PIL import Image
import base64
import io
import os
import time
import uuid
import json
import hashlib

class ModelGemini(VLMBase):
    def __init__(self, config, device: str = "cuda"):
        super().__init__(device)
        self.load_model(config)


    def load_model(self, config, device: str = "cuda"):
        self.client = OpenAI(api_key=config.api_key, 
                            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                            )
        self.model = config.model_name


    def generate_response(self, prompt, image=None, max_new_tokens=1024, do_sample=False, temperature=0.0):
        if image is not None:
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]},
            ]
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]

        response = self.process_messages(messages, max_new_tokens, do_sample, temperature)
        
        return response


    def process_messages(self, messages, max_new_tokens=1024, do_sample=False, temperature=0.0):
        openai_messages = self._to_openai_messages(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            reasoning_effort="none", # "low" is another option should try both
            messages=openai_messages,
            max_tokens=4096,
            temperature=temperature,
        )
        content = response.choices[0].message.content

        self._debug_dump(messages, content)
        
        return content

    def _image_to_data_url(self, image: Any) -> str:
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        if isinstance(image, (bytes, bytearray)):
            b64 = base64.b64encode(image).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://") or image.startswith("data:"):
                return image
            if os.path.exists(image):
                with open(image, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/png;base64,{b64}"
        raise TypeError("Unsupported image type for conversion to data URL")

    def _to_openai_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            if isinstance(content, list):
                parts: List[Dict[str, Any]] = []
                for part in content:
                    ptype = part.get("type")
                    if ptype == "text":
                        parts.append({"type": "text", "text": part.get("text", "")})
                    elif ptype == "image":
                        data_url = self._image_to_data_url(part.get("image"))
                        parts.append({"type": "image_url", "image_url": {"url": data_url}})
                    elif ptype == "image_url" and isinstance(part.get("image_url"), dict):
                        parts.append(part)
                    else:
                        # Fallback: coerce unknown parts into text
                        text_val = part.get("text") or str(part)
                        parts.append({"type": "text", "text": text_val})
                converted.append({"role": role, "content": parts})
            elif isinstance(content, str):
                converted.append({"role": role, "content": content})
            else:
                # If the upstream provides a single image directly
                if isinstance(content, Image.Image) or isinstance(content, (bytes, bytearray)):
                    data_url = self._image_to_data_url(content)
                    converted.append({
                        "role": role,
                        "content": [{"type": "image_url", "image_url": {"url": data_url}}],
                    })
                else:
                    converted.append({"role": role, "content": str(content)})
        return converted

    def _debug_dump(self, messages, content):

        # Normalize conversation for hashing and dumping
        normalized_parts = []
        dump_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            m_content = msg.get("content")

            if isinstance(m_content, list):
                parts_dump: List[Dict[str, Any]] = []
                for part in m_content:
                    ptype = part.get("type")
                    if ptype == "text":
                        text_val = part.get("text", "")
                        normalized_parts.append(f"text:{text_val}")
                        parts_dump.append({"type": "text", "text": text_val})
                    elif ptype == "image":
                        data_url = self._image_to_data_url(part.get("image"))
                        normalized_parts.append(f"image:{data_url}")
                        parts_dump.append({"type": "image", "data_url": data_url})
                    elif ptype == "image_url":
                        # Keep as-is, ensure string url for hashing
                        url_val = part.get("image_url")
                        if isinstance(url_val, dict):
                            url_val = url_val.get("url")
                        url_str = str(url_val)
                        normalized_parts.append(f"image_url:{url_str}")
                        parts_dump.append({"type": "image_url", "url": url_str})
                    else:
                        # Fallback to text representation
                        text_val = part.get("text") or str(part)
                        normalized_parts.append(f"text:{text_val}")
                        parts_dump.append({"type": "text", "text": text_val})
                dump_messages.append({"role": role, "content": parts_dump})
            elif isinstance(m_content, str):
                normalized_parts.append(f"text:{m_content}")
                dump_messages.append({"role": role, "content": m_content})
            else:
                # Possible single image object or bytes
                if isinstance(m_content, Image.Image) or isinstance(m_content, (bytes, bytearray)):
                    data_url = self._image_to_data_url(m_content)
                    normalized_parts.append(f"image:{data_url}")
                    dump_messages.append({
                        "role": role,
                        "content": [{"type": "image", "data_url": data_url}],
                    })
                else:
                    text_val = str(m_content)
                    normalized_parts.append(f"text:{text_val}")
                    dump_messages.append({"role": role, "content": text_val})

        # Include answer in hash
        normalized_parts.append(f"answer:{content}")
        normalized = "\n".join(normalized_parts)
        dump_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()

        base_dir = os.path.join("outputs", "debug")
        os.makedirs(base_dir, exist_ok=True)
        dump_path = os.path.join(base_dir, f"{dump_hash}.json")

        dump_payload = {
            "id": dump_hash,
            "created_at": int(time.time()),
            "messages": dump_messages,
            "answer": content,
        }

        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(dump_payload, f, ensure_ascii=False, indent=2)