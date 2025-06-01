import base64
import torch
import math
import warnings
from io import BytesIO
import requests
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from transformers import Gemma3ForConditionalGeneration, AutoProcessor, AutoTokenizer
from transformers import BitsAndBytesConfig

import torch._dynamo
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 1024


class Gemma3():

    def __init__(self, model_name="google/gemma-3-12b-it") -> None:
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, device_map="cuda", torch_dtype=torch.bfloat16
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model_name = model_name

    def __str__(self):
        return self.model_name.split('/')[-1]

    def __call__(self, prompt: str, image=None, max_tokens=1024, top_p=1,top_k=1, temperature=0.0, **kwargs) -> tuple[str, dict]:
        conversation = [{ "type": "text", "text": prompt}]
        if image is not None:
            buffered = BytesIO()
            image.convert("RGB").save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            image_data = base64.b64encode(img_bytes).decode("utf-8")
            base64_string = f"data:image/png;base64,{image_data}"
            conversation = [{"type": "image", "image": base64_string}] + conversation
        messages = [{"role": "user", "content": conversation}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        res_info = {
            "input": prompt,
            "output": decoded,
            "num_input_tokens": input_len,
            "num_output_tokens": len(generation),
            "logprobs": []  # NOTE: currently the Gemini API does not provide logprobs
        }
        return decoded, res_info
