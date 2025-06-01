"""
Partial code taken from https://github.com/kq-chen/qwen-vl-utils/blob/main/src/qwen_vl_utils/vision_process.py
"""
import base64
import torch
import math
import warnings
from io import BytesIO
import requests
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoProcessor, LlavaForConditionalGeneration
from .utils import convert_pil2url


class LLaVA:
    
    def __init__(self, model_name: str = "01-ai/Yi-VL-6B") -> None:
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model_name = model_name

    def __str__(self):
        return self.model_name.replace('/', '-')

    def __call__(self, prompt: str, image = None, max_tokens=2048, temperature=0.0, **kwargs) -> tuple[str, dict]:
        content = [{'type': 'text', 'text': prompt}]
        if image is not None:
            content.append({'type': 'image'})

        messages= [
            {
                "role": "user",
                "content": content
            }
        ]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        # Preparation for inference
        # print('hi')
        inputs = processor(images=[image] if image is not None else None, 
                           text=text_prompt,
                           padding=True, 
                           return_tensors="pt"
                        ).to(self.model.device, torch.float16)
        # print('shit')
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs,
                                        max_new_tokens=max_tokens,
                                        temperature=temperature,
                                        do_sample=False if temperature == 0.0 else False
                                )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        res_info = {
                'input': prompt,
                'output': output_text[0],
                'num_input_tokens': inputs.input_ids.shape[1],
                'num_output_tokens': len(generated_ids[0][inputs.input_ids.shape[1]:])
            }
        return output_text[0], res_info
        

if __name__ == "__main__":
    from PIL import Image
    image = Image.open('replacement_images/5cd0f0daf3acc.jpg')
    llm = LLaVA()
    res_text, res_info = llm(prompt="請用中文敘述一下",image=image)
    print(res_text)
    print(res_info)
