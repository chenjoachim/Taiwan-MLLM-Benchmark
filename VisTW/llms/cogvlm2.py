"""
Tutorials : https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat

git clone https://github.com/deepseek-ai/DeepSeek-VL
cd DeepSeek-VL
pip install -e .

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
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import convert_pil2url


class CogVLM2:
    
    def __init__(self, model_name: str = "THUDM/cogvlm2-llama3-chat-19B") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        self.model_name = model_name

    def __str__(self):
        return self.model_name.replace('/', '-')

    def __call__(self, prompt: str, image = None, max_tokens=2048, temperature=0.0, **kwargs) -> tuple[str, dict]:
        if image is None:
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=prompt,
                history=[],
                template_version='chat'
            )
        else:
            input_by_model = self.model.build_conversation_input_ids(
                            self.tokenizer,
                            query=prompt,
                            history=[],
                            images=[image.convert('RGB')],
                            template_version='chat'
                        )

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.model.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.model.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.model.device),
            'images': [[input_by_model['images'][0].to(self.model.device).to(torch.bfloat16)]] if image is not None else None,
        }
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": 128002,  
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]

        res_info = {
                'input': prompt,
                'output': response,
                'num_input_tokens': inputs['input_ids'].shape[1],
                'num_output_tokens': outputs.shape[1]
            }
        return response, res_info
        

if __name__ == "__main__":
    from PIL import Image
    image = Image.open('replacement_images/5cd0f0daf3acc.jpg')
    llm = CogVLM2()
    res_text, res_info = llm(prompt="請用中文敘述一下",image=image)
    print(res_text)
    print(res_info)
