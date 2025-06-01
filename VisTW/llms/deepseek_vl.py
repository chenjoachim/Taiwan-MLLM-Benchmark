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
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from .utils import convert_pil2url


class DeepSeekVL:
    
    def __init__(self, model_name: str = "deepseek-ai/deepseek-vl-7b-chat") -> None:
        self.processor = VLChatProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer

        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="auto",
                                                    trust_remote_code=True)
        self.model_name = model_name

    def __str__(self):
        return self.model_name.replace('/', '-')

    def __call__(self, prompt: str, image = None, max_tokens=2048, temperature=0.0, **kwargs) -> tuple[str, dict]:
        conversation = [
            {
                "role": "User",
                "content": prompt
            },
            {
                "role": "Assistant", # wait, wtf?
                "content": ""
            }
        ]
        if image is not None:
            conversation = [
                {
                    "role": "User",
                    "content": prompt+"<image_placeholder>",
                    "images": ["./images/training_pipelines.png"]
                },
                {
                    "role": "Assistant", # wait, wtf?
                    "content": ""
                }
            ]
        
        prepare_inputs = self.processor(
            conversations=conversation,
            images=[image] if image is not None else [],
            force_batchify=True
        ).to(self.model.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True
        )
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        res_info = {
                'input': prompt,
                'output': answer,
                'num_input_tokens': prepare_inputs.input_ids.shape[1],
                'num_output_tokens': outputs.shape[1]
            }
        return answer, res_info

class DeepSeekVL2:
    
    def __init__(self, model_name: str = "deepseek-ai/deepseek-vl2") -> None:
        self.processor = DeepseekVLV2Processor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="auto",
                                                    quantization_config=quantization_config,
                                                    trust_remote_code=True
                                                )
        self.model_name = model_name

    def __str__(self):
        return self.model_name.replace('/', '-')

    def __call__(self, prompt: str, image = None, max_tokens=2048, temperature=0.0, **kwargs) -> tuple[str, dict]:
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        if image is not None:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n<|ref|>{prompt}<|/ref|>.",
                    # "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        
        prepare_inputs = self.processor(
            conversations=conversation,
            images=[image.convert('RGB')] if image is not None else [],
            force_batchify=True
        ).to(self.model.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=temperature,
            use_cache=True
        )
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        res_info = {
                'input': prompt,
                'output': answer,
                'num_input_tokens': prepare_inputs.input_ids.shape[1],
                'num_output_tokens': outputs.shape[1]
            }
        return answer, res_info
        

if __name__ == "__main__":
    from PIL import Image
    image = Image.open('replacement_images/5cd0f0daf3acc.jpg')
    llm = DeepSeekVL2()
    res_text, res_info = llm(prompt="請用中文敘述一下",image=image)
    print(res_text)
    print(res_info)
