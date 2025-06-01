from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from PIL import Image
import requests
import torch

config = AutoConfig.from_pretrained("benchang1110/TaiVisionLM-base-v2",trust_remote_code=True)
processor = AutoProcessor.from_pretrained("benchang1110/TaiVisionLM-base-v2",trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("benchang1110/TaiVisionLM-base-v2",trust_remote_code=True,torch_dtype=torch.float16,attn_implementation="sdpa").to('cuda')
model.eval()
image = Image.open('659b87ba93383.jpg').convert("RGB")
text = "描述圖片, 請回答答案是 A,B,C,D其中一個"
inputs = processor(text=text,images=image, return_tensors="pt", padding=False).to('cuda')
outputs = model.generate(**inputs,max_length=512)[0]
outputs = processor.tokenizer.decode(outputs, skip_special_tokens=False)
print(outputs)

