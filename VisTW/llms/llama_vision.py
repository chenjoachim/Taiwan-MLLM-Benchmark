import base64
import torch
import math
import warnings
from io import BytesIO
import requests
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoTokenizer
from transformers import BitsAndBytesConfig

class LLaMAVision:

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", quantized=True) -> None:
        quantization_config = None
        if quantized:
            double_quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
            )

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=double_quant_config,
        )
        self.processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
        self.model_name = model_name

    def __str__(self):
        return self.model_name.replace('/', '-')

    def __call__(self, prompt: str, image = None, max_tokens=1024, temperature=0.0, **kwargs) -> tuple[str, dict]:
        content = [{'type': 'text', 'text': prompt}]
        if image is not None:
            content.append({'type': 'image'})
        messages = [
            {"role": "user", "content": content}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            images=image,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)
        output = self.model.generate(**inputs,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            do_sample=False if temperature == 0.0 else True
                        )
        output_txt = self.processor.decode(output[0])
        response = output_txt.split('assistant<|end_header_id|>')[-1].replace('<|eot_id|>', '').strip()
        res_info = {
            'input': prompt,
            'output': response,
            'num_input_tokens': inputs.input_ids.shape[1],
            'num_output_tokens': len(output[0][inputs.input_ids.shape[1]:])
        }
        return response, res_info


if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
    processor.save_pretrained("train_vlm/test-exams-big/test_lora_yenting")

    llm = LLaMAVision('train_vlm/test-exams-big/test_lora_yenting')
    image = Image.open('659b87ba93383.jpg')
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "回答一下的多選題問題。並且在回覆的最後記得講格式： 答案: $字母 而字母是 ABCD 的其中一個。回答前請先一步一步(think step by step)想好答案。你必須使用中文回答。公司發行到期還本、每年固定利率之長期債券，如果折價或溢價發行，依一般公認會計原則採利息法認列之利息費用情況為何？\nA. 折價發行之債券，其利息費用逐期遞增，溢價發行者利息費用逐期遞減\nB. 折價發行之債券，其利息費用逐期遞減，溢價發行者利息費用逐期遞增\nC. 無論折價或溢價發行，其利息費用每期相等\nD. 無論折價或溢價發行，其利息費用均逐期遞減"}
        ]}
    ]
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "回答一下的多選題問題。並且在回覆的最後記得講格式： 答案: $字母 而字母是 ABCD 的其中一個。直接給我答案回覆，Do not think step by step。你必須使用中文回答。公司發行到期還本、每年固定利率之長期債券，如果折價或溢價發行，依一般公認會計原則採利息法認列之利息費用情況為何？\nA. 折價發行之債券，其利息費用逐期遞增，溢價發行者利息費用逐期遞減\nB. 折價發行之債券，其利息費用逐期遞減，溢價發行者利息費用逐期遞增\nC. 無論折價或溢價發行，其利息費用每期相等\nD. 無論折價或溢價發行，其利息費用均逐期遞減"}
        ]}
    ]
    # messages = [
    #     {"role": "user", "content": [
    #         {"type": "image"},
    #         {"type": "text", "text": "說明一下這圖片的內容"}
    #     ]}
    # ]
    res_text, res = llm("回答一下的多選題問題。並且在回覆的最後記得講格式： 答案: $字母 而字母是 ABCD 的其中一個。直接給我答案回覆，Do not think step by step。你必須使用中文回答。公司發行到期還本、每年固定利率之長期債券，如果折價或溢價發行，依一般公認會計原則採利息法認列之利息費用情況為何？\nA. 折價發行之債券，其利息費用逐期遞增，溢價發行者利息費用逐期遞減\nB. 折價發行之債券，其利息費用逐期遞減，溢價發行者利息費用逐期遞增\nC. 無論折價或溢價發行，其利息費用每期相等\nD. 無論折價或溢價發行，其利息費用均逐期遞減", image)
    print(res_text)