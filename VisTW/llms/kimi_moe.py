"""
pip install tiktoken blobfile
"""
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor


class KimiVL:

    def __init__(self, model_name: str = "moonshotai/Kimi-VL-A3B-Instruct") -> None:
        import tiktoken
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True, 
        )
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def __str__(self):
        return self.model_name.replace('/', '-')

    def __call__(self, prompt: str, image=None, max_tokens=2048, temperature=0.2, **kwargs) -> tuple[str, dict]:
        if image:
            image = image.convert('RGB')
            inputs = self.tokenizer.apply_chat_template([{"role": "user", "image": image, "content": prompt}],
                                                add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                                return_dict=True)  # chat mode
        else:
            inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                                return_dict=True)  # chat mode

        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": "input.png"}, 
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = self.processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # Create response info dictionary similar to Qwen2VL
        res_info = {
            'input': prompt,
            'output': response,
            'num_input_tokens': inputs.input_ids.shape[1],
            'num_output_tokens': len(generated_ids_trimmed[0])
        }
        return response, res_info


if __name__ == "__main__":
    from PIL import Image
    image = Image.open('failed.jpg')
    llm = KimiVL()
    res_text, res_info = llm(prompt="請用中文敘述一下",image=image)
    print(res_text)
    print(res_info)
