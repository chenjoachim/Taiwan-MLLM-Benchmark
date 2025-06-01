import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


class GLMv4:

    def __init__(self, model_name: str = "THUDM/glm-4v-9b") -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            use_fast=False
        )
        self.model_name = model_name

    def __str__(self):
        return self.model_name.replace('/', '-')

    def __call__(self, prompt: str, image=None, max_tokens=2048, temperature=0.0, **kwargs) -> tuple[str, dict]:
        if image:
            image = image.convert('RGB')
            inputs = self.tokenizer.apply_chat_template([{"role": "user", "image": image, "content": prompt}],
                                                add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                                return_dict=True)  # chat mode
        else:
            inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                                return_dict=True)  # chat mode

        inputs = inputs.to(self.model.device)

        gen_kwargs = {"max_length": max_tokens, "do_sample": True if temperature > 0 else False, "top_k": 1, "temperature": temperature}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]

        response = self.tokenizer.decode(outputs[0])
        # Create response info dictionary similar to Qwen2VL
        res_info = {
            'input': prompt,
            'output': response,
            'num_input_tokens': inputs['input_ids'].shape[1],
            'num_output_tokens': len(outputs)
        }
        return response, res_info


if __name__ == "__main__":
    from PIL import Image
    image = Image.open('failed.jpg')
    llm = GLMv4("THUDM/glm-4v-9b")
    res_text, res_info = llm(prompt="請用中文敘述一下",image=image)
    print(res_text)
    print(res_info)
