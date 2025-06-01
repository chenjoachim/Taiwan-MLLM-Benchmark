"""
Tutorials:
This is a sample implementation of the Freeze model using MediaTek Research's Llama-Breeze2 models.
Usage:
    from PIL import Image
    image = Image.open("64045.png")
    freeze = Freeze()
    result, info = freeze(prompt="請問第二名可獲得多少獎金？", image=image, temperature=0.05, max_tokens=1500)
    print(result)
    print(info)
"""

from transformers import AutoModel, AutoTokenizer, GenerationConfig
import torch
import tempfile
from PIL import Image
from mtkresearch.llm.image_process_v3 import load_image
from mtkresearch.llm.prompt import MRPromptV3

class MRPomptV3Patch(MRPromptV3):
    def _get_content_from_list(self, content_list):
        content = ''
        pixel_values = None
        for x in content_list:
            if x['type'] == 'text':
                content += x['text']
            elif x['type'] == 'image':
                if isinstance(x['image_path'], str):
                    image = Image.open(x['image_path']).convert('RGB')
                elif isinstance(x['image_path'], Image.Image):
                    image = x['image_path'].convert('RGB')

                pixel_values = load_image(image)
                width, height = image.size
                image_content_token_num = self._get_image_content_token_num(width, height)
                image_content_str = ''.join([self.image_content_token] * image_content_token_num)
                content += f'{self.image_start_token}{image_content_str}{self.image_end_token}\n'
            elif x['type'] == 'bbox':
                bboxes = [self._normalize_bbox(box, x["width"], x["height"]) for box in x["coords"]]
                content += f'{self.bbox_start_token}{repr(bboxes)}{self.bbox_end_token}'
            else:
                raise ValueError('unknown type in the content')
        return content, pixel_values


class Breeze2:
    def __init__(self, model_id: str = 'MediaTek-Research/Llama-Breeze2-3B-Instruct') -> None:
        self.model_id = model_id
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='auto',
            img_context_token_id=128212
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.prompt_engine = MRPomptV3Patch()
        self.sys_prompt = (
            'You are a helpful AI assistant built by MediaTek Research. '
            'The user you are helping speaks Traditional Chinese and comes from Taiwan.'
        )

    def __str__(self):
        return self.model_id.replace('/', '-')

    def _save_temp_image(self, image: Image) -> str:
        """Save a PIL image to a temporary PNG file and return the file path."""
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp.name, format="PNG")
        tmp.close()
        return tmp.name

    def _inference(self, prompt: str, gen_config: GenerationConfig, pixel_values=None) -> str:
        """Tokenize the prompt (and optional image pixel values) and generate a response."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
        if pixel_values is None:
            output_tensors = self.model.generate(**inputs, generation_config=gen_config)
        else:
            output_tensors = self.model.generate(
                **inputs,
                generation_config=gen_config,
                pixel_values=pixel_values.to(self.model.device, dtype=self.model.dtype)
            )
        output_str = self.tokenizer.decode(output_tensors[0], skip_special_tokens=True)
        return output_str

    def __call__(self, prompt: str, image: Image = None, max_tokens: int = 2048, temperature: float = 0.00, **kwargs) -> tuple[str, dict]:
        """
        Build the conversation structure, get the prompt (and pixel values if an image is provided)
        from the prompt engine, run inference, and return the parsed result along with info.
        """
        # Build generation config using __call__ parameters.
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            eos_token_id=128009,
            **kwargs
        )
        has_image = False
        if image is not None:
            # Ensure the image is in RGB mode and save it to a temporary file.
            conversations = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": [
                    {"type": "image", "image_path": image},
                    {"type": "text", "text": prompt}
                ]}
            ]
            has_image = True
        else:
            conversations = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ]
        if has_image:
            prompt_str, pixel_values = self.prompt_engine.get_prompt(conversations)
            output_str = self._inference(prompt_str, gen_config, pixel_values=pixel_values)
        else:
            prompt_str = self.prompt_engine.get_prompt(conversations)
            output_str = self._inference(prompt_str, gen_config)
        result = self.prompt_engine.parse_generated_str(output_str)

        # Count tokens for input and output.
        num_input_tokens = len(self.tokenizer(prompt_str).input_ids)
        num_output_tokens = len(self.tokenizer(result['content']).input_ids)

        res_info = {
            "input": prompt,
            "output": result['content'],
            "model_id": self.model_id,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
        }
        return result['content'], res_info

if __name__ == "__main__":
    # Example usage:
    image = Image.open("64045.png")
    freeze = Breeze2()
    result, info = freeze(prompt="請問第二名可獲得多少獎金？", image=image, temperature=0.05, max_tokens=1500)
    print(result)
    print(info)
