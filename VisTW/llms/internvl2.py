import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

class InternVL2:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    def __init__(self, model_name: str = "OpenGVLab/InternVL2-2B") -> None:
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
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

    def build_transform(self, input_size):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) 
            for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
            
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def process_image(self, image, input_size=448, max_num=12):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, 
            image_size=input_size, 
            use_thumbnail=True, 
            max_num=max_num
        )
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def __call__(self, prompt: str, image=None, max_tokens=2048, temperature=0.0, **kwargs) -> tuple[str, dict]:
        generation_config = {
            'max_new_tokens': max_tokens,
            'do_sample': True if temperature > 0.0 else False,
            'temperature': temperature
        }
        
        if image is not None:
            pixel_values = self.process_image(image).to(torch.bfloat16).cuda()
            prompt = f'<image>\n{prompt}'
            input_tokens = self.tokenizer.encode(prompt)
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config
            )
        else:
            input_tokens = self.tokenizer.encode(prompt)
            response = self.model.chat(
                self.tokenizer,
                None,
                prompt,
                generation_config
            )
        output_tokens = self.tokenizer.encode(response)

        # Create response info dictionary similar to Qwen2VL
        res_info = {
            'input': prompt,
            'output': response,
            'num_input_tokens': len(input_tokens),
            'num_output_tokens': len(output_tokens)
        }
        return response, res_info

if __name__ == "__main__":
    from PIL import Image
    image = Image.open('failed.jpg')
    llm = InternVL2("OpenGVLab/InternVL2-1B")
    res_text, res_info = llm(prompt="請用中文敘述一下",image=image)
    print(res_text)
    print(res_info)
