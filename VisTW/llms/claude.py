import io
import os
import base64
from io import BytesIO
from PIL import Image
from anthropic import Anthropic
from llms.utils import retry_with_exponential_backoff

class ClaudeChat():

    def __init__(self, model_name='claude-3-haiku-20240307') -> None:
        self.client = Anthropic(api_key=os.environ['ANTHROPIC_KEY'])
        self.model_name = model_name

    def __str__(self):
        return self.model_name

    def __call__(self, prompt, image=None, max_tokens=512, temperature=0.0, **kwargs) -> str:
        content = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        if image is not None:
            # Resize the image if necessary
            image = image.convert('RGB')
            max_size = 2048
            img_byte_arr = io.BytesIO()
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.LANCZOS)
            # Convert the resized image back to bytes
            image.save(img_byte_arr, format='jpeg')
            img_data = img_byte_arr.getvalue()
            content[0]['content'].append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.standard_b64encode(img_data).decode('utf-8'),
                        },
                    })
            assert len(content[0]['content']) == 2
        message = self.client.messages.create(
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            messages=content,
            model=self.model_name,
        )
        res_text = message.content[0].text
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": message.usage.input_tokens,
            "num_output_tokens": message.usage.output_tokens,
            "logprobs": []  # NOTE: currently the Claude API does not provide logprobs
        }
        return res_text, res_info

if __name__ == "__main__":
    from PIL import Image
    image = Image.open("659b87ba93383.jpg")
    llm = ClaudeChat(model_name="claude-3-haiku-20240307")
    res_text, res_info = llm(prompt="What is in this image", image=image)
    print(res_text)
    print(res_info)
