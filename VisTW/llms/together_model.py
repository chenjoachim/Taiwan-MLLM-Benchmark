import os
from together import Together
from llms.utils import retry_with_exponential_backoff, convert_pil2url

class TogetherModel:

    TOP_LOGPROBS = 1

    def __init__(self, model_name: str = "llama3-70b-8192") -> None:
        self.client = Together(
            # This is the default and can be omitted
            api_key=os.environ.get("TOGETHER_API_KEY"),
        )
        self.model_name = model_name

    def __str__(self):
        return self.model_name.replace('/', '-')

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, image = None, max_tokens=2048, temperature=0.0, **kwargs) -> tuple[str, dict]:
        content = [{'type': 'text', 'text': prompt}]
        if image is not None:
            content.append({'type': 'image_url', 
                            'image_url': {'url': convert_pil2url(image)}
                        })

        res = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=self.TOP_LOGPROBS,
            **kwargs
        )
        res_text = res.choices[0].message.content
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": res.usage.prompt_tokens,
            "num_output_tokens": res.usage.completion_tokens,
            "logprobs": []
        }
        return res_text, res_info

if __name__ == "__main__":
    from PIL import Image
    image = Image.open('replacement_images/5cd0f0daf3acc.jpg')
    llm = TogetherModel(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo")
    res_text, res_info = llm(prompt="請用中文敘述一下",image=image)
    print(res_text)
    print(res_info)
