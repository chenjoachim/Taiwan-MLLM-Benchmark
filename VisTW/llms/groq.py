import os
from groq import Groq
from .utils import retry_with_exponential_backoff, convert_pil2url


class GroqModel:
    model_list = [
        "llama-3.2-11b-text-preview",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-90b-text-preview",
        "llama-3.2-90b-vision-preview"
    ]

    def __init__(self, model_name: str = "llama-3.2-11b-text-preview") -> None:
        self.client = Groq(
            # This is the default and can be omitted
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.model_name = model_name

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, image=None, max_tokens=1024, temperature=0.0, **kwargs) -> tuple[str, dict]:
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
    llm = GroqModel(model_name="llama3-70b-8192")
    res_text, res_info = llm(prompt="Are you an instruction-tuned version of LLama-3?")
    print(res_text)
    print(res_info)
