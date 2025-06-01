import os
from openai import OpenAI

from .utils import retry_with_exponential_backoff, convert_pil2url

class OpenAIChat():
    TOP_LOGPROBS = 4

    def __init__(self, model_name='gpt-3.5-turbo-0125') -> None:
        params = {}
        if os.getenv('CUSTOM_API_URL') and 'gpt-' not in model_name:
            params['base_url'] = os.environ['CUSTOM_API_URL']
            params['api_key'] = os.environ['CUSTOM_API_KEY']
        else:
            params = {'api_key': os.environ['OAI_KEY']}
        self.client = OpenAI(**params)
        self.model_name = model_name

    def __str__(self):
        return self.model_name.split('/')[-1].replace(':free', '')

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, image=None, max_tokens=1024, top_p=0.95, temperature=0.0, **kwargs) -> tuple[str, dict]:
        content = [{'type': 'text', 'text': prompt}]
        if image is not None:
            content.append({'type': 'image_url', 
                            'image_url': {'url': convert_pil2url(image)}
                        })
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': content}],
            # temperature=float(temperature),
            # max_tokens=int(max_tokens),
            # top_p=float(top_p),
            # logprobs=True,
            # top_logprobs=self.TOP_LOGPROBS,
            **kwargs
        )
        if response.choices[0].logprobs:
            log_prob_seq = response.choices[0].logprobs.content
        else:
            log_prob_seq = []

        # assert response.usage.completion_tokens == len(log_prob_seq)
        res_text = response.choices[0].message.content
        # print(response.usage.completion_tokens, len(log_prob_seq), res_text)
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": response.usage.prompt_tokens,
            "num_output_tokens": response.usage.completion_tokens,
            "logprobs": [[{"token": pos_info.token, "logprob": pos_info.logprob} for pos_info in position.top_logprobs] for position in log_prob_seq]
        }
        return res_text, res_info

if __name__ == "__main__":
    llm = OpenAIChat()
    res_text, res_info = llm(prompt="Say apple!")
    print(res_text)
    print()
    from pprint import pprint
    pprint(res_info)
