import base64
from io import BytesIO
import time
try:
    import groq
    has_groq = True
except ImportError:
    has_groq = False

try:
    import together
    has_together = True
except ImportError:
    has_together = False
import openai
import anthropic
import google.api_core.exceptions as g_exceptions
import urllib.request
from colorama import Fore, Style




def convert_pil2url(image):
    buffered = BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    image_data = base64.b64encode(img_bytes).decode("utf-8")
    image_base64 = f"data:image/jpeg;base64,{image_data}"
    return image_base64


def get_llm(model_name: str, series: str = None):
    if series is None:
        if model_name[:3] in ('gpt', 'o1-'):
            series = 'openai'
        elif model_name[:6] == 'claude':
            series = 'anthropic'
        elif model_name[:6] == 'gemini':
            series = 'gemini'
        if series is None:
            raise ValueError('unable to found matching series for current model, please provide the API provider in --series value')
    if series == 'gemini':
        from llms.gemini import Gemini
        return Gemini(model_name)
    elif series == 'gemini_dev':
        from llms.gemini_gen import Gemini
        return Gemini(model_name)
    elif series == 'openai':
        from llms.oai_chat import OpenAIChat
        return OpenAIChat(model_name)
    elif series == 'anthropic':
        from llms.claude import ClaudeChat
        return ClaudeChat(model_name)
    elif series == 'anthropic_vertex':
        from llms.vertex_claude import ClaudeChat
        return ClaudeChat(model_name)
    elif series == 'hf_model':
        from llms.hf_model import HFModel
        return HFModel(model_name)
    elif series == 'qwen':
        from llms.qwen_vl import QwenVL
        return QwenVL(model_name)
    elif series == 'internvl2':
        from llms.internvl2 import InternVL2
        return InternVL2(model_name)
    elif series == 'mllama':
        from llms.llama_vision import LLaMAVision
        return LLaMAVision(model_name)
    elif series == 'gemma3':
        from llms.gemma3 import Gemma3
        return Gemma3(model_name)
    elif series == 'llava':
        from llms.llava import LLaVA
        return LLaVA(model_name)
    elif series == 'ds':
        from llms.deepseek_vl import DeepSeekVL
        return DeepSeekVL(model_name)
    elif series == 'ds2':
        from llms.deepseek_vl import DeepSeekVL2
        return DeepSeekVL2(model_name)
    elif series == 'thudm':
        from llms.cogvlm2 import CogVLM2
        return CogVLM2(model_name)
    elif series == 'kimi':
        from llms.kimi_moe import KimiVL
        return KimiVL(model_name)
    elif series == 'breeze2':
        from llms.breeze2 import Breeze2
        return Breeze2(model_name)
    elif series == "groq":
        from llms.groq_model import GroqModel
        return GroqModel(model_name)
    elif series == "together":
        from llms.together_model import TogetherModel
        return TogetherModel(model_name)
    raise ValueError('series : {} for {} is not yet supported'.format(series, model_name))

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 0.25,
    exponential_base: float = 2,
    max_retries: int = 10
):
    # Define errors based on available libraries.
    errors_tuple = (
        openai.RateLimitError,
        openai.APIError,
        g_exceptions.ResourceExhausted,
        g_exceptions.ServiceUnavailable,
        g_exceptions.GoogleAPIError,
        anthropic.BadRequestError,
        anthropic.InternalServerError,
        anthropic.RateLimitError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        TypeError,
        ValueError, IndexError, UnboundLocalError
    )
    if has_groq:
        errors_tuple += (groq.RateLimitError,
                        groq.InternalServerError,
                        groq.APIConnectionError)
    if has_together:
        errors_tuple += (together.error.RateLimitError,)
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specific errors
            except errors_tuple as e:
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if isinstance(e, ValueError) or (num_retries > max_retries):
                    print(Fore.RED + f"ValueError / Maximum number of retries ({max_retries}) exceeded." + Style.RESET_ALL)
                    result = 'error:{}'.format(e)
                    prompt = kwargs["prompt"] if "prompt" in kwargs else args[1]
                    res_info = {
                        "input": prompt,
                        "output": result,
                        "num_input_tokens": len(prompt) // 4,  # approximation
                        "num_output_tokens": 0,
                        "logprobs": []
                    }
                    return result, res_info
                # Sleep for the delay
                print(Fore.YELLOW + f"Error encountered ({e}). Retry ({num_retries}) after {delay} seconds..." + Style.RESET_ALL)
                time.sleep(delay)
                # Increment the delay
                delay *= exponential_base
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
    return wrapper
