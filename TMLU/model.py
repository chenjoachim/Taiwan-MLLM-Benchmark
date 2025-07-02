import abc
import logging
from abc import abstractmethod

import google.generativeai as genai
import torch
import torch.nn.functional as F
import transformers
from anthropic import Anthropic
from anthropic import APIError as Anthropic_APIError
from google.ai import generativelanguage as glm
from openai import APIError as OpenAI_APIError
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def _get_dtype(
    dtype: str | torch.dtype, config: transformers.AutoConfig | None = None
) -> torch.dtype:
    """From https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/models/huggingface.py"""
    if dtype is None and config is not None:
        _torch_dtype = config.torch_dtype
    elif isinstance(dtype, str) and dtype != 'auto':
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


def log_memory_usage():
    """Simple memory logging utility"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        cached = torch.cuda.memory_reserved() / (1024**3)  # GB
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB", flush=True)


class LM(abc.ABC):
    def __init__(self, max_tokens, temperature):
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def generate(self, prompts):
        pass


class HFLM_vLLM(LM):
    def __init__(
        self,
        model_name,
        tensor_parallel_size,
        max_tokens,
        max_length,
        temperature,
        revision=None,
        dtype=None,
        cache_dir=None,
        tokenizer_name=None
    ):
        super().__init__(max_tokens, temperature)
        if tokenizer_name:
            auto_tokenizer_name = tokenizer_name
        else:
            auto_tokenizer_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            auto_tokenizer_name,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        print(dtype)
        self.llm = LLM(
            model=model_name,
            tokenizer=auto_tokenizer_name,
            tensor_parallel_size=tensor_parallel_size,
            max_num_batched_tokens=8192,
            max_model_len=8192,
            quantization='AWQ' if 'awq' in model_name.lower() else None,
            revision=revision,
            dtype=dtype,
            download_dir=cache_dir,
            device="cuda",
            trust_remote_code=True,
        )
        generation_config = GenerationConfig.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        config = AutoConfig.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        if max_length:
            self.model_max_length = max_length
        elif hasattr(generation_config, 'max_length'):
            self.model_max_length = generation_config.max_length
        elif hasattr(config, 'max_position_embeddings'):
            self.model_max_length = config.max_length.max_position_embeddings
        else:
            logger.error('model max length is unknown.')
            exit()

        self.sampling_params = SamplingParams(
            temperature=self.temperature, max_tokens=self.max_tokens, stop=['問題：']
        )

    def generate(self, dataset, prefill='正確答案：(', apply_chat_template=True):
        if apply_chat_template:
            dataset = dataset.map(
                lambda x: {
                    'prompt': self.tokenizer.apply_chat_template(
                        [{'role': 'user', 'content': x['prompt']}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    + prefill
                },
                load_from_cache_file=False,
            )
        else:
            dataset = dataset.map(
                lambda x: {'prompt': x['prompt'] + prefill},
                load_from_cache_file=False,
            )
        outputs = self.llm.generate(dataset['prompt'], self.sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        answers = [outputs[i].outputs[0].text for i in range(len(outputs))]
        return answers


class HFLM_transformers(LM):
    def __init__(
        self,
        model_name,
        max_tokens,
        max_length,
        temperature,
        revision=None,
        dtype=None,
        cache_dir=None,
        flash_attn=False,
        batch_size=1
    ):
        super().__init__(max_tokens, temperature)
        self.batch_size = batch_size
        self.config = AutoConfig.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        
        # Set padding token for batch processing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(_get_dtype(dtype, self.config))
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=_get_dtype(dtype, self.config),
            device_map="cuda",  # Changed from "cuda" to "auto"
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation='flash_attention_2' if flash_attn else 'sdpa',
        )
        
        # Fixed the attribute access and added fallback logic
        if max_length:
            self.model_max_length = max_length
        elif hasattr(self.llm.generation_config, 'max_length'):
            self.model_max_length = self.llm.generation_config.max_length
        elif hasattr(self.llm.config, 'max_position_embeddings'):
            self.model_max_length = self.llm.config.max_position_embeddings  # Fixed this line
        elif hasattr(self.config, 'max_position_embeddings'):
            self.model_max_length = self.config.max_position_embeddings
        else:
            logger.error('model max length is unknown.')
            self.model_max_length = 2048  # Set a reasonable default instead of exiting
            
        print(f"Model max length: {self.model_max_length}")
        self.llm.eval()

    def get_tokenizer(self):
        return self.tokenizer

    def encode(self, text):
        encoded = self.tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
        return encoded

    def encode_pair(self, context, conti):
        whole_enc = self.encode(context + conti)
        context_enc = self.encode(context)
        context_enc_len = context_enc.shape[1]
        conti_enc = whole_enc[:, context_enc_len:]
        conti_enc_len = conti_enc.shape[1]
        context_enc = context_enc[:, -(self.model_max_length - conti_enc_len) :]
        return context_enc, conti_enc

    def generate(self, dataset, prefill='正確答案：(', apply_chat_template=True):
        print("START BATCH INFERENCE", flush=True)
        log_memory_usage()
        
        # Prepare all prompts
        prompts = []
        for example in dataset:
            if apply_chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': example['prompt']}],
                    tokenize=False,
                    add_generation_prompt=True,
                ) + prefill
            else:
                prompt = example['prompt'] + prefill
            prompts.append(prompt)
        
        print(f"Processing {len(prompts)} prompts with batch size {self.batch_size}", flush=True)
        
        answers = []
        with torch.no_grad():
            # Process in batches
            for i in tqdm(range(0, len(prompts), self.batch_size), desc="Batch inference"):
                batch_prompts = prompts[i:i + self.batch_size]
                
                # Tokenize batch with left padding for generation
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors='pt',
                    padding='longest',  # Use longest for efficiency
                    truncation=True,
                    max_length=self.model_max_length,
                    pad_to_multiple_of=8,  # For efficiency on modern GPUs
                    return_attention_mask=True
                ).to("cuda")
                
                # Store input lengths for each sample in the batch
                input_lengths = []
                for j in range(len(batch_prompts)):
                    # Count non-padding tokens
                    attention_mask = inputs.attention_mask[j]
                    input_length = attention_mask.sum().item()
                    input_lengths.append(input_length)
                
                # Generate for batch
                with torch.inference_mode():
                    outputs = self.llm.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=self.temperature > 0,
                        use_cache=True,
                        cache_implementation='static',
                        disable_compile=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode batch outputs
                for j, output in enumerate(outputs):
                    # Extract only the generated part (after the input)
                    input_length = input_lengths[j]
                    generated_ids = output[input_length:]
                    generated_text = self.tokenizer.decode(
                        generated_ids, 
                        skip_special_tokens=True
                    )
                    answers.append(generated_text)
                
                # Clean up batch
                del inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Log memory usage periodically
                if (i // self.batch_size) % 10 == 0:
                    log_memory_usage()
                
        print("FINISH BATCH INFERENCE", flush=True)
        log_memory_usage()
        return answers

    def get_batch_size(self):
        return self.batch_size

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

class OpenAI_LM(LM):
    def __init__(
        self,
        model_name,
        max_tokens,
        temperature,
        api_key,
        base_url=None,
        timeout=20.0,
        max_retries=100,
    ):
        super().__init__(max_tokens, temperature)

        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )
        self.model = model_name

    def query(self, prompt, prefill=''):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': prefill.strip()},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        answer = response.choices[0].message.content
        return answer

    def generate(self, dataset, prefill=''):
        answers = []
        for example in tqdm(dataset):
            try:
                answer = self.query(example['prompt'], prefill)
                answers.append(answer)
            except OpenAI_APIError as e:
                logger.error(e.message)
                break
        return answers


class Anthropic_LM(LM):
    def __init__(self, model_name, max_tokens, temperature, api_key, timeout=20.0, max_retries=100):
        super().__init__(max_tokens, temperature)
        self.client = Anthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model_name

    def query(self, prompt, prefill=''):
        response = self.client.messages.create(
            model=self.model,
            messages=[
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': prefill.strip()},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        answer = response.content[0].text
        return answer

    def generate(self, dataset, prefill=''):
        answers = []
        for example in tqdm(dataset):
            try:
                answer = self.query(example['prompt'], prefill)
                answers.append(answer)
            except Anthropic_APIError as e:
                logger.error(e.message)
                break
        return answers


class Google_LM(LM):
    def __init__(self, model_name, max_tokens, temperature, api_key, timeout=20.0, max_retries=100):
        super().__init__(max_tokens, temperature)
        genai.configure(api_key=api_key)
        generation_config = glm.GenerationConfig(
            max_output_tokens=max_tokens, temperature=temperature
        )
        safety_settings = [
            {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
        ]
        self.request_options = {'retry': max_retries, 'timeout': timeout}
        self.client = genai.GenerativeModel(
            model_name, generation_config=generation_config, safety_settings=safety_settings
        )

    def query(self, prompt, prefill=''):
        response = self.client.generate_content(
            f'{prompt}{prefill.strip()}',
            # request_options=self.request_options
        )
        answer = response.text
        return answer

    def generate(self, dataset, prefill=''):
        answers = []
        for example in tqdm(dataset):
            for _ in range(self.request_options['retry']):
                try:
                    answer = self.query(example['prompt'], prefill)
                    answers.append(answer)
                    break
                except Exception as e:
                    logger.error(e)
            else:
                break
        return answers
