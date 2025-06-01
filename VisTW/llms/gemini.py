import os
import base64
from io import BytesIO
import logging
import vertexai
from time import sleep
from vertexai.preview.generative_models import GenerativeModel, Image
import vertexai.preview.generative_models as generative_models
from .utils import retry_with_exponential_backoff, convert_pil2url

vertexai.init(project=os.environ['GCP_PROJECT_NAME'], location="us-central1")

class Gemini():

    SAFETY_SETTINGS={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }


    def __init__(self, model_name='gemini-1.0-pro-vision-001') -> None:
        self.model = GenerativeModel(model_name)
        self.model_name = model_name

    def __str__(self):
        return self.model_name

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, image=None, max_tokens=1024, top_p=1,top_k=1, temperature=0.0, **kwargs) -> tuple[str, dict]:
        conversation = [prompt]
        if image is not None:
            buffered = BytesIO()
            image.convert("RGB").save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            image_data = base64.b64encode(img_bytes).decode("utf-8")
            base64_string = f"data:image/png;base64,{image_data}"
            image_file = Image.from_bytes(img_bytes)
            conversation.append(image_file)

        result = self.model.generate_content(
                conversation,
                generation_config={
                    "max_output_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k)
                },
                safety_settings=self.SAFETY_SETTINGS,
                stream=False
        ).candidates[0].content.parts[0].text
        res_info = {
            "input": prompt,
            "output": result,
            "num_input_tokens": self.model.count_tokens(prompt).total_tokens,
            "num_output_tokens": self.model.count_tokens(result).total_tokens,
            "logprobs": []  # NOTE: currently the Gemini API does not provide logprobs
        }
        return result, res_info
