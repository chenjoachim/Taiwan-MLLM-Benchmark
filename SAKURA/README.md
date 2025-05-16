# SAKURA

repo: https://github.com/anthony-wss/SAKURA/tree/main

## Installation

We recommend using `uv` for fast dependency management:

```bash
uv pip venv --python 3.10
source .venv/bin/activate
uv pip install tqdm==4.66.4 openai==1.33.0
```

## Baselines

- `Qwen2.5-Omni-7B`: `Qwen/Qwen2.5-Omni-7B`
- `Qwen2-Audio-7B`: `Qwen/Qwen2-Audio-7B`
- `Phi-4-MM`: `microsoft/Phi-4-multimodal-instruct`

## Usage

### Part 1: Responses Generation

Online model support is currently **work in progress**.

#### Qwen 2.5 Omni

We use the `transformers` library to run Qwen 2.5 Omni.
 Our implementation follows the [Qwen 2.5 Omni cookbook](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb), and the logic is implemented in `inference_qwen2_5Omni.py`.

To run the inference:

```bash
python inference_qwen2_5Omni.py
```

Results will be saved to `./results/<subject>_results.json`.

### Part 2: Model as Judge

1. Set the `OPENAI_API_KEY` environment variables.
2. Run the evaluation:

```bash
python llm_judge.py -i ./results/<subject>_results.json -o judge_outputs/
```

3. Calculate accuracy:

```bash
python calculate_acc.py -i judge_outputs/<model_name>_judgements.json
```

## Results

|            | Qwen2.5-Omni-7B | Qwen2-Audio-7B | Phi-4-MM |
| ---------- | --------------- | -------------- | -------- |
| single hop | WIP             | WIP            | WIP      |
| multi hop  | WIP             | WIP            | WIP      |