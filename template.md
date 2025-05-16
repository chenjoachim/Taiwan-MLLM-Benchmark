# Benchmark Name

> Briefly describe the benchmark, including its purpose and dataset size. Link to the original Github repo. If you maintain a well-organized fork, please link that instead.

This benchmark focuses on evaluating models' **textual knowledge about Taiwan**.
 The full dataset contains **1,000 test samples per subject**, across **26 subjects**.

**Repository:** `repo_url`

## Installation

> Include package dependencies in `requirements.txt`, ideally generated via `pip freeze > requirements.txt`.

To install dependencies:

```bash
pip install -r requirements.txt
```

## Baselines

> Put full huggingface model id here.

- `model_A`: `Qwen/Qwen2.5-Omni-7B`
- `model_B`: `Qwen/Qwen2-Audio-7B-Instruct`
- ...

## Usage

> Provide detailed steps to reproduce your baselines. Include all necessary scripts and configuration details such as generation parameters. (Commit & push `run_eval.py` and `calculate_acc.py` in this case).

1. Run the command to get model generation:

```bash
python run_eval.py --model model_A
```

2. Calculate the metrics:

```bash
python calculate_acc.py
```

## Results

> Add results in the table below. Ensure they match exactly with our Google sheets.

|           | model A | model B | model C |
| --------- | ------- | ------- | ------- |
| subject A | 98.5%   | 41.5%   | 71.5%   |
| subject B | 76.5%   | 32.5%   | 68.5%   |