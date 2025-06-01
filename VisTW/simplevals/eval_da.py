import os
import json
import argparse
from typing import List
from tqdm import tqdm
from datasets import load_dataset
from llms.utils import get_llm
from .utils import (
    write_jsonl,
    normalize_response,
    evaluate_response,
    write_final_log,
    load_existing_entries,
    VALID_DATASETS
)

BASELINE_PROMPT = """回答以下的多選題問題。並且在回覆的最後記得講格式： 答案: $字母 而字母是 ABCD 的其中一個。直接給我答案回覆，不要給我任何習考的不週，Do not think step by step。你必須使用中文回答。

"""

choices = "ABCD"
sampler = get_llm('gpt-4o-mini')

def format_question(row, mode, leetspeak=False):
    question = row['question']
    for idx, choice in enumerate(choices):
        question += f'\n{choice}. {row[choice]}'
    return question


def process_question(row, llm, system_prompt, mode, stats, leetspeak=False):
    log_entry = {"question_id": stats['total'] }
    question = format_question(row, mode, leetspeak=leetspeak)
    ground_truth = row['answer']
    
    log_entry.update({
        "original_question": row['question'],
        "formatted_question": question,
        "ground_truth": ground_truth,
        "full_prompt": system_prompt + "\n\n" + question
    })

    res_text, res_info = llm(log_entry["full_prompt"],
                             image=None if mode == 'text' else row['image'],
                             max_tokens=2048
                )
    log_entry.update({
        "llm_response": res_text,
        "llm_info": res_info,
        "normalized_response": normalize_response(res_text)
    })

    evaluate_response(log_entry, mode, stats, sampler)
    return log_entry


def eval_dataset(llm, subject_name, mode="image", text_ver=False):
    if text_ver:
        dataset = load_dataset('VisTai/mcq-text', subject_name, split='test')
        logging_file = f"{subject_name}_text-da-{mode}_{str(llm)}.jsonl"
    else:
        dataset = load_dataset('VisTai/vistw-mcq', subject_name, split='test')
        logging_file = f"{subject_name}_da-{mode}_{str(llm)}.jsonl"

    stats = load_existing_entries(logging_file)
    full_path = os.path.join('execution_results', logging_file)
    
    with open(full_path, 'a') as log_file:
        for idx, row in enumerate(tqdm(dataset, dynamic_ncols=True, initial=stats['total'])):
            if idx in stats['existing_entries']:
                continue
            system_prompt = BASELINE_PROMPT
            log_entry = process_question(row, llm, system_prompt, mode, stats, leetspeak=False)
            json.dump(log_entry, log_file)
            log_file.write('\n')
            log_file.flush()

    write_final_log(logging_file, stats)
    return stats['hit'], stats['total'], stats['correct'], stats['negative_markings']

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM on MMLU datasets")
    parser.add_argument("model_name", type=str, help="Name of the LLM model to use")
    parser.add_argument("--series", type=str, help="Name of the LLM model to use", default=None)
    parser.add_argument("--datasets", nargs='+', default=['all'], choices=VALID_DATASETS + ['all'],
                        help="List of datasets to evaluate (default: all)")
    parser.add_argument("--mode",
                        choices=["baseline", "image", "text"],
                        default="text",
                        help="Evaluation mode (default: baseline)")
    parser.add_argument("--text_only",
                        action="store_true",
                        default=False,
                        help="Use text-only mode (default: False)")
    return parser.parse_args()

def get_datasets_to_evaluate(selected_datasets: List[str]) -> List[str]:
    if 'all' in selected_datasets:
        return VALID_DATASETS
    return selected_datasets

def main():
    args = parse_arguments()
    llm = get_llm(args.model_name, args.series)
    datasets_to_evaluate = get_datasets_to_evaluate(args.datasets)

    results = {}
    for dataset in datasets_to_evaluate:
        print(f"Evaluating {dataset}...")
        hit, total, correct, negative_markings = eval_dataset(llm, dataset,
                    mode=args.mode, text_ver=args.text_only
                )
        results[dataset] = {
            "score": hit / total,
            "correct": correct,
            "total": total
        }
        if args.mode == "neg_mark":
            results[dataset]["negative_markings"] = negative_markings

        print(f"Dataset: {dataset}")
        print(f"Score: {hit}/{total} ({hit/total:.2%})")
        if args.mode == "neg_mark":
            print(f"Correct answers: {correct}/{total} ({correct/total:.2%})")
            print(f"Negative markings: {negative_markings}/{total} ({negative_markings/total:.2%})")
        print()

    # Print overall results
    total_score = sum(result["score"] for result in results.values()) / len(results)
    print(f"Overall score across all evaluated datasets: {total_score:.2%}")

if __name__ == "__main__":
    main()