import os
import json
import argparse
from typing import List
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from llms.utils import get_llm
from .utils import (
    normalize_response,
)

def load_existing_entries(logging_file, log_dir="freeform_log"):
    full_path = os.path.join(log_dir, logging_file)
    print(full_path)

    stats = {'total': 0, 'existing_entries': {}}
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if 'question_id' in entry:
                    stats['existing_entries'][entry['question_id']] = entry
                    stats['total'] += 1
    return stats

def resize_image(img, scale_factor=0.5):
    """
    Resize an image to 50% of its original size using Pillow.
    
    Args:
        input_path (str): Path to the input image
        scale_factor (float, optional): Scale factor for resizing. Default is 0.5 (50%)
    
    Returns:
        str: Path to the resized image
    """
    try:
        # Open the image        
        # Get original dimensions
        width, height = img.size
        
        # Calculate new dimensions (50% of original)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        return resized_img
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None

def process_question(row, llm, mode, stats, system_prompt='', resize=None):
    log_entry = {"question_id": row['thread_id'] }
    ground_truth = row['ground_truth']
    log_entry.update({
        "original_question": row['question'],
        "ground_truth": ground_truth,
        "system_prompt": system_prompt,
        "full_prompt": (system_prompt + "\n\n" + row['question']) if len(system_prompt) else row['question']
    })

    image = None
    if mode != 'text':
        image = row['image']
        if resize is not None:
            image = resize_image(image, scale_factor=resize)
    
    res_text, res_info = llm(log_entry["full_prompt"],
                             image=image,
                             max_tokens=2048
                )
    stats['total'] += 1
    log_entry.update({
        "llm_response": res_text,
        "llm_info": res_info,
        "normalized_response": normalize_response(res_text)
    })
    return log_entry


def eval_dataset(llm, mode="image", text_ver=False, system_prompt='', resize=None):
    dataset = load_dataset('VisTai/vistw-dialogue', split='test')
    if len(system_prompt) == 0:
        logging_file = f"{str(llm)}.jsonl"
    else:
        logging_file = f"{system_prompt}-{str(llm)}.jsonl"
    
    log_dir = "freeform_log"
    if resize is not None:
        log_dir = f"freeform_log{resize}"
    
    os.makedirs(log_dir, exist_ok=True)
    stats = load_existing_entries(logging_file, log_dir=log_dir)
    full_path = os.path.join(log_dir, logging_file)
    
    with open(full_path, 'a') as log_file:
        for idx, row in enumerate(tqdm(dataset, dynamic_ncols=True, initial=stats['total'])):
            if row['thread_id'] in stats['existing_entries']:
                continue
            log_entry = process_question(row, llm, mode, stats, system_prompt=system_prompt, resize=resize)
            json.dump(log_entry, log_file)
            log_file.write('\n')
            log_file.flush()

    return stats['total']

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM on MMLU datasets")
    parser.add_argument("model_name", type=str, help="Name of the LLM model to use")
    parser.add_argument("--series", type=str, help="Name of the LLM model to use", default=None)
    parser.add_argument("--system_prompt", type=str, help="System prompt to add", default="")
    parser.add_argument("--mode",
                        choices=["baseline", "image", "text"],
                        default="text",
                        help="Evaluation mode (default: baseline)")
    parser.add_argument("--resize", type=float, help="Scale factor to resize input images (e.g., 0.5 for 50%)", default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()
    llm = get_llm(args.model_name, args.series)
    print(f"Evaluating FreeForm...")
    total = eval_dataset(llm, mode=args.mode, system_prompt=args.system_prompt, resize=args.resize)

if __name__ == "__main__":
    main()