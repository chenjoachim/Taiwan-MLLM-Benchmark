import re
import os
import json

VALID_DATASETS = [
    'accounting', 'arts', 'biology', 'chemistry', 'chinese_literature', 'dentistry', 
    'electronic_circuits', 'fundamentals_of_physical_therapy', 'geography', 'mathematics', 
    'mechanics', 'medical', 'music', 'natural_science', 'navigation', 'pharmaceutical_chemistry', 
    'physics', 'sociology', 'statistics', 'structural_engineering', 'veterinary_medicine'
]

MMMU_DATASETS = [
   "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology"
]
def load_existing_entries(logging_file):
    full_path = os.path.join('execution_results', logging_file)
    print(full_path)

    stats = {'total': 0, 'hit': 0, 'correct': 0, 'negative_markings': 0, 'existing_entries': {}}
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if 'question_id' in entry:
                    stats['existing_entries'][entry['question_id']] = entry
                    stats['total'] += 1
                    stats['hit'] += int(entry['current_score'].split('/')[0])
                    if 'correct_answers' in entry:
                        stats['correct'] += entry['correct_answers']
                        stats['negative_markings'] += entry['negative_markings']
                    elif entry.get('is_correct', False):
                        stats['correct'] += 1
    return stats


def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """

    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )

def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .replace("：", ":")
        .strip()
    )


parser_prompt = """Extract the following ANSWER final answer, your answer should be the one which match any of these valid choices:
{valid_choice}

ANSWER:
{response}

Only response in one letter : A,B,C,D,E,F,G. You must only output one letter choice and nothing else.
RESPONSE:"""

answer_comparison = """Determine if the [Response] and [Ground Truth] are choosing the same answer choice or not

[Response]
```
{response}
```

[Ground Truth]
```
Answer choice : {valid_choice}
```

Only response in True/False, you should only output True if the selected choice in [Response] and [Ground Truth] is the same
RESPONSE:"""

def fallback_llm_as_parser(llm, response, valid_choice):
    prompt = parser_prompt.format(valid_choice=valid_choice, response=response)
    res_text, _ = llm(prompt)
    return res_text

def is_both_same(llm, response, valid_choice):
    prompt = answer_comparison.format(valid_choice=valid_choice, response=response)
    print(prompt)
    res_text, _ = llm(prompt)
    return res_text.strip().lower()


def ensure_logging_dir():
    if not os.path.exists('execution_results'):
        os.makedirs('execution_results')

def write_jsonl(filename, data):
    ensure_logging_dir()
    full_path = os.path.join('execution_results', filename)
    with open(full_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')

def write_final_log(logging_file, stats):
    final_log = {"final_score": f"{stats['hit']}/{stats['total']}"}
    if 'negative_markings' in stats:
        final_log.update({
            "correct_answers": f"{stats['correct']}/{stats['total']}",
            "negative_markings": f"{stats['negative_markings']}/{stats['total']}"
        })
    write_jsonl(logging_file, final_log)


def evaluate_neg_mark(log_entry, extracted_answer, stats):
    if extracted_answer == 'E':
        stats['negative_markings'] += 1
        log_entry["result"] = "No answer"
    elif extracted_answer == log_entry["ground_truth"]:
        stats['hit'] += 1
        stats['correct'] += 1
        log_entry["result"] = "Correct"
    else:
        stats['hit'] -= 1
        log_entry["result"] = "Incorrect"

def evaluate_regular(log_entry, extracted_answer, stats):
    if extracted_answer == log_entry["ground_truth"]:
        stats['hit'] += 1
        stats['correct'] += 1
        log_entry["is_correct"] = True
    else:
        log_entry["is_correct"] = False

def evaluate_response(log_entry, mode, stats, sampler):

    if isinstance(log_entry['ground_truth'], list):
        match = re.search(r"答案\s*:\s*(\w+)", log_entry["normalized_response"])
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1))
        else:
            extracted_answer = log_entry['normalized_response']

        if 'true' in is_both_same(sampler, extracted_answer, ",".join(log_entry['ground_truth'])):
            stats['hit'] += 1
            stats['correct'] += 1
            log_entry["is_correct"] = True
        else:
            log_entry["is_correct"] = False
        stats['total'] += 1
        log_entry["current_score"] = f"{stats['hit']}/{stats['total']}"
        return None
    match = re.search(r"答案\s*:\s*(\w+)", log_entry["normalized_response"])
    if match:
        extracted_answer = normalize_extracted_answer(match.group(1))
        log_entry["extracted_answer"] = extracted_answer
        evaluate_regular(log_entry, extracted_answer, stats)
    else:
        match = re.search(r"Answer\s*:\s*(\w+)", log_entry["normalized_response"])
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1))
            log_entry["extracted_answer"] = extracted_answer
            evaluate_regular(log_entry, extracted_answer, stats)
        else:# use llm to parse out which is the more valid choice
            selected_choice = 'A. '+log_entry['formatted_question'].split('A.')[-1]
            llm_response = log_entry['llm_response']
            extracted_answer = fallback_llm_as_parser(sampler, llm_response, selected_choice)
            evaluate_regular(log_entry, extracted_answer, stats)

    stats['total'] += 1
    log_entry["current_score"] = f"{stats['hit']}/{stats['total']}"

