#!/bin/bash
export OAI_KEY="OPENAI KEY HERE"

subjects=(
    "accounting"
    "arts"
    "biology"
    "chemistry"
    "chinese_literature"
    "dentistry"
    "electronic_circuits"
    "fundamentals_of_physical_therapy"
    "geography"
    "mathematics"
    "mechanics"
    "medical"
    "music"
    "natural_science"
    "navigation"
    "pharmaceutical_chemistry"
    "physics"
    "sociology"
    "statistics"
    "structural_engineering"
    "veterinary_medicine"
)

modes=(
    "image"
    "text"
)

for subject in "${subjects[@]}"; do
    for mode in "${modes[@]}"; do
        python -m simplevals.eval gpt-4o-mini-2024-07-18 --datasets $subject --mode $mode
        python -m simplevals.eval_da gpt-4o-mini-2024-07-18 --datasets $subject --mode $mode
    done
    python -m simplevals.eval gpt-4o-mini-2024-07-18 --datasets $subject --mode "text" --text_only
    python -m simplevals.eval_da gpt-4o-mini-2024-07-18 --datasets $subject --mode "text" --text_only

done