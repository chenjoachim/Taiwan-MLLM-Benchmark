export GCP_PROJECT_NAME="GCP VERTEX PROJECT name"
export OAI_KEY="OPEN AI KEY"
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
        python -m simplevals.eval gemini-1.5-flash --datasets $subject --mode $mode
        python -m simplevals.eval_da gemini-1.5-flash --datasets $subject --mode $mode
    done
    python -m simplevals.eval gemini-1.5-flash --datasets $subject --mode "text" --text_only
    python -m simplevals.eval_da gemini-1.5-flash --datasets $subject --mode "text" --text_only
done