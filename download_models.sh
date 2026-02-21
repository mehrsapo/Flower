#!/bin/bash
# Downloads pretrained model weights from HuggingFace and places them
# in the exact same structure expected by the project:
#
#   model/afhq_cat/gradient_step/model_final.pt
#   model/afhq_cat/ot/model_final.pt
#   model/afhq_cat/ot/model_final_no_ot.pt
#   model/celeba/gradient_step/model_final.pt
#   model/celeba/ot/model_final.pt
#   model/celeba/ot/model_final_no_ot.pt
#
# Usage:
#   bash download_model.sh
# ---------------------------------------------------------------------------

HF_REPO="mehrsapo/flower-weights"

BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main"

# Project root is the same directory as this script
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FILES=(
    "model/afhq_cat/gradient_step/model_final.pt"
    "model/afhq_cat/ot/model_final.pt"
    "model/afhq_cat/ot/model_final_no_ot.pt"
    "model/celeba/gradient_step/model_final.pt"
    "model/celeba/ot/model_final.pt"
    "model/celeba/ot/model_final_no_ot.pt"
)

for FILE in "${FILES[@]}"; do
    DEST="$ROOT_DIR/$FILE"
    mkdir -p "$(dirname "$DEST")"
    echo "Downloading $FILE ..."
    wget -q --show-progress -O "$DEST" "${BASE_URL}/${FILE}"
done

echo "All weights downloaded successfully."
