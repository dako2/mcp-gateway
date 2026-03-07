#!/bin/bash
# Step 4.2: Response quality check using LLM
# Usage: bash step4.2_completion_response_check.sh <input_file> [model] [engine]

input_file=${1}
model_path=${2:-"grok-4-1-fast-reasoning"}
engine=${3:-"xai"}
step=${4:-"4.2"}

if [ -z "$input_file" ]; then
    echo "Usage: bash step4.2_completion_response_check.sh <input_file> [model] [engine]"
    exit 1
fi

python completion_endpoint.py --input_file ${input_file} --model_path ${model_path} --engine ${engine} --step ${step} --max_tokens 8192
