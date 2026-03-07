#!/bin/bash
# Step 2.2: Question quality check using LLM
# Usage: bash step2.2_completion_quality_check.sh <input_file> [model] [engine]

input_file=${1}
model_path=${2:-"grok-4-1-fast-reasoning"}
engine=${3:-"xai"}
step=${4:-"2.2"}

if [ -z "$input_file" ]; then
    echo "Usage: bash step2.2_completion_quality_check.sh <input_file> [model] [engine]"
    exit 1
fi

python completion_endpoint.py --input_file ${input_file} --model_path ${model_path} --engine ${engine} --step ${step}
