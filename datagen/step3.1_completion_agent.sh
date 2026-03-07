#!/bin/bash
# Step 3.1: Generate agent trajectories using LLM + MCP tools
# Usage: bash step3.1_completion_agent.sh <input_file> [model] [engine] [max_workers]

input_file=${1}
model_path=${2:-"grok-4-1-fast-reasoning"}
engine=${3:-"xai"}
step=${4:-"3.1"}
max_workers=${5:-8}

if [ -z "$input_file" ]; then
    echo "Usage: bash step3.1_completion_agent.sh <input_file> [model] [engine] [max_workers]"
    exit 1
fi

python completion_openai_agent.py \
    --input_file ${input_file} \
    --model_path ${model_path} \
    --engine ${engine} \
    --step ${step} \
    --agent openai_agent \
    --max_workers ${max_workers} \
    --enable_tool_hint
