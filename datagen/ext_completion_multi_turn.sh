#!/bin/bash
# Multi-turn conversation generation
# Usage: bash ext_completion_multi_turn.sh <input_file> [model] [engine] [num_turns] [max_workers]

input_file=${1}
model_path=${2:-"grok-4-1-fast-reasoning"}
engine=${3:-"xai"}
num_desired_turns=${4:-5}
user_simulation_model=${5:-"${model_path}"}
max_workers=${6:-8}

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Multi-Turn Conversation Generation${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Input file: ${input_file}${NC}"
echo -e "${BLUE}Model: ${model_path}${NC}"
echo -e "${BLUE}Engine: ${engine}${NC}"
echo -e "${BLUE}Turns: ${num_desired_turns}${NC}"
echo -e "${BLUE}Max workers: ${max_workers}${NC}"
echo -e "${BLUE}========================================${NC}"

if [ -z "$input_file" ]; then
    echo "Usage: bash ext_completion_multi_turn.sh <input_file> [model] [engine] [num_turns] [max_workers]"
    exit 1
fi

if [ ! -f "$input_file" ]; then
    echo "Error: Input file does not exist: $input_file"
    exit 1
fi

echo -e "${GREEN}Starting Multi-Turn Generation${NC}"

python ext_multi_turn_openai_agent.py \
    --input_file ${input_file} \
    --model_path ${model_path} \
    --user_simulation_model ${user_simulation_model} \
    --engine ${engine} \
    --num_desired_turns ${num_desired_turns} \
    --max_workers ${max_workers} \
    --agent openai_agent \
    --reasoning_effort high \
    --parallel_function_calls True \
    --max_rounds_per_turn 10

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}Multi-Turn Generation Completed!${NC}"
else
    echo "Multi-Turn Generation Failed with exit code: $exit_code"
fi

exit $exit_code
