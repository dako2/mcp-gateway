#!/bin/bash
# Step 3.1: Generate agent trajectories using LLM + MCP tools
#
# Gateway mode (default):
#   bash step3.1_completion_agent.sh <input_file> [model] [engine] [step] [max_workers]
#   Requires: gateway running at localhost:8000 and data/server_url_map.json
#   To prepare:  cd src && python resolve_urls.py
#                cd src && uvicorn gateway:app --port 8000
#
# Legacy Smithery mode:
#   USE_GATEWAY=0 bash step3.1_completion_agent.sh <input_file> ...

input_file=${1}
model_path=${2:-"grok-4-1-fast-reasoning"}
engine=${3:-"xai"}
step=${4:-"3.1"}
max_workers=${5:-8}
use_gateway=${USE_GATEWAY:-1}
gateway_url=${GATEWAY_URL:-"http://localhost:8000"}

if [ -z "$input_file" ]; then
    echo "Usage: bash step3.1_completion_agent.sh <input_file> [model] [engine] [step] [max_workers]"
    exit 1
fi

gateway_args=""
if [ "$use_gateway" = "1" ]; then
    # Verify gateway is reachable
    if ! curl -sf "${gateway_url}/health" > /dev/null 2>&1; then
        echo "ERROR: Gateway not running at ${gateway_url}"
        echo "Start it first:  cd src && uvicorn gateway:app --port 8000"
        exit 1
    fi
    echo "Gateway OK at ${gateway_url}"
    gateway_args="--use_gateway --gateway_url ${gateway_url}"
fi

python completion_openai_agent.py \
    --input_file ${input_file} \
    --model_path ${model_path} \
    --engine ${engine} \
    --step ${step} \
    --agent openai_agent \
    --max_workers ${max_workers} \
    --enable_tool_hint \
    ${gateway_args}
