#!/bin/bash
# Run Toucan datagen pipeline with xAI Grok models
#
# Usage:
#   # Step 1.2: Generate questions
#   bash run_with_grok.sh completion <input_file>
#
#   # Step 2.2: Question quality check
#   bash run_with_grok.sh quality_check <input_file>
#
#   # Step 3.1: Generate agent traces
#   bash run_with_grok.sh trace_gen <input_file>
#
#   # Step 4.2: Response quality check
#   bash run_with_grok.sh response_check <input_file>
#
# Environment:
#   XAI_API_KEY must be set in ../.env or exported

set -e

# Load .env
if [ -f "../.env" ]; then
    export $(grep -v '^#' ../.env | xargs)
fi

if [ -z "$XAI_API_KEY" ]; then
    echo "ERROR: XAI_API_KEY not set. Add it to ../.env"
    exit 1
fi

MODE=${1:-"completion"}
INPUT_FILE=${2}
MODEL=${3:-"grok-4-1-fast-reasoning"}
MAX_WORKERS=${4:-8}

if [ -z "$INPUT_FILE" ]; then
    echo "Usage: bash run_with_grok.sh <mode> <input_file> [model] [max_workers]"
    echo ""
    echo "Modes:"
    echo "  completion      - Generate completions (Steps 1.2, 2.2, 4.2)"
    echo "  trace_gen       - Generate agent traces (Step 3.1)"
    echo ""
    echo "Models:"
    echo "  grok-4-1-fast-reasoning     (\$0.20/\$0.50 per 1M tokens, default)"
    echo "  grok-4-1-fast-non-reasoning (\$0.20/\$0.50 per 1M tokens)"
    echo "  grok-4-0709                 (\$3/\$15 per 1M tokens, most capable)"
    exit 1
fi

echo "============================================"
echo "  Toucan DataGen with xAI Grok"
echo "============================================"
echo "  Mode:        $MODE"
echo "  Input:       $INPUT_FILE"
echo "  Model:       $MODEL"
echo "  Max Workers: $MAX_WORKERS"
echo "============================================"

case $MODE in
    completion)
        python completion_endpoint.py \
            --input_file "$INPUT_FILE" \
            --model_path "$MODEL" \
            --engine xai \
            --temperature 0.7 \
            --max_tokens 4096
        ;;
    trace_gen)
        python completion_openai_agent.py \
            --input_file "$INPUT_FILE" \
            --model_path "$MODEL" \
            --engine xai \
            --agent openai_agent \
            --max_workers "$MAX_WORKERS" \
            --temperature 0.7 \
            --max_tokens 4096 \
            --max_turns 10 \
            --timeout 120 \
            --reasoning_effort high \
            --parallel_function_calls True
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Use 'completion' or 'trace_gen'"
        exit 1
        ;;
esac

echo ""
echo "Done!"
