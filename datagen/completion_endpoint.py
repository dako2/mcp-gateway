"""Completion endpoint for datagen pipeline. Supports xAI Grok, OpenAI, and OpenRouter APIs."""

import os
import argparse
import copy
import json
from time import time
from tqdm import tqdm
from openai import OpenAI

from utils import load_dataset_from_file, save_dataset, safe_save_checkpoint, get_model_abbreviation


def get_args():
    parser = argparse.ArgumentParser(description="Response Generation Manager.")
    parser.add_argument("--model_path", type=str, default="grok-4-1-fast-reasoning")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of samples per batch")
    parser.add_argument("--checkpoint_every", type=int, default=10, help="Save checkpoint every n batches")
    parser.add_argument("--openai_api_key", type=str, default="", help="OpenAI API Key")
    parser.add_argument("--openrouter_url", type=str, default="https://openrouter.ai/api/v1", help="OpenRouter API URL")
    parser.add_argument("--openrouter_api_key", type=str, default="", help="OpenRouter API Key")
    parser.add_argument("--xai_api_key", type=str, default="", help="xAI API Key (for Grok models)")

    parser.add_argument('--engine', default="xai", type=str, choices=["xai", "openai", "openrouter_api"])
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--step", type=str, default="unknown", help="Processing step identifier.")

    return parser.parse_args()


args = get_args()
print(f"Response Generation Manager. Arguments: {args}")

if args.input_file is None:
    raise ValueError("Please specify the input file path.")

if not args.input_file.endswith("prepared.jsonl") and not args.input_file.endswith("prepared.json"):
    print("Error: Input file must end with prepared.json(l) for completion pipeline.")
    exit(1)

MODEL_NAME = args.model_path
INPUT_FILE_NAME = args.input_file
BATCH_SIZE = args.batch_size
CHECKPOINT_EVERY = args.checkpoint_every

model_abbreviation = get_model_abbreviation(args.model_path)

base_name = INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]
if base_name.endswith("_4prepared"):
    base_name = base_name[:-10]
elif base_name.endswith("_prepared"):
    base_name = base_name[:-9]

if args.num_trials > 1:
    checkpoint_files = [f"{base_name}_{model_abbreviation}_results{i}_checkpoint.json" for i in range(args.num_trials)]
    saved_files = [f"{base_name}_{model_abbreviation}_results{i}.jsonl" for i in range(args.num_trials)]
else:
    checkpoint_file = f"{base_name}_{model_abbreviation}_results_checkpoint.json"
    saved_file = f"{base_name}_{model_abbreviation}_results.jsonl"


def process_batch_openai(batch, client):
    for item in batch:
        message = item["messages"]
        try:
            completion = client.chat.completions.create(
                model=args.model_path,
                messages=message,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p
            )
            response = completion.choices[0].message.content
            item['messages'] = message + [{"role": "assistant", "content": response}]
        except Exception as e:
            print(f"Failed to process item with error: {str(e)}")
            item['messages'] = message + [{"role": "assistant", "content": ""}]
    return batch


def add_generation_config_to_metadata(dataset, model_abbreviation, generation_params):
    config_entry = {
        "model": model_abbreviation,
        "generation_params": generation_params,
        "timestamp": int(time())
    }
    for item in dataset:
        if "metadata" not in item:
            item["metadata"] = {}
        if "synthetic_data_gen_configs" not in item["metadata"]:
            item["metadata"]["synthetic_data_gen_configs"] = []
        item["metadata"]["synthetic_data_gen_configs"].append(config_entry)
    return dataset


def generate_and_update(dataset, checkpoint_file, llm=None):
    processed_dataset = copy.deepcopy(dataset)

    generation_params = {
        "engine": args.engine,
        "model_path": args.model_path,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "num_trials": args.num_trials,
        "step": args.step
    }

    if os.path.exists(checkpoint_file):
        last_checkpoint_idx = len(load_dataset_from_file(checkpoint_file))
        print(f"Checkpoint file found. Resuming from index {last_checkpoint_idx}.")
        processed_dataset[:last_checkpoint_idx] = load_dataset_from_file(checkpoint_file)
        num_batches = (len(processed_dataset) - last_checkpoint_idx + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Remaining batches: {num_batches}")
    else:
        last_checkpoint_idx = 0
        num_batches = (len(processed_dataset) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Total batches: {num_batches}")

    for i in tqdm(range(num_batches)):
        start_idx = i * BATCH_SIZE + last_checkpoint_idx
        end_idx = min((i + 1) * BATCH_SIZE + last_checkpoint_idx, len(processed_dataset))
        batch = processed_dataset[start_idx:end_idx]
        batch = process_batch_openai(batch, llm)
        processed_dataset[start_idx:end_idx] = batch

        if i % CHECKPOINT_EVERY == 0:
            safe_save_checkpoint(processed_dataset[:end_idx], checkpoint_file, convert_to_jsonl=False)
            print(f"Checkpoint saved after batch {i + 1}.")

    processed_dataset = add_generation_config_to_metadata(processed_dataset, model_abbreviation, generation_params)
    return processed_dataset


def main():
    dataset = load_dataset_from_file(INPUT_FILE_NAME)
    if not isinstance(dataset, list):
        dataset = [dataset]

    if args.engine == "xai":
        print("Starting xAI Grok engine...")
        xai_api_key = args.xai_api_key or os.getenv("XAI_API_KEY")
        if not xai_api_key:
            raise ValueError("xAI API Key not provided. Set XAI_API_KEY env var or --xai_api_key.")
        llm = OpenAI(api_key=xai_api_key, base_url="https://api.x.ai/v1")
    elif args.engine == "openai":
        print("Starting OpenAI engine...")
        openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API Key not provided. Set OPENAI_API_KEY env var or --openai_api_key.")
        llm = OpenAI(api_key=openai_api_key)
    elif args.engine == "openrouter_api":
        print("Starting OpenRouter engine...")
        openrouter_api_key = args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OpenRouter API Key not provided. Set OPENROUTER_API_KEY env var or --openrouter_api_key.")
        llm = OpenAI(api_key=openrouter_api_key, base_url=args.openrouter_url)
    else:
        raise ValueError(f"Invalid engine: {args.engine}")

    if args.num_trials == 1:
        updated_dataset = generate_and_update(dataset, checkpoint_file, llm)
        save_dataset(updated_dataset, saved_file, convert_to_jsonl=True)
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        print("Final dataset saved. Checkpoint removed.")
    else:
        for i in range(args.num_trials):
            updated_dataset = generate_and_update(dataset, checkpoint_files[i], llm)
            save_dataset(updated_dataset, saved_files[i], convert_to_jsonl=True)
            if os.path.exists(checkpoint_files[i]):
                os.remove(checkpoint_files[i])
            print(f"Dataset for trial {i} saved. Checkpoint removed.")


if __name__ == "__main__":
    main()
