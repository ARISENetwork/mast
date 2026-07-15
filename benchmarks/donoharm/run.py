#!/usr/bin/env python3
"""
DonoHarm Benchmark Runner

Usage:
    python run.py --model-config PATH --benchmark-config PATH [--limit N] [--threads N]

Arguments:
    --model-config      Path to model YAML (e.g., config/models/gpt-4o.yaml)
    --benchmark-config  Path to benchmark YAML (e.g., config/benchmarks/donoharm.yaml)
    --limit             Optional: limit number of items for testing
    --threads           Optional: number of parallel threads (default: 20)
"""

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm
import litellm
litellm.suppress_debug_info = True
from litellm import completion, completion_cost
from dotenv import load_dotenv

load_dotenv(override=True)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import create_loader


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_config(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def read_existing_ids(output_file: Path) -> set:
    completed = set()
    if output_file.exists():
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        response = obj.get("response")
                        if "error" not in obj and response is not None:
                            item_id = obj.get("id")
                            trial = obj.get("trial", 1)
                            completed.add((item_id, trial))
                    except json.JSONDecodeError:
                        continue
    return completed


def cleanup_duplicates(output_file: Path) -> int:
    if not output_file.exists():
        return 0

    entries = {}
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    item_id = obj.get("id")
                    trial = obj.get("trial", 1)
                    if item_id:
                        entries[(item_id, trial)] = line
                except json.JSONDecodeError:
                    continue

    with open(output_file, 'r') as f:
        original_count = sum(1 for line in f if line.strip())

    with open(output_file, 'w') as f:
        for line in entries.values():
            f.write(line)

    return original_count - len(entries)


class LiteLLMWrapper:
    def __init__(self, config: dict):
        model_config = config.get("model", {})
        bench_config = config.get("benchmark", {})

        self.platform = model_config.get("platform", "").lower()
        if "model_id" not in model_config:
            raise ValueError("model_id is required in model config")
        model_id = model_config["model_id"]

        if self.platform == "openrouter" and not model_id.startswith("openrouter/"):
            self.model = f"openrouter/{model_id}"
        elif self.platform == "local":
            self.model = f"openai/{model_id}"
        elif self.platform == "azure" and not model_id.startswith("azure/"):
            # LiteLLM Azure routing: model="azure/<deployment_name>".
            # Reads AZURE_API_BASE / AZURE_API_KEY / AZURE_API_VERSION from env
            # unless the YAML provides explicit api_base / api_key below.
            self.model = f"azure/{model_id}"
        else:
            self.model = model_id

        self.api_base = model_config.get("api_base")
        self.api_key = model_config.get("api_key")
        self.temperature = bench_config.get("temperature", 0.0)
        self.max_tokens = bench_config.get("max_tokens", 4096)
        self.completion_kwargs = model_config.get("completion_kwargs", {}) or {}

    def generate(self, prompt: str) -> tuple[str, dict]:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        kwargs.update(self.completion_kwargs)
        if "reasoning_effort" in self.completion_kwargs or "thinking" in self.completion_kwargs:
            kwargs["temperature"] = 1
        # A YAML completion_kwargs key set to null strips the parameter entirely
        # (e.g. for models that reject `temperature` as deprecated).
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        response = completion(**kwargs, timeout=300)
        content = response.choices[0].message.content
        try:
            cost = completion_cost(completion_response=response)
        except Exception:
            cost = None
        usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
        }
        usage["cost"] = cost
        return content, usage


def main():
    parser = argparse.ArgumentParser(description="Run DonoHarm benchmark")
    parser.add_argument("--model-config", required=True, help="Path to model config YAML")
    parser.add_argument("--benchmark-config", required=True, help="Path to benchmark config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items")
    parser.add_argument("--threads", type=int, default=20, help="Number of parallel threads")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt variant name (from prompts/)")
    parser.add_argument("--k", type=int, default=11, choices=range(1, 12),
                        help="Variants per case (1-11). 1=base only, "
                             "5=base + 4 perturbations, 11=all (default).")
    args = parser.parse_args()

    if args.threads is not None and (args.threads < 1 or args.threads > 200):
        parser.error("--threads must be between 1 and 200")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    model_config = load_yaml(args.model_config)
    bench_config = load_yaml(args.benchmark_config)
    final_config = merge_config(model_config, bench_config)

    if args.limit is not None:
        if "benchmark" not in final_config:
            final_config["benchmark"] = {}
        final_config["benchmark"]["limit"] = args.limit

    prompt_name = args.prompt or final_config.get("donoharm", {}).get("prompt", "default")

    benchmark_name = bench_config["benchmark"]["name"]
    model_name = model_config["model"]["name"]

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    # All prompts live under a subdirectory named after the prompt; "default"
    # has no special-case treatment so callers and tools never need to branch.
    output_dir = repo_root / "results" / "raw" / benchmark_name / prompt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}.jsonl"

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    key_hint = f"...{api_key[-4:]}" if len(api_key) >= 4 else "(not set)"

    print(f"DonoHarm Benchmark Runner")
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt_name}")
    print(f"API Key: {key_hint}")
    print(f"Output: {output_file}")
    print("-" * 60)

    completed_ids = read_existing_ids(output_file)
    if completed_ids:
        print(f"Resuming: Skipping {len(completed_ids)} completed items")

    loader = create_loader(final_config, prompt_name=prompt_name)
    platform = (final_config.get("model", {}).get("platform") or "").lower()
    model = LiteLLMWrapper(final_config)
    print(f"Provider: litellm ({platform or 'auto'})")

    limit = final_config.get("benchmark", {}).get("limit")
    items = list(loader.load_items(limit=limit))
    if args.k is not None:
        # Keep k variants per case: 1 base + (k - 1) perturbations.
        from collections import defaultdict
        perturb_count: dict[str, int] = defaultdict(int)
        filtered = []
        for item in items:
            if "-" not in item.id:
                filtered.append(item)
            else:
                base = item.id.rsplit("-", 1)[0]
                if perturb_count[base] < args.k - 1:
                    filtered.append(item)
                    perturb_count[base] += 1
        items = filtered
        print(f"k={args.k}: {len(items)} items "
              f"(base + {args.k - 1} perturbations per case)")

    # Breadth-first ordering: every base case first, then the 1st perturbation
    # of each, then the 2nd, etc. An early stop then yields complete base-case
    # coverage (the headline F1_weighted signal) rather than a handful of
    # fully-perturbed cases. Order-only; results are per-item independent.
    def _perturbation_order(item):
        if "-" not in item.id:
            return (0, item.id)
        base, suffix = item.id.rsplit("-", 1)
        level = int(suffix) + 1 if suffix.isdigit() else 999
        return (level, base)
    items.sort(key=_perturbation_order)

    trials = final_config.get("benchmark", {}).get("trials", 1)

    all_tasks = [(item, trial) for item in items for trial in range(1, trials + 1)]
    tasks_to_process = [(item, trial) for item, trial in all_tasks
                        if (item.id, trial) not in completed_ids]

    print(f"Total tasks: {len(all_tasks)} ({len(items)} items x {trials} trials)")
    print(f"To process: {len(tasks_to_process)}")
    print(f"Using {args.threads} threads")

    write_lock = threading.Lock()

    def process_task(item, trial):
        try:
            start_time = time.time()
            response, usage = model.generate(item.prompt)
            runtime = int(round(time.time() - start_time))

            return {
                "id": item.id,
                "trial": trial,
                "response": response,
                "runtime": runtime,
                "usage": usage,
            }
        except Exception as e:
            return {
                "id": item.id,
                "trial": trial,
                "response": None,
                "error": str(e),
                "runtime": None,
                "usage": None,
            }

    def write_result(result):
        with write_lock:
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_task, item, trial): (item, trial)
                   for item, trial in tasks_to_process}

        with tqdm(total=len(tasks_to_process), desc=f"Running {benchmark_name}") as pbar:
            for future in as_completed(futures):
                result = future.result()
                write_result(result)
                pbar.update(1)

    # Retry errors
    max_retries = 3
    successful_pairs = read_existing_ids(output_file)
    initial_errors = len(all_tasks) - len(successful_pairs)

    if initial_errors == 0:
        print("\nNo errors detected, skipping retry phase")
    else:
        print(f"\nChecking for errors: {initial_errors} tasks need retry")
        for retry_num in range(1, max_retries + 1):
            successful_pairs = read_existing_ids(output_file)
            tasks_with_errors = [(item, trial) for item, trial in all_tasks
                                 if (item.id, trial) not in successful_pairs]
            if not tasks_with_errors:
                print(f"All errors resolved after {retry_num - 1} retries")
                break
            print(f"\nRetry {retry_num}/{max_retries}: {len(tasks_with_errors)} tasks with errors")
            with ThreadPoolExecutor(max_workers=args.threads) as executor:
                futures = {executor.submit(process_task, item, trial): (item, trial)
                           for item, trial in tasks_with_errors}
                with tqdm(total=len(tasks_with_errors), desc=f"Retry {retry_num}") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        write_result(result)
                        if result.get("error"):
                            tqdm.write(f"Error on {result['id']} trial {result['trial']}: {result['error']}")
                        pbar.update(1)

    # Cleanup
    duplicates_removed = cleanup_duplicates(output_file)
    if duplicates_removed > 0:
        print(f"Cleaned up {duplicates_removed} old error entries")

    final_successful_pairs = read_existing_ids(output_file)
    final_errors = len(all_tasks) - len(final_successful_pairs)
    if final_errors > 0:
        print(f"Warning: {final_errors} tasks still have errors after {max_retries} retries")

    runtimes = []
    total_input = 0
    total_output = 0
    total_reasoning = 0
    total_cost = 0.0
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    if obj.get("response") is not None and obj.get("runtime") is not None:
                        runtimes.append(obj["runtime"])
                        u = obj.get("usage") or {}
                        if u.get("prompt_tokens") is not None:
                            total_input += u["prompt_tokens"]
                        if u.get("completion_tokens") is not None:
                            total_output += u["completion_tokens"]
                        details = u.get("completion_tokens_details") or {}
                        if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
                            total_reasoning += details["reasoning_tokens"]
                        if u.get("cost") is not None:
                            total_cost += u["cost"]
                except json.JSONDecodeError:
                    continue
    if runtimes:
        avg_runtime = sum(runtimes) / len(runtimes)
        total_tokens = total_input + total_output
        print(f"\nSummary ({len(runtimes)} successful tasks):")
        print(f"  Avg runtime: {avg_runtime:.1f}s")
        print(f"  Total tokens: {total_tokens:,} (input: {total_input:,}, output: {total_output:,}, reasoning: {total_reasoning:,})")
        print(f"  Total cost: ${total_cost:.2f}")

    print(f"\nDone. Output: {output_file}")


if __name__ == "__main__":
    main()
