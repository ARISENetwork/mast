#!/usr/bin/env python3
"""
SCT Benchmark Runner

Usage:
    python run.py --model-config PATH --benchmark-config PATH [--limit N] [--threads N]

Arguments:
    --model-config      Path to model YAML (e.g., config/models/gpt-4o.yaml)
    --benchmark-config  Path to benchmark YAML (e.g., config/benchmarks/sct.yaml)
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
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm
import litellm
litellm.suppress_debug_info = True
from litellm import completion, completion_cost
from dotenv import load_dotenv

# Load environment variables from .env file (override shell env)
load_dotenv(override=True)

# Local imports: data_loader lives alongside this file.
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import create_loader


def load_yaml(path: str) -> dict:
    """Load YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_config(base: dict, override: dict) -> dict:
    """Recursively merge configs (override takes precedence)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def read_existing_ids(output_file: Path) -> set:
    """Read successfully completed (id, trial) pairs from output file.

    Only includes items that completed without error AND have a valid response.
    This allows errored items to be retried on rerun.
    """
    completed = set()
    if output_file.exists():
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        response = obj.get("response")
                        # Only count as completed if no error field AND the
                        # response is non-empty. Blank/whitespace responses are
                        # re-attempted, not accepted.
                        has_nonempty_response = response is not None and (
                            not isinstance(response, str) or bool(response.strip())
                        )
                        if "error" not in obj and has_nonempty_response:
                            item_id = obj.get("id")
                            trial = obj.get("trial", 1)
                            completed.add((item_id, trial))
                    except json.JSONDecodeError:
                        continue
    return completed


def cleanup_duplicates(output_file: Path) -> int:
    """Remove duplicate entries, keeping only the last entry per (id, trial) pair.

    This cleans up old error entries after they've been successfully retried.
    Returns the number of duplicates removed.
    """
    if not output_file.exists():
        return 0

    # Read all entries, keeping track of last entry per (id, trial)
    entries = {}
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    item_id = obj.get("id")
                    trial = obj.get("trial", 1)
                    if item_id:
                        entries[(item_id, trial)] = line  # Later entries overwrite earlier ones
                except json.JSONDecodeError:
                    continue

    # Count original lines
    with open(output_file, 'r') as f:
        original_count = sum(1 for line in f if line.strip())

    # Write deduplicated entries
    with open(output_file, 'w') as f:
        for line in entries.values():
            f.write(line)

    return original_count - len(entries)


class LiteLLMWrapper:
    """Wrapper for litellm completion."""
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
        else:
            self.model = model_id

        self.api_base = model_config.get("api_base")
        self.api_key = model_config.get("api_key")
        self.request_timeout = model_config.get(
            "request_timeout",
            bench_config.get("request_timeout", 300),
        )

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
        # (e.g. for reasoning models that reject a non-default temperature).
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        response = completion(**kwargs, timeout=self.request_timeout)
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
    parser = argparse.ArgumentParser(description="Run SCT benchmark")
    parser.add_argument("--model-config", required=True, help="Path to model config YAML")
    parser.add_argument("--benchmark-config", required=True, help="Path to benchmark config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items")
    parser.add_argument("--threads", type=int, default=20, help="Number of parallel threads")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1. CONFIGURATION
    # -------------------------------------------------------------------------
    model_config = load_yaml(args.model_config)
    bench_config = load_yaml(args.benchmark_config)
    final_config = merge_config(model_config, bench_config)

    if args.limit is not None:
        if "benchmark" not in final_config:
            final_config["benchmark"] = {}
        final_config["benchmark"]["limit"] = args.limit

    # -------------------------------------------------------------------------
    # 2. OUTPUT PATH
    # -------------------------------------------------------------------------
    if "benchmark" not in bench_config or "name" not in bench_config["benchmark"]:
        raise ValueError("Benchmark name is required in benchmark config")
    benchmark_name = bench_config["benchmark"]["name"]

    if "model" not in model_config or "name" not in model_config["model"]:
        raise ValueError("Model name is required in model config")
    model_name = model_config["model"]["name"]

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    # Standardized output path: results/raw/{benchmark}/{model}.jsonl
    output_dir = repo_root / "results" / "raw" / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}.jsonl"

    # Show API key hint for debugging
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    key_hint = f"...{api_key[-4:]}" if len(api_key) >= 4 else "(not set)"

    print(f"SCT Benchmark Runner")
    print(f"Model: {model_name}")
    print(f"API Key: {key_hint}")
    print(f"Output: {output_file}")
    print("-" * 60)

    # -------------------------------------------------------------------------
    # 3. RESUME
    # -------------------------------------------------------------------------
    completed_ids = read_existing_ids(output_file)
    if completed_ids:
        print(f"Resuming: Skipping {len(completed_ids)} completed items")

    # -------------------------------------------------------------------------
    # 4. INIT
    # -------------------------------------------------------------------------
    loader = create_loader(final_config)
    platform = (final_config.get("model", {}).get("platform") or "").lower()
    model = LiteLLMWrapper(final_config)
    print(f"Provider: litellm ({platform or 'auto'})")

    # Get limit and subset from config
    limit = final_config.get("benchmark", {}).get("limit")
    subset = final_config.get("benchmark", {}).get("subset", "all")

    # -------------------------------------------------------------------------
    # 5. LOOP (PARALLEL with TRIALS)
    # -------------------------------------------------------------------------
    items = list(loader.load_items(subset=subset, limit=limit))
    trials = final_config.get("benchmark", {}).get("trials", 1)

    # Expand items into (item, trial) tasks
    all_tasks = [(item, trial) for item in items for trial in range(1, trials + 1)]

    # Filter out already completed (id, trial) pairs
    tasks_to_process = [(item, trial) for item, trial in all_tasks
                        if (item.id, trial) not in completed_ids]

    print(f"Total tasks: {len(all_tasks)} ({len(items)} items × {trials} trials)")
    print(f"To process: {len(tasks_to_process)}")
    print(f"Using {args.threads} threads")

    # Thread-safe file writing
    write_lock = threading.Lock()

    def process_task(item, trial):
        """Process a single benchmark task (item + trial). Returns the result dict."""
        try:
            start_time = time.time()
            response, usage = model.generate(item.prompt)
            if isinstance(response, str) and not response.strip():
                raise ValueError("Model returned an empty/whitespace response")
            end_time = time.time()
            runtime = int(round(end_time - start_time))

            return {
                "id": item.id,
                "trial": trial,
                "response": response,
                "runtime": runtime,
                "metadata": item.metadata,
                "usage": usage,
            }

        except Exception as e:
            return {
                "id": item.id,
                "trial": trial,
                "response": None,
                "error": str(e),
                "runtime": None,
                "metadata": item.metadata,
                "usage": None,
            }

    def write_result(result):
        """Write a result to the output file (thread-safe)."""
        with write_lock:
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()

    # Process tasks in parallel
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_task, item, trial): (item, trial)
                   for item, trial in tasks_to_process}

        with tqdm(total=len(tasks_to_process), desc=f"Running {benchmark_name}") as pbar:
            for future in as_completed(futures):
                item, trial = futures[future]
                result = future.result()
                write_result(result)

                if result.get("error"):
                    tqdm.write(f"Error on {item.id} trial {trial}: {result['error']}")

                pbar.update(1)

    # -------------------------------------------------------------------------
    # 6. RETRY ERRORS (up to 3 times)
    # -------------------------------------------------------------------------
    max_retries = 3

    # Initial error check
    successful_pairs = read_existing_ids(output_file)
    initial_errors = len(all_tasks) - len(successful_pairs)

    if initial_errors == 0:
        print("\nNo errors detected, skipping retry phase")
    else:
        print(f"\nChecking for errors: {initial_errors} tasks need retry")

        for retry_num in range(1, max_retries + 1):
            # Check which (id, trial) pairs still have errors
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
                        item, trial = futures[future]
                        result = future.result()
                        write_result(result)

                        if result.get("error"):
                            tqdm.write(f"Error on {item.id} trial {trial}: {result['error']}")

                        pbar.update(1)

    # -------------------------------------------------------------------------
    # 7. CLEANUP
    # -------------------------------------------------------------------------
    duplicates_removed = cleanup_duplicates(output_file)
    if duplicates_removed > 0:
        print(f"Cleaned up {duplicates_removed} old error entries")

    # Report final error count
    final_successful_pairs = read_existing_ids(output_file)
    final_errors = len(all_tasks) - len(final_successful_pairs)
    if final_errors > 0:
        print(f"Warning: {final_errors} tasks still have errors after {max_retries} retries")

    # Calculate summary stats from successful responses
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
                        usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else obj
                        if usage.get("prompt_tokens") is not None:
                            total_input += usage["prompt_tokens"]
                        if usage.get("completion_tokens") is not None:
                            total_output += usage["completion_tokens"]
                        details = usage.get("completion_tokens_details")
                        rt = usage.get("reasoning_tokens") or (details.get("reasoning_tokens") if isinstance(details, dict) else None)
                        if rt is not None:
                            total_reasoning += rt
                        if usage.get("cost") is not None:
                            total_cost += usage["cost"]
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
