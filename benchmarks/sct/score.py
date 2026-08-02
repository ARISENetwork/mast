#!/usr/bin/env python3
"""
SCT Benchmark Scorer

Scores SCT responses using expert distribution alignment.
No LLM grader needed - SCT uses deterministic scoring.

Handles multiple trials per question by computing per-trial metrics
and aggregating across trials with mean and 95% CI.

Usage:
    python score.py --model-config PATH --benchmark-config PATH [--limit N]

Output:
    results/scores/{benchmark}/{model}.csv

    Columns: category, metric, trials, mean, ci
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

SCT_RATINGS = [-2, -1, 0, 1, 2]
CONFIDENCE_METRIC_LABELS = {
    "overconfidence_rate": "Overconfidence",
    "underconfidence_rate": "Underconfidence",
    "distractor_susceptibility": "Distractor Susceptibility",
}


def load_yaml(path: str) -> dict:
    """Load YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_response(response: str) -> Optional[int]:
    """
    Extract rating from LLM response text.

    Handles two output shapes: a JSON-style {"Rating": X} field (structured
    outputs, e.g. the submission endpoint format) and the plain-text
    "Rating: X" answer line, where X is -2, -1, 0, +1, or +2.

    Args:
        response: Raw text response from LLM.

    Returns:
        Integer rating in range [-2, 2], or None if parsing fails.
    """
    if not response or not isinstance(response, str):
        return None

    try:
        # Structured JSON rating, e.g. {"Rating": 2} or {"rating": "-1"}. The
        # quote before the colon distinguishes this from the plain-text
        # "Rating: X" heading, so this branch only fires for JSON outputs (the
        # plain-text path below is left byte-identical for every other model).
        json_matches = re.findall(r'"[Rr]ating"\s*:\s*"?([+-]?[012])"?', response)
        if json_matches:
            rating = int(json_matches[-1])
            if rating in [-2, -1, 0, 1, 2]:
                return rating

        # Look for "Rating:" followed by the score. Use the LAST occurrence so
        # chain-of-thought reasoning that mentions "**Rating:**" as a heading
        # earlier in the response doesn't preempt the final answer line.
        idx = response.rfind("Rating:")
        if idx == -1:
            return None

        rating_section = response[idx + len("Rating:"):][:10]
        pattern = r'[+-]?[012]'
        matches = re.findall(pattern, rating_section)

        if not matches:
            return None

        # Take first match
        rating_str = matches[0]
        rating = int(rating_str)

        if rating not in [-2, -1, 0, 1, 2]:
            return None

        return rating

    except Exception:
        return None


# =============================================================================
# SCT SCORING
# =============================================================================

def sct_score(expert_distribution: np.ndarray, responses: np.ndarray) -> float:
    """
    Calculate SCT score measuring alignment with expert consensus.

    The score represents how well the responses match the expert panel's
    distribution of answers. A score of 1.0 means perfect alignment with
    the most common expert answer for each question.

    Args:
        expert_distribution: Array of shape (n_questions, 5) containing
            expert response distributions for ratings [-2, -1, 0, 1, 2].
        responses: Array of shape (n_questions,) containing integer
            responses in range [-2, 2].

    Returns:
        Score between 0.0 and 1.0.
    """
    expert_distribution = np.asarray(expert_distribution, dtype=float)
    responses = np.asarray(responses, dtype=int)

    # Normalize so max score per question is 1
    row_max = np.max(expert_distribution, axis=1, keepdims=True)
    row_max = np.where(row_max == 0, 1, row_max)  # Avoid division by zero
    normalized = expert_distribution / row_max

    # Sum scores: response of -2 -> index 0, ..., +2 -> index 4
    total = 0.0
    for i, response in enumerate(responses):
        idx = response + 2  # Convert [-2,2] to [0,4]
        total += normalized[i, idx]

    return total / len(responses)


def percentage_in_expert_set(expert_distribution: np.ndarray, responses: np.ndarray) -> float:
    """
    Calculate percentage of responses that match any expert answer.

    A more lenient metric - counts a response as correct if at least
    one expert gave that same answer.

    Args:
        expert_distribution: Array of shape (n_questions, 5)
        responses: Array of shape (n_questions,) in range [-2, 2]

    Returns:
        Proportion between 0.0 and 1.0.
    """
    expert_distribution = np.asarray(expert_distribution, dtype=float)
    responses = np.asarray(responses, dtype=int)

    matches = 0
    for i, response in enumerate(responses):
        idx = response + 2
        if expert_distribution[i, idx] > 0:
            matches += 1

    return matches / len(responses)


def score_single_response(expert_distribution: List[float], response: int) -> dict:
    """
    Score a single response against expert distribution.

    Args:
        expert_distribution: List of 5 floats for ratings [-2, -1, 0, 1, 2]
        response: Integer response in range [-2, 2]

    Returns:
        Dictionary with score details.
    """
    expert_distribution = np.asarray(expert_distribution, dtype=float)

    row_max = np.max(expert_distribution)
    if row_max == 0:
        normalized_score = 0.0
    else:
        normalized_score = expert_distribution[response + 2] / row_max

    in_expert_set = expert_distribution[response + 2] > 0

    return {
        "response": response,
        "normalized_score": float(normalized_score),
        "in_expert_set": bool(in_expert_set),
    }


def unique_modal_expert_rating(expert_distribution: List[float]) -> Optional[int]:
    """Return the unique modal expert rating, or None when the mode is tied."""
    expert_distribution = np.asarray(expert_distribution, dtype=float)
    if expert_distribution.size != len(SCT_RATINGS):
        return None

    row_max = np.max(expert_distribution)
    if row_max <= 0:
        return None

    modal_indices = np.flatnonzero(expert_distribution == row_max)
    if len(modal_indices) != 1:
        return None

    return SCT_RATINGS[int(modal_indices[0])]


def confidence_error_indicators(
    expected_rating: Optional[int],
    response: int,
) -> Dict[str, float]:
    """Opportunity-normalized directional errors against the expert mode.

    The metrics are defined only when an item creates the relevant opportunity:
    overconfidence for unique-modal +/-1 items, underconfidence for unique-modal
    +/-2 items, and distractor susceptibility for unique-modal 0 items.
    """
    if expected_rating is None:
        return {}

    if expected_rating == 1:
        return {"overconfidence_rate": 1.0 if response == 2 else 0.0}
    if expected_rating == -1:
        return {"overconfidence_rate": 1.0 if response == -2 else 0.0}
    if expected_rating == 2:
        return {"underconfidence_rate": 1.0 if response == 1 else 0.0}
    if expected_rating == -2:
        return {"underconfidence_rate": 1.0 if response == -1 else 0.0}
    if expected_rating == 0:
        return {"distractor_susceptibility": 1.0 if response != 0 else 0.0}

    return {}


def compute_student_score_from_distribution(expert_dist: List[float], student_dist: List[float]) -> float:
    """
    Compute average student SCT score from their response distribution.

    This computes the expected score if students responded according to
    the student distribution.

    Args:
        expert_dist: Expert distribution [count for -2, -1, 0, +1, +2]
        student_dist: Student distribution [count for -2, -1, 0, +1, +2]

    Returns:
        Average student score (0.0-1.0)
    """
    expert_dist = np.asarray(expert_dist, dtype=float)
    student_dist = np.asarray(student_dist, dtype=float)

    # Normalize expert distribution so max = 1
    row_max = np.max(expert_dist)
    if row_max == 0:
        return 0.0
    normalized_expert = expert_dist / row_max

    # Total students
    total_students = np.sum(student_dist)
    if total_students == 0:
        return 0.0

    # Weighted average score
    total_score = 0.0
    for i in range(5):  # For each rating -2 to +2
        n_students = student_dist[i]
        score_for_rating = normalized_expert[i]
        total_score += n_students * score_for_rating

    return total_score / total_students


def aggregate_metric(values):
    """Compute n_trials, mean, and 95% CI from a list of per-trial values."""
    values = np.array(values, dtype=float)
    n = len(values)
    mean_val = float(np.mean(values))
    if n > 1:
        se = float(np.std(values, ddof=1) / np.sqrt(n))
        ci = round(1.96 * se, 3)
    else:
        ci = ''
    return n, round(mean_val, 3), ci


def bootstrap_ci(per_item: List[float], n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Item-level percentile bootstrap CI for the mean of per_item.

    Returns dict with keys: n, mean, ci, ci_lo, ci_hi (95% percentile interval,
    ci is the half-width). Empty input -> NaNs.
    """
    arr = np.asarray(per_item, dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    mean = float(arr.mean())
    if n < 2:
        return {"n": n, "mean": round(mean, 4), "ci": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return {
        "n": n,
        "mean": round(mean, 4),
        "ci": round((hi - lo) / 2, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Score SCT benchmark results")
    parser.add_argument("--model-config", required=True, help="Path to model config YAML")
    parser.add_argument("--benchmark-config", required=True, help="Path to benchmark config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to score")
    parser.add_argument("--threads", type=int, default=None, help="Ignored (no parallel scoring)")
    parser.add_argument("--trial", type=int, default=1,
                        help="Trial number to score (default 1). SCT now uses item-level bootstrap "
                             "CIs over a single trial; multi-trial averaging is no longer the primary path.")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="Bootstrap iterations for CI (default 2000)")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed (default 0)")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1. SETUP
    # -------------------------------------------------------------------------
    model_config = load_yaml(args.model_config)
    bench_config = load_yaml(args.benchmark_config)

    # -------------------------------------------------------------------------
    # 2. PATH RESOLUTION
    # -------------------------------------------------------------------------
    benchmark_name = bench_config.get("benchmark", {}).get("name", "sct")
    model_name = model_config.get("model", {}).get("name", "unknown")

    # Get repo root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    # Standardized paths: results/raw/{benchmark}/{model}.jsonl
    input_file = repo_root / "results" / "raw" / benchmark_name / f"{model_name}.jsonl"
    output_dir = repo_root / "results" / "scores" / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"SCT Benchmark Scorer")
    print(f"Model: {model_name}")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Run the benchmark first with run.py")
        return

    # -------------------------------------------------------------------------
    # 3. LOAD RUBRICS (expert + student distributions)
    # -------------------------------------------------------------------------
    rubrics_file = script_dir / "dataset" / "rubrics.jsonl"
    rubrics = {}  # Full rubric data including student info
    if rubrics_file.exists():
        with open(rubrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    rubrics[obj["id"]] = obj

    print(f"Loaded {len(rubrics)} rubrics")

    # Count questions with student data
    with_student = sum(1 for r in rubrics.values() if "student_distribution" in r)
    print(f"Questions with student comparison data: {with_student}")

    # -------------------------------------------------------------------------
    # 4. LOAD RESPONSES - group by (id, trial)
    # -------------------------------------------------------------------------
    responses: Dict[tuple, dict] = {}  # (id, trial) -> item

    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item_id = item.get("id") or item.get("item_id")
                trial = item.get("trial", 1)
                if item_id:
                    responses[(item_id, trial)] = item

    # Apply --limit (first N unique item IDs)
    unique_ids = sorted(set(id for id, _ in responses.keys()))
    if args.limit is not None:
        unique_ids = unique_ids[:args.limit]
        limited_set = set(unique_ids)
        responses = {k: v for k, v in responses.items() if k[0] in limited_set}
        print(f"Limited to {len(unique_ids)} items")

    # Determine trials from data
    trial_numbers = sorted(set(t for _, t in responses.keys()))
    n_trials = len(trial_numbers)
    print(f"Detected {n_trials} trial(s); scoring trial={args.trial}")
    if args.trial not in trial_numbers:
        print(f"Error: trial {args.trial} not present in raw data (available: {trial_numbers})")
        return

    # -------------------------------------------------------------------------
    # 5. SCORE PER ITEM (single trial; item-level bootstrap CIs)
    # -------------------------------------------------------------------------
    parse_errors = 0
    api_errors = 0
    question_results = []  # For detailed JSONL output

    # Per-subtest item-level scores (kept as flat lists for bootstrap)
    overall_sct_per_item: List[float] = []
    overall_inset_per_item: List[float] = []
    overall_confidence_per_item: Dict[str, List[float]] = {
        metric: [] for metric in CONFIDENCE_METRIC_LABELS
    }
    subtest_sct: Dict[str, List[float]] = defaultdict(list)
    subtest_inset: Dict[str, List[float]] = defaultdict(list)
    subtest_confidence: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {metric: [] for metric in CONFIDENCE_METRIC_LABELS}
    )
    subtest_names = set()

    # Per-item bookkeeping for human comparison (uses the same single-trial scores)
    item_trial_scores: Dict[str, list] = defaultdict(list)
    item_source: Dict[str, str] = {}

    trial = args.trial
    trial_items = {id: item for (id, t), item in responses.items() if t == trial}

    for item_id in tqdm(unique_ids, desc=f"Scoring trial {trial}"):
        if item_id not in trial_items:
            continue

        item = trial_items[item_id]
        rubric = rubrics.get(item_id, {})
        expert_dist = rubric.get("expert_distribution", [0, 0, 0, 0, 0])

        if item.get("error") or item.get("response") is None:
            api_errors += 1
            continue

        rating = parse_response(item.get("response", ""))
        if rating is None:
            parse_errors += 1
            continue

        score_result = score_single_response(expert_dist, rating)
        per_item_score = float(score_result["normalized_score"])
        per_item_inset = 1.0 if score_result["in_expert_set"] else 0.0
        expert_modal_rating = unique_modal_expert_rating(expert_dist)
        confidence_indicators = confidence_error_indicators(
            expert_modal_rating, rating
        )

        overall_sct_per_item.append(per_item_score)
        overall_inset_per_item.append(per_item_inset)
        for metric, value in confidence_indicators.items():
            overall_confidence_per_item[metric].append(value)

        item_trial_scores[item_id].append(per_item_score)

        source_short = item.get("metadata", {}).get("source_short", "unknown")
        subtest_names.add(source_short)
        item_source[item_id] = source_short
        subtest_sct[source_short].append(per_item_score)
        subtest_inset[source_short].append(per_item_inset)
        for metric, value in confidence_indicators.items():
            subtest_confidence[source_short][metric].append(value)

        detail = {
            "id": item_id,
            "trial": trial,
            "rating": rating,
            "expert_modal_rating": expert_modal_rating,
            "normalized_score": per_item_score,
            "in_expert_set": score_result["in_expert_set"],
        }
        detail.update(confidence_indicators)
        question_results.append(detail)

    # -------------------------------------------------------------------------
    # 6. AGGREGATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    total = len(unique_ids)
    n_used = len(overall_sct_per_item)
    print(f"Total questions:   {total}")
    print(f"Scored items:      {n_used} (trial {trial})")
    print(f"Parse errors:      {parse_errors}")
    print(f"API errors:        {api_errors}")
    print(f"Bootstrap:         B={args.n_boot}, seed={args.seed}, percentile method")
    print()

    metrics_list = []

    if overall_sct_per_item:
        b = bootstrap_ci(overall_sct_per_item, n_boot=args.n_boot, seed=args.seed)
        print(f"SCT Score:         {b['mean']:.4f} (±{b['ci']:.4f}, 95% CI [{b['ci_lo']:.4f}, {b['ci_hi']:.4f}], n={b['n']})")
        metrics_list.append({"category": "Overall", "metric": "sct_score",
                             "trials": b["n"], "mean": b["mean"], "ci": b["ci"],
                             "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"]})

        b = bootstrap_ci(overall_inset_per_item, n_boot=args.n_boot, seed=args.seed)
        print(f"In Expert Set:     {b['mean']:.4f} (±{b['ci']:.4f}, 95% CI [{b['ci_lo']:.4f}, {b['ci_hi']:.4f}], n={b['n']})")
        metrics_list.append({"category": "Overall", "metric": "pct_in_expert_set",
                             "trials": b["n"], "mean": b["mean"], "ci": b["ci"],
                             "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"]})

        for metric, label in CONFIDENCE_METRIC_LABELS.items():
            values = overall_confidence_per_item[metric]
            if not values:
                continue
            b = bootstrap_ci(values, n_boot=args.n_boot, seed=args.seed)
            print(f"{label + ':':<18} {b['mean']:.4f} (±{b['ci']:.4f}, 95% CI [{b['ci_lo']:.4f}, {b['ci_hi']:.4f}], n={b['n']})")
            metrics_list.append({"category": "Overall", "metric": metric,
                                 "trials": b["n"], "mean": b["mean"], "ci": b["ci"],
                                 "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"]})
        print()

    # Per-subtest metrics (item-level bootstrap)
    subtest_mean_sct: Dict[str, float] = {}
    if subtest_names:
        print("Subtest Scores:")
        for source_short in sorted(subtest_names):
            sct_items = subtest_sct.get(source_short, [])
            inset_items = subtest_inset.get(source_short, [])
            if not sct_items:
                continue
            b_sct = bootstrap_ci(sct_items, n_boot=args.n_boot, seed=args.seed)
            b_pct = bootstrap_ci(inset_items, n_boot=args.n_boot, seed=args.seed)
            print(f"  {source_short:<15} SCT: {b_sct['mean']:.4f}±{b_sct['ci']:.4f}  "
                  f"InSet: {b_pct['mean']:.4f}±{b_pct['ci']:.4f}  (n={b_sct['n']})")
            subtest_mean_sct[source_short] = b_sct["mean"]
            metrics_list.append({"category": source_short, "metric": "sct_score",
                                 "trials": b_sct["n"], "mean": b_sct["mean"], "ci": b_sct["ci"],
                                 "ci_lo": b_sct["ci_lo"], "ci_hi": b_sct["ci_hi"]})
            metrics_list.append({"category": source_short, "metric": "pct_in_expert_set",
                                 "trials": b_pct["n"], "mean": b_pct["mean"], "ci": b_pct["ci"],
                                 "ci_lo": b_pct["ci_lo"], "ci_hi": b_pct["ci_hi"]})
            for metric in CONFIDENCE_METRIC_LABELS:
                values = subtest_confidence[source_short][metric]
                if not values:
                    continue
                b_metric = bootstrap_ci(values, n_boot=args.n_boot, seed=args.seed)
                metrics_list.append({"category": source_short, "metric": metric,
                                     "trials": b_metric["n"], "mean": b_metric["mean"], "ci": b_metric["ci"],
                                     "ci_lo": b_metric["ci_lo"], "ci_hi": b_metric["ci_hi"]})

    # -------------------------------------------------------------------------
    # 7. HUMAN COMPARISON (console output only)
    # -------------------------------------------------------------------------
    # Compute per-item average model score across trials, compare to student
    human_comparison_data: Dict[str, Dict[str, list]] = {}

    for item_id, scores in item_trial_scores.items():
        rubric = rubrics.get(item_id, {})
        if "student_distribution" not in rubric:
            continue

        expert_dist = rubric.get("expert_distribution", [0, 0, 0, 0, 0])
        if "student_score" in rubric:
            student_score = rubric["student_score"]
        else:
            student_score = compute_student_score_from_distribution(
                expert_dist, rubric["student_distribution"]
            )

        source = item_source.get(item_id, "unknown")
        if source not in human_comparison_data:
            human_comparison_data[source] = {"model_scores": [], "student_scores": []}
        human_comparison_data[source]["model_scores"].append(float(np.mean(scores)))
        human_comparison_data[source]["student_scores"].append(student_score)

    comparison_results = {}
    if human_comparison_data:
        print()
        print("=" * 60)
        print("HUMAN (STUDENT) COMPARISON")
        print("=" * 60)
        print("(Only for subtests with student response data)")
        print()

        all_model_scores = []
        all_student_scores = []

        print(f"{'Subtest':<18} {'Model':>8} {'Student':>8} {'Diff':>8}  {'n':>5}")
        print("-" * 52)

        for source, data in sorted(human_comparison_data.items()):
            if data["model_scores"]:
                model_mean = np.mean(data["model_scores"])
                student_mean = np.mean(data["student_scores"])
                diff = model_mean - student_mean
                n = len(data["model_scores"])

                print(f"{source:<18} {model_mean:>8.4f} {student_mean:>8.4f} {diff:>+8.4f}  {n:>5}")

                all_model_scores.extend(data["model_scores"])
                all_student_scores.extend(data["student_scores"])

                comparison_results[source] = {
                    "model_score": float(model_mean),
                    "student_score": float(student_mean),
                    "difference": float(diff),
                    "n_questions": n,
                }

        # Overall comparison
        if all_model_scores:
            print("-" * 52)
            overall_model = np.mean(all_model_scores)
            overall_student = np.mean(all_student_scores)
            overall_diff = overall_model - overall_student
            print(f"{'OVERALL':<18} {overall_model:>8.4f} {overall_student:>8.4f} {overall_diff:>+8.4f}  {len(all_model_scores):>5}")

            comparison_results["_overall"] = {
                "model_score": float(overall_model),
                "student_score": float(overall_student),
                "difference": float(overall_diff),
                "n_questions": len(all_model_scores),
            }

            # Interpretation
            print()
            if overall_diff > 0.05:
                print(f"Model OUTPERFORMS students by {overall_diff:.1%}")
            elif overall_diff < -0.05:
                print(f"Model UNDERPERFORMS vs students by {abs(overall_diff):.1%}")
            else:
                print(f"Model performs SIMILARLY to students (diff: {overall_diff:+.1%})")

    # -------------------------------------------------------------------------
    # 7b. PUBLISHED BASELINES COMPARISON (console output only)
    # -------------------------------------------------------------------------
    baselines_file = script_dir / "dataset" / "baselines.json"
    if baselines_file.exists() and subtest_mean_sct:
        with open(baselines_file, 'r') as f:
            baselines = json.load(f)

        print()
        print("=" * 60)
        print("PUBLISHED BASELINES COMPARISON")
        print("=" * 60)
        print()

        # Show per-subtest comparison with published human baselines
        print("Human Baselines (from published studies):")
        print(f"{'Subtest':<15} {'Model':>7} {'Student':>8} {'Resident':>8} {'Staff':>7}")
        print("-" * 50)

        for short in sorted(subtest_mean_sct.keys()):
            if short not in baselines:
                continue

            baseline = baselines[short]
            model_score = subtest_mean_sct[short]

            student = baseline.get("human_baselines", {}).get("student", {}).get("score", "")
            resident = baseline.get("human_baselines", {}).get("resident", {}).get("score", "")
            staff = baseline.get("human_baselines", {}).get("staff", {}).get("score", "")

            student_str = f"{student:.3f}" if student else "   -"
            resident_str = f"{resident:.3f}" if resident else "   -"
            staff_str = f"{staff:.3f}" if staff else "   -"

            print(f"{short:<15} {model_score:>7.3f} {student_str:>8} {resident_str:>8} {staff_str:>7}")

        # Show comparison with previously evaluated models
        print()
        print("Model Baselines (from previous evaluations):")
        print(f"{'Model':<20} {'Score':>7}  vs current: {model_name}")
        print("-" * 50)

        # Aggregate baseline model scores across tested subtests
        baseline_model_scores: Dict[str, List[float]] = {}
        current_scores_for_comparison = []

        for short, model_sct in subtest_mean_sct.items():
            if short not in baselines:
                continue

            current_scores_for_comparison.append(model_sct)

            for model_id, model_data in baselines[short].get("model_baselines", {}).items():
                if model_id not in baseline_model_scores:
                    baseline_model_scores[model_id] = []
                baseline_model_scores[model_id].append(model_data["score"])

        if current_scores_for_comparison:
            current_avg = np.mean(current_scores_for_comparison)

            # Sort by score descending
            sorted_models = sorted(
                baseline_model_scores.items(),
                key=lambda x: np.mean(x[1]),
                reverse=True
            )

            for model_id, scores_list in sorted_models:
                avg_score = np.mean(scores_list)
                diff = current_avg - avg_score
                diff_str = f"{diff:+.3f}" if diff != 0 else "  same"
                print(f"{model_id:<20} {avg_score:>7.3f}  {diff_str}")

            print("-" * 50)
            print(f"{'>>> ' + model_name:<20} {current_avg:>7.3f}  (current)")

    # -------------------------------------------------------------------------
    # 8. SAVE OUTPUTS
    # -------------------------------------------------------------------------
    # Write detailed results JSONL
    details_file = output_dir / f"{model_name}_details.jsonl"
    with open(details_file, 'w') as f:
        for result in question_results:
            f.write(json.dumps(result) + "\n")

    # Write metrics CSV (standardized format: category,metric,trials,mean,ci)
    metrics_file = output_dir / f"{model_name}.csv"
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(metrics_file, index=False)

    # Save comparison to JSON if available
    if comparison_results:
        comparison_file = output_dir / f"{model_name}_human_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\nComparison saved to: {comparison_file}")

    print()
    print(f"Details saved to: {details_file}")
    print(f"Metrics saved to: {metrics_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
