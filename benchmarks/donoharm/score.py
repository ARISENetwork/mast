"""Score donoharm benchmark responses using multi-judge LLM evaluation.

Calls judge LLMs with structured output, computes clinical harm metrics,
and aggregates across judges with CI.

Usage:
    python score.py --model-config PATH --benchmark-config PATH [--limit N] [--threads N]
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge.metrics import (  # noqa: E402
    compute_harm,
    compute_metrics_for_case,
)
from data_loader import variant_within_k  # noqa: E402

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


# Stratified cluster bootstrap: base case is the sampling unit, specialty prefix
# (Card/Pulm/Endo/...) is the stratum. Variants × judges are nested within case.
# B=2000 puts MC noise on the 2.5/97.5 percentile endpoints at ~0.005, well below
# the CI width itself. Bump to 5000 if tighter endpoints are needed. Strata with
# <MIN_STRATUM_CASES are merged into an "_other" pool so the bootstrap inside
# each stratum has non-degenerate sampling.
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 0
MIN_STRATUM_CASES = 3
_SPECIALTY_RE = re.compile(r"^([A-Za-z]+)")


def _specialty_of(case_id: str) -> str:
    m = _SPECIALTY_RE.match(case_id)
    return m.group(1) if m else "_other"


def _cluster_bootstrap_ci(
    case_values: dict[str, list[float]],
    n_boot: int = BOOTSTRAP_ITERS,
    seed: int = BOOTSTRAP_SEED,
    case_aggregator: Callable[[list[float]], float] | None = None,
) -> tuple[float, float, float]:
    """Stratified cluster bootstrap 95% CI, base case = sampling unit.

    Estimand: expected score on a new clinical case drawn from the same case
    distribution, evaluated by the currently-configured judge(s). Variants
    (baseline + perturbations) are treated as non-significantly-different
    representations of the same case and pool into a single per-case scalar
    via `case_aggregator` (default mean); judges nest within the same case.
    CI is conditional on the configured judge set - judge-choice uncertainty
    is not captured here.

    Cases are grouped by specialty prefix (Card, Pulm, Endo, ...). Each iteration
    resamples case IDs with replacement within each stratum preserving that
    stratum's size, then takes the mean of the sampled case scalars. The point
    estimate is still the unweighted grand mean of per-case scalars.

    Stratification exploits the fact that specialty difficulty is a large component
    of between-case variance; narrows the CI ~5-15% vs unstratified cluster bootstrap
    without biasing the estimator. Strata with fewer than MIN_STRATUM_CASES cases
    are merged into a single "_other" pool.

    `case_aggregator` swaps the per-case collapse - pass `min` (or a direction-aware
    `worst` callable) for safety-floor metrics. Min is non-smooth so the CI may
    be slightly wider than mean's, which honestly reflects the noisier estimator.

    Returns (point, lo, hi); callers emit lo/hi and a symmetric half-width
    (hi-lo)/2 as `ci` for backwards compatibility.
    """
    case_ids = list(case_values.keys())
    if not case_ids:
        return (float("nan"), float("nan"), float("nan"))

    if case_aggregator is None:
        case_aggregator = lambda v: float(np.mean(v))  # noqa: E731
    case_means = {c: float(case_aggregator(case_values[c])) for c in case_ids}
    point = float(np.mean(list(case_means.values())))

    if len(case_ids) < 2:
        return (point, point, point)

    strata: dict[str, list[str]] = defaultdict(list)
    for c in case_ids:
        strata[_specialty_of(c)].append(c)
    small = [s for s, cs in strata.items() if len(cs) < MIN_STRATUM_CASES]
    if small:
        merged: list[str] = []
        for s in small:
            merged.extend(strata.pop(s))
        if merged:
            strata["_other"].extend(merged)

    rng = np.random.default_rng(seed)
    stratum_arrays = {s: np.array([case_means[c] for c in cs]) for s, cs in strata.items()}
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        parts = []
        for arr in stratum_arrays.values():
            parts.append(arr[rng.integers(0, arr.size, size=arr.size)])
        boot_means[i] = float(np.mean(np.concatenate(parts)))

    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return (point, float(lo), float(hi))


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_config(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def judge_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def parse_variant_id(variant_id: str) -> tuple[str, int]:
    """Parse variant ID into (base_id, suffix).

    'Derm001-3' -> ('Derm001', 3) (perturbation).
    'Derm001'   -> ('Derm001', -1) (baseline, no suffix).
    """
    if "-" in variant_id:
        parts = variant_id.rsplit("-", 1)
        try:
            return parts[0], int(parts[1])
        except ValueError:
            return variant_id, -1
    return variant_id, -1


# F1_floor: per-case worst variant (literal min of F1_weighted across a case's
# variants; k-dependent, drifts down as variant count grows). Single-trial
# models are skipped since the aggregator collapses to the mean.
FLOOR_METRIC_SOURCES = {
    "F1_floor": ("F1_weighted", lambda v: float(min(v))),
}
MIN_VARIANTS_FOR_FLOOR = 2


def _aggregate_metric_set(
    records: list[dict],
    judge_names: list[str],
    metrics_key: str,
    category: str,
) -> list[dict]:
    """Aggregate metrics: per base case mean (across variants and judges), CI
    across cases. Also emits the F1_floor worst-variant companion for models
    with >= MIN_VARIANTS_FOR_FLOOR variants per case."""
    metric_names = ["F1_weighted", "Severe_rate", "Precision_weighted", "Recall_weighted"]

    # Collect per-case means: for each base case, average across all variants and judges
    filtered = [r for r in records if r.get("judge") in judge_names and metrics_key in r]
    if not filtered:
        return []

    # Group by base case ID
    case_metrics: dict[str, list[dict]] = {}
    for r in filtered:
        base_id, _ = parse_variant_id(r["id"])
        case_metrics.setdefault(base_id, []).append(r[metrics_key])

    n_cases = len(case_metrics)
    median_variants = float(np.median([len(ml) for ml in case_metrics.values()]))
    eligible_for_floor = median_variants >= MIN_VARIANTS_FOR_FLOOR

    rows = []
    for metric in metric_names:
        # Collect per-(variant, judge) values grouped by base case
        case_values: dict[str, list[float]] = {}
        for base_id, metrics_list in case_metrics.items():
            values = [m[metric] for m in metrics_list if metric in m and not np.isnan(m[metric])]
            if values:
                case_values[base_id] = values
        if not case_values:
            continue
        mean, lo, hi = _cluster_bootstrap_ci(case_values)
        ci = (hi - lo) / 2
        rows.append({
            "category": category,
            "metric": metric,
            "trials": n_cases,
            "mean": round(mean, 4),
            "ci": round(ci, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        })

    # Floor column: per-case worst variant instead of per-case mean.
    if eligible_for_floor:
        for floor_name, (source_metric, aggregator) in FLOOR_METRIC_SOURCES.items():
            case_values = {}
            for base_id, metrics_list in case_metrics.items():
                values = [
                    m[source_metric]
                    for m in metrics_list
                    if source_metric in m and not np.isnan(m[source_metric])
                ]
                if values:
                    case_values[base_id] = values
            if not case_values:
                continue
            mean, lo, hi = _cluster_bootstrap_ci(case_values, case_aggregator=aggregator)
            ci = (hi - lo) / 2
            rows.append({
                "category": category,
                "metric": floor_name,
                "trials": n_cases,
                "mean": round(mean, 4),
                "ci": round(ci, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
            })

    return rows


def aggregate_across_judges(
    judged_path: Path,
    judge_names: list[str],
    k: int | None = None,
    rubric_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Aggregate metrics: mean across cases per judge, then CI across judges.

    Emits rows from `rec["metrics"]` under the canonical metric names.

    Scoped to the requested `--k`/`--limit`: the judged file is a cache that
    can hold a superset of the current request (the match cache grows but never
    prunes), so the CSV always reflects exactly the variants/cases asked for
    regardless of cache state -- not "whatever happens to be in the file".
    """
    records = []
    if judged_path.exists():
        for line in judged_path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))

    records = [
        r for r in records
        if (k is None or variant_within_k(r["id"], k))
        and (rubric_ids is None or parse_variant_id(r["id"])[0] in rubric_ids)
    ]

    if not records:
        log.error("No judged results to aggregate")
        sys.exit(1)

    rows = _aggregate_metric_set(records, judge_names, "metrics", "Overall")

    return pd.DataFrame(rows)


def _normalize_response_actions(record: dict) -> dict:
    """Drop stale rubric references from responseActions[].match.

    Reconciles the bug where stage-7's apply_global_match_review only flipped
    options[] but left responseActions[].match referencing now-demoted oids.
    A responseAction's match must only contain oids whose options[oid].matched
    is True; anything else is stale and is removed.

    Self-healing: idempotent, safe to run on already-consistent records (no-op
    when every match references a matched option).
    """
    matched_ids = {int(o["id"]) for o in record.get("options", []) if o.get("matched")}
    rebuilt = []
    for ra in record.get("responseActions", []):
        match_str = ra.get("match", "") or ""
        if not match_str:
            rebuilt.append(ra)
            continue
        toks = [t.strip() for t in match_str.split(",") if t.strip()]
        kept = [t for t in toks if int(t) in matched_ids]
        if len(kept) == len(toks):
            rebuilt.append(ra)
        else:
            rebuilt.append({**ra, "match": ",".join(kept)})
    record["responseActions"] = rebuilt
    return record


def _rescore(judged_path: Path, rubrics: dict) -> None:
    """Recompute metrics from existing judged JSONL without calling judge LLMs.

    Recomputes `rec["metrics"]` from the cached match/review verdicts;
    re-running `score.py --rescore` is idempotent.
    """
    from judge.metrics import finalize_metrics, select_persisted_metrics

    lines = judged_path.read_text().splitlines()
    updated = []
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        base_id, _ = parse_variant_id(rec["id"])
        rubric = rubrics.get(base_id)
        if not rubric:
            # Not rescored (rubric not loaded), but still enforce the
            # persisted-metric contract on any pre-existing block.
            if "metrics" in rec:
                rec["metrics"] = select_persisted_metrics(rec["metrics"])
            updated.append(json.dumps(rec))
            continue
        # Heal stale stage-7-vs-responseAction references before metric recompute.
        rec = _normalize_response_actions(rec)
        response_actions = rec.get("responseActions", [])
        harm_results = compute_harm(rec["options"], rubric)
        rec["harm"] = [h for h in harm_results if h["harm_type"]]

        # Canonical: full responseActions (un-stripped). Precision_weighted
        # excludes off-rubric actions; verbosity is handled by recall correction.
        # finalize_metrics folds in the length correction (response_len lives at
        # the record top level so it survives this recompute), derives F1, and
        # reduces to the persisted block. Idempotent.
        rec["metrics"] = finalize_metrics(
            compute_metrics_for_case(harm_results, response_actions, rubric),
            rec.get("response_len"),
        )

        updated.append(json.dumps(rec))
    judged_path.write_text("\n".join(updated) + "\n")
    log.info("Rescored %d records in %s", len(updated), judged_path)


def main():
    # Judge defaults to the validated pinned models. --match-judge/--review-judge
    # let a user repoint the (Gemini-only) judge at another Gemini model when a
    # pinned preview id is unavailable to their key; doing so departs from the
    # reference table.
    from judge.config import DEFAULT_MATCH_JUDGE, DEFAULT_REVIEW_JUDGE
    parser = argparse.ArgumentParser(description="Score donoharm benchmark")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--benchmark-config", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to the first N base cases; matches run.py --limit.")
    parser.add_argument("--threads", type=int, default=40)
    parser.add_argument("--rescore", action="store_true",
                        help="recompute metrics from existing judged JSONL (no LLM calls)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt variant name (matches run.py --prompt)")
    parser.add_argument("--k", type=int, default=11, choices=range(1, 12),
                        help="Variants per case to score (1-11). 1=base only, "
                             "5=base + 4 perturbations, 11=all (default).")
    parser.add_argument("--match-judge", dest="strategy_extractor", default=None,
                        metavar="GEMINI_MODEL",
                        help=f"Match-stage judge model (default: {DEFAULT_MATCH_JUDGE}). "
                             "Must be a Gemini model; changing it departs from the "
                             "reference table.")
    parser.add_argument("--review-judge", dest="strategy_global_match_reviewer", default=None,
                        metavar="GEMINI_MODEL",
                        help=f"Review-stage judge model (default: {DEFAULT_REVIEW_JUDGE}). "
                             "Must be a Gemini model; changing it departs from the "
                             "reference table.")
    args = parser.parse_args()

    # Defaults for fields formerly driven by individual CLI flags. The public
    # bundle fixes these to their production values; the rest of the flags were
    # trimmed. (--match-judge/--review-judge above set strategy_extractor /
    # strategy_global_match_reviewer; left None they fall back to the defaults.)
    args.judged_path = None
    args.raw_dir = None
    args.cases = None
    args.no_aggregate = False
    args.strategy_match_prompt = None
    args.no_global_match_review = False
    args.strategy_global_match_review_prompt = None

    if args.threads is not None and (args.threads < 1 or args.threads > 200):
        parser.error("--threads must be between 1 and 200")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    return score_one(args)


def score_one(args):
    """Score a single (model_config, benchmark_config)."""
    model_config = load_yaml(args.model_config)
    bench_config = load_yaml(args.benchmark_config)
    config = merge_config(model_config, bench_config)

    model_name = model_config["model"]["name"]
    benchmark_name = bench_config["benchmark"]["name"]
    prompt_name = args.prompt or config.get("donoharm", {}).get("prompt", "default")

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    dataset_dir = script_dir / "dataset"

    # Load rubrics
    rubric_dir = dataset_dir / "rubrics"
    rubric_files = sorted(rubric_dir.glob("*.json"))
    if args.limit:
        # --limit N = first N base cases. Rubric filenames are <CaseId>.json,
        # so the sorted slice is the first N case ids -- identical to run.py's
        # base_case_id selection (which also sorts by case id).
        rubric_files = rubric_files[: args.limit]
    case_filter = {c.strip() for c in args.cases.split(",")} if args.cases else None
    rubrics = {}
    for f in rubric_files:
        data = json.loads(f.read_text())
        if case_filter and data["id"] not in case_filter:
            continue
        rubrics[data["id"]] = data
    if case_filter:
        missing = case_filter - set(rubrics.keys())
        if missing:
            log.error("Cases not found in rubrics: %s", ", ".join(sorted(missing)))
            sys.exit(1)

    log.info("Loaded %d rubrics", len(rubrics))

    # All prompts (including default) live under a prompt-named subdirectory.
    raw_dir = (Path(args.raw_dir) if args.raw_dir
               else repo_root / "results" / "raw" / benchmark_name / prompt_name)
    response_path = raw_dir / f"{model_name}.jsonl"
    if not response_path.exists():
        log.error("Response file not found: %s", response_path)
        sys.exit(1)
    responses = []
    for line in response_path.read_text().splitlines():
        if line.strip():
            responses.append(json.loads(line))
    responses = [r for r in responses if parse_variant_id(r["id"])[0] in rubrics]
    if args.k is not None:
        # Keep k variants per case: base + perturbation suffixes 0..k-2.
        # variant_within_k is the single source of truth shared with run.py.
        responses = [r for r in responses if variant_within_k(r["id"], args.k)]
    log.info("Loaded %d responses for %s", len(responses), model_name)

    # Partial-run guard (warn, do not block): a silently incomplete run still
    # writes a CSV that looks directly comparable to the reference table.
    # Expected count comes from the dataset manifest for the cases/k actually
    # in scope, so intentional subsets (--cases, --k, --limit) shrink `rubrics`
    # and don't false-alarm.
    items_path = dataset_dir / "items.jsonl"
    if items_path.exists():
        n_expected = 0
        for line in items_path.read_text().splitlines():
            if not line.strip():
                continue
            item_id = json.loads(line)["id"]
            base, _ = parse_variant_id(item_id)
            if base in rubrics and variant_within_k(item_id, args.k):
                n_expected += 1
        if n_expected and len(responses) < n_expected:
            log.warning(
                "Partial run: %d responses loaded but the dataset offers %d "
                "for the cases/k in scope. Scores below are over an INCOMPLETE "
                "set and are NOT comparable to the reference table.",
                len(responses), n_expected,
            )
    n_bad = sum(
        1 for r in responses
        if r.get("error") or not str(r.get("response") or "").strip()
    )
    if n_bad:
        log.warning(
            "%d of %d loaded responses are empty or carry an error field; "
            "the judge drops these, so they are EXCLUDED from scoring rather "
            "than penalized -- a model that returns empty/errored outputs can "
            "therefore score higher than it should. Retry or fill them in for "
            "a comparable score.",
            n_bad, len(responses),
        )

    # Run the strategy judge pipeline (the only judging pipeline)
    judged_path = Path(args.judged_path) if args.judged_path else raw_dir / f"{model_name}_judged.jsonl"
    judged_path.parent.mkdir(parents=True, exist_ok=True)

    from judge import (
        DEFAULT_MATCH_JUDGE,
        DEFAULT_MATCH_PROMPT,
        DEFAULT_REVIEW_JUDGE,
        DEFAULT_REVIEW_PROMPT,
        JudgeConfig,
        judge_responses,
    )
    from judge.io import (
        judge_short_name as _judge_short_name,
    )

    if args.rescore:
        log.info("Rescoring %s from existing judgments", model_name)
        _rescore(judged_path, rubrics)
    else:
        extractor = args.strategy_extractor or DEFAULT_MATCH_JUDGE
        global_match_reviewer = (
            None if args.no_global_match_review
            else (args.strategy_global_match_reviewer or DEFAULT_REVIEW_JUDGE)
        )
        log.info(
            "Model: %s | extractor: %s | "
            "global-match-reviewer: %s | Cases: %d",
            model_name, judge_name(extractor),
            judge_name(global_match_reviewer) if global_match_reviewer else "(skipped)",
            len(rubrics),
        )

        judge_config = JudgeConfig(
            cache_root=raw_dir / "_strategy",
            match_judge=extractor,
            match_prompt=(Path(args.strategy_match_prompt)
                              if args.strategy_match_prompt
                              else DEFAULT_MATCH_PROMPT),
            review_judge=global_match_reviewer,
            review_prompt=(Path(args.strategy_global_match_review_prompt)
                           if args.strategy_global_match_review_prompt
                           else DEFAULT_REVIEW_PROMPT),
            threads=args.threads,
            cases_filter=(
                [s.strip() for s in args.cases.split(",") if s.strip()]
                if args.cases else None
            ),
        )
        judge_responses(
            model_name=model_name,
            responses=responses,
            rubrics=rubrics,
            config=judge_config,
            judged_path=judged_path,
        )

    if args.no_aggregate:
        log.info("--no-aggregate set; skipping scores CSV")
        return

    # Aggregate. Auto-discover the judge identifier set from the judged
    # file's `judge` field. Falls back to the review-judge name only when
    # the file is empty (defensive).
    jnames = sorted({
        json.loads(line)["judge"]
        for line in judged_path.read_text().splitlines()
        if line.strip()
    }) if judged_path.exists() else []
    if not jnames:
        jnames = [_judge_short_name(DEFAULT_REVIEW_JUDGE)]
    scores_df = aggregate_across_judges(judged_path, jnames, k=args.k, rubric_ids=set(rubrics))

    # Write MAST-format scores CSV under the prompt-named subdirectory.
    scores_root = repo_root / "results" / "scores" / benchmark_name
    scores_path = scores_root / prompt_name / f"{model_name}.csv"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(scores_path, index=False)
    log.info("Wrote metrics to %s", scores_path)
    print(scores_df.to_string(index=False))



if __name__ == "__main__":
    main()
