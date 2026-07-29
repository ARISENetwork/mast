"""Match-graph judge pipeline for donoharm.

Public entry point: `judge_responses(*, model_name, responses, rubrics,
config, judged_path) -> JudgeRunSummary`. Re-exported from
`benchmarks.donoharm.judge`.

LLM-driven steps, in execution order:
  match           (match_stage.py, prompt extract_match.md,
                   fixed flash match judge, judge-independent. Also emits
                   strategies.jsonl via deterministic union-find grouping
                   in the same invocation; no separate group step.)
  review          (review_stage.py, prompt
                   global_match_review.md, the review judge re-reads ALL
                   options with override authority)
  apply-overrides (overrides.apply_overrides_to_strategies, bakes
                   review promotions/demotions into the refined match
                   graph)

Match (including grouping) is judge-independent and cached per model under
  results/raw/donoharm/{prompt}/_strategy/{model}/

Cache subdirs:
  match/<match_judge>.jsonl     extract + match
  review_<judge>/<judge>.jsonl  global match review

Final output: results/raw/donoharm/{prompt}/{model}_judged.jsonl, shape-
compatible with the legacy combined-judge pipeline so downstream
compute_metrics_for_case + compute_resilience consumers don't change.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from .config import JudgeConfig
from .io import (
    derive_record_judge_short,
    judge_short_name as _judge_short_name,
    load_jsonl,
    responses_to_dict,
    write_responses_file,
)
from .overrides import apply_overrides_to_refined
from .summary import JudgeRunSummary

log = logging.getLogger(__name__)


def _cache_keys(path: Path) -> set[tuple[str, int]]:
    return {(r["id"], r.get("trial", 1)) for r in load_jsonl(path)}


def _require_stage_complete(stage: str, expected: set, path: Path) -> None:
    """Abort when a stage silently skipped work.

    Both stages return without writing on exhausted retries, and neither the
    adapter nor score.py reconciles what came back against what went in: a
    failed review still yields a judged record (match-only, stamped with the
    review judge), so a downstream count check cannot see it.
    """
    missing = expected - _cache_keys(path)
    if not missing:
        return
    shown = ", ".join(f"{cid}/t{trial}" for cid, trial in sorted(missing)[:10])
    if len(missing) > 10:
        shown += f", ... (+{len(missing) - 10} more)"
    raise RuntimeError(
        f"{stage} stage is incomplete: {len(missing)} of {len(expected)} expected "
        f"records are missing from {path} ({shown}). Scores from an incomplete "
        f"judge run are not comparable to the reference table. Re-run to fill the "
        f"cache; completed records are cached and will not be re-billed."
    )


def judge_responses(
    *,
    model_name: str,
    responses: list[dict],
    rubrics: dict,
    config: JudgeConfig,
    judged_path: Path,
) -> JudgeRunSummary:
    """Run the donoharm judge pipeline over one model's responses.

    Caller responsibilities:
      - Load `responses` (list of {id, trial, response}) however you want.
      - Load `rubrics` (case_id -> rubric_dict) however you want.
      - Decide where to write `judged_path`.

    Pipeline responsibilities:
      - LLM calls for match, group, review.
      - Cache management under `config.cache_root / model_name`.
      - Resume semantics (idempotent re-runs hit cache; staleness via
        prompt_hash).
      - Adapter step that produces the final per-record schema.
      - Writing `judged_path` (one record per (case_id, trial)).

    Returns:
        JudgeRunSummary with the overall record counts and elapsed time.
        Does NOT return the records themselves (re-read `judged_path` if
        needed).
    """
    from .adapter import adapt_model
    from .stages.match_stage import base_case_id, run_match, emit_strategies
    from .stages.review_stage import run_review

    t_start = time.perf_counter()
    cache_dir = config.cache_root / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    case_filter = set(config.cases_filter) if config.cases_filter else None

    # Materialize responses.jsonl for inspectability + resume hygiene; the
    # stage functions consume the in-memory dict, not this file.
    write_responses_file(responses, cache_dir / "responses.jsonl")
    responses_dict = responses_to_dict(responses)
    # responses_to_dict drops falsy responses. Silently judging fewer records
    # than the caller handed us is the exact bias this pipeline must not have.
    n_dropped = len(responses) - len(responses_dict)
    if n_dropped:
        raise ValueError(
            f"{n_dropped} of {len(responses)} response records are empty or "
            f"duplicate keys; they would be dropped before judging and shrink "
            f"the scored denominator. Filter or fix them before judging."
        )

    log.info(
        "[strategy] cache_dir=%s; responses=%d; global-match-reviewer=%s",
        cache_dir, len(responses),
        _judge_short_name(config.review_judge) if config.review_judge else "(skipped)",
    )

    # match + group: judge-independent. Resume logic inside run_match keys
    # on (case, trial), so partial / stale caches don't get re-billed.
    match_short = _judge_short_name(config.match_judge)
    match_dir = cache_dir / "match"
    match_path = match_dir / f"{match_short}.jsonl"
    run_match(
        judge_model=config.match_judge, threads=config.threads,
        responses=responses_dict, rubrics=rubrics,
        case_filter=case_filter, trial_filter=None, limit=None,
        prompt_path=config.match_prompt, out_dir=match_dir,
        model_name=model_name,
    )
    if not match_path.exists():
        raise RuntimeError(f"match stage produced no output at {match_path}")
    # Mirrors run_match's own task predicates so the expected set is exactly
    # what that stage should have produced.
    expected_match = {
        (case, trial)
        for (_m, case, trial) in responses_dict
        if (not case_filter or case in case_filter) and base_case_id(case) in rubrics
    }
    _require_stage_complete("match", expected_match, match_path)
    strategies_path = cache_dir / "strategies.jsonl"
    emit_strategies(match_path, strategies_path)

    # review: strong judge re-reviews EVERY option's match verdict with
    # authority to override.
    review_path: Path | None = None
    if config.review_judge:
        review_short = _judge_short_name(config.review_judge)
        # run_review expects (model, case, trial) keys; the empty model
        # placeholder gets overlaid via model_name inside run_review.
        refined_by_key = {
            ("", r["id"], r.get("trial", 1)): r for r in load_jsonl(strategies_path)
        }
        match_recs_by_key = {
            ("", r["id"], r.get("trial", 1)): r for r in load_jsonl(match_path)
        }
        review_path = run_review(
            judge_model=config.review_judge,
            refined=refined_by_key, responses=responses_dict,
            rubrics=rubrics, match_recs=match_recs_by_key,
            prompt_path=config.review_prompt,
            out_dir=cache_dir / f"review_{review_short}",
            threads=config.threads, model_name=model_name,
            case_filter=case_filter,
        )
        # Mirrors run_review's task predicates. A review failure writes no
        # cache record but still yields a judged record downstream, so this is
        # the only place the omission is visible.
        response_keys = {(case, trial) for (_m, case, trial) in responses_dict}
        expected_review = {
            (case, trial)
            for (_m, case, trial) in refined_by_key
            if (not case_filter or case in case_filter)
            and base_case_id(case) in rubrics
            and (case, trial) in response_keys
        }
        _require_stage_complete("review", expected_review, review_path)

    # Apply review overrides at the strategy level. Below-threshold
    # (unattributed) promotions fall through unchanged and are picked up by
    # apply_global_match_review in the adapt loop for recall accounting.
    if review_path is not None and review_path.exists():
        refined_post_review_path = apply_overrides_to_refined(
            strategies_path, review_path,
            cache_dir / "strategies_post_review.jsonl",
        )
    else:
        refined_post_review_path = strategies_path

    n_written, n_missing_rubric = adapt_model(
        refined_post_review_path=refined_post_review_path,
        review_path=review_path,
        rubrics=rubrics,
        judged_path=judged_path,
        judge_short=derive_record_judge_short(config.review_judge),
        global_match_reviewer_short=(
            _judge_short_name(config.review_judge) if config.review_judge else None
        ),
    )

    log.info(
        "[strategy] wrote %d judged records (missing_rubric=%d)",
        n_written, n_missing_rubric,
    )

    return JudgeRunSummary(
        n_records=n_written,
        n_missing_rubric=n_missing_rubric,
        elapsed_s=time.perf_counter() - t_start,
    )


