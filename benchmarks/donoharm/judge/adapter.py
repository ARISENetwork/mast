"""Adapter from the per-stage strategy-pipeline outputs to the production judge schema.

Production `score.py` consumes per-judge records of shape:
    {options, responseActions, summary}
where:
  - options[]:        per-rubric-option {id, matched, partial, evidence}
  - responseActions[]: per-action {number, action, category, match, score, rationale}

The strategy pipeline produces:
  - strategies.jsonl (or strategies_post_review.jsonl): match-step + group
    output, post review-override application. Per (case, trial):
    {strategies: [{strategy_id, summary, sub_actions, matched, matched_options}]}.

This adapter takes a (refined record, rubric) tuple and emits the
production schema for one (case, trial, judge).

`build_options_only` is a lighter constructor for callers that only need
the options[] view (notably the review step, which reads
options.matched/partial/evidence and rubric.score but never
responseActions).

Action mapping rules:
  - Each sub_action becomes a responseAction.
  - If the sub_action has match_resolved (anchored to rubric option(s)),
    `match` = comma-separated option IDs, `score` = 0 (rubric-handled).
  - Otherwise the sub_action is off-rubric and gets the benign default
    `score` = 5 with `match` empty.

Options mapping rules:
  - For each rubric option: matched=True if any sub_action's match_resolved
    contains its ID, else False.
  - Partial flag: sourced from refined_record.partial_option_ids, singleton
    options the matcher flagged matched&partial in the match step.
"""
from __future__ import annotations

import json
from pathlib import Path

from .io import load_jsonl
from .metrics import (
    compute_harm,
    compute_metrics_for_case,
    finalize_metrics,
)
from .schemas import validate_record


def _load_response_lens_by_key(responses_path: Path | None) -> dict[tuple[str, int], int]:
    """Read a `responses.jsonl` cache and return `(id, trial) -> len(response)`.

    The cache (written by `io.write_responses_file`) is model-less; the
    adapter is per-model, so (id, trial) is the right key. Empty/missing
    response strings are dropped, so length-correction silently no-ops
    on those records (same behavior as `score._rescore`).
    """
    if responses_path is None or not responses_path.exists():
        return {}
    out: dict[tuple[str, int], int] = {}
    for r in load_jsonl(responses_path):
        resp = r.get("response")
        if not isinstance(resp, str) or not resp:
            continue
        rid = r.get("id")
        if rid is None:
            continue
        out[(str(rid), int(r.get("trial", 1)))] = len(resp)
    return out


def _build_judged_record(
    *,
    ref: dict,
    review_rec: dict | None,
    rubric: dict,
    judge_short: str,
    response_len: int | None = None,
) -> dict:
    """Assemble one production-shape judged record from per-stage outputs.

    Caller has already filtered for rubric presence; this function only
    does the merge + metric computation.
    """
    prod = adapt_judge_to_prod(ref, rubric)

    # Apply review overrides at the prod-shape level. Most are idempotent
    # vs the strategies-level apply; load-bearing only for unattributed
    # promotions (reviewer cited action_id=null) -- option flag flips
    # downstream for recall accounting.
    if review_rec:
        prod = apply_global_match_review(prod, review_rec.get("overrides", []))

    harm_results = compute_harm(prod["options"], rubric)

    # Pass full responseActions (un-stripped) to the scorer. Precision_weighted
    # excludes off-rubric actions from its denominator; verbosity is instead
    # handled by the recall length-bias correction.
    response_actions = prod.get("responseActions", [])
    metrics = compute_metrics_for_case(harm_results, response_actions, rubric)

    rec = {
        "id": ref["id"],
        "trial": ref.get("trial", 1),
        "judge": judge_short,
        "options": prod["options"],
        "responseActions": prod.get("responseActions", []),
        "harm": [h for h in harm_results if h["harm_type"]],
        "metrics": metrics,
    }

    # finalize_metrics applies the recall length correction, derives F1, and
    # reduces to the persisted block, so the file written here is already
    # length-corrected. Set response_len at the record top level so
    # score._rescore can re-finalize without re-judging (same contract).
    if response_len is not None:
        rec["response_len"] = int(response_len)
    rec["metrics"] = finalize_metrics(rec["metrics"], response_len)
    return rec


def adapt_model(
    *,
    refined_post_review_path: Path,
    review_path: Path | None,
    rubrics: dict,
    judged_path: Path,
    judge_short: str,
    responses_path: Path | None = None,
) -> tuple[int, int]:
    """Assemble the per-model `judged.jsonl` from per-stage cache outputs.

    Single source of truth for the final-record adapt loop, called by
    `runner.judge_responses` after the match/review stages.

    Returns (n_written, n_missing_rubric).
    """
    refined_by_key = {
        (r["id"], r.get("trial", 1)): r for r in load_jsonl(refined_post_review_path)
    }
    review_by_key = {
        (r["id"], r.get("trial", 1)): r for r in load_jsonl(review_path)
    } if review_path else {}
    response_lens = _load_response_lens_by_key(responses_path)

    judged_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_missing_rubric = 0
    with judged_path.open("w") as fout:
        for key, ref in refined_by_key.items():
            case_id, trial = key
            rubric = rubrics.get(case_id.split("-", 1)[0])
            if rubric is None:
                n_missing_rubric += 1
                continue
            rec = _build_judged_record(
                ref=ref,
                review_rec=review_by_key.get(key),
                rubric=rubric,
                judge_short=judge_short,
                response_len=response_lens.get((str(case_id), int(trial))),
            )
            validate_record(rec, "judged")
            fout.write(json.dumps(rec) + "\n")
            n_written += 1
    return n_written, n_missing_rubric


def adapt_judge_to_prod(
    refined_record: dict,
    rubric: dict,
) -> dict:
    """Produce a production-schema judge record from strategy-pipeline output.

    Args:
        refined_record: {id, trial, strategies} as produced by
            `match_stage.emit_strategies` (after `apply_overrides_to_refined`
            if a review pass ran).
        rubric: full rubric dict for this case

    Returns:
        dict with keys {options, responseActions, summary} matching
        judge/schemas/judged.schema.json.
    """
    strategies = refined_record.get("strategies", [])

    # responseActions
    response_actions = []
    for s in strategies:
        for sa in s.get("sub_actions", []):
            # responseActions retains the legacy `number` key (consumed by
            # downstream metrics consumers) even though the upstream stage
            # now stores it as `id`.
            n = sa["id"]
            mres = sa.get("match_resolved", [])
            if mres:
                # Anchored to rubric option(s)
                match_str = ",".join(str(x) for x in mres)
                rationale = sa.get("fallback_evidence", "") or ""
            else:
                # Off-rubric sub_action
                rationale = ""
                match_str = ""
            response_actions.append({
                "number": n,
                "action": sa.get("action", ""),
                "category": sa.get("category", ""),
                "match": match_str,
                "rationale": rationale,
            })

    return {
        "options": _build_prod_options(refined_record, rubric),
        "responseActions": response_actions,
    }


def _build_prod_options(refined_record: dict, rubric: dict) -> list[dict]:
    """Construct the prod-shape `options[]` from a refined record + rubric.

    Shared by `adapt_judge_to_prod` and `build_options_only`; making the
    equivalence structural rather than test-policed.
    """
    matched_option_ids: set[int] = set()
    for s in refined_record.get("strategies", []):
        for oid in s.get("matched_options", []):
            matched_option_ids.add(oid)
    partial_option_ids = set(refined_record.get("partial_option_ids", []) or [])
    return [
        {
            "id": opt["id"],
            "matched": opt["id"] in matched_option_ids,
            "partial": opt["id"] in partial_option_ids,
            "evidence": "",
        }
        for opt in rubric["options"]
    ]


def build_options_only(refined_record: dict, rubric: dict) -> dict:
    """Build a review-step input record without responseActions.

    The review prompt only reads `options[].{id, matched, partial, evidence}`
    plus the rubric's static `score` field.
    """
    return {"options": _build_prod_options(refined_record, rubric)}


# Review step: global match review.
# Each override carries a verdict in {yes, partial, no}; we map back to
# (matched, partial) flags and overwrite the per-option fields.
_GLOBAL_VERDICT_TO_FLAGS = {
    "yes": (True, False),
    "partial": (True, True),
    "no": (False, False),
}


def apply_global_match_review(
    prod: dict,
    overrides: list[dict],
) -> dict:
    """Apply review-step global match overrides to a prod-shape record.

    Role:
      The production pipeline applies review overrides at the strategy level
      via `overrides.apply_overrides_to_refined`. For promotions where the reviewer
      cited an extracted action_id and for all demotions, the refined record
      already reflects the new match graph, so this function is idempotent on
      them: it sees `was_matched == new_matched` and the flag-flip branches
      are skipped.

      Where this function is still load-bearing:
        - Unattributed promotions (reviewer emitted action_id=null, or cited
          a hallucinated action_id). The strategy-level apply left the option
          unmatched. This call flips the prod option flag for recall
          accounting (precision is unaffected since no responseAction was
          promoted), and copies the reviewer's verbatim evidence into
          `opt.evidence` for audit.

    Legacy / experimental callers still operate on pre-review prod records,
    where this function is the only place overrides are realized; behavior
    in that mode is unchanged.

    Args:
        prod: dict with `options` and `responseActions`. Not mutated; a new
              dict with replaced `options` AND `responseActions` is returned.
        overrides: list of {option_id, new_verdict, rationale, evidence}
                   from the global-review judge. `new_verdict` is one of
                   {yes, partial, no}. Items absent from overrides keep
                   the upstream verdict.

    Effects (full semantics; many become no-ops in current flow):
      - For each override, set the matching option's `matched` and `partial`
        flags from the new_verdict via `_GLOBAL_VERDICT_TO_FLAGS`.
      - When the verdict flip promotes a previously-unmatched option to
        matched (false -> true via partial / yes), the reviewer's evidence is
        captured in `opt.evidence` so downstream metrics carry an audit trail.
      - When the flip demotes (true -> false via no), opt.evidence is cleared
        AND any responseAction whose `match` string referenced the demoted oid
        has that oid removed from its `match`. If an action's `match` becomes
        empty it becomes off-rubric, which Precision_weighted excludes from its
        denominator. This keeps the precision denominator consistent with the
        new option graph.
      - Promotions (false -> true) do NOT mutate responseActions; we cannot
        reliably identify which action the reviewer credited from evidence
        text alone. Recall picks up the promotion via the option flag flip.

    Coverage semantics: the review-step schema emits overrides
    only; absent options are confirmed at the upstream verdict. The historical
    risk was that "absent" silently conflated "reviewed and
    agreed" with "not reviewed and skipped". Two structural mitigations now
    cover that gap:

    1. The match-stage coverage retry in match_stage.py re-prompts on partial
       nonzero option dropout (`0 < len(emitted_oids) < n_expected`), so the
       reviewer's input always spans the full option set.
    2. The action_id contract on overrides: promotions cite which
       extracted action backs the override, and a null action_id is an
       explicit coverage-gap signal counted in `_promotions_unattributed`.
       Coverage gaps surface as data instead of silent confirms.

    A reviewed_option_ids enumeration would tighten this further but adds
    ~3x output tokens per call without a measured kappa gain; not pursued.
    """
    if not overrides:
        return prod
    by_id = {int(o["option_id"]): o for o in overrides if "option_id" in o}

    new_options = []
    demoted_oids: set[int] = set()
    for opt in prod.get("options", []):
        oid = int(opt["id"])
        ov = by_id.get(oid)
        if not ov:
            new_options.append(opt)
            continue
        new_matched, new_partial = _GLOBAL_VERDICT_TO_FLAGS.get(
            ov["new_verdict"], (bool(opt.get("matched")), bool(opt.get("partial")))
        )
        was_matched = bool(opt.get("matched"))
        new_opt = {**opt, "matched": new_matched, "partial": new_partial}
        if not was_matched and new_matched and ov.get("evidence"):
            new_opt["evidence"] = ov["evidence"]
        if was_matched and not new_matched:
            new_opt["evidence"] = ""
            demoted_oids.add(oid)
        new_options.append(new_opt)

    new_response_actions = prod.get("responseActions", [])
    if demoted_oids:
        rebuilt = []
        for ra in new_response_actions:
            match_str = ra.get("match", "") or ""
            if not match_str:
                rebuilt.append(ra)
                continue
            kept = [
                tok.strip() for tok in match_str.split(",")
                if tok.strip() and int(tok.strip()) not in demoted_oids
            ]
            if len(kept) == len(match_str.split(",")):
                rebuilt.append(ra)
            else:
                rebuilt.append({**ra, "match": ",".join(kept)})
        new_response_actions = rebuilt

    return {**prod, "options": new_options, "responseActions": new_response_actions}
