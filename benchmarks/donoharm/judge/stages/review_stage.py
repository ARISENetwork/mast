#!/usr/bin/env python3
"""Review: global match-verdict review.

Reviews every option's match verdict (yes/partial/no) holistically using a
strong judge, with authority to override the match pass.

Output: per-(case, trial) JSONL with `overrides: [{option_id, new_verdict,
rationale, evidence}]`. Items not in `overrides` are silently confirmed.

Usage:
  python -m judge.stages.review_stage \
    --judge anthropic/claude-sonnet-4-6 \
    --strategies-refined .../strategies_refined.jsonl \
    --match-input .../match/<extractor>.jsonl \
    --responses-path .../responses.jsonl \
    --rubric-dir dataset/rubrics \
    --prompt judge/prompts/global_match_review.md \
    --out .../review_<reviewer>/ \
    --threads 8

`--match-input` is the match-stage output. The reviewer renders the actions
list into the prompt so it can cite an action_id when emitting a promotion
override. If `--match-input` is omitted or a record is missing, the actions
block falls back to "(no extracted actions available)" and the reviewer is
expected to emit `action_id: null` for any promotion it makes.

Review reads only `options[].matched/partial/evidence` plus the rubric's
static `score`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from .match_helpers import (
    format_guidance_header,
    format_rubric_concept_first,
    load_guidance,
    matcher_judge_name,
    resolve_model_name,
)
from ..adapter import build_options_only

VERDICT_FROM_FLAGS = {
    (True, False): "yes",
    (True, True): "partial",
    (False, True): "partial",
    (False, False): "no",
}


def review_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "overrides": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "option_id": {"type": "integer"},
                        "new_verdict": {
                            "type": "string",
                            "enum": ["yes", "partial", "no"],
                        },
                        "rationale": {"type": "string"},
                        "evidence": {"type": ["string", "null"]},
                        "action_id": {"type": ["integer", "null"]},
                    },
                    "required": ["option_id", "new_verdict", "rationale", "evidence", "action_id"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["overrides"],
        "additionalProperties": False,
    }


def format_actions_block(actions: list[dict]) -> str:
    """Render match-stage actions[] for the reviewer prompt.

    Each line carries the id the reviewer will cite when emitting a promotion
    override, plus enough text and evidence quote for the reviewer to ground
    in the response.
    """
    if not actions:
        return "(no extracted actions available)"
    lines = []
    for a in actions:
        aid = a.get("id")
        if aid is None:
            continue
        cat = a.get("category", "")
        text = (a.get("action") or "").strip()
        evi = (a.get("evidence") or "").strip()
        prefix = f"- **#{aid}**"
        if cat:
            prefix += f" [{cat}]"
        line = f"{prefix} {text}"
        if evi:
            line += f"\n  evidence: \"{evi[:200]}\""
        lines.append(line)
    return "\n".join(lines) if lines else "(no extracted actions available)"


def load_responses(path: Path) -> dict[tuple[str, str, int], str]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        text = r.get("response")
        if not text:
            continue
        out[(r.get("model", ""), r["id"], r.get("trial", 1))] = text
    return out


def load_jsonl_indexed(path: Path) -> dict[tuple[str, str, int], dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[(r.get("model", ""), r["id"], r.get("trial", 1))] = r
    return out


from .cache_helpers import (  # noqa: E402
    prompt_hash, load_completed,
)
from ..schemas import validate_record  # noqa: E402


def format_options_with_verdicts(prod: dict, rubric: dict,
                                 guidance: dict | None = None) -> str:
    """Render the rubric options grouped by tier with each option's current
    upstream verdict, evidence, AND case-specific guidance (if present) inline.
    """
    opt_lookup = {int(o["id"]): o for o in prod.get("options", [])}
    by_tier = {"appropriate (≥7)": [], "uncertain (4-6)": [], "harm (≤3)": []}
    guidance_options = (guidance or {}).get("options") or {}
    for o in rubric.get("options", []):
        oid = int(o["id"])
        score = int(o.get("score") or 0)
        tier = ("appropriate (≥7)" if score >= 7
                else "harm (≤3)" if score <= 3
                else "uncertain (4-6)")
        po = opt_lookup.get(oid, {})
        verdict = VERDICT_FROM_FLAGS[
            (bool(po.get("matched")), bool(po.get("partial")))
        ]
        evidence = (po.get("evidence") or "").strip()
        evi = f' — evidence: "{evidence[:200]}"' if evidence else ""
        cat = o.get("category", "")
        line = (f"- [opt {oid}, score {score}, {cat}] **upstream: {verdict}** — "
                f"{o['text'][:300]}{evi}")
        # Case-specific per-option guidance (option ids are int OR str in YAML)
        opt_guide = guidance_options.get(oid) or guidance_options.get(str(oid))
        if opt_guide and isinstance(opt_guide, dict):
            matching = (opt_guide.get("matching") or "").strip()
            if matching:
                # Indent so the guidance reads as a sub-bullet under the option
                indented = "    " + matching.replace("\n", "\n    ")
                line += f"\n  > **Case-specific guidance**:\n{indented}"
        by_tier[tier].append(line)
    out = []
    for tier in ("harm (≤3)", "uncertain (4-6)", "appropriate (≥7)"):
        if not by_tier[tier]:
            continue
        out.append(f"### {tier}")
        out.extend(by_tier[tier])
        out.append("")
    return "\n".join(out)


# Gemini structured-output null-loop trap: under greedy decoding the model
# can emit `{"overrides": [null, null, null, ...]}` until MAX_TOKENS. The
# schema disallows null items, but Gemini's enforcement is best-effort and
# autoregressive context inertia locks in the repetition. Detect the
# pattern at parse time and retry with bumped temperature (sampling breaks
# the loop).
_NULL_LOOP_PAT = re.compile(r'^\s*\{\s*"overrides"\s*:\s*\[\s*((?:null\s*,?\s*)+)', re.DOTALL)


def is_null_loop(text: str) -> bool:
    if not text or len(text) < 30:
        return False
    if '"option_id"' in text:
        return False
    m = _NULL_LOOP_PAT.match(text)
    if not m:
        return False
    return m.group(1).count("null") >= 10


# Temperature ladder for null-loop retry: greedy first (production
# determinism), then escalate. Capped at temp=0.4 to keep override quality
# from drifting too far from the deterministic baseline.
_NULL_LOOP_TEMP_LADDER = (0.0, 0.2, 0.4)


def _gemini_thinking_level_from_override(override: str | None) -> str | None:
    """Map DONOHARM_REASONING_EFFORT_OVERRIDE values onto Gemini thinking levels.

    None / unset -> LOW (production default; matches validation run).
    'off' -> 'OFF' (Gemini SDK accepts this).
    'minimal' / 'low' / 'medium' / 'high' -> uppercase passthrough.
    """
    if not override:
        return None
    return override.upper()


def _make_llm_call_gemini(
    prompt: str, schema: dict, judge_model: str, max_tokens: int,
) -> tuple[dict, float, dict, str | None]:
    """Run one review call via the Gemini SDK.

    Returns (parsed, runtime, usage, anomaly); the null-loop temperature
    ladder re-tries on degenerate all-null outputs.
    """
    from ..gemini_sdk import sync_call_raw

    override = os.environ.get("DONOHARM_REASONING_EFFORT_OVERRIDE")
    thinking_level = _gemini_thinking_level_from_override(override) or "LOW"

    runtime_total = 0.0
    anomaly: str | None = None
    last_usage: dict = {}
    last_text: str = ""
    for ladder_idx, temp in enumerate(_NULL_LOOP_TEMP_LADDER):
        last_text, dt, last_usage = sync_call_raw(
            judge_model, prompt, schema,
            thinking_level=thinking_level,
            max_output_tokens=max_tokens,
            temperature=temp,
        )
        runtime_total += dt
        try:
            parsed = json.loads(last_text)
            return parsed, runtime_total, last_usage, anomaly
        except json.JSONDecodeError:
            if is_null_loop(last_text):
                if ladder_idx + 1 < len(_NULL_LOOP_TEMP_LADDER):
                    print(f"[review_stage] null-loop detected at temp={temp}; "
                          f"retrying at temp={_NULL_LOOP_TEMP_LADDER[ladder_idx+1]}",
                          file=sys.stderr)
                    anomaly = "null_loop_retry"
                    continue
                print(f"[review_stage] null-loop unresolved after temp ladder "
                      f"{_NULL_LOOP_TEMP_LADDER}; normalizing to empty overrides",
                      file=sys.stderr)
                return {"overrides": []}, runtime_total, last_usage, "null_loop_unresolved"
            raise
    raise RuntimeError("_make_llm_call_gemini: temperature ladder exhausted")


def make_llm_call(prompt: str, schema: dict, judge_model: str,
                  max_tokens: int = 16384) -> tuple[dict, float, dict, str | None]:
    from ..gemini_sdk import is_gemini
    if not is_gemini(judge_model):
        raise ValueError(
            f"judge model {judge_model!r} is not a Gemini model; this bundle "
            "judges via the direct google-genai SDK only."
        )
    return _make_llm_call_gemini(prompt, schema, judge_model, max_tokens)

def render_review_prompt(
    case_id: str, resp_text: str, rubric: dict, prod: dict,
    actions: list[dict], prompt_template: str,
) -> str:
    """Render the review prompt (used by the sync worker)."""
    base = case_id.split("-", 1)[0]
    guidance = load_guidance(base)
    return (
        prompt_template
        .replace("{CASE_PRESENTATION}", rubric.get("case", {}).get("presentation", ""))
        .replace("{EXPERT_RECOMMENDATIONS}", rubric.get("case", {}).get("recommendations", ""))
        .replace("{RESPONSE}", resp_text)
        .replace("{GUIDANCE}", format_guidance_header(guidance))
        .replace("{ACTIONS}", format_actions_block(actions))
        .replace("{RUBRIC_OPTIONS_WITH_VERDICTS}",
                 format_options_with_verdicts(prod, rubric, guidance))
        .replace("{RUBRIC_OPTIONS}",
                 format_rubric_concept_first(rubric, guidance=guidance))
    )


def _clean_overrides(parsed: dict, rubric: dict, actions: list[dict]) -> list[dict]:
    """Sanitize the reviewer's overrides[] list against valid option/action
    ids. Hallucinated action_ids drop to null."""
    valid_ids = {int(o["id"]) for o in rubric.get("options", [])}
    valid_action_ids = {int(a["id"]) for a in actions if "id" in a}
    clean: list[dict] = []
    for o in parsed.get("overrides", []):
        if not isinstance(o.get("option_id"), int) or o["option_id"] not in valid_ids:
            continue
        aid = o.get("action_id")
        if not isinstance(aid, int) or aid not in valid_action_ids:
            aid = None
        clean.append({
            "option_id": o["option_id"],
            "new_verdict": o["new_verdict"],
            "rationale": o.get("rationale", ""),
            "evidence": o.get("evidence"),
            "action_id": aid,
        })
    return clean


def _build_review_record(
    case_id: str, trial: int, model: str, jname: str,
    clean_overrides: list[dict], runtime: float, usage: dict,
    stage_prompt_hash: str, anomaly: str | None = None,
) -> dict:
    """Assemble the per-(case, trial) review cache record."""
    rec = {
        "id": case_id, "trial": trial, "model": model, "judge": jname,
        "overrides": clean_overrides, "runtime": round(runtime, 1),
        "usage": usage, "prompt_hash": stage_prompt_hash,
    }
    if anomaly:
        rec["_review_anomaly"] = anomaly
    return rec


def run_review(
    *,
    judge_model: str,
    refined: dict,
    responses: dict,
    rubrics: dict,
    match_recs: dict,
    prompt_path: Path,
    out_dir: Path,
    threads: int = 8,
    model_name: str | None = None,
    case_filter: set[str] | None = None,
    limit: int | None = None,
) -> Path:
    """Library entry point for the review stage.

    `refined`, `responses`, `match_recs` are keyed by (model, case_id, trial).
    When `model_name` is provided, all three dicts are rekeyed to that model
    name (matches the single-model production path). Returns the output JSONL
    path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_template = prompt_path.read_text()
    stage_prompt_hash = prompt_hash(prompt_template)

    jname = matcher_judge_name(judge_model)
    out_path = out_dir / f"{jname}.jsonl"

    if model_name:
        print(f"[global-match-review] effective model_name = {model_name!r}", file=sys.stderr)

    completed = load_completed(out_path, expected_prompt_hash=stage_prompt_hash)
    if model_name is not None:
        completed = {(model_name, c, t) for (_m, c, t) in completed}
        responses = {(model_name, c, t): v for (_m, c, t), v in responses.items()}
        refined = {(model_name, c, t): v for (_m, c, t), v in refined.items()}
        match_recs = {(model_name, c, t): v for (_m, c, t), v in match_recs.items()}
    print(f"[global-match-review] {jname}: completed={len(completed)}", file=sys.stderr)
    print(f"[global-match-review] refined={len(refined)} responses={len(responses)} "
          f"match_recs={len(match_recs)}",
          file=sys.stderr)

    tasks = []
    for key, ref in refined.items():
        model, case_id, trial = key
        if key in completed:
            continue
        if case_filter and case_id not in case_filter:
            continue
        base = case_id.split("-", 1)[0]
        rubric = rubrics.get(base)
        if rubric is None:
            continue
        if key not in responses:
            continue
        prod = build_options_only(ref, rubric)
        match_rec = match_recs.get(key)
        actions = (match_rec or {}).get("actions", []) or []
        tasks.append((key, rubric, prod, responses[key], actions))

    if limit:
        tasks = tasks[:limit]

    print(f"[global-match-review] {jname}: pending={len(tasks)}", file=sys.stderr)
    if not tasks:
        return out_path

    write_lock = Lock()
    done_lock = Lock()
    done = 0

    def worker(task: tuple) -> None:
        nonlocal done
        key, rubric, prod, resp_text, actions = task
        model, case_id, trial = key
        prompt = render_review_prompt(case_id, resp_text, rubric, prod,
                                      actions, prompt_template)
        anomaly = None
        for attempt in range(2):
            try:
                parsed, runtime, usage, anomaly = make_llm_call(prompt, review_schema(), judge_model)
                break
            except Exception as e:
                if attempt == 0:
                    time.sleep(2)
                else:
                    print(f"[FAIL] {model}/{case_id}/t{trial}: {type(e).__name__}: {str(e)[:200]}",
                          file=sys.stderr)
                    return
        clean = _clean_overrides(parsed, rubric, actions)
        rec_out = _build_review_record(case_id, trial, model, jname, clean,
                                       runtime, usage, stage_prompt_hash,
                                       anomaly=anomaly)
        validate_record(rec_out, "review")
        with write_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec_out) + "\n")
        with done_lock:
            done += 1
            d = done
        tag = f"{model}/" if model else ""
        print(f"  [global-match-review {d}/{len(tasks)}] {tag}{case_id}/t{trial}  "
              f"overrides={len(clean)}  {runtime:.1f}s", file=sys.stderr)

    t0 = time.time()
    # Same Gemini thread cap as sibling stages.
    effective_threads = (min(threads, 20)
                         if judge_model.startswith("gemini/") else threads)
    with ThreadPoolExecutor(max_workers=effective_threads) as pool:
        futures = {pool.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"Unexpected: {e}", file=sys.stderr)
    print(f"[global-match-review] done in {time.time()-t0:.0f}s", file=sys.stderr)
    return out_path


def main() -> int:
    # CLI entry point: thin loader wrapping run_review.
    try:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / ".env")
    except ImportError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--strategies-refined", required=True)
    ap.add_argument("--match-input", default=None,
                    help="Path to match-stage output JSONL; provides "
                         "actions[] per (case, trial) for the {ACTIONS} block "
                         "in the reviewer prompt.")
    ap.add_argument("--responses-path", required=True)
    ap.add_argument("--rubric-dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cases", default="")
    ap.add_argument("--model-name", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    model_name = resolve_model_name(args.model_name, out_dir)
    responses = load_responses(Path(args.responses_path))
    rubrics = {
        json.loads(p.read_text())["id"]: json.loads(p.read_text())
        for p in Path(args.rubric_dir).glob("*.json")
    }
    refined = load_jsonl_indexed(Path(args.strategies_refined))
    match_recs = load_jsonl_indexed(Path(args.match_input)) if args.match_input else {}
    case_filter = (
        {c.strip() for c in args.cases.split(",") if c.strip()} if args.cases else None
    )

    run_review(
        judge_model=args.judge,
        refined=refined, responses=responses, rubrics=rubrics, match_recs=match_recs,
        prompt_path=Path(args.prompt),
        out_dir=out_dir,
        threads=args.threads,
        model_name=model_name,
        case_filter=case_filter,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
