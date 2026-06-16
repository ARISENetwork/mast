#!/usr/bin/env python3
"""Match: extract response actions + match them to rubric options, then
deterministically group them into strategies.

LLM-stage output: one record per (case, trial) written to
{--out}/{judge_short}.jsonl per `judge/schemas/match.schema.json`.

Grouping output (also written by this stage, post-LLM): one record per
(case, trial) at {--out}/../strategies.jsonl per the strategies schema.
The grouping pass is pure-Python (union-find on `dependsOn`) and runs every
time `match_stage` is invoked so the strategies file always matches the
latest match output. Disable via `--no-emit-strategies`.

Usage:
  python -m judge.stages.match_stage --judge gemini/gemini-3-flash-preview \\
                 --responses-path PATH \\
                 --rubric-dir dataset/rubrics \\
                 --prompt judge/prompts/extract_match.md \\
                 --out OUT_DIR [--threads 20] [--cases C1,C2] [--model-name M]

Production runs call `run_match` directly via `judge.runner.judge_responses`
with explicit prompt and output paths per the active JudgeConfig. The runner
consumes the colocated `strategies.jsonl` directly; no separate group step
needs to run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from .match_helpers import (
    format_rubric_concept_first,
    format_guidance_header,
    load_guidance,
    matcher_judge_name,
    resolve_model_name,
)
from .cache_helpers import prompt_hash, load_completed
from ..schemas import validate_record

DEFAULT_PROMPT = Path(__file__).parent.parent / "prompts/extract_match.md"
DEFAULT_OUT = Path(__file__).parent / "match_results"
DEFAULT_JUDGE = "gemini/gemini-3-flash-preview"


def schema() -> dict:
    """Stage-1 output schema: extracted actions plus a flat per-option
    matched/partial verdict list."""
    action_props = {
        "id": {"type": "integer"},
        "action": {"type": "string"},
        "category": {"type": "string",
                     "enum": ["Diagnostic", "Medication", "Procedure",
                              "Counseling", "Follow-up"]},
        "evidence": {"type": "string"},
        "dependsOn": {"type": "array", "items": {"type": "integer"}},
        "match": {"type": "string"},
    }
    action_required = ["id", "action", "category", "evidence", "dependsOn", "match"]
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": action_props,
                    "required": action_required,
                    "additionalProperties": False,
                },
            },
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "matched": {"type": "boolean"},
                        "partial": {"type": "boolean"},
                        "matched_action_ids": {"type": "array", "items": {"type": "integer"}},
                        "evidence": {"type": "string"},
                    },
                    "required": ["id", "matched", "partial", "matched_action_ids", "evidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["actions", "options"],
        "additionalProperties": False,
    }


def base_case_id(case_id: str) -> str:
    """Strip perturbation suffix to get the rubric key.

    Item ids are `{base}` for the base case or `{base}-{N}` for
    perturbation N. Rubrics are stored per base id.
    """
    return case_id.split("-", 1)[0]


def load_responses(path: Path) -> dict[tuple[str, str, int], str]:
    """Load (model, case_id, trial) -> response_text from a JSONL file.

    `model` is read from the record (default ""), so the same loader serves
    both per-model production runs and pooled multi-model runs.
    """
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        text = r.get("response")
        if not text:
            continue
        out[(r.get("model", ""), r["id"], r["trial"])] = text
    return out


def render_match_prompt(case_id: str, resp_text: str, rubric: dict,
                        prompt_template: str) -> str:
    """Render the match-stage prompt for one (case, trial) tuple.

    Pure-function: same inputs produce the same prompt every time.
    """
    base = base_case_id(case_id)
    guidance = load_guidance(base)
    return (prompt_template
            .replace("{CASE_PRESENTATION}", rubric.get("case", {}).get("presentation", ""))
            .replace("{GUIDANCE}", format_guidance_header(guidance))
            .replace("{RESPONSE}", resp_text)
            .replace("{RUBRIC_OPTIONS}",
                     format_rubric_concept_first(rubric, guidance=guidance)))


def _build_match_record(case: str, trial: int, model: str, jname: str,
                        parsed: dict, runtime: float, usage: dict,
                        stage_prompt_hash: str) -> dict:
    """Assemble the per-(case, trial) match cache record. Single source of
    truth for how a match record serializes."""
    rec = {
        "id": case, "trial": trial, "model": model, "judge": jname,
        "actions": parsed["actions"], "options": parsed["options"],
        "runtime": round(runtime, 1), "usage": usage,
        "prompt_hash": stage_prompt_hash,
    }
    return rec


def coverage_short_oids(parsed: dict, rubric: dict) -> list[int] | None:
    """Return the list of missing option_ids if the emission is short on
    rubric coverage, else None.

    gemini-3-flash occasionally omits options (~0.06% of records). The sync
    worker() uses this to drive an inline coverage retry on just the short
    items.
    """
    n_expected = len(rubric.get("options", []))
    emitted = {int(o["id"]) for o in parsed.get("options", []) if "id" in o}
    if 0 < len(emitted) < n_expected:
        return sorted({int(o["id"]) for o in rubric.get("options", [])} - emitted)
    return None


def coverage_retry_prompt(prompt: str, n_emitted: int, n_expected: int,
                          missing: list[int]) -> str:
    """Prepend the per-item coverage-retry NOTE (with explicit missing
    option_ids) to the original prompt. worker() uses this for its inline
    retry when the extractor emits fewer options than the rubric."""
    return (
        f"NOTE: A prior attempt at this case emitted only {n_emitted} of "
        f"{n_expected} rubric options (missing option_ids: {missing}). You "
        f"MUST emit every option from the rubric, including those that don't "
        f"match the response (matched=false, partial=false, "
        f"matched_action_ids=[], evidence=\"\"). Emit the complete options "
        f"list now.\n\n" + prompt
    )


def make_llm_call(prompt: str, schema: dict, judge_model: str,
                  max_tokens: int = 32768) -> tuple[dict, float, dict]:
    from ..gemini_sdk import is_gemini, sync_call
    if not is_gemini(judge_model):
        raise ValueError(
            f"judge model {judge_model!r} is not a Gemini model; this bundle "
            "judges via the direct google-genai SDK only."
        )
    return sync_call(judge_model, prompt, schema,
                     max_output_tokens=max_tokens, temperature=0.0)


def _gemini_safe_threads(judge_model: str, threads: int) -> int:
    """Cap threads at 20 when judge is direct Google AI Studio.

    gemini-3-flash on Tier 1 has a 2M input-tokens-per-minute cap; 40 threads
    can burst past it, so Gemini judges are capped at 20 threads.
    """
    if judge_model.startswith("gemini/") and threads > 20:
        return 20
    return threads


def run_match(judge_model: str, threads: int, responses: dict,
              rubrics: dict, case_filter: set | None, trial_filter: set | None,
              limit: int | None, prompt_path: Path, out_dir: Path,
              model_name: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_template = prompt_path.read_text()
    stage_prompt_hash = prompt_hash(prompt_template)
    jname = matcher_judge_name(judge_model)
    out_path = out_dir / f"{jname}.jsonl"
    completed = load_completed(out_path, rubrics, expected_prompt_hash=stage_prompt_hash)
    if model_name is not None:
        completed = {(model_name, c, t) for (_m, c, t) in completed}

    tasks = []
    for (rec_model, case, trial), resp_text in sorted(responses.items()):
        effective = model_name if model_name is not None else rec_model
        if (effective, case, trial) in completed:
            continue
        if case_filter and case not in case_filter:
            continue
        if trial_filter is not None and trial not in trial_filter:
            continue
        if base_case_id(case) not in rubrics:
            continue
        tasks.append((effective, case, trial, resp_text))
    if limit:
        tasks = tasks[:limit]
    print(f"[match] {jname}: pending={len(tasks)} completed={len(completed)}",
          file=sys.stderr)
    if not tasks:
        return

    write_lock = Lock()
    done_lock = Lock()
    done = 0

    def worker(model: str, case: str, trial: int, resp_text: str) -> None:
        nonlocal done
        base = base_case_id(case)
        rubric = rubrics[base]
        prompt = render_match_prompt(case, resp_text, rubric, prompt_template)
        for attempt in range(2):
            try:
                parsed, runtime, usage = make_llm_call(
                    prompt,
                    schema(),
                    judge_model,
                )
                break
            except Exception as e:
                if attempt == 0:
                    time.sleep(2)
                else:
                    print(f"[match FAIL] {model}/{case}/t{trial}: {type(e).__name__}: {str(e)[:200]}",
                          file=sys.stderr)
                    return
        # Coverage retry: re-prompt once when emission is short on rubric
        # options (~0.06% of records on gemini-3-flash).
        missing = coverage_short_oids(parsed, rubric)
        if missing is not None:
            emitted_n = len(parsed.get("options", []))
            n_expected = len(rubric.get("options", []))
            try:
                retry_parsed, retry_runtime, retry_usage = make_llm_call(
                    coverage_retry_prompt(prompt, emitted_n, n_expected, missing),
                    schema(),
                    judge_model,
                )
                retry_oids = {int(o["id"]) for o in retry_parsed.get("options", []) if "id" in o}
                if len(retry_oids) > emitted_n:
                    print(f"  [match coverage retry] {model}/{case}/t{trial}: "
                          f"{emitted_n} -> {len(retry_oids)} options",
                          file=sys.stderr)
                    parsed = retry_parsed
                    runtime += retry_runtime
                    for k in ("prompt_tokens", "completion_tokens",
                              "input_tokens", "output_tokens"):
                        if k in retry_usage:
                            usage[k] = (usage.get(k) or 0) + (retry_usage.get(k) or 0)
            except Exception as e:
                print(f"  [match coverage retry FAIL] {model}/{case}/t{trial}: "
                      f"{type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        rec = _build_match_record(case, trial, model, jname, parsed,
                                  runtime, usage, stage_prompt_hash)
        validate_record(rec, "match")
        with write_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        with done_lock:
            done += 1
            d = done
        n_match = sum(1 for o in parsed["options"] if o["matched"])
        n_acts = len(parsed["actions"])
        tag = f"{model}/" if model else ""
        print(f"  [match {d}/{len(tasks)}] {tag}{case}/t{trial}  "
              f"actions={n_acts} matched={n_match}/{len(parsed['options'])}  "
              f"{runtime:.1f}s", file=sys.stderr)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=_gemini_safe_threads(judge_model, threads)) as pool:
        futures = {pool.submit(worker, m, c, t, r): (m, c, t) for m, c, t, r in tasks}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"Unexpected: {e}", file=sys.stderr)
    print(f"[match] done in {time.time() - t0:.0f}s", file=sys.stderr)


def emit_strategies(match_path: Path, strategies_path: Path) -> int:
    """Run deterministic union-find grouping over the match output and write
    strategies.jsonl. Pure Python; returns the number of records written.
    """
    from .group_into_strategies import process_record
    records: list[dict] = []
    for line in match_path.read_text().splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    strategies_path.parent.mkdir(parents=True, exist_ok=True)
    n_strategies = 0
    with strategies_path.open("w") as f:
        for rec in records:
            processed = process_record(rec)
            validate_record(processed, "strategies")
            f.write(json.dumps(processed) + "\n")
            n_strategies += len(processed["strategies"])
    print(f"[group] wrote {len(records)} records ({n_strategies} strategies) "
          f"to {strategies_path}", file=sys.stderr)
    return len(records)


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / ".env")
    except ImportError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cases", default="")
    ap.add_argument("--trials", default="")
    ap.add_argument("--responses-path", required=True,
                    help="Path to responses JSONL ({id, trial, response}/line).")
    ap.add_argument("--rubric-dir", required=True,
                    help="Directory of per-case rubric JSON files.")
    ap.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model-name", default=None,
                    help="Canonical model name to stamp into output records + "
                         "use as the cache key. Auto-inferred from --out if "
                         "the path is under `.../_strategy/<model>/...`.")
    ap.add_argument("--emit-strategies", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="After the match LLM pass, run deterministic "
                         "grouping and write strategies.jsonl one level above "
                         "--out. Default ON.")
    ap.add_argument("--strategies-out", default=None,
                    help="Override the strategies.jsonl output path. "
                         "Default: <out>/../strategies.jsonl (i.e. alongside "
                         "the match/<judge>/ subdir).")
    args = ap.parse_args()

    prompt_path = Path(args.prompt)
    out_dir = Path(args.out)

    model_name = resolve_model_name(args.model_name, out_dir, None, None)
    if model_name:
        print(f"[match_stage] effective model_name = {model_name!r}", file=sys.stderr)

    responses = load_responses(Path(args.responses_path))
    print(f"Loaded {len(responses)} responses from {args.responses_path}",
          file=sys.stderr)
    rubrics = {}
    for p in Path(args.rubric_dir).glob("*.json"):
        r = json.loads(p.read_text())
        rubrics[r["id"]] = r

    case_filter = ({c.strip() for c in args.cases.split(",") if c.strip()}
                   if args.cases else None)
    trial_filter = ({int(t.strip()) for t in args.trials.split(",") if t.strip()}
                    if args.trials else None)

    run_match(args.judge, args.threads, responses, rubrics,
              case_filter, trial_filter, args.limit,
              prompt_path=prompt_path, out_dir=out_dir,
              model_name=model_name)

    if args.emit_strategies:
        match_path = out_dir / f"{matcher_judge_name(args.judge)}.jsonl"
        strategies_path = (Path(args.strategies_out) if args.strategies_out
                           else out_dir.parent / "strategies.jsonl")
        if match_path.exists():
            emit_strategies(match_path, strategies_path)
        else:
            print(f"[group] skipped (no match output at {match_path})",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
