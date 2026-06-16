# donoharm-judge

LLM judge pipeline that scores free-text clinical responses against a
per-case rubric and returns harm metrics (F1_weighted, severity rates).
Two LLM stages (match, review) wrapped behind
one `judge_responses(...)` call. Validated against physician adjudication labels; see the NOHARM
paper for agreement statistics.

This is the open-source extraction of the donoharm judge used in MAST
(Medical AI Standardized Testing). Inside MAST the package is imported
as `benchmarks.donoharm.judge`; installed via this `pyproject.toml`
it's importable as `donoharm_judge`.

## Install

```bash
# From the benchmark directory:
pip install -e judge
```

Judges run on Gemini via `google-genai` directly (for reliable
`response_json_schema` support).

## Quickstart

```python
from pathlib import Path
from donoharm_judge import judge_responses, JudgeConfig

# responses: one record per (case_id, trial) with the model's free-text answer
responses = [
    {"id": "Card001", "trial": 1, "response": "Order 12-lead ECG, troponin..."},
    # ...
]

# rubrics: keyed by base case_id; each rubric is a dict per the upstream
# donoharm schema (case presentation, options[], specialistConsult)
rubrics = {"Card001": {...}}  # see dataset/rubrics/ for the schema

summary = judge_responses(
    model_name="my-llm",
    responses=responses,
    rubrics=rubrics,
    config=JudgeConfig(
        cache_root=Path("./cache"),
        review_judge="gemini/gemini-3.5-flash",
    ),
    judged_path=Path("./judged.jsonl"),
)

print(summary)              # record counts + elapsed
# Per-record metrics (F1_weighted, harm tier rates, ...) live in judged.jsonl
```

## Architecture

```
   responses + rubrics
           |
           v
  +-------------------+
  | match (extract +  |   Gemini direct, sync
  | rubric mapping)   |
  +-------------------+
           |
           v
  +-------------------+
  | review            |   Gemini 3.5 flash (validated reviewer)
  | (override pass)   |
  +-------------------+
           |
           v
  +-------------------+
  | apply overrides   |   deterministic, no LLM call
  +-------------------+
           |
           v
        adapter -> judged.jsonl
```

Judging runs synchronously: a thread pool issues per-record judge calls
and returns once the cohort is scored.

## Configuration

`JudgeConfig` is a frozen dataclass. The only required field is
`cache_root`; everything else has a default (see the knobs table below).

The remainder of this document is the operational reference (API
details, cache layout, schemas, provider routing).

## Public API

```python
from donoharm_judge import (
    judge_responses,
    JudgeConfig,
    JudgeRunSummary,
)

summary = judge_responses(
    model_name="claude-sonnet-4-6",
    responses=responses_list,        # [{id, trial, response}, ...]
    rubrics=rubrics_by_case_id,       # {case_id: rubric_dict}
    config=JudgeConfig(
        cache_root=Path("results/raw/donoharm/default/_strategy"),
    ),
    judged_path=Path("results/raw/donoharm/default/claude-sonnet-4-6_judged.jsonl"),
)
```

`responses` and `rubrics` come in pre-loaded by the caller; the pipeline
never reads them from disk. `judged_path` is written by the pipeline (one
record per `(case_id, trial)`). `JudgeRunSummary` carries overall record
counts + elapsed; re-read `judged_path` for the records themselves.

### `JudgeConfig` knobs

| Field | Default | Notes |
|---|---|---|
| `cache_root` | (required) | Intermediate stage outputs live under `{cache_root}/{model_name}/...`. MAST production uses `{raw_dir}/_strategy`. |
| `match_judge` | `gemini/gemini-3-flash-preview` | Match-stage judge: extracts response actions + maps to rubric options. |
| `match_prompt` | `prompts/extract_match.md` | |
| `review_judge` | `gemini/gemini-3.5-flash` | Review. `None` disables. |
| `review_prompt` | `prompts/global_match_review.md` | |
| `threads` | 40 | Per-stage thread pool. Provider-cap rules applied internally (Gemini direct AI Studio capped at 20). |
| `cases_filter` | `None` | Optional case-id allowlist for partial runs. |

## Pipeline steps

Order: `match -> review -> apply_overrides -> adapt -> write judged`.

| Step | Module | Output cache path | Schema |
|---|---|---|---|
| Match (extract + group) | `stages/match_stage.py:run_match` (also runs `emit_strategies` inline) | `match/<extractor>.jsonl` + `strategies.jsonl` | `schemas/match.schema.json`, `schemas/strategies.schema.json` |
| Review | `stages/review_stage.py:run_review` | `review_<reviewer>/<reviewer>.jsonl` | `schemas/review.schema.json` |
| Apply overrides | `overrides.py` | `strategies_post_review.jsonl` | (strategies schema + telemetry) |
| Adapter | `adapter.py` | (in-memory) | n/a |
| Write final | `runner.py` | `judged_path` | `schemas/judged.schema.json` |

Stage functions are called directly as library functions from `runner.py`;
each also exposes a `main()` for ad-hoc CLI invocation via
`python -m benchmarks.donoharm.judge.stages.<stage>`.

Review produces an absent-means-confirmed override list that
`apply_overrides_to_strategies` (`overrides.py`) materializes into the
strategies graph. Silent-confirm risk (unreviewed options defaulting to
"confirmed match") is mitigated by the match-stage coverage retry.

## Cache layout

```
{cache_root}/{model_name}/
├── responses.jsonl                     materialized input
├── match/<extractor>.jsonl             match
├── strategies.jsonl                    deterministic grouping (emitted by match)
├── strategies_post_review.jsonl        review applied
└── review_<reviewer>/<reviewer>.jsonl  review
```

Resume semantics: each stage reads its own output file as a
`(case_id, trial) -> record` dict and skips inputs already present.
Staleness detected via `prompt_hash`. Invalidation rule: drop a cached
record only when `cached_oids - current_oids` is non-empty (cached
references option IDs the rubric no longer has).

## Final `_judged.jsonl` schema

`schemas/judged.schema.json` is authoritative. Per-record fields, in
short:

- `id`, `trial`, `judge`
- `options[]`: per-rubric-option verdict after review overrides
  (`{id, matched, partial, evidence?}`)
- `responseActions[]`: model's actions with `{number, action, category,
  match, score?, rationale?}`
- `summary`: free-text per-record note; commonly empty
- `harm[]`, `nonrubric_harms[]`: from `metrics.compute_harm` /
  `metrics.compute_nonrubric_harms`
- `metrics{}`: `F1_raw`/`F1_binary`/`F1_weighted`,
  `F1_uncapped`/`F1_matched`/`F1_capped`, `Precision_*` (incl.
  `Precision_matched`), `Recall_*`, `Offrubric_rate`, `Accuracy`,
  `Accuracy_binary`, `Severe_rate`, `Moderate_rate`, `Mild_rate`.
  **Headline metric is `F1_weighted`**: matched (off-rubric excluded)
  severity-weighted F1 with a Severe cap. `F1_uncapped` is the pre-2026-06
  headline (uncapped partial F1), kept for comparison.
- `global_match_reviewer`: short name of the review judge (may be `null`)
- `runtime`, `grader_usage`, `grader_latency_ms`: always `null` at this
  layer (per-stage timing lives in stage outputs)

## Provider routing

Judges run on Gemini only, via `gemini_sdk.py` (the google-genai SDK
directly, using `sync_call`). The match and review stages reject a
non-Gemini judge model id.

## Files

- `runner.py`: `judge_responses` orchestrator. Calls stage `run_*` functions directly (no subprocess).
- `config.py`: `JudgeConfig`, stage defaults. **Single source of truth for defaults.**
- `summary.py`: `JudgeRunSummary`.
- `gemini_sdk.py`: direct google-genai SDK calls (`sync_call`). Schema conversion via `to_openapi()`.
- `metrics.py`: `compute_harm`, `compute_metrics_for_case`, `compute_nonrubric_harms`, `get_option_score`, harm-weight constants.
- `adapter.py`: stage-output to `_judged.jsonl` record adapter; `apply_global_match_review` materializer.
- `overrides.py`: review override application onto strategies.
- `io.py`: small JSONL read/write helpers.
- `stages/`: per-stage library functions (`run_match`, `run_review`) + helpers (`cache_helpers.py` for prompt-hash resume, `match_helpers.py` for prompt rendering, `group_into_strategies.py` for the deterministic post-match grouping pass). Each stage also exposes a thin CLI via `main()`.
- `prompts/`: production prompts (`extract_match.md`, `global_match_review.md`).
- `schemas/`: four JSON Schema files enforcing the I/O contracts plus a small loader (`load_schema(name)`, `all_schemas()`, `validate_record(rec, name)`). Validation runs at every stage write site; jsonschema is a hard dep.
