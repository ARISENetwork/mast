"""Small I/O + naming helpers shared between the judge pipeline and its callers.

Public surface so `score.py` (and any future caller) can materialize a
responses file in the same shape the stage scripts read, and resolve
cache-path-safe judge names, without dipping into runner internals.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    """Tolerant JSONL loader. Skips empty + malformed lines (truncated writes
    from a killed worker leave a partial last line)."""
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            log.warning("[load_jsonl] dropping malformed line in %s: %s",
                        path.name, e)
    return out


def judge_short_name(judge_model_id: str) -> str:
    """Strip vendor prefixes and sanitize remaining slashes so the result is
    path-safe (used as cache subdir names like `review_{judge_short}/`).

    Mirrors `matcher_judge_name` in `stages/match_helpers.py`; keep the two
    in sync so cache paths line up across orchestrator and stage scripts.
    """
    name = (judge_model_id
            .replace("openrouter/anthropic/", "")
            .replace("openrouter/google/", "")
            .replace("openrouter/openai/", "")
            .replace("anthropic/", "")
            .replace("gemini/", "")
            .replace("-preview", ""))
    return name.replace("/", "-")


def derive_record_judge_short(
    review_judge: str | None,
) -> str:
    """Pick the `judge` short name to stamp into each final judged record.

    Precedence: review > "match_only". score.py auto-discovers the judge
    set from the file rather than reconstructing it, so this label drives
    downstream filtering.
    """
    if review_judge is not None:
        return judge_short_name(review_judge)
    return "match_only"


def responses_to_dict(
    responses: list[dict],
    model_name: str | None = None,
) -> dict[tuple[str, str, int], str]:
    """Convert a list of {id, trial, response} records to a (model, id, trial)
    keyed dict. When `model_name` is set, it stamps every key (matches the
    single-model production path); otherwise the per-record `model` field
    is used. Records with empty `response` are dropped.
    """
    out: dict[tuple[str, str, int], str] = {}
    for r in responses:
        if not r.get("response"):
            continue
        m = model_name if model_name is not None else r.get("model", "")
        out[(m, r["id"], r.get("trial", 1))] = r["response"]
    return out


def write_responses_file(responses: list[dict], path: Path) -> None:
    """Emit a responses JSONL in the shape `match_stage.load_responses` expects.

    One `{id, trial, response}` record per line. Model name is NOT carried
    here; downstream steps receive it via `--model-name` (and also infer
    it from their cache path under `_strategy/{model}/...`). Keeping the
    file format model-less means existing match / review caches
    built before the model field landed stay compatible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in responses:
            f.write(json.dumps({
                "id": r["id"],
                "trial": r.get("trial", 1),
                "response": r["response"],
            }) + "\n")
