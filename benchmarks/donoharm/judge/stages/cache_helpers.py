"""Shared cache helpers for the donoharm judge-pipeline eval scripts.

prompt_hash(text)
    Deterministic 12-hex-char sha256 of a prompt template. Cached records
    store this; load_completed drops records whose stored hash doesn't
    match the current template (silent re-judge on prompt edits).

load_completed(path, rubrics=None, expected_prompt_hash=None)
    Read an existing per-stage JSONL, returning the set of (model, case,
    trial) tuples already cached. If `rubrics` is provided, drop records
    whose options[] id set doesn't match the current rubric (cache
    invalidation on upstream rubric edits). If `expected_prompt_hash` is
    provided, drop records whose stored prompt_hash differs. In both
    cases, rewrites the file in place without the stale records.

resolve_model_name(explicit, *cache_paths)
    Resolve the model name from an explicit arg or by inspecting cache
    paths that contain `.../<prompt>/_strategy/<model>/...`. Returns None
    in pooled / experimental mode so callers fall back to per-record
    `model` fields.

Note: match_helpers.py (this directory) has its own load_completed returning
a 2-tuple (case_id, trial) set; that one is unrelated and stays in place
for match_eval's internal use.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def prompt_hash(text: str) -> str:
    """Stable short hash of prompt content. 12 hex chars of sha256."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def load_completed(
    path: Path,
    rubrics: dict | None = None,
    expected_prompt_hash: str | None = None,
) -> set[tuple[str, str, int]]:
    """Return (model, case, trial) tuples already cached at `path`.

    When `rubrics` is provided, drop any cached record that references
    option ids NOT present in the current rubric (i.e. cached_oids has
    entries the rubric doesn't), and rewrite `path` without those stale
    records. This catches cache entries left behind when a rubric was
    edited upstream (option removed or renumbered): those records carry
    ids the rubric no longer recognizes and would surface as orphan harm
    entries in the final `_judged.jsonl`.

    Records that are merely a SUBSET of the current rubric (cached is
    missing one or more current ids) are kept. That pattern is usually a
    judge omitting an option in its emission, not a rubric edit. Dropping
    those records creates an infinite re-judge loop because the judge
    typically reproduces the same subset on re-run; coverage gaps should
    be addressed at the prompt/judge level, not via cache invalidation.

    When `expected_prompt_hash` is provided, drop any cached record whose
    stored `prompt_hash` doesn't match (silent re-judge on prompt edits).
    Records without a `prompt_hash` field are kept for back-compat with
    caches written before the field was introduced; the first re-run
    after that backfill will stamp them on rewrite.

    Cascade caveat: rubric invalidation only works on stages whose cached
    records carry `options[]` (match). Grouping records don't reference
    rubric ids directly, so they can hold orphaned entries pointing at
    (model, case, trial) tuples whose match-stage source was re-judged. After
    a rubric edit, also delete grouping cache files (or rerun those
    stages with `--force`) to fully refresh the pipeline.
    """
    if not path.exists():
        return set()
    out: set[tuple[str, str, int]] = set()
    fresh_lines: list[str] = []
    n_stale_rubric = 0
    n_stale_prompt = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = r.get("id")
        is_stale_rubric = False
        is_stale_prompt = False
        if rubrics is not None and cid is not None:
            base = cid.split("-", 1)[0]
            rubric = rubrics.get(base)
            if rubric is not None:
                cached_oids = {int(o["id"]) for o in r.get("options", []) if "id" in o}
                current_oids = {int(o["id"]) for o in rubric.get("options", [])}
                # Drop only on unambiguous drift: cached references ids the
                # rubric no longer has. Pure subset (judge omitted options)
                # is kept to avoid an infinite re-judge loop.
                if cached_oids - current_oids:
                    is_stale_rubric = True
        if expected_prompt_hash is not None:
            cached_hash = r.get("prompt_hash")
            if cached_hash is not None and cached_hash != expected_prompt_hash:
                is_stale_prompt = True
        if is_stale_rubric:
            n_stale_rubric += 1
            continue
        if is_stale_prompt:
            n_stale_prompt += 1
            continue
        fresh_lines.append(line)
        out.add((r.get("model", ""), cid, r.get("trial", 1)))
    if n_stale_rubric or n_stale_prompt:
        parts = []
        if n_stale_rubric:
            parts.append(f"{n_stale_rubric} stale-rubric")
        if n_stale_prompt:
            parts.append(f"{n_stale_prompt} stale-prompt")
        print(
            f"  load_completed: rewrote {path.name} dropping "
            f"{', '.join(parts)} record(s) (will re-judge)",
            file=sys.stderr,
        )
        path.write_text("\n".join(fresh_lines) + ("\n" if fresh_lines else ""))
    return out


def resolve_model_name(explicit: str | None, *cache_paths: Path) -> str | None:
    """Resolve the model name that match/severity/review evals should stamp into output
    + use as the (model, id, trial) cache key.

    Resolution order:
      1. `explicit` (from a --model-name CLI arg) if non-empty.
      2. Path inference: if any `cache_paths` is structurally under
         `.../_strategy/<model>/...`, return that <model>.
      3. None (pooled / experimental mode: callers fall back to per-record
         `model` fields in their inputs).

    Path inference is guarded by the `_strategy` literal so pooled multi-model
    runs (which use arbitrary --out paths) don't get miscoerced.
    """
    if explicit:
        return explicit
    for p in cache_paths:
        if p is None:
            continue
        try:
            parts = Path(p).resolve().parts
        except (OSError, ValueError):
            continue
        if "_strategy" in parts:
            idx = parts.index("_strategy")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return None
