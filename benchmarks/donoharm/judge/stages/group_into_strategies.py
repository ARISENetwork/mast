#!/usr/bin/env python3
"""Group canonical-extraction actions into strategies via union-find on `dependsOn`.

Reads match-stage canonical extraction (typically the runner's
`match/<extractor>.jsonl` output), runs transitive-closure union-find on
each (case, trial)'s actions using the `dependsOn` edges, and produces a
per-(case, trial) strategy decomposition:

  - Each strategy = a connected component of the dependency graph
  - `summary` = the root (min-id) action's text. Deterministic; no LLM call.
  - Annotates `matched` per strategy = any sub-action's `match` is non-empty
  - Annotates `matched_options` = list of rubric option IDs covered by sub-actions

Each output line:
  {id, trial, strategies: [{strategy_id, summary, sub_actions, matched,
                             matched_options}]}
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..schemas import validate_record


def union_find_components(actions: list[dict]) -> list[list[int]]:
    """Returns list of action-id lists, one per connected component."""
    by_id = {a["id"]: a for a in actions}
    ids = sorted(by_id.keys())

    parent = {n: n for n in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Add edges from each action's dependsOn list (parent → child unions)
    for a in actions:
        n = a["id"]
        for dep in a.get("dependsOn") or []:
            if dep in parent:
                union(n, dep)

    # Group by root
    components: dict[int, list[int]] = {}
    for n in ids:
        components.setdefault(find(n), []).append(n)
    # Sort components by smallest member so output is stable
    return sorted(components.values(), key=lambda c: c[0])


def summarize_strategy(actions: list[dict]) -> str:
    """Return the strategy's root (min-id) action text as the headline.

    Union-find chose `min(ra, rb)` as the root, so the action with the lowest
    id in the component is the canonical representative. We use its `action`
    text as the deterministic strategy summary (a section header above the
    sub_actions list).
    """
    if not actions:
        return "(no actions)"
    root = min(actions, key=lambda a: a["id"])
    return root.get("action", "") or "(no action text)"


def resolve_match(match_str: str) -> list[int]:
    """Resolve an action's match string (comma-separated numeric option IDs,
    e.g. "11" or "11,12") into a list of rubric option IDs."""
    out: list[int] = []
    for tok in (t.strip() for t in (match_str or "").split(",")):
        if tok.isdigit():
            out.append(int(tok))
    return out


def process_record(rec: dict) -> dict:
    actions = rec.get("actions", [])
    components = union_find_components(actions)
    by_id = {a["id"]: a for a in actions}

    # Singleton-option partial flags from the matcher - propagate to the refined
    # record so the production adapter can set prod_options[oid].partial. Match
    # already emits these per the prompt's singleton "Partial credit" section;
    # without this hop they are silently dropped.
    partial_option_ids = sorted({
        int(o["id"]) for o in rec.get("options", []) or []
        if o.get("matched") and o.get("partial")
    })

    strategies = []
    for sid, comp_ids in enumerate(components):
        comp_actions = [by_id[n] for n in comp_ids]
        summary = summarize_strategy(comp_actions)
        matched_options: list[int] = []
        for a in comp_actions:
            for opt_id in resolve_match(a.get("match", "")):
                matched_options.append(opt_id)
        matched_options = sorted(set(matched_options))

        # Per-sub-action resolved match for downstream display
        sub_actions = []
        for a in comp_actions:
            resolved = resolve_match(a.get("match", ""))
            sub_actions.append({
                "id": a["id"],
                "action": a.get("action", ""),
                "category": a.get("category", ""),
                "evidence": a.get("evidence", ""),
                "match_raw": a.get("match", ""),
                "match_resolved": resolved,
            })

        strategies.append({
            "strategy_id": sid,
            "summary": summary,
            "sub_actions": sub_actions,
            "matched": bool(matched_options),
            "matched_options": matched_options,
        })

    return {
        "id": rec["id"],
        "trial": rec["trial"],
        "model": rec.get("model", ""),
        "strategies": strategies,
        "partial_option_ids": partial_option_ids,
    }


def main() -> int:
    # CLI entry point: see match_stage.main for the dotenv rationale.
    try:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / ".env")
    except ImportError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Path to match-stage canonical extraction JSONL.")
    ap.add_argument("--output", required=True,
                    help="Path to write the per-(case, trial) strategy "
                         "decomposition JSONL.")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    print(f"Loaded {len(records)} canonical-extraction records", file=sys.stderr)

    out_lines: list[str] = [None] * len(records)  # type: ignore
    n_strategies_total = 0
    n_matched_strategies = 0
    n_unmatched = 0

    def worker(i: int, rec: dict):
        return i, process_record(rec)

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {pool.submit(worker, i, r): i for i, r in enumerate(records)}
        done = 0
        for fut in as_completed(futures):
            i, processed = fut.result()
            validate_record(processed, "strategies")
            out_lines[i] = json.dumps(processed)
            n_strategies_total += len(processed["strategies"])
            n_matched_strategies += sum(1 for s in processed["strategies"] if s["matched"])
            n_unmatched += sum(1 for s in processed["strategies"] if not s["matched"])
            done += 1
            if done % 25 == 0:
                print(f"  progress: {done}/{len(records)}", file=sys.stderr)

    dst.write_text("\n".join(out_lines) + "\n")

    print(f"\nWrote {len(records)} records to {dst}", file=sys.stderr)
    print(f"  total strategies: {n_strategies_total}", file=sys.stderr)
    print(f"  matched (≥1 sub-action matched): {n_matched_strategies}", file=sys.stderr)
    print(f"  unmatched: {n_unmatched}", file=sys.stderr)
    print(f"  avg strategies/case: {n_strategies_total/len(records):.1f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
