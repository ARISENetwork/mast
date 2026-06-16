"""Apply review-step overrides to a refined
strategies record, producing a post-review record the adapter consumes.

This is the deterministic step that replaces the runtime adapter logic in
`apply_global_match_review`. Baking the review overrides into the match
graph here means the final metric computation sees the post-review match
decisions.

Attribution model: the review prompt receives match-stage's `actions[]` and
emits a per-override `action_id: int | null`. Non-null cites the extracted
action that supports the override; null is an explicit coverage-gap signal
("the response did this, but match didn't extract it"). This replaces the
prior longest-common-substring heuristic on `evidence` text.

Three override classes:

  - **Demotion** (prior matched -> new `no`):
      Remove the demoted oid from any strategy's `matched_options` and from
      `partial_option_ids`. If a sub_action's `match_resolved` referenced only
      that oid, clear the anchor. If a strategy has no remaining anchored
      sub_actions and no other matched_options, flip strategy.matched=False.
      Severity will then score it as an unmatched strategy.

  - **Promotion** (prior `no` -> new `yes`/`partial`):
      Use the override's `action_id` to attribute the promotion to a specific
      sub_action via direct lookup (sub_action.id == action_id). Hit: inject
      `match_resolved=[oid]` into that sub_action, add oid to its strategy's
      `matched_options`, set strategy.matched=True. Miss (null action_id or a
      hallucinated id not present in any sub_action): leave the strategies
      graph unchanged and bump `_promotions_unattributed`. The option flag
      still flips in the prod-shape record downstream for recall accounting.

  - **Partial-shift** (`yes` <-> `partial`, no change in matched-ness):
      Update `partial_option_ids` set; nothing else changes.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from .io import load_jsonl
from .schemas import validate_record

log = logging.getLogger(__name__)


def _coerce_override(ov):
    """Accept dict, or string-encoded JSON dict. Tolerates judges that wrap
    each override in JSON-encoded-string form inside the overrides[] array
    instead of emitting nested objects. Returns dict or None if uncoercible.
    """
    if isinstance(ov, dict):
        return ov
    if isinstance(ov, str):
        try:
            parsed = json.loads(ov)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


_VERDICT_TO_FLAGS = {
    "yes": (True, False),
    "partial": (True, True),
    "no": (False, False),
}


def _attribute_promotion(
    action_id: int | None,
    strategies: list[dict],
) -> tuple[int | None, int | None]:
    """Locate the (strategy_idx, sub_action_id) the reviewer cited.

    Returns (strategy_idx, sub_action_id). If action_id is null or no
    sub_action carries that id (hallucinated reference), returns (None, None).
    """
    if action_id is None:
        return (None, None)
    for s_idx, s in enumerate(strategies):
        for sa in s.get("sub_actions", []):
            if sa.get("id") == action_id:
                return (s_idx, action_id)
    return (None, None)


def _classify_override(
    oid: int,
    new_verdict: str,
    matched_oids: set[int],
    partial_oids: set[int],
) -> str:
    """Return one of: 'promotion', 'demotion', 'partial-shift', 'noop'."""
    new_matched, new_partial = _VERDICT_TO_FLAGS.get(new_verdict, (None, None))
    if new_matched is None:
        return "noop"
    prior_matched = oid in matched_oids
    prior_partial = oid in partial_oids
    if not prior_matched and new_matched:
        return "promotion"
    if prior_matched and not new_matched:
        return "demotion"
    if prior_matched and new_matched and prior_partial != new_partial:
        return "partial-shift"
    return "noop"


def _strategy_owning(strategies: list[dict], oid: int) -> int | None:
    """Return the index of the strategy whose matched_options contains oid,
    or None if no strategy claims it."""
    for i, s in enumerate(strategies):
        if oid in s.get("matched_options", []):
            return i
    return None


def _strategy_has_remaining_anchors(s: dict) -> bool:
    """True if the strategy still has any anchored sub_actions or matched_options."""
    if s.get("matched_options"):
        return True
    for sa in s.get("sub_actions", []):
        if sa.get("match_resolved"):
            return True
    return False


def apply_overrides_to_strategies(
    refined_record: dict,
    overrides: list[dict],
) -> dict:
    """Apply review overrides to a refined record. Returns a new
    record; the input is not mutated.

    Args:
        refined_record: {id, trial, model, strategies, partial_option_ids}
            from `strategies.jsonl`. Each strategy has {strategy_id, sub_actions,
            matched, matched_options}.
        overrides: list of {option_id, new_verdict, rationale, evidence, action_id}
            from the review step. `action_id` is required (per review schema);
            None signals an explicit coverage gap. May include verdicts that
            match the upstream (no-op).

    Returns:
        New record with mutated `strategies` and `partial_option_ids`. Adds
        `_overrides_applied` (count) and `_promotions_unattributed` (count).
    """
    rec = copy.deepcopy(refined_record)
    strategies: list[dict] = rec.get("strategies", []) or []
    partial_oids: set[int] = {int(x) for x in (rec.get("partial_option_ids") or [])}
    matched_oids: set[int] = set()
    for s in strategies:
        for x in s.get("matched_options", []) or []:
            matched_oids.add(int(x))

    n_applied = 0
    n_promotions_unattributed = 0

    for raw_ov in overrides or []:
        ov = _coerce_override(raw_ov)
        if ov is None:
            continue
        oid_raw = ov.get("option_id")
        verdict = ov.get("new_verdict")
        if not isinstance(oid_raw, int) or verdict not in _VERDICT_TO_FLAGS:
            continue
        oid = int(oid_raw)
        klass = _classify_override(oid, verdict, matched_oids, partial_oids)
        if klass == "noop":
            continue

        n_applied += 1

        if klass == "demotion":
            partial_oids.discard(oid)
            owning_idx = _strategy_owning(strategies, oid)
            if owning_idx is not None:
                s = strategies[owning_idx]
                s["matched_options"] = [
                    int(x) for x in s.get("matched_options", []) if int(x) != oid
                ]
                for sa in s.get("sub_actions", []):
                    mres = sa.get("match_resolved") or []
                    new_mres = [int(x) for x in mres if int(x) != oid]
                    if len(new_mres) != len(mres):
                        sa["match_resolved"] = new_mres
                if not _strategy_has_remaining_anchors(s):
                    s["matched"] = False
            matched_oids.discard(oid)

        elif klass == "promotion":
            new_partial = verdict == "partial"
            action_id_raw = ov.get("action_id")
            action_id = action_id_raw if isinstance(action_id_raw, int) else None
            s_idx, sa_id = _attribute_promotion(action_id, strategies)
            if s_idx is None:
                # Reviewer emitted null (coverage gap) or cited an action_id
                # not present in any sub_action (hallucinated). Leave strategies
                # graph unchanged; option flag flips downstream in the adapter
                # for recall accounting.
                if action_id is not None:
                    log.warning(
                        "[apply-overrides] %s/t%s: reviewer cited action_id=%s for "
                        "option %s but no sub_action carries that id; treating as null",
                        rec.get("id"), rec.get("trial"), action_id, oid,
                    )
                n_promotions_unattributed += 1
                if new_partial:
                    partial_oids.add(oid)
                continue
            s = strategies[s_idx]
            # Inject anchor into the attributed sub_action
            for sa in s.get("sub_actions", []):
                if sa.get("id") == sa_id:
                    mres = list(sa.get("match_resolved") or [])
                    if oid not in mres:
                        mres.append(oid)
                    sa["match_resolved"] = mres
                    break
            mo = list(s.get("matched_options", []) or [])
            if oid not in mo:
                mo.append(oid)
            s["matched_options"] = mo
            s["matched"] = True
            matched_oids.add(oid)
            if new_partial:
                partial_oids.add(oid)

        elif klass == "partial-shift":
            new_partial = verdict == "partial"
            if new_partial:
                partial_oids.add(oid)
            else:
                partial_oids.discard(oid)

    rec["partial_option_ids"] = sorted(partial_oids)
    rec["_overrides_applied"] = n_applied
    rec["_promotions_unattributed"] = n_promotions_unattributed
    return rec


def apply_overrides_to_refined(
    refined_path: Path,
    overrides_path: Path,
    out_path: Path,
) -> Path:
    """Materialize a post-review refined JSONL by applying review overrides
    at the strategy level. Each input refined record is paired with its
    overrides record by (id, trial); the overrides are passed through
    `apply_overrides_to_strategies`. Records without overrides pass through
    unchanged.

    Writes one record per input line to `out_path`. Records that crash the
    apply step are logged and emitted unchanged so the downstream severity
    step still sees them.
    """
    overrides_by_key: dict[tuple[str, int], list[dict]] = {}
    for r in load_jsonl(overrides_path):
        overrides_by_key[(r["id"], r.get("trial", 1))] = r.get("overrides", []) or []

    n_total = 0
    n_with_overrides = 0
    n_unattributed_promotions = 0
    n_applied = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for ref in load_jsonl(refined_path):
            key = (ref["id"], ref.get("trial", 1))
            overrides = overrides_by_key.get(key)
            if overrides:
                try:
                    ref = apply_overrides_to_strategies(ref, overrides)
                    n_with_overrides += 1
                    n_applied += ref.get("_overrides_applied", 0)
                    n_unattributed_promotions += ref.get("_promotions_unattributed", 0)
                except Exception as e:
                    log.warning(
                        "[apply-overrides] %s/t%s: %s (emitting refined unchanged)",
                        key[0], key[1], e,
                    )
            validate_record(ref, "strategies")
            fout.write(json.dumps(ref) + "\n")
            n_total += 1
    log.info(
        "[apply-overrides] %d records (overridden=%d, applied=%d, "
        "unattributed_promotions=%d)",
        n_total, n_with_overrides, n_applied, n_unattributed_promotions,
    )
    return out_path
