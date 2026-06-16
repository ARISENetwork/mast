"""Run-summary dataclass returned by `judge_responses`."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeRunSummary:
    """Returned by `judge_responses`. Caller re-reads `judged_path` if it
    needs the records themselves.

    `n_missing_rubric` counts (case_id, trial) pairs whose `case_id`
    prefix didn't resolve to a rubric in `rubrics`.
    """

    n_records: int = 0
    n_missing_rubric: int = 0
    elapsed_s: float = 0.0
