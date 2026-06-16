"""Configuration dataclasses for the donoharm judging pipeline.

Public API surface. Replaces the kwarg-soup signature of the legacy
orchestrator (13+ kwargs) with a single frozen dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

JUDGE_DIR = Path(__file__).resolve().parent
JUDGE_PROMPTS = JUDGE_DIR / "prompts"

# Stage models. Both default to direct google-genai SDK routing; this
# bundle judges with Gemini only.

# Match extractor: judge-independent, fixed for production parity.
DEFAULT_MATCH_JUDGE = "gemini/gemini-3-flash-preview"

# Global match-review judge. Strong reviewer re-reads ALL options per case
# and emits overrides for any wrong matched/partial verdicts. Selected by
# ablation against physician labels; runs direct via the google-genai SDK.
DEFAULT_REVIEW_JUDGE = "gemini/gemini-3.5-flash"

DEFAULT_MATCH_PROMPT = JUDGE_PROMPTS / "extract_match.md"
DEFAULT_REVIEW_PROMPT = JUDGE_PROMPTS / "global_match_review.md"


@dataclass(frozen=True)
class JudgeConfig:
    """All knobs for one `judge_responses` invocation.

    `cache_root` and `judged_path` are caller-chosen. The pipeline writes
    intermediate stage outputs under `{cache_root}/{model_name}/...`.

    In the MAST production layout `cache_root` is the `_strategy`
    directory under the prompt-named raw dir (i.e. `{raw_dir}/_strategy`),
    NOT the raw dir itself. Pick any per-experiment directory you want
    for ad-hoc runs.
    """

    cache_root: Path

    match_judge: str = DEFAULT_MATCH_JUDGE
    match_prompt: Path = DEFAULT_MATCH_PROMPT

    review_judge: str | None = DEFAULT_REVIEW_JUDGE
    review_prompt: Path = DEFAULT_REVIEW_PROMPT

    threads: int = 40
    cases_filter: list[str] | None = None
