"""Direct google-genai SDK calls for Gemini judges.

Uses the google-genai SDK directly for reliable `response_json_schema`
support. This bundle judges with Gemini only.

    parsed, runtime, usage = sync_call(model, prompt, schema)

Returns a normalized {prompt_tokens, completion_tokens, thinking_tokens}
usage dict so the cost code is call-agnostic.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

_CLIENT = None


def _client():
    """Lazy google-genai client. Read GEMINI_API_KEY on first call so callers
    can `load_dotenv()` before any judge call without forcing import-time env.
    """
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set; google-genai SDK cannot initialize"
            )
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def is_gemini(model_id: str) -> bool:
    """True if the judge model id targets direct google-genai, recognized by
    the `gemini/` prefix.
    """
    return model_id.startswith("gemini/")


def _strip_prefix(model_id: str) -> str:
    """SDK expects bare model id (no `gemini/` prefix)."""
    return model_id[len("gemini/"):] if model_id.startswith("gemini/") else model_id


def to_openapi(node: Any) -> Any:
    """Convert a draft-07 / draft-2020 JSON Schema dict to Gemini's OpenAPI
    subset for `response_schema`.

    Two known divergences require translation:
      1. `additionalProperties` is rejected. Drop it.
      2. `type: ["X", "null"]` (nullable shorthand) becomes
         `type: "X", nullable: true`.

    Pure-function, leaves the input untouched.
    """
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k == "additionalProperties":
                continue
            if k == "type" and isinstance(v, list):
                non_null = [t for t in v if t != "null"]
                if len(v) > len(non_null):
                    out["nullable"] = True
                out["type"] = non_null[0] if non_null else "string"
            else:
                out[k] = to_openapi(v)
        return out
    if isinstance(node, list):
        return [to_openapi(x) for x in node]
    return node


def sync_call_raw(
    model_id: str,
    prompt: str,
    schema: dict,
    *,
    thinking_level: str = "LOW",
    max_output_tokens: int = 32768,
    temperature: float = 0.0,
    retry_on_rate_limit: bool = True,
) -> tuple[str, float, dict]:
    """One synchronous Gemini SDK call returning (raw_text, runtime, usage).

    Callers that need to inspect text before parsing (e.g. review_stage's
    null-loop detector) use this directly. Most callers want the parsed
    convenience wrapper `sync_call` below.

    Usage dict shape matches `fix_pool_to_161.sync_call` for cost parity with
    the validation runs:
        {prompt_tokens, completion_tokens, thinking_tokens}

    Retries on 429 / ResourceExhausted up to 3x with 60-s backoff so direct
    AI Studio bursts past Tier 1's 2M-input-tokens/min cap recover instead of
    failing the whole task.
    """
    from google.genai import types

    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=to_openapi(schema),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(
            thinking_level=thinking_level,
            include_thoughts=False,
        ),
    )
    bare = _strip_prefix(model_id)

    t0 = time.time()
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = _client().models.generate_content(
                model=bare, contents=prompt, config=cfg,
            )
            break
        except Exception as e:  # google-genai raises ClientError on 429
            msg = str(e).lower()
            transient = (
                retry_on_rate_limit
                and ("rate" in msg or "429" in msg or "resource_exhausted" in msg)
            )
            if not transient or attempt == 2:
                raise
            last_err = e
            time.sleep(60)
    else:  # pragma: no cover - loop always breaks or raises
        if last_err is not None:
            raise last_err
        raise RuntimeError("sync_call retry loop fell through")
    runtime = time.time() - t0

    parts = resp.candidates[0].content.parts or []
    texts = [
        p.text for p in parts
        if getattr(p, "text", None) and not getattr(p, "thought", False)
    ]
    text = texts[-1] if texts else ""

    u = resp.usage_metadata
    usage = {
        "prompt_tokens": getattr(u, "prompt_token_count", 0) or 0,
        "completion_tokens": getattr(u, "candidates_token_count", 0) or 0,
        "thinking_tokens": getattr(u, "thoughts_token_count", 0) or 0,
    }
    return text, runtime, usage


def sync_call(
    model_id: str,
    prompt: str,
    schema: dict,
    **kwargs: Any,
) -> tuple[dict, float, dict]:
    """Parsed-convenience wrapper around sync_call_raw.

    Returns (parsed_json, runtime, usage). Raises json.JSONDecodeError on
    invalid JSON; let it propagate to the caller's existing retry/fail path.
    """
    text, runtime, usage = sync_call_raw(model_id, prompt, schema, **kwargs)
    parsed = json.loads(text) if text else {}
    return parsed, runtime, usage


