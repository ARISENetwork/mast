"""JSON Schema files for donoharm judge-pipeline I/O contracts.

  from donoharm_judge.schemas import load_schema, validate_record
  schema = load_schema("match")  # or strategies, review, judged
  validate_record(rec, "match")  # raises jsonschema.ValidationError on bad record

Validation runs unconditionally at every stage write site. Cost is
microseconds per record; catching schema drift at write time is cheaper
than discovering it during analysis. jsonschema is a hard dependency.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent

_SCHEMA_NAMES = (
    "match",
    "strategies",
    "review",
    "judged",
)


def load_schema(name: str) -> dict:
    """Return the parsed schema document by short name.

    Raises ValueError if `name` isn't a known schema.
    """
    if name not in _SCHEMA_NAMES:
        raise ValueError(
            f"unknown schema {name!r}; expected one of {_SCHEMA_NAMES}"
        )
    path = _HERE / f"{name}.schema.json"
    return json.loads(path.read_text())


def all_schemas() -> dict[str, dict]:
    """Return all four schemas keyed by short name."""
    return {n: load_schema(n) for n in _SCHEMA_NAMES}


@lru_cache(maxsize=None)
def _validator(name: str):
    import jsonschema
    return jsonschema.Draft202012Validator(load_schema(name))


def validate_record(record: dict, name: str) -> None:
    """Validate `record` against the named schema.

    Raises `jsonschema.ValidationError` on failure.
    """
    _validator(name).validate(record)


__all__ = ["load_schema", "all_schemas", "validate_record"]
