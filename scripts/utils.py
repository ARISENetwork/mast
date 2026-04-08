#!/usr/bin/env python3
"""
Shared utility functions for MAST validation scripts.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import jsonschema


def load_config() -> Dict[str, Any]:
    """Load API configuration from scripts/config.json."""
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("Please create config.json based on config.example.json")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        sys.exit(1)


def extract_openai_content(response_body: Any) -> str | None:
    """Extract content from an OpenAI-compatible response.

    Returns the text content if the response matches the OpenAI chat
    completions format (choices[0].message.content), otherwise None.
    """
    try:
        if isinstance(response_body, dict) and "choices" in response_body:
            return response_body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        pass
    return None


def validate_schema(response_body: Any, schema: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate response body against a JSON schema."""
    try:
        jsonschema.validate(response_body, schema)
        return True, "Schema validation passed"
    except jsonschema.ValidationError as e:
        return False, f"Schema validation failed: {e.message}"
    except Exception as e:
        return False, f"Schema validation error: {str(e)}"


def save_json_file(data: Any, file_path: Path, indent: int = 2) -> None:
    """Save data as JSON to a file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def get_results_dir(benchmark_name: str) -> Path:
    """Get the results directory path for a benchmark."""
    return Path(__file__).parent.parent / "results" / benchmark_name
