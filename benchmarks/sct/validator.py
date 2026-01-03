#!/usr/bin/env python3
"""
API Validator for SCT (Script Concordance Test) benchmark.

This validator makes HTTPS POST requests to submitter APIs,
validates responses against the schema, and saves results.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import requests

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))
from utils import load_config, validate_schema, save_json_file, get_results_dir


def load_schema() -> Dict[str, Any]:
    """Load JSON schema for response validation."""
    schema_path = Path(__file__).parent / "schema.json"
    with open(schema_path, 'r') as f:
        return json.load(f)


def load_prompt() -> str:
    """Load the prompt template."""
    prompt_path = Path(__file__).parent / "prompt.md"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_input_file(input_path: Path) -> str:
    """Load input text file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def make_api_request(url: str, token: str, payload: str, timeout: int) -> Dict[str, Any]:
    """
    Make HTTPS POST request to submitter's API.

    Args:
        url: API endpoint URL
        token: Bearer token for authentication
        payload: Request payload (prompt + input)
        timeout: Request timeout in seconds

    Returns:
        Dict containing response details
    """
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'text/plain'
    }

    start_time = time.time()
    try:
        response = requests.post(
            url,
            data=payload,
            headers=headers,
            timeout=timeout
        )
        response_time = time.time() - start_time

        # Try to parse JSON response
        try:
            response_body = response.json()
        except json.JSONDecodeError:
            response_body = {"error": "Invalid JSON response", "raw_response": response.text}

        return {
            "success": True,
            "status_code": response.status_code,
            "response_time": round(response_time, 2),
            "headers": dict(response.headers),
            "body": response_body,
            "error": None
        }

    except requests.exceptions.Timeout:
        response_time = time.time() - start_time
        return {
            "success": False,
            "status_code": None,
            "response_time": round(response_time, 2),
            "headers": {},
            "body": None,
            "error": f"Request timed out after {timeout} seconds"
        }

    except requests.exceptions.RequestException as e:
        response_time = time.time() - start_time
        return {
            "success": False,
            "status_code": None,
            "response_time": round(response_time, 2),
            "headers": {},
            "body": None,
            "error": str(e)
        }


def validate_rating(response_body: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that the Rating is within expected range.

    Args:
        response_body: Parsed JSON response

    Returns:
        Tuple of (valid, message)
    """
    if not isinstance(response_body, dict):
        return False, "Response must be a JSON object"

    rating = response_body.get("Rating")
    if rating is None:
        return False, "Missing 'Rating' field"

    if not isinstance(rating, int):
        return False, f"Rating must be an integer, got {type(rating).__name__}"

    if rating < -2 or rating > 2:
        return False, f"Rating must be between -2 and +2, got {rating}"

    return True, "Rating is valid"


def save_response(test_case: str, response_data: Dict[str, Any], payload: str):
    """Save API response to results directory."""
    results_dir = get_results_dir("sct")
    response_file = results_dir / f"{test_case}_response.json"

    full_response = {
        "test_case": test_case,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request": {
            "url": response_data.get("url", "unknown"),
            "payload": payload
        },
        "response": {
            "success": response_data["success"],
            "status_code": response_data["status_code"],
            "response_time": response_data["response_time"],
            "headers": dict(response_data["headers"]),
            "body": response_data["body"],
            "error": response_data["error"]
        }
    }

    save_json_file(full_response, response_file)


def save_body_output(test_case: str, response_body: Any):
    """Save just the response body output to results directory."""
    results_dir = get_results_dir("sct")
    body_file = results_dir / f"{test_case}.json"

    save_json_file(response_body, body_file)


def save_validation_result(test_case: str, validation_data: Dict[str, Any]):
    """Save validation results to results directory."""
    results_dir = get_results_dir("sct")
    validation_file = results_dir / f"{test_case}_validation.json"

    save_json_file(validation_data, validation_file)


def test_api_endpoint(test_case: str) -> Tuple[bool, str]:
    """
    Test a single API endpoint for the given test case.

    Args:
        test_case: Name of the test case (e.g., "example_001")

    Returns:
        Tuple of (passed, message)
    """
    try:
        # Load configuration
        config = load_config()
        if "endpoint" not in config:
            return False, "No endpoint configured in config.json"

        endpoint_config = config["endpoint"]
        url = endpoint_config["url"]
        token = endpoint_config["token"]
        timeout = min(endpoint_config.get("timeout", 30), 300)

        # Load prompt and input
        prompt = load_prompt()
        input_path = Path(__file__).parent / "inputs" / f"{test_case}.txt"

        if not input_path.exists():
            return False, f"Input file not found: {input_path}"

        input_text = load_input_file(input_path)

        # Construct payload
        payload = prompt + "\n" + input_text

        # Make API request
        response_data = make_api_request(url, token, payload, timeout)
        response_data["url"] = url

        # Save raw response
        save_response(test_case, response_data, payload)

        # Save body output if response was successful
        if response_data["success"] and response_data["body"] is not None:
            save_body_output(test_case, response_data["body"])

        # Check for API errors
        if not response_data["success"]:
            validation_result = {
                "test_case": test_case,
                "passed": False,
                "schema_valid": False,
                "rating_valid": False,
                "response_time": response_data["response_time"],
                "errors": [response_data["error"]],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            save_validation_result(test_case, validation_result)
            return False, f"API request failed: {response_data['error']}"

        # Validate response schema
        schema = load_schema()
        schema_valid, schema_message = validate_schema(response_data["body"], schema)

        # Validate rating range
        rating_valid, rating_message = validate_rating(response_data["body"])

        # Overall validation
        passed = schema_valid and rating_valid
        errors = []
        if not schema_valid:
            errors.append(schema_message)
        if not rating_valid:
            errors.append(rating_message)

        validation_result = {
            "test_case": test_case,
            "passed": passed,
            "schema_valid": schema_valid,
            "rating_valid": rating_valid,
            "response_time": response_data["response_time"],
            "errors": errors,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        save_validation_result(test_case, validation_result)

        if passed:
            rating = response_data["body"].get("Rating", "?")
            return True, f"✓ {test_case}: Rating={rating:+d} in {response_data['response_time']}s"
        else:
            return False, f"✗ {test_case}: {'; '.join(errors)}"

    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        validation_result = {
            "test_case": test_case,
            "passed": False,
            "schema_valid": False,
            "rating_valid": False,
            "response_time": None,
            "errors": [error_msg],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        save_validation_result(test_case, validation_result)
        return False, error_msg


def main():
    """Main validation function."""
    if len(sys.argv) != 2:
        print("Usage: python validator.py <test_name>")
        print("Example: python validator.py example_001")
        sys.exit(1)

    test_name = sys.argv[1]
    passed, message = test_api_endpoint(test_name)

    print(message)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
