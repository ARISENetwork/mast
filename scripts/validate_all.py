#!/usr/bin/env python3
"""
Master API validator script for all benchmarks.

This script loads API configurations and runs tests against submitter endpoints,
providing comprehensive validation reports with full auditability.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import subprocess
from utils import load_config


def discover_benchmarks() -> List[str]:
    """Discover all benchmark directories."""
    benchmarks_dir = Path(__file__).parent.parent / "benchmarks"
    if not benchmarks_dir.exists():
        return []

    benchmarks = []
    for item in benchmarks_dir.iterdir():
        if item.is_dir() and (item / "validator.py").exists() and item.name != "template":
            benchmarks.append(item.name)

    return sorted(benchmarks)


def discover_test_cases(benchmark_name: str) -> List[str]:
    """Discover all test cases for a given benchmark (test_*.txt and example_*.txt)."""
    inputs_dir = Path(__file__).parent.parent / "benchmarks" / benchmark_name / "inputs"
    if not inputs_dir.exists():
        return []

    stems = set()
    for pattern in ("test_*.txt", "example_*.txt"):
        for item in inputs_dir.glob(pattern):
            stems.add(item.stem)
    return sorted(stems)


def run_api_validator(benchmark_name: str, test_name: str) -> Tuple[bool, str]:
    """Run the API validator for a specific benchmark and test case."""
    try:
        validator_path = Path(__file__).parent.parent / "benchmarks" / benchmark_name / "validator.py"

        # Run the validator as a subprocess
        result = subprocess.run(
            [sys.executable, str(validator_path), test_name],
            capture_output=True,
            text=True,
            cwd=validator_path.parent
        )

        # Print the validator output immediately
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(f"Error: {result.stderr.strip()}")

        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip() or result.stdout.strip()

    except Exception as e:
        error_msg = f"Failed to run validator: {str(e)}"
        print(error_msg)
        return False, error_msg


def validate_benchmark(benchmark_name: str) -> Dict[str, Any]:
    """Validate all test cases for a given benchmark via API."""
    print(f"\nTesting {benchmark_name.upper()} benchmark...")

    test_cases = discover_test_cases(benchmark_name)
    if not test_cases:
        print(f" No test cases found for {benchmark_name}")
        return {
            "benchmark": benchmark_name,
            "status": "no_tests",
            "message": "No test cases found",
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "test_cases": []
        }

    results = []
    passed = 0
    failed = 0

    for test_case in test_cases:
        print(f"  Testing {test_case}...")
        is_valid, message = run_api_validator(benchmark_name, test_case)

        results.append({
            "test_case": test_case,
            "passed": is_valid,
            "message": message
        })

        if is_valid:
            passed += 1
        else:
            failed += 1

    status = "passed" if failed == 0 else "failed"

    return {
        "benchmark": benchmark_name,
        "status": status,
        "total_tests": len(test_cases),
        "passed": passed,
        "failed": failed,
        "test_cases": results
    }


def print_benchmark_results(results: Dict[str, Any]):
    """Print results for a single benchmark."""
    print(f"\n{'='*60}")
    print(f"{results['benchmark'].upper()} RESULTS")
    print(f"{'='*60}")
    print(f"Status: {results['status'].upper()}")
    print(f"Tests: {results.get('passed', 0)}/{results.get('total_tests', 0)} passed")

    if results.get('failed', 0) > 0:
        print("\nFailed tests:")
        for test_case in results['test_cases']:
            if not test_case['passed']:
                print(f"  - {test_case['test_case']}: {test_case['message']}")

    if results.get('failed', 0) == 0 and results.get('total_tests', 0) > 0:
        print("\nAll API tests passed!")


def validate_config() -> bool:
    """Validate that configuration is properly set up."""
    try:
        config = load_config()
        endpoint = config.get("endpoint")

        if not endpoint:
            print("Error: No endpoint configured in config.json")
            return False

        # Check required fields
        required_fields = ["url", "token"]
        missing_fields = [field for field in required_fields if field not in endpoint]

        if missing_fields:
            print(f"Error: Missing required fields: {', '.join(missing_fields)}")
            return False

        # Validate URL format
        url = endpoint["url"]
        if not url.startswith("http"):
            print(f"Error: Invalid URL format: {url}")
            return False

        # Check timeout
        timeout = endpoint.get("timeout", 30)
        if timeout > 300:
            print(f"Warning: Timeout {timeout}s exceeds maximum 300s, capping at 300s")

        print("Configuration validation passed")
        return True

    except Exception as e:
        print(f"Error: Configuration error: {e}")
        return False


def show_config_info():
    """Show information about the configured endpoint."""
    config = load_config()
    endpoint = config.get("endpoint", {})
    print(f"Endpoint: {endpoint.get('url', 'unknown')}")
    print(f"Timeout: {endpoint.get('timeout', 30)}s")
    verify_ssl = endpoint.get('verify_ssl', True)
    if not verify_ssl:
        print("⚠️  SSL certificate verification is DISABLED")
    print()


def main():
    """Main API validation function."""
    print("Starting API validation for all benchmarks...")
    print("This will make HTTPS requests to configured endpoints\n")

    # Validate configuration first
    if not validate_config():
        print("\nError: Configuration validation failed. Please check your config.json")
        sys.exit(1)

    # Show endpoint info
    show_config_info()

    benchmarks = discover_benchmarks()
    if not benchmarks:
        print("Error: No benchmarks found!")
        sys.exit(1)

    print(f"Will test {len(benchmarks)} benchmark(s): {', '.join(benchmarks)}\n")

    all_results = []
    total_passed = 0
    total_failed = 0

    for benchmark in benchmarks:
        results = validate_benchmark(benchmark)
        all_results.append(results)
        total_passed += results.get('passed', 0)
        total_failed += results.get('failed', 0)
        print_benchmark_results(results)

    # Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total benchmarks tested: {len(benchmarks)}")
    print(f"Total API calls made: {total_passed + total_failed}")
    print(f"Successful responses: {total_passed}")
    print(f"Failed responses: {total_failed}")

    if total_failed == 0:
        print("\nAll API tests passed! Responses saved to results/ directory.")
        print("Check results/noharm/ for detailed API responses and validation results.")
        sys.exit(0)
    else:
        print(f"\n{total_failed} API test(s) failed!")
        print("Check results/ directory for detailed error information.")
        print("Look for _response.json files (raw API responses) and _validation.json files (error details).")
        sys.exit(1)


if __name__ == "__main__":
    main()
