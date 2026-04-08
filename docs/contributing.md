# Contributing to the Leaderboard

This guide explains how to contribute new benchmarks, test cases, or improvements to the leaderboard validation system. This is for the internal MAST team only.

## Adding a New Benchmark

1. **Create the benchmark directory:**
   ```bash
   mkdir benchmarks/your_benchmark_name
   cd benchmarks/your_benchmark_name
   mkdir inputs outputs
   ```

2. **Create required files:**
   - `prompt.md` - Description of the benchmark task
   - `schema.json` - JSON schema for input/output validation
   - `validator.py` - Python validation script
   - `inputs/test_*.txt` - Input test files
   - `outputs/test_*.json` - Expected output files

3. **Follow the templates:**
   - Copy `prompt.md` from an existing benchmark
   - Copy `schema.json` and modify for your benchmark
   - Copy `validator.py` and implement the specific validation logic

## Adding Test Cases

To add new test cases to an existing benchmark:

1. **Add input file:**
   ```bash
   # For benchmark 'donoharm'
   echo "Your test input here" > benchmarks/donoharm/inputs/test_002.txt
   ```

2. **Add corresponding output file:**
   ```bash
   # Create the expected output
   cat > benchmarks/donoharm/outputs/test_002.json << EOF
   {
     "response": "Assessment: ... \n\n1. First recommendation...\n2. Second recommendation..."
   }
   EOF
   ```

3. **Run validation:**
   ```bash
   python scripts/validate_all.py
   ```

## Validator Implementation

Each `validator.py` is an API validator invoked as a subprocess by `validate_all.py`. It should implement:

1. **API Request:** Send the prompt + input to the submitter's endpoint
2. **Schema Validation:** Validate the response against `schema.json`
3. **Result Saving:** Save responses and validation results to `results/`

Key functions (see `benchmarks/donoharm/validator.py` as reference):
- `make_api_request()` - HTTPS POST to the submitter's API
- `test_api_endpoint()` - Orchestrates request, validation, and result saving
- `load_schema()` - Loads the benchmark's JSON schema

## Schema Design

The `schema.json` file is a standard JSON Schema used to validate API responses. It should define the expected response structure directly.

Example (from donoharm):
```json
{
  "type": "object",
  "properties": {
    "response": {
      "type": "string",
      "minLength": 50,
      "description": "Free-text clinical management plan"
    }
  },
  "required": ["response"]
}
```

## Running Tests

- **Validate all benchmarks:** `python scripts/validate_all.py`
- **Validate specific benchmark:** `python benchmarks/donoharm/validator.py test_001`
- **Validate specific test case:** `python scripts/validate_all.py` (finds and runs all)

## Best Practices

1. **Naming:** Use `test_001`, `test_002`, etc. for test cases
2. **Documentation:** Keep `prompt.md` up to date with clear examples
3. **Validation:** Provide helpful error messages in validators
4. **Schema:** Use specific types and constraints in JSON Schema
5. **Testing:** Add diverse test cases covering edge cases

## File Structure

```
benchmarks/
├── benchmark_name/
│   ├── prompt.md          # Task description
│   ├── schema.json        # Input/output schemas
│   ├── validator.py       # Validation logic
│   ├── inputs/            # Test input files (.txt)
│   │   ├── test_001.txt
│   │   └── test_002.txt
│   └── outputs/           # Expected outputs (.json)
│       ├── test_001.json
│       └── test_002.json
└── ...
```

## Dependencies

The validation system uses:
- `jsonschema` for JSON validation
- Standard Python libraries only

Install dependencies:
```bash
pip install jsonschema requests
