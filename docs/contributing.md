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
   # Create the expected JSON output
   cat > benchmarks/donoharm/outputs/test_002.json << EOF
   {
     "result": "expected_result",
     "metadata": {
       "confidence": 0.95,
       "processing_time": 1.0
     }
   }
   EOF
   ```

3. **Run validation:**
   ```bash
   python scripts/validate_all.py
   ```

## Validator Implementation

The `validator.py` script should implement:

1. **Schema Validation:** Uses the `schema.json` to validate output format
2. **Correctness Validation:** Compares actual output with expected output
3. **Error Handling:** Provides clear error messages

Key functions to implement:
- `validate_correctness()` - Main validation logic
- `load_schema()` - Loads the benchmark's schema
- `validate_test_case()` - Orchestrates the validation

## Schema Design

The `schema.json` file should contain:

- `input_schema` - Description of input format (for documentation)
- `output_schema` - JSON Schema for output validation
- `validation_rules` - Any custom validation rules

Example structure:
```json
{
  "output_schema": {
    "type": "object",
    "properties": {
      "result": {"type": "string"},
      "score": {"type": "number", "minimum": 0, "maximum": 100}
    },
    "required": ["result", "score"]
  }
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
pip install jsonschema
