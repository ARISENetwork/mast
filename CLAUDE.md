# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAST (Medical AI Superintelligence Test) is a medical AI benchmarking system. Submitters provide an API endpoint that gets tested against standardized medical scenarios. Results appear on the leaderboard at bench.arise-ai.org.

The system sends HTTPS POST requests with clinical prompts to submitter APIs, validates JSON responses against schemas, and saves results for audit.

## Commands

```bash
# Install dependencies
pip install jsonschema requests

# Run all benchmark validations against configured API
python scripts/validate_all.py

# Run a single test case for a specific benchmark
python benchmarks/donoharm/validator.py test_001

# Setup: copy config template and edit with your API details
cp scripts/config.example.json scripts/config.json
```

No test suite, linter, or CI pipeline is configured.

## Architecture

**Validation pipeline flow:**
```
scripts/validate_all.py  (discovers benchmarks, runs each via subprocess)
  -> benchmarks/{name}/validator.py {test_case}  (makes API call, validates response)
     -> scripts/utils.py  (shared: config loading, schema validation, file I/O)
        -> results/{name}/  (saves _response.json, _validation.json, body .json)
```

`validate_all.py` discovers benchmarks by scanning `benchmarks/` for directories containing `validator.py` (excluding `template/`). Each validator is invoked as a subprocess with `sys.executable`.

**Benchmark structure** (each benchmark follows this convention):
- `prompt.md` - task description concatenated with input as the API payload
- `schema.json` - JSON Schema for response validation
- `validator.py` - makes the API call, validates, saves results
- `inputs/test_*.txt` - test case inputs
- `outputs/test_*.txt` - reference answers

**Cross-module imports:** Benchmark validators import from `scripts/utils.py` via `sys.path.append` (see `benchmarks/donoharm/validator.py:17`).

**Config:** `scripts/config.json` (gitignored) holds the API endpoint URL, bearer token, and timeout. Max timeout is 300s.

## Adding a New Benchmark

Copy `benchmarks/template/`, implement `validator.py` following `benchmarks/donoharm/validator.py` as reference. Test cases use `test_NNN` naming. The benchmark is auto-discovered by `validate_all.py`.

**OpenAI-compatible endpoints:** Validators accept both the native schema (`{"response": "..."}` for donoharm, `{"Rating": N, "Rationale": "..."}` for SCT) and OpenAI chat completions format (`choices[0].message.content`). Extraction logic lives in `scripts/utils.py:extract_openai_content()`.

## Key Constraints

- API payload format: `prompt.md + "\n" + test_input.txt` sent as `Content-Type: text/plain`
- Responses must be valid JSON validated against the benchmark's `schema.json`
- `config.json` contains secrets (bearer token) and must never be committed
- Results are saved to `results/` for auditability (also gitignored)
