# Medical AI Superintelligence Test (MAST) Leaderboard
## Overview

MAST (Medical AI Superintelligence Test) is a suite of clinically realistic benchmarks to evaluate real-world medical capabilities of artificial intelligence models. The system provides a leaderboard where AI models submit API endpoints that are automatically tested against standardized medical scenarios. 

The live leaderboard is available at [benchmarks.arise-ai.org](https://benchmarks.arise-ai.org).

This repository provides instructions and test files to validate your custom model API endpoint. After passing validation, view the [Submission Agreement](docs/submission_agreement.md) and submit the [Registration Form](https://forms.gle/4exSPLbsmWjNmMRQ7) for review by the MAST team. The API and token are used only for benchmark execution and are not stored after evaluation.

## How It Works

1. **Submitters** provide a single API endpoint with authentication token
2. **Leaderboard** runs automated tests against all benchmarks using that endpoint
3. **API calls** are made with standardized prompts and test cases for each benchmark
4. **Responses** are validated for format compliance
5. **Results** are manually reviewed prior to publication on the leaderboard

## Structure

```
mast/
├── benchmarks/
│   ├── donoharm/               # Do No Harm benchmark
│   │   ├── prompts/            # System prompt variants (default.md sent with each case)
│   │   ├── schema.json         # Response validation schema
│   │   ├── validator.py        # API testing logic
│   │   ├── inputs/             # Test input files (.txt)
│   │   └── outputs/            # Reference responses
│   ├── sct/                    # Script Concordance Test benchmark
│   └── template/               # Template for new benchmarks
├── results/                    # API response storage (per-benchmark)
├── scripts/
│   ├── validate_all.py         # Master API tester
│   ├── utils.py                # Shared utilities
│   ├── config.json             # API endpoint config (gitignored)
│   └── config.example.json     # Template for submitters
├── docs/
│   ├── contributing.md         # Contribution guidelines
│   ├── submission_agreement.md # Terms for submitters
│   └── benchmark_descriptions.md  # Detailed benchmark info
└── README.md
```

## Quick Start

### For Submitters

1. **Clone the repository:**
```bash
git clone https://github.com/ARISENetwork/mast.git
cd mast
```

2. **Set up your API endpoint** — provide a hosted endpoint for accessing and benchmarking your model.

3. **Configure your endpoint** by copying and editing the config:
```bash
cp scripts/config.example.json scripts/config.json
# Edit scripts/config.json with your API details
```

4. **Test your endpoint:**
```bash
python scripts/validate_all.py
```

## API Request Format

Each benchmark makes HTTPS POST requests with:

- **Method**: `POST`
- **Headers**:
  - `Authorization: Bearer {token}`
  - `Content-Type: text/plain`
- **Body**: `prompts/default.md + "\n" + test_input.txt`
- **Timeout**: Up to 300 seconds

The body contains the full system prompt followed by the clinical case. See `benchmarks/donoharm/prompts/default.md` for the exact prompt and `benchmarks/donoharm/inputs/test_001.txt` for an example case. Alternative prompt variants used for sensitivity analyses are in the same `prompts/` directory.

## Response Format

APIs must return a JSON object containing a free-text clinical management plan:

```json
{
  "response": "Assessment: Grade 3 infusion reaction to nivolumab...\n\n1. Refer to Allergy/Immunology for urgent evaluation...\n2. Hold next nivolumab dose until allergy clearance...\n3. ..."
}
```

The `response` field must contain at least 50 characters of clinical text. There is no required structure within the text itself — the model should write a management plan as described in the prompt. See `benchmarks/donoharm/outputs/test_001.txt` for an example of a valid response.

**OpenAI-compatible endpoints** are also accepted. If your API returns the standard OpenAI chat completions format (`choices[0].message.content`), the validator will automatically extract the content. This includes endpoints served via OpenRouter or any OpenAI-compatible provider.

## Benchmarks

### First Do NOHARM Benchmark

- **Study**: https://arxiv.org/abs/2512.01241
- **Task**: Provide clinical recommendations for a medical case
- **Input**: Reconstructed from real clinical cases, where a generalist physician electronically consulted a specialist/subspecialist
- **Output**: Free-text management plan (assessment + recommendations)
- **Scoring**: Evaluated by multiple LLM judges against specialist-authored rubrics
- **Validation**: Format compliance (schema validation only)

### SCT (Script Concordance Test) Benchmark

*Coming soon.* See `benchmarks/sct/README.md` for preliminary details.

## Validation Results

All API responses are saved for auditability:

- **`test_XXX_response.json`**: Complete API response with metadata
- **`test_XXX_validation.json`**: Validation results and error details

## Prerequisites
### Python Dependencies
Install required packages:
```bash
pip install jsonschema requests
```

### API Requirements
- **Stable endpoint**: API must remain accessible for at least 72 hours during benchmarking
- **Concurrent requests**: Must support 5-10 simultaneous connections
- **Authentication**: Bearer token authentication required
- **Response time**: Under 300 seconds per request
- **Response format**: Valid JSON — either `{"response": "..."}` or OpenAI-compatible chat completions format

### Resource Requirements

#### Estimated Token Usage
- Input tokens: ~1,500,000
- Output tokens: ~3,000,000-18,000,000 (varies with reasoning depth)

#### Estimated Costs
Approximate inference costs for large frontier-scale models: **$125-$400** for a full benchmark run (but can be higher depending on reasoning effort).

*Costs are approximate and depend on your provider's current pricing. The benchmark is run at multiple response lengths for sensitivity analysis. Reasoning models produce significantly more output tokens due to chain-of-thought, increasing costs substantially.*

## File Formats

### Input Files (.txt)
- Plain text clinical cases
- UTF-8 encoding
- One case per file

### Response Schema
- JSON object with a `response` string field, or OpenAI-compatible chat completions format
- Must conform to `benchmarks/donoharm/schema.json` (after extraction)
- Minimum 50 characters in the response field
