# Medical AI Superintelligence Test (MAST) Leaderboard
## Overview

MAST (Medical AI Superintelligence Test) is a suite of clinically realistic benchmarks to evaluate real-world medical capabilities of artificial intelligence models. The system provides a leaderboard where AI models submit API endpoints that are automatically tested against standardized medical scenarios. 

The live leaderboard is available at [bench.arise-ai.org](https://bench.arise-ai.org).

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
│   └── donoharm/              # Do No Harm benchmark
│       ├── prompt.md           # System prompt sent with each case
│       ├── schema.json         # Response validation schema
│       ├── validator.py        # API testing logic
│       ├── inputs/             # Test input files (.txt)
│       │   └── test_001.txt
│       └── outputs/            # Reference responses
│           └── test_001.txt
├── results/                    # API response storage
│   └── donoharm/               # Per-benchmark results
│       ├── test_001_response.json    # Raw API responses
│       └── test_001_validation.json  # Validation results
├── scripts/
│   ├── validate_all.py         # Master API tester
│   ├── config.json             # API endpoint configurations
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
git clone https://github.com/HealthRex/mast.git
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
- **Body**: `prompt.md + "\n" + test_input.txt`
- **Timeout**: Up to 300 seconds

The body contains the full system prompt followed by the clinical case. See `benchmarks/donoharm/prompt.md` for the exact prompt and `benchmarks/donoharm/inputs/test_001.txt` for an example case.

## Response Format

APIs must return a JSON object containing a free-text clinical management plan:

```json
{
  "response": "Assessment: Grade 3 infusion reaction to nivolumab...\n\n1. Refer to Allergy/Immunology for urgent evaluation...\n2. Hold next nivolumab dose until allergy clearance...\n3. ..."
}
```

The `response` field must contain at least 50 characters of clinical text. There is no required structure within the text itself — the model should write a management plan as described in the prompt. See `benchmarks/donoharm/outputs/test_001.txt` for an example of a valid response.

## Benchmarks

### Do No Harm Benchmark

- **Study**: https://arxiv.org/abs/2512.01241
- **Task**: Provide a complete clinical management plan for a medical case
- **Input**: Real clinical cases authored by specialist physicians
- **Output**: Free-text management plan (assessment + numbered recommendations)
- **Scoring**: Evaluated by multiple LLM judges against physician-authored rubrics
- **Validation**: Format compliance (schema validation only)

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
- **Response format**: Valid JSON with a `response` field containing the management plan

### Resource Requirements

#### Estimated Token Usage
- Input tokens: ~500,000
- Output tokens: ~400,000-2,500,000 (varies with reasoning depth)

#### Estimated Costs
Approximate inference costs of widely-used models for a full benchmark run:
- DeepSeek V3: ~$0.20
- Gemini Flash: ~$0.20
- GPT-4o: ~$5
- Claude Sonnet: ~$8
- GPT-5: ~$18
- Gemini Pro: ~$22

*Costs are approximate and depend on your provider's current pricing. Reasoning models (e.g., o4-mini, DeepSeek R1, Gemini Pro) produce more output tokens due to chain-of-thought, increasing costs.*

## File Formats

### Input Files (.txt)
- Plain text clinical cases
- UTF-8 encoding
- One case per file

### Response Schema
- JSON object with a `response` string field
- Must conform to `benchmarks/donoharm/schema.json`
- Minimum 50 characters in the response field
