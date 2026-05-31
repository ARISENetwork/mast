# Medical AI Superintelligence Test (MAST) Leaderboard
## Overview

MAST (Medical AI Superintelligence Test) is a suite of clinically realistic benchmarks to evaluate real-world medical capabilities of artificial intelligence models. The system provides a leaderboard where AI models submit API endpoints that are automatically tested against standardized medical scenarios. 

The live leaderboard is available at [arise-ai.org/mast/technical](https://arise-ai.org/mast/technical).

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
│   ├── donoharm/               # First Do NOHARM benchmark
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

2. **Set up your API endpoint**: provide a hosted endpoint for accessing and benchmarking your model.

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

The `response` field must contain at least 50 characters of clinical text. There is no required structure within the text itself; the model should write a management plan as described in the prompt. See `benchmarks/donoharm/outputs/test_001.txt` for an example of a valid response.

**OpenAI-compatible endpoints** are also accepted. If your API returns the standard OpenAI chat completions format (`choices[0].message.content`), the validator will automatically extract the content. This includes endpoints served via OpenRouter or any OpenAI-compatible provider.

## Benchmarks

The MAST suite spans the clinical capabilities measured on the public [leaderboard](https://arise-ai.org/mast/technical). Each benchmark links to its code, data, or site (the First Do NOHARM kit is coming shortly). Full descriptions and demos: [arise-ai.org/mast/benchmarks](https://arise-ai.org/mast/benchmarks).

| Benchmark | Clinical capability | Code / data | Paper |
| --- | --- | --- | --- |
| First Do NOHARM v2 | Safety, management reasoning | Coming soon | [arXiv](https://arxiv.org/abs/2512.01241) |
| Script Concordance Test (SCT) | Reasoning under uncertainty | [`benchmarks/sct/`](benchmarks/sct/) | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIdbp2500120) |
| CPC-Bench | Diagnostic reasoning | [cpcbench.com](https://cpcbench.com) | [arXiv](https://arxiv.org/abs/2509.12194) |
| MedAgentBench v2 | Agentic EHR tasks | [GitHub](https://github.com/ARISENetwork/medagentbenchv2) | [Paper](https://psb.stanford.edu/psb-online/proceedings/psb26/chen_eric.pdf) |
| PhysicianBench | Agentic EHR tasks | [GitHub](https://github.com/HealthRex/PhysicianBench) | - |
| ReXrank Mini | Multimodal radiology | [GitHub](https://github.com/rajpurkarlab/ReXrank) | [arXiv](https://arxiv.org/abs/2411.15122) |
| Multimodal Images | Multimodal dermatology | [DDI](https://ddi-dataset.github.io/) · [MIDAS](https://stanfordaimi.azurewebsites.net/datasets/f4c2020f-801a-42dd-a477-a1a8357ef2a5) | [DDI](https://www.science.org/doi/10.1126/sciadv.abq6147) · [MIDAS](https://ai.nejm.org/doi/full/10.1056/AIdbp2400732) |

ReXrank Mini is MAST's curated subset of the full [ReXrank](https://github.com/rajpurkarlab/ReXrank) benchmark, run with that harness.

- **First Do NOHARM v2**: free-text management plans reconstructed from real generalist-to-specialist consults, scored by multiple LLM judges against specialist-authored rubrics. The `benchmarks/donoharm/` validator checks your endpoint's response format (see [API Request Format](#api-request-format) above); a full run-it-yourself kit is coming shortly.
- **Script Concordance Test (SCT)**: probabilistic clinical reasoning under uncertainty. Run it yourself on the 174-item open subset with deterministic scoring (no LLM judge); see `benchmarks/sct/README.md` for setup, the `sct_score` metric, and reference scores.

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
- **Response format**: Valid JSON: either `{"response": "..."}` or OpenAI-compatible chat completions format

### Resource Requirements

Token and inference-cost estimates per benchmark, from a single GPT-5.5 reference run. Treat these as a rough guide only: your model's token counts and cost will differ, often substantially. Output tokens include reasoning tokens; both scale with reasoning effort and your provider's pricing.

| Benchmark | Input tokens | Output tokens | Est. cost (GPT-5.5) |
| --- | --- | --- | --- |
| First Do NOHARM v2 | 0.7M | 0.9M | $31 |
| Script Concordance Test (SCT) | 0.2M | 0.2M | $6 |
| CPC-Bench | 5.6M | 2.2M | $87 |
| MedAgentBench v2 | 12.5M | 0.4M | $74 |
| PhysicianBench | 36.6M | 0.7M | $205 |
| ReXrank Mini | 32.9M | 2.0M | $221 |
| Multimodal Images | 32.0M | 1.0M | $191 |
| **Full suite** | **~121M** | **~7.4M** | **~$815** |

Agentic benchmarks (MedAgentBench, PhysicianBench) consume far more input tokens because each task spans many tool-use turns. Costs cover model inference only; LLM-judge scoring is run by the MAST team. PhysicianBench reflects the GPT-5.5 high-effort run.

## File Formats

### Input Files (.txt)
- Plain text clinical cases
- UTF-8 encoding
- One case per file

### Response Schema
- JSON object with a `response` string field, or OpenAI-compatible chat completions format
- Must conform to `benchmarks/donoharm/schema.json` (after extraction)
- Minimum 50 characters in the response field
