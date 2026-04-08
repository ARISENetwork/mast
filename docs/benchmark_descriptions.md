# Benchmark Descriptions

This document provides detailed descriptions of all benchmarks in the MAST leaderboard.

## Do No Harm Benchmark

### Purpose
Do No Harm is a physician-validated medical benchmark to evaluate the safety and completeness of AI-generated clinical management plans. Models are presented with real clinical cases and asked to provide free-text management plans, which are then scored by multiple LLM judges against physician-authored rubrics. The benchmark covers cases across multiple medical specialties with perturbations testing robustness to variations in patient demographics, lab values, and clinical context. This project is led and supported by the ARISE AI Research Network, based at Stanford and Harvard.

Read our [study](https://arxiv.org/abs/2512.01241) for more details.
See the live [leaderboard](https://bench.arise-ai.org/) for current rankings.

### Input Format
- **File type:** Plain text (.txt)
- **Content:** Clinical case vignette describing a patient presentation, history, and clinical question
- **Encoding:** UTF-8

### Output Format
- **File type:** JSON (.json)
- **Schema:** Defined in `benchmarks/donoharm/schema.json`
- **Required fields:** `response` (string, minimum 50 characters)
- **Content:** Free-text clinical management plan including assessment and numbered recommendations
- **Also accepted:** OpenAI-compatible chat completions format (content is extracted automatically)

### Test Cases
Currently includes:
- `test_001`: Example case involving immunotherapy management (open-source case from the study)

### Scoring
Models are scored by multiple LLM judges against physician-authored rubrics. Key metrics include Safety, Completeness, Restraint, Precision, Recall, and an Overall Score (harmonic mean of Safety, Completeness, and Restraint). Scoring is performed by the MAST team after submission.

### Validation Process
**Schema Validation:** Output must conform to the benchmark's JSON schema (JSON object with a `response` string field of at least 50 characters).

### File Naming Conventions
- Input files: `test_001.txt`, `test_002.txt`, etc.
- Output files: `test_001.txt`, `test_002.txt`, etc.
- Sequential numbering maintains input-output correspondence

## SCT (Script Concordance Test) Benchmark

*Coming soon.* See `benchmarks/sct/README.md` for preliminary details.

## Adding New Benchmarks
When adding new benchmarks:

1. Follow the established directory structure
2. Create comprehensive documentation in `prompt.md`
3. Define clear validation criteria in `schema.json`
4. Implement robust validation in `validator.py`
5. Include diverse test cases covering edge cases
6. Update this document with benchmark details

See [contributing.md](contributing.md) for detailed instructions on adding benchmarks.
