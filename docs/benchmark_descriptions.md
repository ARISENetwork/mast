# Benchmark Descriptions

This document provides detailed descriptions of the benchmarks with runnable kits in this repository. The full MAST suite on the public leaderboard also includes CPC-Bench (diagnostic reasoning), MedAgentBench v2 and PhysicianBench (agentic EHR tasks), ReXrank Mini (radiology), and Multimodal Images (dermatology); see [arise-ai.org/mast/benchmarks](https://arise-ai.org/mast/benchmarks) for the complete set.

## First Do NOHARM Benchmark

### Purpose
First Do NOHARM is a physician-validated medical benchmark to evaluate the safety and completeness of AI-generated clinical management plans. Models are presented with real clinical cases and asked to provide free-text management plans, which are then scored by multiple LLM judges against physician-authored rubrics. The benchmark covers cases across multiple medical specialties with perturbations testing robustness to variations in patient demographics, lab values, and clinical context. This project is led and supported by the ARISE AI Research Network, based at Stanford and Harvard.

Read our [study](https://arxiv.org/abs/2512.01241) for more details.
See the live [leaderboard](https://arise-ai.org/mast/technical) for current rankings.

### Run it yourself
First Do NOHARM ships as a run-it-yourself kit over a 30-case open subset (330 perturbation variants), scored by a Gemini LLM judge (costs a few dollars per run). The headline metric is `F1_weighted`: severity-weighted F1 over rubric-matched actions (no severity cap; Severe harms are reported separately as `Severe_rate`), with a 95% stratified cluster-bootstrap CI. See `benchmarks/donoharm/README.md` for setup, reference scores, and data provenance.

### Input Format
- **File type:** Plain text (.txt)
- **Content:** Clinical case vignette describing a patient presentation, history, and clinical question
- **Encoding:** UTF-8

### Output Format
- **File type:** JSON (.json)
- **Schema:** Defined in `benchmarks/donoharm/submission/schema.json`
- **Required fields:** `response` (string, minimum 50 characters)
- **Content:** Free-text clinical management plan including assessment and numbered recommendations
- **Also accepted:** OpenAI-compatible chat completions format (content is extracted automatically)

### Test Cases
Currently includes:
- `test_001`: Example case involving immunotherapy management (open-source case from the study)

### Scoring
Models are scored by an LLM judge pipeline (Gemini match + review stages) against physician-authored rubrics. The headline metric is `F1_weighted`: severity-weighted F1 over rubric-matched actions, reported alongside weighted Precision/Recall, `Severe_rate`, and the perturbation aggregates `F1_floor` and `Resilience` (see `benchmarks/donoharm/README.md` for definitions). Scoring for leaderboard submissions is performed by the MAST team after submission.

### Validation Process
**Schema Validation:** Output must conform to the benchmark's JSON schema (JSON object with a `response` string field of at least 50 characters).

### File Naming Conventions
- Input files: `test_001.txt`, `test_002.txt`, etc.
- Reference output files: `test_001.txt`, `test_002.txt`, etc. — these contain the plain-text content of the `response` field; the API itself must return the JSON envelope described above
- Sequential numbering maintains input-output correspondence

### Submission
Submitting to the public leaderboard is optional and separate from running the benchmark locally. See `benchmarks/donoharm/submission/` for the endpoint validator and calibration case.

## SCT (Script Concordance Test) Benchmark

### Purpose
The Script Concordance Test measures probabilistic clinical reasoning under uncertainty. Given a clinical scenario, a hypothesis (diagnosis or treatment), and a new piece of information, the model rates how that information shifts the likelihood of the hypothesis on a 5-point scale (-2 to +2). Responses are scored against expert physician panel distributions.

### Run it yourself
SCT ships as a run-it-yourself kit over a 174-item open subset (Adelaide SCT + Open Medical SCT) with deterministic scoring (no LLM judge). The headline metric is `sct_score`: alignment with the expert consensus distribution, scaled 0 to 1, with an item-level bootstrap confidence interval. See `benchmarks/sct/README.md` for setup, reference scores, and data provenance.

### Output Format
- **File type:** JSON (.json)
- **Required fields:** `Rating` (integer, -2 to +2) and `Rationale` (string)
- **Also accepted:** OpenAI-compatible chat completions format (content is extracted automatically)

### Submission
Submitting to the public leaderboard is optional and separate from running the benchmark locally. See `benchmarks/sct/submission/` for the endpoint validator and calibration cases.

## Adding New Benchmarks
When adding new benchmarks:

1. Follow the established directory structure
2. Create comprehensive documentation in `prompt.md`
3. Define clear validation criteria in `schema.json`
4. Implement robust validation in `validator.py`
5. Include diverse test cases covering edge cases
6. Update this document with benchmark details

See [contributing.md](contributing.md) for detailed instructions on adding benchmarks.
