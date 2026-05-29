# SCT - Script Concordance Test Benchmark

> Examples from [SCT-Bench/sctpublic](https://github.com/SCT-Bench/sctpublic)

## Overview

The Script Concordance Test (SCT) evaluates clinical reasoning by measuring how AI models interpret new clinical information in the context of diagnostic or therapeutic hypotheses.

**By default this is run-it-yourself**: install, run the open dataset locally, and get scores. Scoring is deterministic and free (no LLM judge). Submitting to the public leaderboard is optional and covered at the end.

## Task Description

Given a clinical scenario, a hypothesis (diagnosis or treatment), and new information, the model rates how the new information affects the likelihood of the hypothesis on a 5-point scale:

| Rating | Meaning |
|--------|---------|
| -2 | Strongly decreases likelihood |
| -1 | Slightly decreases likelihood |
| 0 | No effect on likelihood |
| +1 | Slightly increases likelihood |
| +2 | Strongly increases likelihood |

The model returns a JSON object: `{"Rating": 1, "Rationale": "Brief clinical justification"}`.

## Run it yourself (open subset, 174 items)

Run from this directory:

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your provider key
python run.py   --model-config config/models/example.yaml --benchmark-config config/benchmark.yaml
python score.py --model-config config/models/example.yaml --benchmark-config config/benchmark.yaml
# -> results/scores/sct/<model>.csv  (headline: sct_score + bootstrap CI)
```

`run.py` writes `results/raw/sct/<model>.jsonl` with id, trial, response, usage, and metadata. You can skip `run.py` and supply your own JSONL with at least `{id, trial, response}`.

### Metric

The headline metric is `sct_score`: alignment with the expert consensus distribution, scaled 0 to 1. Scores include an item-level percentile bootstrap confidence interval (B=2000, seed=0).

### Reference scores

On the open subset (174 items), scored with this bundle's `dataset/` + `score.py` (deterministic, so exactly reproducible). Values are mean +/- 95% item-level percentile bootstrap half-width.

| Model | sct_score |
| --- | --- |
| GPT-5.5 | 0.745 +/- 0.052 |
| Claude Opus 4.7 | 0.756 +/- 0.056 |
| Gemini 3.1 Pro | 0.721 +/- 0.053 |

## Scoring

Responses are scored against expert physician panel distributions:

- **SCT Score (0-1)**: weighted alignment with expert consensus. A response matching the most common expert answer scores 1.0; responses matching less common expert answers score proportionally lower.
- **Expert Set Match**: binary measure of whether the response matches any expert's answer.

See `DATA.md` for data provenance and sources.

## Data Sources

The full benchmark includes 750 questions from 10 medical institutions; this bundle ships the 174-item open subset (Adelaide SCT + Open Medical SCT).

- Adelaide SCT (Medicine, Surgery, Psychiatry, Pediatrics, OB-Gyn)
- IU Emergency Medicine (3 cohorts)
- McGill Neurology
- Singapore Internal Medicine & Neurology
- Indianapolis Physiotherapy
- Montefiore Pediatrics

## References

- **SCT-Bench paper**: [McCoy et al., NEJM AI 2025](https://ai.nejm.org/doi/full/10.1056/AIdbp2500120)
- Script Concordance Testing methodology: [Fournier et al., 2008](https://pubmed.ncbi.nlm.nih.gov/18785963/)
- Public examples and templates: [SCT-Bench/sctpublic](https://github.com/SCT-Bench/sctpublic)

## Submitting to the leaderboard (optional)

If you want your model on the public [leaderboard](https://benchmarks.arise-ai.org), your hosted API endpoint must first pass a format check, then you register it. See [`submission/`](submission/) for the endpoint validator and the 5 calibration cases, and `docs/submission_agreement.md` (repo root) for the Registration Form and terms. This step is only for leaderboard submission; it is not needed to run the benchmark yourself.

## License

Code is MIT-licensed (see `LICENSE`). Data provenance and sources are in `DATA.md`.
