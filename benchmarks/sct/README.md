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

`score.py` reads the rating from either shape a model might return: the plain-text answer line the kit's prompt elicits (`Rating: X`), or a JSON field (`{"Rating": N, ...}`) as produced by structured outputs and by the leaderboard submission endpoint (see [`submission/`](submission/)). Both parse identically, so a JSONL captured from a submission-format endpoint scores without conversion.

A complete worked run is in [`examples/`](examples/): a real GPT-5.5 run over all 174 items plus the score outputs it produced.

### Metric

The headline metric is `sct_score`: alignment with the expert consensus distribution, scaled 0 to 1. Scores include an item-level percentile bootstrap confidence interval (B=2000, seed=0).

### Reference scores

On the open subset (174 items), scored with this bundle's `dataset/` + `score.py` (deterministic, so exactly reproducible). Values are mean +/- 95% item-level percentile bootstrap half-width. The GPT-5.5 row is reproducible from the run shipped in [`examples/`](examples/).

| Model | sct_score |
| --- | --- |
| GPT-5.5 | 0.745 +/- 0.053 |
| Claude Opus 4.7 | 0.756 +/- 0.057 |
| Gemini 3.1 Pro | 0.721 +/- 0.058 |

## Scoring

Responses are scored against expert physician panel distributions:

- **SCT Score (0-1)**: weighted alignment with expert consensus. A response matching the most common expert answer scores 1.0; responses matching less common expert answers score proportionally lower.
- **Expert Set Match**: binary measure of whether the response matches any expert's answer.

The scorecard also carries three directional error rates, each defined only on the items that create the opportunity (so each has its own denominator, reported in the CSV's `trials` column):

- **Overconfidence**: on items where the expert mode is +/-1, how often the model answered +/-2 in the same direction.
- **Underconfidence**: on items where the expert mode is +/-2, how often the model answered +/-1 in the same direction.
- **Distractor Susceptibility**: on items where the expert mode is 0, how often the model moved off 0.

Items whose expert mode is tied are excluded from all three. These are diagnostic breakdowns, not part of the headline `sct_score`.

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

If you want your model on the public [leaderboard](https://arise-ai.org/mast/technical), your hosted API endpoint must first pass a format check, then you register it. See [`submission/`](submission/) for the endpoint validator and the 5 calibration cases, and `docs/submission_agreement.md` (repo root) for the Registration Form and terms. This step is only for leaderboard submission; it is not needed to run the benchmark yourself.

## License

Code is MIT-licensed (see `LICENSE`). Data provenance and sources are in `DATA.md`.
