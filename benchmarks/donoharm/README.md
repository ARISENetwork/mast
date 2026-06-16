# NOHARM: Clinical Harm Benchmark

NOHARM is a clinical-harm benchmark where models write free-text management plans for real-world specialty-consult cases, scored by an LLM judge against specialist-authored rubrics. This bundle is the open subset (30 cases).

## Two paths

- **Run it yourself (this guide).** `run.py` + `score.py` generate your model's plans and judge them locally over the 30-case open subset, producing a scored CSV. This is what most readers want.
- **Endpoint conformance check.** If you plan to submit a hosted API to the leaderboard, `submission/validator.py` (run via the repo-root `validate_all.py`, reading your endpoint URL/token from `scripts/config.json`) POSTs one sample case and checks the reply against `submission/schema.json`. It only confirms your API returns the expected format - it does not run or score the benchmark. ARISE runs the full evaluation against your endpoint after you submit; this is just a pre-submission self-check.

## Cost and Requirements

**IMPORTANT**: Scoring calls Google Gemini judges and **costs money** (a few dollars for a full open-subset run, more for reasoning models). You must provide `GEMINI_API_KEY` in your `.env` file for scoring to work. See the example below.

## Installation

Requires Python >= 3.10.

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your model key AND GEMINI_API_KEY (judge)
```

## Running

```bash
# Inference: generate model responses
python run.py   --model-config config/models/example.yaml --benchmark-config config/benchmark.yaml

# Scoring: judge responses and compute metrics
python score.py --model-config config/models/example.yaml --benchmark-config config/benchmark.yaml

# Output: results/scores/donoharm/default/<model>.csv (headline: F1_weighted + CI)
```

Output paths are anchored two directories above this one (the repository
root in the MAST layout), not the current working directory: raw responses
land in `<repo-root>/results/raw/donoharm/<prompt>/` and score CSVs in
`<repo-root>/results/scores/donoharm/<prompt>/`.

`<prompt>` is `default` unless you pass `--prompt` (to both `run.py` and
`score.py`). The alternates change only the length instruction - `limit500`
(under 500 words), `limitnone` (no word cap), `zero` (no system prompt at all)
- and exist for the paper's length-sensitivity analysis. Use `default` to
match the reference table.

## How It Works

`run.py` generates responses and writes them to `results/raw/donoharm/default/<model>.jsonl` with records containing `{id, trial, response, usage}`. Here `<model>` is the `name` field in your model config (the output/cache namespace), which is separate from `model_id` (the model litellm actually calls). You may also supply your own JSONL file with at least `{id, trial, response}` fields and skip the inference step.

The judge caches per `(case, trial)` under `_strategy/<model>/`, so if you change a model's responses and re-score, clear that model's cache (or score under a new model name) to force a re-judge. Because outputs and the cache are keyed on `name` (not `model_id`), switching `model_id` while keeping the same `name` reuses the previous model's cached responses and judgments - rename `name` when you change models.

Item IDs follow the format `<CaseId>` for base cases and `<CaseId>-<n>` for perturbations (n = 0 to 9).

See `examples/` for a worked example: one full case (`All001`, all 11 variants) - the inference input (`gpt-5.5.jsonl`), the judge output it produces (`gpt-5.5_judged.jsonl`), and the score CSV (`gpt-5.5.csv`).

## Dataset Size

The bundle contains 30 open base cases with up to 11 variants each (330 total available):
- Default (no `--k` flag): all 11 variants, 330 items per model (1 base + 10 perturbations per case); this is what the frontier leaderboard rows and the reference table below use, so the default is the setting to match
- `--k 5`: 150 items per model (1 base + 4 perturbations); a cheaper run whose metrics sit close to k=11 (saturation-validated) but does not exactly reproduce the reference table
- `--k N` (N from 1 to 11): cap variants per case at N

`--k` (and `--limit`) is applied at generation, judging, and scoring, so the
CSV always reflects exactly the variants/cases you ask for. Re-running
`score.py` at a different `--k` just works, with no cache or file to clear
first (the judge cache grows to cover any newly added variants and never
re-bills the ones already judged). `python score.py --rescore` recomputes
metrics from the existing judged file without new judge calls (useful after a
metric-definition change), then scores the requested `--k`/`--limit` scope.

## Metric

The headline metric is **F1_weighted**: a severity-weighted F1 scored over the rubric-matched actions in the model's plan (off-rubric verbosity excluded), capped to 0 on any variant that commits a Severe harm. It is the Severe-capped harmonic mean of `Precision_weighted` (severity-weighted precision over rubric-matched actions, uncorrected) and `Recall_weighted` (severity-weighted recall, length-corrected). Reported with a 95% stratified cluster bootstrap confidence interval (B=2000 bootstrap samples).

The score CSV reports `F1_weighted`, `Precision_weighted`, `Recall_weighted`, `Severe_rate`, and the aggregate-only `F1_floor` (worst-variant `F1_weighted`). See DATA.md for full definitions, rubric scale, and bootstrap methodology.

## Judge

Scoring runs a Gemini judge in two stages, **match + review**, which produce the headline `F1_weighted`; the reference table below was computed this way. Pinned judge model IDs:
- Match: `gemini/gemini-3-flash-preview`
- Review: `gemini/gemini-3.5-flash`

For details on judge reproducibility, see DATA.md. Note that Gemini preview models can drift across revisions, so exact numeric reproduction is not guaranteed, though the pinned IDs document what produced the published numbers.

## Comparing Your Numbers

Reference scores on this open subset (30 cases, **k=11**: all 330 variants), headline metric `F1_weighted` with weighted Precision/Recall alongside. Values are mean +/- 95% stratified cluster bootstrap half-width (B=2000, seeded).

| Model | F1_weighted | Precision (weighted) | Recall (weighted) |
| --- | --- | --- | --- |
| GPT-5.5 | 0.726 +/- 0.072 | 0.886 +/- 0.046 | 0.752 +/- 0.052 |
| Claude Opus 4.7 | 0.619 +/- 0.086 | 0.827 +/- 0.062 | 0.687 +/- 0.061 |
| Gemini 3.1 Pro | 0.569 +/- 0.082 | 0.807 +/- 0.063 | 0.587 +/- 0.068 |

These were produced with the pinned judge (see DATA.md) at k=11, which is the default, so the default run is directly comparable. Your numbers may still differ because the judge routes to Gemini models that can drift across revisions; a cheaper `--k 5` run differs further, since it scores only a subset of the variants behind this table. Treat these as reference points rather than exact targets.

## Citation

If you use NOHARM, please cite:

> Wu et al. *First, do NOHARM: towards clinically safe large language models.* arXiv:2512.01241, 2025. <https://arxiv.org/abs/2512.01241>

```bibtex
@misc{wu2025noharm,
  title         = {First, do {NOHARM}: towards clinically safe large language models},
  author        = {David Wu and Fateme Nateghi Haredasht and Saloni Kumar Maharaj and Priyank Jain and Jessica Tran and Matthew Gwiazdon and Arjun Rustagi and Jenelle Jindal and Jacob M. Koshy and Vinay Kadiyala and Anup Agarwal and Bassman Tappuni and Brianna French and Sirus Jesudasen and Christopher V. Cosgriff and Rebanta Chakraborty and Jillian Caldwell and Susan Ziolkowski and David J. Iberri and Robert Diep and Rahul S. Dalal and Kira L. Newman and Kristin Galetta and J. Carl Pallais and Nancy Wei and Kathleen M. Buchheit and David I. Hong and Ernest Y. Lee and Allen Shih and Vartan Pahalyants and Tamara B. Kaplan and Vishnu Ravi and Sarita Khemani and April S. Liang and Daniel Shirvani and Advait Patil and Nicholas Marshall and Kanav Chopra and Joel Koh and Adi Badhwar and Liam G. McCoy and David J. H. Wu and Yingjie Weng and Sumant Ranji and Kevin Schulman and Nigam H. Shah and Jason Hom and Arnold Milstein and Adam Rodman and Jonathan H. Chen and Ethan Goh},
  year          = {2025},
  eprint        = {2512.01241},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CY},
}
```

## License and Data

- **Code**: MIT (LICENSE)
- **Data and rubrics**: CC-BY-4.0 (see DATA.md for full provenance)
