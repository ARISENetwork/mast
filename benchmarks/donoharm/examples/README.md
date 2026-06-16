# Worked example

A full worked case: all 11 variants of `All001` (the base case plus its 10
perturbations), so you can see the shape of the pipeline's input and outputs,
including the perturbation structure that NOHARM is built around. These files
are illustrative only; they are not part of the dataset, and a real run writes
its own copies under `<repo-root>/results/`.

`All001` is a clean, high-scoring case (per-variant `F1_weighted` 0.986-1.000,
no harms flagged), chosen so the example shows the format without judging edge
cases.

## `gpt-5.5.jsonl` (inference)

11 records, one per variant, exactly what `run.py` writes:

- `id`, `trial` - item identity. `All001` is the base case; `All001-0` .. `All001-9` are perturbations
- `response` - the model's free-text management plan (real GPT-5.5 output)
- `usage`, `runtime` - token counts and wall-clock seconds

You can hand-write a file in this shape (`{id, trial, response}` minimum) and
skip the inference step.

## `gpt-5.5_judged.jsonl` (judge output)

11 records, exactly what `score.py` writes after the match + review judge
stages (each validates against `judge/schemas/judged.schema.json`):

- `options` - per-rubric-option verdicts: `matched` / `partial` (+ `evidence`)
- `responseActions` - the model's actions parsed out of its response
- `harm` - per-option harm records (empty across this case: nothing harmful omitted)
- `metrics` - the per-record metric block; `F1_weighted` is the headline
- `judge` - the review-stage judge id (`gemini-3.5-flash`)

## `gpt-5.5.csv` (score output)

The aggregated scorecard `score.py` writes to
`results/scores/donoharm/default/<model>.csv`: one row per metric with `mean`
and a bootstrap CI (`ci`, `ci_lo`, `ci_hi`). Because this case has >= 2 variants,
the aggregate-only metrics `F1_floor` (worst variant) and `Resilience`
(per-option consistency across perturbations) are included alongside the
weighted/raw families.

One caveat: this CSV covers a single base case, so `ci` is 0 - the stratified
cluster bootstrap resamples base cases, and there is only one here. A real run
over the full open subset (30 cases) produces real CIs; see the reference table
in the top-level README.

## Reproducing it

```bash
# (with your own model key + GEMINI_API_KEY in .env)
python run.py   --model-config config/models/example.yaml --benchmark-config config/benchmark.yaml
python score.py --model-config config/models/example.yaml --benchmark-config config/benchmark.yaml
```

Your records will match these in shape. Exact numbers can vary slightly because
the judge routes to Gemini preview models that can drift across revisions.
