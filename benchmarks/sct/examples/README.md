# Worked example

A complete GPT-5.5 run over the full 174-item open subset: the inference
output, the per-item score details, and the aggregate scorecard. This is the
run behind the GPT-5.5 row in the reference table in `../README.md`, so you can
reproduce that number without spending anything on inference.

Scoring is deterministic, so re-scoring these files reproduces the published
figures exactly.

## `gpt-5.5.jsonl` (inference)

174 records, one per item, exactly what `run.py` writes:

- `id`, `trial` - item identity, matching `dataset/items.jsonl`
- `response` - the model's raw answer text (real GPT-5.5 output). For this
  model the answers are terse (`Rating: -1`) because the reasoning happens in
  tokens the provider does not return; other models emit their rationale here
- `metadata` - `source` / `source_short`; `score.py` uses `source_short` for the
  per-subtest rows
- `usage`, `runtime` - token counts and wall-clock seconds

You can hand-write a file in this shape (`{id, trial, response}` minimum) and
skip the inference step. One field was removed from the published copy: the
provider's per-request `usage.message_id`, which is account-scoped and carries
no information about the benchmark.

## `gpt-5.5_details.jsonl` (per-item scores)

174 records, exactly what `score.py` writes alongside the CSV:

- `rating` - the rating parsed out of `response`
- `expert_modal_rating` - the expert panel's unique modal answer, or `null`
  when the panel's mode is tied
- `normalized_score` - the item's contribution to `sct_score` (expert weight
  for the chosen rating, divided by the panel's max weight)
- `in_expert_set` - whether any expert chose that rating
- `overconfidence_rate` / `underconfidence_rate` / `distractor_susceptibility` -
  present only on items that create the relevant opportunity, which is why they
  appear on some records and not others

## `gpt-5.5.csv` (scorecard)

The aggregated scorecard `score.py` writes to
`results/scores/sct/<model>.csv`: one row per metric per category, with `mean`
and an item-level percentile bootstrap CI (`ci` is the half-width; B=2000,
seed=0). `Overall` covers all 174 items; `adelaide` and `open_medical` are the
two open subtests.

Headline row: `Overall,sct_score,174,0.7453,0.0531,...`

Note the `trials` column is an item count, not a repeat count. It varies by
metric because the three confidence metrics are opportunity-normalized: only
68 items have a unique modal expert answer of +/-1 (overconfidence), 75 have
+/-2 (underconfidence), and 19 have 0 (distractor susceptibility).

## Reproducing it

```bash
# from benchmarks/sct/, with your own provider key in .env
python run.py   --model-config config/models/example.yaml --benchmark-config config/benchmark.yaml
python score.py --model-config config/models/example.yaml --benchmark-config config/benchmark.yaml
```

To re-score these exact files instead of running inference, copy
`gpt-5.5.jsonl` to `<repo-root>/results/raw/sct/gpt-5.5.jsonl`, point a model
config at `name: gpt-5.5`, and run `score.py`. You will get this CSV back
byte-for-byte.
