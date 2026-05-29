# SCT data provenance

The bundled dataset is the **open subset** of the Script Concordance Test
benchmark: the `Open Medical SCT` and `Adelaide SCT` sources (174 items of 750).
Held-out sources are not included.

- Source: HealthRex-ARISE/sctbench
- License: MIT
- Files: `dataset/items.jsonl` (cases), `dataset/rubrics.jsonl` (expert
  consensus distributions), `dataset/baselines.json` (published human/model
  baselines for the open subtests).

Scoring is deterministic against the expert distributions; no LLM judge is
involved.
