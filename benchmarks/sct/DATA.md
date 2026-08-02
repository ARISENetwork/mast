# SCT data provenance

The bundled dataset is the **open subset** of the Script Concordance Test
benchmark: the `Open Medical SCT` and `Adelaide SCT` sources (174 items of 750).
The other eight sources are held out to keep the leaderboard uncontaminated and
are not included here, in any form.

- Source: the SCT-Bench corpus (McCoy et al., NEJM AI 2025). The 174 open items
  and their expert consensus distributions are the ones published at
  [SCT-Bench/sctpublic](https://github.com/SCT-Bench/sctpublic).
- License: MIT, covering both the code and the bundled data (see `LICENSE`).
- Files: `dataset/items.jsonl` (cases), `dataset/rubrics.jsonl` (expert
  consensus distributions), `dataset/baselines.json` (published human/model
  baselines for the open subtests).

Scoring is deterministic against the expert distributions; no LLM judge is
involved.

Case text is carried through byte-for-byte from the published source, including
a pre-existing mojibake artifact in some Adelaide items: the degree symbol
`˚` (U+02DA) appears as the two characters U+00CB U+009A, its UTF-8 bytes
decoded as Latin-1. This is left as-is deliberately: every reference score in
`README.md` was produced against exactly this text, and the same artifact is
present upstream, so silently repairing it would break reproducibility without
fixing the source.
