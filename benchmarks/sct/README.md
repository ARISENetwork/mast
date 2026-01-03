# SCT - Script Concordance Test Benchmark

> Examples from [SCT-Bench/sctpublic](https://github.com/SCT-Bench/sctpublic)

## Overview

The Script Concordance Test (SCT) evaluates clinical reasoning by measuring how AI models interpret new clinical information in the context of diagnostic or therapeutic hypotheses.

## Task Description

Given a clinical scenario, a hypothesis (diagnosis or treatment), and new information, the model must rate how the new information affects the likelihood of the hypothesis using a 5-point scale:

| Rating | Meaning |
|--------|---------|
| -2 | Strongly decreases likelihood |
| -1 | Slightly decreases likelihood |
| 0 | No effect on likelihood |
| +1 | Slightly increases likelihood |
| +2 | Strongly increases likelihood |

## Response Format

Models must return a JSON object:

```json
{
  "Rating": 1,
  "Rationale": "Brief clinical justification for the rating"
}
```

## Scoring

Responses are scored against expert physician panel distributions:

- **SCT Score (0-1)**: Weighted alignment with expert consensus. A response matching the most common expert answer scores 1.0; responses matching less common expert answers score proportionally lower.
- **Expert Set Match**: Binary measure of whether the response matches any expert's answer.

## Example Cases

Five calibration examples are provided in `inputs/` with reference outputs in `outputs/`:

| Example | Scenario | Expected Rating |
|---------|----------|-----------------|
| 001 | Ear infection treatment | -2 |
| 002 | Pediatric GI + fever | -1 |
| 003 | Pregnancy test indication | 0 |
| 004 | Atopic dermatitis + fever | +1 |
| 005 | Trisomy 21 + petechiae | +2 |

## Data Sources

The full benchmark includes 750 questions from 10 medical institutions:

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
