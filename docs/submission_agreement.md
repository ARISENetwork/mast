# Medical AI Superintelligence Test (MAST)  
## Submission Agreement

---

## Section 1. Submission Process

1. Review this document in its entirety.
2. If you are submitting a custom model, please review all instructions on this GitHub repository to validate your API endpoint.
3. Fill out the [Registration Form](https://forms.gle/4exSPLbsmWjNmMRQ7) and agree to the terms of use.
4. We will review and verify your submission.
5. If verified, your model will be evaluated on the private test set for each benchmark.
6. Results will be reviewed by the MAST team for approval.
7. If approved, results will be permanently indexed on the public leaderboard.

---

## Section 2. Registration

All submissions are subject to a review process at the discretion of the MAST team to ensure responsible, ethical, and fair use of the benchmark. Submission of the registration form does **not** guarantee acceptance or publication on the leaderboard. See above for the link to the Registration Form.

---

## Section 3. Eligibility

### Availability

Models must be publicly accessible through at least one of the following:

- A commercial product with a public-facing website available to end users or clinicians
- A public API or inference platform (e.g., OpenRouter, cloud provider marketplace)

### Notability

To maintain leaderboard quality and prevent flooding, submitted models must satisfy **at least two** of the following notability criteria, adapted from [Wikipedia's general notability guideline](https://en.wikipedia.org/wiki/Wikipedia:Notability):

1. **Independent coverage** — the model or its parent organization has received coverage in independent reliable sources (e.g., news outlets, peer-reviewed journals, industry publications)
2. **Platform listing** — the model is listed on a major model hub or inference platform (e.g., HuggingFace, OpenRouter, AWS Bedrock, Google Vertex, Azure AI)
3. **Technical documentation** — a published paper, preprint, or technical report describing the model's architecture or training
4. **Commercial availability** — the model is part of a commercially available product with a public-facing website
5. **Institutional affiliation** — the model is developed by a recognized company, research lab, or academic institution

The MAST team retains final discretion over notability determinations. Submissions that do not clearly meet these criteria may be declined without further review.

### Distinctiveness

Submissions must represent **substantively distinct models**. Trivial variations — such as different system prompts, thin wrappers, or minor configuration changes applied to a model already on the leaderboard — will not be accepted as separate entries.

## Section 4. Submission Requirements

- Models not publicly available via common inference aggregators (e.g., OpenRouter) must provide a **stable custom API endpoint** for MAST evaluation. Submitting organizations are responsible for ensuring endpoint availability, correctness, and compliance with the expected input/output schema throughout the evaluation period.

- Models evaluated via custom API endpoints must operate in a **zero data-retention mode**. Submitting organizations must **not store, log, cache, reuse, or train on** any inputs, prompts, or outputs generated during MAST evaluation.

- Submitting organizations must follow the instructions and test examples provided in the GitHub repository to ensure their endpoint correctly handles the expected inputs and outputs.

- Submitting organizations are responsible for **token costs** incurred during evaluation. Estimated token usage and evaluation parameters are provided for reference and may vary.

- All results will be **manually inspected and verified** by the MAST team prior to public indexing on the leaderboard.

- The MAST team retains **final authority** over verification decisions.

- All verified and non-disqualified results will be posted publicly to the leaderboard. Once submitted and evaluated, model scores are **final** and will remain publicly available.

- The MAST team reserves the right to **publish, analyze, and report** submitted results — including model names, scores, and aggregate statistics — in academic papers, preprints, presentations, and other research communications, without additional consent from submitting organizations.

- To discourage overfitting and repeated probing of the benchmark, the total number of submission attempts will be displayed publicly alongside the score, as well as the date of submission.

- The MAST team reserves the right to **disqualify any submission** made in bad faith or exhibiting suspicious activity, including but not limited to attempts at benchmark reconstruction, prompt leakage, coordinated probing, or violations of the Terms of Use.

- Submission constitutes advance consent for evaluation on future MAST benchmarks and major benchmark revisions. Organizations may decline participation in a newly introduced benchmark; however, models that do not participate will be listed as partially indexed.

---

## Section 5. Terms of Use

Participation in MAST is conditioned on agreement to responsible use of the benchmark. Submitting organizations agree **not** to attempt to reverse engineer, reconstruct, or infer the contents of the private test sets, nor to game or manipulate the evaluation process.

Violations of this policy will result in **immediate disqualification** and **permanent removal** of the organization from all current and future MAST submissions. Additionally, we reserve the right to pursue legal action.

All decisions regarding verification, scoring, disqualification, and leaderboard inclusion are **final and non-appealable**.

---

_Last modified on April 8th, 2026._
