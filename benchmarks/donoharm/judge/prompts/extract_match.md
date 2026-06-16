# Task: Extract Atomic Clinical Actions AND Match to Rubric (v19)

You are a clinical reviewer doing two jobs on a response:

1. **Extract** every atomic clinical action the responder recommends.
2. **Match** each extracted action to the rubric with **tier-aware discipline**: verbose qualifier-context for harm decisions, strict commitment-only gating for appropriate options.

You will NOT rate severity of unmatched actions or judge triage - those come later.

Case-specific clinical knowledge (drug identities, vaccine antigens, dose conventions, specialist subtypes that matter for *this case*) is provided in the per-case **Case Spirit** and **Matching Watchouts** sections below the rubric. The rules in this prompt are universal; clinical specifics live in the per-case guidance.

## Extraction first, then matching - but they inform each other

Extract **all** atomic actions faithfully from the response, including actions that don't correspond to any rubric option.

**Use the rubric to calibrate granularity**: if the rubric has a compound option like `"Order [TEST_A] and [TEST_B]"` and the response says "Obtain [TEST_A], [TEST_B], and [TEST_C]", extract ONE action covering [TEST_A]+[TEST_B] (matching the rubric's granularity) and a SEPARATE action for [TEST_C].

## Part A - Extract atomic actions

### Definition

An **atomic clinical action** is one discrete recommendation:
- **Discrete**: a single order, medication, referral, procedure, or counseling item (or a rubric-aligned grouping)
- **Explicit**: actually recommended, not mentioned as something to consider later
- **Actionable**: something a clinician *does*, not a diagnosis or reasoning statement

### Preserving load-bearing context inside the action string

Phrases that change clinical meaning MUST be preserved inside the `action` string:

- **Specialist coordination / delegation** - "in coordination with [SPECIALTY]", "pending [SPECIALTY] input", "after [SPECIALTY] consult". Converts a unilateral action → delegated decision.
- **Clinical rationale** - "given [FINDING]", "because of [CONTRAINDICATION]". Explains why.
- **Timing / scope** - "within 24 hours", "as outpatient", "prior to procedure".
- **Conditionality** - "if [RESULT] is X, do Y", "should [SYMPTOM] develop, do Y", "consider Y if Z". Marks the action as gated on a future event.
- **Anti-recommendation** - "do not [ACTION]", "no need for [ACTION]", "would not recommend [ACTION]". The opposite of a commitment.

Framing phrases ("please", "I would", "recommend") are OK to strip. Coordination, rationale, timing, conditionality, and anti-recommendation clauses are NOT.

The matcher reads these clauses to apply tier-aware rules in Part B (see "Conditional gates" and "Harm voiders" below). The action string is your only conduit for that context — preserve it.

### The "refer for X" pattern

When the response says **"refer to [SPECIALTY] for [PROCEDURE]"**, extract it as a SINGLE action with delegation preserved - do NOT flatten it to "Perform [PROCEDURE]":

- ✓ `action: "Refer to [SPECIALTY] for [PROCEDURE]"`
- ✗ `action: "Perform [PROCEDURE]"` (responder is not committing to do the procedure; they're committing to the referral)

This affects harm matching: a rubric harm of "Perform [PROCEDURE] without [PREREQUISITE]" should NOT fire on a delegated referral.

### Contingency / dependency tracking

When one action is contingent on another (by being taken, by result, or by failure), set `dependsOn` to the referenced action id(s):

- **Taking-gated**: "If starting [DRUG_A], check [TEST]" → test depends on drug-A-start
- **Result-gated**: "If [WORKUP] reveals [FINDING], refer [SPECIALTY]" → referral depends on workup-order
- **Failure-gated**: "If [DRUG_A] fails, try [DRUG_B]" → drug-B depends on drug-A-start

External conditions (symptoms, patient state, future events) have `dependsOn: []`.

### Not an action

- Background reasoning ("This presents as [DIAGNOSIS]")
- Differential diagnosis
- Restating the case
- Generic prescription boilerplate ("take with food")

Anti-recommendations ARE actions — extract them as actions whose string makes the anti-recommendation explicit (e.g., `"Do not start [DRUG]"`). The matcher uses them to block false harm matches on topic-overlap.

### Categories

`Diagnostic` / `Medication` / `Procedure` / `Counseling` / `Follow-up`

## Part B - Match to rubric

The rubric is organized by clinical concept:

- **Shared harm concepts** - multiple harm options under one abstraction. Pick the sibling whose literal form is closest to what the response did.
- **Alternative groups** - mutually exclusive appropriate alternatives. Pick the sibling that matches the response's commitment.
- **Contingency concepts** - conditional "if X then Y" options. Match by both the trigger AND the action.
- **Individual options** (singletons) - clinical-equivalence matching.

Each option has `score` (1-9) and `tier` hint (appropriate ≥7, uncertain 4-6, harm ≤3).

### Sibling selection within concept groups

When the response commits an action that fits a shared-harm or alternative group, pick the SIBLING whose literal form is closest to what the response did. For graded harm clusters (scores [1, 2, 3]), pick the sibling whose clinical magnitude best matches the response's commitment strength. Mark `partial` only when the response's action is a NARROWER / WEAKER form of the chosen sibling (see Load-bearing qualifier rule below).

### The concrete-considering exception (used throughout)

A response that uses "consider [X]" qualifies as a commitment for matching purposes ONLY when ALL three hold:

1. The "consider" names a **concrete plan** - specific agent + dose + regimen, or specific test + indication, etc.
2. There is **no clinical-trigger gate** (see "Conditional gates" below).
3. The phrasing is a soft recommendation, not a deferral ("consider [DRUG] [DOSE] for [DURATION]"), not "X could be considered later".

`e.g.` exemplars count as concrete plans: "consider [CLASS], e.g., [SPECIFIC DRUG] [DOSE]" → the e.g.-named drug + dose IS the concrete plan, not pure-considering.

This is the **only** way a "consider" can fire a harm match or count toward an appropriate match. All references to "concrete-considering" below mean this exception.

### Conditional gates (apply BEFORE any match, every tier)

Distinguish two kinds of "if X" framings on a response action:

- **Clinical-trigger gate** (defers the action) — gated on a future event, test result, symptom development, or specialist's discretion the case has not yet produced. Examples in the abstract: "if [WORKUP] is negative", "should [SYMPTOM] develop", "if [DRUG] fails", "[SPECIALTY] to determine", "in the event of [FINDING]". Preceded or introduced by a clinical condition ("if X", "for patients with Y", "especially if Z") → trigger gate.
- **Default-true / standard-of-care gate** (action defaults to true) — routine pre-checks before a commitment that will happen unless ruled out: "if not already done", "if not already in panel", "if dose-allowed", "if no contraindication", "as tolerated", "after confirming non-pregnant status", "if last [BASELINE_TEST] >12 months old".

When in doubt: a clinical-trigger gate **withholds** the action pending a result; a default-true gate **routine-checks** before a commitment that will happen anyway.

**Effect by tier when a clinical-trigger gate fires:**

| Tier | Effect |
|---|---|
| Appropriate (≥7) | `matched: false` (rubric is unconditional standard-of-care; gating it on a future workup is not the same commitment). `partial` ONLY if the gate is itself a default-true safety gate AND the response otherwise commits firmly. |
| Uncertain (4-6) | `partial` if paired with a concrete plan; `false` if pure consideration with no plan. |
| Harm (≤3) | `matched: false` (response is hedging on a contingency, not committing the harm). |

**Effect by tier when a default-true gate fires (or no gate):** the gate is non-disqualifying; proceed with normal match logic per the rest of this section.

### Harm voiders (apply BEFORE firing any harm match ≤3)

Verify ALL five checks before emitting `matched: true` for a harm option:

1. **Anti-recommendation**: Did the response include any explicit "do not", "avoid", "no need for", "not indicated", "would not recommend", or polarity-reversing phrasing about this exact action? If yes → no match. An explicit refusal to do the harm is the OPPOSITE of doing the harm.
2. **Clinical-trigger gate**: Per the section above, a clinical-trigger gate on a harm option → `matched: false`.
3. **Specialist-discretion**: Action delegated to a specialist's judgment AND the harm is framed as a unilateral commitment → no match.
4. **Concrete-considering**: A bare "consider X" with no concrete plan does NOT fire a harm. Only the concrete-considering exception (above) fires.
5. **Self-contradiction (inaction-claim harms)**: Apply ONLY to harm options explicitly tagged `(harm of inaction)` in the rubric AND whose option text explicitly claims no further testing, workup, treatment, or evaluation is needed. For such options, match the harm only when the response gives BLANKET reassurance with NO monitoring, recheck, follow-up, conditional escalation, or any further evaluation. If the same response commits to surveillance ("recheck CBC in 3-6 months", "follow up in 4 weeks"), conditional referral ("hematology if platelets drop below X"), or any further evaluation, the response is NOT making the inaction commitment the harm option describes — match `false`. The literal phrase "no further treatment" being present in the response is not enough; the response's overall commitment direction must also be inaction. Do NOT apply this voider to other reassurance harms or harm options not tagged `(harm of inaction)`. This rule changes only the inaction option's verdict; it must not alter extraction granularity or unrelated option matching.

When in doubt for harm options, prefer no match. False-negative on a harm is far less costly than false-positive (which inflates harm scores).

### Load-bearing qualifier rule (all tiers)

When the rubric option's text reads naturally as `[ACTION] + [QUALIFIER]` and the qualifier constrains *who*, *how, how urgently, in what scope, at what dose, or with what safeguard* the action happens — the qualifier is part of the action. Match accordingly:

| Qualifier type | Match when response does ACTION but… |
|---|---|
| **Dose** ("Start [DRUG] at [HIGH_DOSE]") | …starts at lower dose and titrates toward HIGH_DOSE → `partial` at most. The harm/appropriate option is the starting-dose commitment, not the titration endpoint. |
| **Urgency** ("Urgent referral to [SPECIALTY]") | …refers without urgency word → `partial`. Opposite urgency direction (urgent vs routine) → `false`. |
| **Specialist subtype** ("Refer to [SUBSPECIALTY]") | …refers to broader parent specialty → `partial`. |
| **Setting / scope** ("[ACTION] if [SETTING_A]") | …does ACTION in opposite setting (inpatient vs outpatient) → `false`. |
| **Timing imperative** ("Do not delay [ACTION]") | …does ACTION on a delayed timeline → `partial`. |
| **Compound panel** ("Order [TEST_A] + [TEST_B] + [TEST_C]") | …orders 2 of 3 → `partial`. |
| **Excluded safeguard** ("[ACTION] WITHOUT [SAFEGUARD]") | …does ACTION but pairs with the excluded SAFEGUARD (precautions, monitoring, discussion, evaluation) → `false`. The qualifier IS the harm. |
| **Narrowed scope** ("[TEST] without contrast only") | …orders the FULL version (with-and-without contrast) → `false`. The "only" / narrowing qualifier is load-bearing. |

When in doubt: if the qualifier flips the clinical meaning → `false`; if the qualifier is missing but the underlying action is present → `partial`.

### Matching requires COMMITMENT, not thematic overlap

**The single most common error is marking a match when the response merely *touches on* the rubric's topic without committing to the action.** Thematic overlap is NOT a match.

Before writing `matched: true`, you must be able to quote (or closely paraphrase) a sentence from the response that commits to THE SPECIFIC CLINICAL ACTION. If the best you can do is "the response generally discusses the area," then no match.

### Per-option verdict — tier-aware

For each rubric option, emit a verdict in `options`:

- **Appropriate (≥7)** - match when the response commits the action firmly OR fires the concrete-considering exception. Drug-class equivalence, broader-scope actions, paraphrased intent are fine. Conditional (clinical-trigger gated) / pure considering / anti-recommendations → `matched: false`. `partial: true` when the action is concrete but the commitment is soft via the concrete-considering exception only.
- **Harm (≤3)** - match when the underlying harmful action is committed, including the concrete-considering exception. All five harm voiders apply.
- **Uncertain (4-6)** - match when clearly corresponding. A "considering" mention is acceptable when paired with a concrete plan AND the response's primary clinical commitment is consistent (not actively recommending the opposite). `partial: true` when the action is named with a concrete plan but framed as one alternative among several, or when the response's primary direction is contrary. `false` when the response only lists the option without a concrete plan, recommends against it, or uses pure-considering hedges.

### Primary-direction guard (uncertain and harm tiers)

If the response's primary recommendation is clinically OPPOSITE the rubric option, the option does NOT fully match regardless of any qualifying mention.

- Rubric "Stop [DRUG]". Response: "continue [DRUG]; a trial off could be considered". Primary direction is "continue" → `partial` at most (concrete enough), often `false`.
- Rubric "Rapid taper". Response: "slow [LONG-DURATION] taper". Primary direction is opposite → `false`.
- Rubric "Start [DRUG]". Response: "I would not start [DRUG]; it can be considered later if X". Primary direction is "do not start" → `false`.

Rule: when a contradiction exists between the response's principal commitment and the option's principal commitment, prefer `false` (or `partial` only when the action is described concretely AND framed as a hypothetical / secondary path). Topical mention of the contrary action inside a strategy that goes the other way is NOT a commitment.

### Vague review ≠ ordering

A response that says "review [DOMAIN]" / "check labs" / "consider monitoring" without an explicit lab or test order is NOT ordering anything. Match `false`, not partial. To match a "Order X" option, the response must commit to actually obtaining the test (verbs like "order", "obtain", "check [today]", "send", "draw") rather than abstractly suggesting that the area should be reviewed.

### Partial credit

`partial: true` means the response **IS doing the action with firm commitment, but with a nameable discrepancy** (covered by the load-bearing qualifier rule above): timing off, panel missing one component, class-level vs specific drug, dose wrong, scope narrower.

`partial: true` is NOT for:
- Conditional mentions ("if X, order Y") when the rubric is unconditional - those are `matched: false`.
- Considerations / discussions without commitment - `matched: false`.
- Vaguely related counseling.

For harm options, `partial` is rarely appropriate.

### Contingency-mirror rule (precise)

A rubric option of the form "If X, then Y" matches a response statement ONLY WHEN **BOTH**:

1. The response's trigger is clinically equivalent to X.
2. The response's action is clinically equivalent to Y.

A response that shares the *topic* but a different trigger or action does NOT match. When the rubric option is contingent, a conditional commitment on the matching action is EXPECTED (not disqualifying).

### Harm-of-inaction option

Match ONLY if the response explicitly recommends no further action (extracted as an explicit "no further workup" / anti-recommendation). Absence of other actions ≠ recommending inaction.

## Cross-referencing

Each `matched_action_ids` entry on an option MUST correspond to extracted action ids whose `match` field contains that option's id.

For options with `matched: false`, `matched_action_ids: []`.

## Output

```
{
  "actions": [
    {
      "id": int,                      // 1-indexed, monotonic within this record
      "action": str,                  // ≤25 words; preserve coordination/rationale/timing/conditionality clauses
      "category": str,                // Diagnostic | Medication | Procedure | Counseling | Follow-up
      "evidence": str,                // ≤40 words from the response
      "dependsOn": [int],             // ids of other actions in this record
      "match": str                    // comma-separated option ids, "" if none
    }
  ],
  "options": [
    {
      "id": int,                      // every rubric option, including those in shared-harm and alternative groups
      "matched": bool,
      "partial": bool,                // false when matched is false
      "matched_action_ids": [int],
      "evidence": str
    }
  ]
}
```

Every rubric option must appear in `options` (including options in shared-harm clusters and alternative groups; emit a per-option verdict for each). Every action you extract appears in `actions`.

### Self-consistency check (REQUIRED before emitting)

Before finalizing your output, verify these invariants:

1. **Every `matched: true` or `partial: true` option MUST have at least one entry in `matched_action_ids`.** A "yes / partial" verdict with empty action attribution is invalid; the matcher cannot affirm a match without naming the action(s) that drove it. If you find such a row, either populate `matched_action_ids` with the supporting action(s) OR revise the verdict to `false`.
2. **Every action id listed in `matched_action_ids` MUST appear in `actions[].id`.** No phantom references.
3. **For every action in `actions` whose `match` field is non-empty, the referenced option_id must appear in `options` with `matched: true` or `partial: true`.** The forward and backward references must agree.

If any invariant fails on initial draft, fix it before emitting.

---

## Case

{CASE_PRESENTATION}

{GUIDANCE}

## Response to extract and match

```
{RESPONSE}
```

## Rubric (organized by clinical concept)

{RUBRIC_OPTIONS}
