# Interview Notes — LLM Regression Test Suite

## What I Built

A golden-dataset regression testing framework for LLM responses. You define expected
outputs in a JSON file. The runner compares actual responses against those expectations
using similarity scoring, keyword coverage, format validation, and safety checks.
The DriftDetector compares two run sets to surface regressions between model versions.
Reports are exported to CSV and JSON.

Built to demonstrate how to bring software regression testing discipline — golden
masters, version comparison, automated CI gates — to LLM quality management.

## How I Would Explain It in an Interview

> "In traditional software testing, a regression suite runs the same inputs through
> different software versions and flags when outputs change unexpectedly. LLMs need
> exactly the same thing — but the comparison can't be exact string matching because
> responses are never identical.
>
> I built a framework that solves this with a blended similarity score: 50% Jaccard
> token overlap and 50% token F1 (the same measure used in SQuAD). You set a minimum
> threshold per golden record. If a response drops below it, the test fails.
>
> The DriftDetector layer compares two complete run sets — baseline vs current — and
> produces a report showing which tests regressed, which improved, and the average
> score delta across the whole dataset. This is the CI gate: if drift_detected is
> True in the report, the deployment is blocked."

## What QA Problem It Solves

1. **Silent LLM regression** — model update changes response quality without any
   automated test catching it
2. **Prompt change regression** — modifying a system prompt breaks downstream
   response format or content
3. **Safety drift** — fine-tuned model loses ability to refuse harmful requests
4. **JSON schema breakage** — structured output endpoint starts returning prose
5. **Keyword loss** — required domain terms disappear from responses over time

## Key Design Decisions Worth Discussing

**Why blended_score instead of just Jaccard?**
Jaccard is symmetric and penalises verbosity equally on both sides. Token F1 is
asymmetric — it measures recall of the reference in the prediction. Blending both
gives a more balanced signal for responses that are correct but worded differently.

**Why keep similarity and safety in separate modules?**
Different teams own these concerns. The scoring module can be updated without
touching the safety checks, and vice versa. Both are independently unit-testable.

**Why JSON for the golden dataset?**
JSON is version-controllable, human-readable, and diffable in pull requests.
When a golden answer is updated, the PR diff shows exactly what changed.

**Why CSV + JSON output?**
CSV is importable into Google Sheets / Excel for stakeholder review. JSON is
machine-readable for dashboards, Slack bots, or further pipeline automation.

## What I Would Add Next

1. **Embedding-based similarity** — swap blended_score for cosine similarity using
   sentence-transformers so the scorer handles paraphrased answers correctly
2. **LLM-as-judge** — use a secondary cheap LLM call to grade correctness rather
   than lexical overlap, for open-ended factual questions
3. **GitHub Actions comment** — post the drift report as a PR comment automatically
4. **Historical trend chart** — track avg_similarity over time and plot a trend line
   to catch gradual degradation before it becomes a regression
5. **Multi-model comparison** — run the same golden dataset against GPT-4o, Claude,
   and Gemini and compare scores side by side
