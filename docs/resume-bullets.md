# Resume Bullets — LLM Regression Test Suite

## Option A — LLM Evaluation Engineer focus

- Built a golden-dataset regression testing framework for LLM responses (Python,
  pytest) with blended similarity scoring (Jaccard + token F1), keyword coverage
  checks, JSON schema validation, and safety regression detection; exports CSV and
  JSON reports with per-category pass rates

## Option B — AI QA Engineer focus

- Designed and implemented a DriftDetector that compares LLM response quality between
  model versions using 12 golden Q&A records across five categories (factual, code,
  structured, safety, technical); surfaces regressions, improvements, and average
  score delta per run

## Option C — AI SDET focus

- Developed a modular LLM regression suite with separated scoring, safety, runner,
  and reporting layers; integrated with GitHub Actions CI to automatically gate
  deployments when similarity scores drop below per-record thresholds

## Notes on Usage

- Strong talking point: "The blended score handles paraphrase variation — a response
  that says 'Paris is France's capital' scores well against 'The capital of France is
  Paris' because the token F1 component captures recall, not just exact overlap."
- When asked about LLM evaluation frameworks: "This implements the core concepts
  from DeepEval's regression testing and RAGAS's answer relevance metric from scratch.
  The next step is to wire in the actual DeepEval library so tests benefit from its
  LLM-as-judge grading."
