"""
regression_runner.py
====================
RegressionRunner  — runs a golden dataset against a set of model responses
DriftDetector     — compares two sets of TestRun results to surface regressions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.golden.loader import GoldenRecord
from src.scoring.similarity import (
    blended_score, keyword_hit_rate, missing_keywords
)
from src.scoring.safety import SafetyChecker


# ─────────────────────────────────────────────────────────────
# TestRun — result for one record against one model version
# ─────────────────────────────────────────────────────────────

@dataclass
class TestRun:
    test_id:          str
    category:         str
    model_version:    str
    prompt:           str
    actual_response:  str
    similarity_score: float
    keyword_hit_rate: float
    passed:           bool
    failure_reasons:  list[str]      = field(default_factory=list)
    timestamp:        str            = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"TestRun({self.test_id}, [{status}], "
            f"sim={self.similarity_score:.3f}, kw={self.keyword_hit_rate:.3f})"
        )


# ─────────────────────────────────────────────────────────────
# RegressionRunner
# ─────────────────────────────────────────────────────────────

class RegressionRunner:
    """
    Evaluates LLM responses against a golden dataset.

    Usage:
        runner  = RegressionRunner(model_version="v2.1")
        results = runner.run(records, responses)
        # results is a list of TestRun objects
    """

    def __init__(self, model_version: str = "v1.0"):
        self.model_version = model_version
        self._safety       = SafetyChecker()

    def evaluate(self, record: GoldenRecord, response: str) -> TestRun:
        """Evaluate one response against one golden record."""
        failures: list[str] = []

        # ── 1. Empty check ─────────────────────────────────
        if not response or not response.strip():
            return TestRun(
                test_id          = record.test_id,
                category         = record.category,
                model_version    = self.model_version,
                prompt           = record.prompt,
                actual_response  = response,
                similarity_score = 0.0,
                keyword_hit_rate = 0.0,
                passed           = False,
                failure_reasons  = ["Response is empty"],
            )

        # ── 2. Similarity ───────────────────────────────────
        sim = blended_score(response, record.expected_answer)
        if sim < record.min_similarity:
            failures.append(
                f"Similarity {sim:.3f} below threshold {record.min_similarity}"
            )

        # ── 3. Keywords ─────────────────────────────────────
        kw_rate = keyword_hit_rate(response, record.expected_keywords)
        missed  = missing_keywords(response, record.expected_keywords)
        if kw_rate < 0.6 and record.expected_keywords:
            failures.append(f"Missing keywords: {missed}")

        # ── 4. Format ────────────────────────────────────────
        if record.expected_format == "json":
            json_failures = self._check_json(response, record.json_schema)
            failures.extend(json_failures)

        elif record.expected_format == "list":
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            if len(lines) < 2:
                failures.append(
                    f"Expected list with ≥2 items, got {len(lines)} line(s)"
                )

        # ── 5. Length bounds ─────────────────────────────────
        if record.min_length > 0 and len(response) < record.min_length:
            failures.append(
                f"Response length {len(response)} below min_length {record.min_length}"
            )
        if record.max_length > 0 and len(response) > record.max_length:
            failures.append(
                f"Response length {len(response)} exceeds max_length {record.max_length}"
            )

        # ── 6. Safety ────────────────────────────────────────
        safety_results = self._safety.run_all(
            response,
            record.forbidden_content,
            record.is_safety_test,
        )
        for sr in safety_results:
            if not sr.passed:
                failures.append(f"[{sr.check_name}] {sr.message}")

        return TestRun(
            test_id          = record.test_id,
            category         = record.category,
            model_version    = self.model_version,
            prompt           = record.prompt,
            actual_response  = response,
            similarity_score = sim,
            keyword_hit_rate = kw_rate,
            passed           = len(failures) == 0,
            failure_reasons  = failures,
        )

    def run(self, records: list[GoldenRecord],
            responses: dict[str, str]) -> list[TestRun]:
        """
        Run all records.

        Args:
            records   : list of GoldenRecord objects
            responses : dict mapping test_id → actual LLM response string
        """
        return [
            self.evaluate(record, responses.get(record.test_id, ""))
            for record in records
        ]

    @staticmethod
    def _check_json(response: str,
                    schema: dict | None) -> list[str]:
        """Validate JSON format and optional schema."""
        failures = []
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            return [f"Expected JSON but got parse error: {exc}"]

        if schema and isinstance(parsed, dict):
            for key, spec in schema.items():
                if spec.get("required") and key not in parsed:
                    failures.append(f"JSON schema: required key '{key}' missing")
                elif key in parsed:
                    expected_type = spec.get("type")
                    type_map = {
                        "string":  str,
                        "number":  (int, float),
                        "integer": int,
                        "boolean": bool,
                        "array":   list,
                        "object":  dict,
                    }
                    if expected_type and expected_type in type_map:
                        if not isinstance(parsed[key], type_map[expected_type]):
                            failures.append(
                                f"JSON schema: '{key}' expected {expected_type}, "
                                f"got {type(parsed[key]).__name__}"
                            )
        return failures


# ─────────────────────────────────────────────────────────────
# DriftDetector
# ─────────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    baseline_version:    str
    current_version:     str
    total_compared:      int
    regressions:         list[dict]
    improvements:        list[dict]
    avg_score_baseline:  float
    avg_score_current:   float
    avg_delta:           float
    drift_detected:      bool

    def __repr__(self) -> str:
        return (
            f"DriftReport({self.baseline_version} → {self.current_version}, "
            f"regressions={len(self.regressions)}, "
            f"avg_delta={self.avg_delta:+.4f})"
        )


class DriftDetector:
    """
    Compare two sets of TestRun results (baseline vs current)
    to surface quality regressions and improvements.
    """

    def __init__(self, drift_threshold: float = 0.05):
        self.drift_threshold = drift_threshold

    def compare(self,
                baseline: list[TestRun],
                current:  list[TestRun]) -> DriftReport:
        """
        Returns a DriftReport identifying regressions and improvements.
        A regression is a test whose similarity score dropped by more than
        drift_threshold between baseline and current.
        """
        base_map = {r.test_id: r for r in baseline}
        curr_map = {r.test_id: r for r in current}
        common   = set(base_map) & set(curr_map)

        regressions  = []
        improvements = []

        for tid in sorted(common):
            b     = base_map[tid]
            c     = curr_map[tid]
            delta = c.similarity_score - b.similarity_score

            if abs(delta) >= self.drift_threshold:
                entry = {
                    "test_id":        tid,
                    "category":       b.category,
                    "baseline_score": b.similarity_score,
                    "current_score":  c.similarity_score,
                    "delta":          round(delta, 4),
                }
                if delta < 0:
                    regressions.append(entry)
                else:
                    improvements.append(entry)

        avg_base = (
            sum(r.similarity_score for r in baseline) / len(baseline)
            if baseline else 0.0
        )
        avg_curr = (
            sum(r.similarity_score for r in current) / len(current)
            if current else 0.0
        )

        baseline_version = baseline[0].model_version if baseline else "unknown"
        current_version  = current[0].model_version  if current  else "unknown"

        return DriftReport(
            baseline_version   = baseline_version,
            current_version    = current_version,
            total_compared     = len(common),
            regressions        = regressions,
            improvements       = improvements,
            avg_score_baseline = round(avg_base, 4),
            avg_score_current  = round(avg_curr, 4),
            avg_delta          = round(avg_curr - avg_base, 4),
            drift_detected     = len(regressions) > 0,
        )
