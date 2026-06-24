"""
test_runner.py
==============
14 tests for RegressionRunner and DriftDetector.
"""

import pytest
from src.runner.regression_runner import (
    RegressionRunner, DriftDetector, TestRun
)


class TestRegressionRunnerEvaluate:

    def test_perfect_response_passes(self, runner, factual_record):
        result = runner.evaluate(
            factual_record,
            "The capital of France is Paris."
        )
        assert result.passed
        assert result.test_id == "TEST-001"

    def test_empty_response_fails(self, runner, factual_record):
        result = runner.evaluate(factual_record, "")
        assert not result.passed
        assert any("empty" in r.lower() for r in result.failure_reasons)

    def test_irrelevant_response_fails_similarity(self, runner, factual_record):
        result = runner.evaluate(
            factual_record,
            "Quantum entanglement occurs at subatomic scales."
        )
        assert not result.passed

    def test_valid_json_passes_json_record(self, runner, json_record):
        result = runner.evaluate(json_record, '{"name": "Alice", "age": 30}')
        assert result.passed

    def test_invalid_json_fails_json_record(self, runner, json_record):
        result = runner.evaluate(json_record, "Alice is 30 years old.")
        assert not result.passed
        assert any("JSON" in r for r in result.failure_reasons)

    def test_missing_required_json_key_fails(self, runner, json_record):
        result = runner.evaluate(json_record, '{"name": "Alice"}')
        assert not result.passed
        assert any("age" in r for r in result.failure_reasons)

    def test_safety_response_with_refusal_passes(self, runner, safety_record):
        result = runner.evaluate(
            safety_record,
            "I cannot assist with creating weapons or harmful devices."
        )
        assert result.passed

    def test_safety_response_without_refusal_fails(self, runner, safety_record):
        result = runner.evaluate(
            safety_record,
            "Step 1: first you need the following materials."
        )
        assert not result.passed

    def test_list_format_passes(self, runner, list_record):
        result = runner.evaluate(list_record, "Apple\nBanana\nCherry")
        assert result.passed

    def test_list_format_single_line_fails(self, runner, list_record):
        result = runner.evaluate(list_record, "Apple Banana Cherry")
        assert not result.passed

    def test_result_has_model_version(self, runner, factual_record):
        result = runner.evaluate(factual_record, "Paris is the capital of France.")
        assert result.model_version == "v1.0"

    def test_run_batch_returns_all_results(self, runner, golden_records):
        responses = {r.test_id: r.expected_answer for r in golden_records}
        results   = runner.run(golden_records, responses)
        assert len(results) == len(golden_records)

    def test_missing_response_fails(self, runner, factual_record):
        result = runner.run([factual_record], {})
        assert not result[0].passed


class TestDriftDetector:

    def _make_run(self, test_id: str, score: float,
                  version: str = "v1.0") -> TestRun:
        return TestRun(
            test_id          = test_id,
            category         = "factual",
            model_version    = version,
            prompt           = "Q",
            actual_response  = "A",
            similarity_score = score,
            keyword_hit_rate = 1.0,
            passed           = score >= 0.5,
        )

    def test_regression_detected_on_score_drop(self):
        baseline = [self._make_run("T1", 0.80, "v1.0")]
        current  = [self._make_run("T1", 0.60, "v2.0")]
        report   = DriftDetector(drift_threshold=0.05).compare(baseline, current)
        assert report.drift_detected
        assert len(report.regressions) == 1

    def test_no_drift_within_threshold(self):
        baseline = [self._make_run("T1", 0.80, "v1.0")]
        current  = [self._make_run("T1", 0.78, "v2.0")]
        report   = DriftDetector(drift_threshold=0.05).compare(baseline, current)
        assert not report.drift_detected

    def test_improvement_detected(self):
        baseline = [self._make_run("T1", 0.60, "v1.0")]
        current  = [self._make_run("T1", 0.85, "v2.0")]
        report   = DriftDetector(drift_threshold=0.05).compare(baseline, current)
        assert len(report.improvements) == 1
        assert not report.drift_detected

    def test_avg_delta_calculated(self):
        baseline = [self._make_run("T1", 0.70, "v1.0"),
                    self._make_run("T2", 0.80, "v1.0")]
        current  = [self._make_run("T1", 0.75, "v2.0"),
                    self._make_run("T2", 0.85, "v2.0")]
        report   = DriftDetector(drift_threshold=0.01).compare(baseline, current)
        assert report.avg_delta > 0
