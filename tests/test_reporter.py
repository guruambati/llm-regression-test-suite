"""
test_reporter.py
================
10 tests for ReportGenerator: CSV, JSON, and summary output.
"""

import csv
import json
import pytest
from src.runner.regression_runner import RegressionRunner, TestRun
from src.reporter.report_generator import ReportGenerator


@pytest.fixture
def sample_results(runner, golden_records):
    responses = {r.test_id: r.expected_answer for r in golden_records}
    return runner.run(golden_records, responses)

@pytest.fixture
def reporter(sample_results):
    return ReportGenerator(sample_results)


class TestSummary:

    def test_summary_has_required_keys(self, reporter):
        s = reporter.summary()
        for key in ("total", "passed", "failed", "pass_rate",
                    "avg_similarity", "by_category"):
            assert key in s

    def test_total_matches_record_count(self, reporter, golden_records):
        assert reporter.summary()["total"] == len(golden_records)

    def test_pass_rate_in_range(self, reporter):
        rate = reporter.summary()["pass_rate"]
        assert 0.0 <= rate <= 1.0

    def test_by_category_contains_all_categories(self, reporter):
        cats = reporter.summary()["by_category"]
        assert "factual" in cats
        assert "safety"  in cats


class TestCsvReport:

    def test_csv_created(self, reporter, tmp_path):
        path = tmp_path / "report.csv"
        reporter.save_csv(str(path))
        assert path.exists()

    def test_csv_row_count_matches_results(self, reporter, tmp_path, golden_records):
        path = tmp_path / "report.csv"
        reporter.save_csv(str(path))
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(golden_records)

    def test_csv_has_required_columns(self, reporter, tmp_path):
        path = tmp_path / "report.csv"
        reporter.save_csv(str(path))
        with open(path) as f:
            reader = csv.DictReader(f)
            cols   = reader.fieldnames
        for col in ("test_id", "passed", "similarity_score", "failure_reasons"):
            assert col in cols


class TestJsonReport:

    def test_json_created(self, reporter, tmp_path):
        path = tmp_path / "report.json"
        reporter.save_json(str(path))
        assert path.exists()

    def test_json_valid_structure(self, reporter, tmp_path):
        path = tmp_path / "report.json"
        reporter.save_json(str(path))
        data = json.loads(path.read_text())
        assert "summary" in data
        assert "results" in data

    def test_json_results_count(self, reporter, tmp_path, golden_records):
        path = tmp_path / "report.json"
        reporter.save_json(str(path))
        data = json.loads(path.read_text())
        assert len(data["results"]) == len(golden_records)

    def test_creates_parent_directory(self, reporter, tmp_path):
        path = tmp_path / "nested" / "dir" / "report.json"
        reporter.save_json(str(path))
        assert path.exists()
