"""
report_generator.py
===================
Generates CSV and JSON reports from a list of TestRun objects.
Also produces a console-friendly summary dict.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.runner.regression_runner import TestRun


class ReportGenerator:
    """
    Turns a list of TestRun objects into CSV and JSON reports.

    Usage:
        reporter = ReportGenerator(results, model_version="v2.0")
        reporter.save_csv("sample_reports/run_v2.csv")
        reporter.save_json("sample_reports/run_v2.json")
        print(reporter.summary())
    """

    CSV_FIELDS = [
        "test_id", "category", "model_version", "passed",
        "similarity_score", "keyword_hit_rate",
        "failure_reasons", "timestamp",
    ]

    def __init__(self, results: list["TestRun"]):
        self._results = results

    # ── Summary ───────────────────────────────────────────────

    def summary(self) -> dict:
        total   = len(self._results)
        passed  = sum(1 for r in self._results if r.passed)
        failed  = total - passed
        avg_sim = (
            sum(r.similarity_score for r in self._results) / total
            if total else 0.0
        )
        avg_kw  = (
            sum(r.keyword_hit_rate for r in self._results) / total
            if total else 0.0
        )

        by_category: dict[str, dict] = {}
        for r in self._results:
            cat = r.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "passed": 0}
            by_category[cat]["total"]  += 1
            by_category[cat]["passed"] += 1 if r.passed else 0

        return {
            "generated_at":      datetime.now(timezone.utc).isoformat(),
            "model_version":     self._results[0].model_version if self._results else "unknown",
            "total":             total,
            "passed":            passed,
            "failed":            failed,
            "pass_rate":         round(passed / total, 4) if total else 0.0,
            "avg_similarity":    round(avg_sim, 4),
            "avg_keyword_rate":  round(avg_kw, 4),
            "by_category":       by_category,
        }

    # ── CSV ───────────────────────────────────────────────────

    def save_csv(self, path: str = "sample_reports/report.csv") -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for r in self._results:
                writer.writerow({
                    "test_id":          r.test_id,
                    "category":         r.category,
                    "model_version":    r.model_version,
                    "passed":           r.passed,
                    "similarity_score": r.similarity_score,
                    "keyword_hit_rate": r.keyword_hit_rate,
                    "failure_reasons":  "; ".join(r.failure_reasons),
                    "timestamp":        r.timestamp,
                })
        return output

    # ── JSON ──────────────────────────────────────────────────

    def save_json(self, path: str = "sample_reports/report.json") -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "summary": self.summary(),
            "results": [
                {
                    "test_id":          r.test_id,
                    "category":         r.category,
                    "model_version":    r.model_version,
                    "passed":           r.passed,
                    "similarity_score": r.similarity_score,
                    "keyword_hit_rate": r.keyword_hit_rate,
                    "failure_reasons":  r.failure_reasons,
                    "prompt":           r.prompt,
                    "actual_response":  r.actual_response,
                    "timestamp":        r.timestamp,
                }
                for r in self._results
            ],
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output
