"""
loader.py
=========
Loads and validates the golden dataset from a JSON file.

Golden records define the expected behaviour for a given prompt.
The RegressionRunner compares actual LLM responses against these records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoldenRecord:
    """
    One ground-truth test case.

    Fields
    ------
    test_id           : unique identifier, e.g. "GEN-001"
    category          : factual | code | structured | safety | technical
    prompt            : the input sent to the LLM
    expected_answer   : reference answer for similarity scoring
    expected_keywords : tokens that must appear in the response
    expected_format   : "text" | "json" | "list"
    json_schema       : optional dict of {key: {type, required}} for JSON checks
    is_safety_test    : if True, forbidden_content is enforced
    forbidden_content : phrases that must NOT appear in the response
    min_similarity    : minimum blended similarity score to pass
    min_length        : minimum character length (0 = no check)
    max_length        : maximum character length (0 = no check)
    """
    test_id:           str
    category:          str
    prompt:            str
    expected_answer:   str
    expected_keywords: list[str]       = field(default_factory=list)
    expected_format:   str             = "text"
    json_schema:       dict | None     = None
    is_safety_test:    bool            = False
    forbidden_content: list[str]       = field(default_factory=list)
    min_similarity:    float           = 0.3
    min_length:        int             = 0
    max_length:        int             = 0


class GoldenLoader:
    """Loads a golden dataset JSON file into a list of GoldenRecord objects."""

    def load(self, path: str | Path) -> list[GoldenRecord]:
        """
        Load and validate golden records from a JSON file.

        Raises FileNotFoundError if the file does not exist.
        Raises ValueError if a required field is missing from any record.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Golden dataset not found: {path}")

        raw = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(raw, list):
            raise ValueError("Golden dataset must be a JSON array.")

        records = []
        for i, item in enumerate(raw):
            self._validate(item, index=i)
            records.append(GoldenRecord(
                test_id           = item["test_id"],
                category          = item.get("category", "general"),
                prompt            = item["prompt"],
                expected_answer   = item["expected_answer"],
                expected_keywords = item.get("expected_keywords", []),
                expected_format   = item.get("expected_format", "text"),
                json_schema       = item.get("json_schema"),
                is_safety_test    = item.get("is_safety_test", False),
                forbidden_content = item.get("forbidden_content", []),
                min_similarity    = float(item.get("min_similarity", 0.3)),
                min_length        = int(item.get("min_length", 0)),
                max_length        = int(item.get("max_length", 0)),
            ))
        return records

    @staticmethod
    def _validate(item: dict, index: int) -> None:
        required = ("test_id", "prompt", "expected_answer")
        for field_name in required:
            if field_name not in item:
                raise ValueError(
                    f"Record at index {index} is missing required field '{field_name}'."
                )
        if not item["test_id"].strip():
            raise ValueError(f"Record at index {index} has an empty 'test_id'.")
