"""
test_loader.py
==============
10 tests for GoldenLoader: file loading, validation, error handling.
"""

import json
import pytest
from src.golden.loader import GoldenLoader, GoldenRecord


class TestGoldenLoader:

    def test_loads_real_dataset(self, real_dataset_path):
        records = GoldenLoader().load(real_dataset_path)
        assert len(records) == 12

    def test_all_records_are_golden_record_instances(self, real_dataset_path):
        records = GoldenLoader().load(real_dataset_path)
        assert all(isinstance(r, GoldenRecord) for r in records)

    def test_record_fields_populated(self, real_dataset_path):
        records = GoldenLoader().load(real_dataset_path)
        r = records[0]
        assert r.test_id
        assert r.prompt
        assert r.expected_answer

    def test_safety_records_flagged(self, real_dataset_path):
        records  = GoldenLoader().load(real_dataset_path)
        safety   = [r for r in records if r.is_safety_test]
        assert len(safety) == 2

    def test_json_format_records_have_schema(self, real_dataset_path):
        records = GoldenLoader().load(real_dataset_path)
        json_recs = [r for r in records if r.expected_format == "json"]
        assert all(r.json_schema is not None for r in json_recs)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GoldenLoader().load(tmp_path / "nonexistent.json")

    def test_missing_test_id_raises(self, tmp_path):
        bad_data = [{"prompt": "Q", "expected_answer": "A"}]
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(bad_data))
        with pytest.raises(ValueError, match="test_id"):
            GoldenLoader().load(f)

    def test_missing_prompt_raises(self, tmp_path):
        bad_data = [{"test_id": "T1", "expected_answer": "A"}]
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(bad_data))
        with pytest.raises(ValueError, match="prompt"):
            GoldenLoader().load(f)

    def test_non_array_json_raises(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text('{"not": "an array"}')
        with pytest.raises(ValueError, match="array"):
            GoldenLoader().load(f)

    def test_empty_test_id_raises(self, tmp_path):
        bad_data = [{"test_id": "  ", "prompt": "Q", "expected_answer": "A"}]
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(bad_data))
        with pytest.raises(ValueError, match="empty"):
            GoldenLoader().load(f)
