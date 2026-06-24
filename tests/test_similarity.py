"""
test_similarity.py
==================
12 tests for similarity scoring functions.
"""

import pytest
from src.scoring.similarity import (
    jaccard_similarity, token_f1, blended_score,
    keyword_hit_rate, missing_keywords
)


class TestJaccard:

    def test_identical_texts_score_one(self):
        assert jaccard_similarity("python is great", "python is great") == 1.0

    def test_completely_different_texts_score_zero(self):
        score = jaccard_similarity("python programming", "tokyo weather sunny")
        assert score == 0.0

    def test_partial_overlap(self):
        score = jaccard_similarity("python is a language", "python is used widely")
        assert 0.0 < score < 1.0

    def test_both_empty_returns_one(self):
        assert jaccard_similarity("", "") == 1.0

    def test_one_empty_returns_zero(self):
        assert jaccard_similarity("python", "") == 0.0

    def test_case_insensitive(self):
        s1 = jaccard_similarity("PYTHON", "python")
        assert s1 == 1.0


class TestTokenF1:

    def test_identical_texts_score_one(self):
        assert token_f1("the answer is paris", "the answer is paris") == 1.0

    def test_no_overlap_returns_zero(self):
        assert token_f1("cat sat mat", "dog ran fast") == 0.0

    def test_partial_overlap(self):
        score = token_f1("paris is a city in france", "paris is the capital of france")
        assert 0.0 < score < 1.0

    def test_both_empty_returns_one(self):
        assert token_f1("", "") == 1.0


class TestBlendedScore:

    def test_perfect_match_returns_one(self):
        assert blended_score("paris is the capital", "paris is the capital") == 1.0

    def test_no_match_returns_zero(self):
        assert blended_score("cats meow softly", "rockets orbit moon") == 0.0

    def test_result_in_range(self):
        score = blended_score("the capital is paris in france", "paris is the capital")
        assert 0.0 <= score <= 1.0


class TestKeywordCoverage:

    def test_all_keywords_found(self):
        rate = keyword_hit_rate("Paris is the capital of France.", ["paris", "france"])
        assert rate == 1.0

    def test_no_keywords_found(self):
        rate = keyword_hit_rate("Cats are mammals.", ["python", "django"])
        assert rate == 0.0

    def test_partial_keywords(self):
        rate = keyword_hit_rate("Python is great.", ["python", "django"])
        assert rate == 0.5

    def test_empty_keyword_list_returns_one(self):
        assert keyword_hit_rate("any response", []) == 1.0

    def test_missing_keywords_list(self):
        missed = missing_keywords("Python is great.", ["python", "django", "flask"])
        assert "django" in missed
        assert "flask"  in missed
        assert "python" not in missed
