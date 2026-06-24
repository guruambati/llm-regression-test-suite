"""
similarity.py
=============
Semantic similarity scoring using lexical overlap.

Three measures:
  - jaccard_similarity : token set overlap ratio
  - token_f1           : precision/recall F1 over token sets (SQuAD-style)
  - blended_score      : 50/50 blend of Jaccard and F1

All functions are pure — no side effects, no external dependencies.
Values are always in [0.0, 1.0].
"""

from __future__ import annotations

import re


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, punctuation stripped."""
    return set(re.findall(r'\b[a-z0-9]+\b', text.lower()))


def jaccard_similarity(a: str, b: str) -> float:
    """
    Jaccard similarity between the token sets of two strings.
    Returns 1.0 when both strings are empty.
    """
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def token_f1(prediction: str, reference: str) -> float:
    """
    Token-level F1 score between prediction and reference.
    Measures how much of the reference is covered (recall)
    and how precise the prediction is (precision).
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens  = _tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common    = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(ref_tokens)

    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def blended_score(prediction: str, reference: str) -> float:
    """
    50/50 blend of Jaccard and token F1.
    More balanced than either alone:
      - Jaccard penalises verbose responses
      - F1 is more recall-oriented
    Returns a value in [0.0, 1.0].
    """
    j = jaccard_similarity(prediction, reference)
    f = token_f1(prediction, reference)
    return round((j + f) / 2, 4)


def keyword_hit_rate(response: str, keywords: list[str]) -> float:
    """
    Fraction of expected keywords present in the response (case-insensitive).
    Returns 1.0 when the keyword list is empty.
    """
    if not keywords:
        return 1.0
    lower   = response.lower()
    hits    = [kw for kw in keywords if kw.lower() in lower]
    return round(len(hits) / len(keywords), 4)


def missing_keywords(response: str, keywords: list[str]) -> list[str]:
    """Return keywords that are absent from the response."""
    lower = response.lower()
    return [kw for kw in keywords if kw.lower() not in lower]
