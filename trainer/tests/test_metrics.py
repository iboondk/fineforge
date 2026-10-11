import pytest

from fineforge_train.bench.metrics import (
    aggregate,
    exact_match,
    normalize_answer,
    rouge_l,
    token_f1,
    tokens,
)


def test_normalize_drops_case_punctuation_articles_and_extra_space():
    assert normalize_answer("  The   Cat. ") == "cat"
    assert normalize_answer("A dog, an owl!") == "dog owl"


def test_tokens_follow_normalization():
    assert tokens("The red-blue  sky") == ["redblue", "sky"]


def test_exact_match_after_normalization():
    assert exact_match("The Cat.", "cat") == 1.0
    assert exact_match("cat", "dog") == 0.0


def test_token_f1_partial_overlap_known_value():
    # 2 shared tokens out of 3 predicted and 2 reference tokens:
    # precision 2/3, recall 1, F1 = 2 * (2/3) / (5/3) = 0.8
    assert token_f1("red blue green", "red blue") == pytest.approx(0.8)


def test_token_f1_articles_are_ignored():
    assert token_f1("a b c", "b c") == pytest.approx(1.0)


def test_token_f1_no_overlap_is_zero():
    assert token_f1("x", "y") == 0.0


def test_token_f1_empty_cases():
    assert token_f1("", "") == 1.0
    assert token_f1("the", "") == 1.0  # "the" normalizes to nothing
    assert token_f1("", "cat") == 0.0
    assert token_f1("cat", "") == 0.0


def test_rouge_l_known_value():
    # LCS "x z" has length 2: P = 2/3, R = 2/2 = 1, F = 0.8
    assert rouge_l("x y z", "x z") == pytest.approx(0.8)


def test_rouge_l_empty_cases_are_zero():
    assert rouge_l("", "") == 0.0
    assert rouge_l("", "cat") == 0.0
    assert rouge_l("cat", "") == 0.0


def test_aggregate_means_and_n():
    preds = ["The Cat.", "red blue green"]
    refs = ["cat", "red blue"]
    out = aggregate(preds, refs)
    assert out["n"] == 2
    assert out["exact_match"] == pytest.approx(0.5)
    assert out["token_f1"] == pytest.approx((1.0 + 0.8) / 2)
    assert out["rouge_l"] == pytest.approx((1.0 + 0.8) / 2)


def test_aggregate_rejects_mismatch_and_empty():
    with pytest.raises(ValueError):
        aggregate(["a"], [])
    with pytest.raises(ValueError):
        aggregate([], [])
