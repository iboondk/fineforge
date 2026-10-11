"""Pure-Python answer metrics: exact match, token F1 and ROUGE-L.

Stdlib only, so the benchmark scoring can be unit-tested without torch.
"""

from __future__ import annotations

import string
from collections import Counter

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLES = frozenset({"a", "an", "the"})


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, drop articles, and collapse whitespace."""
    text = s.lower().translate(_PUNCT_TABLE)
    kept = [tok for tok in text.split() if tok not in _ARTICLES]
    return " ".join(kept)


def tokens(s: str) -> list[str]:
    """Normalized whitespace tokens of an answer."""
    return normalize_answer(s).split()


def exact_match(pred: str, ref: str) -> float:
    """1.0 when the normalized strings are identical, otherwise 0.0."""
    return 1.0 if normalize_answer(pred) == normalize_answer(ref) else 0.0


def token_f1(pred: str, ref: str) -> float:
    """SQuAD-style bag-of-tokens F1 over normalized tokens.

    Both sides empty gives 1.0. Exactly one side empty gives 0.0.
    """
    pred_toks = tokens(pred)
    ref_toks = tokens(ref)
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    common = sum((Counter(pred_toks) & Counter(ref_toks)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_toks)
    recall = common / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence, using two rolling rows."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F-measure (beta = 1) over normalized tokens, based on the LCS.

    Any zero-length side gives 0.0.
    """
    pred_toks = tokens(pred)
    ref_toks = tokens(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    lcs = _lcs_length(pred_toks, ref_toks)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_toks)
    recall = lcs / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def aggregate(preds: list[str], refs: list[str]) -> dict:
    """Mean of each metric over paired predictions and references, plus the count n.

    Raises ValueError on mismatched lengths or on an empty input, because a mean
    over zero samples would be an invented number.
    """
    if len(preds) != len(refs):
        raise ValueError(f"got {len(preds)} predictions but {len(refs)} references")
    n = len(preds)
    if n == 0:
        raise ValueError("cannot aggregate metrics over zero samples")
    return {
        "exact_match": sum(exact_match(p, r) for p, r in zip(preds, refs)) / n,
        "token_f1": sum(token_f1(p, r) for p, r in zip(preds, refs)) / n,
        "rouge_l": sum(rouge_l(p, r) for p, r in zip(preds, refs)) / n,
        "n": n,
    }
