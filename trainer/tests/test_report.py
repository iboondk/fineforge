import re

from fineforge_train.bench.report import (
    PLACEHOLDER_HEADLINE,
    main,
    relative_improvement,
    render,
)

# Clearly fake data for rendering tests. These are not benchmark results.
FAKE_RESULTS = {
    "model_id": "test-model",
    "adapter_path": "test/adapter",
    "eval_file": "test/eval.jsonl",
    "eval_sha256": "ab" * 32,
    "n": 4,
    "seed": 7,
    "generation": {"max_new_tokens": 8, "do_sample": False},
    "timestamp_utc": "2000-01-01T00:00:00+00:00",
    "base": {"exact_match": 0.5, "token_f1": 0.5, "rouge_l": 0.0, "n": 4},
    "tuned": {"exact_match": 0.63, "token_f1": 0.63, "rouge_l": 0.25, "n": 4},
}


def test_relative_improvement_values():
    assert relative_improvement(0.5, 0.63) == "+26.0%"
    assert relative_improvement(0.5, 0.25) == "-50.0%"


def test_relative_improvement_base_zero_is_na():
    assert relative_improvement(0.0, 0.3) == "n/a"


def test_placeholder_has_tbd_in_all_nine_metric_cells():
    text = render(None)
    rows = [line for line in text.splitlines() if re.match(r"^\| (exact_match|token_f1|rouge_l) \|", line)]
    assert len(rows) == 3
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")][1:]
        assert cells == ["TBD", "TBD", "TBD"]
    assert text.startswith("# FineForge benchmark")
    assert "| Metric | Base | Tuned | Relative change |" in text
    assert PLACEHOLDER_HEADLINE in text.splitlines()
    assert PLACEHOLDER_HEADLINE == "Headline: relative improvement = TBD (target ≈ 26%; not yet measured)"


def test_placeholder_contains_no_four_decimal_numbers():
    assert not re.search(r"[0-9]+\.[0-9]{4}", render(None))


def test_render_fake_results_shows_values_and_relative_change():
    text = render(FAKE_RESULTS)
    assert "| exact_match | 0.5000 | 0.6300 | +26.0% |" in text
    assert "| rouge_l | 0.0000 | 0.2500 | n/a |" in text
    assert "Headline: relative improvement (token_f1) = +26.0% (measured; target ≈ 26%)" in text
    assert "TBD" not in text
    assert "## Provenance" in text
    assert "- eval_sha256: " + "ab" * 32 in text
    assert "- n: 4" in text
    assert "- seed: 7" in text
    assert "- timestamp: 2000-01-01T00:00:00+00:00" in text


def test_main_without_results_writes_placeholder(tmp_path):
    out = tmp_path / "benchmark.md"
    assert main(["--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == render(None)


def test_main_with_missing_results_file_writes_placeholder(tmp_path):
    out = tmp_path / "benchmark.md"
    assert main(["--results", str(tmp_path / "nope.json"), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == render(None)
