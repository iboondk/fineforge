"""Tests for scripts/make_sample_data.py and scripts/build_notebook.py.

Run from the project root with the trainer venv:

    trainer/.venv/bin/python -m pytest -q scripts/test_scripts.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import build_notebook  # noqa: E402
import make_sample_data  # noqa: E402

SAMPLE = ROOT / "data" / "sample" / "sample.jsonl"
NOTEBOOK = ROOT / "notebooks" / "fineforge_colab.ipynb"


def _normalized_canonical(obj: dict) -> str:
    if "messages" in obj:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in obj["messages"])
    else:
        text = (
            f"instruction: {obj['instruction']}\n"
            f"input: {obj.get('input', '')}\n"
            f"output: {obj['output']}"
        )
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def _check_shape(obj: dict) -> None:
    """Assert one record matches the dataset contract in docs/dataset-format.md."""
    assert obj.get("source") == "synthetic-fineforge-sample"
    if "messages" in obj:
        msgs = obj["messages"]
        assert isinstance(msgs, list) and len(msgs) >= 2
        for i, m in enumerate(msgs):
            assert set(m) >= {"role", "content"}
            assert m["role"] in {"system", "user", "assistant"}
            assert isinstance(m["content"], str) and m["content"].strip()
            if m["role"] == "system":
                assert i == 0
        assert any(m["role"] == "user" for m in msgs)
        assert msgs[-1]["role"] == "assistant"
    else:
        assert isinstance(obj["instruction"], str) and obj["instruction"].strip()
        assert isinstance(obj["output"], str) and obj["output"].strip()
        assert isinstance(obj.get("input", ""), str)


def test_sample_generator_is_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    assert make_sample_data.main(["--out", str(a), "--n", "200", "--seed", "0"]) == 0
    assert make_sample_data.main(["--out", str(b), "--n", "200", "--seed", "0"]) == 0
    assert a.read_bytes() == b.read_bytes()


def test_sample_generator_seed_changes_output(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    make_sample_data.main(["--out", str(a), "--seed", "0"])
    make_sample_data.main(["--out", str(b), "--seed", "1"])
    assert a.read_bytes() != b.read_bytes()


def test_sample_lines_parse_and_have_right_shape(tmp_path: Path) -> None:
    out = tmp_path / "s.jsonl"
    assert make_sample_data.main(["--out", str(out), "--n", "200"]) == 0
    raw = out.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    lines = raw.decode("utf-8").split("\n")[:-1]
    assert len(lines) == 200
    shapes = set()
    for line in lines:
        obj = json.loads(line)
        _check_shape(obj)
        shapes.add("chat" if "messages" in obj else "instruction")
        # sort_keys=True means keys come out in sorted order.
        assert list(json.loads(line).keys()) == sorted(obj.keys())
    assert shapes == {"chat", "instruction"}


def test_sample_contains_exact_duplicates(tmp_path: Path) -> None:
    out = tmp_path / "s.jsonl"
    make_sample_data.main(["--out", str(out), "--n", "200"])
    objs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    keys = [_normalized_canonical(o) for o in objs]
    dupes = len(keys) - len(set(keys))
    assert dupes > 0
    assert dupes / len(keys) <= 0.10


def test_committed_sample_matches_defaults(tmp_path: Path) -> None:
    assert SAMPLE.exists(), "data/sample/sample.jsonl must be generated and committed"
    out = tmp_path / "default.jsonl"
    make_sample_data.main(["--out", str(out)])
    assert out.read_bytes() == SAMPLE.read_bytes()


def test_notebook_build_structure() -> None:
    nb = build_notebook.build()
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] == 5
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["colab"] == {"provenance": []}
    assert nb["metadata"]["kernelspec"]["name"] == "python3"

    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    for cell in nb["cells"]:
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell["id"])
        assert cell["cell_type"] in {"markdown", "code"}
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_notebook_contains_pipeline_steps() -> None:
    nb = build_notebook.build()
    source = "".join("".join(c["source"]) for c in nb["cells"])
    required = [
        "fineforge validate",
        "fineforge dedupe",
        "fineforge shard",
        "fineforge_train.train",
        "fineforge_train.bench.run",
        "fineforge_train.bench.report",
        "fineforge_train.merge",
        "gguf convert",
        "gguf quantize",
    ]
    for needle in required:
        assert needle in source, f"missing pipeline step: {needle}"


def test_notebook_has_no_tokens_or_numbers_from_runs() -> None:
    nb = build_notebook.build()
    source = "".join("".join(c["source"]) for c in nb["cells"])
    assert not re.search(r"hf_[A-Za-z0-9]{10,}", source)
    assert "REPO_URL" in source
    assert "https://github.com/iboondk/fineforge" in source
    assert "userdata" in source
    assert "26%" not in source


def test_committed_notebook_matches_builder() -> None:
    assert NOTEBOOK.exists(), "notebooks/fineforge_colab.ipynb must be generated and committed"
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb == build_notebook.build()


def test_build_notebook_main_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "nb.ipynb"
    assert build_notebook.main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text)["nbformat"] == 4
