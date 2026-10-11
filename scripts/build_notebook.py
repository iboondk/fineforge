#!/usr/bin/env python3
"""Build the reproducible Colab notebook as nbformat 4.5 JSON (stdlib only).

    python scripts/build_notebook.py [--out notebooks/fineforge_colab.ipynb]

``build()`` returns the notebook as a dict. ``main()`` writes it with indent=1 and a
trailing newline. Cell ids are derived from cell position and label, so rebuilding
gives byte-identical output.

Benchmark numbers come only from running the notebook. This builder writes no results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "notebooks" / "fineforge_colab.ipynb"

REPO_URL = "https://github.com/iboondk/fineforge"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp"


def _md(label: str, text: str) -> tuple[str, str, str]:
    return ("markdown", label, text)


def _code(label: str, text: str) -> tuple[str, str, str]:
    return ("code", label, text)


# Each entry is (cell_type, label, source). Order is the notebook order.
CELLS: list[tuple[str, str, str]] = [
    _md(
        "intro",
        """# FineForge: reproducible Colab run

This notebook runs the whole FineForge pipeline on a Colab GPU:
validate, dedupe and shard a JSONL dataset, QLoRA fine-tune a small model, benchmark
the base model against the adapter, merge the adapter, and export a GGUF file.

**Honesty note.** The benchmark numbers come only from running this notebook. The repo
does not ship measured results. `results/benchmark.md` shows `TBD` until you run the
benchmark cells yourself, and the headline is a target, not a result.

The sample data is **synthetic** (templated toy records). Replace it with your own
dataset in the shape described in `docs/dataset-format.md`.""",
    ),
    _md("gpu-header", "## 1. Check the GPU"),
    _code("nvidia-smi", "!nvidia-smi"),
    _md("clone-header", "## 2. Clone the repository"),
    _code(
        "clone",
        f'''REPO_URL = "{REPO_URL}"
# If you re-run this cell, delete the existing `fineforge` folder first.
!git clone $REPO_URL fineforge
%cd fineforge''',
    ),
    _md(
        "rust-header",
        "## 3. Install Rust and build the `fineforge` CLI",
    ),
    _code(
        "rust-install",
        """!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
import os
os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]""",
    ),
    _code(
        "cargo-build",
        "!cargo build --release --manifest-path cli/Cargo.toml",
    ),
    _md("data-header", "## 4. Data: sample or your own"),
    _code(
        "make-sample",
        """# Replace with your own dataset: write JSONL in the docs/dataset-format.md shape and
# point RAW at it. The sample generator writes synthetic toy records only.
RAW = "data/sample/sample.jsonl"
!python scripts/make_sample_data.py --out $RAW""",
    ),
    _md("validate-header", "## 5. Validate the dataset"),
    _code(
        "validate",
        """!mkdir -p data/out
!./cli/target/release/fineforge validate $RAW --report data/out/validate.json""",
    ),
    _md("dedupe-header", "## 6. Dedupe (exact and near-duplicate)"),
    _code(
        "dedupe",
        """!./cli/target/release/fineforge dedupe $RAW -o data/out/dedup.jsonl --near 0.85 --stats data/out/dedupe_stats.json""",
    ),
    _md("shard-header", "## 7. Shard into train and eval splits"),
    _code(
        "shard",
        """!./cli/target/release/fineforge shard data/out/dedup.jsonl --out-dir data/out --shards 2 --eval-ratio 0.1 --seed 42""",
    ),
    _md("install-header", "## 8. Install the trainer"),
    _code("pip-install", "!pip install -e 'trainer[train]'"),
    _md(
        "hf-header",
        """## 9. Optional: Hugging Face login

Only needed for gated models. The token is read from a Colab secret named `HF_TOKEN`,
never typed into the notebook.""",
    ),
    _code(
        "hf-login",
        """try:
    from google.colab import userdata
    from huggingface_hub import login

    login(token=userdata.get("HF_TOKEN"))
    print("Logged in to Hugging Face.")
except Exception as exc:  # no secret set, or not running on Colab
    print("Skipping Hugging Face login:", exc)""",
    ),
    _md(
        "train-header",
        """## 10. Train (QLoRA)

The dry run prints the plan as JSON and trains nothing.""",
    ),
    _code(
        "train-dry-run",
        """%cd trainer
!python -m fineforge_train.train --config configs/qlora.yaml --dry-run""",
    ),
    _code(
        "train",
        "!python -m fineforge_train.train --config configs/qlora.yaml",
    ),
    _md("bench-header", "## 11. Benchmark: base model vs adapter"),
    _code(
        "bench-run",
        f"""!mkdir -p ../results
!python -m fineforge_train.bench.run --base-model {BASE_MODEL} --adapter runs/qlora/adapter --eval-file ../data/out/eval.jsonl --out ../results/results.json --seed 42""",
    ),
    _code(
        "bench-report",
        """!python -m fineforge_train.bench.report --results ../results/results.json --out ../results/benchmark.md
from IPython.display import Markdown, display

with open("../results/benchmark.md", encoding="utf-8") as fh:
    display(Markdown(fh.read()))""",
    ),
    _md("merge-header", "## 12. Merge the adapter into the base model"),
    _code(
        "merge",
        f"!python -m fineforge_train.merge --base {BASE_MODEL} --adapter runs/qlora/adapter --out runs/qlora/merged",
    ),
    _md(
        "gguf-header",
        """## 13. Export GGUF (llama.cpp)

Clones llama.cpp, installs its conversion requirements, and builds `llama-quantize`.
The `fineforge gguf` commands wrap the llama.cpp scripts and binaries. They do not
reimplement them.""",
    ),
    _code(
        "llama-cpp-build",
        f"""%cd ..
!git clone --depth 1 {LLAMA_CPP_URL} llama.cpp
!pip install -r llama.cpp/requirements.txt
!cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
!cmake --build llama.cpp/build --target llama-quantize -j""",
    ),
    _code(
        "gguf-convert-quantize",
        """import os

os.environ["LLAMA_CPP_DIR"] = os.path.abspath("llama.cpp")
QUANT_BIN = os.path.abspath("llama.cpp/build/bin/llama-quantize")
MERGED = "trainer/runs/qlora/merged"

!./cli/target/release/fineforge gguf convert $MERGED --outfile data/out/model-f16.gguf --outtype f16
!./cli/target/release/fineforge gguf quantize data/out/model-f16.gguf data/out/model-Q4_K_M.gguf --type Q4_K_M --bin $QUANT_BIN""",
    ),
    _md(
        "done",
        """## Done

Your numbers are in `results/results.json` and `results/benchmark.md`, both written by
this run. Quote only the values from your own run, and record the seed and eval file
hash that the results JSON contains.""",
    ),
]


def _source_lines(text: str) -> list[str]:
    """Split text into nbformat source lines, each keeping its trailing newline."""
    lines = text.split("\n")
    out = [line + "\n" for line in lines[:-1]]
    if lines[-1] != "":
        out.append(lines[-1])
    return out


def _cell_id(index: int, label: str) -> str:
    digest = hashlib.sha256(f"fineforge-{index}-{label}".encode("utf-8")).hexdigest()
    return f"{label}-{digest[:8]}"


def build() -> dict:
    """Return the notebook as an nbformat 4.5 dict."""
    cells = []
    for index, (cell_type, label, text) in enumerate(CELLS):
        cell: dict = {
            "cell_type": cell_type,
            "id": _cell_id(index, label),
            "metadata": {},
            "source": _source_lines(text),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/build_notebook.py",
        description="Write the FineForge Colab notebook (nbformat 4.5).",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output .ipynb path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"wrote notebook to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
