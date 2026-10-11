# FineForge architecture

An LLM fine-tuning toolkit with two parts:

1. **`fineforge` (Rust CLI)**: validates, dedupes and shards JSONL datasets, and wraps llama.cpp's GGUF convert and quantize steps.
2. **`fineforge_train` (Python)**: LoRA/QLoRA training, adapter merging, and a benchmark that compares the base model with the tuned one. It ships with a Colab notebook that can be reproduced end to end.

The repository includes a Colab notebook that reproduces a run and a benchmark table. The table shows `TBD` (to be measured) until a real run produces numbers.

---

## 1. Architecture

```
raw.jsonl
   │  fineforge validate      (schema + content checks, JSON report)
   │  fineforge dedupe        (exact SHA-256 + optional MinHash near-dup)
   │  fineforge shard         (deterministic shuffle, eval split, N train shards, manifest)
   ▼
data/out/{train-00000.jsonl, ..., eval.jsonl, manifest.json}
   │  python -m fineforge_train.train --config configs/qlora.yaml
   ▼
runs/<name>/adapter/           (PEFT LoRA adapter)
   │  python -m fineforge_train.bench.run   (base vs base+adapter on eval.jsonl)
   │  python -m fineforge_train.bench.report  -> results/benchmark.md
   │  python -m fineforge_train.merge        -> runs/<name>/merged/ (HF weights)
   ▼
fineforge gguf convert   (llama.cpp convert_hf_to_gguf.py) -> model-f16.gguf
fineforge gguf quantize  (llama-quantize)                  -> model-Q4_K_M.gguf
```

The Rust and Python halves share only the **dataset contract** (Section 3) and files on disk. Neither one calls the other's code.

## 2. Tech choices

| Concern | Choice | Why |
|---|---|---|
| CLI | Rust 2021, `clap` 4 (derive), `serde`/`serde_json`, `anyhow`, `sha2` | Fast streaming over large JSONL, and ships as a single binary |
| RNG | Hand-written SplitMix64 inside the crate | Same shuffle on every platform and crate version, with no `rand` version churn |
| Near-dup | MinHash over word 5-gram shingles, 64 seeded hashes, Jaccard estimate | Easy to unit-test, no extra dependencies |
| CLI tests | `assert_cmd`, `predicates`, `tempfile` | Black-box tests of exit codes and output files |
| GGUF | Shell out to llama.cpp (`convert_hf_to_gguf.py`, `llama-quantize`) with `--dry-run` | llama.cpp is not reimplemented. Dry-run makes the wrapper testable without a model |
| Trainer | Python 3.10+, `transformers`, `peft`, `trl` (SFTTrainer), `bitsandbytes` (4-bit QLoRA), `datasets`, `pyyaml` | Standard stack that runs on a Colab T4 |
| Default base model | `Qwen/Qwen2.5-0.5B-Instruct` (overridable) | Small enough for a free Colab T4, with or without QLoRA |
| Python packaging | `pyproject.toml` (setuptools). Core deps: `pyyaml`. Extra `[train]` holds the heavy ML deps. Extra `[dev]` holds `pytest` | Unit tests run without torch or a GPU |
| Heavy imports | `torch`/`transformers`/`peft`/`trl` imported **inside functions only** | `--dry-run` paths and tests work on a laptop without CUDA |
| Metrics | Pure-Python exact match, token F1 and ROUGE-L (LCS) | Deterministic, no `evaluate` download, easy to test |
| Secrets | `.env.example` with `HF_TOKEN=` only, never a real value | Gated models need a token, so it is documented but not committed |

## 3. Dataset contract (JSONL, UTF-8, one object per line)

Two accepted shapes:

- **chat**: `{"messages": [{"role": "system"|"user"|"assistant", "content": "<non-empty str>"}, ...]}`
  - at least 2 messages, at least one `user`, the **last** message is `assistant`, and `system` may only appear at index 0.
- **instruction**: `{"instruction": "<non-empty str>", "input": "<str, optional>", "output": "<non-empty str>"}`

Any extra top-level keys (for example `"id"` or `"source"`) are allowed and kept as they are.

**Canonical text** (used for dedupe in Rust and must stay stable): for chat records, join `"{role}: {content}"` with `\n`. For instruction records, use `"instruction: {instruction}\ninput: {input}\noutput: {output}"` (`input` is `""` if missing). **Normalization** for hashing lowercases the text, then collapses every run of Unicode whitespace to a single space and trims the ends.

## 4. Directory layout

```
fineforge/
├── docs/                         # architecture, ADRs, dataset format, benchmark methodology
├── .env.example                      # HF_TOKEN=  (placeholder only)
├── .gitignore                        # target/, runs/, data/out/, *.gguf, __pycache__/, .venv/
├── cli/                              # Rust crate "fineforge"
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs                   # clap dispatch only
│   │   ├── record.rs                 # Record enum, parse_line, canonical_text, normalize
│   │   ├── validate.rs
│   │   ├── dedupe.rs                 # exact + minhash
│   │   ├── rng.rs                    # SplitMix64 + Fisher–Yates
│   │   ├── shard.rs
│   │   └── gguf.rs                   # command builders + runner
│   └── tests/
│       ├── fixtures/{valid.jsonl, mixed.jsonl, dupes.jsonl}
│       ├── validate_cli.rs
│       ├── dedupe_shard_cli.rs
│       └── gguf_cli.rs
├── trainer/                          # Python package
│   ├── pyproject.toml
│   ├── configs/{lora.yaml, qlora.yaml}
│   ├── fineforge_train/
│   │   ├── __init__.py
│   │   ├── config.py                 # TrainConfig dataclass + load_config
│   │   ├── data.py                   # load_jsonl, to_messages, to_prompt_completion
│   │   ├── train.py                  # CLI, --dry-run
│   │   ├── merge.py                  # merge adapter -> HF dir
│   │   └── bench/
│   │       ├── __init__.py
│   │       ├── metrics.py
│   │       ├── run.py                # base vs tuned generation + scoring
│   │       └── report.py             # results JSON -> markdown table
│   └── tests/{test_config.py, test_data.py, test_train_dryrun.py, test_metrics.py, test_report.py}
├── data/sample/sample.jsonl          # small toy dataset (synthetic, labeled as such)
├── scripts/
│   ├── make_sample_data.py           # writes data/sample/sample.jsonl deterministically
│   └── build_notebook.py             # writes notebooks/fineforge_colab.ipynb (stdlib json)
├── notebooks/fineforge_colab.ipynb   # generated, committed
└── results/benchmark.md              # placeholder table until a real run
```

## 5. Benchmark reporting rules

- `results/benchmark.md` is generated by `bench.report`. With no results file it renders `TBD` in every metric cell, plus this line: `Headline: relative improvement = TBD (target ≈ 26%; not yet measured)`.
- Relative improvement is `(tuned - base) / base * 100`, computed only from real measured numbers, and rendered as `n/a` when base == 0.
- Each results JSON records the model id, adapter path, eval file SHA-256, sample count, seed, generation params and a timestamp, so anyone can reproduce the run.
