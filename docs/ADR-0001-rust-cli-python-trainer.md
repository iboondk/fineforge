# ADR-0001: Split the toolkit into a Rust CLI and a Python trainer

## Status

Accepted.

## Context

FineForge does two kinds of work with different needs. Dataset work (validate, dedupe, shard) streams large JSONL files and should be fast and ship as one binary. Training and benchmarking need the Hugging Face stack (`torch`, `transformers`, `peft`, `trl`), which only exists in Python and runs on a Colab GPU.

## Decision

- `cli/` is a Rust crate named `fineforge` (clap 4, serde_json, sha2, anyhow). It owns `validate`, `dedupe`, `shard` and the `gguf` wrapper.
- `trainer/` is a Python package named `fineforge_train`. It owns LoRA/QLoRA training, merging and the base-vs-tuned benchmark.
- The two halves share only the dataset contract (docs/dataset-format.md) and files on disk. Neither calls the other's code.

## Consequences

- Each half can be tested alone. `cargo test` needs no Python, and `pytest` in `trainer/` needs no GPU or torch (see the CI workflow).
- The contract must be kept in sync by hand. `scripts/test_scripts.py` checks the sample data and the notebook against the same shape rules.
- Users need two toolchains to build from source. A release binary avoids the Rust toolchain for dataset prep.
