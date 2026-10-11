# FineForge write-up

## Problem

Fine-tuning a small language model on a custom dataset usually goes wrong before training starts. Raw JSONL files contain malformed lines, exact and near duplicates, and no reproducible train/eval split. After training, comparing the tuned model with the base model and converting it for local inference are more manual steps. FineForge covers that path in one repository.

## Architecture

The toolkit has two halves that share only a dataset contract and files on disk.

- A Rust CLI, `fineforge`, validates, dedupes and shards JSONL datasets, and wraps llama.cpp's GGUF convert and quantize tools.
- A Python package, `fineforge_train`, runs LoRA and QLoRA training, merges adapters, and benchmarks the base model against the tuned one.

The contract accepts chat-message and instruction records, and defines one canonical text form that the Rust deduper depends on.

## Key decisions

- **Rust for dataset work.** Streaming large JSONL files is fast, and the result ships as one binary (ADR-0001).
- **Shell out to llama.cpp.** GGUF conversion is not reimplemented. Each command has `--dry-run`, so the wrapper is testable without a model (ADR-0002).
- **Deterministic shuffle.** SplitMix64 is written in-crate, so the same seed gives byte-identical shards on every platform (ADR-0003).
- **Lazy heavy imports.** `torch`, `transformers`, `peft` and `trl` load only inside functions, so unit tests and dry runs need no GPU (ADR-0005).
- **Honest benchmark.** The report shows `TBD` until a real run produces numbers, and each results file records its provenance (ADR-0004).

## Results

No benchmark results exist yet, and this write-up claims none. The project targets a relative improvement of about 26% over the base model. That target has not been measured, and `results/benchmark.md` says so.

What has been checked is the software: the Rust and Python test suites, and the dry-run paths for training, merging and GGUF conversion. These confirm the pipeline logic. They say nothing about model quality.

## What I would do next

1. Run the Colab notebook on a T4 and commit the resulting results JSON, so the benchmark shows measured values.
2. Replace the synthetic sample with a licensed dataset and document its provenance.
3. Add an end-to-end CI job that shards the sample data and checks the manifest against a stored hash.
4. Publish prebuilt binaries from tagged releases.

The main risk is the eval split. A fixed split with a recorded hash is a first step, but a person should review the eval set before any claim is made.
