# ADR-0002: Wrap llama.cpp by spawning its tools instead of reimplementing GGUF

## Status

Accepted.

## Context

GGUF conversion and quantization are already implemented by llama.cpp (`convert_hf_to_gguf.py` and `llama-quantize`). Reimplementing them would be large, fragile and would drift from upstream.

## Decision

- The `fineforge gguf convert` and `fineforge gguf quantize` subcommands only build a command line and spawn the external tool. Command builders live in `cli/src/gguf.rs` and are unit-tested as strings.
- Every subcommand accepts `--dry-run`, which prints `DRY-RUN: <command>` and exits 0. This makes the wrapper testable on machines without llama.cpp.
- Tool locations come from flags or environment variables (`--script` / `LLAMA_CPP_DIR`, `--bin` / `LLAMA_QUANTIZE`). A missing tool returns exit code 2 with a message that contains `not found`.
- Quantization types and output types are checked against fixed allow-lists before anything runs.

## Consequences

- The wrapper is only as correct as the flags it passes. Upstream flag changes need a manual check.
- The real conversion and quantization steps are not covered by CI, because CI has no llama.cpp and no model weights.
