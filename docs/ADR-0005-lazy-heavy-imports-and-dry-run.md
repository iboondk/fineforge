# ADR-0005: Import torch and the ML stack only inside functions, with --dry-run

## Status

Accepted.

## Context

The trainer's unit tests and the `--dry-run` paths should run on a laptop with no GPU, no CUDA and no ML packages. Module-level imports of `torch`, `transformers`, `peft` and `trl` would make every import of the package slow or failing.

## Decision

- Core dependencies are `pyyaml` only. The heavy stack is the optional `train` extra, and tests use the `dev` extra (`pytest`).
- `torch`, `transformers`, `peft`, `trl` and `bitsandbytes` are imported inside the functions that need them.
- `python -m fineforge_train.train --config ... --dry-run` prints the run plan as JSON and returns without importing the ML stack. `merge` and `bench.run` have the same `--dry-run` behaviour.

## Consequences

- `pytest` runs in CI without torch installed (see `.github/workflows/ci.yml`).
- A missing ML dependency shows up at run time, not at import time. The README tells users to install the `train` extra before a real run.
