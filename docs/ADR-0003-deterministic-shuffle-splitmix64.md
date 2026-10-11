# ADR-0003: Use a hand-written SplitMix64 and Fisher-Yates shuffle for sharding

## Status

Accepted.

## Context

`fineforge shard` must produce byte-identical shards and eval split for the same input and seed (PLAN.md milestone M2). A general-purpose RNG crate can change its output between versions, and that would silently change every dataset split.

## Decision

- `cli/src/rng.rs` implements SplitMix64 directly, with a Fisher-Yates shuffle on top of it.
- No `rand` dependency is used for sharding.
- Near-duplicate detection uses MinHash over word 5-gram shingles with 64 seeded hashes, also written in-crate (`cli/src/dedupe.rs`).

## Consequences

- The output is stable across platforms and crate upgrades. The algorithm is small enough to review by reading it.
- The implementation must be maintained. Any change to the generator changes every split, so it must be treated as a breaking change.
- Tests in `cli/tests/dedupe_shard_cli.rs` compare outputs from repeated runs with the same seed.
