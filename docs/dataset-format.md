# Dataset format

FineForge reads and writes JSONL: UTF-8, one JSON object per line. Two record shapes are accepted.

## Shapes

- **chat**: `{"messages": [{"role": "system" | "user" | "assistant", "content": "<non-empty string>"}, ...]}`
  - at least 2 messages
  - at least one `user` message
  - the **last** message is `assistant`
  - `system` may appear only at index 0
- **instruction**: `{"instruction": "<non-empty string>", "input": "<string, optional>", "output": "<non-empty string>"}`

Extra top-level keys (for example `"id"` or `"source"`) are allowed and kept as they are.

## Canonical text

`fineforge dedupe` hashes a canonical form of each record. The form must stay stable, because changing it changes every dedupe result.

- **chat**: join `"{role}: {content}"` for each message with `\n`.
- **instruction**: `"instruction: {instruction}\ninput: {input}\noutput: {output}"`. `input` is `""` when missing.

Normalization before hashing lowercases the text, collapses every run of Unicode whitespace to a single space, and trims both ends.

## Validation

`fineforge validate <file>` checks every line against the rules above and writes a JSON report. It exits non-zero when any record is invalid.
