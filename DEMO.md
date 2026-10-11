# FineForge demo script

This script records a short GIF or Loom (about 90 seconds) that shows the dataset
pipeline end to end. It uses only the bundled synthetic sample data, so no account,
token or GPU is needed. Training is shown as a dry run only, because a real run needs a
Colab GPU.

## Before you record

1. Build the CLI: `cd cli && cargo build --release && cd ..`
2. Create the trainer environment: `cd trainer && python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]" && cd ..`
3. Use a clean terminal with a large font (16pt or more) and a dark theme.
4. Clear old outputs: `rm -rf data/out`
5. Set the prompt to something short, for example `$ `.

## Scene 1: the problem (0:00 to 0:10)

- Show `data/sample/sample.jsonl` with `head -n 3 data/sample/sample.jsonl`.
- Narration: "Raw fine-tuning data has bad lines, duplicates and no fixed split. FineForge handles that first."

## Scene 2: validate (0:10 to 0:25)

```bash
./cli/target/release/fineforge validate data/sample/sample.jsonl
```

- Expected: a line `total=... valid=... invalid=...`, and exit code 0 when every line is valid.
- Point out `echo $?` to show the exit code.

## Scene 3: dedupe (0:25 to 0:40)

```bash
mkdir -p data/out
./cli/target/release/fineforge dedupe data/sample/sample.jsonl -o data/out/deduped.jsonl --near 0.8 --stats data/out/dedupe-stats.json
```

- Expected: a summary line with `exact_dupes` and `near_dupes`. Read the actual numbers from your own run and do not copy numbers from this script.

## Scene 4: shard (0:40 to 0:55)

```bash
./cli/target/release/fineforge shard data/out/deduped.jsonl --out-dir data/out --shards 2 --eval-ratio 0.1 --seed 42
ls data/out
```

- Show `manifest.json`, `eval.jsonl` and the `train-*.jsonl` shards.
- Narration: "The same seed always gives the same split, byte for byte."

## Scene 5: GGUF command preview (0:55 to 1:05)

```bash
./cli/target/release/fineforge gguf convert path/to/hf-model --outfile model-f16.gguf --script /path/to/llama.cpp/convert_hf_to_gguf.py --dry-run
./cli/target/release/fineforge gguf quantize model-f16.gguf model-Q4_K_M.gguf --type Q4_K_M --dry-run
```

- Narration: "llama.cpp does the conversion. FineForge builds the command, and dry-run shows it before anything runs."

## Scene 6: training plan (1:05 to 1:20)

```bash
cd trainer
. .venv/bin/activate
python -m fineforge_train.train --config configs/qlora.yaml --dry-run
cd ..
```

- Narration: "The dry run prints the plan as JSON without importing torch. The real run happens on a Colab T4."

## Scene 7: benchmark status (1:20 to 1:30)

```bash
cat results/benchmark.md
```

- Show the `TBD` cells and the headline line.
- Narration: "We do not publish a number until a real run produces one. The target is marked as not yet measured."

## Closing card (1:30 to 1:35)

- Show the repository URL (replace the placeholder with the real one before publishing) and the line "MIT licensed".

## Recording tips

- Run every command once before recording so that the output is cached and the timing is predictable.
- For a GIF, keep each scene under 15 seconds and trim the dry-run JSON with `| head -n 30`.
- Do not show the `HF_TOKEN` value. Keep `.env` out of the frame. Use `.env.example` in the shot.
