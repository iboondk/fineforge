.PHONY: all build venv test demo clean

BIN := cli/target/release/fineforge
PY  := trainer/.venv/bin/python

# One command: build, test, and run the sample pipeline end to end (no GPU needed).
all: test demo

build:
	cd cli && cargo build --release

venv:
	@$(PY) -c "import fineforge_train, pytest" 2>/dev/null || { \
	  python3 -m venv trainer/.venv && $(PY) -m pip install -q -e "trainer[dev]"; }

test: build venv
	cd cli && cargo test --all-targets
	cd trainer && .venv/bin/python -m pytest -q
	$(PY) -m pytest -q scripts/test_scripts.py

demo: build venv
	rm -rf data/out && mkdir -p data/out
	$(BIN) validate data/sample/sample.jsonl
	$(BIN) dedupe data/sample/sample.jsonl -o data/out/deduped.jsonl --near 0.8 --stats data/out/dedupe-stats.json
	$(BIN) shard data/out/deduped.jsonl --out-dir data/out --shards 2 --eval-ratio 0.1 --seed 42
	cd trainer && .venv/bin/python -m fineforge_train.train --config configs/qlora.yaml --dry-run
	$(BIN) gguf quantize model-f16.gguf model-Q4_K_M.gguf --type Q4_K_M --dry-run

clean:
	rm -rf data/out runs
