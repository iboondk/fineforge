"""Base-vs-tuned benchmark runner.

    python -m fineforge_train.bench.run --base-model ID --adapter DIR --eval-file FILE \
        [--out results/results.json] [--limit N] [--max-new-tokens 256] \
        [--batch-size 8] [--seed 42] [--dry-run]

``--dry-run`` only loads and checks the eval file. It prints the sample count, the
eval SHA-256 and the output path as JSON, and it never imports torch or any other
heavy ML dependency. Those imports live inside ``generate()`` and ``main()``'s real path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

from fineforge_train import data
from fineforge_train.bench.metrics import aggregate

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def results_record(
    model_id: str,
    adapter: str,
    eval_file: str,
    seed: int,
    gen_params: dict,
    base_scores: dict,
    tuned_scores: dict,
) -> dict:
    """Build the JSON-serialisable results record described in docs/benchmark-methodology.md."""
    if base_scores["n"] != tuned_scores["n"]:
        raise ValueError("base and tuned scores were computed on different sample counts")
    return {
        "model_id": model_id,
        "adapter_path": adapter,
        "eval_file": eval_file,
        "eval_sha256": _sha256_file(eval_file),
        "n": base_scores["n"],
        "seed": seed,
        "generation": dict(gen_params),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base": dict(base_scores),
        "tuned": dict(tuned_scores),
    }


def generate(model, tokenizer, messages_list: list[list[dict]], max_new_tokens: int, batch_size: int) -> list[str]:
    """Greedy-decode one reply per prompt, batching with left padding.

    Only the newly generated tokens are decoded, so the prompt is not echoed back.
    """
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs: list[str] = []
    for start in range(0, len(messages_list), batch_size):
        batch = messages_list[start:start + batch_size]
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        for row in new_tokens:
            outputs.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return outputs


def _load_eval(eval_file: str, limit: int | None) -> tuple[list[list[dict]], list[str]]:
    prompts: list[list[dict]] = []
    refs: list[str] = []
    for rec in data.load_jsonl(eval_file):
        prompt, ref = data.split_prompt_reference(rec)
        prompts.append(prompt)
        refs.append(ref)
    if limit is not None:
        prompts, refs = prompts[:limit], refs[:limit]
    return prompts, refs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fineforge_train.bench.run",
        description="Score a base model and its LoRA-tuned version on an eval file.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="base model id or local path")
    parser.add_argument("--adapter", help="PEFT adapter directory (required unless --dry-run)")
    parser.add_argument("--eval-file", required=True, help="JSONL eval file (see docs/dataset-format.md)")
    parser.add_argument("--out", default="results/results.json", help="output JSON path")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N records")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check the eval file and print the plan as JSON without loading any model",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    try:
        prompts, refs = _load_eval(args.eval_file, args.limit)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps({
            "n": len(prompts),
            "eval_sha256": _sha256_file(args.eval_file),
            "would_write": args.out,
        }, indent=2))
        return 0

    if not args.adapter:
        parser.error("--adapter is required for a real run")
    if not prompts:
        print("error: eval file has no scorable records", file=sys.stderr)
        return 2

    return _real_run(args, prompts, refs)


def _real_run(args: argparse.Namespace, prompts: list[list[dict]], refs: list[str]) -> int:
    import random

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    if torch.cuda.is_available():
        base = base.to("cuda")
    base.eval()

    base_preds = generate(base, tokenizer, prompts, args.max_new_tokens, args.batch_size)
    base_scores = aggregate(base_preds, refs)

    tuned = PeftModel.from_pretrained(base, args.adapter)
    tuned.eval()
    tuned_preds = generate(tuned, tokenizer, prompts, args.max_new_tokens, args.batch_size)
    tuned_scores = aggregate(tuned_preds, refs)

    gen_params = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "batch_size": args.batch_size,
        "decoding": "greedy",
    }
    record = results_record(
        model_id=args.base_model,
        adapter=args.adapter,
        eval_file=args.eval_file,
        seed=args.seed,
        gen_params=gen_params,
        base_scores=base_scores,
        tuned_scores=tuned_scores,
    )
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(json.dumps({"wrote": args.out, "n": record["n"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
