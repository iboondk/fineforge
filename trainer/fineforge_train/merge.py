"""Merge a LoRA/QLoRA adapter into its base model and save HF weights.

    python -m fineforge_train.merge --base BASE --adapter DIR --out DIR [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys


def merge(base: str, adapter: str, out: str) -> None:
    """Load base in float16, apply the adapter, merge, and save with safetensors."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fineforge_train.merge",
        description="Merge a LoRA adapter into its base model.",
    )
    parser.add_argument("--base", required=True, help="base model id or local path")
    parser.add_argument("--adapter", required=True, help="adapter directory (e.g. runs/x/adapter)")
    parser.add_argument("--out", required=True, help="output directory for merged weights")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned action as JSON and exit without loading any model",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        plan = {
            "action": "merge_lora",
            "base": args.base,
            "adapter": args.adapter,
            "out": args.out,
            "dtype": "float16",
            "safe_serialization": True,
        }
        print(json.dumps(plan, indent=2))
        return 0

    try:
        merge(args.base, args.adapter, args.out)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
