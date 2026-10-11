"""LoRA / QLoRA training CLI.

    python -m fineforge_train.train --config configs/qlora.yaml [--set KEY=VALUE ...] [--dry-run]

``--dry-run`` prints the plan as JSON and returns without importing torch or any
other heavy ML dependency. Those imports live inside ``train()``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

from fineforge_train import data
from fineforge_train.config import TrainConfig, load_config


def _load_records(paths: list[str]) -> list[dict]:
    """Load every record from the given files and check each one's shape."""
    records: list[dict] = []
    for path in paths:
        for rec in data.load_jsonl(path):
            try:
                data.to_messages(rec)
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from exc
            records.append(rec)
    return records


def build_plan(cfg: TrainConfig) -> dict:
    """Describe the run as a JSON-serialisable dict without training anything."""
    train_records = _load_records(cfg.train_files)
    eval_records = _load_records([cfg.eval_file]) if cfg.eval_file else []
    return {
        "base_model": cfg.base_model,
        "method": cfg.method,
        "quantization": "4bit-nf4" if cfg.method == "qlora" else None,
        "lora": {
            "r": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
            "target_modules": list(cfg.target_modules),
        },
        "num_train_records": len(train_records),
        "num_eval_records": len(eval_records),
        "effective_batch_size": cfg.batch_size * cfg.grad_accum,
        "output_dir": cfg.output_dir,
    }


def train(cfg: TrainConfig) -> None:
    """Fine-tune the base model and save the adapter to <output_dir>/adapter."""
    import dataclasses
    import inspect

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    plan = build_plan(cfg)  # validates the data before any model is loaded
    compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, quantization_config=bnb_config, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        if cfg.bf16:
            load_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            load_dtype = torch.float16
        else:
            load_dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=load_dtype)

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM",
    )

    train_ds = Dataset.from_list(
        [{"messages": data.to_messages(rec)} for rec in _load_records(cfg.train_files)]
    )
    eval_ds = None
    if cfg.eval_file:
        eval_ds = Dataset.from_list(
            [{"messages": data.to_messages(rec)} for rec in _load_records([cfg.eval_file])]
        )

    # Argument names differ across trl releases, so check what this install accepts.
    sft_fields = {f.name for f in dataclasses.fields(SFTConfig)}
    length_key = "max_seq_length" if "max_seq_length" in sft_fields else "max_length"
    sft_kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        seed=cfg.seed,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        bf16=cfg.bf16,
        **{length_key: cfg.max_seq_len},
    )
    if eval_ds is not None:
        sft_kwargs["eval_strategy"] = "epoch"
    sft_args = SFTConfig(**sft_kwargs)

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    tok_key = "processing_class" if "processing_class" in trainer_params else "tokenizer"
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        **{tok_key: tokenizer},
    )
    trainer.train()

    adapter_dir = os.path.join(cfg.output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    with open(os.path.join(cfg.output_dir, "train_plan.json"), "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2)
        fh.write("\n")


def _parse_set(parser: argparse.ArgumentParser, items: list[str]) -> dict:
    overrides: dict = {}
    for item in items:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        try:
            overrides[key] = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            parser.error(f"--set {key}: could not parse value {raw!r}: {exc}")
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fineforge_train.train",
        description="LoRA/QLoRA fine-tuning for FineForge datasets.",
    )
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key (value parsed as YAML); repeatable",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the run plan as JSON and exit without training",
    )
    args = parser.parse_args(argv)
    overrides = _parse_set(parser, args.set)

    try:
        cfg = load_config(args.config, overrides)
        if args.dry_run:
            print(json.dumps(build_plan(cfg), indent=2))
            return 0
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
