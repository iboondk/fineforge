"""Training configuration: the TrainConfig dataclass and YAML loading."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field, fields
from typing import Any

import yaml

METHODS = ("lora", "qlora")
_POSITIVE_INT_FIELDS = ("lora_r", "batch_size", "grad_accum")


@dataclass(kw_only=True)
class TrainConfig:
    """All settings for one LoRA or QLoRA fine-tuning run."""

    train_files: list[str]
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    eval_file: str | None = None
    output_dir: str = "runs/default"
    method: str = "qlora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    max_seq_len: int = 1024
    epochs: float = 1.0
    lr: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 4
    seed: int = 42
    bf16: bool = False
    max_steps: int = -1


_FIELD_NAMES = tuple(f.name for f in fields(TrainConfig))
_INT_FIELDS = {"lora_r", "lora_alpha", "max_seq_len", "batch_size", "grad_accum", "seed", "max_steps"}
_FLOAT_FIELDS = {"lora_dropout", "epochs", "lr"}
_STR_FIELDS = {"base_model", "output_dir", "method"}
_LIST_FIELDS = {"train_files", "target_modules"}


def _coerce(name: str, value: Any) -> Any:
    """Check and convert one raw value to the type its field declares."""
    if name in _INT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return value
    if name in _FLOAT_FIELDS:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a number, got {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        try:
            # --set values go through yaml.safe_load, which reads "2e-4" as a string.
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number, got {value!r}") from None
    if name == "bf16":
        if not isinstance(value, bool):
            raise ValueError(f"bf16 must be true or false, got {value!r}")
        return value
    if name in _STR_FIELDS:
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string, got {value!r}")
        return value
    if name == "eval_file":
        if value is not None and not isinstance(value, str):
            raise ValueError(f"eval_file must be a string or null, got {value!r}")
        return value
    if name in _LIST_FIELDS:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{name} must be a list of strings, got {value!r}")
        return list(value)
    raise ValueError(f"unknown config key: {name}")  # pragma: no cover - guarded earlier


def _expand_train_files(patterns: list[str]) -> list[str]:
    """Expand glob patterns relative to the current directory, sorted per pattern."""
    expanded: list[str] = []
    for pattern in patterns:
        if glob.has_magic(pattern):
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise ValueError(
                    f"train_files glob matched no files: {pattern!r} "
                    f"(relative to {os.getcwd()})"
                )
            expanded.extend(matches)
        else:
            if not os.path.isfile(pattern):
                raise ValueError(f"train file not found: {pattern!r} (relative to {os.getcwd()})")
            expanded.append(pattern)
    return expanded


def _validate(cfg: TrainConfig) -> None:
    if cfg.method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}, got {cfg.method!r}")
    if not cfg.train_files:
        raise ValueError("train_files is required and must be a non-empty list")
    for name in _POSITIVE_INT_FIELDS:
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be a positive integer, got {getattr(cfg, name)}")


def load_config(path: str, overrides: dict | None = None) -> TrainConfig:
    """Read a YAML config, apply overrides, validate, and expand train_files globs."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")

    data = dict(raw)
    if overrides:
        data.update(overrides)

    unknown = sorted(str(k) for k in data if k not in _FIELD_NAMES)
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(unknown)}")

    values = {name: _coerce(name, value) for name, value in data.items()}
    values.setdefault("train_files", [])
    cfg = TrainConfig(**values)
    _validate(cfg)
    cfg.train_files = _expand_train_files(cfg.train_files)
    return cfg
