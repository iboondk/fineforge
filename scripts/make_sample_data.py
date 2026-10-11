#!/usr/bin/env python3
"""Write a small SYNTHETIC toy dataset for the FineForge pipeline.

    python scripts/make_sample_data.py [--out data/sample/sample.jsonl] [--n 200] [--seed 0]

Every record is built from templates and a small hard-coded table, so the output is a
pure function of (--n, --seed): running twice with the same arguments writes
byte-identical files. Records use the two shapes from docs/dataset-format.md (instruction
and chat) and carry ``"source": "synthetic-fineforge-sample"``.

About 5% of the lines are case or whitespace variants of other records, so that
``fineforge dedupe`` has exact duplicates to remove. Stdlib only.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

SOURCE = "synthetic-fineforge-sample"
DUPLICATE_RATIO = 0.05
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "sample" / "sample.jsonl"

# Capital facts (well-known, public geography).
CAPITALS = [
    ("France", "Paris"),
    ("Japan", "Tokyo"),
    ("Canada", "Ottawa"),
    ("Kenya", "Nairobi"),
    ("Peru", "Lima"),
    ("Norway", "Oslo"),
    ("Egypt", "Cairo"),
    ("Australia", "Canberra"),
    ("Brazil", "Brasília"),
    ("Portugal", "Lisbon"),
    ("Thailand", "Bangkok"),
    ("Mongolia", "Ulaanbaatar"),
]

# (from unit, to unit, multiplication factor). Factors are standard conversions.
CONVERSIONS = [
    ("kilometers", "miles", 0.621371),
    ("meters", "feet", 3.28084),
    ("kilograms", "pounds", 2.20462),
    ("liters", "US gallons", 0.264172),
    ("inches", "centimeters", 2.54),
    ("miles", "kilometers", 1.609344),
]

# English to French phrases.
TRANSLATIONS = [
    ("Good morning", "Bonjour"),
    ("Thank you", "Merci"),
    ("Good night", "Bonne nuit"),
    ("Where is the station?", "Où est la gare ?"),
    ("I like tea.", "J'aime le thé."),
]

NAMES = ["Maya", "Omar", "Lena", "Kenji", "Ada", "Tomas"]
SYSTEM_PROMPT = "You are a concise assistant."


def _arithmetic(rng: random.Random) -> tuple[str, str, str]:
    name = rng.choice(NAMES)
    if rng.random() < 0.5:
        boxes = rng.randint(2, 12)
        per_box = rng.randint(2, 9)
        eaten = rng.randint(1, boxes * per_box - 1)
        left = boxes * per_box - eaten
        problem = (
            f"A box holds {per_box} apples. {name} buys {boxes} boxes and eats "
            f"{eaten} apples. How many apples does {name} have left?"
        )
        answer = f"{name} has {left} apples left."
    else:
        speed = rng.randint(20, 90)
        hours = rng.randint(2, 9)
        problem = (
            f"A train travels {speed} km each hour for {hours} hours. "
            f"How far does it go in total?"
        )
        answer = f"It goes {speed * hours} km in total."
    return ("Solve the word problem and give the final number.", problem, answer)


def _conversion(rng: random.Random) -> tuple[str, str, str]:
    src, dst, factor = rng.choice(CONVERSIONS)
    value = rng.randint(2, 500)
    result = value * factor
    instruction = f"Convert {value} {src} to {dst}. Round to two decimals."
    answer = f"{value} {src} is about {result:.2f} {dst}."
    return (instruction, "", answer)


def _capital(rng: random.Random) -> tuple[str, str, str]:
    country, capital = rng.choice(CAPITALS)
    if rng.random() < 0.5:
        return (f"What is the capital of {country}?", "", f"The capital of {country} is {capital}.")
    return (f"Name the capital city of {country}.", "", capital)


def _translation(rng: random.Random) -> tuple[str, str, str]:
    english, french = rng.choice(TRANSLATIONS)
    return ("Translate the sentence to French.", english, french)


GENERATORS = [_arithmetic, _conversion, _capital, _translation]


def _triple_key(triple: tuple[str, str, str]) -> str:
    """Case and whitespace insensitive key, used to keep base records distinct."""
    return " | ".join(" ".join(part.lower().split()) for part in triple)


def _shape(rng: random.Random, triple: tuple[str, str, str]) -> dict:
    """Wrap a (instruction, input, output) triple as an instruction or chat record."""
    instruction, input_text, output = triple
    if rng.random() < 0.5:
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "source": SOURCE,
        }
    messages = []
    if rng.random() < 0.25:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    user_text = f"{instruction}\n{input_text}" if input_text else instruction
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": output})
    return {"messages": messages, "source": SOURCE}


def _mangle(text: str, rng: random.Random) -> str:
    """Return a case or whitespace variant that normalizes to the same text."""
    choice = rng.randrange(3)
    if choice == 0:
        variant = text.upper()
    elif choice == 1:
        variant = text.lower()
    else:
        variant = "  " + text.replace(" ", "  ") + "  "
    if variant == text:
        variant = " " + text + " "
    return variant


def _variant(rng: random.Random, rec: dict) -> dict:
    """Copy a record and vary its text, so it dedupes to the original."""
    out = copy.deepcopy(rec)
    if "messages" in out:
        for msg in out["messages"]:
            msg["content"] = _mangle(msg["content"], rng)
    else:
        out["instruction"] = _mangle(out["instruction"], rng)
        if out["input"]:
            out["input"] = _mangle(out["input"], rng)
        out["output"] = _mangle(out["output"], rng)
    return out


def build_records(n: int, seed: int) -> list[dict]:
    """Return n records: unique synthetic base records plus about 5% variant duplicates."""
    if n < 0:
        raise ValueError("--n must be >= 0")
    rng = random.Random(seed)

    n_dup = round(n * DUPLICATE_RATIO)
    n_base = n - n_dup

    base: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(1000, n_base * 200)
    while len(base) < n_base:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(f"could not generate {n_base} distinct records")
        gen = rng.choice(GENERATORS)
        triple = gen(rng)
        key = _triple_key(triple)
        if key in seen:
            continue
        seen.add(key)
        base.append(_shape(rng, triple))

    records = list(base)
    for _ in range(n_dup):
        source = rng.choice(base)
        records.insert(rng.randint(0, len(records)), _variant(rng, source))

    for idx, rec in enumerate(records):
        rec["id"] = f"synthetic-{idx:05d}"
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/make_sample_data.py",
        description="Write a deterministic SYNTHETIC toy dataset for FineForge.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output JSONL path")
    parser.add_argument("--n", type=int, default=200, help="number of records (default 200)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.n < 0:
        print("error: --n must be >= 0", file=sys.stderr)
        return 2
    try:
        records = build_records(args.n, args.seed)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    write_jsonl(out, records)
    print(f"wrote {len(records)} synthetic records to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
