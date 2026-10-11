"""JSONL loading and conversion of dataset records into chat messages.

Two record shapes are accepted (see docs/dataset-format.md):

* chat:        {"messages": [{"role": ..., "content": ...}, ...]}
* instruction: {"instruction": ..., "input": ... (optional), "output": ...}
"""

from __future__ import annotations

import json

_ROLES = ("system", "user", "assistant")


def load_jsonl(path: str) -> list[dict]:
    """Read a UTF-8 JSONL file, skipping blank lines.

    Raises ValueError with the file and 1-based line number on bad JSON.
    """
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: each line must be a JSON object")
            records.append(obj)
    return records


def _non_empty_str(rec: dict, key: str) -> str:
    value = rec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"record field {key!r} must be a non-empty string")
    return value


def _chat_messages(rec: dict) -> list[dict]:
    raw = rec["messages"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("'messages' must be a non-empty list")
    messages: list[dict] = []
    for idx, msg in enumerate(raw):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{idx}] must be an object")
        if msg.get("role") not in _ROLES:
            raise ValueError(f"messages[{idx}] has invalid role {msg.get('role')!r}")
        if not isinstance(msg.get("content"), str) or not msg["content"].strip():
            raise ValueError(f"messages[{idx}] content must be a non-empty string")
        messages.append(dict(msg))
    return messages


def to_messages(rec: dict) -> list[dict]:
    """Return a chat-style list of {"role", "content"} dicts for one record."""
    if "messages" in rec:
        return _chat_messages(rec)
    if "instruction" in rec and "output" in rec:
        instruction = _non_empty_str(rec, "instruction")
        output = _non_empty_str(rec, "output")
        raw_input = rec.get("input")
        if raw_input is not None and not isinstance(raw_input, str):
            raise ValueError("record field 'input' must be a string when present")
        if raw_input and raw_input.strip():
            user = f"{instruction}\n\n{raw_input}"
        else:
            user = instruction
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": output},
        ]
    raise ValueError(
        "unrecognised record shape: expected a 'messages' list, "
        "or 'instruction' and 'output' keys"
    )


def split_prompt_reference(rec: dict) -> tuple[list[dict], str]:
    """Split a record into (prompt messages, reference answer) for benchmarking.

    The prompt is every message except the final assistant turn, which is returned
    as the reference string.
    """
    messages = to_messages(rec)
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("the final message must be an assistant turn to use as the reference")
    return messages[:-1], messages[-1]["content"]
