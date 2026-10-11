import json

import pytest

from fineforge_train.data import load_jsonl, split_prompt_reference, to_messages


def test_instruction_without_input():
    rec = {"instruction": "Say hi", "output": "Hi!"}
    assert to_messages(rec) == [
        {"role": "user", "content": "Say hi"},
        {"role": "assistant", "content": "Hi!"},
    ]


def test_instruction_with_input_joins_with_blank_line():
    rec = {"instruction": "Summarise", "input": "Long text", "output": "Short"}
    assert to_messages(rec) == [
        {"role": "user", "content": "Summarise\n\nLong text"},
        {"role": "assistant", "content": "Short"},
    ]


def test_instruction_with_empty_input_is_treated_as_no_input():
    rec = {"instruction": "Say hi", "input": "", "output": "Hi!"}
    assert to_messages(rec)[0]["content"] == "Say hi"


def test_chat_passthrough_returns_copy():
    msgs = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    rec = {"messages": msgs, "id": "x1"}
    out = to_messages(rec)
    assert out == msgs
    assert out is not msgs
    out[0]["content"] = "changed"
    assert msgs[0]["content"] == "Be brief."


@pytest.mark.parametrize("rec", [
    {},
    {"foo": "bar"},
    {"instruction": "only instruction"},
    {"output": "only output"},
    {"instruction": "", "output": "x"},
    {"instruction": "q", "output": 5},
    {"messages": []},
    {"messages": [{"role": "robot", "content": "x"}]},
    {"messages": "not a list"},
])
def test_bad_shape_raises(rec):
    with pytest.raises(ValueError):
        to_messages(rec)


def test_split_prompt_reference_chat():
    rec = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]}
    prompt, ref = split_prompt_reference(rec)
    assert prompt == [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    assert ref == "a"


def test_split_prompt_reference_instruction():
    prompt, ref = split_prompt_reference({"instruction": "Add", "input": "1+1", "output": "2"})
    assert prompt == [{"role": "user", "content": "Add\n\n1+1"}]
    assert ref == "2"


def test_split_requires_final_assistant_turn():
    rec = {"messages": [
        {"role": "user", "content": "q"},
        {"role": "user", "content": "again"},
    ]}
    with pytest.raises(ValueError):
        split_prompt_reference(rec)


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert load_jsonl(str(path)) == [{"a": 1}, {"a": 2}]


def test_load_jsonl_bad_json_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\n\n{"a": \n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:3"):
        load_jsonl(str(path))


def test_load_jsonl_rejects_non_object_lines(tmp_path):
    path = tmp_path / "arr.jsonl"
    path.write_text(json.dumps([1, 2]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":1"):
        load_jsonl(str(path))
