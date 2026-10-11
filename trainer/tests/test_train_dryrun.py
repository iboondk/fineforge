import json
import sys

import yaml

from fineforge_train.train import main


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return str(path)


def test_dry_run_plan_without_torch(tmp_path, capsys):
    _write_jsonl(tmp_path / "train-00000.jsonl", [
        {"instruction": "q1", "output": "a1"},
        {"instruction": "q2", "input": "x", "output": "a2"},
        {"messages": [
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ]},
    ])
    eval_file = _write_jsonl(tmp_path / "eval.jsonl", [
        {"instruction": "e1", "output": "ea1"},
    ])
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "train_files": [str(tmp_path / "train-*.jsonl")],
        "eval_file": eval_file,
        "output_dir": str(tmp_path / "runs" / "qlora"),
        "method": "qlora",
    }), encoding="utf-8")

    rc = main(["--config", str(cfg_path), "--dry-run", "--set", "lora_r=8"])
    out = capsys.readouterr().out

    assert rc == 0
    plan = json.loads(out)
    assert plan["num_train_records"] == 3
    assert plan["num_eval_records"] == 1
    assert plan["lora"]["r"] == 8
    assert plan["quantization"] == "4bit-nf4"
    assert plan["effective_batch_size"] == 16
    assert plan["method"] == "qlora"
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_dry_run_lora_has_no_quantization(tmp_path, capsys):
    train = _write_jsonl(tmp_path / "t.jsonl", [{"instruction": "q", "output": "a"}])
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"train_files": [train], "method": "lora"}), encoding="utf-8")

    assert main(["--config", str(cfg_path), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["quantization"] is None
    assert plan["num_eval_records"] == 0
    assert plan["effective_batch_size"] == 16


def test_dry_run_bad_override_returns_error(tmp_path, capsys):
    train = _write_jsonl(tmp_path / "t.jsonl", [{"instruction": "q", "output": "a"}])
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"train_files": [train]}), encoding="utf-8")

    assert main(["--config", str(cfg_path), "--dry-run", "--set", "nope=1"]) == 2
    assert "nope" in capsys.readouterr().err
