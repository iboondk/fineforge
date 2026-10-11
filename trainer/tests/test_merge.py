import json
import sys

from fineforge_train.merge import main


def test_merge_dry_run_prints_planned_action(capsys):
    rc = main([
        "--base", "Qwen/Qwen2.5-0.5B-Instruct",
        "--adapter", "runs/qlora/adapter",
        "--out", "runs/qlora/merged",
        "--dry-run",
    ])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan == {
        "action": "merge_lora",
        "base": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter": "runs/qlora/adapter",
        "out": "runs/qlora/merged",
        "dtype": "float16",
        "safe_serialization": True,
    }
    assert "torch" not in sys.modules
