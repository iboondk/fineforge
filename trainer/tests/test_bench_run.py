import json
import hashlib
import sys

from fineforge_train.bench.run import main, results_record


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return str(path)


def test_dry_run_reports_n_and_hash_without_torch(tmp_path, capsys):
    eval_file = _write_jsonl(tmp_path / "eval.jsonl", [
        {"instruction": "q1", "output": "a1"},
        {"messages": [
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]},
    ])
    out_path = str(tmp_path / "results.json")

    rc = main(["--eval-file", eval_file, "--out", out_path, "--dry-run"])
    printed = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert printed["n"] == 2
    assert printed["would_write"] == out_path
    assert printed["eval_sha256"] == hashlib.sha256(open(eval_file, "rb").read()).hexdigest()
    assert "torch" not in sys.modules
    assert not (tmp_path / "results.json").exists()


def test_dry_run_limit(tmp_path, capsys):
    eval_file = _write_jsonl(tmp_path / "eval.jsonl", [
        {"instruction": "q1", "output": "a1"},
        {"instruction": "q2", "output": "a2"},
        {"instruction": "q3", "output": "a3"},
    ])
    assert main(["--eval-file", eval_file, "--limit", "2", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 2


def test_results_record_has_required_keys(tmp_path):
    eval_file = _write_jsonl(tmp_path / "eval.jsonl", [{"instruction": "q", "output": "a"}])
    scores = {"exact_match": 1.0, "token_f1": 1.0, "rouge_l": 1.0, "n": 1}
    rec = results_record("m", "adp", eval_file, 42, {"max_new_tokens": 8}, scores, scores)
    assert set(rec) == {
        "model_id", "adapter_path", "eval_file", "eval_sha256", "n", "seed",
        "generation", "timestamp_utc", "base", "tuned",
    }
    assert rec["n"] == 1 and rec["seed"] == 42
    assert len(rec["eval_sha256"]) == 64
    assert "torch" not in sys.modules
