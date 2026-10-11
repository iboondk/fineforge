from pathlib import Path

import pytest
import yaml

from fineforge_train.config import TrainConfig, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_defaults():
    cfg = TrainConfig(train_files=["a.jsonl"])
    assert cfg.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.eval_file is None
    assert cfg.output_dir == "runs/default"
    assert cfg.method == "qlora"
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert cfg.max_seq_len == 1024
    assert cfg.epochs == 1.0
    assert cfg.lr == 2e-4
    assert cfg.batch_size == 4
    assert cfg.grad_accum == 4
    assert cfg.seed == 42
    assert cfg.bf16 is False
    assert cfg.max_steps == -1


def test_train_files_is_required():
    with pytest.raises(TypeError):
        TrainConfig()


def test_load_config_applies_yaml_and_overrides(tmp_path):
    train = _write(tmp_path / "train-0.jsonl", "")
    cfg_path = _write(
        tmp_path / "c.yaml",
        yaml.safe_dump({"train_files": [train], "lora_r": 32, "method": "lora"}),
    )
    cfg = load_config(cfg_path, {"lora_r": 4, "lr": "1e-4"})
    assert cfg.lora_r == 4  # override wins over YAML
    assert cfg.method == "lora"
    assert cfg.lr == pytest.approx(1e-4)  # "1e-4" is a string after yaml.safe_load
    assert cfg.train_files == [train]


def test_unknown_key_is_rejected_by_name(tmp_path):
    train = _write(tmp_path / "t.jsonl", "")
    cfg_path = _write(tmp_path / "c.yaml", yaml.safe_dump({"train_files": [train], "bogus_key": 1}))
    with pytest.raises(ValueError, match="bogus_key"):
        load_config(cfg_path)


def test_unknown_override_is_rejected(tmp_path):
    train = _write(tmp_path / "t.jsonl", "")
    cfg_path = _write(tmp_path / "c.yaml", yaml.safe_dump({"train_files": [train]}))
    with pytest.raises(ValueError, match="nope"):
        load_config(cfg_path, {"nope": 3})


def test_bad_method_is_rejected(tmp_path):
    train = _write(tmp_path / "t.jsonl", "")
    cfg_path = _write(tmp_path / "c.yaml", yaml.safe_dump({"train_files": [train], "method": "full"}))
    with pytest.raises(ValueError, match="method"):
        load_config(cfg_path)


def test_empty_train_files_is_rejected(tmp_path):
    cfg_path = _write(tmp_path / "c.yaml", yaml.safe_dump({"train_files": []}))
    with pytest.raises(ValueError, match="train_files"):
        load_config(cfg_path)


def test_missing_train_files_is_rejected(tmp_path):
    cfg_path = _write(tmp_path / "c.yaml", yaml.safe_dump({"lr": 1e-4}))
    with pytest.raises(ValueError, match="train_files"):
        load_config(cfg_path)


@pytest.mark.parametrize("key", ["lora_r", "batch_size", "grad_accum"])
def test_non_positive_ints_are_rejected(tmp_path, key):
    train = _write(tmp_path / "t.jsonl", "")
    cfg_path = _write(tmp_path / "c.yaml", yaml.safe_dump({"train_files": [train], key: 0}))
    with pytest.raises(ValueError, match=key):
        load_config(cfg_path)


def test_glob_expansion_is_sorted_and_relative_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train-00002.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "train-00000.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "train-00001.jsonl").write_text("", encoding="utf-8")
    cfg_path = _write(tmp_path / "c.yaml", yaml.safe_dump({"train_files": ["train-*.jsonl"]}))
    cfg = load_config(cfg_path)
    assert cfg.train_files == [
        "train-00000.jsonl",
        "train-00001.jsonl",
        "train-00002.jsonl",
    ]


def test_glob_with_no_match_names_the_pattern(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = _write(tmp_path / "c.yaml", yaml.safe_dump({"train_files": ["missing-*.jsonl"]}))
    with pytest.raises(ValueError, match="missing-"):
        load_config(cfg_path)


@pytest.mark.parametrize("name,method,out", [
    ("lora.yaml", "lora", "runs/lora"),
    ("qlora.yaml", "qlora", "runs/qlora"),
])
def test_shipped_configs_parse(name, method, out):
    raw = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
    assert raw["train_files"] == ["../data/out/train-*.jsonl"]
    assert raw["eval_file"] == "../data/out/eval.jsonl"
    assert raw["method"] == method
    assert raw["output_dir"] == out
