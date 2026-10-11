use std::fs;
use std::path::PathBuf;

use assert_cmd::Command;
use predicates::str::contains;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn valid_fixture_exits_zero() {
    Command::cargo_bin("fineforge")
        .unwrap()
        .arg("validate")
        .arg(fixture("valid.jsonl"))
        .assert()
        .success()
        .stdout(contains("invalid=0"));
}

#[test]
fn mixed_fixture_writes_report_and_exits_one() {
    let dir = tempfile::tempdir().unwrap();
    let report = dir.path().join("r.json");

    Command::cargo_bin("fineforge")
        .unwrap()
        .arg("validate")
        .arg(fixture("mixed.jsonl"))
        .arg("--report")
        .arg(&report)
        .assert()
        .code(1)
        .stdout(contains("total=6 valid=3 invalid=3"));

    let text = fs::read_to_string(&report).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["invalid"], 3);
    let lines: Vec<u64> = v["errors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["line"].as_u64().unwrap())
        .collect();
    assert_eq!(lines, vec![4, 5, 6]);
}

#[test]
fn missing_input_exits_two() {
    Command::cargo_bin("fineforge")
        .unwrap()
        .arg("validate")
        .arg(fixture("does-not-exist.jsonl"))
        .assert()
        .code(2);
}
