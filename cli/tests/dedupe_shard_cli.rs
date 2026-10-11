use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use predicates::str::contains;
use serde_json::Value;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn fineforge() -> Command {
    Command::cargo_bin("fineforge").unwrap()
}

/// Reads every regular file in `dir` into a name -> bytes map.
fn read_dir_bytes(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            let name = entry.file_name().to_string_lossy().into_owned();
            out.insert(name, fs::read(entry.path()).unwrap());
        }
    }
    out
}

fn run_shard(input: &Path, out_dir: &Path, seed: &str) {
    fineforge()
        .arg("shard")
        .arg(input)
        .arg("--out-dir")
        .arg(out_dir)
        .args(["--seed", seed, "--shards", "2", "--eval-ratio", "0.25"])
        .assert()
        .success();
}

fn manifest_sums(dir: &Path) -> (u64, u64, u64) {
    let text = fs::read_to_string(dir.join("manifest.json")).unwrap();
    let v: Value = serde_json::from_str(&text).unwrap();
    let eval_count = v["eval_count"].as_u64().unwrap();
    let eval_file = v["eval"]["count"].as_u64().unwrap();
    let train: u64 = v["train_shards"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["count"].as_u64().unwrap())
        .sum();
    (v["total"].as_u64().unwrap(), eval_count, eval_file + train)
}

#[test]
fn dedupe_without_near_removes_exact_dupes() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.jsonl");
    let stats = dir.path().join("stats.json");

    fineforge()
        .arg("dedupe")
        .arg(fixture("dupes.jsonl"))
        .arg("-o")
        .arg(&out)
        .arg("--stats")
        .arg(&stats)
        .assert()
        .success()
        .stdout(contains(
            "total=8 kept=5 exact_dupes=2 near_dupes=0 invalid_skipped=1",
        ));

    let v: Value = serde_json::from_str(&fs::read_to_string(&stats).unwrap()).unwrap();
    assert_eq!(v["exact_dupes"], 2);
    assert_eq!(v["invalid_skipped"], 1);
}

#[test]
fn dedupe_with_near_removes_near_dupe() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.jsonl");

    fineforge()
        .arg("dedupe")
        .arg(fixture("dupes.jsonl"))
        .arg("-o")
        .arg(&out)
        .args(["--near", "0.7"])
        .assert()
        .success()
        .stdout(contains(
            "total=8 kept=4 exact_dupes=2 near_dupes=1 invalid_skipped=1",
        ));

    let kept = fs::read_to_string(&out).unwrap();
    assert_eq!(kept.lines().count(), 4);
    assert!(kept.ends_with('\n'));
}

#[test]
fn shard_is_byte_identical_across_runs_with_same_seed() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let input = fixture("valid.jsonl");

    run_shard(&input, a.path(), "7");
    run_shard(&input, b.path(), "7");

    let files_a = read_dir_bytes(a.path());
    let files_b = read_dir_bytes(b.path());
    assert!(files_a.contains_key("manifest.json"));
    assert!(files_a.contains_key("eval.jsonl"));
    assert_eq!(files_a, files_b);

    let (total, eval_count, sum) = manifest_sums(a.path());
    assert_eq!(total, 4);
    assert_eq!(eval_count, 1);
    assert_eq!(sum, 4);
}

#[test]
fn shard_with_different_seed_still_sums_to_total() {
    let dir = tempfile::tempdir().unwrap();
    run_shard(&fixture("valid.jsonl"), dir.path(), "8");

    let (total, eval_count, sum) = manifest_sums(dir.path());
    assert_eq!(total, 4);
    assert_eq!(eval_count, 1);
    assert_eq!(sum, 4);
}

#[test]
fn shard_rejects_invalid_input_with_line_number() {
    let dir = tempfile::tempdir().unwrap();
    fineforge()
        .arg("shard")
        .arg(fixture("mixed.jsonl"))
        .arg("--out-dir")
        .arg(dir.path().join("out"))
        .assert()
        .code(1)
        .stderr(contains("line 4"));
}
