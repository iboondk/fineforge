use std::fs;

use assert_cmd::Command;
use predicates::str::contains;

fn fineforge() -> Command {
    let mut cmd = Command::cargo_bin("fineforge").unwrap();
    cmd.env_remove("LLAMA_CPP_DIR");
    cmd.env_remove("LLAMA_QUANTIZE");
    cmd
}

#[test]
fn quantize_dry_run_prints_exact_command() {
    fineforge()
        .args([
            "gguf",
            "quantize",
            "in.gguf",
            "out.gguf",
            "--type",
            "Q5_K_M",
            "--bin",
            "/x/llama-quantize",
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout("DRY-RUN: /x/llama-quantize in.gguf out.gguf Q5_K_M\n");
}

#[test]
fn quantize_dry_run_defaults_to_llama_quantize_and_q4_k_m() {
    fineforge()
        .args(["gguf", "quantize", "a.gguf", "b.gguf", "--dry-run"])
        .assert()
        .success()
        .stdout("DRY-RUN: llama-quantize a.gguf b.gguf Q4_K_M\n");
}

#[test]
fn quantize_dry_run_honors_llama_quantize_env() {
    fineforge()
        .env("LLAMA_QUANTIZE", "/env/llama-quantize")
        .args(["gguf", "quantize", "a.gguf", "b.gguf", "--dry-run"])
        .assert()
        .success()
        .stdout("DRY-RUN: /env/llama-quantize a.gguf b.gguf Q4_K_M\n");
}

#[test]
fn convert_dry_run_contains_outtype_flag() {
    fineforge()
        .args([
            "gguf",
            "convert",
            "m",
            "--outfile",
            "o.gguf",
            "--script",
            "/s/convert_hf_to_gguf.py",
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(contains("--outtype f16"));
}

#[test]
fn convert_without_script_or_env_exits_two() {
    fineforge()
        .args(["gguf", "convert", "m", "--outfile", "o.gguf"])
        .assert()
        .code(2)
        .stderr(contains("set --script or LLAMA_CPP_DIR"));
}

#[test]
fn convert_without_script_or_env_exits_two_in_dry_run() {
    fineforge()
        .args(["gguf", "convert", "m", "--outfile", "o.gguf", "--dry-run"])
        .assert()
        .code(2)
        .stderr(contains("set --script or LLAMA_CPP_DIR"));
}

#[test]
fn quantize_rejects_unknown_type_and_lists_allowed() {
    let assert = fineforge()
        .args([
            "gguf",
            "quantize",
            "in.gguf",
            "out.gguf",
            "--type",
            "Q9",
            "--dry-run",
        ])
        .assert()
        .failure();
    assert
        .stderr(contains("Q9"))
        .stderr(contains("Q4_0"))
        .stderr(contains("Q4_K_M"))
        .stderr(contains("Q5_K_M"))
        .stderr(contains("Q6_K"))
        .stderr(contains("Q8_0"));
}

#[test]
fn convert_rejects_unknown_outtype_and_lists_allowed() {
    fineforge()
        .args([
            "gguf",
            "convert",
            "m",
            "--outfile",
            "o.gguf",
            "--script",
            "/s/convert_hf_to_gguf.py",
            "--outtype",
            "q4_0",
            "--dry-run",
        ])
        .assert()
        .failure()
        .stderr(contains("f32, f16, bf16, q8_0, auto"));
}

#[test]
fn quantize_real_run_with_missing_binary_reports_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.gguf");
    fs::write(&input, b"not really gguf").unwrap();
    let output = dir.path().join("out.gguf");

    fineforge()
        .args(["gguf", "quantize"])
        .arg(&input)
        .arg(&output)
        .args(["--bin", "/nonexistent/llama-quantize"])
        .assert()
        .failure()
        .code(2)
        .stderr(contains("not found"));
}

#[test]
fn quantize_real_run_with_missing_input_exits_two() {
    fineforge()
        .args([
            "gguf",
            "quantize",
            "/nonexistent/in.gguf",
            "out.gguf",
            "--bin",
            "/nonexistent/llama-quantize",
        ])
        .assert()
        .code(2)
        .stderr(contains("does not exist"));
}
