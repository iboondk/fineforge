//! Command builders and a runner for the llama.cpp GGUF convert and quantize
//! steps. llama.cpp is never reimplemented here: we only build and spawn the
//! external commands, or print them in dry-run mode.

use std::io::ErrorKind;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};

/// Output types accepted by `convert_hf_to_gguf.py --outtype`.
pub const ALLOWED_OUTTYPES: &[&str] = &["f32", "f16", "bf16", "q8_0", "auto"];

/// Quantization types accepted by `llama-quantize`.
pub const ALLOWED_QTYPES: &[&str] = &["Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"];

/// Builds `<python> <script> <model_dir> --outfile <outfile> --outtype <outtype>`.
pub fn convert_cmd(
    model_dir: &Path,
    outfile: &Path,
    outtype: &str,
    script: &Path,
    python: &str,
) -> Command {
    let mut cmd = Command::new(python);
    cmd.arg(script)
        .arg(model_dir)
        .arg("--outfile")
        .arg(outfile)
        .arg("--outtype")
        .arg(outtype);
    cmd
}

/// Builds `<bin> <input> <output> <qtype>`.
pub fn quantize_cmd(input: &Path, output: &Path, qtype: &str, bin: &str) -> Command {
    let mut cmd = Command::new(bin);
    cmd.arg(input).arg(output).arg(qtype);
    cmd
}

/// Renders the program and its arguments joined by single spaces.
pub fn render(cmd: &Command) -> String {
    let mut parts = vec![cmd.get_program().to_string_lossy().into_owned()];
    parts.extend(cmd.get_args().map(|a| a.to_string_lossy().into_owned()));
    parts.join(" ")
}

/// Rejects any `--outtype` value that is not in [`ALLOWED_OUTTYPES`].
pub fn validate_outtype(value: &str) -> Result<()> {
    if ALLOWED_OUTTYPES.contains(&value) {
        Ok(())
    } else {
        bail!(
            "invalid --outtype '{value}'; allowed values: {}",
            ALLOWED_OUTTYPES.join(", ")
        )
    }
}

/// Rejects any `--type` value that is not in [`ALLOWED_QTYPES`].
pub fn validate_qtype(value: &str) -> Result<()> {
    if ALLOWED_QTYPES.contains(&value) {
        Ok(())
    } else {
        bail!(
            "invalid --type '{value}'; allowed values: {}",
            ALLOWED_QTYPES.join(", ")
        )
    }
}

/// In dry-run mode prints `DRY-RUN: <command>` to stdout and returns 0.
/// Otherwise runs the command with inherited stdio and returns its exit code.
/// A missing program yields an error whose message contains `not found`.
pub fn run(mut cmd: Command, dry_run: bool) -> Result<i32> {
    if dry_run {
        println!("DRY-RUN: {}", render(&cmd));
        return Ok(0);
    }

    let program = cmd.get_program().to_string_lossy().into_owned();
    cmd.stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    match cmd.status() {
        Ok(status) => Ok(status.code().unwrap_or(1)),
        Err(e) if e.kind() == ErrorKind::NotFound => bail!(
            "'{program}' not found ({e}); install llama.cpp or set the matching flag or env var \
             (--bin / LLAMA_QUANTIZE for quantize, --script / LLAMA_CPP_DIR for convert, --python for convert)"
        ),
        Err(e) => Err(e).with_context(|| format!("failed to run '{program}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn render_convert_command() {
        let cmd = convert_cmd(
            &PathBuf::from("m"),
            &PathBuf::from("o.gguf"),
            "f16",
            &PathBuf::from("/s/convert_hf_to_gguf.py"),
            "python3",
        );
        assert_eq!(
            render(&cmd),
            "python3 /s/convert_hf_to_gguf.py m --outfile o.gguf --outtype f16"
        );
    }

    #[test]
    fn render_quantize_command() {
        let cmd = quantize_cmd(
            &PathBuf::from("in.gguf"),
            &PathBuf::from("out.gguf"),
            "Q5_K_M",
            "/x/llama-quantize",
        );
        assert_eq!(render(&cmd), "/x/llama-quantize in.gguf out.gguf Q5_K_M");
    }

    #[test]
    fn rejects_unknown_qtype_and_lists_allowed() {
        let err = validate_qtype("Q9").unwrap_err().to_string();
        assert!(err.contains("Q9"));
        for q in ALLOWED_QTYPES {
            assert!(err.contains(q), "message should list {q}: {err}");
        }
        assert!(validate_qtype("q4_k_m").is_err());
    }

    #[test]
    fn accepts_all_allowed_qtypes() {
        for q in ALLOWED_QTYPES {
            assert!(validate_qtype(q).is_ok(), "{q} should be accepted");
        }
    }

    #[test]
    fn rejects_unknown_outtype_and_lists_allowed() {
        let err = validate_outtype("q4_0").unwrap_err().to_string();
        for t in ALLOWED_OUTTYPES {
            assert!(err.contains(t), "message should list {t}: {err}");
        }
    }

    #[test]
    fn accepts_all_allowed_outtypes() {
        for t in ALLOWED_OUTTYPES {
            assert!(validate_outtype(t).is_ok(), "{t} should be accepted");
        }
    }
}
