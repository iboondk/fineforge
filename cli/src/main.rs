mod dedupe;
mod gguf;
mod record;
mod rng;
mod shard;
mod validate;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process;

use clap::{Parser, Subcommand};

use crate::dedupe::{dedupe_lines, DedupeStats};
use crate::record::parse_line;
use crate::shard::shard;
use crate::validate::{validate_reader, ValidateReport};

#[derive(Parser)]
#[command(
    name = "fineforge",
    version,
    about = "Dataset tooling for LLM fine-tuning"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate a JSONL dataset against the record contract.
    Validate {
        /// Path to the input JSONL file.
        input: PathBuf,
        /// Write the full report as pretty JSON to this path.
        #[arg(long)]
        report: Option<PathBuf>,
        /// Maximum number of per-line errors kept in the report.
        #[arg(long, default_value_t = 100)]
        max_errors: usize,
    },
    /// Drop exact duplicates, and optionally near duplicates, from a JSONL dataset.
    Dedupe {
        /// Path to the input JSONL file.
        input: PathBuf,
        /// Path to write the deduplicated JSONL file.
        #[arg(short = 'o', long)]
        output: PathBuf,
        /// Jaccard threshold in [0, 1] for MinHash near-duplicate removal.
        #[arg(long)]
        near: Option<f64>,
        /// Write the statistics as pretty JSON to this path.
        #[arg(long)]
        stats: Option<PathBuf>,
    },
    /// Deterministically shuffle, split off an eval set, and write train shards.
    Shard {
        /// Path to the input JSONL file.
        input: PathBuf,
        /// Output directory (created if missing).
        #[arg(long = "out-dir")]
        out_dir: PathBuf,
        /// Number of train shards.
        #[arg(long, default_value_t = 1)]
        shards: usize,
        /// Fraction of records placed in eval.jsonl, in [0, 1).
        #[arg(long = "eval-ratio", default_value_t = 0.1)]
        eval_ratio: f64,
        /// Shuffle seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Wrap llama.cpp GGUF convert and quantize steps.
    Gguf {
        #[command(subcommand)]
        action: GgufAction,
    },
}

#[derive(Subcommand)]
enum GgufAction {
    /// Convert a Hugging Face model directory to GGUF with convert_hf_to_gguf.py.
    Convert {
        /// Hugging Face model directory.
        model_dir: PathBuf,
        /// Output GGUF file path.
        #[arg(long)]
        outfile: PathBuf,
        /// Output type: f32, f16, bf16, q8_0 or auto.
        #[arg(long, default_value = "f16")]
        outtype: String,
        /// Path to convert_hf_to_gguf.py (default: $LLAMA_CPP_DIR/convert_hf_to_gguf.py).
        #[arg(long)]
        script: Option<PathBuf>,
        /// Python interpreter used to run the script.
        #[arg(long, default_value = "python3")]
        python: String,
        /// Print the command instead of running it.
        #[arg(long)]
        dry_run: bool,
    },
    /// Quantize a GGUF file with llama-quantize.
    Quantize {
        /// Input GGUF file.
        input: PathBuf,
        /// Output GGUF file.
        output: PathBuf,
        /// Quantization type: Q4_0, Q4_K_M, Q5_K_M, Q6_K or Q8_0.
        #[arg(long = "type", default_value = "Q4_K_M")]
        qtype: String,
        /// Path to llama-quantize (default: $LLAMA_QUANTIZE, then llama-quantize on PATH).
        #[arg(long)]
        bin: Option<String>,
        /// Print the command instead of running it.
        #[arg(long)]
        dry_run: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    let code = match cli.command {
        Command::Gguf { action } => run_gguf(action),
        Command::Validate {
            input,
            report,
            max_errors,
        } => run_validate(&input, report.as_deref(), max_errors),
        Command::Dedupe {
            input,
            output,
            near,
            stats,
        } => run_dedupe(&input, &output, near, stats.as_deref()),
        Command::Shard {
            input,
            out_dir,
            shards,
            eval_ratio,
            seed,
        } => run_shard(&input, &out_dir, shards, eval_ratio, seed),
    };
    process::exit(code);
}

/// Returns the child's exit code on a real run or dry-run success (0). Returns
/// 2 on usage errors, missing inputs, or a missing external tool.
fn run_gguf(action: GgufAction) -> i32 {
    let result = match action {
        GgufAction::Convert {
            model_dir,
            outfile,
            outtype,
            script,
            python,
            dry_run,
        } => gguf_convert(&model_dir, &outfile, &outtype, script, &python, dry_run),
        GgufAction::Quantize {
            input,
            output,
            qtype,
            bin,
            dry_run,
        } => gguf_quantize(&input, &output, &qtype, bin, dry_run),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            2
        }
    }
}

fn gguf_convert(
    model_dir: &Path,
    outfile: &Path,
    outtype: &str,
    script: Option<PathBuf>,
    python: &str,
    dry_run: bool,
) -> anyhow::Result<i32> {
    gguf::validate_outtype(outtype)?;
    let script = resolve_convert_script(script)?;
    if !dry_run && !model_dir.exists() {
        anyhow::bail!("model directory {} does not exist", model_dir.display());
    }
    let cmd = gguf::convert_cmd(model_dir, outfile, outtype, &script, python);
    gguf::run(cmd, dry_run)
}

fn resolve_convert_script(flag: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    if let Some(path) = flag {
        return Ok(path);
    }
    match std::env::var_os("LLAMA_CPP_DIR") {
        Some(dir) if !dir.is_empty() => Ok(PathBuf::from(dir).join("convert_hf_to_gguf.py")),
        _ => anyhow::bail!("set --script or LLAMA_CPP_DIR to locate convert_hf_to_gguf.py"),
    }
}

fn gguf_quantize(
    input: &Path,
    output: &Path,
    qtype: &str,
    bin: Option<String>,
    dry_run: bool,
) -> anyhow::Result<i32> {
    gguf::validate_qtype(qtype)?;
    let bin = bin
        .or_else(|| {
            std::env::var("LLAMA_QUANTIZE")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "llama-quantize".to_string());
    if !dry_run && !input.exists() {
        anyhow::bail!("input file {} does not exist", input.display());
    }
    let cmd = gguf::quantize_cmd(input, output, qtype, &bin);
    gguf::run(cmd, dry_run)
}

/// Returns the process exit code: 0 when all lines are valid, 1 when any line
/// is invalid, 2 on I/O error.
fn run_validate(input: &Path, report_path: Option<&Path>, max_errors: usize) -> i32 {
    let file = match File::open(input) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: cannot open {}: {e}", input.display());
            return 2;
        }
    };

    let report: ValidateReport = match validate_reader(BufReader::new(file), max_errors) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e:#}");
            return 2;
        }
    };

    println!(
        "total={} valid={} invalid={}",
        report.total, report.valid, report.invalid
    );

    if let Some(path) = report_path {
        let json = match serde_json::to_string_pretty(&report) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("error: cannot serialize report: {e}");
                return 2;
            }
        };
        if let Err(e) = std::fs::write(path, format!("{json}\n")) {
            eprintln!("error: cannot write report {}: {e}", path.display());
            return 2;
        }
    }

    if report.invalid == 0 {
        0
    } else {
        1
    }
}

/// Returns 0 on success, 2 on usage or I/O error.
fn run_dedupe(input: &Path, output: &Path, near: Option<f64>, stats_path: Option<&Path>) -> i32 {
    if let Some(t) = near {
        if !(0.0..=1.0).contains(&t) {
            eprintln!("error: --near must be between 0 and 1, got {t}");
            return 2;
        }
    }

    let file = match File::open(input) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: cannot open {}: {e}", input.display());
            return 2;
        }
    };

    let lines: Vec<String> = match BufReader::new(file).lines().collect::<Result<_, _>>() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: cannot read {}: {e}", input.display());
            return 2;
        }
    };

    let (kept, stats): (Vec<String>, DedupeStats) = dedupe_lines(lines.into_iter(), near);

    let mut body = String::new();
    for line in &kept {
        body.push_str(line);
        body.push('\n');
    }
    if let Err(e) = std::fs::write(output, body) {
        eprintln!("error: cannot write {}: {e}", output.display());
        return 2;
    }

    println!(
        "total={} kept={} exact_dupes={} near_dupes={} invalid_skipped={}",
        stats.total, stats.kept, stats.exact_dupes, stats.near_dupes, stats.invalid_skipped
    );

    if let Some(path) = stats_path {
        let json = match serde_json::to_string_pretty(&stats) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("error: cannot serialize stats: {e}");
                return 2;
            }
        };
        if let Err(e) = std::fs::write(path, format!("{json}\n")) {
            eprintln!("error: cannot write stats {}: {e}", path.display());
            return 2;
        }
    }

    0
}

/// Returns 0 on success, 1 when an input line is an invalid record, 2 on I/O or
/// parameter error.
fn run_shard(input: &Path, out_dir: &Path, shards: usize, eval_ratio: f64, seed: u64) -> i32 {
    let file = match File::open(input) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: cannot open {}: {e}", input.display());
            return 2;
        }
    };

    let mut lines: Vec<String> = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("error: cannot read line {}: {e}", idx + 1);
                return 2;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        if let Err(reason) = parse_line(&line) {
            eprintln!("error: invalid record at line {}: {reason}", idx + 1);
            return 1;
        }
        lines.push(line);
    }

    match shard(lines, out_dir, shards, eval_ratio, seed) {
        Ok(manifest) => {
            println!(
                "total={} eval={} train_shards={} out_dir={}",
                manifest.total,
                manifest.eval_count,
                manifest.train_shards.len(),
                out_dir.display()
            );
            0
        }
        Err(e) => {
            eprintln!("error: {e:#}");
            2
        }
    }
}
