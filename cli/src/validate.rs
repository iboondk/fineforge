use std::io::BufRead;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::record::parse_line;

#[derive(Debug, Serialize)]
pub struct ValidateReport {
    pub total: usize,
    pub valid: usize,
    pub invalid: usize,
    pub errors: Vec<LineError>,
}

#[derive(Debug, Serialize)]
pub struct LineError {
    /// 1-based physical line number in the input file.
    pub line: usize,
    pub reason: String,
}

/// Validates every non-blank line of a JSONL stream.
///
/// Blank lines (after trim) are skipped and not counted. `errors` keeps at most
/// `max_errors` entries, but `invalid` counts every failing line.
pub fn validate_reader<R: BufRead>(r: R, max_errors: usize) -> Result<ValidateReport> {
    let mut report = ValidateReport {
        total: 0,
        valid: 0,
        invalid: 0,
        errors: Vec::new(),
    };

    for (idx, line) in r.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read line {}", idx + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        report.total += 1;

        match parse_line(&line) {
            Ok(_) => report.valid += 1,
            Err(reason) => {
                report.invalid += 1;
                if report.errors.len() < max_errors {
                    report.errors.push(LineError {
                        line: idx + 1,
                        reason,
                    });
                }
            }
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const CHAT: &str =
        r#"{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}"#;
    const BAD: &str = r#"{"instruction":"x","output":""}"#;

    fn run(input: &str, max: usize) -> ValidateReport {
        validate_reader(Cursor::new(input.as_bytes()), max).unwrap()
    }

    #[test]
    fn counts_valid_and_invalid() {
        let input = format!("{CHAT}\n{BAD}\n{CHAT}\n");
        let rep = run(&input, 100);
        assert_eq!((rep.total, rep.valid, rep.invalid), (3, 2, 1));
        assert_eq!(rep.errors.len(), 1);
        assert_eq!(rep.errors[0].line, 2);
        assert_eq!(rep.errors[0].reason, "output must be a non-empty string");
    }

    #[test]
    fn blank_lines_are_skipped_but_keep_line_numbers() {
        let input = format!("\n   \n{CHAT}\n\t\n{BAD}\n");
        let rep = run(&input, 100);
        assert_eq!((rep.total, rep.valid, rep.invalid), (2, 1, 1));
        assert_eq!(rep.errors[0].line, 5);
    }

    #[test]
    fn errors_are_truncated_but_invalid_counts_all() {
        let input = format!("{BAD}\n{BAD}\n{BAD}\n");
        let rep = run(&input, 2);
        assert_eq!(rep.invalid, 3);
        assert_eq!(rep.errors.len(), 2);
        assert_eq!(rep.errors[1].line, 2);
    }

    #[test]
    fn zero_max_errors_keeps_no_entries() {
        let rep = run(BAD, 0);
        assert_eq!(rep.invalid, 1);
        assert!(rep.errors.is_empty());
    }

    #[test]
    fn empty_input_is_all_zero() {
        let rep = run("", 10);
        assert_eq!((rep.total, rep.valid, rep.invalid), (0, 0, 0));
    }

    #[test]
    fn report_serializes_to_json() {
        let rep = run(BAD, 10);
        let v = serde_json::to_value(&rep).unwrap();
        assert_eq!(v["invalid"], 1);
        assert_eq!(v["errors"][0]["line"], 1);
    }
}
