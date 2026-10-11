use std::collections::HashSet;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::record::{canonical_text, normalize, parse_line, Record};

/// Word-shingle length used for near-duplicate detection.
const SHINGLE_K: usize = 5;
/// Number of MinHash functions (signature length).
const MINHASH_NUM: usize = 64;

/// Hex SHA-256 of the normalized canonical text. Two records with the same key
/// are exact duplicates.
pub fn exact_key(r: &Record) -> String {
    let digest = Sha256::digest(normalize(&canonical_text(r)).as_bytes());
    hex::encode(digest)
}

/// Word k-grams of already-normalized text, each joined by a single space. If
/// the text has fewer than `k` words, the single shingle is the whole text.
pub fn shingles(text: &str, k: usize) -> HashSet<String> {
    let k = k.max(1);
    let words: Vec<&str> = text.split(' ').filter(|w| !w.is_empty()).collect();
    let mut out = HashSet::new();
    if words.len() < k {
        out.insert(text.to_string());
    } else {
        for window in words.windows(k) {
            out.insert(window.join(" "));
        }
    }
    out
}

/// First 8 bytes (big-endian) of SHA-256(seed as little-endian bytes ++ shingle).
fn hash64(seed: u64, shingle: &str) -> u64 {
    let mut h = Sha256::new();
    h.update(seed.to_le_bytes());
    h.update(shingle.as_bytes());
    let d = h.finalize();
    u64::from_be_bytes(d[..8].try_into().expect("sha256 output is 32 bytes"))
}

/// MinHash signature: for each seed `i` in `0..num`, the minimum hash over all
/// shingles. An empty set yields `num` copies of `u64::MAX`.
pub fn minhash(sh: &HashSet<String>, num: usize) -> Vec<u64> {
    if sh.is_empty() {
        return vec![u64::MAX; num];
    }
    (0..num as u64)
        .map(|i| sh.iter().map(|s| hash64(i, s)).min().unwrap_or(u64::MAX))
        .collect()
}

/// Estimated Jaccard similarity: the fraction of equal signature positions.
pub fn jaccard_est(a: &[u64], b: &[u64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let equal = a
        .iter()
        .zip(b.iter())
        .take(n)
        .filter(|(x, y)| x == y)
        .count();
    equal as f64 / n as f64
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct DedupeStats {
    pub total: usize,
    pub kept: usize,
    pub exact_dupes: usize,
    pub near_dupes: usize,
    pub invalid_skipped: usize,
}

/// Keeps the first occurrence of each record. Blank lines are ignored, invalid
/// lines are skipped and counted, and kept lines are returned unchanged.
///
/// When `near` is `Some(t)`, a record whose MinHash estimate reaches `t`
/// against any kept record is dropped as a near duplicate.
pub fn dedupe_lines(
    lines: impl Iterator<Item = String>,
    near: Option<f64>,
) -> (Vec<String>, DedupeStats) {
    let mut kept: Vec<String> = Vec::new();
    let mut stats = DedupeStats::default();
    let mut seen_exact: HashSet<String> = HashSet::new();
    let mut signatures: Vec<Vec<u64>> = Vec::new();

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        stats.total += 1;

        let record = match parse_line(&line) {
            Ok((record, _)) => record,
            Err(_) => {
                stats.invalid_skipped += 1;
                continue;
            }
        };

        let key = exact_key(&record);
        if seen_exact.contains(&key) {
            stats.exact_dupes += 1;
            continue;
        }

        if let Some(threshold) = near {
            let sh = shingles(&normalize(&canonical_text(&record)), SHINGLE_K);
            let sig = minhash(&sh, MINHASH_NUM);
            if signatures
                .iter()
                .any(|kept_sig| jaccard_est(kept_sig, &sig) >= threshold)
            {
                stats.near_dupes += 1;
                continue;
            }
            signatures.push(sig);
        }

        seen_exact.insert(key);
        kept.push(line);
        stats.kept += 1;
    }

    (kept, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = r#"{"instruction":"Translate","input":"Hi","output":"Salut"}"#;
    const A_LOOSE: &str = r#"{"instruction":"  TRANSLATE ","input":"hi","output":"salut  "}"#;

    #[test]
    fn exact_key_ignores_case_and_whitespace() {
        let (r1, _) = parse_line(A).unwrap();
        let (r2, _) = parse_line(A_LOOSE).unwrap();
        assert_eq!(exact_key(&r1), exact_key(&r2));
        assert_eq!(exact_key(&r1).len(), 64);
    }

    #[test]
    fn shingles_are_word_kgrams() {
        let sh = shingles("a b c d e f", 5);
        assert_eq!(sh.len(), 2);
        assert!(sh.contains("a b c d e"));
        assert!(sh.contains("b c d e f"));
    }

    #[test]
    fn short_text_is_single_shingle() {
        let sh = shingles("only three words", 5);
        assert_eq!(sh.len(), 1);
        assert!(sh.contains("only three words"));
    }

    #[test]
    fn minhash_is_deterministic_and_empty_is_max() {
        let sh = shingles("a b c d e f g", 5);
        assert_eq!(minhash(&sh, 8), minhash(&sh, 8));
        assert_eq!(minhash(&HashSet::new(), 4), vec![u64::MAX; 4]);
    }

    #[test]
    fn identical_sets_have_jaccard_one() {
        let sig = minhash(&shingles("a b c d e f g h", 5), 64);
        assert_eq!(jaccard_est(&sig, &sig), 1.0);
    }

    #[test]
    fn dedupe_drops_exact_and_invalid() {
        let input = vec![
            A.to_string(),
            A_LOOSE.to_string(),
            r#"{"instruction":"x","output":""}"#.to_string(),
        ];
        let (kept, stats) = dedupe_lines(input.into_iter(), None);
        assert_eq!(kept, vec![A.to_string()]);
        assert_eq!(
            (
                stats.total,
                stats.kept,
                stats.exact_dupes,
                stats.invalid_skipped
            ),
            (3, 1, 1, 1)
        );
        assert_eq!(stats.near_dupes, 0);
    }
}
