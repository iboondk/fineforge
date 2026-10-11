use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::rng::shuffle;

#[derive(Debug, Clone, Serialize)]
pub struct ShardInfo {
    pub file: String,
    pub count: usize,
    pub sha256: String,
}

#[derive(Debug, Serialize)]
pub struct Manifest {
    pub seed: u64,
    pub eval_ratio: f64,
    pub total: usize,
    pub eval_count: usize,
    pub train_shards: Vec<ShardInfo>,
    pub eval: ShardInfo,
}

/// Deterministically shuffles `lines` with `seed`, splits off the first
/// `floor(len * eval_ratio)` as `eval.jsonl`, and deals the rest round-robin into
/// `num_shards` train files. Writes `manifest.json` next to them.
pub fn shard(
    mut lines: Vec<String>,
    out_dir: &Path,
    num_shards: usize,
    eval_ratio: f64,
    seed: u64,
) -> Result<Manifest> {
    if num_shards == 0 {
        bail!("num_shards must be at least 1");
    }
    if !(0.0..1.0).contains(&eval_ratio) {
        bail!("eval_ratio must be in [0, 1), got {eval_ratio}");
    }

    fs::create_dir_all(out_dir).with_context(|| format!("cannot create {}", out_dir.display()))?;

    shuffle(&mut lines, seed);

    let total = lines.len();
    let eval_count = (total as f64 * eval_ratio).floor() as usize;
    let (eval_lines, train_lines) = lines.split_at(eval_count);

    let eval = write_shard(out_dir, "eval.jsonl", eval_lines.iter().map(String::as_str))?;

    let mut buckets: Vec<Vec<&str>> = vec![Vec::new(); num_shards];
    for (i, line) in train_lines.iter().enumerate() {
        buckets[i % num_shards].push(line.as_str());
    }

    let mut train_shards = Vec::with_capacity(num_shards);
    for (i, bucket) in buckets.into_iter().enumerate() {
        let name = format!("train-{i:05}.jsonl");
        train_shards.push(write_shard(out_dir, &name, bucket.into_iter())?);
    }

    let manifest = Manifest {
        seed,
        eval_ratio,
        total,
        eval_count,
        train_shards,
        eval,
    };

    let json = serde_json::to_string_pretty(&manifest).context("cannot serialize manifest")?;
    let manifest_path = out_dir.join("manifest.json");
    fs::write(&manifest_path, format!("{json}\n"))
        .with_context(|| format!("cannot write {}", manifest_path.display()))?;

    Ok(manifest)
}

fn write_shard<'a>(
    dir: &Path,
    name: &str,
    lines: impl Iterator<Item = &'a str>,
) -> Result<ShardInfo> {
    let mut bytes: Vec<u8> = Vec::new();
    let mut count = 0usize;
    for line in lines {
        bytes.extend_from_slice(line.as_bytes());
        bytes.push(b'\n');
        count += 1;
    }

    let path = dir.join(name);
    fs::write(&path, &bytes).with_context(|| format!("cannot write {}", path.display()))?;

    Ok(ShardInfo {
        file: name.to_string(),
        count,
        sha256: hex::encode(Sha256::digest(&bytes)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| format!(r#"{{"instruction":"q{i}","output":"a{i}"}}"#))
            .collect()
    }

    #[test]
    fn counts_sum_to_total() {
        let dir = tempfile::tempdir().unwrap();
        let m = shard(sample(10), dir.path(), 3, 0.2, 1).unwrap();
        assert_eq!(m.total, 10);
        assert_eq!(m.eval_count, 2);
        assert_eq!(m.eval.count, 2);
        let train: usize = m.train_shards.iter().map(|s| s.count).sum();
        assert_eq!(train, 8);
        assert!(dir.path().join("manifest.json").exists());
        assert!(dir.path().join("train-00002.jsonl").exists());
    }

    #[test]
    fn rejects_zero_shards_and_bad_ratio() {
        let dir = tempfile::tempdir().unwrap();
        assert!(shard(sample(3), dir.path(), 0, 0.1, 1).is_err());
        assert!(shard(sample(3), dir.path(), 1, 1.0, 1).is_err());
        assert!(shard(sample(3), dir.path(), 1, -0.1, 1).is_err());
    }

    #[test]
    fn same_seed_gives_same_hashes() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let ma = shard(sample(20), a.path(), 2, 0.25, 9).unwrap();
        let mb = shard(sample(20), b.path(), 2, 0.25, 9).unwrap();
        assert_eq!(ma.eval.sha256, mb.eval.sha256);
        assert_eq!(ma.train_shards[1].sha256, mb.train_shards[1].sha256);
    }
}
