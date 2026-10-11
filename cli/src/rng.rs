/// SplitMix64 pseudo-random generator. Deterministic for a given seed, which is
/// what makes `shard` output byte-identical across runs.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns a value in `0..n`. Panics if `n == 0`.
    pub fn gen_range(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// In-place Fisher-Yates shuffle driven by SplitMix64 with the given seed.
pub fn shuffle<T>(v: &mut [T], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for i in (1..v.len()).rev() {
        let j = rng.gen_range(i as u64 + 1) as usize;
        v.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_seed_zero_matches_reference() {
        assert_eq!(SplitMix64::new(0).next_u64(), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn shuffle_same_seed_is_identical() {
        let mut a: Vec<u32> = (0..100).collect();
        let mut b: Vec<u32> = (0..100).collect();
        shuffle(&mut a, 7);
        shuffle(&mut b, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut v: Vec<u32> = (0..100).collect();
        shuffle(&mut v, 123);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<u32>>());
        assert_ne!(v, (0..100).collect::<Vec<u32>>());
    }

    #[test]
    fn gen_range_stays_in_bounds() {
        let mut rng = SplitMix64::new(99);
        for _ in 0..1000 {
            assert!(rng.gen_range(5) < 5);
        }
    }
}
