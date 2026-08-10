use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SamplingControls {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub seed: u32,
}

impl Default for SamplingControls {
    fn default() -> Self {
        // Conservative defaults that work well for most modern chat models.
        Self {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            repeat_penalty: 1.1,
            seed: 0xC0DE_C0DE,
        }
    }
}

impl SamplingControls {
    /// Deterministic greedy sampling (temperature ~0, fixed seed).
    pub fn deterministic() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            min_p: 0.0,
            repeat_penalty: 1.0,
            seed: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let s = SamplingControls::default();
        assert!(s.temperature > 0.0);
        assert!(s.top_p > 0.0 && s.top_p <= 1.0);
        assert!(s.top_k > 0);
    }

    #[test]
    fn deterministic_is_greedy() {
        let s = SamplingControls::deterministic();
        assert_eq!(s.temperature, 0.0);
        assert_eq!(s.top_k, 1);
    }
}
