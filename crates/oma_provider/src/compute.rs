//! Informational enum describing which compute backend this build activated.
//!
//! Backend selection is a compile-time decision made via feature flags
//! (`vulkan` default, `rocm`, `dual`). This enum exists for logs, the
//! `doctor` subcommand, and anywhere user-facing diagnostics report the
//! active backend. There is no runtime auto-detection (ADR-1, ADR-7).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    Vulkan,
    Rocm,
    /// Both backends compiled; the active one is decided at model-load time
    /// by `llama-cpp-2` itself based on which GPU library loads first.
    Dual,
    /// Neither feature enabled (unusual — we should have at least `vulkan`).
    Cpu,
}

impl ComputeBackend {
    pub const fn compiled() -> Self {
        #[cfg(feature = "dual")]
        {
            Self::Dual
        }
        #[cfg(all(feature = "rocm", not(feature = "dual")))]
        {
            Self::Rocm
        }
        #[cfg(all(feature = "vulkan", not(feature = "rocm"), not(feature = "dual")))]
        {
            Self::Vulkan
        }
        #[cfg(not(any(feature = "vulkan", feature = "rocm", feature = "dual")))]
        {
            Self::Cpu
        }
    }

    pub const fn describe(&self) -> &'static str {
        match self {
            Self::Vulkan => "Vulkan (mesa radv)",
            Self::Rocm => "ROCm (HIP)",
            Self::Dual => "Vulkan + ROCm (selected at runtime)",
            Self::Cpu => "CPU only (no GPU feature enabled)",
        }
    }
}

impl std::fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Vulkan => "vulkan",
            Self::Rocm => "rocm",
            Self::Dual => "dual",
            Self::Cpu => "cpu",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_matches_feature() {
        let backend = ComputeBackend::compiled();
        #[cfg(feature = "dual")]
        assert_eq!(backend, ComputeBackend::Dual);
        #[cfg(all(feature = "rocm", not(feature = "dual")))]
        assert_eq!(backend, ComputeBackend::Rocm);
        #[cfg(all(feature = "vulkan", not(feature = "rocm"), not(feature = "dual")))]
        assert_eq!(backend, ComputeBackend::Vulkan);
    }

    #[test]
    fn describe_is_non_empty() {
        for b in [
            ComputeBackend::Vulkan,
            ComputeBackend::Rocm,
            ComputeBackend::Dual,
            ComputeBackend::Cpu,
        ] {
            assert!(!b.describe().is_empty());
        }
    }
}
