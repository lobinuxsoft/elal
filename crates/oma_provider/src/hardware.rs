//! Hardware introspection for the local machine.
//!
//! Currently scoped to GPU VRAM detection on Linux via DRM sysfs nodes.
//! AMD GPUs expose `mem_info_vram_total` / `mem_info_vram_used` under
//! `/sys/class/drm/card*/device/`. NVIDIA GPUs do not — for those we'd
//! need to shell out to `nvidia-smi`. Out of scope here (project targets
//! AMD RDNA on Bazzite).
//!
//! All probes are best-effort: sysfs may be missing in a container, the
//! card number changes between machines, and a non-AMD GPU returns
//! nothing. Callers receive `None` and pick a conservative fallback.

use std::fs;
use std::path::Path;

/// VRAM totals for a single GPU, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramInfo {
    pub total_bytes: u64,
    pub used_bytes: u64,
}

impl VramInfo {
    pub fn free_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.used_bytes)
    }
}

/// Probe the largest DRM card that exposes AMD VRAM counters.
///
/// On a single-GPU system this returns that GPU. On multi-GPU systems
/// it returns the one with the largest VRAM, which is a reasonable
/// proxy for "the discrete GPU we plan to offload onto" without
/// per-vendor heuristics.
///
/// Returns `None` when no DRM card exposes the AMD-specific sysfs nodes
/// (NVIDIA-only systems, virtualised guests without sysfs, non-Linux).
pub fn detect_primary_gpu_vram() -> Option<VramInfo> {
    let drm = Path::new("/sys/class/drm");
    if !drm.is_dir() {
        return None;
    }

    let mut best: Option<VramInfo> = None;
    for entry in fs::read_dir(drm).ok()?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // We only want top-level cards (`card0`, `card1`, …) — skip
        // `cardN-DP-1` and similar connector-level entries.
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let device = entry.path().join("device");
        let info = read_amd_vram(&device);
        if let Some(info) = info
            && best
                .map(|b| info.total_bytes > b.total_bytes)
                .unwrap_or(true)
        {
            best = Some(info);
        }
    }
    best
}

/// Read AMD-specific VRAM sysfs nodes from a `card*/device` directory.
fn read_amd_vram(device: &Path) -> Option<VramInfo> {
    let total = read_u64(&device.join("mem_info_vram_total"))?;
    let used = read_u64(&device.join("mem_info_vram_used")).unwrap_or(0);
    if total == 0 {
        return None;
    }
    Some(VramInfo {
        total_bytes: total,
        used_bytes: used,
    })
}

fn read_u64(path: &Path) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vram_free_saturates_when_used_exceeds_total() {
        let info = VramInfo {
            total_bytes: 100,
            used_bytes: 200,
        };
        assert_eq!(info.free_bytes(), 0);
    }

    #[test]
    fn vram_free_normal_case() {
        let info = VramInfo {
            total_bytes: 16 * 1024 * 1024 * 1024,
            used_bytes: 4 * 1024 * 1024 * 1024,
        };
        assert_eq!(info.free_bytes(), 12 * 1024 * 1024 * 1024);
    }

    /// Smoke: probe should not panic on any host. On non-AMD or non-Linux
    /// systems we accept `None`; on the developer's box we expect `Some`.
    #[test]
    fn detect_does_not_panic() {
        let _ = detect_primary_gpu_vram();
    }
}
