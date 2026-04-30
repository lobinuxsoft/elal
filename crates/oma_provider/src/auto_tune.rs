//! Automatic `n_ctx` sizing based on available VRAM and model architecture.
//!
//! The KV cache scales linearly with context length. A 30B model with
//! `n_ctx_train = 262144` will happily ask llama.cpp for ~24 GiB of KV
//! buffer if we accept the model's native default — and that exceeds
//! consumer GPU VRAM, forcing a spillover into RAM that ends in OOM.
//!
//! This module picks a `n_ctx` that the configured GPU can actually
//! hold, biased conservatively (round down to a power of two, leave a
//! safety margin for compute buffers, cap at a sane upper bound).

use llama_cpp_2::model::LlamaModel;

use crate::hardware::VramInfo;

/// Hard upper bound for auto-tuned context. Models advertise up to 256K
/// tokens, but agent prompts rarely benefit beyond 32K and KV cache cost
/// scales linearly. Users who genuinely need more pass `--n-ctx`.
pub const AUTO_TUNE_MAX_CTX: u32 = 32_768;

/// Hard lower bound. Below this, agent prompts (system + tools + first
/// user message) start to truncate.
pub const AUTO_TUNE_MIN_CTX: u32 = 2_048;

/// Headroom reserved on top of the KV cache for compute buffers,
/// scratch tensors, and llama.cpp internal state. Conservative.
const VRAM_SAFETY_MARGIN_BYTES: u64 = 1024 * 1024 * 1024; // 1 GiB

/// Bytes per element in the KV cache. llama.cpp defaults to fp16 (2 B)
/// for both K and V tensors, hence 2 × 2 = 4.
const KV_BYTES_PER_ELEMENT: u64 = 4;

#[derive(Debug, Clone)]
pub struct AutoTuneResult {
    /// Final `n_ctx` to apply.
    pub n_ctx: u32,
    /// One-line human-readable reason — surfaced in CLI logs.
    pub reason: String,
    /// `true` if the user-requested override was honoured (no auto-tune).
    pub user_override: bool,
}

/// Choose an `n_ctx` for the given model and hardware.
///
/// - `requested == 0` → auto-tune from VRAM and model architecture.
/// - `requested > 0`  → respect the override, but cap at `n_ctx_train`.
pub fn auto_tune_n_ctx(
    model: &LlamaModel,
    requested: u32,
    vram: Option<VramInfo>,
) -> AutoTuneResult {
    let native = model.n_ctx_train();

    if requested != 0 {
        let n_ctx = requested.min(native);
        let reason = if n_ctx == requested {
            format!("user override (n_ctx={requested})")
        } else {
            format!("user override clamped to n_ctx_train ({native})")
        };
        return AutoTuneResult {
            n_ctx,
            reason,
            user_override: true,
        };
    }

    let kv_per_token = kv_cache_bytes_per_token(model);

    let from_vram = vram.and_then(|v| max_ctx_for_vram(v.free_bytes(), kv_per_token));

    let candidate = match from_vram {
        Some(c) => c.min(native).min(AUTO_TUNE_MAX_CTX),
        None => AUTO_TUNE_MIN_CTX.min(native),
    };

    let n_ctx = round_down_pow2(candidate.max(AUTO_TUNE_MIN_CTX));

    let reason = match (vram, from_vram) {
        (Some(v), Some(_)) => {
            let free_mib = v.free_bytes() / (1024 * 1024);
            let kv_kib = kv_per_token / 1024;
            format!(
                "auto-tuned: {free_mib} MiB free VRAM, {kv_kib} KiB/token KV, capped at {AUTO_TUNE_MAX_CTX}",
            )
        }
        _ => format!("auto-tuned fallback: VRAM probe unavailable, defaulting to {n_ctx}"),
    };

    AutoTuneResult {
        n_ctx,
        reason,
        user_override: false,
    }
}

/// Compute KV cache cost per token in bytes, derived from model
/// architecture metadata exposed by llama-cpp-2.
///
/// Formula: `n_layer × n_head_kv × head_dim × bytes_per_element`
/// where `head_dim = n_embd / n_head` and `bytes_per_element = 4`
/// (fp16 K + fp16 V).
fn kv_cache_bytes_per_token(model: &LlamaModel) -> u64 {
    let n_layer = u64::from(model.n_layer());
    let n_head_kv = u64::from(model.n_head_kv());
    let n_head = u64::from(model.n_head().max(1));
    let n_embd = u64::try_from(model.n_embd().max(1)).unwrap_or(1);
    let head_dim = n_embd / n_head;
    n_layer * n_head_kv * head_dim * KV_BYTES_PER_ELEMENT
}

/// Largest `n_ctx` that fits in `free_bytes` after subtracting the
/// safety margin. Returns `None` if there isn't even room for the
/// minimum context.
fn max_ctx_for_vram(free_bytes: u64, kv_per_token: u64) -> Option<u32> {
    if kv_per_token == 0 {
        return None;
    }
    let usable = free_bytes.saturating_sub(VRAM_SAFETY_MARGIN_BYTES);
    let max_tokens = usable / kv_per_token;
    if max_tokens < u64::from(AUTO_TUNE_MIN_CTX) {
        return None;
    }
    Some(u32::try_from(max_tokens).unwrap_or(u32::MAX))
}

/// Round `value` down to the largest power of two ≤ `value`, clamped
/// to `[AUTO_TUNE_MIN_CTX, AUTO_TUNE_MAX_CTX]`.
fn round_down_pow2(value: u32) -> u32 {
    if value < AUTO_TUNE_MIN_CTX {
        return AUTO_TUNE_MIN_CTX;
    }
    let clamped = value.min(AUTO_TUNE_MAX_CTX);
    let leading = clamped.leading_zeros();
    1u32 << (31 - leading)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_down_pow2_clamps_below_min() {
        assert_eq!(round_down_pow2(100), AUTO_TUNE_MIN_CTX);
        assert_eq!(round_down_pow2(2047), AUTO_TUNE_MIN_CTX);
    }

    #[test]
    fn round_down_pow2_at_min_returns_min() {
        assert_eq!(round_down_pow2(2048), 2048);
    }

    #[test]
    fn round_down_pow2_typical_values() {
        assert_eq!(round_down_pow2(4096), 4096);
        assert_eq!(round_down_pow2(5000), 4096);
        assert_eq!(round_down_pow2(8191), 4096);
        assert_eq!(round_down_pow2(8192), 8192);
        assert_eq!(round_down_pow2(43_690), 32_768);
    }

    #[test]
    fn round_down_pow2_clamps_above_max() {
        assert_eq!(round_down_pow2(100_000), AUTO_TUNE_MAX_CTX);
        assert_eq!(round_down_pow2(u32::MAX), AUTO_TUNE_MAX_CTX);
    }

    #[test]
    fn max_ctx_for_vram_handles_typical_qwen3_coder_30b() {
        // Qwen3-Coder-30B-A3B: 48 layers, 8 KV heads, head_dim 128
        let kv_per_token: u64 = 48 * 8 * 128 * 4;
        // 8 GiB free VRAM after model loads
        let free: u64 = 8 * 1024 * 1024 * 1024;
        let result = max_ctx_for_vram(free, kv_per_token).unwrap();
        // Sanity: must fit at least 16K, less than 64K
        assert!(result >= 16_384, "got {result}");
        assert!(result < 64_000, "got {result}");
    }

    #[test]
    fn max_ctx_for_vram_returns_none_when_oom() {
        // Only 100 MiB free, KV cost 192 KiB/token → less than min
        let free: u64 = 100 * 1024 * 1024;
        let kv: u64 = 192 * 1024;
        assert!(max_ctx_for_vram(free, kv).is_none());
    }

    #[test]
    fn max_ctx_for_vram_zero_kv_returns_none() {
        assert!(max_ctx_for_vram(8 * 1024 * 1024 * 1024, 0).is_none());
    }
}
