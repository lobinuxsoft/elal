//! Automatic `n_ctx` sizing based on available VRAM and model architecture.
//!
//! The KV cache scales linearly with context length. A 30B model with
//! `n_ctx_train = 262144` will happily ask llama.cpp for ~24 GiB of KV
//! buffer if we accept the model's native default — and that exceeds
//! consumer GPU VRAM, forcing a spillover into RAM that ends in OOM.
//!
//! This module enforces a hard rule: **`model + KV` must fit in
//! `≤ VRAM_BUDGET_FRACTION × total VRAM`**. The remaining 20% is
//! headroom for everything that is NOT us — desktop compositor, other
//! GPU workloads, llama.cpp scratch buffers, kernel reservation. Free
//! VRAM at probe time is intentionally NOT used: another process can
//! free or claim memory between our probe and llama.cpp's allocation,
//! so a budget anchored on `total` is the only stable target.
//!
//! When the model file alone already exceeds the budget the loader
//! refuses to proceed and returns a hard error — silent spillover into
//! RAM is exactly the failure mode this module exists to prevent.

use std::path::Path;

use llama_cpp_2::model::LlamaModel;

use crate::error::LlmError;
use crate::hardware::VramInfo;

/// Hard upper bound for auto-tuned context. Models advertise up to 256K
/// tokens, but agent prompts rarely benefit beyond 32K and KV cache cost
/// scales linearly. Users who genuinely need more pass `--n-ctx`.
pub const AUTO_TUNE_MAX_CTX: u32 = 32_768;

/// Hard lower bound. Below this, agent prompts (system + tools + first
/// user message) start to truncate.
pub const AUTO_TUNE_MIN_CTX: u32 = 2_048;

/// Maximum fraction of total VRAM that `model + KV cache` may occupy.
/// The remainder is reserved for everything else on the GPU (desktop
/// compositor, other workloads, llama.cpp compute scratch).
pub const VRAM_BUDGET_FRACTION: f64 = 0.80;

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
/// - `requested == 0` → auto-tune from VRAM and model architecture,
///   capped by [`VRAM_BUDGET_FRACTION`] of total VRAM.
/// - `requested > 0`  → respect the override, but cap at `n_ctx_train`.
///   No VRAM check; the user explicitly opted in.
///
/// Returns [`LlmError::Load`] when auto-tuning runs and the model file
/// alone exceeds the VRAM budget — there is no `n_ctx` value that
/// would let it fit, so silent spillover would deceive the user.
pub fn auto_tune_n_ctx(
    model: &LlamaModel,
    requested: u32,
    vram: Option<VramInfo>,
    model_path: &Path,
) -> Result<AutoTuneResult, LlmError> {
    let native = model.n_ctx_train();

    if requested != 0 {
        let n_ctx = requested.min(native);
        let reason = if n_ctx == requested {
            format!("user override (n_ctx={requested})")
        } else {
            format!("user override clamped to n_ctx_train ({native})")
        };
        return Ok(AutoTuneResult {
            n_ctx,
            reason,
            user_override: true,
        });
    }

    let kv_per_token = kv_cache_bytes_per_token(model);

    let Some(vram) = vram else {
        // VRAM probe unavailable (non-AMD GPU, no DRM sysfs, CPU-only).
        // Fall back to the minimum context, capped at the model's
        // native max — better to underclock than to OOM blindly.
        let n_ctx = round_down_pow2(AUTO_TUNE_MIN_CTX.min(native));
        return Ok(AutoTuneResult {
            n_ctx,
            reason: format!("auto-tuned fallback: VRAM probe unavailable, defaulting to {n_ctx}"),
            user_override: false,
        });
    };

    let model_bytes = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);

    let candidate = max_ctx_for_vram(vram.total_bytes, model_bytes, kv_per_token)?;
    let clamped = candidate.min(native).min(AUTO_TUNE_MAX_CTX);
    let n_ctx = round_down_pow2(clamped.max(AUTO_TUNE_MIN_CTX));

    let total_gib = bytes_to_gib(vram.total_bytes);
    let model_gib = bytes_to_gib(model_bytes);
    let kv_kib = kv_per_token / 1024;
    let reason = format!(
        "auto-tuned: 80% of {total_gib:.1} GiB VRAM = {budget_gib:.1} GiB budget, model {model_gib:.2} GiB, {kv_kib} KiB/token KV → n_ctx={n_ctx}",
        budget_gib = total_gib * VRAM_BUDGET_FRACTION,
    );

    Ok(AutoTuneResult {
        n_ctx,
        reason,
        user_override: false,
    })
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

/// Largest `n_ctx` that fits inside the VRAM budget after subtracting
/// the model file size. Returns:
/// - `Err` when the model alone breaks the 80% rule (no `n_ctx` makes
///   it fit) or when there isn't even room for [`AUTO_TUNE_MIN_CTX`].
/// - `Ok(max_tokens)` otherwise.
fn max_ctx_for_vram(
    total_bytes: u64,
    model_bytes: u64,
    kv_per_token: u64,
) -> Result<u32, LlmError> {
    if kv_per_token == 0 {
        return Err(LlmError::Load(
            "model architecture metadata missing — cannot estimate KV cost".into(),
        ));
    }
    let budget = (total_bytes as f64 * VRAM_BUDGET_FRACTION) as u64;
    if model_bytes >= budget {
        return Err(LlmError::Load(format!(
            "model file ({:.2} GiB) exceeds {pct}% of total VRAM ({:.2} GiB) — \
             pass --n-gpu-layers 0 to run on CPU, --n-ctx <small> to override, \
             or pick a smaller / more aggressively quantised GGUF",
            bytes_to_gib(model_bytes),
            bytes_to_gib(total_bytes),
            pct = (VRAM_BUDGET_FRACTION * 100.0) as u32,
        )));
    }
    let kv_budget = budget - model_bytes;
    let max_tokens = kv_budget / kv_per_token;
    if max_tokens < u64::from(AUTO_TUNE_MIN_CTX) {
        return Err(LlmError::Load(format!(
            "VRAM budget after model ({} MiB) leaves room for only {max_tokens} tokens, \
             below the {AUTO_TUNE_MIN_CTX}-token minimum — use a smaller model or \
             --n-gpu-layers 0",
            kv_budget / (1024 * 1024),
        )));
    }
    Ok(u32::try_from(max_tokens).unwrap_or(u32::MAX))
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

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
#[path = "auto_tune_tests.rs"]
mod tests;
