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
fn max_ctx_for_vram_handles_qwen3_1_7b_on_16gib_card() {
    // Qwen3-1.7B Q5_K_M: 1.4 GiB on disk, 28 layers, 8 KV heads, head_dim 128.
    let kv_per_token: u64 = 28 * 8 * 128 * 4;
    let total: u64 = 16 * 1024 * 1024 * 1024;
    let model: u64 = 14 * 1024 * 1024 * 1024 / 10; // 1.4 GiB
    let result = max_ctx_for_vram(total, model, kv_per_token).expect("must fit");
    // Budget: 16 * 0.80 = 12.8 GiB ; minus 1.4 model = 11.4 GiB for KV.
    // KV cost: 28 * 8 * 128 * 4 = 114 688 B/token → ~104K tokens fit.
    // The CLAMP to AUTO_TUNE_MAX_CTX happens later in auto_tune_n_ctx.
    assert!(result >= AUTO_TUNE_MAX_CTX, "got {result}");
}

#[test]
fn max_ctx_for_vram_rejects_model_alone_over_budget() {
    // Qwen3-Coder-30B Q3_K_S: ~13 GiB on a 16 GiB card → 13 > 16*0.80=12.8.
    let kv_per_token: u64 = 48 * 8 * 128 * 4;
    let total: u64 = 16 * 1024 * 1024 * 1024;
    let model: u64 = 13 * 1024 * 1024 * 1024;
    let err = max_ctx_for_vram(total, model, kv_per_token).expect_err("must reject");
    let msg = format!("{err}");
    assert!(msg.contains("exceeds"), "{msg}");
    assert!(msg.contains("80%"), "{msg}");
}

#[test]
fn max_ctx_for_vram_rejects_when_kv_budget_below_min() {
    // Tight budget: 12 GiB total → 9.6 GiB budget. Model takes 9.5 GiB.
    // Remaining 0.1 GiB / 192 KiB per token = ~545 tokens — below MIN.
    let kv_per_token: u64 = 192 * 1024;
    let total: u64 = 12 * 1024 * 1024 * 1024;
    let model: u64 = 95 * 1024 * 1024 * 1024 / 10;
    let err = max_ctx_for_vram(total, model, kv_per_token).expect_err("must reject");
    let msg = format!("{err}");
    assert!(msg.contains("below the"), "{msg}");
    assert!(msg.contains(&format!("{AUTO_TUNE_MIN_CTX}")), "{msg}");
}

#[test]
fn max_ctx_for_vram_zero_kv_returns_err() {
    let err =
        max_ctx_for_vram(8 * 1024 * 1024 * 1024, 1024 * 1024 * 1024, 0).expect_err("must reject");
    assert!(format!("{err}").contains("metadata missing"));
}

#[test]
fn max_ctx_for_vram_honors_80_percent_boundary() {
    // Model at exactly 80% of total → reject (>= boundary, no room for KV).
    let total: u64 = 10 * 1024 * 1024 * 1024;
    let model: u64 = 8 * 1024 * 1024 * 1024;
    let err = max_ctx_for_vram(total, model, 1024).expect_err("must reject");
    assert!(format!("{err}").contains("80%"));
}

#[test]
fn max_ctx_for_vram_just_under_boundary_succeeds() {
    // Model at 79% of total → tiny room left, but enough for MIN_CTX.
    let kv_per_token: u64 = 1024; // small KV → many tokens fit per byte
    let total: u64 = 10 * 1024 * 1024 * 1024;
    let model: u64 = 79 * 1024 * 1024 * 1024 / 10; // 7.9 GiB
    let result = max_ctx_for_vram(total, model, kv_per_token).expect("must fit");
    assert!(result >= AUTO_TUNE_MIN_CTX, "got {result}");
}
