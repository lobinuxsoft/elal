# VRAM Budget & `n_ctx` Auto-Tune

How `EmbeddedProvider::load` decides the context window size and why
loading certain models triggers a hard refusal.

## The rule

> **`model + KV cache  ≤  vram.total_bytes × 0.80`**

The remaining 20% is reserved for everything that is NOT us — desktop
compositor, other GPU workloads, `llama.cpp` compute scratch buffers,
kernel reservation. The auto-tuner picks the largest power-of-two
`n_ctx` that fits inside the budget after the model file size has
been deducted.

## Why anchor on TOTAL VRAM, not FREE

Free VRAM at probe time is unstable. Between our probe and
`llama.cpp`'s actual KV-cache allocation, another process can claim
or release memory; basing the budget on free bytes silently invites
OOM the moment a browser tab opens or a compositor pre-allocates.
Total bytes are a hardware constant — anchoring there means the
budget is the same on every load and the user can predict it from
`elal doctor`.

The 20% headroom assumes the user understands that other GPU
workloads also need to fit. It is **not** a safety net against the
user opening 6 GiB of Chrome on a 16 GiB card.

## What happens when the model alone exceeds the budget

`auto_tune_n_ctx` returns `Err(LlmError::Load)` with a clear message:

```
model file (12.8 GiB) exceeds 80% of total VRAM (12.74 GiB) —
pass --n-gpu-layers 0 to run on CPU, --n-ctx <small> to override,
or pick a smaller / more aggressively quantised GGUF
```

There is **no silent fallback**. Spilling into RAM was the original
failure mode this module exists to prevent (Qwen3-Coder-30B-A3B
advertises `n_ctx_train = 262144` and would have asked for ~24 GiB
of KV buffer otherwise).

The same hard refusal kicks in when `model_bytes < budget` but the
remaining space leaves room for fewer than `AUTO_TUNE_MIN_CTX = 2048`
tokens. A 600-token context wouldn't fit a system prompt + tool
catalog + first user message anyway.

## Override: `--n-ctx <u32>`

Both `elal agent` and `elal chat` accept `--n-ctx`:

| Value         | Behaviour                                                |
| ------------- | -------------------------------------------------------- |
| `0` (default) | Auto-tune from VRAM under the 80% rule.                  |
| `> 0`         | Honour as user override. Clamped to `n_ctx_train` of the loaded model. **Skips the VRAM check entirely** — the user is opting in. |

The override is an escape hatch for "I know what I am doing, fit it
or die trying" cases. It is the only path to load a model whose
file size is over the 80% budget on a given GPU.

## CPU fallback: `--n-gpu-layers 0`

When the model can't live in VRAM at all, push it to RAM:

```
elal agent --new --model <huge.gguf> --n-gpu-layers 0
```

VRAM probe is irrelevant on this path; `auto_tune_n_ctx` falls back
to its conservative default (`AUTO_TUNE_MIN_CTX`, capped at
`n_ctx_train`). RAM has no equivalent budget rule yet — user is
responsible for not OOM-killing the box.

## Worked examples (RX 9070 XT, 15.92 GiB total)

`80% × 15.92 GiB = 12.74 GiB budget`

| Model                                  | File size  | KV/token  | Auto-tuned `n_ctx` | Notes                  |
| -------------------------------------- | ---------- | --------- | ------------------ | ---------------------- |
| Qwen3-1.7B Q5_K_M                      | 1.40 GiB   | 14 KiB    | `32_768` (capped)  | Plenty of headroom.    |
| Qwen3-Coder-30B-A3B Q3_K_S             | 12.38 GiB  | 48 KiB    | `4_096`            | KV budget 0.36 GiB → ~7.5 K tokens, rounded down to nearest pow2. |
| Hypothetical 14 GiB model              | 14.00 GiB  | —         | `Err`              | Model alone over budget. |

## Where to look

- Rule + auto-tuner — `crates/elal_provider/src/auto_tune.rs`
  (`VRAM_BUDGET_FRACTION`, `auto_tune_n_ctx`, `max_ctx_for_vram`).
- VRAM probe — `crates/elal_provider/src/hardware.rs`
  (`detect_primary_gpu_vram`; AMD via DRM sysfs only — non-AMD GPUs
  return `None` and trigger the CPU-style fallback).
- CLI plumbing — `elal agent --n-ctx`, `elal chat --n-ctx`,
  `elal doctor` GPU VRAM line.
- Refuse-to-load message — `LlmError::Load` propagated through
  `EmbeddedProvider::load` in `crates/elal_provider/src/embedded/mod.rs`.

## Out of scope (open follow-ups)

- NVIDIA / Intel VRAM detection. Today the probe only reads AMD DRM
  sysfs; other vendors get `None` → minimum-context fallback.
- RAM budget rule for `--n-gpu-layers 0` paths.
- Partial GPU offload heuristic (`--n-gpu-layers <k>` between 0 and
  -1) — currently the user picks `k` blind.
- Runtime backend auto-detection across compiled-in backends
  (Vulkan / ROCm / CUDA). Tracked separately.
