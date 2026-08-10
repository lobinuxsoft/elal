# Elal

**A local-first AI coding agent in Rust, with the inference engine embedded in the binary.**

No API keys. No network calls for inference. The model runs on your GPU, in your process.

> *Elal* is the culture hero of the Tehuelche people of Patagonia — the one who taught
> humans to make fire, to hunt, to survive. A guide, not an oracle.

---

## What this is

Elal embeds [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) directly, so a single
binary loads a GGUF model onto the GPU and drives an autonomous tool-calling loop over
your codebase. There is no inference server to run alongside it and no remote endpoint
in the hot path.

It is built around three constraints that shape every decision in the codebase:

- **The model runs locally.** Inference never leaves the machine.
- **VRAM is finite and shared.** The context window is auto-tuned against real GPU
  memory instead of being hardcoded and hoping for the best.
- **Sessions outlive the process.** Conversations are journaled to disk and resumable,
  including the KV cache.

## Status

**Alpha.** The agent loop, the embedded provider, the tool layer and session persistence
are implemented and usable. Seven crates in the workspace are reserved namespaces holding
four lines each — they are placeholders for planned work (MCP transport, client/server
split, sandboxed execution), not shipped functionality. The table below is explicit about
which is which.

## Features

- **Embedded GGUF inference** via `llama-cpp-2` — no sidecar process, no HTTP hop.
- **Dual GPU backend**: Vulkan by default, ROCm behind a feature flag, or both compiled in.
- **VRAM-aware context sizing**: `model + KV cache ≤ 80% of total VRAM`. The remaining 20%
  is left to the compositor, other workloads and llama.cpp scratch buffers. Models that
  cannot fit are refused with a diagnostic instead of thrashing.
- **Autonomous tool-calling loop** with an approval mode, so the agent asks before it acts.
- **Persistent sessions**: rollout journal on disk, resumable transcripts, optional KV-cache
  snapshots to skip prompt re-evaluation on resume.
- **Terminal UI** built on `ratatui`, plus a scriptable CLI.

## Architecture

A Cargo workspace. Implemented crates:

| Crate | Role |
|---|---|
| `elal_core` | Session state, rollout journal, replay, configuration |
| `elal_tools` | Tool implementations: file ops, search, web, shell |
| `elal_provider` | Embedded llama.cpp provider, backend selection, VRAM auto-tune |
| `elal_cli` | Binary entry point: `tui`, `doctor`, `chat`, `agent`, `sessions` |
| `elal_protocol` | Wire types: messages, turns, approval modes |
| `elal_models` | Model registry, GGUF discovery, download, VRAM-bounded swap |

Reserved namespaces, not yet implemented: `elal_client`, `elal_server`, `elal_mcp`,
`elal_safety`, `elal_tasks`, `elal_tui`, `elal_utils`.

## Requirements

- Rust **1.85+** (edition 2024)
- A Vulkan-capable GPU and drivers, or an AMD GPU with ROCm
- A GGUF model file

## Build

```bash
# Vulkan (default)
cargo build --release

# ROCm
cargo build --release --no-default-features --features rocm

# Both backends compiled in, selected at runtime
cargo build --release --no-default-features --features dual
```

### Troubleshooting: `fatal error: 'stdbool.h' file not found`

`llama-cpp-sys-2` generates its bindings with `bindgen`, which loads `libclang` at build
time. On distributions that ship several LLVM versions side by side — Fedora Atomic and
its derivatives, for instance, where a ROCm-bundled clang sits next to the system one —
`bindgen` can pick a `libclang` whose builtin headers it then fails to locate. Point it at
them explicitly:

```bash
BINDGEN_EXTRA_CLANG_ARGS="-I$(dirname $(find /usr/lib /usr/lib64 -name stdbool.h -path '*clang*' | head -1))" \
  cargo build --release
```

## Usage

```bash
# Check what the runtime sees: GPU, VRAM, backend, models
elal doctor

# Interactive TUI
elal tui

# One-shot generation against a model file
elal chat --model /path/to/model.gguf --prompt "explain this repository"

# Interactive agent loop with a persistent session
elal agent --new
elal agent --continue

# Inspect what is on disk
elal sessions list
elal sessions show <session-id>
```

## Configuration

Project-level settings live in a `.elal.toml` file discovered by walking up from the
working directory. See [`.elal.toml.example`](.elal.toml.example).

Global config and data:

| Path | Contents |
|---|---|
| `$XDG_CONFIG_HOME/elal/config.toml` | Global defaults |
| `$XDG_DATA_HOME/elal/` | Models, sessions, rollout journals |

Environment overrides: `ELAL_DATA_ROOT` (data root), `ELAL_LOG` (log level).

## Further reading

- [`docs/VRAM_BUDGET.md`](docs/VRAM_BUDGET.md) — how `n_ctx` is derived from available VRAM
  and why some models are refused outright.
- [`docs/SESSION_PERSISTENCE.md`](docs/SESSION_PERSISTENCE.md) — on-disk session format and
  replay semantics.

## License

MIT. See [LICENSE](LICENSE).
