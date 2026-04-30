# Session Persistence

How `oma agent` keeps a conversation across process restarts: on-disk
layout, JSONL contract, sidecar atomicity, resume semantics, and the
opt-in KV-cache snapshot path.

## On-disk layout

Sessions live under `OMA_DATA_ROOT` (defaults to `dirs::data_dir()/oh-my-agent`,
which is `~/.local/share/oh-my-agent` on Linux). One directory per UTC
calendar day; one rollout file per session:

```
<data_root>/sessions/<YYYY>/<MM>/<DD>/rollout-<rfc3339-secs>-<id>.jsonl
<data_root>/sessions/<YYYY>/<MM>/<DD>/rollout-<rfc3339-secs>-<id>.jsonl.meta.json
<data_root>/sessions/<YYYY>/<MM>/<DD>/rollout-<rfc3339-secs>-<id>.jsonl.kv   (opt-in)
```

- `rollout-...jsonl` — append-only journal of events (durable source of truth).
- `<rollout>.meta.json` — sidecar with mutating `SessionRecord` fields,
  rewritten atomically on every turn close.
- `<rollout>.kv` — `LlamaContext` state snapshot, only when the session
  was opened with `--save-kv-cache`.

Override via `OMA_DATA_ROOT=<path>` for sandboxes / integration tests.

## JSONL contract (`oma_protocol::session`)

Every line in the rollout file is a tagged JSON object that deserializes
to `RolloutLine`. The first line is always `RolloutLine::SessionMeta`;
subsequent lines are `Turn`, `Item`, or `SessionTitleUpdated` in append
order. Lines are never edited or reordered.

| `kind`                   | Payload                                  | Emitted when                                                  |
| ------------------------ | ---------------------------------------- | ------------------------------------------------------------- |
| `session_meta`           | `SessionMetaLine { timestamp, session }` | First write of a fresh session.                               |
| `turn`                   | `TurnLine { timestamp, turn }`           | `Persistence::begin_turn` (Running) and `end_turn` (terminal).|
| `item`                   | `ItemLine { timestamp, item }`           | Each `TurnItem` produced during the turn.                     |
| `session_title_updated`  | `SessionTitleUpdatedLine { ... }`        | Title changes; mirrors the new value.                         |

`TurnItem` covers `UserMessage`, `AgentMessage`, `Reasoning`, `ToolCall`,
`ToolResult`, `ContextCompaction`. Each `ItemRecord` carries a strictly
increasing per-session `seq` so replay can deduplicate / sort even when
clocks jitter.

`SCHEMA_VERSION` is bumped only on breaking shape changes; replay
tolerates older versions explicitly. Writers always emit the current
constant.

## Sidecar atomicity

`RolloutStore::write_meta_sidecar` writes `<rollout>.meta.json` via
temp-file + rename (`<sidecar>.tmp` → `<sidecar>`). Same-filesystem
rename is atomic on POSIX; cross-fs writes surface as `std::io::Error`.

The sidecar mirrors mutating fields (`updated_at`, `total_*_tokens`,
`title`, `first_user_message`) so `oma sessions list` answers in O(1)
per session without replaying the full JSONL. The JSONL remains the
authoritative source — when the sidecar is missing or corrupt, the
query layer falls back to a full replay.

## Resume semantics (`Agent::resume_session`)

`oma agent --continue` and `--resume <id>`:

1. `oma_core::session::query::{find_latest, locate}` resolves the
   target rollout. `--continue` matches by `cwd`; `--resume` by
   `SessionId`.
2. `load_session(rollout)` replays the JSONL into `LoadedSession`:
   - `state.messages: Vec<Message>` — full conversation, system prompt
     EXCLUDED (`Agent::ensure_system_prompt` re-injects it).
   - `last_turn_seq` / `last_item_seq` — highest sequence numbers seen,
     so subsequent appends do not collide.
   - `record` — the most recent `SessionRecord` (sidecar-first when
     present, otherwise reconstructed from the meta line).
3. `Agent::resume_session(store, loaded)` builds a stateful agent that
   reuses the existing rollout file — no second `SessionMeta` line is
   ever written.
4. Replay tolerates a truncated tail: a malformed final line stops
   replay at the last good record. Useful when a process crashes
   mid-turn — you keep the work, just not the partial line.

## KV-cache snapshots (`--save-kv-cache`)

Opt-in. When the flag is set, every turn:

1. Loads `<rollout>.kv` into `LlamaContext` before tokenising. The
   snapshot covers a known prefix of the conversation; tokens past
   that prefix go through normal prompt-eval.
2. After generation, atomically rewrites `<rollout>.kv` (`.kv.tmp` →
   `.kv`) so the next turn can skip eval over the new full prefix.
3. Validates the snapshot against the loaded model's SHA-256 via
   `validate_compatible`. A mismatch refuses the load and falls back
   to full prompt-eval — you keep correctness over speed.

Tradeoff: snapshots can grow to ~1 GiB at full 32K context for a
mid-size model. They live next to the rollout, so they share its
lifetime. Skip the flag for short-lived sessions or when disk is
tight.

The CLI surfaces the decision at startup:

- `[oma agent] kv-cache enabled at <path>` — snapshot will load (if
  present) and rewrite at end of turn.
- `[oma agent] --save-kv-cache: refusing to load snapshot — <reason>` —
  SHA mismatch or missing-but-still-empty case; the flag still enables
  saving going forward.

## Failure modes

- **Disk full / EIO mid-turn** — `tracing::error!("session persistence
  write failed")`, the turn keeps running. Local sessions prefer
  degraded continuity over aborting.
- **Corrupt rollout** — `list_sessions` silently skips files that fail
  both sidecar and replay. One bad file cannot block the rest.
- **Schema drift** — older `schema_version` is tolerated by replay;
  unknown future versions surface as a parse error and are skipped.
- **KV snapshot stale after model swap** — caught by `validate_compatible`,
  fallback is automatic.

## Where to look

- Wire types — `crates/oma_protocol/src/session.rs`
- Writer — `crates/oma_core/src/session/rollout.rs`
- Replay — `crates/oma_core/src/session/replay.rs`
- Query / listing — `crates/oma_core/src/session/query.rs`
- Agent integration — `crates/oma_core/src/agent/persist.rs`
- KV snapshots — `crates/oma_provider/src/kv_snapshot.rs`
- CLI surface — `crates/oma_cli/src/{agent_cmd,sessions_cmd}.rs`
