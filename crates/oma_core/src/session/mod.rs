//! Session persistence — in-memory state, JSONL rollout, replay, query.
//!
//! Submodules are introduced chunk-by-chunk per issue #25:
//! - `state` (chunk 1): [`SessionState`], [`SessionConfig`], [`TokenBudget`].
//! - `rollout` (chunk 2): JSONL append-only writer.
//! - `replay` (chunk 3): rollout → `SessionState` rehydration.
//! - `query` (chunk 4): list / find-latest over the sessions directory.
//! - `kv_snapshot` (chunk 7): llama.cpp `state_save_file` / `state_load_file`
//!   adapter.

mod query;
mod replay;
mod rollout;
mod state;

pub use query::{ListedSession, find_latest, list_sessions, locate};
pub use replay::{LoadedSession, load_session};
pub use rollout::RolloutStore;
pub use state::{SessionConfig, SessionState, TokenBudget};
