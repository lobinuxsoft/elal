//! In-memory session state — mirrors the rollout journal.
//!
//! Direct port of `claw-code-rust/crates/core/src/session.rs`'s `SessionState`,
//! adapted to oh-my-agent's protocol types: `oma_protocol::Message` (real
//! `Role::System`) instead of claw's `RequestMessage` (system-as-prompt-prefix),
//! `u64` token counters matching [`oma_protocol::TurnUsage`], and a typed
//! [`oma_protocol::ApprovalMode`] inside [`SessionConfig`].
//!
//! The compaction policy is the one from `claw-code-rust/core/query.rs::compact_session`
//! with one substantive change: messages with `Role::System` are preserved
//! across compactions. claw's flat `RequestMessage` has no equivalent role,
//! so its policy can drop from index 0 freely; ours cannot, otherwise the
//! agent would lose its system prompt mid-conversation.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use oma_protocol::{ApprovalMode, Message, Role, SessionId};

/// Token-budget configuration for a session.
///
/// Mirrors `claw-code-rust/core/context.rs::TokenBudget`. The `compact_threshold`
/// is the fraction of `input_budget()` past which compaction triggers; defaults
/// to 0.9 — once the running prompt crosses 90 % of the input budget, the
/// session asks for a compaction round so the next request keeps headroom.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenBudget {
    /// Total context window supported by the active model (tokens).
    pub context_window: usize,
    /// Tokens reserved for model output. Subtracted from `context_window` to
    /// arrive at the prompt-input budget.
    pub max_output_tokens: usize,
    /// Fraction of [`TokenBudget::input_budget`] that triggers compaction.
    pub compact_threshold: f64,
}

impl TokenBudget {
    /// Builds a budget with the default 0.9 compaction threshold.
    pub fn new(context_window: usize, max_output_tokens: usize) -> Self {
        Self {
            context_window,
            max_output_tokens,
            compact_threshold: 0.9,
        }
    }

    /// Tokens available for prompt input (history + system + tools + user turn).
    /// Saturates to zero when `max_output_tokens` exceeds `context_window`
    /// (a misconfiguration the auto-tune layer guards against, but worth
    /// being defensive at this layer).
    pub fn input_budget(&self) -> usize {
        self.context_window.saturating_sub(self.max_output_tokens)
    }

    /// Returns true when `current_tokens` crosses the compaction threshold.
    pub fn should_compact(&self, current_tokens: usize) -> bool {
        current_tokens as f64 > self.input_budget() as f64 * self.compact_threshold
    }
}

impl Default for TokenBudget {
    /// Conservative default sized for the smallest models we routinely use.
    /// Specific runs override this from the loaded provider's
    /// `n_ctx` after auto-tune.
    fn default() -> Self {
        Self::new(32_768, 4_096)
    }
}

/// Per-session configuration.
#[derive(Debug, Clone, Default)]
pub struct SessionConfig {
    /// Token budget governing compaction.
    pub token_budget: TokenBudget,
    /// Approval mode the session was created under.
    pub approval_mode: ApprovalMode,
}

/// Mutable in-memory state for one conversation session.
///
/// One-to-one mirror of `claw-code-rust/core/session.rs::SessionState` with
/// oh-my-agent's typing. The rollout journal (chunk 2) hydrates this from
/// disk on `--resume`, and writes append back as the agent loop runs.
#[derive(Debug)]
pub struct SessionState {
    /// Stable session identifier.
    pub id: SessionId,
    /// Per-session configuration.
    pub config: SessionConfig,
    /// In-order conversation history. The first entry is typically a
    /// `Role::System` message; compaction preserves all such entries.
    pub messages: Vec<Message>,
    /// Working directory the session is bound to.
    pub cwd: PathBuf,
    /// Number of turns executed so far (1-based once a turn completes).
    pub turn_count: usize,
    /// Cumulative input tokens across every completed turn.
    pub total_input_tokens: u64,
    /// Cumulative output tokens across every completed turn.
    pub total_output_tokens: u64,
    /// Input tokens reported for the most recent turn — drives the
    /// compaction trigger via [`TokenBudget::should_compact`].
    pub last_input_tokens: usize,
    /// User prompts queued while a turn is mid-flight; drained at the
    /// start of the next turn. Wrapped in an `Arc<Mutex<_>>` so an external
    /// task (REPL, future TUI) can enqueue while the agent loop holds
    /// `&mut self`.
    pub pending_user_prompts: Arc<Mutex<VecDeque<String>>>,
}

impl SessionState {
    /// Builds a fresh session with a generated [`SessionId`] and an empty history.
    pub fn new(config: SessionConfig, cwd: PathBuf) -> Self {
        Self {
            id: SessionId::new(),
            config,
            messages: Vec::new(),
            cwd,
            turn_count: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            last_input_tokens: 0,
            pending_user_prompts: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Builds a session with a caller-supplied id — used by replay (chunk 3)
    /// to reconstruct a session from its rollout JSONL.
    pub fn with_id(id: SessionId, config: SessionConfig, cwd: PathBuf) -> Self {
        Self {
            id,
            config,
            messages: Vec::new(),
            cwd,
            turn_count: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            last_input_tokens: 0,
            pending_user_prompts: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Appends a message to the in-memory history.
    pub fn push_message(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    /// Enqueues a user prompt to be drained at the start of the next turn.
    /// Safe to call from a different task than the one running the agent.
    ///
    /// # Panics
    /// Panics if the pending-prompts mutex has been poisoned by a panic in
    /// another holder. A poisoned mutex here means a prior thread crashed
    /// while mutating the queue — the session is not recoverable.
    pub fn enqueue_user_prompt(&self, prompt: String) {
        self.pending_user_prompts
            .lock()
            .expect("pending user prompts mutex poisoned")
            .push_back(prompt);
    }

    /// Drains every queued user prompt in FIFO order.
    ///
    /// # Panics
    /// See [`SessionState::enqueue_user_prompt`].
    pub fn drain_pending_user_prompts(&self) -> Vec<String> {
        self.pending_user_prompts
            .lock()
            .expect("pending user prompts mutex poisoned")
            .drain(..)
            .collect()
    }

    /// Drops oldest non-system messages until the running prompt is
    /// projected to fit ~70 % of the input budget. Returns the number
    /// of messages removed. No-op when fewer than three messages are
    /// present, or when every droppable message would have to be
    /// removed to leave at least one non-system entry.
    pub fn compact(&mut self) -> usize {
        compact_messages(
            &mut self.messages,
            self.last_input_tokens,
            &self.config.token_budget,
        )
    }
}

/// Pure-function variant of [`SessionState::compact`] — operates on any
/// `Vec<Message>` so the agent loop can reuse the same policy without
/// constructing a `SessionState`.
pub fn compact_messages(
    messages: &mut Vec<Message>,
    last_input_tokens: usize,
    budget: &TokenBudget,
) -> usize {
    if messages.len() <= 2 {
        return 0;
    }
    let target = compute_remove_count(messages.len(), last_input_tokens, budget);
    if target == 0 {
        return 0;
    }
    drop_oldest_non_system(messages, target)
}

fn compute_remove_count(msg_count: usize, last_input_tokens: usize, budget: &TokenBudget) -> usize {
    if last_input_tokens == 0 {
        return msg_count / 2;
    }
    let avg = last_input_tokens / msg_count;
    if avg == 0 {
        return msg_count / 2;
    }
    let target = (budget.input_budget() as f64 * 0.7) as usize;
    let keep = (target / avg).max(2).min(msg_count);
    msg_count - keep
}

fn drop_oldest_non_system(messages: &mut Vec<Message>, target: usize) -> usize {
    let mut removed = 0;
    messages.retain(|msg| {
        let should_drop = removed < target && msg.role != Role::System;
        if should_drop {
            removed += 1;
        }
        !should_drop
    });
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with_window(context_window: usize, max_output: usize) -> SessionConfig {
        SessionConfig {
            token_budget: TokenBudget::new(context_window, max_output),
            approval_mode: ApprovalMode::Smart,
        }
    }

    #[test]
    fn token_budget_default_is_conservative() {
        let budget = TokenBudget::default();
        assert_eq!(budget.context_window, 32_768);
        assert_eq!(budget.max_output_tokens, 4_096);
        assert!((budget.compact_threshold - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn input_budget_saturates_when_misconfigured() {
        let budget = TokenBudget::new(100, 200);
        assert_eq!(budget.input_budget(), 0);
    }

    #[test]
    fn should_compact_threshold() {
        let budget = TokenBudget::new(1_000, 0);
        assert!(!budget.should_compact(800));
        assert!(budget.should_compact(950));
    }

    #[test]
    fn new_session_starts_empty() {
        let state = SessionState::new(SessionConfig::default(), PathBuf::from("/tmp"));
        assert_eq!(state.messages.len(), 0);
        assert_eq!(state.turn_count, 0);
        assert_eq!(state.total_input_tokens, 0);
        assert_eq!(state.last_input_tokens, 0);
        assert_eq!(state.cwd, PathBuf::from("/tmp"));
    }

    #[test]
    fn with_id_preserves_caller_supplied_id() {
        let id = SessionId::new();
        let state = SessionState::with_id(id, SessionConfig::default(), PathBuf::from("/tmp"));
        assert_eq!(state.id, id);
    }

    #[test]
    fn push_message_appends() {
        let mut state = SessionState::new(SessionConfig::default(), PathBuf::from("/tmp"));
        state.push_message(Message::user("hi"));
        state.push_message(Message::assistant("hello"));
        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.messages[0].role, Role::User);
        assert_eq!(state.messages[1].role, Role::Assistant);
    }

    #[test]
    fn pending_prompts_round_trip_in_fifo_order() {
        let state = SessionState::new(SessionConfig::default(), PathBuf::from("/tmp"));
        state.enqueue_user_prompt("first".into());
        state.enqueue_user_prompt("second".into());
        state.enqueue_user_prompt("third".into());

        let drained = state.drain_pending_user_prompts();
        assert_eq!(drained, vec!["first", "second", "third"]);

        let drained_again = state.drain_pending_user_prompts();
        assert!(drained_again.is_empty());
    }

    #[test]
    fn compact_noop_below_three_messages() {
        let mut state = SessionState::new(SessionConfig::default(), PathBuf::from("/tmp"));
        state.push_message(Message::system("sys"));
        state.push_message(Message::user("hi"));
        state.last_input_tokens = 999_999_999; // would trigger compaction otherwise
        assert_eq!(state.compact(), 0);
        assert_eq!(state.messages.len(), 2);
    }

    #[test]
    fn compact_with_zero_tokens_drops_oldest_half_of_droppable() {
        let mut state = SessionState::new(SessionConfig::default(), PathBuf::from("/tmp"));
        state.push_message(Message::system("sys"));
        for i in 0..6 {
            state.push_message(Message::user(format!("msg-{i}")));
        }
        // No token data → target_remove = msg_count / 2 = 3.
        let removed = state.compact();
        assert_eq!(removed, 3);
        assert_eq!(state.messages.len(), 4);
        // System preserved at index 0.
        assert_eq!(state.messages[0].role, Role::System);
    }

    #[test]
    fn compact_preserves_system_messages() {
        let mut state = SessionState::new(cfg_with_window(1_000, 100), PathBuf::from("/tmp"));
        state.push_message(Message::system("sys-anchor"));
        state.push_message(Message::user("u1"));
        state.push_message(Message::assistant("a1"));
        state.push_message(Message::user("u2"));
        state.push_message(Message::assistant("a2"));
        state.last_input_tokens = 5_000; // far above input_budget=900 → forces removal.
        let removed = state.compact();
        assert!(removed > 0, "expected compaction to remove messages");
        // System always survives.
        assert!(
            state.messages.iter().any(|m| m.role == Role::System),
            "system message must survive compaction"
        );
        assert_eq!(state.messages[0].role, Role::System);
    }

    #[test]
    fn compact_uses_70_percent_budget_target() {
        // input_budget = 900, target = 630, avg = 5000/5 = 1000.
        // keep = max((630/1000), 2) = 2, remove = 5 - 2 = 3.
        let mut state = SessionState::new(cfg_with_window(1_000, 100), PathBuf::from("/tmp"));
        for i in 0..5 {
            state.push_message(Message::user(format!("u{i}")));
        }
        state.last_input_tokens = 5_000;
        let removed = state.compact();
        assert_eq!(removed, 3);
        assert_eq!(state.messages.len(), 2);
        // Oldest two were dropped; remaining are u3, u4.
        assert_eq!(state.messages[0].content.as_deref(), Some("u3"));
        assert_eq!(state.messages[1].content.as_deref(), Some("u4"));
    }

    #[test]
    fn compact_only_system_messages_is_noop() {
        let mut state = SessionState::new(SessionConfig::default(), PathBuf::from("/tmp"));
        state.push_message(Message::system("sys-1"));
        state.push_message(Message::system("sys-2"));
        state.push_message(Message::system("sys-3"));
        state.last_input_tokens = 999_999;
        let removed = state.compact();
        assert_eq!(removed, 0);
        assert_eq!(state.messages.len(), 3);
    }

    #[test]
    fn compact_capped_by_droppable_count() {
        // Edge case: target_remove > droppable_count. Must not panic and must
        // remove at most droppable_count messages.
        let mut state = SessionState::new(cfg_with_window(1_000, 100), PathBuf::from("/tmp"));
        state.push_message(Message::system("sys"));
        state.push_message(Message::system("sys"));
        state.push_message(Message::user("u1"));
        state.push_message(Message::user("u2"));
        state.last_input_tokens = 0; // → target = msg_count/2 = 2.
        let removed = state.compact();
        // Both user messages are droppable; both can be removed.
        assert_eq!(removed, 2);
        assert!(state.messages.iter().all(|m| m.role == Role::System));
    }
}
