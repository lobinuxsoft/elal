//! Context-window budget enforcement.
//!
//! Local models run with modest context windows (8K–128K). When the
//! conversation history approaches the budget, we must drop messages or
//! the next provider call will fail at decode time.
//!
//! MVP policy is intentionally simple:
//!
//! 1. The **system** message is never dropped — it carries the instructions
//!    the model needs to behave correctly.
//! 2. The **most recent user message** is never dropped — otherwise the
//!    model would be answering a different question than the one asked.
//! 3. Everything else is dropped in order, oldest first, until the
//!    remaining history fits under the budget.
//!
//! When even the protected messages exceed the budget (rare, requires
//! pathological system prompts), [`fit_to_budget`] stops — truncating a
//! protected message is worse than letting the provider reject the
//! request with a clear error.
//!
//! Summarisation of dropped segments is explicitly deferred to a
//! follow-up; we will add it when we see it matter in practice, not
//! preemptively.

use oma_protocol::{Message, Role};

/// Rough token-count heuristic. Four characters per token is the
/// conventional English-text estimate; we add a small overhead for role
/// markers and JSON envelope. Good enough for budget decisions — the
/// actual decode error is the hard stop.
pub(crate) fn approx_token_count(message: &Message) -> usize {
    let content_len = message.content.as_ref().map(|s| s.len()).unwrap_or(0);
    let tool_calls_len: usize = message
        .tool_calls
        .as_ref()
        .map(|calls| {
            calls
                .iter()
                .map(|c| c.function.name.len() + c.function.arguments.len())
                .sum()
        })
        .unwrap_or(0);
    let reasoning_len = message
        .reasoning_content
        .as_ref()
        .map(|s| s.len())
        .unwrap_or(0);
    (content_len + tool_calls_len + reasoning_len) / 4 + 8
}

/// Drop oldest non-protected messages until the total approximate token
/// count fits under `max_tokens`. Returns how many messages were dropped.
pub(crate) fn fit_to_budget(messages: &mut Vec<Message>, max_tokens: usize) -> usize {
    let mut dropped = 0;
    loop {
        let total: usize = messages.iter().map(approx_token_count).sum();
        if total <= max_tokens {
            return dropped;
        }
        let last_user_idx = messages.iter().rposition(|m| matches!(m.role, Role::User));
        // Find the oldest message that is neither the system prompt nor the
        // most recent user message.
        let drop_idx = messages.iter().enumerate().find_map(|(i, m)| {
            let is_system = matches!(m.role, Role::System);
            let is_protected_user = last_user_idx == Some(i);
            (!is_system && !is_protected_user).then_some(i)
        });
        match drop_idx {
            Some(i) => {
                messages.remove(i);
                dropped += 1;
            }
            // Nothing more we can drop without violating the protection
            // rules — the provider will reject this prompt if it is still
            // too big, and that is a clearer failure than silently
            // truncating system context.
            None => return dropped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oma_protocol::Message;

    fn user_filler(chars: usize) -> Message {
        Message::user("x".repeat(chars))
    }
    fn assistant_filler(chars: usize) -> Message {
        Message::assistant("x".repeat(chars))
    }

    #[test]
    fn approx_token_count_grows_with_content() {
        let small = Message::user("hi");
        let big = Message::user("x".repeat(400));
        assert!(approx_token_count(&big) > approx_token_count(&small));
    }

    #[test]
    fn under_budget_is_noop() {
        let mut msgs = vec![
            Message::system("sys"),
            Message::user("hello"),
            Message::assistant("hi"),
        ];
        let dropped = fit_to_budget(&mut msgs, 10_000);
        assert_eq!(dropped, 0);
        assert_eq!(msgs.len(), 3);
    }

    #[test]
    fn over_budget_drops_oldest_non_protected() {
        let mut msgs = vec![
            Message::system("sys"),
            user_filler(4000),      // old user — droppable
            assistant_filler(4000), // old assistant — droppable
            user_filler(4000),      // protected most-recent-user
        ];
        let dropped = fit_to_budget(&mut msgs, 1200);
        assert!(
            dropped >= 1,
            "expected to drop at least one non-protected message"
        );
        // System stays.
        assert!(matches!(msgs[0].role, Role::System));
        // Protected user stays (the last message, or close to it).
        assert!(matches!(msgs.last().unwrap().role, Role::User));
    }

    #[test]
    fn system_is_never_dropped_even_if_alone() {
        // Absurdly tight budget: only the system fits, barely.
        let mut msgs = vec![Message::system("s".repeat(8000))];
        let dropped = fit_to_budget(&mut msgs, 100);
        assert_eq!(dropped, 0);
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn protected_last_user_updates_as_messages_drop() {
        // Two user messages — dropping the first one should be fine,
        // the second is the "last user" and should stay.
        let mut msgs = vec![user_filler(4000), assistant_filler(4000), user_filler(1000)];
        fit_to_budget(&mut msgs, 400);
        // Last message must still be the short user.
        let last = msgs.last().unwrap();
        assert!(matches!(last.role, Role::User));
        assert!(last.content.as_deref().unwrap().len() <= 1000);
    }
}
