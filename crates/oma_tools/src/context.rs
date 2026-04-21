//! Runtime context passed to [`Tool::execute`](crate::Tool::execute).
//!
//! Tools receive only what they need through [`ToolContext`] — no giant app
//! context object. Progress and partial output stream back to the agent/TUI
//! through the bundled [`mpsc::Sender<ToolEvent>`].

use std::path::Path;

use oma_protocol::SessionId;
use tokio::sync::mpsc;

/// Execution context borrowed by the tool for the duration of one invocation.
///
/// Borrowed (not owned) so the agent can keep its own state without cloning
/// paths per-call. The `events` sender is cloned into the context because
/// [`mpsc::Sender`] is designed to be cheaply cloneable.
pub struct ToolContext<'a> {
    /// Effective working directory for the current invocation.
    pub working_dir: &'a Path,
    /// Root of the current project, if discovery found one.
    pub project_root: Option<&'a Path>,
    /// Session this invocation belongs to.
    pub session_id: SessionId,
    /// Channel the tool can use to stream progress back to the agent/TUI.
    pub events: mpsc::Sender<ToolEvent>,
}

/// Incremental events a tool can emit while it is still running.
///
/// Delivered on a best-effort basis — if the receiver is slow, drops are
/// preferred to blocking the tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolEvent {
    /// Human-readable status line (e.g. "reading page 2/5").
    Progress(String),
    /// Partial output chunk for tools that stream results (e.g. long shell
    /// commands).
    Partial(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn context_forwards_events_to_receiver() {
        let cwd = PathBuf::from("/tmp");
        let (tx, mut rx) = mpsc::channel(4);
        let ctx = ToolContext {
            working_dir: &cwd,
            project_root: None,
            session_id: SessionId::new(),
            events: tx,
        };

        ctx.events
            .send(ToolEvent::Progress("hello".into()))
            .await
            .unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received, ToolEvent::Progress("hello".into()));
    }

    #[test]
    fn events_are_cloneable() {
        let a = ToolEvent::Partial("chunk".into());
        let b = a.clone();
        assert_eq!(a, b);
    }
}
