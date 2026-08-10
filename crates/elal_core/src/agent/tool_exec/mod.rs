//! Tool dispatch — runs a batch of [`PendingToolCall`]s against the
//! [`ToolRegistry`], consulting the approval resolver and surfacing
//! `ApprovalRequired` prompts over the event channel when the user's
//! attention is needed.
//!
//! Soft failures (unknown tool, invalid JSON args, tool execution error,
//! user denial) are converted into `AgentEvent::ToolResult { is_error: true }`
//! and a matching `Message::tool` payload so the model can observe the
//! failure and recover. Hard `TurnError` surfaces only for unrecoverable
//! situations such as the consumer dropping the events channel.

use std::collections::HashSet;
use std::path::Path;

use elal_protocol::{ApprovalDecisionValue, ApprovalMode, ApprovalScopeValue, Message, SessionId};
use elal_tools::{ApprovalDecision, ToolContext, ToolRegistry, resolve};
use tokio::sync::mpsc;

use super::event::{AgentEvent, ApprovalRequest, UserAction};
use super::stream_router::PendingToolCall;
use super::turn::TurnError;

/// Outcome of dispatching a batch of pending tool calls.
pub(crate) struct DispatchOutcome {
    pub events: Vec<AgentEvent>,
    pub tool_messages: Vec<Message>,
    pub executed: u32,
    /// True when the user answered an approval prompt with `Cancel`,
    /// meaning the whole turn should end immediately.
    pub cancelled: bool,
}

/// Everything `dispatch` needs besides the pending calls themselves. One
/// struct so the function signature stays readable.
pub(crate) struct DispatchContext<'a> {
    pub tools: &'a ToolRegistry,
    pub working_dir: &'a Path,
    pub session_id: SessionId,
    pub mode: ApprovalMode,
    /// Tool names the user has pre-approved at session scope. Mutated
    /// when an `Approve` response carries `Session` / `Tool` scope.
    pub approved_for_session: &'a mut HashSet<String>,
    /// Tool names the user has pre-approved for the current turn only.
    /// Mutated when an `Approve` response carries `Turn` scope.
    pub approved_for_turn: &'a mut HashSet<String>,
    pub events_tx: &'a mpsc::Sender<AgentEvent>,
    pub actions_rx: &'a mut mpsc::Receiver<UserAction>,
}

enum Gate {
    /// Proceed with execution.
    Allow,
    /// Treat as a soft failure with the attached message.
    Deny(String),
    /// User cancelled — stop the whole turn.
    Cancel,
}

/// Dispatch the batch, honoring approval mode + per-tool hints + scope
/// allowlists. Returns a [`DispatchOutcome`] with the consumer-facing
/// events and the messages to append to history.
pub(crate) async fn dispatch(
    pending: Vec<PendingToolCall>,
    mut ctx: DispatchContext<'_>,
) -> Result<DispatchOutcome, TurnError> {
    let mut out = DispatchOutcome {
        events: Vec::with_capacity(pending.len()),
        tool_messages: Vec::with_capacity(pending.len()),
        executed: 0,
        cancelled: false,
    };

    for pc in pending {
        let tool = match ctx.tools.get(&pc.name) {
            Some(t) => t.clone(),
            None => {
                push_soft_error(
                    &pc.id,
                    format!("tool '{}' is not registered", pc.name),
                    &mut out,
                );
                continue;
            }
        };

        let args: serde_json::Value = match serde_json::from_str(&pc.arguments) {
            Ok(v) => v,
            Err(e) => {
                push_soft_error(
                    &pc.id,
                    format!("arguments for '{}' were not valid JSON: {e}", pc.name),
                    &mut out,
                );
                continue;
            }
        };

        let gate = check_approval(&pc, tool.as_ref(), &args, &mut ctx, &mut out).await?;
        match gate {
            Gate::Allow => {}
            Gate::Deny(msg) => {
                push_soft_error(&pc.id, msg, &mut out);
                continue;
            }
            Gate::Cancel => {
                out.cancelled = true;
                return Ok(out);
            }
        }

        let (tool_events_tx, _tool_events_rx) = mpsc::channel(16);
        let tool_ctx = ToolContext {
            working_dir: ctx.working_dir,
            project_root: None,
            session_id: ctx.session_id,
            events: tool_events_tx,
        };

        match tool.execute(args, &tool_ctx).await {
            Ok(result) => {
                out.events.push(AgentEvent::ToolResult {
                    id: pc.id.clone(),
                    content: result.content.clone(),
                    is_error: result.is_error,
                });
                out.tool_messages.push(Message::tool(pc.id, result.content));
                out.executed += 1;
            }
            Err(err) => {
                push_soft_error(
                    &pc.id,
                    format!("tool '{}' failed: {err}", pc.name),
                    &mut out,
                );
            }
        }
    }

    Ok(out)
}

/// Resolve approval for `pc`. Returns [`Gate::Allow`] when the call may
/// proceed, [`Gate::Deny`] when the user denies (handled as soft failure),
/// or [`Gate::Cancel`] when the user cancels the entire turn.
async fn check_approval(
    pc: &PendingToolCall,
    tool: &dyn elal_tools::Tool,
    args: &serde_json::Value,
    ctx: &mut DispatchContext<'_>,
    out: &mut DispatchOutcome,
) -> Result<Gate, TurnError> {
    // Session / turn allowlists short-circuit the resolver entirely — the
    // user already opted in.
    if ctx.approved_for_session.contains(&pc.name) || ctx.approved_for_turn.contains(&pc.name) {
        return Ok(Gate::Allow);
    }

    let decision = resolve(ctx.mode, tool.spec().approval_hint, tool.name(), args);
    match decision {
        ApprovalDecision::Allow => Ok(Gate::Allow),
        ApprovalDecision::Prompt => prompt_user(pc, tool, args, ctx, out).await,
    }
}

/// Emit `AgentEvent::ApprovalRequired`, wait for the matching
/// `UserAction::ApprovalResponse`, update allowlists, and translate the
/// response into a [`Gate`].
async fn prompt_user(
    pc: &PendingToolCall,
    tool: &dyn elal_tools::Tool,
    args: &serde_json::Value,
    ctx: &mut DispatchContext<'_>,
    out: &mut DispatchOutcome,
) -> Result<Gate, TurnError> {
    let request_id = format!("{}-{}", ctx.session_id, pc.id);
    let args_preview = serde_json::to_string_pretty(args).unwrap_or_else(|_| args.to_string());
    let request = ApprovalRequest {
        request_id: request_id.clone(),
        tool_name: pc.name.clone(),
        describe_action: tool.describe_action(args),
        args_preview,
    };

    // Flush any buffered ToolResult / Text events first so the UI shows
    // them before the modal. Then emit the prompt.
    for event in out.events.drain(..) {
        ctx.events_tx
            .send(event)
            .await
            .map_err(|_| TurnError::ChannelClosed)?;
    }
    ctx.events_tx
        .send(AgentEvent::ApprovalRequired(request))
        .await
        .map_err(|_| TurnError::ChannelClosed)?;

    // Wait for the matching response. Any non-matching message is ignored
    // — lets consumers queue unrelated actions without confusing the loop.
    loop {
        let action = ctx
            .actions_rx
            .recv()
            .await
            .ok_or(TurnError::ChannelClosed)?;
        let UserAction::ApprovalResponse {
            request_id: rid,
            decision,
            scope,
        } = action;
        if rid != request_id {
            continue;
        }
        return Ok(translate_response(&pc.name, decision, scope, ctx));
    }
}

fn translate_response(
    tool_name: &str,
    decision: ApprovalDecisionValue,
    scope: ApprovalScopeValue,
    ctx: &mut DispatchContext<'_>,
) -> Gate {
    match decision {
        ApprovalDecisionValue::Approve => {
            match scope {
                ApprovalScopeValue::Session | ApprovalScopeValue::Tool => {
                    ctx.approved_for_session.insert(tool_name.to_string());
                }
                ApprovalScopeValue::Turn => {
                    ctx.approved_for_turn.insert(tool_name.to_string());
                }
                // Once / PathPrefix / Host — transient for this
                // invocation only. Path and Host scopes would benefit
                // from richer matching; that's a follow-up when tools
                // start exposing path/host metadata per call.
                _ => {}
            }
            Gate::Allow
        }
        ApprovalDecisionValue::Deny => Gate::Deny(format!("tool '{tool_name}' denied by user")),
        ApprovalDecisionValue::Cancel => Gate::Cancel,
    }
}

fn push_soft_error(id: &str, message: String, out: &mut DispatchOutcome) {
    out.events.push(AgentEvent::ToolResult {
        id: id.to_string(),
        content: message.clone(),
        is_error: true,
    });
    out.tool_messages
        .push(Message::tool(id.to_string(), message));
}

#[cfg(test)]
mod tests;
