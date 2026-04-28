//! `update_plan` built-in — structured plan tool intended for planner / orchestrator
//! roles in a multi-model agent setup.
//!
//! Differs from [`todo_write`](crate::todo_write) in two ways: it accepts an
//! optional `explanation` paragraph that frames the plan, and it is meant to
//! be emitted by a *planning* model (potentially distinct from the worker
//! model) before any execution happens. The status vocabulary is the same so
//! a downstream worker can echo the same items back via `todo_write`.

use async_trait::async_trait;
use oma_protocol::ToolDefinition;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::context::ToolContext;
use crate::result::{ToolError, ToolResult};
use crate::spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
use crate::trait_def::Tool;

const DESCRIPTION: &str = "Emit a structured plan with an optional explanation paragraph and an ordered \
                           list of `{step, status}` items. Use this when planning multi-step work before \
                           starting execution. At most one step may be `in_progress` at a time. The plan is \
                           rendered back as a formatted checklist for both the user and any downstream model.";

/// Status values shared with [`todo_write`](crate::todo_write::TodoStatus).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Pending,
    InProgress,
    Completed,
}

/// One step in a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanItem {
    pub step: String,
    pub status: PlanStatus,
}

/// Tool: lets a planner role emit a structured plan with optional context.
pub struct UpdatePlanTool;

#[async_trait]
impl Tool for UpdatePlanTool {
    fn name(&self) -> &str {
        "update_plan"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "update_plan",
            DESCRIPTION,
            json!({
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "Optional paragraph framing the plan."
                    },
                    "plan": {
                        "type": "array",
                        "description": "Ordered list of plan steps.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": { "type": "string" },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"]
                                }
                            },
                            "required": ["step", "status"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["plan"],
                "additionalProperties": false
            }),
        )
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "update_plan",
            tier: ToolTier::Read,
            approval_hint: ApprovalHint::Never,
            side_effects: SideEffects::None,
        }
    }

    fn describe_action(&self, args: &serde_json::Value) -> String {
        let count = args
            .get("plan")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        format!("Update plan ({count} steps)")
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _ctx: &ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let explanation = args
            .get("explanation")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let plan: Vec<PlanItem> = serde_json::from_value(
            args.get("plan").cloned().unwrap_or(json!([])),
        )
        .map_err(|e| ToolError::InvalidArgs {
            name: "update_plan".into(),
            reason: format!("`plan` did not match the schema: {e}"),
        })?;

        let in_progress = plan
            .iter()
            .filter(|p| p.status == PlanStatus::InProgress)
            .count();
        if in_progress > 1 {
            return Ok(ToolResult::soft_error(format!(
                "{in_progress} steps are in_progress; at most one is allowed at a time."
            )));
        }

        let body = render_plan(&plan);
        let content = if explanation.is_empty() {
            body
        } else {
            format!("{explanation}\n\n{body}")
        };
        let structured = json!({
            "explanation": explanation,
            "plan": plan,
        });
        Ok(ToolResult::ok(content).with_structured(structured))
    }
}

fn render_plan(plan: &[PlanItem]) -> String {
    if plan.is_empty() {
        return "(empty plan)".into();
    }
    let mut out = String::new();
    for (i, p) in plan.iter().enumerate() {
        let mark = match p.status {
            PlanStatus::Pending => "[ ]",
            PlanStatus::InProgress => "[~]",
            PlanStatus::Completed => "[x]",
        };
        out.push_str(&format!("{}. {} {}\n", i + 1, mark, p.step));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use tokio::sync::mpsc;

    fn ctx<'a>(wd: &'a Path) -> (ToolContext<'a>, mpsc::Receiver<crate::context::ToolEvent>) {
        let (tx, rx) = mpsc::channel(4);
        (
            ToolContext {
                working_dir: wd,
                project_root: None,
                session_id: oma_protocol::SessionId::new(),
                events: tx,
            },
            rx,
        )
    }

    #[tokio::test]
    async fn renders_plan_with_explanation() {
        let wd = PathBuf::from("/tmp");
        let (cx, _rx) = ctx(&wd);
        let args = json!({
            "explanation": "Refactor the parser before adding the new dialect.",
            "plan": [
                { "step": "read existing parser",  "status": "completed" },
                { "step": "extract token kinds",   "status": "in_progress" },
                { "step": "add new dialect",       "status": "pending" }
            ]
        });
        let out = UpdatePlanTool.execute(args, &cx).await.unwrap();
        assert!(!out.is_error);
        assert!(out.content.starts_with("Refactor the parser"));
        assert!(out.content.contains("[~] extract token kinds"));
        assert!(out.structured.is_some());
    }

    #[tokio::test]
    async fn empty_explanation_omits_paragraph() {
        let wd = PathBuf::from("/tmp");
        let (cx, _rx) = ctx(&wd);
        let args = json!({
            "explanation": "   ",
            "plan": [{ "step": "a", "status": "pending" }]
        });
        let out = UpdatePlanTool.execute(args, &cx).await.unwrap();
        assert!(out.content.starts_with("1. [ ] a"));
    }

    #[tokio::test]
    async fn rejects_multiple_in_progress() {
        let wd = PathBuf::from("/tmp");
        let (cx, _rx) = ctx(&wd);
        let args = json!({
            "plan": [
                { "step": "a", "status": "in_progress" },
                { "step": "b", "status": "in_progress" }
            ]
        });
        let out = UpdatePlanTool.execute(args, &cx).await.unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("at most one"));
    }

    #[tokio::test]
    async fn malformed_plan_is_invalid_args() {
        let wd = PathBuf::from("/tmp");
        let (cx, _rx) = ctx(&wd);
        let args = json!({ "plan": [{ "step": "x", "status": "blocked" }] });
        let err = UpdatePlanTool.execute(args, &cx).await.unwrap_err();
        matches!(err, ToolError::InvalidArgs { .. })
            .then_some(())
            .expect("expected InvalidArgs");
    }

    #[test]
    fn describe_action_reports_step_count() {
        let args = json!({
            "plan": [
                { "step": "a", "status": "pending" },
                { "step": "b", "status": "pending" }
            ]
        });
        assert_eq!(
            UpdatePlanTool.describe_action(&args),
            "Update plan (2 steps)"
        );
    }

    #[test]
    fn spec_is_pure_no_approval() {
        let s = UpdatePlanTool.spec();
        assert_eq!(s.name, "update_plan");
        assert_eq!(s.approval_hint, ApprovalHint::Never);
        assert_eq!(s.side_effects, SideEffects::None);
    }
}
