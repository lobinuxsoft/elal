//! The core [`Tool`] trait every capability must implement.

use async_trait::async_trait;
use oma_protocol::ToolDefinition;

use crate::context::ToolContext;
use crate::result::{ToolError, ToolResult};
use crate::spec::ToolSpec;

/// Contract every tool — built-in, MCP-sourced, or user-provided — satisfies.
///
/// Design notes:
/// - Async `execute` because most tools hit I/O (files, subprocesses,
///   network). Synchronous tools simply return immediately.
/// - [`Self::definition`] produces the JSON-Schema-bearing contract the
///   model sees. [`Self::spec`] is the runtime-facing descriptor used by the
///   registry and approval resolver.
/// - [`Self::describe_action`] generates the one-line preview shown in the
///   approval modal — separated from `execute` so the UI can render it
///   without running the tool.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Stable lowercase name used by the LLM to invoke this tool.
    ///
    /// Must match the `name` field in the tool's [`ToolSpec`] and the
    /// `function.name` in its [`ToolDefinition`].
    fn name(&self) -> &str;

    /// Model-visible definition: name, description, JSON-schema parameters.
    fn definition(&self) -> ToolDefinition;

    /// Runtime descriptor — tier, approval hint, side effects.
    fn spec(&self) -> ToolSpec;

    /// Human-readable summary of what this invocation would do.
    ///
    /// Shown in the approval modal so the user can decide without reading
    /// JSON. Pure function over `args`; must not perform I/O.
    fn describe_action(&self, args: &serde_json::Value) -> String;

    /// Execute the tool against validated arguments.
    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext<'_>,
    ) -> Result<ToolResult, ToolError>;
}
