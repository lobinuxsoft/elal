//! In-memory registry mapping tool names to their implementations.

use std::collections::HashMap;
use std::sync::Arc;

use elal_protocol::ToolDefinition;

use crate::trait_def::Tool;

/// In-memory registry of runtime tools.
///
/// The registry owns `Arc<dyn Tool>` handles so implementations can be
/// shared across async tasks without cloning the tool's internal state.
/// Lookups are by the tool's [`Tool::name`] which must be unique.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Inserts (or replaces) a tool keyed by its [`Tool::name`].
    ///
    /// Returns the previous entry under the same name if one existed, so
    /// callers can log accidental collisions during startup.
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Option<Arc<dyn Tool>> {
        let key = tool.name().to_string();
        self.tools.insert(key, tool)
    }

    /// Looks up a tool by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Returns `true` when a tool with this name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Collects the model-visible definitions for every registered tool.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    /// Registers the default set of built-in tools.
    ///
    /// Populated incrementally by Phase 2b (#5). Downstream crates
    /// (`elal_tasks`, etc.) register their own tools explicitly against this
    /// registry and are NOT included here to keep `elal_tools` free of
    /// reverse dependencies.
    pub fn register_defaults(&mut self) {
        self.register(Arc::new(crate::read::ReadTool));
        self.register(Arc::new(crate::list_dir::ListDirTool));
        self.register(Arc::new(crate::shell_exec::ShellExecTool));
        self.register(Arc::new(crate::glob::GlobTool));
        self.register(Arc::new(crate::grep::GrepTool));
        self.register(Arc::new(crate::file_write::FileWriteTool));
        self.register(Arc::new(crate::apply_patch::ApplyPatchTool));
        self.register(Arc::new(crate::edit::EditTool));
        self.register(Arc::new(crate::webfetch::WebFetchTool));
        self.register(Arc::new(crate::websearch::WebSearchTool::new()));
        self.register(Arc::new(crate::git::GitTool));
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ToolContext;
    use crate::result::{ToolError, ToolResult};
    use crate::spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
    use async_trait::async_trait;

    struct DummyTool {
        name: &'static str,
    }

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &str {
            self.name
        }

        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                self.name,
                "dummy tool used in tests",
                serde_json::json!({"type": "object"}),
            )
        }

        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: self.name,
                tier: ToolTier::Read,
                approval_hint: ApprovalHint::Never,
                side_effects: SideEffects::None,
            }
        }

        fn describe_action(&self, _args: &serde_json::Value) -> String {
            format!("run {}", self.name)
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolContext<'_>,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::ok("ok"))
        }
    }

    #[test]
    fn new_registry_is_empty() {
        let r = ToolRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn register_inserts_tool() {
        let mut r = ToolRegistry::new();
        r.register(Arc::new(DummyTool { name: "read" }));
        assert!(r.contains("read"));
        assert_eq!(r.len(), 1);
        assert!(r.get("read").is_some());
        assert!(r.get("missing").is_none());
    }

    #[test]
    fn register_same_name_replaces_and_returns_previous() {
        let mut r = ToolRegistry::new();
        let a = Arc::new(DummyTool { name: "read" });
        let b = Arc::new(DummyTool { name: "read" });
        assert!(r.register(a.clone()).is_none());
        let prev = r.register(b);
        assert!(prev.is_some());
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn definitions_returns_one_entry_per_tool() {
        let mut r = ToolRegistry::new();
        r.register(Arc::new(DummyTool { name: "read" }));
        r.register(Arc::new(DummyTool { name: "write" }));
        let defs = r.definitions();
        assert_eq!(defs.len(), 2);
        let names: Vec<_> = defs.iter().map(|d| d.function.name.clone()).collect();
        assert!(names.contains(&"read".to_string()));
        assert!(names.contains(&"write".to_string()));
    }

    #[test]
    fn register_defaults_registers_phase_2b_builtins() {
        let mut r = ToolRegistry::new();
        r.register_defaults();
        let expected = [
            "read",
            "list_dir",
            "shell_exec",
            "glob",
            "grep",
            "write",
            "apply_patch",
            "edit",
            "webfetch",
            "websearch",
            "git",
        ];
        for name in expected {
            assert!(r.contains(name), "expected built-in '{name}' registered");
        }
        assert_eq!(r.len(), expected.len(), "exactly 11 built-ins expected");
    }

    #[test]
    fn register_defaults_tool_names_match_spec_names() {
        let mut r = ToolRegistry::new();
        r.register_defaults();
        for def in r.definitions() {
            let tool = r
                .get(&def.function.name)
                .expect("tool retrievable by its declared name");
            assert_eq!(
                tool.name(),
                def.function.name,
                "Tool::name and ToolDefinition.function.name must agree"
            );
            assert_eq!(
                tool.spec().name,
                def.function.name,
                "ToolSpec.name and ToolDefinition.function.name must agree"
            );
        }
    }

    #[test]
    fn every_registered_tool_has_sensible_spec() {
        let mut r = ToolRegistry::new();
        r.register_defaults();
        for def in r.definitions() {
            let tool = r.get(&def.function.name).unwrap();
            let spec = tool.spec();
            if spec.approval_hint == crate::spec::ApprovalHint::Maybe {
                assert_eq!(
                    def.function.name, "git",
                    "only 'git' is expected to use ApprovalHint::Maybe in Phase 2b"
                );
            }
        }
    }

    #[test]
    fn register_defaults_tools_have_valid_schemas() {
        let mut r = ToolRegistry::new();
        r.register_defaults();
        for def in r.definitions() {
            assert!(
                def.function.parameters.is_object(),
                "tool '{}' produced non-object JSON schema",
                def.function.name
            );
            assert!(
                !def.function.description.is_empty(),
                "tool '{}' has empty description",
                def.function.name
            );
        }
    }
}
