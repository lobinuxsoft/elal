use std::path::{Path, PathBuf};

use elal_protocol::ApprovalMode;
use serde::{Deserialize, Serialize};

use crate::error::{ElalError, Result};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GlobalConfig {
    pub default_model: Option<String>,
    pub models_dir: PathBuf,
    pub max_context_tokens: usize,
    /// GPU layers to offload. `-1` means "all", `0` means CPU only.
    pub n_gpu_layers: i32,
    pub approval_mode: ApprovalMode,
}

const APP_DIR: &str = "elal";

/// Directory used before the rename to `elal`. Resolved only when the current
/// one is absent, so an existing install keeps its models and settings instead
/// of coming up empty after an upgrade.
const LEGACY_DIR: &str = "oh-my-agent";

/// Returns `base/elal`, falling back to the pre-rename directory when only that
/// one exists.
fn app_dir(base: PathBuf) -> PathBuf {
    let current = base.join(APP_DIR);
    let legacy = base.join(LEGACY_DIR);
    if !current.exists() && legacy.exists() {
        return legacy;
    }
    current
}

impl Default for GlobalConfig {
    fn default() -> Self {
        let models_dir = dirs::data_dir()
            .map(|d| app_dir(d).join("models"))
            .unwrap_or_else(|| PathBuf::from("./models"));
        Self {
            default_model: None,
            models_dir,
            max_context_tokens: 16_384,
            n_gpu_layers: -1,
            approval_mode: ApprovalMode::default(),
        }
    }
}

impl GlobalConfig {
    pub fn global_config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| app_dir(d).join("config.toml"))
    }

    pub fn load_or_default() -> Result<Self> {
        match Self::global_config_path() {
            Some(path) if path.exists() => Self::load_from(&path),
            _ => Ok(Self::default()),
        }
    }

    pub fn load_from(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let cfg = toml::from_str(&raw)?;
        Ok(cfg)
    }

    pub fn save_to(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let raw = toml::to_string_pretty(self)?;
        std::fs::write(path, raw)?;
        Ok(())
    }

    /// Merge overrides from a project config on top of this one.
    pub fn apply_project_overrides(&mut self, project: &ProjectConfig) {
        if let Some(agent) = &project.agent {
            if let Some(m) = &agent.default_model {
                self.default_model = Some(m.clone());
            }
            if let Some(n) = agent.max_context_tokens {
                self.max_context_tokens = n;
            }
            if let Some(n) = agent.n_gpu_layers {
                self.n_gpu_layers = n;
            }
            if let Some(m) = agent.approval_mode {
                self.approval_mode = m;
            }
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectConfig {
    pub agent: Option<AgentOverrides>,
    pub tools: Option<ToolsConfig>,
    pub context: Option<ContextConfig>,
}

impl ProjectConfig {
    pub const FILE_NAME: &str = ".elal.toml";

    /// Project config name used before the rename. Still discovered so existing
    /// checkouts keep their per-project settings without an edit.
    pub const LEGACY_NAME: &str = ".oh-my-agent.toml";

    /// Walk up from `start` looking for a project config file.
    pub fn discover(start: &Path) -> Option<PathBuf> {
        let mut cur = Some(start);
        while let Some(dir) = cur {
            for name in [Self::FILE_NAME, Self::LEGACY_NAME] {
                let candidate = dir.join(name);
                if candidate.exists() {
                    return Some(candidate);
                }
            }
            cur = dir.parent();
        }
        None
    }

    pub fn load_from(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let cfg = toml::from_str(&raw)?;
        Ok(cfg)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentOverrides {
    pub default_model: Option<String>,
    pub max_context_tokens: Option<usize>,
    pub n_gpu_layers: Option<i32>,
    pub approval_mode: Option<ApprovalMode>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolsConfig {
    pub disabled: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextConfig {
    pub exclude_globs: Option<Vec<String>>,
}

/// Load the effective config: global defaults, then project overrides if a
/// `.elal.toml` is found walking upward from `cwd`.
pub fn load_effective(cwd: &Path) -> Result<GlobalConfig> {
    let mut cfg = GlobalConfig::load_or_default()?;
    if let Some(project_path) = ProjectConfig::discover(cwd) {
        let project = ProjectConfig::load_from(&project_path)?;
        cfg.apply_project_overrides(&project);
    }
    Ok(cfg)
}

/// Surface a stable error type for bad config values (validation layer).
pub fn invalid(msg: impl Into<String>) -> ElalError {
    ElalError::Config(msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn default_config_has_sensible_values() {
        let c = GlobalConfig::default();
        assert_eq!(c.max_context_tokens, 16_384);
        assert_eq!(c.n_gpu_layers, -1);
        assert_eq!(c.approval_mode, ApprovalMode::Smart);
    }

    #[test]
    fn global_config_roundtrip_via_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        let original = GlobalConfig {
            default_model: Some("Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf".into()),
            models_dir: dir.path().join("models"),
            max_context_tokens: 32_768,
            n_gpu_layers: -1,
            approval_mode: ApprovalMode::Always,
        };
        original.save_to(&path).unwrap();
        let loaded = GlobalConfig::load_from(&path).unwrap();
        assert_eq!(original, loaded);
    }

    #[test]
    fn project_overrides_replace_global_fields() {
        let mut global = GlobalConfig::default();
        let project = ProjectConfig {
            agent: Some(AgentOverrides {
                default_model: Some("override.gguf".into()),
                max_context_tokens: Some(8_192),
                n_gpu_layers: None,
                approval_mode: Some(ApprovalMode::Never),
            }),
            tools: None,
            context: None,
        };
        global.apply_project_overrides(&project);
        assert_eq!(global.default_model.as_deref(), Some("override.gguf"));
        assert_eq!(global.max_context_tokens, 8_192);
        assert_eq!(global.n_gpu_layers, -1, "n_gpu_layers not overridden");
        assert_eq!(global.approval_mode, ApprovalMode::Never);
    }

    #[test]
    fn project_discover_walks_up() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("a").join("b").join("c");
        std::fs::create_dir_all(&nested).unwrap();
        let cfg_path = dir.path().join(ProjectConfig::FILE_NAME);
        std::fs::write(&cfg_path, "").unwrap();
        let found = ProjectConfig::discover(&nested);
        assert_eq!(found.as_deref(), Some(cfg_path.as_path()));
    }

    #[test]
    fn project_discover_returns_none_when_absent() {
        let dir = TempDir::new().unwrap();
        assert!(ProjectConfig::discover(dir.path()).is_none());
    }

    #[test]
    fn load_effective_merges_when_project_exists() {
        let dir = TempDir::new().unwrap();
        let cfg_path = dir.path().join(ProjectConfig::FILE_NAME);
        std::fs::write(
            &cfg_path,
            "[agent]\ndefault_model = \"local.gguf\"\nmax_context_tokens = 4096\n",
        )
        .unwrap();
        let cfg = load_effective(dir.path()).unwrap();
        assert_eq!(cfg.default_model.as_deref(), Some("local.gguf"));
        assert_eq!(cfg.max_context_tokens, 4096);
    }
}
