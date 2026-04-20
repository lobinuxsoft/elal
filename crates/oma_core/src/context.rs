use std::path::{Path, PathBuf};
use std::process::Command;

use oma_protocol::SessionId;

#[derive(Debug, Clone)]
pub struct AgentContext {
    pub working_dir: PathBuf,
    pub project_root: Option<PathBuf>,
    pub git_info: Option<GitInfo>,
    pub os_info: OsInfo,
    pub session_id: SessionId,
}

impl AgentContext {
    /// Discover context rooted at `working_dir`.
    ///
    /// `project_root` walks up looking for `.oh-my-agent.toml`; falls back to
    /// the git root if that's available; otherwise `None`.
    pub fn discover(working_dir: PathBuf) -> Self {
        let git_info = GitInfo::discover(&working_dir);
        let project_root = crate::config::ProjectConfig::discover(&working_dir)
            .and_then(|p| p.parent().map(PathBuf::from))
            .or_else(|| git_info.as_ref().map(|g| g.root.clone()));

        Self {
            working_dir,
            project_root,
            git_info,
            os_info: OsInfo::detect(),
            session_id: SessionId::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitInfo {
    pub root: PathBuf,
    pub branch: Option<String>,
}

impl GitInfo {
    pub fn discover(working_dir: &Path) -> Option<Self> {
        let root = run_git(working_dir, &["rev-parse", "--show-toplevel"])?;
        let branch = run_git(working_dir, &["rev-parse", "--abbrev-ref", "HEAD"]);
        Some(Self {
            root: PathBuf::from(root),
            branch: branch.filter(|b| b != "HEAD"),
        })
    }
}

fn run_git(cwd: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .current_dir(cwd)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let trimmed = s.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OsInfo {
    pub family: String,
    pub os: String,
    pub arch: String,
}

impl OsInfo {
    pub fn detect() -> Self {
        Self {
            family: std::env::consts::FAMILY.to_string(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn discover_returns_values_in_temp_dir() {
        let dir = TempDir::new().unwrap();
        let ctx = AgentContext::discover(dir.path().to_path_buf());
        assert_eq!(ctx.working_dir, dir.path());
        assert!(!ctx.os_info.os.is_empty());
        assert!(!ctx.os_info.arch.is_empty());
    }

    #[test]
    fn git_info_absent_outside_repo() {
        let dir = TempDir::new().unwrap();
        assert!(GitInfo::discover(dir.path()).is_none());
    }

    #[test]
    fn os_info_detected() {
        let info = OsInfo::detect();
        assert!(!info.family.is_empty());
        assert!(!info.os.is_empty());
        assert!(!info.arch.is_empty());
    }
}
