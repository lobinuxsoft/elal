use thiserror::Error;

#[derive(Debug, Error)]
pub enum ElalError {
    #[error("config error: {0}")]
    Config(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("toml parse error: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("toml serialize error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    /// Structural failures from session replay/persistence — empty rollout,
    /// missing meta header, schema-version mismatch, etc.
    #[error("session: {0}")]
    Session(String),
    // Sub-crate error variants will be added here as crates are implemented:
    // Llm(#[from] elal_provider::LlmError)  — #3
    // Tool(#[from] elal_tools::ToolError)   — #4
}

pub type Result<T> = std::result::Result<T, ElalError>;
