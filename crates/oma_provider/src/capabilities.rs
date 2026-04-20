use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::sampling::SamplingControls;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen25,
    Qwen25Coder,
    Qwen3,
    Llama3,
    Mistral,
    DeepSeekCoderV2,
    DeepSeekR1,
    Phi35,
    Hermes3,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolCallingTier {
    /// The model emits tool_calls via its native chat template correctly.
    Native,
    /// Tool calls must be elicited through ReAct-style prompting; lower reliability.
    Prompted,
    /// Tool calling is not supported in any reliable form.
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatTemplateOverride {
    /// Jinja-style override for the native `chat_template` in GGUF metadata.
    pub template: String,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SpecialTokens {
    pub thinking_start: Option<String>,
    pub thinking_end: Option<String>,
    pub tool_call_prefix: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCapabilities {
    pub family: ModelFamily,
    pub supports_tool_calls: ToolCallingTier,
    pub supports_reasoning: bool,
    pub supports_grammar: bool,
    pub native_max_context: u32,
    pub recommended_sampling: SamplingControls,
    pub chat_template: Option<ChatTemplateOverride>,
    pub special_tokens: SpecialTokens,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            family: ModelFamily::Unknown,
            supports_tool_calls: ToolCallingTier::Prompted,
            supports_reasoning: false,
            supports_grammar: false,
            native_max_context: 4096,
            recommended_sampling: SamplingControls::default(),
            chat_template: None,
            special_tokens: SpecialTokens::default(),
        }
    }
}

impl ModelCapabilities {
    fn qwen25_coder() -> Self {
        Self {
            family: ModelFamily::Qwen25Coder,
            supports_tool_calls: ToolCallingTier::Native,
            supports_reasoning: false,
            supports_grammar: true,
            native_max_context: 32_768,
            recommended_sampling: SamplingControls {
                temperature: 0.7,
                top_p: 0.8,
                top_k: 20,
                min_p: 0.0,
                repeat_penalty: 1.05,
                seed: 0xC0DE_C0DE,
            },
            chat_template: None,
            special_tokens: SpecialTokens::default(),
        }
    }

    fn qwen25() -> Self {
        Self {
            family: ModelFamily::Qwen25,
            ..Self::qwen25_coder()
        }
    }

    fn qwen3() -> Self {
        Self {
            family: ModelFamily::Qwen3,
            supports_reasoning: true,
            special_tokens: SpecialTokens {
                thinking_start: Some("<think>".into()),
                thinking_end: Some("</think>".into()),
                tool_call_prefix: None,
            },
            ..Self::qwen25_coder()
        }
    }

    fn mistral() -> Self {
        Self {
            family: ModelFamily::Mistral,
            supports_tool_calls: ToolCallingTier::Native,
            supports_reasoning: false,
            supports_grammar: true,
            native_max_context: 32_768,
            recommended_sampling: SamplingControls::default(),
            chat_template: None,
            special_tokens: SpecialTokens::default(),
        }
    }

    fn deepseek_coder_v2() -> Self {
        Self {
            family: ModelFamily::DeepSeekCoderV2,
            supports_tool_calls: ToolCallingTier::Native,
            supports_reasoning: false,
            supports_grammar: true,
            native_max_context: 128_000,
            recommended_sampling: SamplingControls::default(),
            chat_template: None,
            special_tokens: SpecialTokens::default(),
        }
    }

    fn deepseek_r1_distill() -> Self {
        Self {
            family: ModelFamily::DeepSeekR1,
            supports_tool_calls: ToolCallingTier::Prompted,
            supports_reasoning: true,
            supports_grammar: true,
            native_max_context: 64_000,
            recommended_sampling: SamplingControls {
                temperature: 0.6,
                top_p: 0.95,
                top_k: 40,
                min_p: 0.05,
                repeat_penalty: 1.0,
                seed: 0xC0DE_C0DE,
            },
            chat_template: None,
            special_tokens: SpecialTokens {
                thinking_start: Some("<think>".into()),
                thinking_end: Some("</think>".into()),
                tool_call_prefix: None,
            },
        }
    }

    fn llama3() -> Self {
        Self {
            family: ModelFamily::Llama3,
            supports_tool_calls: ToolCallingTier::Native,
            supports_reasoning: false,
            supports_grammar: true,
            native_max_context: 131_072,
            recommended_sampling: SamplingControls::default(),
            chat_template: None,
            special_tokens: SpecialTokens::default(),
        }
    }

    fn phi35() -> Self {
        Self {
            family: ModelFamily::Phi35,
            supports_tool_calls: ToolCallingTier::Prompted,
            supports_reasoning: false,
            supports_grammar: true,
            native_max_context: 131_072,
            recommended_sampling: SamplingControls::default(),
            chat_template: None,
            special_tokens: SpecialTokens::default(),
        }
    }

    fn hermes3() -> Self {
        Self {
            family: ModelFamily::Hermes3,
            supports_tool_calls: ToolCallingTier::Native,
            supports_reasoning: false,
            supports_grammar: true,
            native_max_context: 128_000,
            recommended_sampling: SamplingControls::default(),
            chat_template: None,
            special_tokens: SpecialTokens::default(),
        }
    }
}

/// Resolve conservative default capabilities for a GGUF file by filename pattern.
///
/// This is intentionally dumb: it matches substrings in the file stem.
/// Dynamic resolution from GGUF metadata lives in #3b.
pub fn resolve_by_filename(path: &Path) -> ModelCapabilities {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    if stem.contains("deepseek-r1") || stem.contains("r1-distill") {
        ModelCapabilities::deepseek_r1_distill()
    } else if stem.contains("deepseek-coder-v2") {
        ModelCapabilities::deepseek_coder_v2()
    } else if stem.contains("qwen3") {
        ModelCapabilities::qwen3()
    } else if stem.contains("qwen2.5-coder") || stem.contains("qwen2_5-coder") {
        ModelCapabilities::qwen25_coder()
    } else if stem.contains("qwen2.5") || stem.contains("qwen2_5") {
        ModelCapabilities::qwen25()
    } else if stem.contains("mistral-small") || stem.contains("mistral") {
        ModelCapabilities::mistral()
    } else if stem.contains("llama-3") || stem.contains("meta-llama-3") {
        ModelCapabilities::llama3()
    } else if stem.contains("phi-3") || stem.contains("phi3") {
        ModelCapabilities::phi35()
    } else if stem.contains("hermes-3") || stem.contains("hermes3") {
        ModelCapabilities::hermes3()
    } else {
        ModelCapabilities::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn resolve(name: &str) -> ModelFamily {
        resolve_by_filename(&PathBuf::from(name)).family
    }

    #[test]
    fn qwen25_coder_resolved() {
        assert_eq!(
            resolve("Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf"),
            ModelFamily::Qwen25Coder,
        );
    }

    #[test]
    fn qwen25_base_resolved() {
        assert_eq!(
            resolve("Qwen2.5-7B-Instruct-Q5_K_M.gguf"),
            ModelFamily::Qwen25,
        );
    }

    #[test]
    fn deepseek_r1_distill_resolved() {
        assert_eq!(
            resolve("DeepSeek-R1-Distill-Qwen-14B-Q5_K_M.gguf"),
            ModelFamily::DeepSeekR1,
        );
        let caps = resolve_by_filename(&PathBuf::from("DeepSeek-R1-Distill-Qwen-14B-Q5_K_M.gguf"));
        assert!(caps.supports_reasoning);
        assert_eq!(
            caps.special_tokens.thinking_start.as_deref(),
            Some("<think>")
        );
    }

    #[test]
    fn llama3_resolved() {
        assert_eq!(
            resolve("Meta-Llama-3-8B-Instruct-Q5_K_M.gguf"),
            ModelFamily::Llama3,
        );
    }

    #[test]
    fn unknown_model_falls_back_to_conservative_defaults() {
        let caps = resolve_by_filename(&PathBuf::from("mystery-model-v9000.gguf"));
        assert_eq!(caps.family, ModelFamily::Unknown);
        assert_eq!(caps.supports_tool_calls, ToolCallingTier::Prompted);
        assert!(!caps.supports_reasoning);
    }

    #[test]
    fn qwen3_has_reasoning() {
        let caps = resolve_by_filename(&PathBuf::from("Qwen3-14B-Instruct-Q5_K_M.gguf"));
        assert_eq!(caps.family, ModelFamily::Qwen3);
        assert!(caps.supports_reasoning);
    }
}
