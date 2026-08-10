//! Upgrade filename-based [`ModelCapabilities`](crate::ModelCapabilities) with
//! metadata read from the loaded GGUF.
//!
//! The filename heuristic in [`resolve_by_filename`](crate::resolve_by_filename)
//! is a cheap first pass. This module complements it with the authoritative
//! truth baked into the file itself (`general.architecture`, `general.name`,
//! …) using [`LlamaModel::meta_val_str`]. When the metadata unambiguously
//! identifies a known family, the capabilities returned here override the
//! filename fallback.

use llama_cpp_2::model::LlamaModel;

use crate::capabilities::ModelCapabilities;

/// Upgrade `fallback` with whatever the model's metadata tells us about its
/// family. If the metadata is ambiguous, unavailable, or unrecognised, the
/// fallback is returned untouched so filename-based resolution stays in effect.
pub fn resolve_from_model(model: &LlamaModel, fallback: ModelCapabilities) -> ModelCapabilities {
    let arch = read_meta(model, "general.architecture");
    let name = read_meta(model, "general.name");
    let basename = read_meta(model, "general.basename");

    match family_from_metadata(arch.as_deref(), name.as_deref(), basename.as_deref()) {
        Some(caps) => caps,
        None => fallback,
    }
}

/// Reads a GGUF metadata value as a `String`. Returns `None` when the key is
/// missing, empty, or unreadable — the caller treats any of those as "no
/// signal" and falls back to the filename-based capabilities.
fn read_meta(model: &LlamaModel, key: &str) -> Option<String> {
    model
        .meta_val_str(key)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Classify the model from its architecture, name, and basename metadata. The
/// matchers are case-insensitive and tolerant of the common variants GGUF
/// authors use in the wild.
fn family_from_metadata(
    arch: Option<&str>,
    name: Option<&str>,
    basename: Option<&str>,
) -> Option<ModelCapabilities> {
    let arch_lc = arch.map(str::to_lowercase).unwrap_or_default();
    let name_lc = name.map(str::to_lowercase).unwrap_or_default();
    let basename_lc = basename.map(str::to_lowercase).unwrap_or_default();
    let label = [name_lc.as_str(), basename_lc.as_str()].join(" ");

    // Reasoning families first — they share architecture names with their non-thinking cousins.
    if label.contains("r1-distill") || label.contains("deepseek-r1") {
        return Some(ModelCapabilities::deepseek_r1_distill());
    }
    if label.contains("deepseek-coder-v2") {
        return Some(ModelCapabilities::deepseek_coder_v2());
    }

    match arch_lc.as_str() {
        "qwen3" | "qwen3moe" => Some(ModelCapabilities::qwen3()),
        "qwen2" if label.contains("coder") => Some(ModelCapabilities::qwen25_coder()),
        "qwen2" => Some(ModelCapabilities::qwen25()),
        "llama" => {
            if label.contains("hermes") {
                Some(ModelCapabilities::hermes3())
            } else if label.contains("llama-3") || label.contains("llama3") {
                Some(ModelCapabilities::llama3())
            } else {
                None
            }
        }
        "mistral" | "mistral3" => Some(ModelCapabilities::mistral()),
        "phi3" | "phi4" => Some(ModelCapabilities::phi35()),
        "deepseek2" => Some(ModelCapabilities::deepseek_coder_v2()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::{ModelFamily, ToolCallingTier};

    #[test]
    fn qwen3_architecture_resolves_to_qwen3() {
        let caps = family_from_metadata(Some("qwen3"), Some("Qwen3 1.7B"), Some("Qwen3")).unwrap();
        assert_eq!(caps.family, ModelFamily::Qwen3);
        assert!(caps.supports_reasoning);
    }

    #[test]
    fn qwen3_moe_architecture_resolves_to_qwen3() {
        let caps = family_from_metadata(
            Some("qwen3moe"),
            Some("Qwen3-Coder-30B-A3B"),
            Some("Qwen3-Coder"),
        )
        .unwrap();
        assert_eq!(caps.family, ModelFamily::Qwen3);
    }

    #[test]
    fn qwen2_coder_name_resolves_to_qwen25_coder() {
        let caps = family_from_metadata(
            Some("qwen2"),
            Some("Qwen2.5-Coder-7B-Instruct"),
            Some("Qwen2.5-Coder"),
        )
        .unwrap();
        assert_eq!(caps.family, ModelFamily::Qwen25Coder);
    }

    #[test]
    fn qwen2_plain_resolves_to_qwen25_instruct() {
        let caps =
            family_from_metadata(Some("qwen2"), Some("Qwen2.5-7B-Instruct"), Some("Qwen2.5"))
                .unwrap();
        assert_eq!(caps.family, ModelFamily::Qwen25);
    }

    #[test]
    fn deepseek_r1_distill_name_overrides_base_architecture() {
        // Distills are Qwen/Llama architecturally but the name says R1.
        let caps = family_from_metadata(
            Some("qwen2"),
            Some("DeepSeek-R1-Distill-Qwen-14B"),
            Some("DeepSeek-R1-Distill-Qwen"),
        )
        .unwrap();
        assert_eq!(caps.family, ModelFamily::DeepSeekR1);
        assert!(caps.supports_reasoning);
    }

    #[test]
    fn llama3_architecture_with_matching_name() {
        let caps = family_from_metadata(
            Some("llama"),
            Some("Meta-Llama-3-8B-Instruct"),
            Some("Meta-Llama-3"),
        )
        .unwrap();
        assert_eq!(caps.family, ModelFamily::Llama3);
    }

    #[test]
    fn hermes_on_llama_arch_resolves_to_hermes3() {
        let caps = family_from_metadata(
            Some("llama"),
            Some("Hermes-3-Llama-3.1-8B"),
            Some("Hermes-3"),
        )
        .unwrap();
        assert_eq!(caps.family, ModelFamily::Hermes3);
    }

    #[test]
    fn llama_arch_without_llama3_or_hermes_returns_none() {
        // e.g. some random fine-tune of llama-2 we don't recognize.
        assert!(
            family_from_metadata(Some("llama"), Some("MystRand-7B"), Some("MystRand")).is_none()
        );
    }

    #[test]
    fn mistral_architecture_resolves_to_mistral() {
        let caps = family_from_metadata(
            Some("mistral"),
            Some("Mistral-Small-3.2-24B-Instruct"),
            Some("Mistral-Small"),
        )
        .unwrap();
        assert_eq!(caps.family, ModelFamily::Mistral);
        assert!(matches!(caps.supports_tool_calls, ToolCallingTier::Native));
    }

    #[test]
    fn phi4_architecture_resolves_to_phi() {
        let caps =
            family_from_metadata(Some("phi4"), Some("Phi-4-14B-Instruct"), Some("Phi-4")).unwrap();
        assert_eq!(caps.family, ModelFamily::Phi35);
    }

    #[test]
    fn deepseek2_architecture_resolves_to_coder_v2() {
        let caps = family_from_metadata(
            Some("deepseek2"),
            Some("DeepSeek-Coder-V2-Lite-Base"),
            Some("DeepSeek-Coder-V2-Lite"),
        )
        .unwrap();
        assert_eq!(caps.family, ModelFamily::DeepSeekCoderV2);
    }

    #[test]
    fn unknown_architecture_returns_none() {
        assert!(family_from_metadata(Some("exotic-arch-v9"), None, None).is_none());
    }

    #[test]
    fn empty_signals_return_none() {
        assert!(family_from_metadata(None, None, None).is_none());
    }

    #[test]
    fn resolver_is_case_insensitive() {
        let caps = family_from_metadata(Some("QWEN3"), Some("QWEN3-14B"), Some("Qwen3")).unwrap();
        assert_eq!(caps.family, ModelFamily::Qwen3);
    }
}
