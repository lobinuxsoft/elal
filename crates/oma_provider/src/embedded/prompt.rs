//! Chat template application: turn [`oma_protocol::Message`]s into the
//! single prompt string that will be tokenised and fed to the model.
//!
//! Two paths:
//!
//! - [`build_plain_prompt`] for requests without tools — the fast path that
//!   uses `LlamaModel::apply_chat_template` directly.
//! - [`build_oaicompat_prompt`] for tool-bearing requests — wraps
//!   `apply_chat_template_oaicompat` with `enable_thinking: false` and
//!   returns the [`ToolPromptSetup`] bundle the streaming loop needs to
//!   configure its sampler and delta parser.

use llama_cpp_2::model::{
    GrammarTrigger, GrammarTriggerType, LlamaChatMessage, LlamaChatTemplate, LlamaModel,
};
use llama_cpp_2::openai::{ChatParseStateOaicompat, OpenAIChatTemplateParams};
use llama_cpp_2::token::LlamaToken;
use oma_protocol::{Message, Role, ToolDefinition};

use crate::error::LlmError;
use crate::oaicompat;

/// Apply the model's embedded chat template to a message slice. This is the
/// fast path used when the request carries no tool definitions — plain text
/// in, plain prompt out.
pub(super) fn build_plain_prompt(
    model: &LlamaModel,
    chat_template: Option<&LlamaChatTemplate>,
    messages: &[Message],
) -> Result<String, LlmError> {
    let tmpl = chat_template
        .ok_or_else(|| LlmError::ChatTemplate("model has no embedded chat template".into()))?;

    let chat: Vec<LlamaChatMessage> = messages
        .iter()
        .map(|m| {
            let role = role_to_string(m.role);
            let content = m.content.clone().unwrap_or_default();
            LlamaChatMessage::new(role, content).map_err(|e| LlmError::ChatTemplate(e.to_string()))
        })
        .collect::<Result<_, _>>()?;

    model
        .apply_chat_template(tmpl, &chat, true)
        .map_err(|e| LlmError::ChatTemplate(e.to_string()))
}

/// Translate our protocol [`Role`] into the wire string llama-cpp-2 expects.
/// Kept as a free function because both `build_plain_prompt` and the
/// oaicompat adapter need it.
pub(super) fn role_to_string(role: Role) -> String {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
    .to_string()
}

/// Output of [`build_oaicompat_prompt`] — everything the streaming loop
/// needs to configure itself for a tool-bearing request.
pub(super) struct ToolPromptSetup {
    /// Rendered prompt, ready to tokenise.
    pub prompt: String,
    /// Streaming parser that converts raw generated text into OpenAI-format
    /// JSON deltas. Created from the template result's PEG parser.
    pub parse_state: Option<ChatParseStateOaicompat>,
    /// Optional grammar configuration — absent when the template declined to
    /// constrain output (model family without native tool grammar support).
    pub grammar: Option<GrammarConfig>,
    /// Extra stop sequences declared by the template. TODO: wire into the
    /// generation loop as an early-stop condition; the grammar sampler
    /// handles the primary stop cases today.
    #[allow(dead_code)]
    pub additional_stops: Vec<String>,
}

/// Grammar configuration bundle the sampler builder consumes.
pub(super) struct GrammarConfig {
    pub grammar: String,
    pub lazy: bool,
    pub trigger_patterns: Vec<String>,
    pub trigger_tokens: Vec<LlamaToken>,
}

/// Build the tool-aware prompt via
/// `apply_chat_template_with_tools_oaicompat` and assemble the streaming
/// parser + grammar configuration derived from the template result.
pub(super) fn build_oaicompat_prompt(
    model: &LlamaModel,
    chat_template: Option<&LlamaChatTemplate>,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<ToolPromptSetup, LlmError> {
    let tmpl = chat_template
        .ok_or_else(|| LlmError::ChatTemplate("model has no embedded chat template".into()))?;

    let messages_json = oaicompat::messages_to_json(messages)?;
    let tools_json = oaicompat::tools_to_json(tools)?;

    // `enable_thinking: false` is the upstream-documented workaround for the
    // llama.cpp lazy-grammar crash (`GGML_ASSERT(!stacks.empty())` in
    // llama-grammar.cpp:940) that fires when a thinking-capable Qwen3 model
    // emits `</think>\n\n<tool_call>` — the trigger regex matches but the
    // grammar stack has already been consumed by the preceding `<think>`
    // block. See ggml-org/llama.cpp#20345 and #21017.
    let params = OpenAIChatTemplateParams {
        messages_json: &messages_json,
        tools_json: Some(&tools_json),
        tool_choice: None,
        json_schema: None,
        grammar: None,
        reasoning_format: None,
        chat_template_kwargs: None,
        add_generation_prompt: true,
        use_jinja: true,
        parallel_tool_calls: true,
        enable_thinking: false,
        add_bos: true,
        add_eos: false,
        parse_tool_calls: true,
    };

    let result = model
        .apply_chat_template_oaicompat(tmpl, &params)
        .map_err(|e| LlmError::ChatTemplate(e.to_string()))?;

    let parse_state = if result.parse_tool_calls {
        Some(
            result
                .streaming_state_oaicompat()
                .map_err(|e| LlmError::ChatTemplate(format!("streaming parser init: {e}")))?,
        )
    } else {
        None
    };

    // Grammar enforcement is intentionally disabled. The upstream lazy-grammar
    // path crashes on `<tool_call>` triggers when any tokens precede the tag
    // (`</think>`, transitional text, the empty `<think></think>` placeholder
    // emitted with `enable_thinking: false`, etc.) — see
    // ggml-org/llama.cpp#21017, #20345, #20260. Following the same model as
    // the upstream sibling project (claw-code-rust): rely entirely on the
    // PEG-based `ChatParseStateOaicompat` parser above to classify deltas
    // post-hoc into `tool_calls`, `reasoning_content`, and `content`.
    let _ = split_triggers; // keep the helper alive for future re-enable
    let grammar: Option<GrammarConfig> = None;

    Ok(ToolPromptSetup {
        prompt: result.prompt,
        parse_state,
        grammar,
        additional_stops: result.additional_stops,
    })
}

/// Convert llama-cpp-2's mixed-type trigger list into the two homogeneous
/// slices `LlamaSampler::grammar_lazy_patterns` wants.
///
/// - Token triggers contribute their token id to `trigger_tokens`.
/// - Pattern / PatternFull triggers pass their regex through verbatim.
/// - Word triggers are escaped as literal regexes so they match exactly.
fn split_triggers(triggers: &[GrammarTrigger]) -> (Vec<String>, Vec<LlamaToken>) {
    let mut patterns = Vec::new();
    let mut tokens = Vec::new();
    for t in triggers {
        match t.trigger_type {
            GrammarTriggerType::Token => {
                if let Some(tok) = t.token {
                    tokens.push(tok);
                }
            }
            GrammarTriggerType::Word => patterns.push(regex_escape(&t.value)),
            GrammarTriggerType::Pattern | GrammarTriggerType::PatternFull => {
                patterns.push(t.value.clone())
            }
        }
    }
    (patterns, tokens)
}

/// Minimal regex-literal escaper — covers the characters that actually show
/// up in GGUF trigger words (tags, markers, punctuation).
fn regex_escape(s: &str) -> String {
    const META: &[char] = &[
        '\\', '.', '+', '*', '?', '(', ')', '|', '[', ']', '{', '}', '^', '$', '/',
    ];
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        if META.contains(&ch) {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_mapping_matches_openai_wire_format() {
        assert_eq!(role_to_string(Role::System), "system");
        assert_eq!(role_to_string(Role::User), "user");
        assert_eq!(role_to_string(Role::Assistant), "assistant");
        assert_eq!(role_to_string(Role::Tool), "tool");
    }
}
