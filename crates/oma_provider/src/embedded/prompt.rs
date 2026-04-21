//! Chat template application: turn [`oma_protocol::Message`]s into the
//! single prompt string that will be tokenised and fed to the model.

use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use oma_protocol::{Message, Role};

use crate::error::LlmError;

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
