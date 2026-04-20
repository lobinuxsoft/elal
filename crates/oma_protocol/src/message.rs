use serde::{Deserialize, Serialize};

use crate::role::Role;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Reasoning output emitted by models such as QwQ and DeepSeek-R1.
    /// Hidden from the user by default; retained for introspection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            reasoning_content: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

impl ToolCall {
    pub fn function(id: impl Into<String>, name: impl Into<String>, arguments: String) -> Self {
        Self {
            id: id.into(),
            kind: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_message_serializes_openai_compatible() {
        let msg = Message::user("hello");
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "hello");
        assert!(json.get("tool_calls").is_none());
        assert!(json.get("reasoning_content").is_none());
    }

    #[test]
    fn assistant_with_tool_call_roundtrips() {
        let msg = Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall::function(
                "call_1",
                "read_file",
                "{\"path\":\"a.rs\"}".into(),
            )]),
            tool_call_id: None,
            reasoning_content: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, back);
    }

    #[test]
    fn tool_message_carries_call_id() {
        let msg = Message::tool("call_1", "file contents");
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "tool");
        assert_eq!(json["tool_call_id"], "call_1");
        assert_eq!(json["content"], "file contents");
    }

    #[test]
    fn reasoning_content_is_preserved() {
        let msg = Message {
            role: Role::Assistant,
            content: Some("final answer".into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: Some("step 1 → step 2".into()),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("reasoning_content"));
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, back);
    }
}
