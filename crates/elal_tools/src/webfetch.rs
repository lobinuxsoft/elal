//! `webfetch` built-in — GET a URL and return its body as Markdown.

use std::time::Duration;

use async_trait::async_trait;
use elal_protocol::ToolDefinition;
use htmd::HtmlToMarkdown;
use reqwest::Client;
use serde_json::json;

use crate::context::ToolContext;
use crate::result::{ToolError, ToolResult};
use crate::spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
use crate::trait_def::Tool;

const DESCRIPTION: &str = include_str!("webfetch.txt");
const DEFAULT_MAX_BYTES: usize = 2 * 1024 * 1024;
const REQUEST_TIMEOUT_SECS: u64 = 30;
const USER_AGENT: &str = "elal/0.1 (+https://github.com/lobinuxsoft/elal)";

/// Fetches a URL and converts HTML responses to Markdown.
pub struct WebFetchTool;

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "webfetch"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "webfetch",
            DESCRIPTION,
            json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Fully-qualified URL (http or https)."
                    },
                    "max_bytes": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum body size in bytes (default 2 MB)."
                    }
                },
                "required": ["url"]
            }),
        )
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "webfetch",
            tier: ToolTier::Read,
            approval_hint: ApprovalHint::Never,
            side_effects: SideEffects::Network,
        }
    }

    fn describe_action(&self, args: &serde_json::Value) -> String {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("<missing url>");
        format!("Fetch {url}")
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _ctx: &ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let url =
            args.get("url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArgs {
                    name: "webfetch".into(),
                    reason: "missing 'url' field".into(),
                })?;
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(ToolError::InvalidArgs {
                name: "webfetch".into(),
                reason: "'url' must start with http:// or https://".into(),
            });
        }
        let max_bytes = match args.get("max_bytes") {
            Some(v) if v.is_null() => DEFAULT_MAX_BYTES,
            Some(v) => {
                let n = v.as_u64().ok_or_else(|| ToolError::InvalidArgs {
                    name: "webfetch".into(),
                    reason: "'max_bytes' must be a positive integer".into(),
                })?;
                if n == 0 {
                    return Err(ToolError::InvalidArgs {
                        name: "webfetch".into(),
                        reason: "'max_bytes' must be >= 1".into(),
                    });
                }
                n as usize
            }
            None => DEFAULT_MAX_BYTES,
        };

        let client = Client::builder()
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| ToolError::Execution(format!("http client init: {e}")))?;

        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| ToolError::Execution(format!("fetch failed: {e}")))?;

        let status = response.status();
        let final_url = response.url().to_string();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| ToolError::Execution(format!("read body failed: {e}")))?;

        let truncated = bytes.len() > max_bytes;
        let slice = if truncated {
            &bytes[..max_bytes]
        } else {
            &bytes[..]
        };
        let body_text = String::from_utf8_lossy(slice).into_owned();

        let markdown = if content_type.starts_with("text/html") {
            HtmlToMarkdown::new()
                .convert(&body_text)
                .unwrap_or_else(|_| body_text.clone())
        } else {
            body_text
        };

        let mut content = format!(
            "<url>{final_url}</url>\n<status>{status}</status>\n<content_type>{content_type}</content_type>\n<body>\n{markdown}\n</body>",
            status = status.as_u16(),
        );
        if truncated {
            content.push_str(&format!("\n(Body truncated at {} bytes)", max_bytes));
        }
        let is_error = !status.is_success();
        let result = if is_error {
            ToolResult::soft_error(content)
        } else {
            ToolResult::ok(content)
        };
        Ok(result.with_structured(json!({
            "status": status.as_u16(),
            "content_type": content_type,
            "final_url": final_url,
            "truncated": truncated,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::Method::GET;
    use httpmock::MockServer;
    use std::path::Path;
    use tempfile::TempDir;
    use tokio::sync::mpsc;

    fn ctx<'a>(wd: &'a Path) -> (ToolContext<'a>, mpsc::Receiver<crate::context::ToolEvent>) {
        let (tx, rx) = mpsc::channel(4);
        (
            ToolContext {
                working_dir: wd,
                project_root: None,
                session_id: elal_protocol::SessionId::new(),
                events: tx,
            },
            rx,
        )
    }

    #[tokio::test]
    async fn html_response_is_converted_to_markdown() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/page");
            then.status(200)
                .header("content-type", "text/html; charset=utf-8")
                .body(
                    "<html><body><h1>Title</h1><p>Body <a href=\"/x\">link</a></p></body></html>",
                );
        });
        let dir = TempDir::new().unwrap();
        let tool = WebFetchTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"url": server.url("/page")}), &c)
            .await
            .unwrap();
        mock.assert();
        assert!(!out.is_error);
        assert!(out.content.contains("# Title"));
        assert!(out.content.contains("[link](/x)"));
    }

    #[tokio::test]
    async fn plain_text_response_is_returned_as_is() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/json");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"hello":"world"}"#);
        });
        let dir = TempDir::new().unwrap();
        let tool = WebFetchTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"url": server.url("/json")}), &c)
            .await
            .unwrap();
        mock.assert();
        assert!(out.content.contains(r#"{"hello":"world"}"#));
    }

    #[tokio::test]
    async fn non_2xx_is_soft_error_but_still_returns_body() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/err");
            then.status(404).body("not found");
        });
        let dir = TempDir::new().unwrap();
        let tool = WebFetchTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"url": server.url("/err")}), &c)
            .await
            .unwrap();
        mock.assert();
        assert!(out.is_error);
        assert!(out.content.contains("<status>404</status>"));
        assert!(out.content.contains("not found"));
    }

    #[tokio::test]
    async fn truncates_body_past_max_bytes() {
        let server = MockServer::start();
        let payload = "x".repeat(1024);
        let mock = server.mock(|when, then| {
            when.method(GET).path("/big");
            then.status(200)
                .header("content-type", "text/plain")
                .body(payload.clone());
        });
        let dir = TempDir::new().unwrap();
        let tool = WebFetchTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"url": server.url("/big"), "max_bytes": 16}), &c)
            .await
            .unwrap();
        mock.assert();
        assert!(out.content.contains("Body truncated at 16 bytes"));
    }

    #[tokio::test]
    async fn missing_url_is_invalid_args() {
        let dir = TempDir::new().unwrap();
        let tool = WebFetchTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool.execute(json!({}), &c).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[tokio::test]
    async fn non_http_scheme_is_invalid() {
        let dir = TempDir::new().unwrap();
        let tool = WebFetchTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(json!({"url": "ftp://example.com"}), &c)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[test]
    fn spec_declares_never_approval_network_side_effects() {
        let spec = WebFetchTool.spec();
        assert_eq!(spec.approval_hint, ApprovalHint::Never);
        assert_eq!(spec.side_effects, SideEffects::Network);
    }
}
