//! `websearch` built-in — DuckDuckGo HTML scraping search.

use std::time::Duration;

use async_trait::async_trait;
use elal_protocol::ToolDefinition;
use reqwest::Client;
use scraper::{Html, Selector};
use serde_json::json;

use crate::context::ToolContext;
use crate::result::{ToolError, ToolResult};
use crate::spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
use crate::trait_def::Tool;

const DESCRIPTION: &str = include_str!("websearch.txt");
const DEFAULT_MAX_RESULTS: usize = 5;
const HARD_MAX_RESULTS: usize = 20;
const REQUEST_TIMEOUT_SECS: u64 = 30;
const USER_AGENT: &str = "elal/0.1 (+https://github.com/lobinuxsoft/elal)";
const DUCKDUCKGO_URL: &str = "https://html.duckduckgo.com/html/";

/// DuckDuckGo HTML search via the public non-API endpoint.
pub struct WebSearchTool {
    endpoint: String,
}

impl Default for WebSearchTool {
    fn default() -> Self {
        Self {
            endpoint: DUCKDUCKGO_URL.to_string(),
        }
    }
}

impl WebSearchTool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Used by tests to point the scraper at a local mock server.
    #[cfg(test)]
    pub fn with_endpoint(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
        }
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "websearch"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "websearch",
            DESCRIPTION,
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string."
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": HARD_MAX_RESULTS,
                        "description": "Maximum number of results (default 5, max 20)."
                    }
                },
                "required": ["query"]
            }),
        )
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "websearch",
            tier: ToolTier::Read,
            approval_hint: ApprovalHint::Never,
            side_effects: SideEffects::Network,
        }
    }

    fn describe_action(&self, args: &serde_json::Value) -> String {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("<missing query>");
        format!("Search web for `{query}`")
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _ctx: &ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let query =
            args.get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArgs {
                    name: "websearch".into(),
                    reason: "missing 'query' field".into(),
                })?;
        if query.trim().is_empty() {
            return Err(ToolError::InvalidArgs {
                name: "websearch".into(),
                reason: "'query' must not be empty".into(),
            });
        }
        let max_results = match args.get("max_results") {
            Some(v) if v.is_null() => DEFAULT_MAX_RESULTS,
            Some(v) => {
                let n = v.as_u64().ok_or_else(|| ToolError::InvalidArgs {
                    name: "websearch".into(),
                    reason: "'max_results' must be a positive integer".into(),
                })?;
                (n as usize).clamp(1, HARD_MAX_RESULTS)
            }
            None => DEFAULT_MAX_RESULTS,
        };

        let client = Client::builder()
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| ToolError::Execution(format!("http client init: {e}")))?;

        let response = client
            .get(&self.endpoint)
            .query(&[("q", query)])
            .send()
            .await
            .map_err(|e| ToolError::Execution(format!("search request failed: {e}")))?;
        let status = response.status();
        if !status.is_success() {
            return Ok(ToolResult::soft_error(format!(
                "DuckDuckGo responded with HTTP {}",
                status.as_u16()
            )));
        }
        let html_text = response
            .text()
            .await
            .map_err(|e| ToolError::Execution(format!("read response failed: {e}")))?;

        let results = parse_results(&html_text, max_results);
        let mut content = format!("<query>{query}</query>\n<results>\n", query = query,);
        for (i, r) in results.iter().enumerate() {
            content.push_str(&format!(
                "{}. {}\n   {}\n   {}\n",
                i + 1,
                r.title,
                r.url,
                r.snippet,
            ));
        }
        content.push_str(&format!("\n({} results)\n</results>", results.len()));

        Ok(ToolResult::ok(content).with_structured(json!({
            "count": results.len(),
            "results": results.iter().map(|r| json!({
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
            })).collect::<Vec<_>>(),
        })))
    }
}

#[derive(Debug, Clone)]
struct SearchHit {
    title: String,
    url: String,
    snippet: String,
}

fn parse_results(html: &str, max_results: usize) -> Vec<SearchHit> {
    let doc = Html::parse_document(html);
    // DuckDuckGo HTML endpoint wraps each hit in a `.result__body`.
    let body_sel = Selector::parse(".result__body").expect("valid selector");
    let title_sel = Selector::parse(".result__a").expect("valid selector");
    let snippet_sel = Selector::parse(".result__snippet").expect("valid selector");

    let mut out = Vec::new();
    for body in doc.select(&body_sel) {
        if out.len() >= max_results {
            break;
        }
        let Some(title_el) = body.select(&title_sel).next() else {
            continue;
        };
        let title: String = title_el.text().collect();
        let href = title_el.value().attr("href").unwrap_or("").to_string();
        let snippet: String = body
            .select(&snippet_sel)
            .next()
            .map(|s| s.text().collect())
            .unwrap_or_default();
        if title.trim().is_empty() || href.is_empty() {
            continue;
        }
        out.push(SearchHit {
            title: title.trim().to_string(),
            url: normalize_url(&href),
            snippet: snippet.trim().to_string(),
        });
    }
    out
}

/// DuckDuckGo wraps results in a redirect like `//duckduckgo.com/l/?uddg=...`.
/// Extract the real URL if possible, else return as-is.
fn normalize_url(href: &str) -> String {
    let marker = "uddg=";
    if let Some(start) = href.find(marker) {
        let encoded = &href[start + marker.len()..];
        let end = encoded.find('&').unwrap_or(encoded.len());
        if let Ok(decoded) = urldecode(&encoded[..end]) {
            return decoded;
        }
    }
    if href.starts_with("//") {
        format!("https:{href}")
    } else {
        href.to_string()
    }
}

fn urldecode(s: &str) -> Result<String, ()> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hi = from_hex(bytes[i + 1])?;
                let lo = from_hex(bytes[i + 2])?;
                out.push((hi << 4) | lo);
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|_| ())
}

fn from_hex(b: u8) -> Result<u8, ()> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(()),
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

    const SAMPLE_HTML: &str = r##"
    <html><body>
      <div class="result__body">
        <h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frust-lang.org%2F&amp;rut=xxx">The Rust Language</a></h2>
        <a class="result__snippet" href="#">A systems programming language.</a>
      </div>
      <div class="result__body">
        <h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdoc.rust-lang.org%2Fbook%2F">The Rust Book</a></h2>
        <a class="result__snippet">Official online book.</a>
      </div>
      <div class="result__body">
        <h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F">Example</a></h2>
        <a class="result__snippet">Placeholder.</a>
      </div>
    </body></html>
    "##;

    #[tokio::test]
    async fn parses_scraped_results_from_mock() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET)
                .path("/html/")
                .query_param("q", "rust lang");
            then.status(200)
                .header("content-type", "text/html")
                .body(SAMPLE_HTML);
        });
        let tool = WebSearchTool::with_endpoint(server.url("/html/"));
        let dir = TempDir::new().unwrap();
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"query": "rust lang"}), &c)
            .await
            .unwrap();
        mock.assert();
        assert!(!out.is_error);
        assert!(out.content.contains("The Rust Language"));
        assert!(out.content.contains("https://rust-lang.org/"));
        assert!(out.content.contains("The Rust Book"));
        assert!(out.content.contains("Example"));
    }

    #[tokio::test]
    async fn max_results_caps_output() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/html/");
            then.status(200)
                .header("content-type", "text/html")
                .body(SAMPLE_HTML);
        });
        let tool = WebSearchTool::with_endpoint(server.url("/html/"));
        let dir = TempDir::new().unwrap();
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"query": "x", "max_results": 1}), &c)
            .await
            .unwrap();
        assert!(out.content.contains("The Rust Language"));
        assert!(!out.content.contains("The Rust Book"));
    }

    #[tokio::test]
    async fn server_500_is_soft_error() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/html/");
            then.status(500);
        });
        let tool = WebSearchTool::with_endpoint(server.url("/html/"));
        let dir = TempDir::new().unwrap();
        let (c, _rx) = ctx(dir.path());
        let out = tool.execute(json!({"query": "x"}), &c).await.unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("500"));
    }

    #[tokio::test]
    async fn missing_query_is_invalid() {
        let dir = TempDir::new().unwrap();
        let tool = WebSearchTool::new();
        let (c, _rx) = ctx(dir.path());
        let err = tool.execute(json!({}), &c).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[tokio::test]
    async fn empty_query_is_invalid() {
        let dir = TempDir::new().unwrap();
        let tool = WebSearchTool::new();
        let (c, _rx) = ctx(dir.path());
        let err = tool.execute(json!({"query": "  "}), &c).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[test]
    fn urldecode_roundtrip() {
        assert_eq!(urldecode("hello%20world").unwrap(), "hello world");
        assert_eq!(urldecode("a+b").unwrap(), "a b");
        assert_eq!(
            urldecode("https%3A%2F%2Fexample.com%2F").unwrap(),
            "https://example.com/"
        );
    }

    #[test]
    fn normalize_url_extracts_uddg_target() {
        assert_eq!(
            normalize_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Frust-lang.org%2F&rut=xxx"),
            "https://rust-lang.org/"
        );
        assert_eq!(
            normalize_url("//example.com/foo"),
            "https://example.com/foo"
        );
        assert_eq!(
            normalize_url("https://direct.example.com/"),
            "https://direct.example.com/"
        );
    }

    #[test]
    fn spec_declares_never_approval_network_side_effects() {
        let spec = WebSearchTool::new().spec();
        assert_eq!(spec.approval_hint, ApprovalHint::Never);
        assert_eq!(spec.side_effects, SideEffects::Network);
    }
}
