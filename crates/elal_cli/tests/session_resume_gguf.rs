//! GGUF-backed smoke test for the full agent resume loop. Spawns two
//! `elal agent` processes — the first opens a fresh session and emits one
//! prompt before EOF, the second runs `--continue`, emits a second prompt,
//! and exits. Asserts the shared rollout has accumulated tokens across
//! both turns.
//!
//! Gated behind `ELAL_TEST_MODEL` so quick `cargo test` runs skip the model
//! load entirely. Run with:
//!
//! ```ignore
//! ELAL_TEST_MODEL=/path/to/model.gguf cargo test \
//!     --test session_resume_gguf -- --ignored --nocapture
//! ```

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;

mod common;
use common::Sandbox;

#[test]
#[ignore = "requires ELAL_TEST_MODEL pointing at a GGUF on disk"]
fn agent_resume_smoke_with_real_gguf() {
    let model =
        std::env::var("ELAL_TEST_MODEL").expect("ELAL_TEST_MODEL must point at a GGUF file");
    let model_path = PathBuf::from(&model);
    assert!(
        model_path.exists(),
        "ELAL_TEST_MODEL does not exist: {model}"
    );

    let sb = Sandbox::new();
    let cwd = std::env::current_dir().expect("cwd");

    drive_one_turn(&sb, &model_path, "--new", "say hi in one word\n");
    drive_one_turn(&sb, &model_path, "--continue", "say bye in one word\n");

    let listing = elal_core::list_sessions(&sb.store, Some(cwd.as_path()))
        .expect("list sessions after smoke");
    assert_eq!(listing.len(), 1, "exactly one session for this cwd");
    let entry = &listing[0];
    assert!(
        entry.record.total_input_tokens > 0,
        "input tokens accumulated across both turns"
    );
    assert!(
        entry.record.total_output_tokens > 0,
        "output tokens accumulated across both turns"
    );
}

fn drive_one_turn(sb: &Sandbox, model_path: &Path, mode_flag: &str, prompt: &str) {
    // Default `--n-gpu-layers -1` so the smoke exercises the same Vulkan
    // path real users hit (ADR-1). CPU-only would validate the resume
    // contract while bypassing the GPU code paths that matter.
    let mut child = sb
        .cmd()
        .args([
            "agent",
            mode_flag,
            "--model",
            model_path.to_str().expect("utf-8 model path"),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn agent process");
    {
        let stdin = child.stdin.as_mut().expect("stdin handle");
        stdin.write_all(prompt.as_bytes()).expect("write prompt");
    }
    drop(child.stdin.take());
    let out = child.wait_with_output().expect("agent process exits");
    assert!(
        out.status.success(),
        "agent {mode_flag} failed: {}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}
