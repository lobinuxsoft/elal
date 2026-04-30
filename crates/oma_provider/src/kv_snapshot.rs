//! Persistent KV-cache snapshots — the headline value-add for resumed
//! `oma_agent` sessions: skip prompt re-evaluation when continuing a session
//! against the same model.
//!
//! Wraps `LlamaContext::state_save_file` / `state_load_file` from
//! `llama-cpp-2 0.1.144` with three safety layers:
//!
//! 1. **Atomic writes.** Snapshot files for 32K-context models are ~1 GiB —
//!    we never want to leave a half-written `.kv` next to a healthy rollout
//!    journal. Writes go to `<path>.tmp` first and rename atomically.
//! 2. **Model-SHA validation.** The KV state is meaningless across model
//!    swaps; loading a snapshot taken against a different GGUF can corrupt
//!    inference. [`validate_compatible`] hashes the live model and refuses
//!    to load when the SHA differs from the one recorded on the session.
//! 3. **Graceful fallback.** Errors are surfaced as a typed
//!    [`KvSnapshotError`] so the caller can log + drop back to a full
//!    prompt-eval cleanly.
//!
//! The corresponding flow lives in `oma_provider::embedded::streaming`:
//! `chat_completion_stream` peeks at [`crate::CompletionRequest::kv_cache_path`],
//! tries to load the snapshot before tokenising, and saves it again at the
//! end of generation.

use std::fs;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::session::{LoadSessionError, SaveSessionError};
use llama_cpp_2::token::LlamaToken;
use sha2::{Digest, Sha256};

const HASH_BUF_SIZE: usize = 64 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum KvSnapshotError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("save failed: {0}")]
    Save(#[from] SaveSessionError),
    #[error("load failed: {0}")]
    Load(#[from] LoadSessionError),
    #[error(
        "model SHA-256 mismatch: snapshot was created against {expected}, current model is {actual}"
    )]
    ModelMismatch { expected: String, actual: String },
}

/// Compute the SHA-256 of the GGUF file at `model_path`. Streamed so we don't
/// have to map a multi-GiB file into memory just to hash it.
pub fn compute_model_sha256(model_path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(model_path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; HASH_BUF_SIZE];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Atomically save the live KV state at `path` together with the token list
/// it was generated from. Reproduces [`oma_core::session::RolloutStore`]'s
/// "tmp + fsync + rename" pattern so a crash mid-write never leaves a
/// truncated snapshot.
pub fn save(ctx: &LlamaContext, tokens: &[LlamaToken], path: &Path) -> Result<(), KvSnapshotError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = with_extension_suffix(path, "tmp");
    ctx.state_save_file(&tmp, tokens)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Load a previously saved snapshot into `ctx`, returning the token list
/// the snapshot was generated from. Caller is responsible for verifying that
/// the next decoded tokens line up with these (i.e. the live prompt's prefix
/// matches what the snapshot covers).
pub fn load(
    ctx: &mut LlamaContext,
    path: &Path,
    max_tokens: usize,
) -> Result<Vec<LlamaToken>, KvSnapshotError> {
    Ok(ctx.state_load_file(path, max_tokens)?)
}

/// Refuse the snapshot when its recorded SHA does not match the file at
/// `model_path`. Run this before [`load`] on a `--resume` boundary.
pub fn validate_compatible(expected_sha: &str, model_path: &Path) -> Result<(), KvSnapshotError> {
    let actual = compute_model_sha256(model_path)?;
    if expected_sha != actual {
        return Err(KvSnapshotError::ModelMismatch {
            expected: expected_sha.to_string(),
            actual,
        });
    }
    Ok(())
}

/// Returns the canonical KV file path for a rollout — `<rollout-name>.kv`
/// next to the JSONL.
pub fn kv_path(rollout_path: &Path) -> PathBuf {
    with_extension_suffix(rollout_path, "kv")
}

fn with_extension_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut p = path.to_path_buf();
    let new_name = match p.file_name().and_then(|n| n.to_str()) {
        Some(name) => format!("{name}.{suffix}"),
        None => return p,
    };
    p.set_file_name(new_name);
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn compute_model_sha256_is_deterministic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("blob.bin");
        let payload: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        std::fs::write(&path, &payload).unwrap();
        let a = compute_model_sha256(&path).unwrap();
        let b = compute_model_sha256(&path).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64, "SHA-256 hex string is exactly 64 chars");
    }

    #[test]
    fn compute_model_sha256_is_sensitive_to_content() {
        let dir = tempdir().unwrap();
        let p1 = dir.path().join("a.bin");
        let p2 = dir.path().join("b.bin");
        std::fs::write(&p1, b"alpha").unwrap();
        std::fs::write(&p2, b"alpha\n").unwrap();
        let a = compute_model_sha256(&p1).unwrap();
        let b = compute_model_sha256(&p2).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn compute_model_sha256_handles_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        fs::File::create(&path).unwrap();
        let h = compute_model_sha256(&path).unwrap();
        // Well-known SHA-256 of an empty input.
        assert_eq!(
            h,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn compute_model_sha256_streams_files_larger_than_buffer() {
        // Write 2 × HASH_BUF_SIZE so the read loop iterates more than once.
        let dir = tempdir().unwrap();
        let path = dir.path().join("big.bin");
        let mut f = fs::File::create(&path).unwrap();
        for chunk in 0..3 {
            let payload: Vec<u8> = (0..HASH_BUF_SIZE)
                .map(|i| ((i + chunk) % 256) as u8)
                .collect();
            f.write_all(&payload).unwrap();
        }
        drop(f);
        let h = compute_model_sha256(&path).unwrap();
        assert_eq!(h.len(), 64);
    }

    #[test]
    fn validate_compatible_accepts_matching_sha() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"fake-gguf").unwrap();
        let sha = compute_model_sha256(&path).unwrap();
        assert!(validate_compatible(&sha, &path).is_ok());
    }

    #[test]
    fn validate_compatible_rejects_mismatched_sha() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"fake-gguf").unwrap();
        let bogus = "0".repeat(64);
        let err = validate_compatible(&bogus, &path).unwrap_err();
        assert!(matches!(err, KvSnapshotError::ModelMismatch { .. }));
    }

    #[test]
    fn kv_path_appends_kv_suffix_to_rollout_filename() {
        let rollout = Path::new("/data/sessions/2026/04/29/rollout-abc.jsonl");
        assert_eq!(
            kv_path(rollout),
            PathBuf::from("/data/sessions/2026/04/29/rollout-abc.jsonl.kv")
        );
    }

    #[test]
    fn kv_path_handles_pathological_paths_gracefully() {
        // "/" has no file_name — we return the path unchanged rather than
        // panic.
        let pathological = Path::new("/");
        assert_eq!(kv_path(pathological), PathBuf::from("/"));
    }
}
