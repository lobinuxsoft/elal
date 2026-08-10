//! Unit tests for `query.rs`. Tree walk, sidecar-first reading, cwd
//! filtering, mtime-based ordering, and id-by-filename location.

use super::*;
use chrono::TimeZone;
use elal_protocol::ApprovalMode;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::thread::sleep;
use std::time::Duration;
use tempfile::tempdir;

fn fixed_ts() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 4, 29, 14, 32, 11).unwrap()
}

fn make_store_with_session(cwd: &Path) -> (tempfile::TempDir, RolloutStore, SessionRecord) {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let record = store.create_session_record(
        SessionId::new(),
        fixed_ts(),
        cwd.to_path_buf(),
        Some(PathBuf::from("/m.gguf")),
        ApprovalMode::Smart,
    );
    store.append_session_meta(&record).unwrap();
    (dir, store, record)
}

#[test]
fn list_empty_data_root_returns_empty() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let entries = list_sessions(&store, None).expect("list should succeed");
    assert!(entries.is_empty());
}

#[test]
fn list_returns_each_session() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let mut ids = Vec::new();
    for _ in 0..3 {
        let record = store.create_session_record(
            SessionId::new(),
            fixed_ts(),
            PathBuf::from("/some/cwd"),
            None,
            ApprovalMode::Smart,
        );
        store.append_session_meta(&record).unwrap();
        ids.push(record.id);
    }
    let entries = list_sessions(&store, None).unwrap();
    assert_eq!(entries.len(), 3);
    let listed_ids: Vec<_> = entries.iter().map(|e| e.record.id).collect();
    for id in ids {
        assert!(listed_ids.contains(&id));
    }
}

#[test]
fn list_orders_by_last_activity_descending() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    // Three sessions written sequentially. The mtime resolution on most
    // filesystems is at least 1 ms, but to be safe we wait briefly between
    // writes so the mtimes are guaranteed to differ.
    let mut ids_in_write_order = Vec::new();
    for _ in 0..3 {
        let record = store.create_session_record(
            SessionId::new(),
            fixed_ts(),
            PathBuf::from("/cwd"),
            None,
            ApprovalMode::Smart,
        );
        store.append_session_meta(&record).unwrap();
        ids_in_write_order.push(record.id);
        sleep(Duration::from_millis(20));
    }
    // Touch the second-written session to make it "newest".
    let middle_path = store.rollout_path(fixed_ts(), ids_in_write_order[1]);
    let mut f = OpenOptions::new().append(true).open(&middle_path).unwrap();
    sleep(Duration::from_millis(20));
    f.write_all(b"\n").unwrap();
    drop(f);

    let entries = list_sessions(&store, None).unwrap();
    assert_eq!(entries.len(), 3);
    // The middle-written-then-touched session must be first.
    assert_eq!(entries[0].record.id, ids_in_write_order[1]);
}

#[test]
fn list_filters_by_cwd() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let want = PathBuf::from("/projects/elal");
    let other = PathBuf::from("/projects/other");
    let r1 = store.create_session_record(
        SessionId::new(),
        fixed_ts(),
        want.clone(),
        None,
        ApprovalMode::Smart,
    );
    store.append_session_meta(&r1).unwrap();
    let r2 = store.create_session_record(
        SessionId::new(),
        fixed_ts(),
        other.clone(),
        None,
        ApprovalMode::Smart,
    );
    store.append_session_meta(&r2).unwrap();

    let entries = list_sessions(&store, Some(&want)).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].record.id, r1.id);

    let entries_other = list_sessions(&store, Some(&other)).unwrap();
    assert_eq!(entries_other.len(), 1);
    assert_eq!(entries_other[0].record.id, r2.id);
}

#[test]
fn find_latest_returns_most_recent_for_cwd() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let cwd = PathBuf::from("/projects/elal");
    let mut ids = Vec::new();
    for _ in 0..3 {
        let record = store.create_session_record(
            SessionId::new(),
            fixed_ts(),
            cwd.clone(),
            None,
            ApprovalMode::Smart,
        );
        store.append_session_meta(&record).unwrap();
        ids.push(record.id);
        sleep(Duration::from_millis(20));
    }
    let found = find_latest(&store, &cwd).unwrap().expect("must find one");
    // Last-written wins because mtimes are increasing.
    assert_eq!(found.record.id, *ids.last().unwrap());
}

#[test]
fn find_latest_returns_none_when_no_match() {
    let (_dir, store, _) = make_store_with_session(&PathBuf::from("/projects/elal"));
    let result = find_latest(&store, &PathBuf::from("/elsewhere")).unwrap();
    assert!(result.is_none());
}

#[test]
fn find_latest_returns_none_on_empty_root() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let result = find_latest(&store, &PathBuf::from("/anywhere")).unwrap();
    assert!(result.is_none());
}

#[test]
fn locate_finds_by_id() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let r1 = store.create_session_record(
        SessionId::new(),
        fixed_ts(),
        PathBuf::from("/a"),
        None,
        ApprovalMode::Smart,
    );
    store.append_session_meta(&r1).unwrap();
    let r2 = store.create_session_record(
        SessionId::new(),
        fixed_ts(),
        PathBuf::from("/b"),
        None,
        ApprovalMode::Smart,
    );
    store.append_session_meta(&r2).unwrap();

    let found = locate(&store, r2.id).unwrap().expect("must find r2");
    assert_eq!(found, r2.rollout_path);
}

#[test]
fn locate_returns_none_when_id_not_present() {
    let (_dir, store, _) = make_store_with_session(&PathBuf::from("/cwd"));
    let result = locate(&store, SessionId::new()).unwrap();
    assert!(result.is_none());
}

#[test]
fn locate_ignores_non_jsonl_files() {
    let (_dir, store, record) = make_store_with_session(&PathBuf::from("/cwd"));
    let parent = record.rollout_path.parent().unwrap();
    fs::write(parent.join("readme.txt"), b"not a rollout").unwrap();
    fs::write(parent.join("rollout-bogus-not-a-uuid.jsonl"), b"\n").unwrap();
    // Only the original session id resolves.
    let found = locate(&store, record.id).unwrap();
    assert_eq!(found, Some(record.rollout_path.clone()));
}

#[test]
fn sidecar_takes_precedence_over_jsonl() {
    let (_dir, store, record) = make_store_with_session(&PathBuf::from("/cwd"));
    // Mutate the in-memory record (simulating a turn that updated totals)
    // and write only the sidecar — the JSONL still holds the original meta
    // line. The query layer must surface the sidecar version.
    let mut bumped = record.clone();
    bumped.total_input_tokens = 9999;
    bumped.title = Some("from sidecar".into());
    store.write_meta_sidecar(&bumped).unwrap();

    let entries = list_sessions(&store, None).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].record.total_input_tokens, 9999);
    assert_eq!(entries[0].record.title.as_deref(), Some("from sidecar"));
}

#[test]
fn corrupt_rollout_without_sidecar_is_skipped_not_fatal() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    // One healthy session.
    let healthy = store.create_session_record(
        SessionId::new(),
        fixed_ts(),
        PathBuf::from("/cwd"),
        None,
        ApprovalMode::Smart,
    );
    store.append_session_meta(&healthy).unwrap();

    // One corrupt rollout with no sidecar — content is invalid JSON, no
    // SessionMeta line.
    let corrupt_dir = dir
        .path()
        .join("sessions")
        .join("2026")
        .join("04")
        .join("29");
    fs::create_dir_all(&corrupt_dir).unwrap();
    let corrupt_id = SessionId::new();
    let corrupt_path = corrupt_dir.join(format!("rollout-2026-04-29T00-00-00Z-{corrupt_id}.jsonl"));
    fs::write(&corrupt_path, b"this is not json\n").unwrap();

    let entries = list_sessions(&store, None).expect("must not fail on corrupt rollout");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].record.id, healthy.id);
}

#[test]
fn empty_jsonl_file_is_skipped() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let parent = dir
        .path()
        .join("sessions")
        .join("2026")
        .join("04")
        .join("29");
    fs::create_dir_all(&parent).unwrap();
    let empty = parent.join(format!("rollout-x-{}.jsonl", SessionId::new()));
    fs::File::create(&empty).unwrap();
    let entries = list_sessions(&store, None).unwrap();
    assert!(entries.is_empty());
}

#[test]
fn parse_session_id_from_filename_handles_real_filename() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let id = SessionId::new();
    let path = store.rollout_path(fixed_ts(), id);
    assert_eq!(parse_session_id_from_filename(&path), Some(id));
}

#[test]
fn parse_session_id_rejects_non_uuid_tail() {
    assert_eq!(
        parse_session_id_from_filename(Path::new("rollout-foo-bar.jsonl")),
        None
    );
    assert_eq!(
        parse_session_id_from_filename(Path::new("readme.txt")),
        None
    );
    assert_eq!(parse_session_id_from_filename(Path::new("")), None);
}
