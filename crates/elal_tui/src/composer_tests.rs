//! Editing invariants, with an eye on the ones UTF-8 breaks.

use crate::composer::Composer;

fn composer_with(text: &str) -> Composer {
    let mut c = Composer::new("");
    for ch in text.chars() {
        if ch == '\n' {
            c.insert_newline();
        } else {
            c.insert_char(ch);
        }
    }
    c
}

#[test]
fn typing_accumulates() {
    let c = composer_with("hola");
    assert_eq!(c.text(), "hola");
    assert!(!c.is_empty());
}

#[test]
fn backspace_removes_one_char() {
    let mut c = composer_with("hola");
    c.backspace();
    assert_eq!(c.text(), "hol");
}

#[test]
fn multibyte_chars_delete_whole() {
    // Each of these is 2+ bytes: deleting by byte index would panic or corrupt.
    let mut c = composer_with("ñandú");
    c.backspace();
    assert_eq!(c.text(), "ñand");
    c.move_home();
    c.delete();
    assert_eq!(c.text(), "and");
}

#[test]
fn cursor_lands_after_a_multibyte_char() {
    let c = composer_with("ñ");
    let (x, _) = c.cursor_offset(40);
    assert_eq!(x, 3, "prompt is 2 columns, the char is 1");
}

#[test]
fn newline_splits_at_the_cursor() {
    let mut c = composer_with("abcd");
    c.move_left();
    c.insert_newline();
    assert_eq!(c.text(), "abc\nd");
}

#[test]
fn backspace_joins_lines() {
    let mut c = composer_with("ab\ncd");
    c.move_home();
    c.backspace();
    assert_eq!(c.text(), "abcd");
}

#[test]
fn delete_pulls_up_the_next_line() {
    let mut c = composer_with("ab\ncd");
    c.move_up();
    c.move_end();
    c.delete();
    assert_eq!(c.text(), "abcd");
}

#[test]
fn moving_up_clamps_to_a_shorter_line() {
    let mut c = composer_with("ab\nlonger");
    c.move_end();
    c.move_up();
    c.insert_char('!');
    assert_eq!(c.text(), "ab!\nlonger", "column clamped to the short line");
}

#[test]
fn submitting_empties_the_composer() {
    let mut c = composer_with("send me");
    assert_eq!(c.take().as_deref(), Some("send me"));
    assert!(c.is_empty());
    assert_eq!(c.text(), "");
}

#[test]
fn whitespace_alone_is_not_submitted() {
    let mut c = composer_with("   ");
    assert_eq!(c.take(), None);
}

#[test]
fn height_grows_with_wrapping() {
    // Width 12 leaves 10 usable columns after the prompt.
    let c = composer_with(&"x".repeat(25));
    assert_eq!(c.height(12), 3);
}

#[test]
fn height_counts_every_line() {
    let c = composer_with("one\ntwo\nthree");
    assert_eq!(c.height(40), 3);
}

#[test]
fn an_empty_composer_is_one_row() {
    let c = Composer::new("type here");
    assert_eq!(c.height(40), 1);
}

#[test]
fn cursor_follows_the_wrap() {
    let c = composer_with(&"x".repeat(15));
    // 10 usable columns: the cursor sits on the second row, column 5.
    assert_eq!(c.cursor_offset(12), (7, 1));
}
