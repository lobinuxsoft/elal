//! What has to hold for scrollback insertion to be safe to call mid-keystroke.

use ratatui::layout::Position;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::text::Span;

use crate::history::Mode;
use crate::history::ResetScrollRegion;
use crate::history::SetScrollRegion;
use crate::history::insert_lines;
use crate::history::insert_lines_with_mode;
use crate::terminal::Terminal;
use crate::vt100_backend::Vt100Backend;

/// A terminal whose viewport sits on the last row, so inserts land right above it.
fn terminal(width: u16, height: u16) -> Terminal<Vt100Backend> {
    let mut term = Terminal::new(Vt100Backend::new(width, height)).expect("terminal");
    term.set_viewport_area(Rect::new(0, height - 1, width, 1));
    term
}

#[test]
fn inserted_line_reaches_the_screen() {
    let mut term = terminal(40, 8);

    insert_lines(&mut term, vec![Line::from("hello scrollback")]).expect("insert");

    let rows = term.backend().rows();
    assert!(
        rows.iter().any(|r| r.contains("hello scrollback")),
        "line not on screen, rows: {rows:?}"
    );
}

#[test]
fn cursor_returns_to_its_place() {
    let mut term = terminal(40, 8);
    let resting = Position { x: 7, y: 7 };
    term.set_cursor_position(resting).expect("place cursor");

    insert_lines(&mut term, vec![Line::from("a line")]).expect("insert");

    assert_eq!(
        term.get_cursor_position().expect("read cursor"),
        resting,
        "insertion must leave the cursor where the user was typing"
    );
}

#[test]
fn viewport_moves_down_by_rows_written() {
    let mut term = terminal(40, 8);
    let before = term.viewport_area;

    insert_lines(&mut term, vec![Line::from("one"), Line::from("two")]).expect("insert");

    assert_eq!(term.viewport_area.y, before.y, "no room below to move into");
    assert_eq!(term.visible_history_rows(), 2);
}

#[test]
fn viewport_slides_when_there_is_room() {
    let mut term = Terminal::new(Vt100Backend::new(32, 8)).expect("terminal");
    term.set_viewport_area(Rect::new(0, 4, 32, 2));

    insert_lines(&mut term, vec![Line::from("pushes the viewport")]).expect("insert");

    assert_eq!(term.viewport_area, Rect::new(0, 5, 32, 2));
    assert_eq!(term.visible_history_rows(), 1);
}

#[test]
fn a_wrapped_line_counts_every_row() {
    let mut term = Terminal::new(Vt100Backend::new(20, 10)).expect("terminal");
    term.set_viewport_area(Rect::new(0, 2, 20, 1));

    // 45 columns of text over a 20-column screen: three physical rows.
    insert_lines(&mut term, vec![Line::from("x".repeat(45))]).expect("insert");

    assert_eq!(term.viewport_area.y, 5, "viewport moved by wrapped rows");
}

#[test]
fn line_style_survives_on_every_span() {
    let mut term = terminal(40, 10);
    let line = Line::from(vec!["> ".into(), "quoted".into()]).style(Color::Green);

    insert_lines(&mut term, vec![line]).expect("insert");

    let screen = term.backend().vt100().screen();
    let colored = (0..10).any(|row| {
        (0..40).any(|col| {
            screen
                .cell(row, col)
                .is_some_and(|c| c.has_contents() && c.fgcolor() != vt100::Color::Default)
        })
    });
    assert!(colored, "line-level style never reached the spans");
}

#[test]
fn plain_text_after_a_colored_span_resets() {
    let mut term = terminal(40, 6);
    let line = Line::from(vec![
        Span::styled("1. ", Style::default().fg(Color::LightBlue)),
        Span::raw("plain"),
    ]);

    insert_lines(&mut term, vec![line]).expect("insert");

    let screen = term.backend().vt100().screen();
    let row = (0..6)
        .find(|&r| (0..40).any(|c| screen.cell(r, c).is_some_and(|cell| cell.contents() == "1")))
        .expect("row with the marker");

    assert_ne!(
        screen.cell(row, 0).expect("marker cell").fgcolor(),
        vt100::Color::Default,
        "marker should keep its color"
    );
    assert_eq!(
        screen.cell(row, 3).expect("text cell").fgcolor(),
        vt100::Color::Default,
        "text after the marker should not inherit it"
    );
}

#[test]
fn zellij_mode_writes_without_scroll_regions() {
    let mut term = Terminal::new(Vt100Backend::new(32, 8)).expect("terminal");
    term.set_viewport_area(Rect::new(0, 4, 32, 2));

    insert_lines_with_mode(&mut term, vec![Line::from("zellij line")], Mode::Zellij)
        .expect("insert");

    let rows = term.backend().rows();
    assert!(
        rows.iter().any(|r| r.contains("zellij line")),
        "line not on screen, rows: {rows:?}"
    );
    assert_eq!(term.viewport_area, Rect::new(0, 5, 32, 2));
}

#[test]
fn scroll_region_commands_are_ansi() {
    use crossterm::Command;

    let mut out = String::new();
    SetScrollRegion(2..8).write_ansi(&mut out).expect("write");
    ResetScrollRegion.write_ansi(&mut out).expect("write");

    assert_eq!(out, "\x1b[2;8r\x1b[r");
}
