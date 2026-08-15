//! The input area: a multi-line editor that grows with what is typed.
//!
//! Deliberately not a general text editor. It holds one message being written,
//! so there is no selection, no undo and no scrolling — the widget grows instead,
//! and the viewport grows with it.
//!
//! Positions are in **characters**, never bytes: `String` is UTF-8, so byte
//! indices split multi-byte characters and panic on slicing.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::text::Span;
use ratatui::widgets::Widget;

/// What a key press did to the composer, for the caller to act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Nothing worth redrawing for.
    None,
    /// The text changed; redraw.
    Redraw,
    /// The user submitted this message.
    Submit(String),
}

/// The prompt marker, and the indent every wrapped row lines up against.
const PROMPT: &str = "❯ ";

#[derive(Debug, Default)]
pub struct Composer {
    lines: Vec<String>,
    /// Cursor row within `lines`.
    row: usize,
    /// Cursor column, counted in characters of the current row.
    col: usize,
    /// Shown when empty, so the composer is never a blank box.
    placeholder: String,
}

impl Composer {
    pub fn new(placeholder: impl Into<String>) -> Self {
        Self {
            lines: vec![String::new()],
            row: 0,
            col: 0,
            placeholder: placeholder.into(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.lines.iter().all(String::is_empty)
    }

    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    /// Rows the widget needs at `width`, so the caller can size the viewport.
    pub fn height(&self, width: u16) -> u16 {
        let usable = usable_width(width);
        let rows: usize = self
            .lines
            .iter()
            .map(|line| wrapped_rows(line.chars().count(), usable))
            .sum();
        u16::try_from(rows).unwrap_or(u16::MAX).max(1)
    }

    pub fn insert_char(&mut self, ch: char) {
        let line = &mut self.lines[self.row];
        let byte = byte_index(line, self.col);
        line.insert(byte, ch);
        self.col += 1;
    }

    /// Split the current line at the cursor — the Shift+Enter newline.
    pub fn insert_newline(&mut self) {
        let line = &mut self.lines[self.row];
        let byte = byte_index(line, self.col);
        let tail = line.split_off(byte);
        self.lines.insert(self.row + 1, tail);
        self.row += 1;
        self.col = 0;
    }

    /// Delete backwards, joining with the previous line at column 0.
    pub fn backspace(&mut self) {
        if self.col > 0 {
            let line = &mut self.lines[self.row];
            let byte = byte_index(line, self.col - 1);
            line.remove(byte);
            self.col -= 1;
        } else if self.row > 0 {
            let current = self.lines.remove(self.row);
            self.row -= 1;
            self.col = self.lines[self.row].chars().count();
            self.lines[self.row].push_str(&current);
        }
    }

    /// Delete forwards, pulling up the next line at end of line.
    pub fn delete(&mut self) {
        let len = self.lines[self.row].chars().count();
        if self.col < len {
            let line = &mut self.lines[self.row];
            let byte = byte_index(line, self.col);
            line.remove(byte);
        } else if self.row + 1 < self.lines.len() {
            let next = self.lines.remove(self.row + 1);
            self.lines[self.row].push_str(&next);
        }
    }

    pub fn move_left(&mut self) {
        if self.col > 0 {
            self.col -= 1;
        } else if self.row > 0 {
            self.row -= 1;
            self.col = self.lines[self.row].chars().count();
        }
    }

    pub fn move_right(&mut self) {
        if self.col < self.lines[self.row].chars().count() {
            self.col += 1;
        } else if self.row + 1 < self.lines.len() {
            self.row += 1;
            self.col = 0;
        }
    }

    /// Move up a line, keeping the column where the shorter line allows.
    pub fn move_up(&mut self) {
        if self.row > 0 {
            self.row -= 1;
            self.col = self.col.min(self.lines[self.row].chars().count());
        }
    }

    pub fn move_down(&mut self) {
        if self.row + 1 < self.lines.len() {
            self.row += 1;
            self.col = self.col.min(self.lines[self.row].chars().count());
        }
    }

    pub fn move_home(&mut self) {
        self.col = 0;
    }

    pub fn move_end(&mut self) {
        self.col = self.lines[self.row].chars().count();
    }

    /// Hand over the message and reset to empty. `None` when there is nothing
    /// but whitespace — Enter on an empty prompt should do nothing.
    pub fn take(&mut self) -> Option<String> {
        let text = self.text();
        if text.trim().is_empty() {
            return None;
        }
        self.lines = vec![String::new()];
        self.row = 0;
        self.col = 0;
        Some(text)
    }

    /// Where the terminal cursor goes, relative to the widget's origin.
    pub fn cursor_offset(&self, width: u16) -> (u16, u16) {
        let usable = usable_width(width);
        let rows_above: usize = self.lines[..self.row]
            .iter()
            .map(|line| wrapped_rows(line.chars().count(), usable))
            .sum();
        // `usable_width` floors at 1, so this cannot divide by zero.
        let (row_in_line, col) = (self.col / usable, self.col % usable);
        let x = col + PROMPT.chars().count();
        (
            u16::try_from(x).unwrap_or(u16::MAX),
            u16::try_from(rows_above + row_in_line).unwrap_or(u16::MAX),
        )
    }

    /// The composer as styled lines: prompt on the first row, indent on the rest.
    fn render_lines(&self, width: u16) -> Vec<Line<'static>> {
        if self.is_empty() {
            return vec![Line::from(vec![
                Span::styled(PROMPT, Style::default().cyan()),
                Span::styled(self.placeholder.clone(), Style::default().dark_gray()),
            ])];
        }

        let usable = usable_width(width);
        let indent = " ".repeat(PROMPT.chars().count());
        let mut out = Vec::new();
        for (i, line) in self.lines.iter().enumerate() {
            for (j, chunk) in wrap_chars(line, usable).into_iter().enumerate() {
                let marker = if i == 0 && j == 0 {
                    Span::styled(PROMPT, Style::default().cyan())
                } else {
                    Span::raw(indent.clone())
                };
                out.push(Line::from(vec![marker, Span::raw(chunk)]));
            }
        }
        out
    }
}

impl Widget for &Composer {
    fn render(self, area: Rect, buf: &mut Buffer) {
        for (i, line) in self.render_lines(area.width).into_iter().enumerate() {
            let Ok(offset) = u16::try_from(i) else { break };
            if offset >= area.height {
                break;
            }
            buf.set_line(area.x, area.y + offset, &line, area.width);
        }
    }
}

/// Columns left for text once the prompt is drawn. At least 1, so a pathological
/// width cannot divide by zero.
fn usable_width(width: u16) -> usize {
    usize::from(width)
        .saturating_sub(PROMPT.chars().count())
        .max(1)
}

/// Rows a line of `len` characters occupies. An empty line still takes one.
const fn wrapped_rows(len: usize, usable: usize) -> usize {
    if len == 0 { 1 } else { len.div_ceil(usable) }
}

/// Split into chunks of `usable` characters. Hard wrapping, not word wrapping:
/// the composer shows what was typed, and moving words around would put the
/// cursor somewhere the user did not put it.
fn wrap_chars(line: &str, usable: usize) -> Vec<String> {
    if line.is_empty() {
        return vec![String::new()];
    }
    let chars: Vec<char> = line.chars().collect();
    chars
        .chunks(usable)
        .map(|chunk| chunk.iter().collect())
        .collect()
}

/// Byte offset of character `col`, or the end of the string.
fn byte_index(line: &str, col: usize) -> usize {
    line.char_indices()
        .nth(col)
        .map_or_else(|| line.len(), |(byte, _)| byte)
}
