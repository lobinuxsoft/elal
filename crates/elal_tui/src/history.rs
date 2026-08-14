//! Writing transcript lines into the terminal's own scrollback.
//!
//! Ported from `claw-code-rust/crates/tui/src/insert_history.rs` (MIT, © 2026
//! wangtsiao). Full license text in `THIRD_PARTY_LICENSES.md`.
//!
//! Lines committed here stop being ours: the terminal owns them, so selection,
//! copy and scrollback work natively and the transcript outlives the process.

use std::fmt;
use std::io;
use std::io::Write;

use crossterm::Command;
use crossterm::cursor::MoveDown;
use crossterm::cursor::MoveTo;
use crossterm::cursor::MoveToColumn;
use crossterm::cursor::RestorePosition;
use crossterm::cursor::SavePosition;
use crossterm::queue;
use crossterm::style::Color as CColor;
use crossterm::style::Colors;
use crossterm::style::Print;
use crossterm::style::SetAttribute;
use crossterm::style::SetBackgroundColor;
use crossterm::style::SetColors;
use crossterm::style::SetForegroundColor;
use crossterm::terminal::Clear;
use crossterm::terminal::ClearType;
use ratatui::backend::IntoCrossterm;
use ratatui::layout::Size;
use ratatui::prelude::Backend;
use ratatui::style::Color;
use ratatui::style::Modifier;
use ratatui::text::Line;
use ratatui::text::Span;

use crate::terminal::Terminal;

/// Which escape sequences to use when making room above the viewport.
///
/// [`Mode::Standard`] uses a `DECSTBM` scroll region plus Reverse Index (`ESC M`)
/// to slide existing rows down without repainting them. Zellij drops those
/// silently, so [`Mode::Zellij`] falls back to emitting newlines at the bottom of
/// the screen and writing at absolute positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Standard,
    Zellij,
}

impl Mode {
    /// Zellij announces itself through `ZELLIJ` in the environment.
    pub fn detect() -> Self {
        if std::env::var_os("ZELLIJ").is_some() {
            Self::Zellij
        } else {
            Self::Standard
        }
    }
}

/// Commit `lines` to the scrollback above the viewport.
pub fn insert_lines<B>(terminal: &mut Terminal<B>, lines: Vec<Line>) -> io::Result<()>
where
    B: Backend<Error = io::Error> + Write,
{
    insert_lines_with_mode(terminal, lines, Mode::Standard)
}

/// Commit `lines` using the escape strategy `mode` selects.
///
/// The viewport moves down by however many rows were inserted, and the cursor
/// ends where it started — a user mid-keystroke must not see it jump.
pub fn insert_lines_with_mode<B>(
    terminal: &mut Terminal<B>,
    lines: Vec<Line>,
    mode: Mode,
) -> io::Result<()>
where
    B: Backend<Error = io::Error> + Write,
{
    let screen_size = terminal.backend().size().unwrap_or(Size::new(0, 0));

    let mut area = terminal.viewport_area;
    let mut should_update_area = false;
    let last_cursor_pos = terminal.last_known_cursor_pos;
    let writer = terminal.backend_mut();

    // A logical line wider than the screen occupies several physical rows, and
    // it is physical rows that the viewport has to move by.
    let wrap_width = usize::from(screen_size.width.max(1));
    let wrapped_rows: usize = lines
        .iter()
        .map(|line| line.width().max(1).div_ceil(wrap_width))
        .sum();
    let wrapped_rows = wrapped_rows as u16;

    if matches!(mode, Mode::Zellij) {
        let space_below = screen_size.height.saturating_sub(area.bottom());
        let shift_down = wrapped_rows.min(space_below);
        let scroll_up_amount = wrapped_rows.saturating_sub(shift_down);

        if scroll_up_amount > 0 {
            // No scroll region available: scroll the whole screen by printing
            // newlines from the bottom row.
            queue!(writer, MoveTo(0, screen_size.height.saturating_sub(1)))?;
            for _ in 0..scroll_up_amount {
                queue!(writer, Print("\n"))?;
            }
        }

        if shift_down > 0 {
            area.y += shift_down;
            should_update_area = true;
        }

        let cursor_top = area.top().saturating_sub(scroll_up_amount + shift_down);
        queue!(writer, MoveTo(0, cursor_top))?;

        for (i, line) in lines.iter().enumerate() {
            if i > 0 {
                queue!(writer, Print("\r\n"))?;
            }
            write_line(writer, line, wrap_width)?;
        }
    } else {
        let cursor_top = if area.bottom() < screen_size.height {
            // There is room below: push the viewport down instead of scrolling
            // the screen, so nothing already on screen has to be repainted.
            let scroll_amount = wrapped_rows.min(screen_size.height - area.bottom());

            let top_1based = area.top() + 1;
            queue!(writer, SetScrollRegion(top_1based..screen_size.height))?;
            queue!(writer, MoveTo(0, area.top()))?;
            for _ in 0..scroll_amount {
                queue!(writer, Print("\x1bM"))?;
            }
            queue!(writer, ResetScrollRegion)?;

            let cursor_top = area.top().saturating_sub(1);
            area.y += scroll_amount;
            should_update_area = true;
            cursor_top
        } else {
            area.top().saturating_sub(1)
        };

        // Confine scrolling to the rows above the viewport, then write from the
        // bottom of that region. Only those rows move; the viewport stays put.
        //
        // ┌─Screen───────────────────────┐
        // │┌╌Scroll region╌╌╌╌╌╌╌╌╌╌╌╌╌╌┐│
        // │┆                            ┆│
        // │█╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘│
        // │╭─Viewport───────────────────╮│
        // │╰────────────────────────────╯│
        // └──────────────────────────────┘
        queue!(writer, SetScrollRegion(1..area.top()))?;

        // MoveTo rather than `Terminal::set_cursor_position`: that would update
        // `last_known_cursor_pos`, and this whole function has to be
        // cursor-position-neutral.
        queue!(writer, MoveTo(0, cursor_top))?;

        for line in &lines {
            queue!(writer, Print("\r\n"))?;
            write_line(writer, line, wrap_width)?;
        }

        queue!(writer, ResetScrollRegion)?;
    }

    queue!(writer, MoveTo(last_cursor_pos.x, last_cursor_pos.y))?;

    if should_update_area {
        terminal.set_viewport_area(area);
    }
    if wrapped_rows > 0 {
        terminal.note_history_rows(wrapped_rows);
    }

    Ok(())
}

/// Write one logical line: clear the rows it will wrap onto, apply its colors,
/// then emit its spans. Cursor placement is the caller's job.
fn write_line<W: Write>(writer: &mut W, line: &Line, wrap_width: usize) -> io::Result<()> {
    let physical_rows = line.width().max(1).div_ceil(wrap_width) as u16;
    if physical_rows > 1 {
        // Wrapped rows are written by the terminal itself, so anything already
        // there survives unless we clear it first.
        queue!(writer, SavePosition)?;
        for _ in 1..physical_rows {
            queue!(writer, MoveDown(1), MoveToColumn(0))?;
            queue!(writer, Clear(ClearType::UntilNewLine))?;
        }
        queue!(writer, RestorePosition)?;
    }
    queue!(
        writer,
        SetColors(Colors::new(
            line.style
                .fg
                .map_or(CColor::Reset, IntoCrossterm::into_crossterm),
            line.style
                .bg
                .map_or(CColor::Reset, IntoCrossterm::into_crossterm),
        ))
    )?;
    queue!(writer, Clear(ClearType::UntilNewLine))?;
    // Fold the line style into every span: the escape stream has no notion of a
    // line-level style, so a green blockquote would lose its color otherwise.
    let merged: Vec<Span> = line
        .spans
        .iter()
        .map(|s| Span {
            style: s.style.patch(line.style),
            content: s.content.clone(),
        })
        .collect();
    write_spans(writer, merged.iter())
}

fn write_spans<'a, I>(writer: &mut impl Write, content: I) -> io::Result<()>
where
    I: IntoIterator<Item = &'a Span<'a>>,
{
    let mut fg = Color::Reset;
    let mut bg = Color::Reset;
    let mut last_modifier = Modifier::empty();
    for span in content {
        let mut modifier = Modifier::empty();
        modifier.insert(span.style.add_modifier);
        modifier.remove(span.style.sub_modifier);
        if modifier != last_modifier {
            crate::terminal::queue_modifier_diff(writer, last_modifier, modifier)?;
            last_modifier = modifier;
        }
        let next_fg = span.style.fg.unwrap_or(Color::Reset);
        let next_bg = span.style.bg.unwrap_or(Color::Reset);
        if next_fg != fg || next_bg != bg {
            queue!(
                writer,
                SetColors(Colors::new(
                    next_fg.into_crossterm(),
                    next_bg.into_crossterm(),
                ))
            )?;
            fg = next_fg;
            bg = next_bg;
        }

        queue!(writer, Print(span.content.clone()))?;
    }

    queue!(
        writer,
        SetForegroundColor(CColor::Reset),
        SetBackgroundColor(CColor::Reset),
        SetAttribute(crossterm::style::Attribute::Reset),
    )
}

/// `DECSTBM` — confine scrolling to rows `start..end`, 1-based and inclusive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetScrollRegion(pub std::ops::Range<u16>);

impl Command for SetScrollRegion {
    fn write_ansi(&self, f: &mut impl fmt::Write) -> fmt::Result {
        write!(f, "\x1b[{};{}r", self.0.start, self.0.end)
    }

    #[cfg(windows)]
    fn execute_winapi(&self) -> io::Result<()> {
        panic!("SetScrollRegion has no WinAPI form; it must go out as ANSI");
    }

    #[cfg(windows)]
    fn is_ansi_code_supported(&self) -> bool {
        // Windows Virtual Terminal understands DECSTBM. Claiming support keeps
        // crossterm from falling back to WinAPI, which has no equivalent.
        true
    }
}

/// Release the scroll region set by [`SetScrollRegion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResetScrollRegion;

impl Command for ResetScrollRegion {
    fn write_ansi(&self, f: &mut impl fmt::Write) -> fmt::Result {
        write!(f, "\x1b[r")
    }

    #[cfg(windows)]
    fn execute_winapi(&self) -> io::Result<()> {
        panic!("ResetScrollRegion has no WinAPI form; it must go out as ANSI");
    }

    #[cfg(windows)]
    fn is_ansi_code_supported(&self) -> bool {
        true
    }
}
