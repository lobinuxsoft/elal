//! A ratatui backend that renders into an in-memory VT100 screen.
//!
//! Inspired by `claw-code-rust/crates/tui/src/test_backend.rs` (MIT, © 2026
//! wangtsiao), but written against a `vt100::Parser` directly instead of
//! wrapping `CrosstermBackend`: 0.30 moved that backend's `writer()` accessors
//! behind an unstable feature, and tests should not depend on those.
//!
//! Asserting on a parsed screen is the only way to tell whether a scroll region
//! actually moved a row — the raw escape bytes look identical either way.
//!
//! Size and cursor position come from the parser. Never ask crossterm: those two
//! queries go to the real stdout regardless of the writer, which during a test
//! run is the terminal running `cargo test`.

use std::fmt;
use std::io;
use std::io::Write;

use crossterm::cursor::Hide;
use crossterm::cursor::MoveTo;
use crossterm::cursor::Show;
use crossterm::queue;
use crossterm::style::Colors;
use crossterm::style::Print;
use crossterm::style::SetAttribute;
use crossterm::style::SetColors;
use crossterm::terminal::Clear;
use ratatui::backend::Backend;
use ratatui::backend::ClearType;
use ratatui::backend::IntoCrossterm;
use ratatui::backend::WindowSize;
use ratatui::buffer::Cell;
use ratatui::layout::Position;
use ratatui::layout::Size;

pub struct Vt100Backend {
    parser: vt100::Parser,
}

impl Vt100Backend {
    pub fn new(width: u16, height: u16) -> Self {
        // Crossterm strips color when stdout is not a tty, and then every color
        // assertion would pass for the wrong reason.
        crossterm::style::force_color_output(true);
        Self {
            parser: vt100::Parser::new(height, width, 0),
        }
    }

    pub const fn vt100(&self) -> &vt100::Parser {
        &self.parser
    }

    /// The screen as one string per row.
    pub fn rows(&self) -> Vec<String> {
        let (_, cols) = self.parser.screen().size();
        self.parser.screen().rows(0, cols).collect()
    }
}

impl Write for Vt100Backend {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.parser.process(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl fmt::Display for Vt100Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.parser.screen().contents())
    }
}

impl Backend for Vt100Backend {
    type Error = io::Error;

    fn draw<'a, I>(&mut self, content: I) -> io::Result<()>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        for (x, y, cell) in content {
            queue!(
                self,
                MoveTo(x, y),
                SetAttribute(crossterm::style::Attribute::Reset),
                SetColors(Colors::new(
                    cell.fg.into_crossterm(),
                    cell.bg.into_crossterm(),
                )),
                Print(cell.symbol().to_string()),
            )?;
        }
        Ok(())
    }

    fn hide_cursor(&mut self) -> io::Result<()> {
        queue!(self, Hide)
    }

    fn show_cursor(&mut self) -> io::Result<()> {
        queue!(self, Show)
    }

    fn get_cursor_position(&mut self) -> io::Result<Position> {
        Ok(self.parser.screen().cursor_position().into())
    }

    fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> io::Result<()> {
        let position = position.into();
        queue!(self, MoveTo(position.x, position.y))
    }

    fn clear(&mut self) -> io::Result<()> {
        queue!(self, Clear(crossterm::terminal::ClearType::All))
    }

    fn clear_region(&mut self, clear_type: ClearType) -> io::Result<()> {
        let ct = match clear_type {
            ClearType::All => crossterm::terminal::ClearType::All,
            ClearType::AfterCursor => crossterm::terminal::ClearType::FromCursorDown,
            ClearType::BeforeCursor => crossterm::terminal::ClearType::FromCursorUp,
            ClearType::CurrentLine => crossterm::terminal::ClearType::CurrentLine,
            ClearType::UntilNewLine => crossterm::terminal::ClearType::UntilNewLine,
        };
        queue!(self, Clear(ct))
    }

    fn size(&self) -> io::Result<Size> {
        let (rows, cols) = self.parser.screen().size();
        Ok(Size::new(cols, rows))
    }

    fn window_size(&mut self) -> io::Result<WindowSize> {
        Ok(WindowSize {
            columns_rows: self.size()?,
            // Nothing under test reads pixel size.
            pixels: Size {
                width: 640,
                height: 480,
            },
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
