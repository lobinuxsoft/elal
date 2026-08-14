//! A terminal that owns a viewport instead of the screen.
//!
//! Derived from `ratatui::Terminal` (MIT, © 2016-2022 Florian Dehau, © 2023-2025
//! The Ratatui Developers) via `claw-code-rust/crates/tui/src/custom_terminal.rs`.
//! Full license text in `THIRD_PARTY_LICENSES.md`.
//!
//! The difference that justifies the fork: [`Terminal::viewport_area`] is public
//! and movable. Rows written into the scrollback push the viewport down, and
//! [`crate::history::insert_lines`] tells the terminal it happened. Ratatui's own
//! terminal assumes its area only changes on resize.

use std::io;
use std::io::Write;

use ratatui::backend::Backend;
use ratatui::backend::ClearType;
use ratatui::buffer::Buffer;
use ratatui::layout::Position;
use ratatui::layout::Rect;
use ratatui::layout::Size;

use super::flush::DrawCommand;
use super::flush::diff_buffers;
use super::flush::draw;
use super::frame::Frame;

#[derive(Debug)]
pub struct Terminal<B>
where
    B: Backend<Error = io::Error> + Write,
{
    backend: B,
    /// Current and previous frame. Compared on flush so only changes are written.
    buffers: [Buffer; 2],
    current: usize,
    pub hidden_cursor: bool,
    /// The rows this terminal draws into. Everything above belongs to the shell.
    pub viewport_area: Rect,
    pub last_known_screen_size: Size,
    /// Where the cursor sat after the last flush. History insertion restores it,
    /// so writing scrollback never disturbs what the user is typing.
    pub last_known_cursor_pos: Position,
    /// Rows of transcript currently visible above the viewport.
    visible_history_rows: u16,
}

impl<B> Drop for Terminal<B>
where
    B: Backend<Error = io::Error> + Write,
{
    fn drop(&mut self) {
        if self.hidden_cursor {
            let _ = self.show_cursor();
        }
    }
}

impl<B> Terminal<B>
where
    B: Backend<Error = io::Error> + Write,
{
    /// Start a session just below the current cursor row, leaving whatever the
    /// shell already printed untouched.
    pub fn new(mut backend: B) -> io::Result<Self> {
        let screen_size = backend.size()?;
        let cursor_pos = backend.get_cursor_position().unwrap_or_else(|err| {
            // Some PTYs never answer CPR (`ESC[6n`). Starting at the origin is
            // wrong by a few rows at worst; failing here would kill the UI.
            tracing::warn!("failed to read initial cursor position, assuming origin: {err}");
            Position { x: 0, y: 0 }
        });
        Ok(Self {
            backend,
            buffers: [Buffer::empty(Rect::ZERO), Buffer::empty(Rect::ZERO)],
            current: 0,
            hidden_cursor: false,
            viewport_area: Rect::new(0, cursor_pos.y.saturating_add(1), 0, 0),
            last_known_screen_size: screen_size,
            last_known_cursor_pos: cursor_pos,
            visible_history_rows: 0,
        })
    }

    pub fn get_frame(&mut self) -> Frame<'_> {
        Frame {
            cursor_position: None,
            viewport_area: self.viewport_area,
            buffer: self.current_buffer_mut(),
        }
    }

    fn current_buffer(&self) -> &Buffer {
        &self.buffers[self.current]
    }

    fn current_buffer_mut(&mut self) -> &mut Buffer {
        &mut self.buffers[self.current]
    }

    fn previous_buffer(&self) -> &Buffer {
        &self.buffers[1 - self.current]
    }

    fn previous_buffer_mut(&mut self) -> &mut Buffer {
        &mut self.buffers[1 - self.current]
    }

    /// Forget what is on screen so the next draw repaints everything.
    fn reset_draw_buffers(&mut self) {
        self.current_buffer_mut().reset();
        self.previous_buffer_mut().reset();
    }

    pub const fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Write the difference between the last two frames to the backend.
    pub fn flush(&mut self) -> io::Result<()> {
        let updates = diff_buffers(self.previous_buffer(), self.current_buffer());
        if let Some(&DrawCommand::Put { x, y, .. }) = updates.iter().rfind(|c| c.is_put()) {
            self.last_known_cursor_pos = Position { x, y };
        }
        draw(&mut self.backend, updates.into_iter())
    }

    pub fn resize(&mut self, screen_size: Size) {
        self.last_known_screen_size = screen_size;
    }

    /// Move or resize the viewport, resizing both diff buffers to match.
    pub fn set_viewport_area(&mut self, area: Rect) {
        self.current_buffer_mut().resize(area);
        self.previous_buffer_mut().resize(area);
        self.viewport_area = area;
        self.visible_history_rows = self.visible_history_rows.min(area.top());
    }

    pub fn autoresize(&mut self) -> io::Result<()> {
        let screen_size = self.size()?;
        if screen_size != self.last_known_screen_size {
            self.resize(screen_size);
        }
        Ok(())
    }

    /// Render one frame: resize if needed, run `render_callback`, flush the diff,
    /// then place the cursor where the callback asked.
    ///
    /// The callback must render the whole viewport every time, including the
    /// parts that did not change — the diff is what decides that, not the caller.
    pub fn draw<F>(&mut self, render_callback: F) -> io::Result<()>
    where
        F: FnOnce(&mut Frame),
    {
        self.autoresize()?;

        let mut frame = self.get_frame();
        render_callback(&mut frame);
        // Take the position out before dropping the frame: it borrows the buffer
        // mutably and flushing needs that borrow back.
        let cursor_position = frame.cursor_position;

        self.flush()?;

        match cursor_position {
            None => self.hide_cursor()?,
            Some(position) => {
                self.show_cursor()?;
                self.set_cursor_position(position)?;
            }
        }

        self.swap_buffers();
        Backend::flush(&mut self.backend)
    }

    pub fn hide_cursor(&mut self) -> io::Result<()> {
        self.backend.hide_cursor()?;
        self.hidden_cursor = true;
        Ok(())
    }

    pub fn show_cursor(&mut self) -> io::Result<()> {
        self.backend.show_cursor()?;
        self.hidden_cursor = false;
        Ok(())
    }

    pub fn get_cursor_position(&mut self) -> io::Result<Position> {
        self.backend.get_cursor_position()
    }

    pub fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> io::Result<()> {
        let position = position.into();
        self.backend.set_cursor_position(position)?;
        self.last_known_cursor_pos = position;
        Ok(())
    }

    /// Clear the viewport and force a full repaint next draw.
    pub fn clear(&mut self) -> io::Result<()> {
        if self.viewport_area.is_empty() {
            return Ok(());
        }
        self.backend
            .set_cursor_position(self.viewport_area.as_position())?;
        self.backend.clear_region(ClearType::AfterCursor)?;
        self.reset_draw_buffers();
        Ok(())
    }

    /// Clear the live viewport but keep the transcript already committed above it.
    ///
    /// This is the exit path: the shell prompt resumes where the viewport was,
    /// and the conversation stays in the scrollback where the user can read it.
    pub fn clear_viewport(&mut self) -> io::Result<()> {
        if self.viewport_area.is_empty() {
            return Ok(());
        }
        self.set_cursor_position(self.viewport_area.as_position())?;
        self.backend.clear_region(ClearType::AfterCursor)?;
        Write::flush(&mut self.backend)?;
        self.reset_draw_buffers();
        Ok(())
    }

    /// Park the cursor on the first column below `area`, clamped to the last row.
    pub fn set_cursor_below(&mut self, area: Rect) -> io::Result<()> {
        let screen_height = self.size()?.height;
        let target_y = area.bottom().min(screen_height.saturating_sub(1));
        self.set_cursor_position(Position { x: 0, y: target_y })
    }

    /// Discard the diff state after something moved screen content behind
    /// ratatui's back — raw newline scrolling, for one. Without this the next
    /// draw would diff against rows that are no longer where it thinks.
    pub fn invalidate_viewport(&mut self) {
        self.reset_draw_buffers();
    }

    pub const fn visible_history_rows(&self) -> u16 {
        self.visible_history_rows
    }

    /// Record that `inserted_rows` of transcript were written above the viewport.
    pub(crate) fn note_history_rows(&mut self, inserted_rows: u16) {
        self.visible_history_rows = self
            .visible_history_rows
            .saturating_add(inserted_rows)
            .min(self.viewport_area.top());
    }

    pub fn swap_buffers(&mut self) {
        self.previous_buffer_mut().reset();
        self.current = 1 - self.current;
    }

    pub fn size(&self) -> io::Result<Size> {
        self.backend.size()
    }
}
