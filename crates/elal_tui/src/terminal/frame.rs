//! The render target handed to draw callbacks.
//!
//! Derived from `ratatui::Frame` (MIT, © 2016-2022 Florian Dehau, © 2023-2025 The
//! Ratatui Developers) via `claw-code-rust/crates/tui/src/custom_terminal.rs`.
//! Full license text in `THIRD_PARTY_LICENSES.md`.

use ratatui::buffer::Buffer;
use ratatui::layout::Position;
use ratatui::layout::Rect;
use ratatui::widgets::Widget;

/// A consistent view of the viewport for the duration of one draw pass.
#[derive(Debug, Hash)]
pub struct Frame<'a> {
    /// Where the cursor goes once the frame is flushed. `None` hides it.
    pub(crate) cursor_position: Option<Position>,
    pub(crate) viewport_area: Rect,
    pub(crate) buffer: &'a mut Buffer,
}

impl Frame<'_> {
    /// The area being rendered into. Stable for the whole draw pass, so widgets
    /// can call it as often as they need.
    ///
    /// Prefer this over the size carried by a resize event: that size describes
    /// the screen, not the inline viewport, and the two differ by every row of
    /// scrollback written so far.
    pub const fn area(&self) -> Rect {
        self.viewport_area
    }

    /// Ratatui 0.30 gates `WidgetRef` behind an unstable feature, so this takes
    /// `Widget` — which is implemented for `&T` wherever a widget is reusable.
    pub fn render_widget<W: Widget>(&mut self, widget: W, area: Rect) {
        widget.render(area, self.buffer);
    }

    /// Show the cursor at `position` after this frame is flushed.
    ///
    /// Leaving this uncalled hides the cursor. Do not mix it with the
    /// `Terminal::{show,hide}_cursor` pair — they fight over the same state.
    pub fn set_cursor_position<P: Into<Position>>(&mut self, position: P) {
        self.cursor_position = Some(position.into());
    }

    pub fn buffer_mut(&mut self) -> &mut Buffer {
        self.buffer
    }
}
