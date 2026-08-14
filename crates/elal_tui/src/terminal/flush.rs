//! Buffer diffing and the escape sequences that apply it.
//!
//! Derived from `ratatui`'s terminal internals (MIT, © 2016-2022 Florian Dehau,
//! © 2023-2025 The Ratatui Developers) via
//! `claw-code-rust/crates/tui/src/custom_terminal.rs`. Full license text in
//! `THIRD_PARTY_LICENSES.md`.

use std::io;
use std::io::Write;

use crossterm::cursor::MoveTo;
use crossterm::queue;
use crossterm::style::Colors;
use crossterm::style::Print;
use crossterm::style::SetAttribute;
use crossterm::style::SetBackgroundColor;
use crossterm::style::SetColors;
use crossterm::style::SetForegroundColor;
use crossterm::terminal::Clear;
use ratatui::backend::IntoCrossterm;
use ratatui::buffer::Buffer;
use ratatui::buffer::Cell;
use ratatui::buffer::CellDiffOption;
use ratatui::layout::Position;
use ratatui::style::Color;
use ratatui::style::Modifier;
use unicode_width::UnicodeWidthStr;

/// Display width of a cell symbol, ignoring OSC escape sequences.
///
/// OSC payloads (OSC 8 hyperlinks, `ESC ] … BEL`) consume no columns, but
/// `UnicodeWidthStr::width` counts the characters inside them — the URL would
/// push every following cell out of place. Strip them before measuring.
pub(super) fn display_width(s: &str) -> usize {
    if !s.contains('\x1B') {
        return s.width();
    }

    let mut visible = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch == '\x1B' && chars.clone().next() == Some(']') {
            chars.next();
            for c in chars.by_ref() {
                if c == '\x07' {
                    break;
                }
            }
            continue;
        }
        visible.push(ch);
    }
    visible.width()
}

#[derive(Debug)]
pub(super) enum DrawCommand {
    Put { x: u16, y: u16, cell: Cell },
    ClearToEnd { x: u16, y: u16, bg: Color },
}

impl DrawCommand {
    pub(super) const fn is_put(&self) -> bool {
        matches!(self, Self::Put { .. })
    }
}

/// Commands that turn `previous` into `next`.
///
/// Beyond a plain cell-by-cell comparison this collapses each row's blank tail
/// into one `ClearToEnd`: a row of trailing spaces costs one escape instead of
/// one write per column, which is most of the traffic on a wide terminal.
pub(super) fn diff_buffers(previous: &Buffer, next: &Buffer) -> Vec<DrawCommand> {
    let previous_cells = &previous.content;
    let next_cells = &next.content;

    let mut updates = vec![];
    let mut last_nonblank_columns = vec![0; previous.area.height as usize];
    for y in 0..previous.area.height {
        let row_start = y as usize * previous.area.width as usize;
        let row_end = row_start + previous.area.width as usize;
        let row = &next_cells[row_start..row_end];
        let bg = row.last().map_or(Color::Reset, |cell| cell.bg);

        // Rightmost column that still matters: any non-space glyph, any cell whose
        // background differs from the row's trailing background, or any modifier.
        // Multi-width glyphs extend it through their full displayed width.
        let mut last_nonblank_column = 0usize;
        let mut column = 0usize;
        while column < row.len() {
            let cell = &row[column];
            let width = display_width(cell.symbol());
            if cell.symbol() != " " || cell.bg != bg || cell.modifier != Modifier::empty() {
                last_nonblank_column = column + width.saturating_sub(1);
            }
            column += width.max(1); // zero-width symbols still occupy a slot
        }

        if last_nonblank_column + 1 < row.len() {
            let (x, y) = previous.pos_of(row_start + last_nonblank_column + 1);
            updates.push(DrawCommand::ClearToEnd { x, y, bg });
        }

        last_nonblank_columns[y as usize] = last_nonblank_column as u16;
    }

    // Cells invalidated by drawing over a preceding multi-width character.
    let mut invalidated: usize = 0;
    // Cells to skip because a preceding multi-width character already covers them.
    let mut to_skip: usize = 0;
    for (i, (current, previous_cell)) in next_cells.iter().zip(previous_cells.iter()).enumerate() {
        // 0.30 replaced the `skip` flag with `diff_option`. `Skip` is owned by an
        // escape sequence (image, hyperlink) and must not be repainted;
        // `AlwaysUpdate` belongs to a renderer we do not control, so it is
        // repainted even when the cell compares equal.
        let skip = current.diff_option == CellDiffOption::Skip;
        let always = current.diff_option == CellDiffOption::AlwaysUpdate;
        if !skip && (always || current != previous_cell || invalidated > 0) && to_skip == 0 {
            let (x, y) = previous.pos_of(i);
            let row = i / previous.area.width as usize;
            if x <= last_nonblank_columns[row] {
                updates.push(DrawCommand::Put {
                    x,
                    y,
                    cell: next_cells[i].clone(),
                });
            }
        }

        to_skip = display_width(current.symbol()).saturating_sub(1);

        let affected_width =
            display_width(current.symbol()).max(display_width(previous_cell.symbol()));
        invalidated = affected_width.max(invalidated).saturating_sub(1);
    }
    updates
}

/// Write `commands` to `writer`, tracking style state so unchanged attributes
/// are not re-emitted, and skipping the cursor move when the next cell is the
/// one immediately to the right.
pub(super) fn draw<I>(writer: &mut impl Write, commands: I) -> io::Result<()>
where
    I: Iterator<Item = DrawCommand>,
{
    let mut fg = Color::Reset;
    let mut bg = Color::Reset;
    let mut modifier = Modifier::empty();
    let mut last_pos: Option<Position> = None;
    for command in commands {
        let (x, y) = match command {
            DrawCommand::Put { x, y, .. } | DrawCommand::ClearToEnd { x, y, .. } => (x, y),
        };
        if !matches!(last_pos, Some(p) if x == p.x + 1 && y == p.y) {
            queue!(writer, MoveTo(x, y))?;
        }
        last_pos = Some(Position { x, y });
        match command {
            DrawCommand::Put { cell, .. } => {
                if cell.modifier != modifier {
                    queue_modifier_diff(writer, modifier, cell.modifier)?;
                    modifier = cell.modifier;
                }
                if cell.fg != fg || cell.bg != bg {
                    queue!(
                        writer,
                        SetColors(Colors::new(
                            cell.fg.into_crossterm(),
                            cell.bg.into_crossterm(),
                        ))
                    )?;
                    fg = cell.fg;
                    bg = cell.bg;
                }

                queue!(writer, Print(cell.symbol()))?;
            }
            DrawCommand::ClearToEnd { bg: clear_bg, .. } => {
                queue!(writer, SetAttribute(crossterm::style::Attribute::Reset))?;
                modifier = Modifier::empty();
                queue!(writer, SetBackgroundColor(clear_bg.into_crossterm()))?;
                bg = clear_bg;
                queue!(writer, Clear(crossterm::terminal::ClearType::UntilNewLine))?;
            }
        }
    }

    queue!(
        writer,
        SetForegroundColor(crossterm::style::Color::Reset),
        SetBackgroundColor(crossterm::style::Color::Reset),
        SetAttribute(crossterm::style::Attribute::Reset),
    )
}

/// Emit only the attribute changes between `from` and `to`.
///
/// Removals go first: `NormalIntensity` clears BOLD and DIM together, so DIM has
/// to be re-applied afterwards when only BOLD was dropped.
pub(crate) fn queue_modifier_diff<W: Write>(
    w: &mut W,
    from: Modifier,
    to: Modifier,
) -> io::Result<()> {
    use crossterm::style::Attribute as CAttribute;

    let removed = from - to;
    if removed.contains(Modifier::REVERSED) {
        queue!(w, SetAttribute(CAttribute::NoReverse))?;
    }
    if removed.contains(Modifier::BOLD) {
        queue!(w, SetAttribute(CAttribute::NormalIntensity))?;
        if to.contains(Modifier::DIM) {
            queue!(w, SetAttribute(CAttribute::Dim))?;
        }
    }
    if removed.contains(Modifier::ITALIC) {
        queue!(w, SetAttribute(CAttribute::NoItalic))?;
    }
    if removed.contains(Modifier::UNDERLINED) {
        queue!(w, SetAttribute(CAttribute::NoUnderline))?;
    }
    if removed.contains(Modifier::DIM) {
        queue!(w, SetAttribute(CAttribute::NormalIntensity))?;
    }
    if removed.contains(Modifier::CROSSED_OUT) {
        queue!(w, SetAttribute(CAttribute::NotCrossedOut))?;
    }
    if removed.contains(Modifier::SLOW_BLINK) || removed.contains(Modifier::RAPID_BLINK) {
        queue!(w, SetAttribute(CAttribute::NoBlink))?;
    }

    let added = to - from;
    if added.contains(Modifier::REVERSED) {
        queue!(w, SetAttribute(CAttribute::Reverse))?;
    }
    if added.contains(Modifier::BOLD) {
        queue!(w, SetAttribute(CAttribute::Bold))?;
    }
    if added.contains(Modifier::ITALIC) {
        queue!(w, SetAttribute(CAttribute::Italic))?;
    }
    if added.contains(Modifier::UNDERLINED) {
        queue!(w, SetAttribute(CAttribute::Underlined))?;
    }
    if added.contains(Modifier::DIM) {
        queue!(w, SetAttribute(CAttribute::Dim))?;
    }
    if added.contains(Modifier::CROSSED_OUT) {
        queue!(w, SetAttribute(CAttribute::CrossedOut))?;
    }
    if added.contains(Modifier::SLOW_BLINK) {
        queue!(w, SetAttribute(CAttribute::SlowBlink))?;
    }
    if added.contains(Modifier::RAPID_BLINK) {
        queue!(w, SetAttribute(CAttribute::RapidBlink))?;
    }

    Ok(())
}
