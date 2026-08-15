//! The event loop, and the terminal state it is responsible for restoring.
//!
//! Raw mode is entered on [`App::new`] and left on drop — including on panic, so
//! a crash cannot leave the user with a terminal that does not echo.

use std::io;
use std::io::Stdout;
use std::io::Write;
use std::io::stdout;

use crossterm::event::DisableBracketedPaste;
use crossterm::event::EnableBracketedPaste;
use crossterm::event::Event;
use crossterm::event::EventStream;
use crossterm::event::KeyCode;
use crossterm::event::KeyEvent;
use crossterm::event::KeyEventKind;
use crossterm::event::KeyModifiers;
use crossterm::execute;
use crossterm::terminal::disable_raw_mode;
use crossterm::terminal::enable_raw_mode;
use futures::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::text::Span;

use crate::composer::Composer;
use crate::history;
use crate::history::Mode;
use crate::terminal::Terminal;

/// What the loop should do after handling one event.
enum Flow {
    Continue,
    Exit,
}

pub struct App {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    composer: Composer,
    /// Chosen once at startup: the multiplexer does not change mid-session.
    mode: Mode,
}

impl App {
    pub fn new(placeholder: impl Into<String>) -> io::Result<Self> {
        enable_raw_mode()?;
        let mut out = stdout();
        // Bracketed paste keeps a multi-line paste from arriving as a burst of
        // Enter presses, each submitting a fragment of the message.
        execute!(out, EnableBracketedPaste)?;

        let mut terminal = Terminal::new(CrosstermBackend::new(out))?;
        let composer = Composer::new(placeholder);
        let area = viewport_for(&terminal, &composer)?;
        terminal.set_viewport_area(area);

        Ok(Self {
            terminal,
            composer,
            mode: Mode::detect(),
        })
    }

    /// Read keys until the user exits, committing each submitted message to the
    /// scrollback. Returns every message that was submitted.
    pub async fn run(&mut self) -> io::Result<Vec<String>> {
        let mut submitted = Vec::new();
        let mut events = EventStream::new();

        self.draw()?;
        while let Some(event) = events.next().await {
            match event? {
                Event::Key(key) => match self.on_key(key, &mut submitted) {
                    Flow::Exit => break,
                    Flow::Continue => self.draw()?,
                },
                Event::Paste(text) => {
                    for ch in text.chars() {
                        if ch == '\n' {
                            self.composer.insert_newline();
                        } else {
                            self.composer.insert_char(ch);
                        }
                    }
                    self.draw()?;
                }
                Event::Resize(_, _) => {
                    // The viewport is anchored to the bottom of the screen, so a
                    // resize moves it even when the composer did not change.
                    self.reflow()?;
                    self.terminal.invalidate_viewport();
                    self.draw()?;
                }
                _ => {}
            }
        }

        Ok(submitted)
    }

    fn on_key(&mut self, key: KeyEvent, submitted: &mut Vec<String>) -> Flow {
        // Windows reports press and release; acting on both types every key twice.
        if key.kind != KeyEventKind::Press {
            return Flow::Continue;
        }

        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let alt = key.modifiers.contains(KeyModifiers::ALT);

        match key.code {
            KeyCode::Char('c' | 'd') if ctrl => return Flow::Exit,
            // Shift+Enter and Alt+Enter insert a newline; plain Enter submits.
            // Terminals that cannot distinguish Shift+Enter still have Alt+Enter.
            KeyCode::Enter if shift || alt => self.composer.insert_newline(),
            KeyCode::Enter => {
                if let Some(text) = self.composer.take() {
                    let _ = self.commit(&text);
                    submitted.push(text);
                }
            }
            KeyCode::Char(ch) if !ctrl => self.composer.insert_char(ch),
            KeyCode::Backspace => self.composer.backspace(),
            KeyCode::Delete => self.composer.delete(),
            KeyCode::Left => self.composer.move_left(),
            KeyCode::Right => self.composer.move_right(),
            KeyCode::Up => self.composer.move_up(),
            KeyCode::Down => self.composer.move_down(),
            KeyCode::Home => self.composer.move_home(),
            KeyCode::End => self.composer.move_end(),
            _ => return Flow::Continue,
        }
        Flow::Continue
    }

    /// Commit a submitted message to the scrollback, where it stops being ours.
    fn commit(&mut self, text: &str) -> io::Result<()> {
        let mut lines: Vec<Line> = vec![Line::default()];
        for (i, line) in text.lines().enumerate() {
            let marker = if i == 0 { "❯ " } else { "  " };
            lines.push(Line::from(vec![
                Span::styled(marker, Style::default().cyan()),
                Span::raw(line.to_string()),
            ]));
        }
        history::insert_lines_with_mode(&mut self.terminal, lines, self.mode)?;
        self.reflow()
    }

    /// Resize and reposition the viewport to fit the composer as it is now.
    fn reflow(&mut self) -> io::Result<()> {
        let area = viewport_for(&self.terminal, &self.composer)?;
        if area != self.terminal.viewport_area {
            self.terminal.set_viewport_area(area);
        }
        Ok(())
    }

    fn draw(&mut self) -> io::Result<()> {
        self.reflow()?;
        let composer = &self.composer;
        self.terminal.draw(|frame| {
            let area = frame.area();
            frame.render_widget(composer, area);
            let (x, y) = composer.cursor_offset(area.width);
            frame.set_cursor_position((area.x + x, area.y + y));
        })
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // Leave the transcript in the scrollback and put the shell prompt back
        // where the viewport was.
        let _ = self.terminal.clear_viewport();
        let _ = self.terminal.show_cursor();
        let _ = execute!(self.terminal.backend_mut(), DisableBracketedPaste);
        let _ = self.terminal.backend_mut().flush();
        let _ = disable_raw_mode();
    }
}

/// The rows the composer needs, pinned to the bottom of the screen.
fn viewport_for<B>(terminal: &Terminal<B>, composer: &Composer) -> io::Result<Rect>
where
    B: ratatui::backend::Backend<Error = io::Error> + Write,
{
    let screen = terminal.size()?;
    let height = composer.height(screen.width).min(screen.height);
    Ok(Rect::new(
        0,
        screen.height.saturating_sub(height),
        screen.width,
        height,
    ))
}
