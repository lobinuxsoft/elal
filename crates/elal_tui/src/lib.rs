//! `elal_tui` — terminal UI for elal.
//!
//! The UI renders **inline**: the conversation is written into the terminal's own
//! scrollback with [`history::insert_lines`], and only the live viewport (composer,
//! status, in-flight output) is drawn by ratatui. Nothing enters the alternate
//! screen, so selection, copy and scrollback stay native and the transcript
//! survives after the process exits.
//!
//! That trade forces a custom [`terminal::Terminal`]: ratatui's own terminal owns
//! the whole screen and cannot be told that rows appeared above its viewport.
//!
//! See <https://github.com/lobinuxsoft/elal/issues/7>.

pub mod app;
pub mod composer;
pub mod history;
pub mod terminal;

#[cfg(test)]
mod composer_tests;
#[cfg(test)]
mod history_tests;
#[cfg(test)]
mod vt100_backend;
