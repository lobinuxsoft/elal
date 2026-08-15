//! Inline terminal: a ratatui viewport that shares the screen with the shell.

mod flush;
mod frame;
mod inline;

pub use frame::Frame;
pub use inline::Terminal;

pub(crate) use flush::queue_modifier_diff;
