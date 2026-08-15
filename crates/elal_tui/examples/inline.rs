//! Try the inline composer on a real terminal:
//!
//! ```sh
//! cargo run -p elal_tui --example inline
//! ```
//!
//! Type, press Enter, and the message lands in the scrollback. Ctrl+C exits —
//! and the messages should still be there afterwards, selectable and scrollable
//! like any other shell output. That is the whole point of the design.

use elal_tui::app::App;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let submitted = {
        let mut app = App::new("write something, Enter to send, Ctrl+C to quit")?;
        app.run().await?
    };

    println!("{} message(s) submitted", submitted.len());
    Ok(())
}
