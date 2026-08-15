# Third-party licenses

## claw-code-rust

Source: <https://github.com/7df-lab/claw-code-rust> — MIT, © 2026 wangtsiao

Elal's session and conversation-record layers carry code ported from that project. The files below
contain adapted implementations, not just a shared idea:

- `crates/elal_core/src/session/state.rs` — `SessionState` and the compaction policy
- `crates/elal_core/src/session/rollout.rs` — the `RolloutStore` append-only journal
- `crates/elal_protocol/src/session.rs` — conversation record shapes (`TurnItem` subset)
- `crates/elal_protocol/src/approval.rs` — approval mode shape
- `crates/elal_tui/src/history.rs` — scrollback insertion and the `DECSTBM` scroll-region dance
- `crates/elal_tui/src/terminal/` — the inline viewport terminal (see the ratatui notice below)

Each of those files names its origin in its module documentation.

Other files reference `claw-code-rust` only as an architectural reference — where a responsibility
sits, how a policy is shaped — with no code taken. Ideas are not licensed; those need no notice and
are not listed here.

### License text

```
MIT License

Copyright (c) 2026 wangtsiao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ratatui

Source: <https://github.com/ratatui/ratatui> — MIT, © 2016-2022 Florian Dehau, © 2023-2025 The
Ratatui Developers

`crates/elal_tui/src/terminal/` derives from `ratatui::Terminal` and its buffer-diffing internals.
The fork exists because the upstream terminal owns the whole screen: it has no way to be told that
rows appeared *above* its viewport, which is the entire premise of rendering inline. Ratatui is also
a regular dependency of that crate — the fork replaces one type, not the library.

### License text

```
The MIT License (MIT)

Copyright (c) 2016-2022 Florian Dehau
Copyright (c) 2023-2025 The Ratatui Developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
