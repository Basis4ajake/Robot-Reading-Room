# Robot Reading Room — GUI

The desktop control center for Local Knowledge Library: a Tauri (Rust + OS webview) shell around a plain HTML/JS/CSS frontend — no framework, no build step. The frontend in `src/` talks to the Python FastAPI backend over HTTP; Tauri just gives it a native window, file-picker dialogs, and the ability to open external links (e.g. the Ollama model catalog) in the system browser instead of the app's own webview.

## Running it

1. Start the backend first, from the repo root (see the root `README.md` / `docs/how_to_use.md` for setup):

   ```bash
   python -m local_knowledge_library.api
   ```

2. In this directory, run the Tauri dev shell:

   ```bash
   npm run tauri dev
   ```

   This opens a native window loading `src/index.html` directly (no bundler — `tauri.conf.json` points `frontendDist` straight at `src/`, so editing any file there and reopening the window picks it up immediately).

To build a distributable binary: `npm run tauri build`.

## Layout

- `src/` — the actual frontend (HTML/CSS/JS), served as-is with no build step.
- `src-tauri/` — the Rust shell: window config (`tauri.conf.json`), plugin registration (`src/lib.rs` — currently `opener` for external links and `dialog` for the native file picker used when adding sources).

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
