# AGENTS.md — pylonrack-llama

> Machine-readable implementation reference. Not human documentation.
> Audience: AI agents continuing implementation.
> Audit: append knowledge after each implementation session.

---

## SYSTEM_IDENTITY

```
name: pylonrack-llama
type: PylonRack slot application
language: Python 3.11+
repo: github.com/marianvid/pylonrack-llama
local_path: /Volumes/Marian_Backup/work/pylonrack-slots/llama/
rack_protocol: PylonRack WebSocket protocol (see pylonrack/AGENTS.md)
ws_port: 8765 (from rack.json, read at runtime)
venv: .venv/ (auto-created by start.sh on first run)
deps: websockets>=12.0, psutil>=5.9, requests>=2.31, huggingface_hub>=0.23
```

---

## MODULE_MAP

```
server.py          — entry point, WebSocket handler, AppState, message builders
config.py          — AppConfig + ServerConfig dataclasses, settings.json loader
llama_server.py    — LlamaServer: process lifecycle, metrics, log rotation, lsof-based PID detection
model_scanner.py   — scan HF Cache for .gguf files, build GGUFModel list
git_updater.py     — GitUpdater: git fetch/pull + cmake rebuild, progress callbacks
settings.json      — user config (gitignored)
rack.json          — PylonRack slot manifest
start.sh           — venv bootstrap + exec python3 server.py
tests/             — pytest test suite
```

---

## CONFIG_SCHEMA

### settings.json
```json
{
  "llama_bin":     "/path/to/llama.cpp/build/bin/llama-server",
  "llama_repo":    "/path/to/llama.cpp",
  "hf_cache":      "/path/to/HuggingFace/hub",
  "log_file":      "~/.pylonrack/llama-server.log",
  "openwebui_url": "http://localhost:1234",
  "server": {
    "host":         "0.0.0.0",
    "port":         1234,
    "ctx_size":     131072,
    "n_gpu_layers": 99,
    "parallel":     2,
    "threads":      8,
    "batch_size":   512,
    "ubatch_size":  256
  }
}
```

---

## APPSTATE

```python
class AppState:
    cfg:               AppConfig
    llama:             LlamaServer
    updater:           GitUpdater
    models:            list[GGUFModel]
    selected_model:    GGUFModel | None
    update_in_progress: bool
    update_available:  bool
    binary_stale:      bool        # binary version < git commit count by >10
    log_subscribers:   set         # WebSocket connections subscribed to live log
    llama_version:     str         # "bNNNN" populated async after startup
```

---

## PROTOCOL_IMPLEMENTATION

### Manifest sent to rack
```python
{
  "type": "manifest",
  "name": "llama.cpp",
  "version": "1.0",
  "heartbeat_interval": 5,
  "controls": [
    {"id": "model_select",  "type": "dropdown", "label": "Model"},
    {"id": "toggle",        "type": "button",   "label": "Start",  "style": "primary"},
    {"id": "status_label",  "type": "label",    "value": "Idle",   "style": "default"},
    _update_control(state),   # see UPDATE_BUTTON_STATES below
  ],
  "ui_url": state.cfg.openwebui_url
}
```

### UPDATE_BUTTON_STATES (3 states)
```python
# binary_stale=True → red, badge=True
{"id": "update", "label": "bNNNN", "style": "error",     "badge": True}
# update_available=True → orange, badge=True
{"id": "update", "label": "bNNNN", "style": "warning",   "badge": True}
# default → grey, no badge
{"id": "update", "label": "bNNNN", "style": "secondary", "badge": False}
```
Tooltip in rack (Swift):
- error+badge   → "Binary is outdated — sources newer than binary. Click to rebuild."
- warning+badge → "Update available — click to pull latest sources & rebuild"
- default       → "llama.cpp is up to date"

### controls_update emitted by _controls_update(state)
```python
controls: [
  {"id": "model_select",  "value": selected_model.display_name or ""},
  {"id": "toggle",        "label": "Stop"|"Start", "style": "destructive"|"primary"},
  {"id": "status_label",  "value": _status_text(state), "style": _status_style(state)},
  {"id": "update",        "label": ..., "style": ..., "badge": ...},  # full _update_control(state)
]
```

### pong
```python
{"type": "pong", "status": "running"|"warning", "message": "X GB RAM · N req" | status_text}
# "running" only when llama.is_running — rack shows WebView on "running", placeholder on "warning"
```

### action: update (CRITICAL SEQUENCE)
```python
1. if update_in_progress: return
2. was_running = llama.is_running
3. if was_running: llama.stop()
4. update_in_progress = True; binary_stale = False; update_available = False
5. send {"type": "show_log"}          ← rack auto-switches to log panel
6. log_subscribers.add(ws)            ← subscribe ws to live log
7. broadcast_update()                 ← shows "Updating…" in controls
8. run updater.update(on_line) in executor  ← each line → _push_log_line(ws, line) live
9. update_in_progress = False
10. if ok: llama_version = _get_llama_version(cfg); push "✓ Build complete — bNNNN"
11. broadcast_update()
12. if was_running and ok: llama.start(path); broadcast_update(); send reload_ui
```

### show_log (server → rack)
```python
{"type": "show_log"}
# rack sets bodyMode = .log and calls requestLog()
# used by update handler to auto-open log panel during build
```

### log streaming (total=-1 = append)
```python
{"type": "log_response", "lines": [line], "total": -1}
# rack appends to logLines (does NOT replace)
# used for: llama stdout piping, update build output
```

---

## GIT_UPDATER (CRITICAL)

### cmake PATH issue
```
PROBLEM: start.sh launches server.py without sourcing .zshrc → cmake not on PATH
FIX: use shutil.which("cmake") with fallback to absolute path
```
```python
import shutil
CMAKE = shutil.which("cmake") or "/opt/homebrew/bin/cmake"
```

### Build commands (per llama.cpp official docs)
```
# From: llama.cpp/docs/build.md — Metal Build section
# Metal is enabled BY DEFAULT on macOS — do NOT add -DGGML_METAL=ON
# Standard build:
cmake -B build
cmake --build build --config Release -j
```
WRONG: `cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release`
WRONG: `cmake --build build --target llama-server --clean-first`
CORRECT:
```python
steps = [
    (["git", "pull", "--ff-only"],                          "Pulling latest commits…"),
    ([CMAKE, "-B", "build"],                                "Configuring CMake…"),
    ([CMAKE, "--build", "build", "--config", "Release", "-j"], "Building…"),
]
```

### is_binary_stale() — version comparison method
```python
# Compare binary version number vs git commit count
# NOT mtime comparison (git resets timestamps on checkout)
binary_version = parse from `llama-server --version` stderr: "version: NNNN"
git_version    = `git rev-list --count HEAD`
stale = binary_version - git_version < -10   # binary is >10 commits behind git
# llama-server --version takes 10-15s (Metal init) — always run in executor
```

### has_update() — check remote
```python
# git fetch --quiet; git rev-list HEAD..origin/HEAD --count
# True if count > 0
```

### Background check sequence
```python
_check_updates_periodically(state, handler):
    while True:
        version      = await executor(_get_llama_version)   # slow: 10-15s
        has_update   = await executor(updater.has_update)
        binary_stale = await executor(updater.is_binary_stale)
        
        changed = has_update != state.update_available or binary_stale != state.binary_stale
        state.llama_version    = version
        state.update_available = has_update
        state.binary_stale     = binary_stale
        
        if changed:
            # Send ONLY update button — prevents full controls re-render flicker
            for ws in handler.clients:
                ws.send(_update_control(state) wrapped in controls_update)
        
        await sleep(1800)
```

---

## LLAMA_SERVER

### PID detection (macOS — no root required)
```python
# CRITICAL: psutil.net_connections() fails without root on macOS → always use lsof
result = subprocess.run(["lsof", "-iTCP:<port>", "-sTCP:LISTEN", "-t"], ...)
for pid in result.stdout.splitlines():
    if "llama" in psutil.Process(pid).name().lower(): return pid
```

### is_running
```python
# Catches both: processes started by this instance AND externally started
if self._process and self._process.poll() is None: return True
return self._find_pid() is not None
```

### stop() — waits for death (not fire-and-forget)
```python
proc.terminate(); proc.wait(timeout=5)  # wait — not fire-and-forget
# Also terminate external pid via _find_pid()
psutil.Process(ext_pid).terminate(); p.wait(timeout=5)  # wait here too
```

### Live log streaming
```python
# llama stdout → _pipe_log thread → on_log_line callback → 
# asyncio.run_coroutine_threadsafe → _push_log_line(ws, line) →
# ws.send({"type": "log_response", "lines": [line], "total": -1})
```

---

## MODEL_MANAGER (server.py handlers)

```
list_local_models → refresh_models(); return [{display_name, full_path, size_gb}]
hf_search         → huggingface_hub.list_models(search=q, filter="gguf", sort="downloads", limit=50)
                    WRONG: direction=-1 (not supported)
                    Empty query → returns top downloaded models
hf_model_files    → list_repo_files + get_paths_info; skip mmproj
hf_download       → hf_hub_download(repo_id, filename, cache_dir)
delete_model      → os.remove(path); refresh_models()
```
All return via `action_result` with `data: {type: "...", ...}` — rack routes via `actionResultToken`.

---

## START_SH

```bash
#!/bin/zsh
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt -q
fi
exec .venv/bin/python3 server.py
```
- `exec` replaces shell — env vars propagate to python3
- Does NOT source .zshrc → cmake, git must be found via absolute path or shutil.which
- startup_delay in rack.json: 5s (server.py init is ~0.1s, but venv creation = 15-30s first run)

---

## KNOWN_ISSUES

```
[RESOLVED] psutil.net_connections() → lsof
[RESOLVED] model_select did not update dropdown → controls_update with value
[RESOLVED] WebView stale model → reload_ui
[RESOLVED] stop() fire-and-forget → wait for death
[RESOLVED] cmake not on PATH → shutil.which + absolute fallback
[RESOLVED] cmake wrong flags → follow llama.cpp docs (no -DGGML_METAL, no --target)
[RESOLVED] update check after 30min delay → runs at startup
[RESOLVED] flicker on update badge → send only update button, not full controls_update
[RESOLVED] binary_stale via mtime → version number comparison instead
[ACTIVE] llama.start() blocks up to 120s — no progress in rack during startup
[ACTIVE] update build takes minutes — no ETA shown
[ACTIVE] git pull --ff-only fails on merge commits in upstream
[ACTIVE] No validation llama_bin exists before start
[ACTIVE] hf_download progress: single 0% → 100% (no intermediate progress from hf_hub_download)
```

---

## AUDIT_LOG

```
2026-05-27 — Initial AGENTS.md

2026-05-28 — Major audit after debugging session
  Added: cmake PATH fix (shutil.which), correct build commands per llama.cpp docs,
  binary_stale 3-state update button, show_log protocol message,
  log streaming (total=-1), background check sequence, model manager handlers,
  anti-flicker controls_update (update button only), stop() wait semantics.
  Test count: 37/37 passing (test_llama_server, test_model_scanner, test_model_manager,
  test_server_state, test_webview_lifecycle, test_git_updater).
  Confirmed working: full update sequence (git pull + cmake build, 395 log lines live).
  Root cause of cmake failure: PATH not set when launched via start.sh without .zshrc.
```
