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
ws_port: 8765 (from rack.json, read at runtime via PYLON_PORT)
venv: .venv/ (auto-created by start.sh on first run)
deps: websockets>=12.0, psutil>=5.9, requests>=2.31, huggingface_hub>=0.23, gguf>=0.9
```

---

## MODULE_MAP

```
server.py          — entry point, WebSocket handler, AppState, SlotHandler, message builders
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

### settings.json (full schema including all additions)
```json
{
  "llama_bin":      "/path/to/llama.cpp/build/bin/llama-server",
  "llama_repo":     "/path/to/llama.cpp",
  "hf_cache":       "/path/to/HuggingFace/hub",
  "log_file":       "~/.pylonrack/llama-server.log",
  "openwebui_url":  "http://localhost:1234",
  "selected_model": "/full/path/to/model.gguf",
  "draft_model":    "/full/path/to/draft.gguf",
  "draft_map": {
    "/full/path/model_a.gguf": "/full/path/draft_a.gguf"
  },
  "settings_map": {
    "/full/path/model_a.gguf": {
      "ctx_size": 131072, "n_gpu_layers": 99, "threads": 8,
      "batch_size": 512, "ubatch_size": 256,
      "temperature": 0.8, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.1,
      "flash_attn": true, "mlock": false
    }
  },
  "server": {
    "host": "0.0.0.0", "port": 1234,
    "ctx_size": 131072, "n_gpu_layers": 99,
    "parallel": 2, "threads": 8,
    "batch_size": 512, "ubatch_size": 256,
    "temperature": 0.8, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.1,
    "flash_attn": true, "mlock": false
  }
}
```

### Persistence rules
```
selected_model  → saved on every model switch
draft_model     → saved on save_settings (redundant with draft_map, kept for compat)
draft_map       → updated on save_settings per current model; loaded into AppState._draft_map
settings_map    → updated on save_settings per current model; loaded into AppState._settings_map
server block    → updated on save_settings (global defaults, used for new models)
```

---

## APPSTATE

```python
class AppState:
    cfg:                AppConfig
    llama:              LlamaServer
    updater:            GitUpdater
    models:             list[GGUFModel]
    selected_model:     GGUFModel | None   # restored from settings.json on init
    update_in_progress: bool
    update_available:   bool
    binary_stale:       bool               # binary version < git commit count by >10
    log_subscribers:    set                # WebSocket connections subscribed to live log
    llama_version:      str                # "bNNNN" populated async after startup
    draft_model:        str | None         # full_path of draft model (None = disabled)
    _saved_model_path:  str | None         # from settings.json, used in refresh_models()
    _draft_map:         dict               # {model_path: draft_path}
    _settings_map:      dict               # {model_path: {ctx_size, temp, ...}}
```

### refresh_models() behaviour
```python
# Scans HF cache for .gguf files
# If selected_model is None: restores from _saved_model_path if available, else models[0]
# IMPORTANT: incomplete downloads filtered out by model_scanner
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
    _update_control(state),
  ],
  "ui_url": state.cfg.openwebui_url
}
```

### pong
```python
{"type": "pong", "status": "running"|"warning", "message": "X GB RAM · N req" | status_text}
# "running" only when llama.is_running
```

### action handlers (control_id routing)
```
toggle           → start/stop llama server
model_select     → switch selected model, persist, push settings to client
update           → git pull + cmake rebuild (show_log → live log → restart if was_running)
list_local_models→ refresh_models() + return models list
hf_search        → asyncio.create_task(_handle_hf_search) — non-blocking
hf_model_files   → asyncio.create_task(_handle_hf_model_files) — non-blocking
hf_download      → asyncio.create_task(_handle_hf_download) — non-blocking, streaming progress
delete_model     → os.remove + refresh_models
get_settings     → return per-model settings (from settings_map or GGUF defaults)
save_settings    → persist to settings.json (settings_map + draft_map), restart if running
check_draft_compat → asyncio.create_task, reads vocab_size from both GGUFs via gguf package
```

### CRITICAL: all HF + settings handlers use asyncio.create_task
```python
# WRONG (blocks event loop → ping/pong fails → Swift reconnects):
await self._handle_hf_download(ws, value)

# CORRECT (non-blocking → ping/pong continues):
asyncio.create_task(self._handle_hf_download(ws, value))
```

### log_request handler
```python
# skip=0: last N lines (initial fetch)
# skip=N: N lines before the last N (load more / prepend)
# Uses _tail_log_file(log_path, n, skip)
# prepend=True in response when skip > 0
```

### get_settings response
```python
{
  "type": "action_result", "action": "settings",
  "data": {
    "type": "settings",
    "server": {  # from settings_map[model] or _read_model_defaults(model)
      "ctx_size": ..., "n_gpu_layers": ..., "threads": ...,
      "batch_size": ..., "ubatch_size": ...,
      "temperature": ..., "top_p": ..., "top_k": ..., "repeat_penalty": ...,
      "flash_attn": ..., "mlock": ...
    },
    "draft_model": draft_map.get(model_path),  # None if no draft
    "hf_cache": str(cfg.hf_cache_path),
  }
}
```

### save_settings sequence
```python
1. Parse settings dict from msg["settings"]
2. Update raw settings.json server block
3. Save draft_model to state + draft_map[current_model]
4. Save settings to settings_map[current_model]
5. Write settings.json
6. Reload config: cfg = load(); llama._cfg = cfg
7. Send settings_saved response
8. If model selected: restart server (stop if running, start with new settings + draft)
9. On start failure: send show_log
```

### Model switch sequence (model_select action)
```python
1. Find matching GGUFModel by display_name
2. state.selected_model = match
3. state.draft_model = _draft_map.get(match.full_path)
4. Persist selected_model + draft_model to settings.json
5. asyncio.create_task(_handle_get_settings(ws))  ← push settings to client immediately
6. Stop running server if any
7. Start server with new model
8. broadcast_update()
9. reload_ui if started ok, else show_log
```

### update action sequence
```python
1. if update_in_progress: return
2. was_running = llama.is_running
3. if was_running: llama.stop()
4. update_in_progress = True; binary_stale = False; update_available = False
5. send show_log
6. log_subscribers.add(ws)
7. broadcast_update()  ← "Updating…"
8. run updater.update(on_line) in executor — each line pushed via log_response total=-1
9. update_in_progress = False
10. if ok: llama_version = _get_llama_version(); push "✓ Build complete — bNNNN"
11. broadcast_update()
12. if was_running and ok: llama.start(path, draft_model); broadcast_update(); reload_ui
    else if start failed: show_log
```

---

## GIT_UPDATER

### cmake PATH issue
```python
import shutil
CMAKE = shutil.which("cmake") or "/opt/homebrew/bin/cmake"
```

### Build commands
```python
steps = [
    (["git", "pull", "--ff-only"],                             "Pulling latest commits…"),
    ([CMAKE, "-B", "build"],                                   "Configuring CMake…"),
    ([CMAKE, "--build", "build", "--config", "Release", "-j"], "Building…"),
]
# Metal enabled by default on macOS — do NOT add -DGGML_METAL=ON
```

### Binary stale detection
```python
binary_version = parse from `llama-server --version` stderr: "version: NNNN"
git_version    = `git rev-list --count HEAD`
stale = binary_version - git_version < -10
# llama-server --version takes 10-15s (Metal init) — always run in executor
```

---

## LLAMA_SERVER

### _build_command
```python
cmd = [bin, "--host", host, "--port", port, "-m", model_path,
       "--ctx-size", ctx_size, "--n-gpu-layers", n_gpu_layers,
       "--parallel", parallel, "--threads", threads,
       "--batch-size", batch_size, "--ubatch-size", ubatch_size,
       "--temp", temperature, "--top-p", top_p,
       "--top-k", top_k, "--repeat-penalty", repeat_penalty,
       "--reasoning", "off", "--metrics"]
if flash_attn: cmd += ["--flash-attn", "on"]
if mlock:      cmd += ["--mlock"]
if draft_model_path: cmd += ["-md", draft_model_path]
```

### start(model_path, draft_model_path=None)
```python
# draft_model_path passed from AppState.draft_model at call site
# All llama.start() calls: lambda: self._state.llama.start(path, self._state.draft_model)
```

### PID detection (macOS — no root required)
```python
result = subprocess.run(["lsof", "-iTCP:<port>", "-sTCP:LISTEN", "-t"], ...)
```

---

## FILE WATCHER (_watch_log_file)

```python
# Runs as asyncio.create_task at startup
# Starts from end of log file
# Polls every 200ms for new bytes
# Handles file rotation/truncation (pos reset on shrink)
# Pushes new lines to all log_subscribers via log_response total=-1
# Works regardless of server running state
```

### _tail_log_file(log_path, n, skip=0)
```python
# skip=0: last n lines
# skip=N: lines[-(n+skip):-skip] — n lines before the last skip lines
```

---

## MODEL_SCANNER

```python
# Scans HF Cache rglob("*.gguf")
# Skips: mmproj files, files with .incomplete marker, files in tmp directories
# GGUFModel(display_name, full_path, size_gb)
# display_name: "org/repo / filename" format
```

### Incomplete download detection
```python
# Skip if .gguf.incomplete exists in same directory
# Skip if any parent dir named "tmp", "temp", or "blobs.tmp"
```

---

## DRAFT MODEL / SPECULATIVE DECODING

```
- draft_model stored per-model in draft_map (persisted in settings.json)
- On model switch: draft_model auto-restored from draft_map
- On save_settings: draft_map updated for current model
- check_draft_compat: reads vocab_size via gguf.GGUFReader for both models
  Returns: {compatible: bool, main_vocab: int, draft_vocab: int}
  Runs in executor (non-blocking, ~3-5s for 2 files)
- llama-server flag: -md (not --draft-model — removed in recent versions)
```

---

## KNOWN_ISSUES

```
[RESOLVED] psutil.net_connections() → lsof
[RESOLVED] model_select dropdown update
[RESOLVED] WebView stale model → reload_ui
[RESOLVED] stop() fire-and-forget → wait
[RESOLVED] cmake not on PATH → shutil.which
[RESOLVED] cmake wrong flags
[RESOLVED] update check delay → runs at startup
[RESOLVED] flicker on update badge
[RESOLVED] binary_stale via mtime → version comparison
[RESOLVED] hf_download blocks event loop → asyncio.create_task
[RESOLVED] hf_download no progress → requests streaming 1MB chunks
[RESOLVED] incomplete downloads visible in Local Models → scanner filter
[RESOLVED] log unavailable when server stopped → file watcher from disk
[RESOLVED] download_complete not updating dropdown → control_data pushed after complete
[RESOLVED] selected_model lost on restart → persisted in settings.json
[RESOLVED] draft_model lost on model switch → draft_map per-model
[RESOLVED] per-model settings lost → settings_map
[RESOLVED] settings panel stale on model switch → get_settings pushed automatically
[RESOLVED] llama start failure silent → show_log on start failure
[RESOLVED] Q4_0_8_8 format removed in new llama.cpp → use Q4_K_M
[ACTIVE] llama.start() blocks up to 120s — no progress in rack during startup
[ACTIVE] update build takes minutes — no ETA shown
[ACTIVE] git pull --ff-only fails on merge commits in upstream
[ACTIVE] No validation llama_bin exists before start
```

---

## AUDIT_LOG

```
2026-05-27 — Initial AGENTS.md

2026-05-28 — Major audit after debugging session
  Added: cmake PATH fix, correct build commands, binary_stale 3-state,
  show_log protocol, log streaming (total=-1), background check sequence,
  model manager handlers, anti-flicker controls_update, stop() wait semantics.

2026-05-29 — Major feature session audit
  FIXED / ADDED:
  - Download: streaming progress via requests, asyncio.create_task for all HF ops
  - Log: file watcher (_watch_log_file) reads from disk, works when server stopped
  - Log: log_request skip param for "load earlier lines" (prepend=true response)
  - Model persistence: selected_model saved/restored via settings.json
  - Draft model: per-model association via draft_map, auto-restored on model switch
  - Per-model settings: settings_map, ctx_size from GGUF metadata on new model
  - Settings push: get_settings pushed automatically on model switch
  - save_settings: persists settings_map + draft_map, restarts server
  - check_draft_compat: vocab_size comparison via gguf package
  - Speculative decoding: -md flag (not --draft-model)
  - Start failure: auto show_log
  - Incomplete download filter: model_scanner skips .incomplete + tmp dirs
  - after download_complete: control_data(model_select items) pushed

  NEW DEPS: gguf>=0.9

  settings.json NEW FIELDS: selected_model, draft_model, draft_map, settings_map,
    server.temperature, server.top_p, server.top_k, server.repeat_penalty,
    server.flash_attn, server.mlock
```
