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
deps: websockets>=12.0, psutil>=5.9, requests>=2.31
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
test.html          — static test page for WebView testing (not production)
```

---

## CONFIG_SCHEMA

### settings.json
```json
{
  "llama_bin":     "/path/to/llama-server",
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

### AppConfig fields
```
llama_bin:    Path → bin_path property (expanduser)
llama_repo:   Path → repo_path property
hf_cache:     Path → hf_cache_path property
log_file:     Path → log_path property
openwebui_url: str → used as ui_url in manifest
server:       ServerConfig dataclass
```

### ServerConfig defaults
```
host=0.0.0.0, port=1234, ctx_size=131072, n_gpu_layers=99
parallel=2, threads=8, batch_size=512, ubatch_size=256
```

---

## APPSTATE

```python
class AppState:
    cfg: AppConfig
    llama: LlamaServer
    updater: GitUpdater
    models: list[GGUFModel]           # populated by refresh_models()
    selected_model: GGUFModel | None  # first model by default
    update_in_progress: bool
    update_available: bool            # set by background check every 1800s

    def refresh_models():             # scan cfg.hf_cache_path
    def selected_path() -> str | None # shortcut to selected_model.full_path
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
    {"id": "update",        "type": "button",   "label": "Update", "style": "secondary", "badge": False},
    {"id": "status_label",  "type": "label",    "value": "Idle",   "style": "default"},
  ],
  "ui_url": cfg.openwebui_url   # loaded from settings.json
}
```

### controls_update emitted by _controls_update(state)
```python
controls: [
  {"id": "model_select",  "value": selected_model.display_name or ""},
  {"id": "toggle",        "label": "Stop"|"Start", "style": "destructive"|"primary"},
  {"id": "status_label",  "value": _status_text(state), "style": _status_style(state)},
  {"id": "update",        "badge": state.update_available},
]
```
Status text/style logic:
```
update_in_progress → "Updating…" / "warning"
llama.is_running   → "Running"   / "success"
default            → "Idle"      / "default"
```

### pong emitted on ping
```python
{
  "type": "pong",
  "status": "running"|"warning",
  "message": f"{ram_used} GB RAM · {reqs} req" if running else status_text
}
```

### control_data handler (model_select only)
```python
# rack sends: {"type": "control_data", "control_id": "model_select"}
# response: {"type": "control_data", "control_id": "model_select", "items": [display_names]}
```

### action handlers

**model_select:**
```python
1. find GGUFModel by display_name
2. state.selected_model = match
3. was_running = llama.is_running
4. send controls_update with model_select.value = display_name (immediate feedback)
5. if was_running:
   a. llama.stop()
   b. send controls_update: toggle="Starting…"/secondary, status_label=f"Loading {name}…"/warning
   c. ok = await executor(llama.start, match.full_path)
   d. send _controls_update(state)
   e. if ok: send {"type": "reload_ui"}  ← triggers WKWebView recreation in rack
6. else: send _controls_update(state)
```

**toggle:**
```python
if running: llama.stop(); broadcast_update
else:
  path = selected_path() or refresh + selected_path()
  if no path: send error label, return
  broadcast_update immediately (shows "Starting…" state via controls_update)
  ok = await executor(llama.start, path)
  if ok: log "started on port X"
  broadcast_update
```

**update:**
```python
if update_in_progress: return
was_running = llama.is_running
if was_running: llama.stop()
state.update_in_progress = True
broadcast_update
lines = []
ok = await executor(updater.update, lambda line: lines.append(line))
state.update_in_progress = False
state.update_available = False
send log_response with lines (rack shows in log panel)
broadcast_update
if was_running and ok: restart with current model
```

---

## LLAMA_SERVER

### PID detection (macOS — no root required)
```python
def _find_pid():
    result = subprocess.run(
        ["lsof", "-iTCP:<port>", "-sTCP:LISTEN", "-t"],
        capture_output=True, text=True, timeout=5
    )
    for line in result.stdout.strip().splitlines():
        pid = int(line.strip())
        if "llama" in psutil.Process(pid).name().lower():
            return pid
    return None
```
CRITICAL: psutil.net_connections() fails without root on macOS. Always use lsof.

### is_running check
```python
@property
def is_running(self):
    if self._process and self._process.poll() is None: return True
    return self._find_pid() is not None
```
Catches both: processes launched by this instance AND externally started instances on same port.

### start() sequence
```python
1. if is_running: return True (idempotent)
2. build command: llama-server + all params + --flash-attn on --reasoning off --metrics
3. Popen(cmd, stdout=PIPE, stderr=STDOUT)
4. start daemon thread: _pipe_log(proc.stdout) → rotating log file (1MB × 10)
5. _wait_ready(): poll GET /v1/models every 2s up to READY_TIMEOUT=120s
6. return True if ready, False if timeout or process died
```

### stop() sequence
```python
1. if self._process and running: proc.terminate(); proc.wait(timeout=5) or proc.kill()
2. pid = _find_pid()  # also terminate externally started instance
3. if pid: psutil.Process(pid).terminate()
4. clear _process, _start_time, _model_path
```

### Log rotation
```python
LOG_MAX_BYTES = 1MB, LOG_BACKUPS = 10
# Rotates in _pipe_log: when file > MAX, rename .log→.log.1→.log.2 etc.
```

### metrics()
```python
GET http://localhost:<port>/metrics  # Prometheus format
extract: "llamacpp:requests_processing <float>"
returns: {"requests_processing": int}
```

### ram()
```python
system: psutil.virtual_memory() → {used_gb, total_gb}
llama: psutil.Process(pid).memory_info().rss → {used_gb}
```

---

## MODEL_SCANNER

```python
@dataclass(frozen=True)
class GGUFModel:
    display_name: str    # "org/repo / stem-without-repo-prefix"
    full_path: str
    size_gb: float

def scan(hf_cache_path: Path) -> list[GGUFModel]:
    # rglob("*.gguf") on hf_cache_path
    # skip mmproj files (projection models, not main models)
    # find ancestor dir starting with "models--"
    # display_name: "org/repo / stem" (strip repo prefix from filename)
    # sorted by path
```

HF Cache structure: `hub/models--org--name/snapshots/<hash>/<file>.gguf`

---

## GIT_UPDATER

```python
class GitUpdater:
    def has_update() -> bool:
        # git fetch --quiet; git rev-list HEAD..origin/HEAD --count
        # returns True if count != "0" and != ""

    def update(progress_callback) -> bool:
        # steps: git pull --ff-only → cmake -B build -DGGML_METAL=ON → cmake --build build --config Release -j
        # streams stdout to progress_callback(line)
        # returns False on any non-zero returncode
```
Requires: cmake on PATH, git on PATH.
Build target: Apple Silicon Metal (DGGML_METAL=ON).

---

## BACKGROUND_TASKS

```python
_check_updates_periodically(state):
    while True:
        await sleep(1800)  # 30 minutes
        if not update_in_progress and repo_path.exists():
            has = await executor(updater.has_update)
            state.update_available = has
# Task created in main() via asyncio.create_task()
```

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
CRITICAL: `exec` replaces shell — PYLON_PORT env propagates correctly to python3.
First run: venv creation + pip install ≈ 15-30s. Use startup_delay in rack.json if needed.
Subsequent runs: instant.

---

## KNOWN_ISSUES

```
[RESOLVED] psutil.net_connections() → lsof based _find_pid()
[RESOLVED] model_select did not update dropdown value in rack → controls_update with value field
[RESOLVED] WebView showed stale model → reload_ui message after restart
[RESOLVED] Orphan llama-server after stop → _find_pid() now finds and terminates external instances
[ACTIVE] start.sh venv output goes to process log in rack (harmless, but visible on first run)
[ACTIVE] llama.start() blocks executor thread for up to 120s — rack shows "Starting…" state but no progress bar
[ACTIVE] update button rebuild takes minutes — no time estimate shown to user
[ACTIVE] git_updater.update() uses --ff-only; merge commits in upstream will fail silently
[ACTIVE] model_scanner does not filter by quantization — shows all quant variants
[ACTIVE] No validation that llama_bin exists before start attempt
[ACTIVE] openwebui_url hardcoded in settings.json — not auto-detected; llama.cpp serves UI on same port as API (default 1234)
```

---

## AUDIT_LOG

```
2026-05-27 — Initial AGENTS.md
  Captured: full protocol impl, AppState, LlamaServer internals, lsof fix,
  model_select restart flow, reload_ui mechanism, start.sh venv pattern.
  Known issues documented above.
```
