# pylonrack-llama

PylonRack slot application for **llama.cpp** — manages a `llama-server` process and exposes it to the rack with model selection, start/stop, update detection, metrics, and Open WebUI integration.

---

## What it does

- **Model dropdown** — scans your HuggingFace cache and lists all `.gguf` files
- **Start / Stop** — launches and stops `llama-server` with your configured parameters
- **Update button** — checks the llama.cpp git repo for new commits, rebuilds on demand; badge appears when an update is available
- **Metrics** — RAM usage and active requests shown in the rack heartbeat
- **Log** — llama-server stdout piped to a rotating log file, accessible via the rack log panel
- **Open WebUI** — displayed in the rack body panel (runs separately, linked via `ui_url`)

---

## Requirements

- Python 3.11+
- A working [llama.cpp](https://github.com/ggerganov/llama.cpp) build (`llama-server` binary)
- HuggingFace cache with `.gguf` model files
- Open WebUI running separately (optional, for the body panel)
- [PylonRack](https://github.com/marianvid/pylonrack) installed

---

## Installation

### 1. Clone

```
git clone https://github.com/marianvid/pylonrack-llama
cd pylonrack-llama
```

### 2. Install dependencies

Using conda (recommended — same env as PylonRack tools):

```
conda activate pylonrack
pip install -r requirements.txt
```

Or with any Python 3.11+ environment:

```
pip install -r requirements.txt
```

Dependencies: `websockets`, `psutil`, `requests`

### 3. Configure

Copy and edit `settings.json`:

```json
{
  "llama_bin":     "/path/to/llama.cpp/build/bin/llama-server",
  "llama_repo":    "/path/to/llama.cpp",
  "hf_cache":      "/path/to/HuggingFace/hub",
  "log_file":      "~/.pylonrack/llama-server.log",
  "openwebui_url": "http://localhost:8080",
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

| Key | Description |
|-----|-------------|
| `llama_bin` | Absolute path to compiled `llama-server` binary |
| `llama_repo` | Absolute path to the llama.cpp git repo (for update detection and rebuild) |
| `hf_cache` | Absolute path to your HuggingFace hub cache directory |
| `log_file` | Where llama-server stdout is written (rotating, 1MB × 10 files) |
| `openwebui_url` | URL of your Open WebUI instance — displayed in the rack body panel |
| `server.port` | Port llama-server listens on |
| `server.ctx_size` | Context size in tokens |
| `server.n_gpu_layers` | GPU layers offloaded to Metal |
| `server.parallel` | Number of parallel slots |
| `server.threads` | CPU threads |
| `server.batch_size` / `ubatch_size` | Batch sizes |

`settings.json` is gitignored — your paths stay local.

### 4. Update `rack.json` start command

Edit `rack.json` to use your Python interpreter:

```json
{
  "start": "/path/to/your/python3 server.py"
}
```

For conda:
```json
{
  "start": "conda run -n pylonrack python3 server.py"
}
```

Or create a `start.sh`:
```bash
#!/bin/bash
source ~/.bashrc
conda activate pylonrack
python3 server.py
```

```json
{
  "start": "bash start.sh"
}
```

---

## Adding to PylonRack

1. Open PylonRack (menu bar icon)
2. Click `+` in the slot list
3. Click **Browse…** and select this folder (`pylonrack-llama/`)
4. Click **Add**
5. Press **▶** to activate the slot

The slot starts `python3 server.py`, which starts the WebSocket server. Select a model from the dropdown and press **Start** to launch llama-server.

---

## Controls

| Control | Type | Description |
|---------|------|-------------|
| Model | Dropdown | Select from all `.gguf` files in HF cache |
| Start / Stop | Button | Toggle llama-server |
| Update | Button | `git pull` + cmake rebuild; badge = update available |
| Status | Label | Current state: Idle / Running / Updating… |

---

## Update mechanism

- Every 30 minutes, the slot runs `git fetch` against the llama.cpp repo
- If new commits are available, the **Update** button gets an orange badge
- Pressing **Update**: stops llama-server (if running), runs `git pull` + `cmake --build`, then optionally restarts
- Rebuild output streams to the rack log panel in real time
- Requires `cmake` on `PATH`

---

## File structure

```
pylonrack-llama/
├── rack.json           ← PylonRack slot manifest
├── settings.json       ← local configuration (gitignored)
├── server.py           ← WebSocket server (PylonRack protocol)
├── config.py           ← configuration loader with defaults
├── llama_server.py     ← llama-server process lifecycle
├── model_scanner.py    ← HF cache scanner
├── git_updater.py      ← git fetch/pull + cmake rebuild
└── requirements.txt
```

---

## License

MIT — use freely, no warranty.
