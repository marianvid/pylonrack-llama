# pylonrack-llama

PylonRack slot application for **llama.cpp** — manages a `llama-server` process and exposes it to the rack with model selection, start/stop, update, metrics, and Open WebUI integration.

## Setup

```
pip install -r requirements.txt
```

## Configuration

Edit `settings.json`:

| Key | Default | Description |
|-----|---------|-------------|
| `llama_bin` | `/Volumes/Marian_Backup/git/llama.cpp/build/bin/llama-server` | Path to llama-server binary |
| `llama_repo` | — | Path to llama.cpp git repo (for updates) |
| `hf_cache` | `/Volumes/Marian_Backup/HF_Cache/hub` | HuggingFace cache directory |
| `log_file` | `~/.pylonrack/llama-server.log` | Log file path |
| `openwebui_url` | `http://localhost:8080` | Open WebUI URL (shown in rack body) |
| `server.port` | `1234` | llama-server port |
| `server.ctx_size` | `131072` | Context size |
| `server.n_gpu_layers` | `99` | GPU layers |
| `server.parallel` | `2` | Parallel slots |
| `server.threads` | `8` | CPU threads |
| `server.batch_size` | `512` | Batch size |
| `server.ubatch_size` | `256` | Micro-batch size |

## rack.json

The slot listens on port `8765` by default (configurable in `rack.json`).

## Controls

| Control | Type | Description |
|---------|------|-------------|
| Model | dropdown | Select GGUF model from HF Cache |
| Start / Stop | button | Toggle llama-server |
| Update | button | `git pull` + rebuild llama.cpp (badge = update available) |
| Status | label | Current state |

The body panel shows Open WebUI embedded.

## Running standalone

```
python3 server.py
```
