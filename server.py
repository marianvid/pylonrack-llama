"""server.py — PylonRack slot application for llama.cpp.

Protocol: PylonRack WebSocket protocol v1
Controls:
  - model_select  (dropdown) — select GGUF model
  - toggle        (button)   — start / stop llama-server
  - update        (button)   — pull + rebuild llama.cpp
  - status_label  (label)    — current state text
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
from pathlib import Path

import websockets
from websockets import ServerConnection as WebSocketServerProtocol

import config as cfg_module
from llama_server import LlamaServer
from model_scanner import GGUFModel, scan
from git_updater import GitUpdater

import re

def _get_llama_version(cfg) -> str:
    """Extract build number from llama-server --version output."""
    try:
        result = subprocess.run(
            [str(cfg.bin_path), "--version"],
            capture_output=True, text=True, timeout=15
        )
        m = re.search(r"version:\s*(\d+)", result.stderr)
        return f"b{m.group(1)}" if m else "llama"
    except Exception:
        return "llama"


def _tail_log_file(log_path: Path, n: int = 100) -> list[str]:
    """Read last n lines from log file. Returns [] if file doesn't exist."""
    try:
        if not log_path.exists():
            return []
        with open(log_path, "rb") as f:
            # Efficient tail: seek from end
            f.seek(0, 2)
            size = f.tell()
            block = min(size, n * 200)  # estimate ~200 bytes per line
            f.seek(max(0, size - block))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
            return lines[-n:]
    except Exception:
        return []


async def _watch_log_file(state: AppState, handler: "SlotHandler") -> None:
    """Watch log file for new lines and push to all log subscribers."""
    log_path = state.cfg.log_path
    # Wait for file to exist
    while not log_path.exists():
        await asyncio.sleep(1)

    # Start from end of file
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        pos = f.tell()

    while True:
        await asyncio.sleep(0.2)
        try:
            current_size = log_path.stat().st_size
            if current_size < pos:
                # File was rotated/truncated
                pos = 0
            if current_size > pos:
                with open(log_path, "rb") as f:
                    f.seek(pos)
                    new_data = f.read()
                    pos = f.tell()
                new_lines = new_data.decode("utf-8", errors="replace").splitlines()
                if new_lines and state.log_subscribers:
                    for ws in list(state.log_subscribers):
                        try:
                            await ws.send(json.dumps({
                                "type":  "log_response",
                                "lines": new_lines,
                                "total": -1,
                            }))
                        except Exception:
                            state.log_subscribers.discard(ws)
        except Exception:
            pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    """Central state — single source of truth for the slot application."""

    def __init__(self) -> None:
        self.cfg             = cfg_module.load()
        self.llama           = LlamaServer(self.cfg)
        self.updater         = GitUpdater(self.cfg.repo_path)
        self.models:    list[GGUFModel] = []
        self.selected_model: GGUFModel | None = None
        self.update_in_progress: bool = False
        self.draft_model: str | None = None  # full_path of draft model, None = disabled
        self.update_available:   bool = False
        self.binary_stale:       bool = False
        self.log_subscribers: set = set()
        self.llama_version:   str = "llama"  # populated async after startup

    @staticmethod
    async def _push_log_line(ws, line: str) -> None:
        """Push a single log line to a subscriber (used by update handler)."""
        try:
            import json
            await ws.send(json.dumps({
                "type":  "log_response",
                "lines": [line],
                "total": -1,
            }))
        except Exception:
            pass

    def refresh_models(self) -> None:
        self.models = scan(self.cfg.hf_cache_path)
        if self.models and self.selected_model is None:
            self.selected_model = self.models[0]

    def selected_path(self) -> str | None:
        return self.selected_model.full_path if self.selected_model else None


# ---------------------------------------------------------------------------
# PylonRack message builders
# ---------------------------------------------------------------------------

def _update_control(state: AppState) -> dict:
    """Build the update button control based on current state."""
    if state.binary_stale:
        return {
            "id":    "update",
            "type":  "button",
            "label": state.llama_version,
            "style": "error",
            "badge": True,
        }
    elif state.update_available:
        return {
            "id":    "update",
            "type":  "button",
            "label": state.llama_version,
            "style": "warning",
            "badge": True,
        }
    else:
        return {
            "id":    "update",
            "type":  "button",
            "label": state.llama_version,
            "style": "secondary",
            "badge": False,
        }


def _manifest(state: AppState) -> dict:
    return {
        "type":    "manifest",
        "name":    "llama.cpp",
        "version": "1.0",
        "heartbeat_interval": 5,
        "controls": [
            {"id": "model_select", "type": "dropdown", "label": "Model"},
            {"id": "toggle",       "type": "button",   "label": "Start", "style": "primary"},
            {"id": "status_label", "type": "label",    "value": "Idle",  "style": "default"},
            _update_control(state),
        ],
        "ui_url": state.cfg.openwebui_url,
    }


def _controls_update(state: AppState) -> dict:
    running = state.llama.is_running
    ctrl    = _update_control(state)
    controls = [
        {
            "id":    "model_select",
            "value": state.selected_model.display_name if state.selected_model else "",
        },
        {
            "id":    "toggle",
            "label": "Stop" if running else "Start",
            "style": "destructive" if running else "primary",
        },
        {
            "id":    "status_label",
            "value": _status_text(state),
            "style": _status_style(state),
        },
        {
            "id":    "update",
            "label": ctrl["label"],
            "style": ctrl["style"],
            "badge": ctrl["badge"],
        },
    ]
    return {"type": "controls_update", "controls": controls}


def _pong(state: AppState) -> dict:
    running  = state.llama.is_running
    metrics  = state.llama.metrics() if running else {}
    ram      = state.llama.ram()
    reqs     = metrics.get("requests_processing", 0)
    ram_used = ram["llama"]["used_gb"]
    msg      = f"{ram_used} GB RAM · {reqs} req" if running else _status_text(state)
    # status "warning" when connected to slot but llama not running — rack shows no WebView
    return {
        "type":    "pong",
        "status":  "running" if running else "warning",
        "message": msg,
    }


def _status_text(state: AppState) -> str:
    if state.update_in_progress:
        return "Updating…"
    if state.llama.is_running:
        return "Running"
    return "Idle"


def _status_style(state: AppState) -> str:
    if state.update_in_progress:
        return "warning"
    if state.llama.is_running:
        return "success"
    return "default"


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

class SlotHandler:
    """Handles one WebSocket connection from the rack."""

    def __init__(self, state: AppState) -> None:
        self._state = state
        self.clients: set = set()   # all currently connected WebSocket clients

    async def handle(self, ws: WebSocketServerProtocol) -> None:
        log.info("Rack connected from %s", ws.remote_address)
        self.clients.add(ws)
        try:
            async for raw in ws:
                await self._dispatch(ws, raw)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            self._state.log_subscribers.discard(ws)
        log.info("Rack disconnected")

    async def _dispatch(self, ws: WebSocketServerProtocol, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type", "")

        if msg_type == "manifest":
            await self._send(ws, _manifest(self._state))
            await self._send(ws, _controls_update(self._state))

        elif msg_type == "ping":
            await self._send(ws, _pong(self._state))

        elif msg_type == "control_data":
            await self._handle_control_data(ws, msg)

        elif msg_type == "action":
            await self._handle_action(ws, msg)

        elif msg_type == "log_request":
            # Subscribe to live log stream
            self._state.log_subscribers.add(ws)
            # Send existing tail from file immediately
            n     = msg.get("lines", 100)
            lines = _tail_log_file(self._state.cfg.log_file_path, n)
            await self._send(ws, {"type": "log_response", "lines": lines, "total": len(lines)})

        elif msg_type == "shutdown":
            log.info("Shutdown requested by rack")
            if self._state.llama.is_running:
                self._state.llama.stop()

    async def _handle_control_data(self, ws: WebSocketServerProtocol, msg: dict) -> None:
        control_id = msg.get("control_id", "")
        if control_id == "model_select":
            self._state.refresh_models()
            await self._send(ws, {
                "type":       "control_data",
                "control_id": "model_select",
                "items":      [m.display_name for m in self._state.models],
            })

    async def _handle_action(self, ws: WebSocketServerProtocol, msg: dict) -> None:
        control_id = msg.get("control_id", "")
        value      = msg.get("value")

        if control_id == "model_select" and value:
            match = next((m for m in self._state.models if m.display_name == value), None)
            if match:
                self._state.selected_model = match
                self._state.draft_model = None  # reset draft — may be incompatible with new model
                was_running = self._state.llama.is_running

                # Update dropdown selection in rack immediately
                await self._send(ws, {
                    "type": "controls_update",
                    "controls": [{"id": "model_select", "value": value}],
                })

                if was_running:
                    # Restart with new model
                    self._state.llama.stop()
                    await self._send(ws, {
                        "type": "controls_update",
                        "controls": [
                            {"id": "toggle",       "label": "Starting…", "style": "secondary"},
                            {"id": "status_label", "value": "Loading…",  "style": "warning"},
                        ],
                    })
                    loop = asyncio.get_event_loop()
                    ok   = await loop.run_in_executor(None, lambda: self._state.llama.start(match.full_path, self._state.draft_model))
                    await self._broadcast_update(ws)
                    if ok:
                        # Signal rack to reload WebView
                        await self._send(ws, {"type": "reload_ui"})
                else:
                    await self._broadcast_update(ws)

        elif control_id == "toggle":
            await self._handle_toggle(ws)

        elif control_id == "update":
            await self._handle_update(ws)

        elif control_id == "list_local_models":
            await self._handle_list_local_models(ws)

        elif control_id == "hf_search":
            asyncio.create_task(self._handle_hf_search(ws, value))

        elif control_id == "hf_model_files":
            asyncio.create_task(self._handle_hf_model_files(ws, value))

        elif control_id == "hf_download":
            # Fire as background task — don't await, so ping/pong continues during download
            asyncio.create_task(self._handle_hf_download(ws, value))

        elif control_id == "delete_model":
            await self._handle_delete_model(ws, value)

        elif control_id == "get_settings":
            await self._handle_get_settings(ws)

        elif control_id == "check_draft_compat":
            asyncio.create_task(self._handle_check_draft_compat(ws, value))

        elif control_id == "save_settings":
            await self._handle_save_settings(ws, msg.get("settings", {}))

    async def _handle_toggle(self, ws: WebSocketServerProtocol) -> None:
        if self._state.llama.is_running:
            # Send immediate feedback BEFORE blocking stop()
            await self._send(ws, {
                "type": "controls_update",
                "controls": [
                    {"id": "toggle",       "label": "Stopping…", "style": "secondary"},
                    {"id": "status_label", "value": "Stopping…", "style": "warning"},
                ],
            })
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._state.llama.stop)
            await self._broadcast_update(ws)
        else:
            path = self._state.selected_path()
            if not path:
                self._state.refresh_models()
                path = self._state.selected_path()
            if not path:
                await self._send(ws, {
                    "type": "controls_update",
                    "controls": [{"id": "status_label", "value": "No model selected", "style": "error"}],
                })
                return

            # Send immediate feedback BEFORE blocking start()
            await self._send(ws, {
                "type": "controls_update",
                "controls": [
                    {"id": "toggle",       "label": "Starting…", "style": "secondary"},
                    {"id": "status_label", "value": "Starting…", "style": "warning"},
                ],
            })

            loop = asyncio.get_event_loop()
            ok   = await loop.run_in_executor(None, lambda: self._state.llama.start(path, self._state.draft_model))

            if ok:
                log.info("llama-server started on port %d", self._state.cfg.server.port)
            else:
                log.error("llama-server failed to start")
            await self._broadcast_update(ws)

    async def _handle_update(self, ws: WebSocketServerProtocol) -> None:
        if self._state.update_in_progress:
            return

        # Stop llama if running
        was_running = self._state.llama.is_running
        if was_running:
            self._state.llama.stop()

        self._state.update_in_progress = True
        self._state.binary_stale       = False
        self._state.update_available   = False

        # Tell rack to open log panel
        await self._send(ws, {"type": "show_log"})

        # Subscribe this connection to live log
        self._state.log_subscribers.add(ws)

        # Show "Updating…" in controls immediately
        await self._broadcast_update(ws)

        loop = asyncio.get_event_loop()

        def _run_update():
            def on_line(line: str):
                asyncio.run_coroutine_threadsafe(
                    self._state._push_log_line(ws, line), loop
                )
            return self._state.updater.update(on_line)

        ok = await loop.run_in_executor(None, _run_update)
        self._state.update_in_progress = False

        # Refresh version in background (slow)
        if ok:
            new_version = await loop.run_in_executor(
                None, lambda: _get_llama_version(self._state.cfg)
            )
            self._state.llama_version = new_version
            self._state.binary_stale  = False
            await self._state._push_log_line(ws, f"✓ Build complete — {new_version}")

        await self._broadcast_update(ws)

        # Restart if was running
        if was_running and ok:
            path = self._state.selected_path()
            if path:
                await self._state._push_log_line(ws, "Restarting llama-server…")
                ok2 = await loop.run_in_executor(None, lambda: self._state.llama.start(path, self._state.draft_model))
                await self._broadcast_update(ws)
                if ok2:
                    await self._send(ws, {"type": "reload_ui"})

    async def _broadcast_update(self, ws: WebSocketServerProtocol) -> None:
        await self._send(ws, _controls_update(self._state))

    # MARK: - Model Manager handlers

    async def _handle_list_local_models(self, ws: WebSocketServerProtocol) -> None:
        self._state.refresh_models()
        models = [
            {"display_name": m.display_name, "full_path": m.full_path, "size_gb": m.size_gb}
            for m in self._state.models
        ]
        await self._send(ws, {
            "type":   "action_result",
            "action": "local_models",
            "data":   {"type": "local_models", "models": models},
        })

    async def _handle_hf_search(self, ws: WebSocketServerProtocol, query: str | None) -> None:
        loop = asyncio.get_event_loop()
        # Empty query = show top downloaded GGUF models
        search_query = query.strip() if query else ""

        def _search():
            try:
                from huggingface_hub import list_models
                results = []
                for m in list_models(
                    search=search_query if search_query else None,
                    filter="gguf",
                    sort="downloads",
                    limit=50,
                ):
                    results.append({
                        "id":          m.id,
                        "downloads":   getattr(m, "downloads", None),
                        "likes":       getattr(m, "likes", None),
                        "description": (getattr(m, "description", None) or "")[:200],
                    })
                return results
            except Exception as e:
                log.error("HF search error: %s", e)
                return []

        results = await loop.run_in_executor(None, _search)
        await self._send(ws, {
            "type":   "action_result",
            "action": "hf_search_results",
            "data":   {"type": "hf_search_results", "results": results},
        })

    async def _handle_hf_model_files(self, ws: WebSocketServerProtocol, repo_id: str | None) -> None:
        if not repo_id:
            return
        loop = asyncio.get_event_loop()

        def _list_files():
            try:
                from huggingface_hub import list_repo_files, get_paths_info
                files = []
                for info in get_paths_info(repo_id, [
                    f for f in list_repo_files(repo_id)
                    if f.endswith(".gguf") and "mmproj" not in f.lower()
                ]):
                    files.append({
                        "name": info.path.split("/")[-1],
                        "size": info.size or 0,
                    })
                return sorted(files, key=lambda f: f["name"])
            except Exception as e:
                log.error("HF files error: %s", e)
                return []

        files = await loop.run_in_executor(None, _list_files)
        await self._send(ws, {
            "type":   "action_result",
            "action": "hf_model_files_result",
            "data":   {"type": "hf_model_files_result", "files": files},
        })

    async def _handle_hf_download(self, ws: WebSocketServerProtocol, value: str | None) -> None:
        if not value or "/" not in value:
            return
        # value = "org/repo/filename.gguf"
        parts    = value.split("/")
        filename = parts[-1]
        repo_id  = "/".join(parts[:-1])
        loop     = asyncio.get_event_loop()

        async def send_progress(progress: float):
            await self._send(ws, {
                "type":   "action_result",
                "action": "download_progress",
                "data":   {"type": "download_progress", "progress": progress},
            })

        # Get download URL via huggingface_hub, then stream with requests for real progress
        def _get_url():
            try:
                from huggingface_hub import hf_hub_url
                return hf_hub_url(repo_id=repo_id, filename=filename), None
            except Exception as e:
                return None, str(e)

        url, err = await loop.run_in_executor(None, _get_url)
        if err:
            await self._send(ws, {
                "type": "action_result", "action": "download_error",
                "data": {"type": "download_error", "message": err},
            })
            return

        # Determine destination path inside HF cache structure
        dest_dir = self._state.cfg.hf_cache_path / f"models--{repo_id.replace('/', '--')}" / "blobs"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        # Stream download with progress updates sent back via WebSocket
        progress_queue: asyncio.Queue = asyncio.Queue()

        def _stream_download():
            try:
                import requests
                headers = {}
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if hf_token:
                    headers["Authorization"] = f"Bearer {hf_token}"
                resp = requests.get(url, headers=headers, stream=True, timeout=30)
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                last_reported = -1
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                pct = downloaded / total
                                # Report only on whole-percent changes to avoid flooding
                                pct_int = int(pct * 100)
                                if pct_int != last_reported:
                                    last_reported = pct_int
                                    asyncio.run_coroutine_threadsafe(
                                        progress_queue.put(pct), loop
                                )
                asyncio.run_coroutine_threadsafe(progress_queue.put(None), loop)  # sentinel
                return str(dest_path), None
            except Exception as e:
                asyncio.run_coroutine_threadsafe(progress_queue.put(None), loop)
                return None, str(e)

        await send_progress(0.0)

        # Run download in executor, drain progress queue concurrently
        download_future = loop.run_in_executor(None, _stream_download)

        while True:
            progress = await progress_queue.get()
            if progress is None:
                break
            await send_progress(progress)

        path, error = await download_future

        if error:
            await self._send(ws, {
                "type": "action_result", "action": "download_error",
                "data": {"type": "download_error", "message": error},
            })
        else:
            self._state.refresh_models()
            # Send updated model list + dropdown items so new model is immediately selectable
            models = [
                {"display_name": m.display_name, "full_path": m.full_path, "size_gb": m.size_gb}
                for m in self._state.models
            ]
            await self._send(ws, {
                "type": "action_result", "action": "download_complete",
                "data": {"type": "download_complete"},
            })
            await self._send(ws, {
                "type":   "action_result", "action": "local_models",
                "data":   {"type": "local_models", "models": models},
            })
            # Update dropdown items in controls bar
            await self._send(ws, {
                "type": "control_data",
                "control_id": "model_select",
                "items": [m.display_name for m in self._state.models],
            })


    async def _handle_check_draft_compat(self, ws: WebSocketServerProtocol, draft_path: str | None) -> None:
        """Read vocab_size from current + draft model and compare."""
        if not draft_path or not self._state.selected_model:
            return
        loop = asyncio.get_event_loop()

        def _read_two():
            try:
                import gguf
                def vocab(path):
                    r = gguf.GGUFReader(str(path), 'r')
                    t = r.fields.get("tokenizer.ggml.tokens")
                    return len(t.data) if t else 0
                main_v  = vocab(self._state.selected_model.full_path)
                draft_v = vocab(draft_path)
                return main_v, draft_v, None
            except Exception as e:
                return 0, 0, str(e)

        main_v, draft_v, err = await loop.run_in_executor(None, _read_two)
        if err:
            await self._send(ws, {
                "type": "action_result", "action": "draft_compat",
                "data": {"type": "draft_compat", "compatible": False,
                         "main_vocab": 0, "draft_vocab": 0},
            })
        else:
            await self._send(ws, {
                "type": "action_result", "action": "draft_compat",
                "data": {"type": "draft_compat",
                         "compatible":  main_v == draft_v and main_v > 0,
                         "main_vocab":  main_v,
                         "draft_vocab": draft_v},
            })

    async def _handle_get_settings(self, ws: WebSocketServerProtocol) -> None:
        s = self._state.cfg.server
        await self._send(ws, {
            "type": "action_result", "action": "settings",
            "data": {
                "type": "settings",
                "server": {
                    "ctx_size":       s.ctx_size,
                    "n_gpu_layers":   s.n_gpu_layers,
                    "threads":        s.threads,
                    "batch_size":     s.batch_size,
                    "ubatch_size":    s.ubatch_size,
                    "temperature":    s.temperature,
                    "top_p":          s.top_p,
                    "top_k":          s.top_k,
                    "repeat_penalty": s.repeat_penalty,
                    "flash_attn":     s.flash_attn,
                    "mlock":          s.mlock,
                },
                "draft_model": self._state.draft_model,
                "hf_cache":    str(self._state.cfg.hf_cache_path),
            },
        })

    async def _handle_save_settings(self, ws: WebSocketServerProtocol, settings: dict) -> None:
        """Save settings to settings.json, reload config, restart server if running."""
        import json as _json
        from pathlib import Path as _Path

        settings_path = _Path(__file__).parent / "settings.json"
        raw = _json.loads(settings_path.read_text()) if settings_path.exists() else {}

        # Map from Swift keys → settings.json server block
        server_keys = {
            "ctx_size", "n_gpu_layers", "parallel", "threads",
            "batch_size", "ubatch_size",
            "temperature", "top_p", "top_k", "repeat_penalty",
            "flash_attn", "mlock",
        }
        if "server" not in raw:
            raw["server"] = {}
        for k, v in settings.items():
            if k in server_keys:
                raw["server"][k] = v

        # Save draft_model path in state (not in settings.json — it's runtime state)
        if "draft_model" in settings:
            self._state.draft_model = settings["draft_model"] or None

        settings_path.write_text(_json.dumps(raw, indent=2))

        # Reload config in-place
        import config as cfg_module
        self._state.cfg = cfg_module.load()
        self._state.llama._cfg = self._state.cfg

        was_running = self._state.llama.is_running

        await self._send(ws, {
            "type": "action_result", "action": "settings_saved",
            "data": {"type": "settings_saved", "restarting": was_running},
        })

        if was_running:
            loop = asyncio.get_event_loop()
            path = self._state.selected_path()
            await self._send(ws, {
                "type": "controls_update",
                "controls": [
                    {"id": "toggle",       "label": "Restarting…", "style": "secondary"},
                    {"id": "status_label", "value": "Restarting…", "style": "warning"},
                ],
            })
            await loop.run_in_executor(None, self._state.llama.stop)
            if path:
                ok = await loop.run_in_executor(None, lambda: self._state.llama.start(path, self._state.draft_model))
                await self._broadcast_update(ws)
                if ok:
                    await self._send(ws, {"type": "reload_ui"})
            else:
                await self._broadcast_update(ws)

    async def _handle_delete_model(self, ws: WebSocketServerProtocol, path: str | None) -> None:
        if not path:
            return
        loop = asyncio.get_event_loop()

        def _delete():
            try:
                import os
                os.remove(path)
                return True, None
            except Exception as e:
                return False, str(e)

        ok, error = await loop.run_in_executor(None, _delete)
        self._state.refresh_models()
        # Update model_select dropdown after delete
        if self._state.selected_model and self._state.selected_model.full_path == path:
            self._state.selected_model = self._state.models[0] if self._state.models else None

        await self._send(ws, {
            "type":   "action_result",
            "action": "delete_complete",
            "data":   {"type": "delete_complete"},
        })
        await self._broadcast_update(ws)

    @staticmethod
    async def _send(ws: WebSocketServerProtocol, data: dict) -> None:
        try:
            await ws.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            pass


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

async def _check_updates_periodically(state: AppState, handler: "SlotHandler") -> None:
    """Check version + updates at startup then every 30 minutes."""
    while True:
        if not state.update_in_progress and state.cfg.repo_path.exists():
            loop = asyncio.get_event_loop()

            # Get version (slow due to Metal init — do it in background)
            version = await loop.run_in_executor(None, lambda: _get_llama_version(state.cfg))
            if version != state.llama_version:
                state.llama_version = version

            # Check updates and binary staleness
            has_update   = await loop.run_in_executor(None, state.updater.has_update)
            binary_stale = await loop.run_in_executor(None, state.updater.is_binary_stale)

            changed = (has_update   != state.update_available or
                       binary_stale != state.binary_stale)
            state.update_available = has_update
            state.binary_stale     = binary_stale
            state.llama_version    = version

            if changed:
                # Send only the update button — avoids re-rendering all controls (no flicker)
                ctrl = _update_control(state)
                for ws in list(handler.clients):
                    try:
                        await ws.send(json.dumps({
                            "type":     "controls_update",
                            "controls": [ctrl],
                        }))
                    except Exception:
                        pass
                log.info("Update check: version=%s has_update=%s binary_stale=%s",
                         version, has_update, binary_stale)
        await asyncio.sleep(1800)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    rack_json = Path(__file__).parent / "rack.json"
    manifest  = json.loads(rack_json.read_text())
    # PYLON_PORT env var overrides rack.json port (used by tests)
    port = int(os.environ.get("PYLON_PORT", manifest.get("port", 8765)))

    state   = AppState()
    state.refresh_models()

    handler = SlotHandler(state)
    asyncio.create_task(_check_updates_periodically(state, handler))
    asyncio.create_task(_watch_log_file(state, handler))
    log.info("PylonRack llama.cpp slot starting on ws://localhost:%d", port)

    async with websockets.serve(handler.handle, "localhost", port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
