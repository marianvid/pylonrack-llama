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
import threading
from pathlib import Path

import websockets
from websockets import ServerConnection as WebSocketServerProtocol

import config as cfg_module
from llama_server import LlamaServer
from model_scanner import GGUFModel, scan
from git_updater import GitUpdater

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
        self.update_available:   bool = False
        self.log_subscribers: set = set()   # active WebSocket connections subscribed to log

        # Wire up live log push
        self.llama.on_log_line = self._on_log_line

    def _on_log_line(self, line: str) -> None:
        """Called from log thread — schedule push to all subscribers."""
        import asyncio
        for ws in list(self.log_subscribers):
            try:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    self._push_log_line(ws, line), loop
                )
            except Exception:
                pass

    @staticmethod
    async def _push_log_line(ws, line: str) -> None:
        try:
            import json
            await ws.send(json.dumps({
                "type":  "log_response",
                "lines": [line],
                "total": -1,   # -1 = streaming append (not a full fetch)
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

def _manifest(cfg) -> dict:
    return {
        "type":    "manifest",
        "name":    "llama.cpp",
        "version": "1.0",
        "heartbeat_interval": 5,
        "controls": [
            {"id": "model_select",  "type": "dropdown", "label": "Model"},
            {"id": "toggle",        "type": "button",   "label": "Start",  "style": "primary"},
            {"id": "update",        "type": "button",   "label": "Update", "style": "secondary", "badge": False},
            {"id": "status_label",  "type": "label",    "value": "Idle",   "style": "default"},
        ],
        "ui_url": cfg.openwebui_url,
    }


def _controls_update(state: AppState) -> dict:
    running = state.llama.is_running
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
            "badge": state.update_available,
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

    async def handle(self, ws: WebSocketServerProtocol) -> None:
        log.info("Rack connected from %s", ws.remote_address)
        try:
            async for raw in ws:
                await self._dispatch(ws, raw)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._state.log_subscribers.discard(ws)
        log.info("Rack disconnected")

    async def _dispatch(self, ws: WebSocketServerProtocol, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type", "")

        if msg_type == "manifest":
            await self._send(ws, _manifest(self._state.cfg))
            # Push current control state
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
            # Send existing tail immediately
            n     = msg.get("lines", 50)
            lines = self._state.llama.log_tail(n)
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
                    ok   = await loop.run_in_executor(None, self._state.llama.start, match.full_path)
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
            await self._handle_hf_search(ws, value)

        elif control_id == "hf_model_files":
            await self._handle_hf_model_files(ws, value)

        elif control_id == "hf_download":
            await self._handle_hf_download(ws, value)

        elif control_id == "delete_model":
            await self._handle_delete_model(ws, value)

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
            ok   = await loop.run_in_executor(None, self._state.llama.start, path)

            if ok:
                log.info("llama-server started on port %d", self._state.cfg.server.port)
            else:
                log.error("llama-server failed to start")
            await self._broadcast_update(ws)

    async def _handle_update(self, ws: WebSocketServerProtocol) -> None:
        if self._state.update_in_progress:
            return

        was_running = self._state.llama.is_running
        if was_running:
            self._state.llama.stop()

        self._state.update_in_progress = True
        await self._broadcast_update(ws)

        loop = asyncio.get_event_loop()

        def _run_update():
            lines = []
            def on_line(line):
                lines.append(line)
            ok = self._state.updater.update(on_line)
            return ok, lines

        ok, lines = await loop.run_in_executor(None, _run_update)

        self._state.update_in_progress = False
        self._state.update_available   = False

        # Send log lines as log_response so rack shows them in log panel
        await self._send(ws, {
            "type":  "log_response",
            "lines": lines,
            "total": len(lines),
        })

        await self._broadcast_update(ws)

        if was_running and ok:
            path = self._state.selected_path()
            if path:
                loop2 = asyncio.get_event_loop()
                ok2 = await loop2.run_in_executor(None, self._state.llama.start, path)
                if ok2:
                    await self._broadcast_update(ws)

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

        def _download():
            try:
                from huggingface_hub import hf_hub_download
                import os

                dest_dir = self._state.cfg.hf_cache_path / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "downloaded"
                dest_dir.mkdir(parents=True, exist_ok=True)

                # Use hf_hub_download with cache_dir
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(self._state.cfg.hf_cache_path),
                    local_dir=None,
                )
                return path, None
            except Exception as e:
                return None, str(e)

        # Send progress updates while downloading
        await send_progress(0.0)
        path, error = await loop.run_in_executor(None, _download)

        if error:
            await self._send(ws, {
                "type":   "action_result",
                "action": "download_error",
                "data":   {"type": "download_error", "message": error},
            })
        else:
            self._state.refresh_models()
            await self._send(ws, {
                "type":   "action_result",
                "action": "download_complete",
                "data":   {"type": "download_complete"},
            })

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

async def _check_updates_periodically(state: AppState) -> None:
    """Poll for git updates every 30 minutes."""
    while True:
        await asyncio.sleep(1800)
        if not state.update_in_progress and state.cfg.repo_path.exists():
            has = await asyncio.get_event_loop().run_in_executor(
                None, state.updater.has_update
            )
            state.update_available = has


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    rack_json = Path(__file__).parent / "rack.json"
    manifest  = json.loads(rack_json.read_text())
    port      = manifest.get("port", 8765)

    state   = AppState()
    state.refresh_models()

    # Initial update check in background
    asyncio.create_task(_check_updates_periodically(state))

    handler = SlotHandler(state)
    log.info("PylonRack llama.cpp slot starting on ws://localhost:%d", port)

    async with websockets.serve(handler.handle, "localhost", port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
