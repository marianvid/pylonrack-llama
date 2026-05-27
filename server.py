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
            n      = msg.get("lines", 50)
            lines  = self._state.llama.log_tail(n)
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
                            {"id": "status_label", "value": f"Loading {match.display_name}…", "style": "warning"},
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
