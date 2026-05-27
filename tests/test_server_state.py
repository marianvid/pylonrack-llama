"""tests/test_server_state.py — Tests for server state: update detection, version, resize event."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg_module
from git_updater import GitUpdater

WS_URL = "ws://localhost:8765"


async def _handshake():
    import websockets
    ws = await websockets.connect(WS_URL, open_timeout=5)
    await ws.send(json.dumps({"type": "manifest"}))
    for _ in range(5):
        try:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
            if msg.get("type") == "controls_update":
                break
        except asyncio.TimeoutError:
            break
    return ws


class TestUpdateDetection(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        try:
            self.ws = await _handshake()
        except Exception as e:
            self.skipTest(f"Server not available: {e}")

    async def asyncTearDown(self):
        if hasattr(self, "ws"):
            try: await self.ws.close()
            except: pass

    async def test_pong_contains_update_badge_when_update_available(self):
        """If has_update() is True, controls_update must have update.badge=True."""
        cfg = cfg_module.load()
        updater = GitUpdater(cfg.repo_path)
        has = updater.has_update()

        # Get current controls state via ping
        await self.ws.send(json.dumps({"type": "ping"}))
        # Collect responses
        responses = []
        for _ in range(5):
            try:
                r = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=2))
                responses.append(r)
                if r.get("type") == "pong":
                    break
            except asyncio.TimeoutError:
                break

        # Request fresh controls_update
        await self.ws.send(json.dumps({"type": "manifest"}))
        controls_update = None
        for _ in range(5):
            try:
                r = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=2))
                if r.get("type") == "controls_update":
                    controls_update = r
                    break
            except asyncio.TimeoutError:
                break

        self.assertIsNotNone(controls_update, "No controls_update received")
        update_ctrl = next(
            (c for c in controls_update.get("controls", []) if c.get("id") == "update"),
            None
        )
        self.assertIsNotNone(update_ctrl, "No 'update' control in controls_update")
        self.assertEqual(
            update_ctrl.get("badge"), has,
            f"update.badge should be {has} (has_update()={has}) but got {update_ctrl.get('badge')}"
        )

    async def test_update_check_runs_at_startup(self):
        """Server must check for updates soon after connect — not wait 30min."""
        cfg = cfg_module.load()
        updater = GitUpdater(cfg.repo_path)
        has = updater.has_update()

        if not has:
            self.skipTest("No update available — cannot verify badge")

        # Wait max 10s for controls_update with badge=True
        await self.ws.send(json.dumps({"type": "manifest"}))
        deadline = asyncio.get_event_loop().time() + 10
        found = False
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=2))
                if r.get("type") == "controls_update":
                    update_ctrl = next(
                        (c for c in r.get("controls", []) if c.get("id") == "update"),
                        None
                    )
                    if update_ctrl and update_ctrl.get("badge") is True:
                        found = True
                        break
            except asyncio.TimeoutError:
                break

        self.assertTrue(found,
            "Server did not send update badge=True within 10s of connect. "
            "Check that _check_updates_periodically runs at startup, not after 30min delay.")


class TestLlamaVersion(unittest.TestCase):

    def test_llama_binary_reports_version(self):
        """llama-server binary must respond to --version."""
        cfg = cfg_module.load()
        result = subprocess.run(
            [str(cfg.bin_path), "--version"],
            capture_output=True, text=True, timeout=5
        )
        output = (result.stdout + result.stderr).lower()
        self.assertTrue(
            "version" in output,
            f"Expected 'version' in output, got: {output[:200]}"
        )

    def test_llama_version_extractable(self):
        """Version string must be parseable from stderr."""
        cfg = cfg_module.load()
        result = subprocess.run(
            [str(cfg.bin_path), "--version"],
            capture_output=True, text=True, timeout=5
        )
        # Format: "version: 9189 (64b38b561)"
        import re
        match = re.search(r"version:\s*(\d+)", result.stderr)
        self.assertIsNotNone(match, f"Could not parse version from: {result.stderr[:200]}")
        version = int(match.group(1))
        self.assertGreater(version, 0)
        print(f"\nllama.cpp version: {version}")


class TestWebViewResizeSignal(unittest.IsolatedAsyncioTestCase):
    """Verify server sends reload_ui after start — which triggers WebView resize."""

    async def asyncSetUp(self):
        try:
            self.ws = await _handshake()
        except Exception as e:
            self.skipTest(f"Server not available: {e}")

    async def asyncTearDown(self):
        if hasattr(self, "ws"):
            try: await self.ws.close()
            except: pass

    async def test_pong_status_is_running_when_llama_running(self):
        """When llama is running, pong status must be 'running' (not 'warning')."""
        cfg = cfg_module.load()
        import socket
        s = socket.socket()
        s.settimeout(1)
        llama_running = s.connect_ex(("localhost", cfg.server.port)) == 0
        s.close()

        if not llama_running:
            self.skipTest("llama-server not running")

        await self.ws.send(json.dumps({"type": "ping"}))
        for _ in range(5):
            try:
                r = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=2))
                if r.get("type") == "pong":
                    self.assertEqual(r.get("status"), "running",
                        f"Expected status=running when llama is up, got: {r.get('status')}")
                    return
            except asyncio.TimeoutError:
                break
        self.fail("No pong received")


if __name__ == "__main__":
    import socket
    s = socket.socket()
    s.settimeout(2)
    if s.connect_ex(("localhost", 8765)) != 0:
        print("ERROR: server not running on localhost:8765")
        sys.exit(1)
    s.close()
    unittest.main(verbosity=2)
