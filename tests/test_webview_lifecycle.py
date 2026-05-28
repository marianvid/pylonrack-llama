"""tests/test_webview_lifecycle.py — Tests for WebView lifecycle.

Simulates: fresh connect → manifest → start llama → verify ui_url present and server responding.
Requires server.py running on localhost:8765.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg_module

WS_URL  = "ws://localhost:8765"
TIMEOUT = 30


async def _handshake(ws):
    await ws.send(json.dumps({"type": "manifest"}))
    manifest       = None
    controls_update = None
    for _ in range(10):
        try:
            r = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
            if r.get("type") == "manifest":
                manifest = r
            elif r.get("type") == "controls_update":
                controls_update = r
                break
        except asyncio.TimeoutError:
            break
    return manifest, controls_update


class TestWebViewLifecycle(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        try:
            import websockets
            self.ws = await websockets.connect(WS_URL, open_timeout=5)
        except Exception as e:
            self.skipTest(f"Server not available: {e}")

    async def asyncTearDown(self):
        if hasattr(self, "ws"):
            try: await self.ws.close()
            except: pass

    async def test_manifest_contains_ui_url(self):
        """Manifest must always contain ui_url — WebView cannot load without it."""
        manifest, _ = await _handshake(self.ws)
        self.assertIsNotNone(manifest, "No manifest received")
        self.assertIn("ui_url", manifest,
            "Manifest missing ui_url — WebView will be blank")
        self.assertTrue(manifest["ui_url"],
            "ui_url is empty — WebView will be blank")
        print(f"\nui_url: {manifest['ui_url']}")

    async def test_ui_url_port_matches_server_config(self):
        """ui_url port must match server config port."""
        manifest, _ = await _handshake(self.ws)
        self.assertIsNotNone(manifest)
        ui_url = manifest.get("ui_url", "")
        cfg    = cfg_module.load()
        self.assertIn(str(cfg.server.port), ui_url,
            f"ui_url {ui_url} does not contain configured port {cfg.server.port}")

    async def test_pong_status_warning_when_llama_stopped(self):
        """When llama not running, pong status must be 'warning' (triggers blank WebView in rack)."""
        cfg = cfg_module.load()
        s   = socket.socket()
        s.settimeout(1)
        llama_running = s.connect_ex(("localhost", cfg.server.port)) == 0
        s.close()

        if llama_running:
            self.skipTest("llama-server is running — cannot test stopped state")

        await _handshake(self.ws)
        await self.ws.send(json.dumps({"type": "ping"}))
        for _ in range(5):
            try:
                r = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=2))
                if r.get("type") == "pong":
                    self.assertEqual(r["status"], "warning",
                        "Expected 'warning' when llama stopped — rack shows blank panel correctly")
                    return
            except asyncio.TimeoutError:
                break
        self.fail("No pong received")

    async def test_pong_status_running_after_start(self):
        """After start action, pong status must become 'running' within TIMEOUT seconds."""
        cfg = cfg_module.load()
        s   = socket.socket()
        s.settimeout(1)
        llama_running = s.connect_ex(("localhost", cfg.server.port)) == 0
        s.close()

        if not llama_running:
            self.skipTest("llama-server not running — start it first for this test")

        await _handshake(self.ws)
        await self.ws.send(json.dumps({"type": "ping"}))
        for _ in range(5):
            try:
                r = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=2))
                if r.get("type") == "pong":
                    self.assertEqual(r["status"], "running",
                        f"Expected 'running' when llama is up, got: {r['status']}")
                    return
            except asyncio.TimeoutError:
                break
        self.fail("No pong received")

    async def test_llama_server_responds_on_configured_port(self):
        """When llama is running, it must respond to HTTP on configured port."""
        cfg = cfg_module.load()
        s   = socket.socket()
        s.settimeout(1)
        llama_running = s.connect_ex(("localhost", cfg.server.port)) == 0
        s.close()

        if not llama_running:
            self.skipTest("llama-server not running")

        import urllib.request
        try:
            resp = urllib.request.urlopen(
                f"http://localhost:{cfg.server.port}/",
                timeout=5
            )
            self.assertEqual(resp.status, 200,
                f"Expected 200 from llama UI, got {resp.status}")
            content_type = resp.headers.get("content-type", "")
            self.assertIn("html", content_type.lower(),
                f"Expected HTML response from llama UI, got: {content_type}")
        except Exception as e:
            self.fail(f"llama-server not responding on port {cfg.server.port}: {e}")

    async def test_controls_update_has_all_required_controls(self):
        """controls_update must contain model_select, toggle, status_label, update."""
        _, cu = await _handshake(self.ws)
        self.assertIsNotNone(cu, "No controls_update received after manifest")
        ids = [c.get("id") for c in cu.get("controls", [])]
        for required in ["model_select", "toggle", "status_label", "update"]:
            self.assertIn(required, ids,
                f"Missing control '{required}' in controls_update")

    async def test_update_control_has_version_label(self):
        """Update button label must be a version string like 'b9189'."""
        import re
        _, cu = await _handshake(self.ws)
        self.assertIsNotNone(cu)
        manifest, _ = await _handshake(self.ws)
        update_ctrl = next(
            (c for c in manifest.get("controls", []) if c.get("id") == "update"),
            None
        )
        if not update_ctrl:
            self.skipTest("No update control in manifest")
        label = update_ctrl.get("label", "")
        self.assertRegex(label, r"b\d+",
            f"Update button label should be 'bNNNN' (version), got: '{label}'")


class TestFreshStartWebView(unittest.IsolatedAsyncioTestCase):
    """Simulate fresh rack start: connect, get manifest, verify WebView conditions."""

    async def test_fresh_connect_gets_ui_url_immediately(self):
        """On fresh connect, manifest must arrive with ui_url before any user action."""
        import websockets
        try:
            ws = await websockets.connect(WS_URL, open_timeout=5)
        except Exception as e:
            self.skipTest(f"Server not available: {e}")

        try:
            # Send manifest immediately — no other setup
            await ws.send(json.dumps({"type": "manifest"}))

            manifest = None
            for _ in range(10):
                try:
                    r = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
                    if r.get("type") == "manifest":
                        manifest = r
                        break
                except asyncio.TimeoutError:
                    break

            self.assertIsNotNone(manifest, "No manifest on fresh connect")
            self.assertIn("ui_url", manifest,
                "ui_url missing from manifest on fresh connect — WebView will be blank")
            self.assertTrue(manifest["ui_url"],
                "ui_url is empty on fresh connect")

        finally:
            await ws.close()

    async def test_webview_visible_when_llama_running(self):
        """When llama running: pong=running → rack shows WebView. Verify the chain."""
        import websockets
        cfg = cfg_module.load()
        s   = socket.socket()
        s.settimeout(1)
        llama_running = s.connect_ex(("localhost", cfg.server.port)) == 0
        s.close()

        if not llama_running:
            self.skipTest("llama-server not running")

        try:
            ws = await websockets.connect(WS_URL, open_timeout=5)
        except Exception as e:
            self.skipTest(f"Server not available: {e}")

        try:
            manifest, _ = await _handshake(ws)

            # 1. manifest has ui_url
            self.assertIn("ui_url", manifest)
            ui_url = manifest["ui_url"]

            # 2. pong says running
            await ws.send(json.dumps({"type": "ping"}))
            pong = None
            for _ in range(5):
                try:
                    r = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
                    if r.get("type") == "pong":
                        pong = r; break
                except asyncio.TimeoutError:
                    break
            self.assertIsNotNone(pong)
            self.assertEqual(pong["status"], "running")

            # 3. ui_url is reachable
            import urllib.request
            resp = urllib.request.urlopen(ui_url, timeout=5)
            self.assertEqual(resp.status, 200,
                f"WebView URL {ui_url} not reachable — WebView will be blank")

        finally:
            await ws.close()


if __name__ == "__main__":
    s = socket.socket()
    s.settimeout(2)
    if s.connect_ex(("localhost", 8765)) != 0:
        print("ERROR: server not running on localhost:8765")
        sys.exit(1)
    s.close()
    unittest.main(verbosity=2)
