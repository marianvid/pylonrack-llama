"""tests/test_model_manager.py — Integration tests for model manager protocol.

Requires server.py to be running on localhost:8765.
Run with: python3 -m pytest tests/test_model_manager.py -v -s
Or standalone: python3 tests/test_model_manager.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


WS_URL = "ws://localhost:8765"
TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _connect_and_handshake():
    """Connect, do manifest handshake, return ws + any pending messages."""
    import websockets
    ws = await websockets.connect(WS_URL, open_timeout=5)
    await ws.send(json.dumps({"type": "manifest"}))
    # Drain manifest + controls_update
    for _ in range(5):
        try:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
            if msg.get("type") == "controls_update":
                break
        except asyncio.TimeoutError:
            break
    return ws


async def _send_action(ws, control_id: str, value: str | None = None, recv_timeout: float = TIMEOUT):
    """Send action and collect responses until action_result arrives."""
    msg = {"type": "action", "control_id": control_id}
    if value:
        msg["value"] = value
    await ws.send(json.dumps(msg))

    responses = []
    deadline = asyncio.get_event_loop().time() + recv_timeout
    while asyncio.get_event_loop().time() < deadline:
        remaining = deadline - asyncio.get_event_loop().time()
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=min(2.0, remaining))
            r = json.loads(raw)
            responses.append(r)
            if r.get("type") == "action_result":
                return responses
        except asyncio.TimeoutError:
            break
    return responses


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelManagerProtocol(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        try:
            import websockets
            # Fresh connection per test — avoids response mixing between tests
            self.ws = await _connect_and_handshake()
        except Exception as e:
            self.skipTest(f"Server not available on {WS_URL}: {e}")

    async def asyncTearDown(self):
        if hasattr(self, "ws"):
            try:
                await self.ws.close()
            except Exception:
                pass

    # MARK: - list_local_models

    async def test_list_local_models_returns_action_result(self):
        """list_local_models must return action_result with type=local_models."""
        responses = await _send_action(self.ws, "list_local_models")
        result = next((r for r in responses if r.get("type") == "action_result"), None)
        self.assertIsNotNone(result, f"No action_result received. Got: {[r.get('type') for r in responses]}")
        self.assertEqual(result.get("action"), "local_models")

    async def test_list_local_models_contains_models_array(self):
        """list_local_models data must contain models array."""
        responses = await _send_action(self.ws, "list_local_models")
        result = next((r for r in responses if r.get("type") == "action_result"), None)
        self.assertIsNotNone(result)
        data = result.get("data", {})
        self.assertIn("models", data, f"No 'models' key in data: {data}")
        self.assertIsInstance(data["models"], list)

    async def test_list_local_models_has_required_fields(self):
        """Each model must have display_name, full_path, size_gb."""
        responses = await _send_action(self.ws, "list_local_models")
        result = next((r for r in responses if r.get("type") == "action_result"), None)
        self.assertIsNotNone(result)
        models = result.get("data", {}).get("models", [])
        self.assertGreater(len(models), 0, "No models found — check HF cache path in settings.json")
        for m in models:
            self.assertIn("display_name", m)
            self.assertIn("full_path", m)
            self.assertIn("size_gb", m)
            self.assertTrue(m["full_path"].endswith(".gguf"),
                f"full_path should end with .gguf: {m['full_path']}")
            self.assertGreater(m["size_gb"], 0)

    async def test_list_local_models_no_mmproj(self):
        """list_local_models must not return mmproj files."""
        responses = await _send_action(self.ws, "list_local_models")
        result = next((r for r in responses if r.get("type") == "action_result"), None)
        models = result.get("data", {}).get("models", []) if result else []
        for m in models:
            self.assertNotIn("mmproj", m["full_path"].lower(),
                f"mmproj file found in results: {m['full_path']}")

    # MARK: - hf_search

    async def test_hf_search_returns_action_result(self):
        """hf_search must return action_result with type=hf_search_results."""
        responses = await _send_action(self.ws, "hf_search", value="qwen", recv_timeout=20)
        result = next((r for r in responses if r.get("type") == "action_result"), None)
        self.assertIsNotNone(result, f"No action_result received. Got: {[r.get('type') for r in responses]}")
        self.assertEqual(result.get("action"), "hf_search_results")

    async def test_hf_search_returns_results_array(self):
        """hf_search data must contain results array with at least 1 item."""
        responses = await _send_action(self.ws, "hf_search", value="llama", recv_timeout=20)
        result = next((r for r in responses if r.get("type") == "action_result"), None)
        self.assertIsNotNone(result)
        results = result.get("data", {}).get("results", [])
        self.assertGreater(len(results), 0, "HF search returned 0 results")

    async def test_hf_search_result_has_required_fields(self):
        """Each HF search result must have id field."""
        responses = await _send_action(self.ws, "hf_search", value="gemma", recv_timeout=20)
        result = next((r for r in responses if r.get("type") == "action_result"), None)
        self.assertIsNotNone(result)
        results = result.get("data", {}).get("results", [])
        for r in results[:5]:
            self.assertIn("id", r, f"Missing 'id' in result: {r}")
            self.assertIsInstance(r["id"], str)

    async def test_hf_search_empty_query_no_crash(self):
        """hf_search with empty value must not crash server."""
        # Send empty value — server should silently skip
        await self.ws.send(json.dumps({"type": "action", "control_id": "hf_search", "value": ""}))
        # Server should still be alive — send ping
        await self.ws.send(json.dumps({"type": "ping"}))
        try:
            pong = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=5))
            self.assertEqual(pong.get("type"), "pong")
        except asyncio.TimeoutError:
            self.fail("Server did not respond to ping after empty search")

    # MARK: - hf_model_files

    async def test_hf_model_files_returns_files(self):
        """hf_model_files for a known repo must return GGUF files."""
        responses = await _send_action(
            self.ws, "hf_model_files",
            value="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            recv_timeout=20
        )
        result = next((r for r in responses if r.get("type") == "action_result"), None)
        self.assertIsNotNone(result, "No action_result received")
        self.assertEqual(result.get("action"), "hf_model_files_result")
        files = result.get("data", {}).get("files", [])
        self.assertGreater(len(files), 0, "No files found for known repo")
        for f in files:
            self.assertIn("name", f)
            self.assertIn("size", f)
            self.assertTrue(f["name"].endswith(".gguf"))
            self.assertNotIn("mmproj", f["name"].lower())


# ---------------------------------------------------------------------------
# Standalone runner with connection check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import socket

    def check_server():
        s = socket.socket()
        s.settimeout(2)
        result = s.connect_ex(("localhost", 8765))
        s.close()
        return result == 0

    if not check_server():
        print("ERROR: server not running on localhost:8765")
        print("Start the slot in PylonRack first, then run tests.")
        sys.exit(1)

    print(f"Server found on localhost:8765 — running tests...\n")
    unittest.main(verbosity=2)
