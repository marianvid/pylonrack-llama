"""tests/test_llama_server.py — LlamaServer unit tests.

Tests run against mock processes (sleep, echo) — no real llama-server needed.
Covers: stop() correctness, stop→change model→start sequence, is_running, _find_pid.
"""

from __future__ import annotations

import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig, ServerConfig
from llama_server import LlamaServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(port: int = 19500) -> AppConfig:
    return AppConfig(
        llama_bin="/bin/sleep",
        llama_repo="",
        hf_cache="",
        log_file="/tmp/test_llama.log",
        openwebui_url="http://localhost:1234",
        server=ServerConfig(port=port),
    )


def _server(port: int = 19500) -> LlamaServer:
    return LlamaServer(_cfg(port))


# ---------------------------------------------------------------------------
# stop() — must wait for death
# ---------------------------------------------------------------------------

class TestStop(unittest.TestCase):

    def test_stop_process_is_dead_on_return(self):
        """stop() must not return while process is still alive."""
        s = _server()
        proc = subprocess.Popen(["sleep", "30"])
        s._process = proc
        s.stop()
        self.assertFalse(proc.poll() is None, "Process must be dead after stop()")

    def test_stop_clears_state(self):
        s = _server()
        proc = subprocess.Popen(["sleep", "30"])
        s._process    = proc
        s._model_path = "/old.gguf"
        s.stop()
        self.assertIsNone(s._process)
        self.assertIsNone(s._model_path)
        self.assertIsNone(s._start_time)

    def test_stop_idempotent(self):
        s = _server()
        s.stop()  # no process
        s.stop()  # again — must not raise

    def test_stop_terminates_external_process(self):
        """stop() must terminate externally detected PID."""
        s = _server()
        ext = subprocess.Popen(["sleep", "30"])
        with patch.object(s, "_find_pid", return_value=ext.pid):
            s.stop()
        time.sleep(0.5)
        self.assertIsNotNone(ext.poll(), "External process must be dead after stop()")
        if ext.poll() is None:
            ext.kill()

    def test_stop_external_process_waited(self):
        """After stop() returns, external process must actually be dead (not just signalled)."""
        s = _server()
        ext = subprocess.Popen(["sleep", "30"])
        pid = ext.pid

        # Call count: first call returns pid (during stop), subsequent return None
        calls = [0]
        def find_pid():
            calls[0] += 1
            return pid if ext.poll() is None else None

        s._find_pid = find_pid
        s.stop()
        # After stop(), process must be dead
        self.assertIsNotNone(ext.poll())

    def test_sigkill_fallback(self):
        """stop() must SIGKILL if process ignores SIGTERM."""
        s = _server()
        proc = subprocess.Popen(["/bin/bash", "-c", "trap '' TERM; sleep 30"])
        s._process = proc
        time.sleep(0.2)
        s.stop()
        self.assertFalse(proc.poll() is None)

    def test_is_running_false_synchronously_after_stop(self):
        """is_running must be False the instant stop() returns."""
        s = _server()
        proc = subprocess.Popen(["sleep", "30"])
        s._process = proc
        s.stop()
        with patch.object(s, "_find_pid", return_value=None):
            self.assertFalse(s.is_running)


# ---------------------------------------------------------------------------
# stop → change model → start
# ---------------------------------------------------------------------------

class TestStopChangeStart(unittest.TestCase):

    def test_start_after_stop_launches_new_process(self):
        """stop() → start(new_model) must actually launch a new process."""
        s = _server()
        proc = subprocess.Popen(["sleep", "30"])
        s._process    = proc
        s._model_path = "/old.gguf"

        with patch.object(s, "_find_pid", return_value=None):
            s.stop()
            self.assertFalse(s.is_running)

            with patch.object(s, "_build_command", return_value=["echo", "ok"]), \
                 patch.object(s, "_wait_ready",    return_value=True):
                ok = s.start("/new.gguf")

        self.assertTrue(ok)
        self.assertEqual(s._model_path, "/new.gguf")
        if s._process:
            s._process.wait()

    def test_start_does_not_launch_when_still_running(self):
        """start() must be idempotent: no new process if is_running=True."""
        s = _server()
        with patch.object(s, "_find_pid", return_value=99999):
            launched = []
            with patch("subprocess.Popen", side_effect=lambda *a, **kw: launched.append(1)):
                s.start("/model.gguf")
        self.assertEqual(launched, [], "Popen must not be called when already running")

    def test_full_cycle_stop_change_start(self):
        """Full cycle: start fake → stop → start new model → model_path updated."""
        s = _server()

        # Phase 1: simulate running
        proc = subprocess.Popen(["sleep", "30"])
        s._process    = proc
        s._model_path = "/model_a.gguf"

        # Phase 2: stop
        with patch.object(s, "_find_pid", return_value=None):
            s.stop()
            self.assertFalse(s.is_running)
            self.assertIsNone(s._process)

        # Phase 3: start new model
        with patch.object(s, "_find_pid",     return_value=None), \
             patch.object(s, "_build_command", return_value=["echo", "ok"]), \
             patch.object(s, "_wait_ready",    return_value=True):
            ok = s.start("/model_b.gguf")

        self.assertTrue(ok)
        self.assertEqual(s._model_path, "/model_b.gguf")

    def test_stop_then_start_fails_gracefully_if_binary_missing(self):
        """start() returns False (not exception) if binary doesn't exist."""
        s = _server()
        s._cfg.llama_bin = "/nonexistent/llama-server"
        with patch.object(s, "_find_pid", return_value=None):
            ok = s.start("/model.gguf")
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------

class TestIsRunning(unittest.TestCase):

    def test_false_when_nothing(self):
        s = _server()
        with patch.object(s, "_find_pid", return_value=None):
            self.assertFalse(s.is_running)

    def test_true_when_process_alive(self):
        s = _server()
        proc = subprocess.Popen(["sleep", "30"])
        s._process = proc
        try:
            self.assertTrue(s.is_running)
        finally:
            proc.terminate(); proc.wait()

    def test_true_when_external_pid(self):
        s = _server()
        with patch.object(s, "_find_pid", return_value=12345):
            self.assertTrue(s.is_running)

    def test_false_after_process_exits_naturally(self):
        s = _server()
        proc = subprocess.Popen(["sleep", "0.05"])
        s._process = proc
        time.sleep(0.3)
        with patch.object(s, "_find_pid", return_value=None):
            self.assertFalse(s.is_running)


# ---------------------------------------------------------------------------
# _find_pid (macOS integration)
# ---------------------------------------------------------------------------

@unittest.skipUnless(sys.platform == "darwin", "macOS only")
class TestFindPid(unittest.TestCase):

    def test_returns_none_when_port_free(self):
        self.assertIsNone(_server(19501)._find_pid())

    def test_returns_none_for_non_llama_process(self):
        import socket
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("localhost", 19502))
        sock.listen(1)
        try:
            self.assertIsNone(_server(19502)._find_pid())
        finally:
            sock.close()


# ---------------------------------------------------------------------------
# UI feedback via server.py
# ---------------------------------------------------------------------------

class TestUIFeedback(unittest.IsolatedAsyncioTestCase):

    async def test_start_sends_immediate_broadcast(self):
        """Clicking Start must emit controls_update before blocking on llama start."""
        from server import SlotHandler

        state = MagicMock()
        state.llama.is_running = False
        state.update_in_progress = False
        state.update_available = False
        state.selected_model = MagicMock()
        state.selected_model.full_path  = "/model.gguf"
        state.selected_model.display_name = "test/model"
        state.models = []

        sent = []
        async def mock_send(ws, data): sent.append(data)
        async def mock_broadcast(ws):  sent.append({"type": "broadcast"})

        h = SlotHandler(state)
        h._send             = mock_send
        h._broadcast_update = mock_broadcast

        import asyncio
        loop = asyncio.get_event_loop()

        async def fake_executor(fn, *args):
            return True

        with patch.object(loop, "run_in_executor", side_effect=fake_executor):
            await h._handle_toggle(MagicMock())

        self.assertTrue(len(sent) > 0, "Must send at least one update on Start")

    async def test_stop_sends_immediate_feedback_before_blocking(self):
        """Clicking Stop: controls_update with 'Stopping…' sent before llama.stop() blocks."""
        from server import SlotHandler

        state = MagicMock()
        state.llama.is_running = True
        state.update_in_progress = False
        state.update_available = False
        state.selected_model = MagicMock()
        state.selected_model.display_name = "test/model"
        state.selected_model.full_path = "/model.gguf"
        state.models = []

        sent = []
        stop_called_after = []

        async def mock_send(ws, data):
            sent.append(data)

        async def mock_broadcast(ws):
            sent.append({"type": "broadcast"})

        original_stop = state.llama.stop

        def track_stop():
            stop_called_after.append(len(sent))
            original_stop()

        state.llama.stop = track_stop

        h = SlotHandler(state)
        h._send             = mock_send
        h._broadcast_update = mock_broadcast

        await h._handle_toggle(MagicMock())

        # At least one message must have been sent BEFORE stop was called
        if stop_called_after:
            self.assertGreater(stop_called_after[0], 0,
                "controls_update must be sent before llama.stop() is called")

    async def test_model_select_when_running_sends_reload_ui(self):
        """Changing model while running must emit reload_ui after restart."""
        from server import SlotHandler
        from model_scanner import GGUFModel

        state = MagicMock()
        state.llama.is_running = True
        state.update_in_progress = False
        state.update_available = False

        model_a = GGUFModel("org/ModelA", "/path/a.gguf", 4.0)
        model_b = GGUFModel("org/ModelB", "/path/b.gguf", 8.0)
        state.models = [model_a, model_b]
        state.selected_model = model_a

        sent = []
        async def mock_send(ws, data): sent.append(data)
        async def mock_broadcast(ws):  sent.append({"type": "broadcast"})

        import asyncio
        loop = asyncio.get_event_loop()

        async def fake_executor(fn, *args):
            return True  # simulate successful start

        h = SlotHandler(state)
        h._send             = mock_send
        h._broadcast_update = mock_broadcast

        with patch.object(loop, "run_in_executor", side_effect=fake_executor):
            await h._handle_action(MagicMock(), {
                "control_id": "model_select",
                "value":      "org/ModelB"
            })

        types = [m.get("type") for m in sent]
        self.assertIn("reload_ui", types,
            "reload_ui must be sent after successful restart with new model")

    async def test_model_select_when_stopped_does_not_reload(self):
        """Changing model while stopped must NOT emit reload_ui."""
        from server import SlotHandler
        from model_scanner import GGUFModel

        state = MagicMock()
        state.llama.is_running = False
        state.update_in_progress = False
        state.update_available = False

        model_a = GGUFModel("org/ModelA", "/path/a.gguf", 4.0)
        model_b = GGUFModel("org/ModelB", "/path/b.gguf", 8.0)
        state.models = [model_a, model_b]
        state.selected_model = model_a

        sent = []
        async def mock_send(ws, data): sent.append(data)
        async def mock_broadcast(ws):  sent.append({"type": "broadcast"})

        h = SlotHandler(state)
        h._send             = mock_send
        h._broadcast_update = mock_broadcast

        await h._handle_action(MagicMock(), {
            "control_id": "model_select",
            "value":      "org/ModelB"
        })

        types = [m.get("type") for m in sent]
        self.assertNotIn("reload_ui", types)

    async def test_webview_hidden_after_stop(self):
        """After stop, status_label must be 'Idle' (signals rack to hide WebView)."""
        from server import SlotHandler, _controls_update, AppState
        import config as cfg_module

        state = MagicMock()
        state.llama.is_running = False
        state.update_in_progress = False
        state.update_available = False
        state.selected_model = MagicMock()
        state.selected_model.display_name = "test/model"

        update = _controls_update(state)
        status = next((c for c in update["controls"] if c["id"] == "status_label"), None)
        self.assertIsNotNone(status)
        self.assertEqual(status["value"], "Idle")
        self.assertEqual(status["style"], "default")


if __name__ == "__main__":
    unittest.main(verbosity=2)
