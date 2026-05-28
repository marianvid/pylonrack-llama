"""conftest.py — Auto-start/stop server.py for integration tests.

No manual PylonRack interaction needed.
Server starts on TEST_PORT (8766) so production port (8765) is unaffected.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

LLAMA_DIR  = Path(__file__).parent.parent
LLAMA_VENV = LLAMA_DIR / ".venv" / "bin" / "python3"
TEST_PORT  = 8766

_server_proc = None


def _port_in_use(port: int) -> bool:
    s = socket.socket()
    s.settimeout(0.5)
    result = s.connect_ex(("localhost", port))
    s.close()
    return result == 0


def _wait_ready(port: int, timeout: float = 10.0) -> bool:
    """Wait until server accepts connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_in_use(port):
            return True
        time.sleep(0.2)
    return False


def pytest_sessionstart(session):
    """Start server.py before any tests."""
    global _server_proc

    # Kill any leftover on test port
    subprocess.run(
        ["/bin/zsh", "-c", f"lsof -ti tcp:{TEST_PORT} | xargs kill -9 2>/dev/null; true"],
        timeout=3
    )
    time.sleep(0.5)

    env = os.environ.copy()
    env["PYLON_PORT"] = str(TEST_PORT)

    _server_proc = subprocess.Popen(
        [str(LLAMA_VENV), "server.py"],
        cwd=str(LLAMA_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    ready = _wait_ready(TEST_PORT, timeout=10)
    if not ready:
        _server_proc.kill()
        _server_proc = None
        print(f"\nWARNING: server.py did not start on port {TEST_PORT}")


def pytest_sessionfinish(session, exitstatus):
    """Kill server.py after all tests."""
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()


@pytest.fixture(scope="session")
def server_port():
    """Test server port. Skip if not running."""
    if _server_proc is None or _server_proc.poll() is not None:
        if _port_in_use(8765):
            return 8765  # fallback to production port
        pytest.skip("Server not running")
    return TEST_PORT


@pytest.fixture(scope="session")
def server_pid():
    """Server PID for process tests."""
    if _server_proc is None:
        pytest.skip("Server not auto-started")
    return _server_proc.pid
