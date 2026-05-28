"""tests/test_process_cleanup.py — Verify all slot processes are killed on PylonRack quit.

Uses conftest.py auto-started server — no manual PylonRack interaction needed.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import unittest
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LLAMA_VENV = "/Volumes/Marian_Backup/work/pylonrack-slots/llama/.venv/bin/python3"
LLAMA_DIR  = "/Volumes/Marian_Backup/work/pylonrack-slots/llama"


def _find_pids(pattern: str) -> list[int]:
    result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(l) for l in result.stdout.strip().splitlines() if l.strip()]


def _alive(pid: int) -> bool:
    """Check if pid is alive AND is still a server.py process (not PID reuse)."""
    try:
        os.kill(pid, 0)
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            capture_output=True, text=True
        )
        return "server.py" in result.stdout
    except (ProcessLookupError, PermissionError):
        return False


def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.2)
    return not _alive(pid)


class TestProcessCleanup(unittest.TestCase):
    """
    PID snapshot captured once in setUpClass from auto-started server.
    All tests work against that snapshot — order independent.
    """

    server_pids: list[int] = []
    llama_pids:  list[int] = []

    @classmethod
    def setUpClass(cls):
        cls.server_pids = _find_pids("server.py")
        cls.llama_pids  = _find_pids("llama-server")
        print(f"\n[setUpClass] server.py PIDs: {cls.server_pids}")
        print(f"[setUpClass] llama-server PIDs: {cls.llama_pids}")

    def setUp(self):
        if not self.server_pids:
            self.skipTest("No server.py running — conftest.py should have started it")

    def test_1_server_processes_found(self):
        self.assertGreater(len(self.server_pids), 0)
        print(f"\nserver.py PIDs: {self.server_pids}")

    def test_2_no_orphan_llama_without_server(self):
        if not self.llama_pids:
            return
        self.assertGreater(len(self.server_pids), 0,
            f"ORPHAN: llama-server {self.llama_pids} running but no server.py")

    def test_3_server_pid_is_actually_alive(self):
        for pid in self.server_pids:
            self.assertTrue(_alive(pid), f"server.py PID {pid} died unexpectedly")

    def test_4_sigterm_kills_server(self):
        pid = self.server_pids[0]
        self.assertTrue(_alive(pid))
        os.kill(pid, signal.SIGTERM)
        self.assertTrue(_wait_dead(pid, timeout=5),
            f"server.py PID {pid} still alive 5s after SIGTERM")
        print(f"\nPID {pid} dead after SIGTERM ✓")

    def test_5_applicationWillTerminate_kills_all(self):
        """Exact command from AppDelegate.applicationWillTerminate."""
        still_alive = [p for p in self.server_pids if _alive(p)]

        if not still_alive:
            # Start fresh with direct python3 (PID = python3, not shell wrapper)
            proc = subprocess.Popen(
                [LLAMA_VENV, "server.py"],
                cwd=LLAMA_DIR,
                env={**os.environ, "PYLON_PORT": "8767"},  # different port to avoid conflict
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(2)
            fresh_pids = [proc.pid]
            print(f"\nStarted fresh server.py PID: {fresh_pids}")
        else:
            fresh_pids = still_alive

        if not fresh_pids:
            self.skipTest("Could not start fresh server.py")

        for pid in fresh_pids:
            self.assertTrue(_alive(pid), f"PID {pid} not alive before kill")

        # Exact command from AppDelegate.applicationWillTerminate
        subprocess.run(
            ["/bin/zsh", "-c",
             "pkill -TERM -f server.py 2>/dev/null; "
             "sleep 2; "
             "pkill -KILL -f server.py 2>/dev/null"],
            timeout=8
        )

        survivors = [p for p in fresh_pids if not _wait_dead(p, timeout=5)]
        self.assertEqual(survivors, [],
            f"server.py PIDs {survivors} still alive after applicationWillTerminate")
        print(f"\nAll {len(fresh_pids)} server.py processes killed ✓")

    def test_6_no_orphans_after_cleanup(self):
        server_survivors = _find_pids("server.py")
        self.assertEqual(server_survivors, [],
            f"server.py orphans: {server_survivors}")
        print(f"\nNo server.py orphans ✓")


if __name__ == "__main__":
    unittest.main(verbosity=2)
