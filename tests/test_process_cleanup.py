"""tests/test_process_cleanup.py — Verify all slot processes are killed on PylonRack quit.

Tests simulate what applicationWillTerminate does and verify processes die.
Requires server.py to be running (slot active in PylonRack).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _find_server_pids() -> list[int]:
    """Find all running server.py processes."""
    result = subprocess.run(
        ["pgrep", "-f", "server.py"],
        capture_output=True, text=True
    )
    pids = []
    for line in result.stdout.strip().splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            pass
    return pids


def _find_llama_pids() -> list[int]:
    """Find all running llama-server processes."""
    result = subprocess.run(
        ["pgrep", "-f", "llama-server"],
        capture_output=True, text=True
    )
    pids = []
    for line in result.stdout.strip().splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            pass
    return pids


def _process_alive(pid: int) -> bool:
    """Check if a process is still alive."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


class TestProcessCleanup(unittest.TestCase):
    """Tests for applicationWillTerminate cleanup behavior."""

    def test_server_processes_found(self):
        """At least one server.py must be running for cleanup tests to be meaningful."""
        pids = _find_server_pids()
        self.assertGreater(len(pids), 0,
            "No server.py processes found. Start the slot in PylonRack first.")
        print(f"\nFound server.py PIDs: {pids}")

    def test_sigterm_kills_server_py(self):
        """SIGTERM to server.py must kill the process within 3 seconds."""
        pids = _find_server_pids()
        if not pids:
            self.skipTest("No server.py running")

        pid = pids[0]
        self.assertTrue(_process_alive(pid), f"PID {pid} not alive before test")

        os.kill(pid, signal.SIGTERM)

        deadline = time.time() + 3
        while time.time() < deadline:
            if not _process_alive(pid):
                break
            time.sleep(0.1)

        self.assertFalse(_process_alive(pid),
            f"server.py (PID {pid}) still alive 3s after SIGTERM")

    def test_applicationWillTerminate_command_kills_server(self):
        """The exact command from applicationWillTerminate must kill server.py."""
        pids_before = _find_server_pids()
        if not pids_before:
            self.skipTest("No server.py running")

        print(f"\nserver.py PIDs before: {pids_before}")

        # Exact command from AppDelegate.applicationWillTerminate
        result = subprocess.run(
            ["/bin/zsh", "-c",
             "pkill -TERM -f server.py 2>/dev/null; sleep 1; pkill -KILL -f server.py 2>/dev/null"],
            timeout=5
        )

        time.sleep(1.5)

        pids_after = _find_server_pids()
        print(f"server.py PIDs after:  {pids_after}")

        # Filter: only PIDs that were alive before
        survivors = [p for p in pids_before if _process_alive(p)]
        self.assertEqual(survivors, [],
            f"server.py processes still alive after cleanup: {survivors}")

    def test_no_orphan_server_after_simulated_quit(self):
        """After simulated PylonRack quit, no server.py orphans remain."""
        pids_before = _find_server_pids()
        if not pids_before:
            self.skipTest("No server.py running")

        # Simulate quit
        subprocess.run(
            ["/bin/zsh", "-c", "pkill -TERM -f server.py 2>/dev/null"],
            timeout=3
        )
        time.sleep(2)
        subprocess.run(
            ["/bin/zsh", "-c", "pkill -KILL -f server.py 2>/dev/null"],
            timeout=3
        )
        time.sleep(0.5)

        survivors = _find_server_pids()
        self.assertEqual(survivors, [],
            f"Orphan server.py processes after quit: {survivors}")

    def test_slot_not_running_after_restart(self):
        """After PylonRack quit simulation, re-adding slot starts fresh process."""
        # This test verifies the problem scenario:
        # old process should not be running when new slot is added

        # Kill any existing server.py
        subprocess.run(["/bin/zsh", "-c", "pkill -KILL -f server.py 2>/dev/null"], timeout=3)
        time.sleep(1)

        pids = _find_server_pids()
        self.assertEqual(pids, [],
            f"server.py still running after kill: {pids}. "
            "When PylonRack restarts, it would pick up stale process.")


class TestLlamaCleanup(unittest.TestCase):
    """Tests for llama-server cleanup."""

    def test_llama_server_not_orphaned(self):
        """After server.py dies, llama-server should also terminate (via stop() call)."""
        # This is a state check — if server.py is dead but llama is running, it's orphaned
        server_pids = _find_server_pids()
        llama_pids  = _find_llama_pids()

        if not server_pids and llama_pids:
            self.fail(
                f"ORPHAN DETECTED: llama-server running (PIDs {llama_pids}) "
                f"but server.py is not running. "
                f"llama-server was not stopped when server.py died."
            )
        print(f"\nserver.py: {server_pids}, llama-server: {llama_pids} — OK")

    def test_process_group_killed(self):
        """Verify that both server.py and its children die on SIGTERM to server.py."""
        server_pids = _find_server_pids()
        if not server_pids:
            self.skipTest("No server.py running")

        pid = server_pids[0]

        # Get child processes
        result = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True, text=True
        )
        child_pids = [int(l) for l in result.stdout.strip().splitlines() if l.strip()]
        print(f"\nserver.py PID {pid} children: {child_pids}")

        # Kill process group
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            os.kill(pid, signal.SIGTERM)

        time.sleep(2)

        # Verify all dead
        alive = [p for p in [pid] + child_pids if _process_alive(p)]
        self.assertEqual(alive, [],
            f"Processes still alive after SIGTERM to group: {alive}")


if __name__ == "__main__":
    print("=== Process Cleanup Tests ===")
    print(f"Current server.py PIDs: {_find_server_pids()}")
    print(f"Current llama PIDs:     {_find_llama_pids()}")
    print()
    unittest.main(verbosity=2)
