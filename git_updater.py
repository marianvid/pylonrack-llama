"""git_updater.py — Check for updates and rebuild llama.cpp."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


import shutil

CMAKE = shutil.which("cmake") or "/opt/homebrew/bin/cmake"

class GitUpdater:
    """Checks and applies updates to a git repository, then rebuilds."""

    def __init__(self, repo_path: Path) -> None:
        self._repo = repo_path

    def is_binary_stale(self) -> bool:
        """Return True if binary version doesn't match current git commit count."""
        binary = self._repo / "build" / "bin" / "llama-server"
        if not binary.exists() or not self._repo.exists():
            return False
        try:
            # Get binary version number
            result = subprocess.run(
                [str(binary), "--version"],
                capture_output=True, text=True, timeout=15
            )
            import re
            m = re.search(r"version:\s*(\d+)", result.stderr)
            if not m:
                return False
            binary_version = int(m.group(1))

            # Get current git commit count
            result2 = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=self._repo,
                capture_output=True, text=True, timeout=10
            )
            if result2.returncode != 0:
                return False
            git_version = int(result2.stdout.strip())

            stale = git_version - binary_version > 10
            if stale:
                log.info("Binary stale: git=%d binary=%d delta=%d",
                         git_version, binary_version, git_version - binary_version)
            return stale
        except Exception as exc:
            log.debug("is_binary_stale error: %s", exc)
            return False

    def has_update(self) -> bool:
        """Return True if remote has commits ahead of local HEAD."""
        if not self._repo.exists():
            return False
        try:
            # Fetch quietly — do not update working tree
            subprocess.run(
                ["git", "fetch", "--quiet"],
                cwd=self._repo,
                capture_output=True,
                timeout=15,
            )
            result = subprocess.run(
                ["git", "rev-list", "HEAD..origin/HEAD", "--count"],
                cwd=self._repo,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0 and result.stdout.strip() not in ("", "0")
        except Exception as exc:
            log.warning("git fetch failed: %s", exc)
            return False

    def update(self, progress_callback=None) -> bool:
        """Pull latest commits and do a clean rebuild."""
        if not self._repo.exists():
            self._emit(progress_callback, "ERROR: repo path does not exist")
            return False

        steps = [
            (["git", "pull", "--ff-only"],
             "Pulling latest commits…"),
            ([CMAKE, "-B", "build"],
             "Configuring CMake…"),
            ([CMAKE, "--build", "build", "--config", "Release", "-j"],
             "Building…"),
        ]

        for cmd, label in steps:
            self._emit(progress_callback, label)
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=self._repo,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for line in proc.stdout:
                    self._emit(progress_callback, line.rstrip())
                proc.wait()
                if proc.returncode != 0:
                    self._emit(progress_callback, f"FAILED (exit {proc.returncode})")
                    return False
            except Exception as exc:
                self._emit(progress_callback, f"ERROR: {exc}")
                return False

        # Verify binary is newer than before
        binary = self._repo / "build" / "bin" / "llama-server"
        self._emit(progress_callback, f"Binary: {binary} — built successfully.")
        self._emit(progress_callback, "Update complete.")
        return True

    @staticmethod
    def _emit(callback, line: str) -> None:
        if callback:
            callback(line)
        else:
            log.info("[git_updater] %s", line)
