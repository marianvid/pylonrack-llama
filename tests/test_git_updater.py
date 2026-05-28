"""tests/test_git_updater.py — Tests for git_updater build correctness."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg_module
from git_updater import GitUpdater


class TestBinaryFreshness(unittest.TestCase):
    """Verify binary is not older than source files."""

    def setUp(self):
        self.cfg     = cfg_module.load()
        self.binary  = self.cfg.bin_path
        self.repo    = self.cfg.repo_path

    def test_binary_exists(self):
        self.assertTrue(self.binary.exists(),
            f"llama-server binary not found at {self.binary}")

    def test_binary_not_older_than_sources(self):
        """Binary mtime must be >= most recent source file mtime.
        If this fails: sources are newer than binary — need rebuild."""
        if not self.binary.exists():
            self.skipTest("Binary not found")
        if not self.repo.exists():
            self.skipTest("Repo not found")

        binary_mtime = self.binary.stat().st_mtime

        # Find most recently modified C/C++/header source file
        newest_source = None
        newest_mtime  = 0
        for ext in ("*.cpp", "*.c", "*.h", "*.cu", "*.metal"):
            for f in self.repo.rglob(ext):
                # Skip build directory
                if "build" in f.parts:
                    continue
                mt = f.stat().st_mtime
                if mt > newest_mtime:
                    newest_mtime  = mt
                    newest_source = f

        if not newest_source:
            self.skipTest("No source files found")

        from datetime import datetime
        bin_date = datetime.fromtimestamp(binary_mtime).strftime("%Y-%m-%d %H:%M:%S")
        src_date = datetime.fromtimestamp(newest_mtime).strftime("%Y-%m-%d %H:%M:%S")

        print(f"\nBinary:        {bin_date}")
        print(f"Newest source: {src_date} ({newest_source.name})")

        self.assertGreaterEqual(
            binary_mtime, newest_mtime,
            f"\nBINARY IS STALE:\n"
            f"  Binary mtime:  {bin_date}\n"
            f"  Newest source: {src_date} ({newest_source})\n"
            f"  Action needed: rebuild with cmake --build --clean-first"
        )

    def test_binary_git_version_matches_source(self):
        """Binary version number should match git commit count."""
        if not self.binary.exists():
            self.skipTest("Binary not found")
        if not self.repo.exists():
            self.skipTest("Repo not found")

        # Get binary version
        result = subprocess.run(
            [str(self.binary), "--version"],
            capture_output=True, text=True, timeout=15
        )
        import re
        m = re.search(r"version:\s*(\d+)", result.stderr)
        if not m:
            self.skipTest("Cannot parse binary version")
        binary_version = int(m.group(1))

        # Get git commit count
        result2 = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=self.repo,
            capture_output=True, text=True, timeout=5
        )
        if result2.returncode != 0:
            self.skipTest("Cannot get git commit count")
        git_version = int(result2.stdout.strip())

        print(f"\nBinary version: b{binary_version}")
        print(f"Git commits:    {git_version}")

        # Allow small delta (version numbering may differ slightly)
        delta = abs(git_version - binary_version)
        self.assertLessEqual(delta, 50,
            f"\nBINARY VERSION MISMATCH:\n"
            f"  Binary: b{binary_version}\n"
            f"  Git:    {git_version} commits\n"
            f"  Delta:  {delta} — binary may be significantly out of date"
        )


class TestGitUpdaterLogic(unittest.TestCase):

    def setUp(self):
        self.cfg     = cfg_module.load()
        self.updater = GitUpdater(self.cfg.repo_path)

    def test_has_update_returns_bool(self):
        result = self.updater.has_update()
        self.assertIsInstance(result, bool)

    def test_update_steps_include_cmake_build(self):
        """update() must use standard cmake commands per llama.cpp docs."""
        import inspect
        source = inspect.getsource(self.updater.update)
        # Per llama.cpp docs: cmake -B build && cmake --build build --config Release
        self.assertIn("cmake", source.lower())
        self.assertIn("-B", source, "cmake must configure with -B build")
        self.assertIn("--build", source, "cmake must build with --build")
        # --clean-first is NOT used (per llama.cpp docs, standard incremental build)
        self.assertNotIn("--clean-first", source,
            "cmake must NOT use --clean-first (not in llama.cpp official docs)")

    def test_update_steps_target_llama_server(self):
        """update() builds all targets per llama.cpp docs (no --target restriction)."""
        import inspect
        source = inspect.getsource(self.updater.update)
        # Per llama.cpp docs: cmake --build build --config Release (no --target)
        self.assertIn("--build", source)
        self.assertIn("Release", source)

    def test_update_nonexistent_repo_returns_false(self):
        updater = GitUpdater(Path("/nonexistent/path"))
        result  = updater.update()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
