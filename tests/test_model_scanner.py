"""tests/test_model_scanner.py — model_scanner unit tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_scanner import scan, GGUFModel


class TestModelScanner(unittest.TestCase):

    def _fake_gguf(self, path: str, size: int = 4_000_000_000):
        p = MagicMock(spec=Path)
        p.name   = Path(path).name
        p.stem   = Path(path).stem
        p.parent = Path(path).parent
        p.parents = list(Path(path).parents)
        p.stat.return_value.st_size = size
        return p

    def test_mmproj_files_excluded(self):
        """mmproj files must never appear in scan results."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "hub"
            repo = base / "models--org--model" / "snapshots" / "abc123"
            repo.mkdir(parents=True)
            (repo / "model-Q4_K_M.gguf").write_bytes(b"fake")
            (repo / "mmproj-model-f16.gguf").write_bytes(b"fake")
            (repo / "mmproj-F16.gguf").write_bytes(b"fake")

            results = scan(base)

        names = [m.full_path.split("/")[-1] for m in results]
        self.assertNotIn("mmproj-model-f16.gguf", names, "mmproj must be excluded")
        self.assertNotIn("mmproj-F16.gguf", names, "mmproj must be excluded")
        self.assertIn("model-Q4_K_M.gguf", names, "main model must be included")

    def test_empty_cache_returns_empty(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(scan(Path(tmp) / "hub"), [])

    def test_nonexistent_path_returns_empty(self):
        self.assertEqual(scan(Path("/nonexistent/path")), [])

    def test_display_name_format(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "hub"
            repo = base / "models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF" / "snapshots" / "abc"
            repo.mkdir(parents=True)
            (repo / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf").write_bytes(b"x" * 100)

            results = scan(base)

        self.assertEqual(len(results), 1)
        self.assertIn("bartowski", results[0].display_name)
        self.assertIn("Meta-Llama", results[0].display_name)

    def test_size_gb_calculated(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "hub"
            repo = base / "models--org--model" / "snapshots" / "abc"
            repo.mkdir(parents=True)
            # Write ~4GB fake
            f = repo / "model.gguf"
            f.write_bytes(b"x")
            # Patch stat
            results = scan(base)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0].size_gb, float)

    def test_results_sorted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "hub"
            for name in ["z_model", "a_model", "m_model"]:
                repo = base / f"models--org--{name}" / "snapshots" / "abc"
                repo.mkdir(parents=True)
                (repo / f"{name}.gguf").write_bytes(b"x")

            results = scan(base)

        paths = [r.full_path for r in results]
        self.assertEqual(paths, sorted(paths), "Results must be sorted by path")


if __name__ == "__main__":
    unittest.main(verbosity=2)
