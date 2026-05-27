"""model_scanner.py — Scan HF Cache for GGUF models."""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class GGUFModel:
    display_name: str   # human-readable label shown in dropdown
    full_path:    str   # absolute path to .gguf file
    size_gb:      float # approximate size in GB

    def __str__(self) -> str:
        return self.display_name


def _make_display_name(repo_dir: Path, gguf_file: Path) -> str:
    """Build a concise display name from repo directory + file name."""
    # repo dir looks like: models--org--name  or  models--org--name--subdir
    repo = repo_dir.name
    repo = re.sub(r"^models--", "", repo)
    repo = repo.replace("--", "/")

    stem = gguf_file.stem
    # Remove redundant repo prefix from filename if present
    short_repo = repo.split("/")[-1].lower()
    short_stem = stem.lower()
    if short_stem.startswith(short_repo):
        stem = stem[len(short_repo):].lstrip("-_")

    return f"{repo} / {stem}" if stem else repo


def scan(hf_cache_path: Path) -> list[GGUFModel]:
    """Return all GGUF models found under hf_cache_path, sorted by name."""
    if not hf_cache_path.exists():
        return []

    models: list[GGUFModel] = []

    for gguf_file in sorted(hf_cache_path.rglob("*.gguf")):
        # Skip projection/vision models — not inference models
        if "mmproj" in gguf_file.name.lower():
            continue
        repo_dir = gguf_file.parent
        for ancestor in gguf_file.parents:
            if ancestor.name.startswith("models--"):
                repo_dir = ancestor
                break

        size_gb = round(gguf_file.stat().st_size / (1024 ** 3), 1)
        display = _make_display_name(repo_dir, gguf_file)

        models.append(GGUFModel(
            display_name=display,
            full_path=str(gguf_file),
            size_gb=size_gb,
        ))

    return models
