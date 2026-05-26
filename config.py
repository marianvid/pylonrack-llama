"""config.py — Load and validate settings.json with defaults."""

from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass, field


SETTINGS_FILE = Path(__file__).parent / "settings.json"


@dataclass
class ServerConfig:
    host:        str = "0.0.0.0"
    port:        int = 1234
    ctx_size:    int = 131072
    n_gpu_layers: int = 99
    parallel:    int = 2
    threads:     int = 8
    batch_size:  int = 512
    ubatch_size: int = 256


@dataclass
class AppConfig:
    llama_bin:    str = "/usr/local/bin/llama-server"
    llama_repo:   str = ""
    hf_cache:     str = ""
    log_file:     str = "~/.pylonrack/llama-server.log"
    openwebui_url: str = "http://localhost:8080"
    server:       ServerConfig = field(default_factory=ServerConfig)

    @property
    def log_path(self) -> Path:
        return Path(os.path.expanduser(self.log_file))

    @property
    def bin_path(self) -> Path:
        return Path(os.path.expanduser(self.llama_bin))

    @property
    def repo_path(self) -> Path:
        return Path(os.path.expanduser(self.llama_repo))

    @property
    def hf_cache_path(self) -> Path:
        return Path(os.path.expanduser(self.hf_cache))


def load() -> AppConfig:
    """Load settings.json, fall back to defaults for missing keys."""
    if not SETTINGS_FILE.exists():
        return AppConfig()

    raw = json.loads(SETTINGS_FILE.read_text())

    server_raw = raw.get("server", {})
    server = ServerConfig(
        host        = server_raw.get("host",         ServerConfig.host),
        port        = server_raw.get("port",         ServerConfig.port),
        ctx_size    = server_raw.get("ctx_size",     ServerConfig.ctx_size),
        n_gpu_layers = server_raw.get("n_gpu_layers", ServerConfig.n_gpu_layers),
        parallel    = server_raw.get("parallel",     ServerConfig.parallel),
        threads     = server_raw.get("threads",      ServerConfig.threads),
        batch_size  = server_raw.get("batch_size",   ServerConfig.batch_size),
        ubatch_size = server_raw.get("ubatch_size",  ServerConfig.ubatch_size),
    )

    return AppConfig(
        llama_bin     = os.path.expanduser(raw.get("llama_bin",     AppConfig.llama_bin)),
        llama_repo    = os.path.expanduser(raw.get("llama_repo",    AppConfig.llama_repo)),
        hf_cache      = os.path.expanduser(raw.get("hf_cache",      AppConfig.hf_cache)),
        log_file      = raw.get("log_file",      AppConfig.log_file),
        openwebui_url = raw.get("openwebui_url", AppConfig.openwebui_url),
        server        = server,
    )
