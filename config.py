"""config.py — Load and validate settings.json with defaults."""

from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass, field


SETTINGS_FILE = Path(__file__).parent / "settings.json"


@dataclass
class ServerConfig:
    host:         str   = "0.0.0.0"
    port:         int   = 1234
    ctx_size:     int   = 131072
    n_gpu_layers: int   = 99
    parallel:     int   = 2
    threads:      int   = 8
    batch_size:   int   = 512
    ubatch_size:  int   = 256
    # Chat / sampling parameters
    temperature:    float = 0.8
    top_p:          float = 0.95
    top_k:          int   = 40
    repeat_penalty: float = 1.1
    # Hardware toggles
    flash_attn: bool = True
    mlock:      bool = False


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
        host          = server_raw.get("host",           ServerConfig.host),
        port          = server_raw.get("port",           ServerConfig.port),
        ctx_size      = server_raw.get("ctx_size",       ServerConfig.ctx_size),
        n_gpu_layers  = server_raw.get("n_gpu_layers",   ServerConfig.n_gpu_layers),
        parallel      = server_raw.get("parallel",       ServerConfig.parallel),
        threads       = server_raw.get("threads",        ServerConfig.threads),
        batch_size    = server_raw.get("batch_size",     ServerConfig.batch_size),
        ubatch_size   = server_raw.get("ubatch_size",    ServerConfig.ubatch_size),
        temperature   = server_raw.get("temperature",    ServerConfig.temperature),
        top_p         = server_raw.get("top_p",          ServerConfig.top_p),
        top_k         = server_raw.get("top_k",          ServerConfig.top_k),
        repeat_penalty= server_raw.get("repeat_penalty", ServerConfig.repeat_penalty),
        flash_attn    = server_raw.get("flash_attn",     ServerConfig.flash_attn),
        mlock         = server_raw.get("mlock",          ServerConfig.mlock),
    )

    return AppConfig(
        llama_bin     = os.path.expanduser(raw.get("llama_bin",     AppConfig.llama_bin)),
        llama_repo    = os.path.expanduser(raw.get("llama_repo",    AppConfig.llama_repo)),
        hf_cache      = os.path.expanduser(raw.get("hf_cache",      AppConfig.hf_cache)),
        log_file      = raw.get("log_file",      AppConfig.log_file),
        openwebui_url = raw.get("openwebui_url", AppConfig.openwebui_url),
        server        = server,
    )
