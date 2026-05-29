"""llama_server.py — llama-server process lifecycle manager.

Responsibilities:
  - Start / stop llama-server
  - Pipe stdout to a rotating log file
  - Poll for readiness
  - Expose status and metrics
  - No sandbox-exec (per spec)
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import requests

from config import AppConfig

log = logging.getLogger(__name__)

LOG_MAX_BYTES = 1 * 1024 * 1024  # 1 MB per file
LOG_BACKUPS   = 10
READY_TIMEOUT = 120              # seconds to wait for /v1/models


class LlamaServer:
    """Manages a single llama-server process."""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg         = cfg
        self._process:    subprocess.Popen | None = None
        self._start_time: datetime | None         = None
        self._model_path: str | None              = None
        self._log_thread: threading.Thread | None = None
        self.on_log_line: callable | None         = None  # callback(line: str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        if self._process and self._process.poll() is None:
            return True
        return self._find_pid() is not None

    def start(self, model_path: str, draft_model_path: str | None = None) -> bool:
        """Start llama-server with the given model path. Returns True on success."""
        if self.is_running:
            return True

        cmd = self._build_command(model_path, draft_model_path)
        log.info("Starting llama-server: %s", " ".join(cmd))

        try:
            self._cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self._cfg.bin_path.parent),
            )
            self._start_time  = datetime.now(timezone.utc)
            self._model_path  = model_path
            self._log_thread  = threading.Thread(
                target=self._pipe_log,
                args=(self._process.stdout,),
                daemon=True,
                name="llama-log",
            )
            self._log_thread.start()
        except Exception as exc:
            log.error("Failed to launch llama-server: %s", exc)
            return False

        return self._wait_ready()

    def stop(self) -> None:
        """Stop llama-server gracefully, then forcefully if needed."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)

        # Also terminate any externally started instance on the same port
        pid = self._find_pid()
        if pid:
            try:
                p = psutil.Process(pid)
                p.terminate()
                # Wait for external process to actually die — not fire-and-forget
                try:
                    p.wait(timeout=5)
                except psutil.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        self._process    = None
        self._start_time = None
        self._model_path = None

    def status(self) -> dict:
        running = self.is_running
        uptime  = None
        if running and self._start_time:
            uptime = int((datetime.now(timezone.utc) - self._start_time).total_seconds())
        return {
            "running":        running,
            "pid":            self._active_pid(),
            "model":          self._model_path,
            "uptime_seconds": uptime,
        }

    def metrics(self) -> dict:
        try:
            url  = f"http://localhost:{self._cfg.server.port}/metrics"
            resp = requests.get(url, timeout=2)
            resp.raise_for_status()
            for line in resp.text.splitlines():
                if line.startswith("llamacpp:requests_processing "):
                    return {"requests_processing": int(float(line.split()[1]))}
        except Exception:
            pass
        return {"requests_processing": 0}

    def ram(self) -> dict:
        mem    = psutil.virtual_memory()
        system = {
            "used_gb":  round(mem.used  / (1024 ** 3), 1),
            "total_gb": round(mem.total / (1024 ** 3), 1),
        }
        llama = {"used_gb": 0.0}
        pid   = self._active_pid()
        if pid:
            try:
                llama["used_gb"] = round(
                    psutil.Process(pid).memory_info().rss / (1024 ** 3), 1
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return {"system": system, "llama": llama}

    def log_tail(self, n: int = 50) -> list[str]:
        log_path = self._cfg.log_path
        if not log_path.exists():
            return []
        return log_path.read_text(errors="replace").splitlines()[-n:]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_command(self, model_path: str, draft_model_path: str | None = None) -> list[str]:
        s = self._cfg.server
        cmd = [
            str(self._cfg.bin_path),
            "--host",           s.host,
            "--port",           str(s.port),
            "-m",               model_path,
            "--ctx-size",       str(s.ctx_size),
            "--n-gpu-layers",   str(s.n_gpu_layers),
            "--parallel",       str(s.parallel),
            "--threads",        str(s.threads),
            "--batch-size",     str(s.batch_size),
            "--ubatch-size",    str(s.ubatch_size),
            "--temp",           str(s.temperature),
            "--top-p",          str(s.top_p),
            "--top-k",          str(s.top_k),
            "--repeat-penalty", str(s.repeat_penalty),
            "--reasoning",      "off",
            "--metrics",
        ]
        if s.flash_attn:
            cmd += ["--flash-attn", "on"]
        if s.mlock:
            cmd.append("--mlock")
        if draft_model_path:
            cmd += ["-md", draft_model_path]
        return cmd

    def _wait_ready(self) -> bool:
        url      = f"http://localhost:{self._cfg.server.port}/v1/models"
        deadline = time.time() + READY_TIMEOUT
        while time.time() < deadline:
            try:
                if requests.get(url, timeout=2).status_code == 200:
                    return True
            except Exception:
                pass
            if self._process and self._process.poll() is not None:
                return False
            time.sleep(2)
        return False

    def _find_pid(self) -> int | None:
        """Find llama-server PID listening on our port. Uses lsof (no root needed on macOS)."""
        port = self._cfg.server.port
        try:
            result = subprocess.run(
                ["lsof", "-iTCP:%d" % port, "-sTCP:LISTEN", "-t"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().splitlines():
                pid = int(line.strip())
                try:
                    name = psutil.Process(pid).name().lower()
                    if "llama" in name:
                        return pid
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as exc:
            log.debug("_find_pid error: %s", exc)
        return None

    def _active_pid(self) -> int | None:
        if self._process and self._process.poll() is None:
            return self._process.pid
        return self._find_pid()

    def _pipe_log(self, pipe) -> None:
        log_path = self._cfg.log_path
        fh = open(log_path, "ab")
        try:
            for raw in pipe:
                try:
                    fh.write(raw)
                    fh.flush()
                    if fh.tell() >= LOG_MAX_BYTES:
                        fh.close()
                        self._rotate_log(log_path)
                        fh = open(log_path, "ab")
                    # Push line to subscribers
                    line = raw.decode(errors="replace").rstrip()
                    if line and self.on_log_line:
                        self.on_log_line(line)
                except Exception:
                    pass
        finally:
            try:
                fh.close()
            except Exception:
                pass

    @staticmethod
    def _rotate_log(log_path: Path) -> None:
        for i in range(LOG_BACKUPS, 0, -1):
            src = Path(f"{log_path}.{i}")
            if src.exists():
                if i == LOG_BACKUPS:
                    src.unlink()
                else:
                    src.rename(f"{log_path}.{i + 1}")
        if log_path.exists():
            log_path.rename(f"{log_path}.1")
