"""Always-on run instrumentation: a debug log plus a ``stats.json`` per run.

Each CLI invocation writes ``<state>/runs/<timestamp>-<command>/`` containing
``run.log`` (DEBUG-level log of this package, INFO+ of everything else) and
``stats.json`` (environment, arguments, stage timings, per-operation timing
aggregates, resource peaks, and any error). Send those two files when
reporting a problem. Disable with ``--no-stats`` or ``AUTO_REDACT_STATS=0``;
relocate with ``AUTO_REDACT_STATS_DIR``.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

SAMPLE_INTERVAL = 5.0
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(threadName)s %(name)s: %(message)s"

_active: "Run | None" = None


def default_root() -> Path:
    override = os.environ.get("AUTO_REDACT_STATS_DIR")
    if override:
        return Path(override)
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "auto-redact" / "runs"


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return None


def _cuda_info() -> dict | None:
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {
            "used_mb": (total - free) / 1048576,  # whole GPU, incl. other processes (ollama)
            "total_mb": total / 1048576,
            "torch_allocated_mb": torch.cuda.memory_allocated() / 1048576,
            "torch_peak_mb": torch.cuda.max_memory_allocated() / 1048576,
        }
    except Exception:
        return None


def _environment() -> dict:
    env: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ld_library_path_set": bool(os.environ.get("LD_LIBRARY_PATH")),
    }
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            env["mem_total_mb"] = int(handle.readline().split()[1]) // 1024
    except (OSError, ValueError):
        pass
    from importlib import metadata

    for pkg in ("auto-redact", "torch", "transformers", "ultralytics", "opencv-python", "numpy"):
        try:
            env.setdefault("versions", {})[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu"] = {"name": props.name, "total_mb": props.total_memory // 1048576}
    except Exception as error:
        env["torch_error"] = repr(error)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            out = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, timeout=10)
            env["ffmpeg"] = out.stdout.splitlines()[0] if out.stdout else None
        except Exception:
            pass
    return env


class Run:
    def __init__(self, command: str, args: dict, root: Path | None = None) -> None:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = (root or default_root()) / f"{stamp}-{command}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()
        self.stats: dict = {
            "command": command,
            "args": args,
            "started": _dt.datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "stages": [],
            "ops": {},
            "notes": {},
            "peaks": {"rss_mb": 0.0, "gpu_used_mb": 0.0},
            "error": None,
        }
        self._handler: logging.Handler | None = None
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None

    def attach_log(self) -> None:
        if self._handler is None:
            self._handler = logging.FileHandler(self.dir / "run.log", encoding="utf-8")
            self._handler.setFormatter(logging.Formatter(_LOG_FORMAT))
            self._handler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        if self._handler not in root.handlers:
            root.addHandler(self._handler)
        logging.getLogger("auto_redact").setLevel(logging.DEBUG)

    def start(self) -> None:
        self.attach_log()
        log = logging.getLogger("auto_redact.telemetry")
        log.debug("args: %s", json.dumps(self.stats["args"], default=str))
        self.stats["environment"] = _environment()
        log.debug("environment: %s", json.dumps(self.stats["environment"], default=str))
        self._sampler = threading.Thread(target=self._sample_loop, name="telemetry", daemon=True)
        self._sampler.start()

    def _sample(self) -> None:
        rss, cuda = _rss_mb(), _cuda_info()
        peaks = self.stats["peaks"]
        if rss:
            peaks["rss_mb"] = max(peaks["rss_mb"], rss)
        if cuda:
            peaks["gpu_used_mb"] = max(peaks["gpu_used_mb"], cuda["used_mb"])
            peaks["torch_peak_mb"] = cuda["torch_peak_mb"]
        logging.getLogger("auto_redact.telemetry").debug(
            "sample: rss_mb=%s cuda=%s", None if rss is None else round(rss), cuda and {k: round(v) for k, v in cuda.items()}
        )

    def _sample_loop(self) -> None:
        while not self._stop.wait(SAMPLE_INTERVAL):
            try:
                self._sample()
            except Exception:
                pass

    @contextlib.contextmanager
    def stage(self, name: str):
        log = logging.getLogger("auto_redact.telemetry")
        start = time.perf_counter()
        log.debug("stage %s: start", name)
        record = {"name": name, "seconds": None, "ok": False}
        self.stats["stages"].append(record)
        try:
            yield
            record["ok"] = True
        finally:
            record["seconds"] = round(time.perf_counter() - start, 3)
            log.debug("stage %s: %.2fs ok=%s", name, record["seconds"], record["ok"])

    def record_op(self, name: str, seconds: float) -> None:
        with self._lock:
            op = self.stats["ops"].setdefault(name, {"count": 0, "total_s": 0.0, "max_s": 0.0})
            op["count"] += 1
            op["total_s"] += seconds
            op["max_s"] = max(op["max_s"], seconds)

    def note(self, key: str, value) -> None:
        self.stats["notes"][key] = value
        logging.getLogger("auto_redact.telemetry").debug("note %s=%s", key, value)

    def finish(self, exit_code: int | None, exc: BaseException | None = None) -> None:
        self._stop.set()
        try:
            self._sample()
        except Exception:
            pass
        if exc is not None:
            self.stats["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            }
            logging.getLogger("auto_redact.telemetry").error("failed: %s", self.stats["error"]["traceback"])
        self.stats["exit_code"] = exit_code
        self.stats["wall_seconds"] = round(time.perf_counter() - self._t0, 3)
        for op in self.stats["ops"].values():
            op["mean_s"] = round(op["total_s"] / op["count"], 4)
            op["total_s"] = round(op["total_s"], 3)
            op["max_s"] = round(op["max_s"], 3)
        (self.dir / "stats.json").write_text(
            json.dumps(self.stats, indent=2, default=str) + "\n", encoding="utf-8"
        )
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler.close()


def enabled() -> bool:
    return os.environ.get("AUTO_REDACT_STATS", "1").lower() not in ("0", "false", "no", "off")


def begin(command: str, args: dict) -> "Run | None":
    global _active
    if not enabled():
        return None
    try:
        _active = Run(command, args)
        _active.start()
    except Exception as error:  # instrumentation must never break a run
        print(f"auto-redact: stats disabled ({error})", file=sys.stderr)
        _active = None
    return _active


def end(exit_code: int | None, exc: BaseException | None = None) -> Path | None:
    global _active
    run, _active = _active, None
    if run is None:
        return None
    try:
        run.finish(exit_code, exc)
    except Exception:
        return None
    return run.dir


def reattach_log() -> None:
    """Re-add the file handler after ``logging.basicConfig(force=True)``."""
    if _active is not None:
        _active.attach_log()


@contextlib.contextmanager
def stage(name: str):
    if _active is None:
        yield
    else:
        with _active.stage(name):
            yield


@contextlib.contextmanager
def timed(name: str):
    """Accumulate count/total/max wall time for a repeated operation."""
    if _active is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        _active.record_op(name, time.perf_counter() - start)


def note(key: str, value) -> None:
    if _active is not None:
        _active.note(key, value)
