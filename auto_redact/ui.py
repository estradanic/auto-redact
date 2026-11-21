"""Terminal progress bars for the CLI (tqdm), shown only on a TTY."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import Callable, Iterator

from tqdm import tqdm


def format_time(seconds: float | int | None) -> str:
    """Format a number of seconds as ``M:SS`` (or ``H:MM:SS`` when needed).

    Used to turn frame counts into video time in progress bars. ``None`` and
    negative values render as ``0:00``.
    """
    if seconds is None:
        seconds = 0
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds < 0 or seconds != seconds:  # NaN guard
        seconds = 0.0
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# Aliases / helpers that other callers (and tests) may use for the same thing.
format_seconds = format_time
format_duration = format_time


def frames_to_time(frames, fps):
    """Convert a frame count to a formatted video time using ``fps``."""
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = 0.0
    if not fps:
        return format_time(0)
    return format_time((frames or 0) / fps)


class _TimeTqdm(tqdm):
    """A ``tqdm`` bar that can carry a ``timestr`` field in its format."""

    @property
    def format_dict(self):  # type: ignore[override]
        d = super().format_dict
        d["timestr"] = getattr(self, "timestr", "")
        return d


ProgressCallback = Callable[[int, int], None]


@contextlib.contextmanager
def progress_bar(
    desc: str,
    unit: str | None = None,
    position: int = 0,
    fps: float | None = None,
) -> Iterator[ProgressCallback]:
    """Yield an ``on_progress(done, total)`` callback driving one bar.

    The bar is hidden automatically when stderr isn't a terminal. When ``fps``
    is given, ``done``/``total`` are frame counts and an extra ``time/total``
    field (``M:SS/total``) is shown alongside the frame counter.
    """
    if desc == "prefilter" and position == 0:
        position = 1  # shown under the (not yet advancing) classify bar
    bar = _TimeTqdm(
        desc=desc,
        unit=unit or ("track" if desc in ("classify", "prefilter") else "frame"),
        file=sys.stderr,
        disable=None,
        dynamic_ncols=True,
        position=position,
        leave=True,
        bar_format=(
            "{desc:<10} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "{timestr} [{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )
    bar.timestr = ""

    def update(done: int, total: int) -> None:
        # Keep the bar's total in sync first so the percentage and the
        # "done/total" counter agree even on the final update.
        if total is not None and total >= 0:
            bar.total = total
        # update() maintains tqdm's smoothed rate; setting bar.n directly
        # leaves it as a cumulative average that includes startup time.
        delta = done - bar.n
        if delta >= 0:
            bar.update(delta)
        else:
            # A late/out-of-order report must not leave the bar behind.
            bar.n = done
        if fps and fps > 0 and total is not None and total >= 0:
            bar.timestr = f"{format_time((done or 0) / fps)}/{format_time((total or 0) / fps)}"
        try:
            bar.refresh()
        except Exception:
            pass
    try:
        yield update
    finally:
        bar.close()


class TqdmLoggingHandler(logging.Handler):
    """Log through ``tqdm.write`` so messages don't tear active bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_logging() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    from . import telemetry
    telemetry.reattach_log()
    # Silence noisy third-party loggers (HTTP/model-loading chatter).
    for name in ("transformers", "huggingface_hub", "httpx", "httpx2", "httpcore", "httpcore2", "urllib3", "filelock"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as tlog
        tlog.set_verbosity_error()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import logging as hlog
        hlog.set_verbosity_error()
    except Exception:
        pass
