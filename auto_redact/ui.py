"""Terminal progress bars for the CLI (tqdm), shown only on a TTY."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import Callable, Iterator

from tqdm import tqdm

ProgressCallback = Callable[[int, int], None]


@contextlib.contextmanager
def progress_bar(desc: str, unit: str | None = None, position: int = 0) -> Iterator[ProgressCallback]:
    """Yield an ``on_progress(done, total)`` callback driving one bar.

    The bar is hidden automatically when stderr isn't a terminal.
    """
    if desc == "prefilter" and position == 0:
        position = 1  # shown under the (not yet advancing) classify bar
    bar = tqdm(
        desc=desc,
        unit=unit or ("track" if desc in ("classify", "prefilter") else "frame"),
        file=sys.stderr,
        disable=None,
        dynamic_ncols=True,
        position=position,
        leave=True,
        bar_format="{desc:<10} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}]",
    )

    def update(done: int, total: int) -> None:
        bar.total = total
        # update() maintains tqdm's smoothed rate; setting bar.n directly
        # leaves it as a cumulative average that includes startup time.
        bar.update(done - bar.n)

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
