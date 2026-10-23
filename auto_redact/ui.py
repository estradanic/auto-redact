"""Terminal progress bars for the CLI (tqdm), shown only on a TTY."""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import Callable, Iterator

from tqdm import tqdm

ProgressCallback = Callable[[int, int], None]


@contextlib.contextmanager
def progress_bar(desc: str, unit: str | None = None, position: int = 0) -> Iterator[ProgressCallback]:
    """Yield an ``on_progress(done, total)`` callback driving one bar.

    The bar is hidden automatically when stderr isn't a terminal.
    """
    bar = tqdm(
        desc=desc,
        unit=unit or ("track" if desc == "classify" else "frame"),
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
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
