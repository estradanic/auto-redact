"""Play a video in VLC while it is still being produced.

VLC stops at the end of a file it has opened, so a growing file can't be
played directly. Instead VLC reads MPEG-TS from a pipe (``fd://0``), which has
no end until we close it: when we have nothing new, VLC buffers, and it resumes
as soon as more bytes arrive. A feeder thread tails the growing ``.ts`` and
writes whatever is new into the pipe.

VLC is told the container explicitly: on a pipe it has no file name to go by
and otherwise guesses an MPEG program stream, which mangles the audio.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# A pipe can't be seeked, so the jump hotkeys (e.g. the arrow keys) are unbound.
_NO_SEEK_KEYS = [
    f"--key-jump{sign}{size}=Unset"
    for size in ("extrashort", "short", "medium", "long")
    for sign in ("-", "+")
]

TS_PACKET = 188
_READ_SIZE = TS_PACKET * 512
_POLL = 0.1


class VlcError(RuntimeError):
    pass


# Variables that make a child VLC load the wrong libraries. The desktop app's
# AppImage sets LD_LIBRARY_PATH, and importing OpenCV (cv2) points the Qt
# variables at its own bundled Qt plugins, which abort VLC's Qt interface.
_FOREIGN_ENV = {"LD_LIBRARY_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR", "QT_PLUGIN_PATH"}


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _FOREIGN_ENV}


def require_vlc() -> str:
    path = shutil.which("vlc")
    if path is None:
        raise VlcError("vlc was not found on PATH")
    return path


def play_file(path: Path) -> None:
    """Open a finished video in VLC and leave it running."""
    subprocess.Popen(
        [require_vlc(), "--play-and-exit", str(path)],
        env=_env(), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


class LivePlayer:
    """VLC fed from a growing MPEG-TS file."""

    def __init__(self, source: Path, progress: Callable[[str], None] = lambda m: None,
                 vlc_args: list[str] | None = None) -> None:
        self.source = Path(source)
        self.progress = progress
        self.vlc_args = vlc_args if vlc_args is not None else []
        self._done = threading.Event()
        self._opened = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._stderr = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            [require_vlc(), "--demux", "avformat", "--avformat-format", "mpegts",
             "--play-and-exit", *_NO_SEEK_KEYS, *self.vlc_args, "fd://0"],
            env=_env(), stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=self._stderr,
        )
        self._thread = threading.Thread(target=self._feed, name="vlc-feed", daemon=True)
        self._thread.start()
        self.progress(f"[vlc] playing {self.source.name} as it is produced")

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def finish(self, timeout: float = 5.0) -> None:
        """Declare that nothing more will be appended to the source.

        Waits briefly until the feeder has the file open, so the caller can
        delete it afterwards: an open file stays readable once unlinked.
        """
        self._done.set()
        if self.alive and self._thread is not None and self._thread.is_alive():
            self._opened.wait(timeout)

    def wait(self) -> None:
        """Block until everything has been fed and VLC has exited."""
        if self._thread is not None:
            self._thread.join()
        if self._proc is not None:
            self._proc.wait()

    def stop(self) -> None:
        self._done.set()
        if self.alive:
            self._proc.terminate()

    def _report_failure(self) -> None:
        """Say so when VLC died on its own; otherwise nothing would show."""
        code = self._proc.poll()
        if code is None or self._stderr is None:
            return
        if code == 0:
            if not self._done.is_set():
                self.progress("[vlc] VLC closed early; live playback stopped (processing continues)")
            return
        try:
            self._stderr.seek(0)
            tail = self._stderr.read().decode("utf-8", errors="replace").strip().splitlines()[-5:]
        except (OSError, ValueError):
            tail = []
        self.progress(f"[vlc] VLC exited unexpectedly (code {code}); no live playback")
        for line in tail:
            self.progress(f"[vlc]   {line}")

    def _feed(self) -> None:
        stdin = self._proc.stdin
        handle = None
        pending = b""
        try:
            while self.alive:
                finished = self._done.is_set()
                if handle is None:
                    try:
                        handle = open(self.source, "rb")
                    except FileNotFoundError:
                        if finished:
                            self._opened.set()
                            return
                        self._done.wait(_POLL)
                        continue
                    self._opened.set()
                data = handle.read(_READ_SIZE)
                if not data:
                    if finished:
                        return
                    self._done.wait(_POLL)
                    continue
                pending += data
                whole = len(pending) - len(pending) % TS_PACKET
                if whole:
                    stdin.write(pending[:whole])
                    stdin.flush()
                    pending = pending[whole:]
        except (BrokenPipeError, ValueError, OSError):
            logger.debug("vlc closed its input", exc_info=True)
        finally:
            self._opened.set()
            self._report_failure()
            if handle is not None:
                handle.close()
            try:
                stdin.close()
            except (BrokenPipeError, ValueError, OSError):
                pass
