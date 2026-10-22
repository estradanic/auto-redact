# Agent notes

Practical notes for AI agents (and humans) working in this repo.

## Python: the desktop app breaks venvs via `LD_LIBRARY_PATH`

When shell commands run through the opencode desktop app (an AppImage), direct
invocations of a venv interpreter silently ignore the venv. Unset
`LD_LIBRARY_PATH` when invoking Python:

```sh
env -u LD_LIBRARY_PATH .venv/bin/python -m pytest
env -u LD_LIBRARY_PATH .venv/bin/python -m pip install <pkg>
```

## Setup

The project venv lives in the repo root (`./.venv`). After pulling changes that
add dependencies, refresh it from the repo root:

```sh
.venv/bin/python -m pip install -e .
```

## Tests

The test suite needs `ffmpeg`/`ffprobe` on `PATH` (tests generate synthetic
clips on the fly and skip if they are missing) plus `pytest`:

```sh
env -u LD_LIBRARY_PATH .venv/bin/python -m pytest
```