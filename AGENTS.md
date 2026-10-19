# Agent notes

Practical notes for AI agents (and humans) working in this repo, so hard-won
findings don't have to be rediscovered.

## Python: opencode desktop (AppImage) breaks venvs via `LD_LIBRARY_PATH`

**Symptom.** When shell commands run through the opencode desktop app (an
AppImage), direct invocations of a venv interpreter silently ignore the venv:

```console
$ .venv/bin/python -c "import sys; print(sys.prefix, sys.executable)"
/usr /home/nic/Applications/opencode-desktop-linux-x86_64.AppImage
$ .venv/bin/python -m pip
/path/to/AppImage: No module named pip
```

Imports then resolve to **system** site-packages instead of the venv's, so
"module not installed" errors appear for packages that are installed — or,
worse, the wrong version is picked up with no error at all.

**Cause.** The AppImage runtime prepends its mount directories to
`LD_LIBRARY_PATH` (e.g. `/tmp/.mount_XXXXXX/usr/lib`). That directory ships a
bundled `libpython3.14.so`, which the dynamic linker prefers over the system
one. The bundled interpreter's path/config resolution then bypasses the venv's
`pyvenv.cfg`. The mount directory names are random per launch, so the exact
symptoms can vary between runs.

Note this can bite even absolute interpreter paths, and console scripts with
venv shebangs (e.g. `.venv/bin/pip`) have been observed to keep working — do
not rely on that; use the fix below uniformly.

**Fix.** Unset `LD_LIBRARY_PATH` when invoking Python:

```sh
env -u LD_LIBRARY_PATH .venv/bin/python -m pytest
env -u LD_LIBRARY_PATH .venv/bin/python -m pip install <pkg>
```

A fully scrubbed environment also works if needed:

```sh
env -i HOME="$HOME" PATH=/usr/bin:/bin .venv/bin/python -m pytest
```

**Setup.** The project venv lives in the repo root (`./.venv`, system Python
3.14) and has the package installed editable. After pulling changes that add
dependencies, refresh it from the repo root:

```sh
.venv/bin/python -m pip install -e .
```

**Tests.** The test suite needs `ffmpeg`/`ffprobe` on `PATH` (tests generate
synthetic clips on the fly and skip if they are missing) plus `pytest`:

```sh
env -u LD_LIBRARY_PATH .venv/bin/python -m pytest
```
