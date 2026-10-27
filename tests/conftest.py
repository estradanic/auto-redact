import os

import pytest

os.environ.setdefault("AUTO_REDACT_STATS", "0")


@pytest.fixture(autouse=True)
def isolated_state_dir(tmp_path_factory, monkeypatch):
    """Never let a test write run data into the real ~/.local/state."""
    monkeypatch.setenv("AUTO_REDACT_STATE_DIR", str(tmp_path_factory.mktemp("state")))
