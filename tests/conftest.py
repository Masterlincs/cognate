import pytest

from mtgemb import util


@pytest.fixture(autouse=True)
def _isolated_logs(tmp_path, monkeypatch):
    """log_event writes via util.RUNS: keep test events out of the real runs/*.jsonl."""
    monkeypatch.setattr(util, "RUNS", tmp_path / "runs-log")
