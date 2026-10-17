import pytest

from mtgemb import util


def test_atomic_write_bytes_replaces_content(tmp_path):
    target = tmp_path / "state.bin"
    util.atomic_write_bytes(target, b"first")
    util.atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"second"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("failure_point", ["replace", "fsync"])
def test_atomic_write_failure_keeps_previous_content_and_cleans_temp(
    tmp_path, monkeypatch, failure_point
):
    target = tmp_path / "state.bin"
    target.write_bytes(b"previous")

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    if failure_point == "replace":
        monkeypatch.setattr(util.os, "replace", fail)
    else:
        monkeypatch.setattr(util.os, "fsync", fail)
    with pytest.raises(OSError, match="simulated write failure"):
        util.atomic_write_bytes(target, b"replacement")
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]


def test_sha256_json_ignores_mapping_key_order():
    assert util.sha256_json({"a": 1, "b": {"x": 2, "y": 3}}) == util.sha256_json(
        {"b": {"y": 3, "x": 2}, "a": 1}
    )
