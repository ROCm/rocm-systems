"""Tests for the mount-level helpers: _parse_proc_mounts() and
inspect_filesystems() (which entries are considered and how duplicates and
non-block mounts are filtered)."""

# These tests intentionally exercise the script's internal (underscore)
# helpers, so protected-access is expected.
# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument
# pylint: disable=protected-access

import io
import stat as stat_mod
from collections import namedtuple

import pytest

_Stat = namedtuple("stat_result", "st_mode st_rdev")


def _patch_mounts(monkeypatch, ais_check, text):
    def fake_open(path, *_args, **_kwargs):
        assert path == "/proc/mounts"
        return io.StringIO(text)

    monkeypatch.setattr(ais_check, "open", fake_open, raising=False)


def test_parse_decodes_octal_escapes(monkeypatch, ais_check):
    _patch_mounts(
        monkeypatch,
        ais_check,
        "/dev/sda1 /mnt/has\\040space ext4 rw 0 0\n",
    )

    entries = list(ais_check._parse_proc_mounts())

    assert entries == [("/dev/sda1", "/mnt/has space", "ext4")]


def test_parse_skips_short_lines(monkeypatch, ais_check):
    _patch_mounts(monkeypatch, ais_check, "garbage line\n/dev/sda1 /mnt ext4 rw 0 0\n")

    entries = list(ais_check._parse_proc_mounts())

    assert entries == [("/dev/sda1", "/mnt", "ext4")]


def test_parse_read_error_warns(monkeypatch, capsys, ais_check):
    def boom(*_a, **_k):
        raise OSError("nope")

    monkeypatch.setattr(ais_check, "open", boom, raising=False)

    assert not list(ais_check._parse_proc_mounts())
    assert "Unable to read /proc/mounts" in capsys.readouterr().err


@pytest.fixture
def stub_inspect(monkeypatch, ais_check):
    """
    Replace _parse_proc_mounts with a canned list and os.stat with a controller
    so inspect_filesystems' filtering can be tested without touching sysfs.
    _inspect_filesystem is stubbed to echo what it was called with.
    """

    def configure(mounts, block_sources):
        monkeypatch.setattr(ais_check, "_parse_proc_mounts", lambda: iter(mounts))

        def fake_stat(path, *_a, **_k):
            if path in block_sources:
                return _Stat(stat_mod.S_IFBLK, 0)
            return _Stat(stat_mod.S_IFREG, 0)

        monkeypatch.setattr(ais_check.os, "stat", fake_stat)

        calls = []

        def fake_inspect(source, mountpoint, fstype):
            calls.append((source, mountpoint, fstype))
            return {
                "source": source,
                "mountpoint": mountpoint,
                "fstype": fstype,
                "supported": True,
                "reasons": [],
            }

        monkeypatch.setattr(ais_check, "_inspect_filesystem", fake_inspect)
        return calls

    return configure


def test_skips_non_absolute_sources(stub_inspect, ais_check):
    calls = stub_inspect(
        [("tmpfs", "/run", "tmpfs"), ("proc", "/proc", "proc")],
        block_sources=set(),
    )

    results = ais_check.inspect_filesystems()

    assert results == []
    assert calls == []


def test_skips_non_block_absolute_sources(stub_inspect, ais_check):
    # An absolute source that is not a block device (e.g. a bind mount or loop
    # path that stats as a regular file) is filtered out.
    calls = stub_inspect(
        [("/some/file", "/mnt/bind", "ext4")],
        block_sources=set(),
    )

    assert ais_check.inspect_filesystems() == []
    assert calls == []


def test_includes_block_backed_mount(stub_inspect, ais_check):
    stub_inspect(
        [("/dev/nvme0n1p1", "/data", "ext4")],
        block_sources={"/dev/nvme0n1p1"},
    )

    results = ais_check.inspect_filesystems()

    assert len(results) == 1
    assert results[0]["mountpoint"] == "/data"


def test_deduplicates_identical_source_and_mount(stub_inspect, ais_check):
    calls = stub_inspect(
        [
            ("/dev/sdd", "/", "ext4"),
            ("/dev/sdd", "/", "ext4"),
        ],
        block_sources={"/dev/sdd"},
    )

    results = ais_check.inspect_filesystems()

    assert len(results) == 1
    assert len(calls) == 1


def test_same_device_different_mounts_kept(stub_inspect, ais_check):
    stub_inspect(
        [
            ("/dev/sdd", "/", "ext4"),
            ("/dev/sdd", "/mnt/distro", "ext4"),
        ],
        block_sources={"/dev/sdd"},
    )

    results = ais_check.inspect_filesystems()

    assert {r["mountpoint"] for r in results} == {"/", "/mnt/distro"}


def test_stat_error_on_source_is_skipped(monkeypatch, ais_check):
    monkeypatch.setattr(
        ais_check,
        "_parse_proc_mounts",
        lambda: iter([("/dev/gone", "/mnt", "ext4")]),
    )

    def boom(*_a, **_k):
        raise OSError("vanished")

    monkeypatch.setattr(ais_check.os, "stat", boom)

    assert ais_check.inspect_filesystems() == []
