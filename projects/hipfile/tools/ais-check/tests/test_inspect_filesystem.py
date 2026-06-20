"""Tests for _inspect_filesystem(): mapping a mount to a support verdict.

A fake sysfs tree is built under tmp_path and os.stat is patched so that a
source device path maps to a chosen (major, minor) whose sysfs node lives at
``<tmp>/sys/dev/block/<maj>:<min>``. This exercises the real device walk while
keeping everything hermetic.
"""

# These tests intentionally exercise the script's internal _inspect_filesystem
# helper, so protected-access is expected.
# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument
# pylint: disable=protected-access

import os
import stat as stat_mod
from collections import namedtuple

import pytest

# Minimal stat_result stand-in: only the fields _inspect_filesystem reads.
_Stat = namedtuple("stat_result", "st_mode st_rdev")

_DEV = os.makedev(259, 0)  # an arbitrary block device number


@pytest.fixture
def inspect_with_backing(monkeypatch, tmp_path, ais_check, make_sysfs_disk):
    """
    Higher-level fixture: create a backing device tree, make os.stat map a
    source path to a block device, and redirect the script's
    /sys/dev/block/<maj>:<min> base to that tree.

    Returns ``(run, tmp_path, make_sysfs_disk)`` where ``run`` performs the
    inspection and ``make_sysfs_disk`` builds the fake device tree.
    """
    sysblock = tmp_path / "sysblock"
    sysblock.mkdir()
    real_stat = os.stat

    def run(
        *, fstype, backing_dir=None, is_block=True, stat_error=False, source="/dev/x"
    ):
        if backing_dir is not None:
            node = sysblock / f"{os.major(_DEV)}:{os.minor(_DEV)}"
            if not node.exists():
                node.symlink_to(backing_dir, target_is_directory=True)

        def fake_stat(path, *args, **kwargs):
            if path == source:
                if stat_error:
                    raise OSError("stat failed")
                mode = stat_mod.S_IFBLK if is_block else stat_mod.S_IFREG
                return _Stat(mode, _DEV)
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(ais_check.os, "stat", fake_stat)
        monkeypatch.setattr(ais_check, "_SYS_DEV_BLOCK", str(sysblock), raising=False)
        return ais_check._inspect_filesystem(source, "/mnt/x", fstype)

    return run, tmp_path, make_sysfs_disk


def test_supported_nvme_ext4(inspect_with_backing, ais_check):
    run, tmp_path, make_sysfs_disk = inspect_with_backing
    disk = make_sysfs_disk(tmp_path, "nvme0n1")

    result = run(fstype="ext4", backing_dir=disk)

    assert result["supported"] is True
    assert result["reasons"] == []


def test_supported_nvme_xfs(inspect_with_backing, ais_check):
    run, tmp_path, make_sysfs_disk = inspect_with_backing
    disk = make_sysfs_disk(tmp_path, "nvme0n1")

    result = run(fstype="xfs", backing_dir=disk)

    assert result["supported"] is True


def test_bad_fstype_reason(inspect_with_backing, ais_check):
    run, tmp_path, make_sysfs_disk = inspect_with_backing
    disk = make_sysfs_disk(tmp_path, "nvme0n1")

    result = run(fstype="btrfs", backing_dir=disk)

    assert result["supported"] is False
    assert any("not ext4 or xfs" in r for r in result["reasons"])


def test_non_nvme_reason(inspect_with_backing, ais_check):
    run, tmp_path, make_sysfs_disk = inspect_with_backing
    disk = make_sysfs_disk(tmp_path, "sda")

    result = run(fstype="ext4", backing_dir=disk)

    assert result["supported"] is False
    assert any("not on an NVMe drive" in r for r in result["reasons"])
    assert any("sda" in r for r in result["reasons"])


def test_lvm_reason(inspect_with_backing, ais_check):
    run, tmp_path, make_sysfs_disk = inspect_with_backing
    disk = make_sysfs_disk(tmp_path, "nvme0n1")
    lv = make_sysfs_disk(tmp_path, "dm-0", dm_uuid="LVM-z", slaves=[disk])

    result = run(fstype="ext4", backing_dir=lv)

    assert result["supported"] is False
    assert any("uses LVM" in r for r in result["reasons"])


def test_multiple_reasons_accumulate(inspect_with_backing, ais_check):
    run, tmp_path, make_sysfs_disk = inspect_with_backing
    disk = make_sysfs_disk(tmp_path, "sda")
    lv = make_sysfs_disk(tmp_path, "dm-0", dm_uuid="LVM-z", slaves=[disk])

    result = run(fstype="btrfs", backing_dir=lv)

    assert result["supported"] is False
    joined = " ".join(result["reasons"])
    assert "not ext4 or xfs" in joined
    assert "uses LVM" in joined
    assert "not on an NVMe drive" in joined


def test_partition_on_nvme_supported(inspect_with_backing, ais_check):
    run, tmp_path, make_sysfs_disk = inspect_with_backing
    disk = make_sysfs_disk(tmp_path, "nvme0n1")
    part = make_sysfs_disk(disk, "nvme0n1p1", partition=True)

    result = run(fstype="ext4", backing_dir=part)

    assert result["supported"] is True


def test_stat_error_reports_reason(inspect_with_backing, ais_check):
    run, _, _ = inspect_with_backing

    result = run(fstype="ext4", stat_error=True)

    assert result["supported"] is False
    assert any("stat" in r for r in result["reasons"])


def test_non_block_source_reports_reason(inspect_with_backing, ais_check):
    run, _, _ = inspect_with_backing

    result = run(fstype="ext4", is_block=False)

    assert result["supported"] is False
    assert any("block device" in r for r in result["reasons"])


def test_missing_sysfs_node_reports_reason(inspect_with_backing, ais_check):
    run, _, _ = inspect_with_backing

    # is_block True but no backing_dir -> sysfs node absent.
    result = run(fstype="ext4", backing_dir=None)

    assert result["supported"] is False
    assert any("sysfs" in r for r in result["reasons"])
