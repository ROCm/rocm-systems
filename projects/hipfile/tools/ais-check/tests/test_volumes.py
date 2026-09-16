# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for the volume-scan logic absorbed from ais-volumes."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument

import pytest


def _mount(ais_check, *, fstype, options="", source="/dev/nvme0n1", mountpoint="/data"):
    return ais_check.Mount("259:0", mountpoint, fstype, source, options)


def test_fs_supported_xfs_always(ais_check):
    assert ais_check.fs_supported(_mount(ais_check, fstype="xfs")) is True


@pytest.mark.parametrize(
    "options,expected",
    [
        ("rw", True),  # no data= option -> ordered default
        ("rw,data=ordered", True),
        ("rw,data=writeback", False),
        ("rw,data=journal", False),
    ],
)
def test_fs_supported_ext4_data_mode(ais_check, options, expected):
    assert (
        ais_check.fs_supported(_mount(ais_check, fstype="ext4", options=options))
        is expected
    )


def test_fs_supported_other_fs_rejected(ais_check):
    assert ais_check.fs_supported(_mount(ais_check, fstype="btrfs")) is False


def test_fstype_label_folds_ext4_journal_mode(ais_check):
    assert ais_check.fstype_label(_mount(ais_check, fstype="ext4")) == "ext4 (ordered)"
    labelled = ais_check.fstype_label(
        _mount(ais_check, fstype="ext4", options="rw,data=writeback")
    )
    assert labelled == "ext4 (writeback)"
    assert ais_check.fstype_label(_mount(ais_check, fstype="xfs")) == "xfs"


def test_collect_skips_pseudo_and_non_block(ais_check, monkeypatch):
    monkeypatch.setattr(ais_check, "backing_storage", lambda devno: ("nvme", "nvme0n1"))
    monkeypatch.setattr(ais_check, "probe_odirect", lambda mp: True)

    mounts = [
        _mount(ais_check, fstype="tmpfs", source="tmpfs", mountpoint="/run"),
        _mount(ais_check, fstype="xfs", source="/dev/nvme0n1", mountpoint="/data"),
    ]
    rows = ais_check.collect(mounts)

    assert [r["mountpoint"] for r in rows] == ["/data"]
    assert rows[0]["capable"] is True


def test_collect_skips_network_mount_before_resolution(ais_check, monkeypatch):
    def _fail_backing(devno):
        raise AssertionError("backing_storage must not run on a non-block mount")

    def _fail_probe(mp):
        raise AssertionError("probe_odirect must not run on a non-block mount")

    monkeypatch.setattr(ais_check, "backing_storage", _fail_backing)
    monkeypatch.setattr(ais_check, "probe_odirect", _fail_probe)

    # NFS export: virtual device (major 0) and a "server:/export" source.
    nfs = ais_check.Mount("0:42", "/mnt/nfs", "nfs4", "server:/export", "rw")
    assert ais_check.collect([nfs]) == []


def test_collect_unsupported_fs_not_probed(ais_check, monkeypatch):
    monkeypatch.setattr(ais_check, "backing_storage", lambda devno: ("nvme", "nvme0n1"))

    def _fail_probe(mp):
        raise AssertionError("probe_odirect must not run on unsupported fs")

    monkeypatch.setattr(ais_check, "probe_odirect", _fail_probe)

    rows = ais_check.collect([_mount(ais_check, fstype="btrfs")])

    assert rows[0]["capable"] is False
    assert rows[0]["odirect"] is None


def test_capable_volumes_aggregates_any(ais_check, monkeypatch):
    monkeypatch.setattr(ais_check, "parse_mountinfo", lambda: [])
    monkeypatch.setattr(
        ais_check,
        "collect",
        lambda mounts: [{"capable": False}, {"capable": True}],
    )
    rows, any_capable = ais_check.capable_volumes()
    assert any_capable is True
    assert len(rows) == 2
