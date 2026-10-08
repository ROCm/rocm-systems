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


def _fake_topology(ais_check, monkeypatch, prefixes, slaves, start=None):
    """
    Drive stack_over_nvme against an in-memory sysfs tree.

    `prefixes` maps a node path to its dm target prefix (or None for a physical
    leaf); `slaves` maps a node path to the child node paths it is stacked on.
    realpath is stubbed to identity so nodes are addressed by plain strings.

    `start` optionally stands in for the realpath of the mount's own block
    device, so a test can name the root node after its sysfs device (md0,
    dm-0) while still passing a realistic maj:min. The walk classifies MD
    layers by name, unlike dm layers which it reads out of `prefixes`.
    """

    def _realpath(path):
        if start is not None and path.startswith("/sys/dev/block/"):
            return start
        return path

    monkeypatch.setattr(ais_check.os.path, "realpath", _realpath)
    monkeypatch.setattr(ais_check, "dm_uuid_prefix", prefixes.get)
    monkeypatch.setattr(ais_check, "block_slaves", slaves.get)


def test_stack_over_nvme_lvm_direct_on_nvme(ais_check, monkeypatch):
    lv = "/sys/dev/block/253:0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={lv: "lvm", "/dev/nvme0n1p1": None},
        slaves={lv: ["/dev/nvme0n1p1"]},
    )
    assert ais_check.stack_over_nvme("253:0") is True


def test_stack_over_nvme_lvm_spanning_multiple_nvme(ais_check, monkeypatch):
    lv = "/sys/dev/block/253:0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={lv: "lvm", "/dev/nvme0n1p1": None, "/dev/nvme1n1p1": None},
        slaves={lv: ["/dev/nvme0n1p1", "/dev/nvme1n1p1"]},
    )
    assert ais_check.stack_over_nvme("253:0") is True


def test_stack_over_nvme_lvm_nested_lvm(ais_check, monkeypatch):
    lv, inner = "/sys/dev/block/253:1", "/sys/dev/block/253:0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={lv: "lvm", inner: "lvm", "/dev/nvme0n1p1": None},
        slaves={lv: [inner], inner: ["/dev/nvme0n1p1"]},
    )
    assert ais_check.stack_over_nvme("253:1") is True


def test_stack_over_nvme_lvm_raid_lv(ais_check, monkeypatch):
    # A two-member RAID LV: the visible LV is stacked on four sub-LVs (data +
    # metadata per member), each of which is itself an LVM dm node carrying an
    # "LVM-" uuid, and each of those sits on an NVMe partition. The existing
    # walk descends the whole tree, so no dm-raid special case is needed --
    # this test exists to keep it that way.
    lv = "/sys/dev/block/253:6"
    members = {
        "/sys/dev/block/253:2": "/dev/nvme1n1p20",  # rmeta_0
        "/sys/dev/block/253:3": "/dev/nvme1n1p20",  # rimage_0
        "/sys/dev/block/253:4": "/dev/nvme1n1p21",  # rmeta_1
        "/sys/dev/block/253:5": "/dev/nvme1n1p21",  # rimage_1
    }
    prefixes = {lv: "lvm"}
    slaves = {lv: list(members)}
    for sub_lv, pv in members.items():
        prefixes[sub_lv] = "lvm"
        prefixes[pv] = None
        slaves[sub_lv] = [pv]

    _fake_topology(ais_check, monkeypatch, prefixes=prefixes, slaves=slaves)
    assert ais_check.stack_over_nvme("253:6") is True


def test_stack_over_nvme_lvm_raid_lv_with_non_nvme_member_rejected(
    ais_check, monkeypatch
):
    # Every member must qualify. A RAID LV with one member on SCSI cannot serve
    # every read over P2P-DMA, so the volume as a whole does not qualify.
    lv, nvme_member, scsi_member = (
        "/sys/dev/block/253:4",
        "/sys/dev/block/253:2",
        "/sys/dev/block/253:3",
    )
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={
            lv: "lvm",
            nvme_member: "lvm",
            scsi_member: "lvm",
            "/dev/nvme1n1p20": None,
            "/dev/sda1": None,
        },
        slaves={
            lv: [nvme_member, scsi_member],
            nvme_member: ["/dev/nvme1n1p20"],
            scsi_member: ["/dev/sda1"],
        },
    )
    assert ais_check.stack_over_nvme("253:4") is False


def test_stack_over_nvme_lvm_on_scsi_rejected(ais_check, monkeypatch):
    lv = "/sys/dev/block/253:0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={lv: "lvm", "/dev/sda1": None},
        slaves={lv: ["/dev/sda1"]},
    )
    assert ais_check.stack_over_nvme("253:0") is False


def test_stack_over_nvme_lvm_on_mpath_rejected(ais_check, monkeypatch):
    lv, mpath = "/sys/dev/block/253:1", "/sys/dev/block/253:0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={lv: "lvm", mpath: "mpath", "/dev/nvme0n1p1": None},
        slaves={lv: [mpath], mpath: ["/dev/nvme0n1p1"]},
    )
    assert ais_check.stack_over_nvme("253:1") is False


def test_stack_over_nvme_unresolvable_topology(ais_check, monkeypatch):
    lv = "/sys/dev/block/253:0"
    _fake_topology(ais_check, monkeypatch, prefixes={lv: "lvm"}, slaves={})
    assert ais_check.stack_over_nvme("253:0") is None


def test_stack_over_nvme_md_raid_on_nvme(ais_check, monkeypatch):
    # An MD array is not device-mapper, so it carries no dm/uuid; the walk
    # recognizes it by its kernel name and descends into its member devices.
    md = "/sys/devices/virtual/block/md0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={"/dev/nvme0n1p1": None, "/dev/nvme1n1p1": None},
        slaves={md: ["/dev/nvme0n1p1", "/dev/nvme1n1p1"]},
        start=md,
    )
    assert ais_check.stack_over_nvme("9:0") is True


def test_stack_over_nvme_md_raid_with_non_nvme_member_rejected(ais_check, monkeypatch):
    # Reads can be served from any mirror, so one non-NVMe member is enough to
    # disqualify the array.
    md = "/sys/devices/virtual/block/md0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={"/dev/nvme0n1p1": None, "/dev/sda1": None},
        slaves={md: ["/dev/nvme0n1p1", "/dev/sda1"]},
        start=md,
    )
    assert ais_check.stack_over_nvme("9:0") is False


def test_stack_over_nvme_md_raid_partition(ais_check, monkeypatch):
    # A partition of an MD array has no slaves of its own in sysfs: it is a
    # child directory of the array, which is where the member devices hang.
    md = "/sys/devices/virtual/block/md0"
    part = md + "/md0p1"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={"/dev/nvme0n1p1": None},
        slaves={md: ["/dev/nvme0n1p1"]},
        start=part,
    )
    monkeypatch.setattr(ais_check.os.path, "exists", lambda p: p == part + "/partition")
    assert ais_check.stack_over_nvme("259:3") is True


def test_stack_over_nvme_md_raid_partition_real_layout(
    ais_check, monkeypatch, tmp_path
):
    # The same step-up, but against a sysfs-shaped directory tree with the real
    # dm_uuid_prefix/block_slaves/partition probes, so the layout the walk
    # assumes (a partition is a child directory of its disk, carries a
    # "partition" file, and has no slaves of its own) is checked rather than
    # stubbed out.
    real_realpath = ais_check.os.path.realpath

    md = tmp_path / "md0"
    (md / "slaves").mkdir(parents=True)
    part = md / "md0p1"
    part.mkdir()
    (part / "partition").write_text("1\n")
    member = tmp_path / "nvme0n1p1"
    member.mkdir()
    (md / "slaves" / "nvme0n1p1").symlink_to(member)

    monkeypatch.setattr(
        ais_check.os.path,
        "realpath",
        lambda p: str(part) if p == "/sys/dev/block/259:3" else real_realpath(p),
    )
    assert ais_check.stack_over_nvme("259:3") is True


def test_stack_over_nvme_lvm_on_md_raid_on_nvme(ais_check, monkeypatch):
    # LVM and MD are both pass-through layers, so stacking one on the other
    # keeps the direct path.
    lv, md = "/sys/dev/block/253:0", "/dev/md0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={lv: "lvm", md: None, "/dev/nvme0n1p1": None},
        slaves={lv: [md], md: ["/dev/nvme0n1p1"]},
    )
    assert ais_check.stack_over_nvme("253:0") is True


def test_stack_over_nvme_md_raid_on_lvm_on_nvme(ais_check, monkeypatch):
    md, lv = "/sys/devices/virtual/block/md0", "/dev/dm-0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={lv: "lvm", "/dev/nvme0n1p1": None},
        slaves={md: [lv], lv: ["/dev/nvme0n1p1"]},
        start=md,
    )
    assert ais_check.stack_over_nvme("9:0") is True


def test_stack_over_nvme_md_raid_on_mpath_rejected(ais_check, monkeypatch):
    md, mpath = "/sys/devices/virtual/block/md0", "/dev/dm-0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={mpath: "mpath", "/dev/nvme0n1p1": None},
        slaves={md: [mpath], mpath: ["/dev/nvme0n1p1"]},
        start=md,
    )
    assert ais_check.stack_over_nvme("9:0") is False


def test_stack_over_nvme_md_raid_unresolvable_topology(ais_check, monkeypatch):
    md = "/sys/devices/virtual/block/md0"
    _fake_topology(ais_check, monkeypatch, prefixes={}, slaves={}, start=md)
    assert ais_check.stack_over_nvme("9:0") is None


def test_backing_supported_nvme(ais_check):
    assert ais_check.backing_supported("nvme", "259:0") is True


@pytest.mark.parametrize("backing,devno", [("lvm", "253:0"), ("md", "9:0")])
def test_backing_supported_stacked_layers_delegate(
    ais_check, monkeypatch, backing, devno
):
    monkeypatch.setattr(ais_check, "stack_over_nvme", lambda devno: True)
    assert ais_check.backing_supported(backing, devno) is True
    monkeypatch.setattr(ais_check, "stack_over_nvme", lambda devno: False)
    assert ais_check.backing_supported(backing, devno) is False


def test_backing_supported_other_layers_rejected(ais_check):
    assert ais_check.backing_supported("mpath", "253:0") is False
    assert ais_check.backing_supported("crypt", "253:0") is False
    assert ais_check.backing_supported("loop", "7:0") is False


def test_backing_supported_unknown_is_none(ais_check):
    assert ais_check.backing_supported(None, "0:0") is None


def test_collect_lvm_on_nvme_capable(ais_check, monkeypatch):
    monkeypatch.setattr(ais_check, "backing_storage", lambda devno: ("lvm", "dm-0"))
    monkeypatch.setattr(ais_check, "stack_over_nvme", lambda devno: True)
    monkeypatch.setattr(ais_check, "probe_odirect", lambda mp: True)

    rows = ais_check.collect(
        [_mount(ais_check, fstype="ext4", source="/dev/mapper/vg-lv")]
    )

    assert rows[0]["backing"] == "lvm"
    assert rows[0]["capable"] is True


def test_collect_lvm_not_on_nvme_not_capable(ais_check, monkeypatch):
    monkeypatch.setattr(ais_check, "backing_storage", lambda devno: ("lvm", "dm-0"))
    monkeypatch.setattr(ais_check, "stack_over_nvme", lambda devno: False)
    monkeypatch.setattr(ais_check, "probe_odirect", lambda mp: True)

    rows = ais_check.collect(
        [_mount(ais_check, fstype="ext4", source="/dev/mapper/vg-lv")]
    )

    assert rows[0]["capable"] is False


def test_collect_md_raid_on_nvme_capable(ais_check, monkeypatch):
    monkeypatch.setattr(ais_check, "backing_storage", lambda devno: ("md", "md0"))
    monkeypatch.setattr(ais_check, "stack_over_nvme", lambda devno: True)
    monkeypatch.setattr(ais_check, "probe_odirect", lambda mp: True)

    rows = ais_check.collect([_mount(ais_check, fstype="xfs", source="/dev/md0")])

    assert rows[0]["backing"] == "md"
    assert rows[0]["capable"] is True


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
