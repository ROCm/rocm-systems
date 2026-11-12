# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for the volume-scan logic absorbed from ais-volumes."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument

import errno
from collections import namedtuple
from unittest.mock import Mock, mock_open

import pytest

_Completed = namedtuple("CompletedProcess", "returncode stdout")


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


@pytest.mark.parametrize("fstype", ["nfs", "nfs4"])
@pytest.mark.parametrize(
    "options,expected",
    [
        ("rw,proto=rdma", True),
        ("ro,proto=rdma6,vers=4.2", True),
        ("rw,proto=tcp", False),
        ("rw,proto=tcp6", False),
        ("rw,proto=udp", False),
        ("rw,proto=udp6", False),
        ("rw", False),
        ("rw,proto", False),
        ("rw,proto=", False),
        ("rw,mountproto=rdma", False),
        ("rw,proto=rdma-extra", False),
        ("rw,proto=rdma6-extra", False),
    ],
)
def test_fs_supported_nfs_transport(ais_check, fstype, options, expected):
    mount = _mount(ais_check, fstype=fstype, options=options)
    assert ais_check.fs_supported(mount) is expected


@pytest.mark.parametrize("proto", ["rdma", "rdma6", "tcp"])
def test_nfs_transport_from_mountinfo_super_options(ais_check, monkeypatch, proto):
    mountinfo = (
        "42 25 0:42 / /mnt/nfs rw,relatime shared:7 - "
        f"nfs4 server:/export rw,vers=4.2,proto={proto},addr=192.0.2.1\n"
    )
    mocked_open = mock_open(read_data=mountinfo)
    monkeypatch.setattr("builtins.open", mocked_open)

    mounts = ais_check.parse_mountinfo()

    mocked_open.assert_called_once_with("/proc/self/mountinfo", "r", encoding="utf-8")
    assert len(mounts) == 1
    assert ais_check.fs_supported(mounts[0]) is (proto in ("rdma", "rdma6"))


def test_fstype_label_folds_ext4_journal_mode(ais_check):
    assert ais_check.fstype_label(_mount(ais_check, fstype="ext4")) == "ext4 (ordered)"
    labelled = ais_check.fstype_label(
        _mount(ais_check, fstype="ext4", options="rw,data=writeback")
    )
    assert labelled == "ext4 (writeback)"
    assert ais_check.fstype_label(_mount(ais_check, fstype="xfs")) == "xfs"


def _fake_topology(  # pylint: disable=too-many-arguments
    ais_check,
    monkeypatch,
    prefixes,
    slaves,
    *,
    start=None,
    levels=None,
    names=None,
    raid_types=None,
):
    """
    Drive stack_over_nvme against an in-memory sysfs tree.

    `prefixes` maps a node path to its dm target prefix (or None for a physical
    leaf); `slaves` maps a node path to the child node paths it is stacked on;
    `levels` maps an MD node path to its personality (None for a node whose
    md/level cannot be read). realpath is stubbed to identity so nodes are
    addressed by plain strings.

    Device-mapper nodes answer to their node path as their dm name unless
    `names` maps them to one, and report no dm-raid segments unless
    `raid_types` maps that name to a set of them (or to None, standing in for a
    table that can't be read).

    `start` optionally stands in for the realpath of the mount's own block
    device, so a test can name the root node after its sysfs device (md0,
    dm-0) -- which is how the walk recognizes an MD layer -- while still
    passing a realistic maj:min.
    """

    def _realpath(path):
        if start is not None and path.startswith("/sys/dev/block/"):
            return start
        return path

    monkeypatch.setattr(ais_check.os.path, "realpath", _realpath)
    monkeypatch.setattr(ais_check, "dm_uuid_prefix", prefixes.get)
    monkeypatch.setattr(ais_check, "block_slaves", slaves.get)
    monkeypatch.setattr(ais_check, "md_level", (levels or {}).get)
    monkeypatch.setattr(
        ais_check, "dm_name", lambda node: (names or {}).get(node, node)
    )
    monkeypatch.setattr(
        ais_check,
        "dm_raid_segment_types",
        lambda name: (raid_types or {}).get(name, set()),
    )


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


def _raid_lv_topology(members):
    """
    Build the (prefixes, slaves, names) of a RAID LV for _fake_topology.

    `members` maps a member index to the physical volume its sub-LVs sit on.
    The visible LV is "/sys/dev/block/253:0" ("vg-lv"), stacked on a data and a
    metadata sub-LV per member, each of which is itself an LVM dm node on that
    PV.
    """
    lv = "/sys/dev/block/253:0"
    prefixes, slaves, names = {lv: "lvm"}, {lv: []}, {lv: "vg-lv"}
    minor = 1
    for index, pv in sorted(members.items()):
        for role in ("rmeta", "rimage"):
            sub_lv = f"/sys/dev/block/253:{minor}"
            minor += 1
            prefixes[sub_lv] = "lvm"
            names[sub_lv] = f"vg-lv_{role}_{index}"
            slaves[sub_lv] = [pv]
            slaves[lv].append(sub_lv)
        prefixes[pv] = None
    return prefixes, slaves, names


@pytest.mark.parametrize("raid_type", ["raid0", "raid1", "raid10", "raid10_near"])
def test_stack_over_nvme_lvm_raid_lv_passthrough_types(
    ais_check, monkeypatch, raid_type
):
    # A two-member RAID LV: the walk descends the sub-LV tree as it does for a
    # linear LV, and the dm-raid target type decides the layer's own verdict.
    prefixes, slaves, names = _raid_lv_topology(
        {0: "/dev/nvme1n1p20", 1: "/dev/nvme1n1p21"}
    )
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes=prefixes,
        slaves=slaves,
        names=names,
        raid_types={"vg-lv": {raid_type}},
    )
    assert ais_check.stack_over_nvme("253:0") is True


@pytest.mark.parametrize("raid_type", ["raid4", "raid5_ls", "raid6_nr"])
def test_stack_over_nvme_lvm_parity_raid_lv_rejected(ais_check, monkeypatch, raid_type):
    # A parity RAID LV is rejected like a parity MD array: same personality
    # code, reached through dm-raid.
    prefixes, slaves, names = _raid_lv_topology(
        {0: "/dev/nvme1n1p20", 1: "/dev/nvme1n1p21", 2: "/dev/nvme1n1p22"}
    )
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes=prefixes,
        slaves=slaves,
        names=names,
        raid_types={"vg-lv": {raid_type}},
    )
    assert ais_check.stack_over_nvme("253:0") is False


def test_stack_over_nvme_lvm_raid_lv_unreadable_table_unverified(
    ais_check, monkeypatch
):
    # Without the dm table the RAID level is unknown. The rimage sub-LVs give
    # the LV away as a RAID LV, so the verdict is unverified rather than a yes.
    prefixes, slaves, names = _raid_lv_topology({0: "/dev/nvme1n1p20"})
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes=prefixes,
        slaves=slaves,
        names=names,
        raid_types={"vg-lv": None},
    )
    assert ais_check.stack_over_nvme("253:0") is None


def test_stack_over_nvme_plain_lv_unreadable_table_still_capable(
    ais_check, monkeypatch
):
    # A linear LV has no rimage sub-LVs, so it needs no dm table to classify
    # and keeps its verdict when dmsetup is unavailable (an unprivileged run).
    lv = "/sys/dev/block/253:0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={lv: "lvm", "/dev/nvme0n1p1": None},
        slaves={lv: ["/dev/nvme0n1p1"]},
        names={lv: "vg-lv"},
        raid_types={"vg-lv": None},
    )
    assert ais_check.stack_over_nvme("253:0") is True


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


@pytest.mark.parametrize("level", ["linear", "raid0", "raid1", "raid10"])
def test_stack_over_nvme_md_passthrough_levels_on_nvme(ais_check, monkeypatch, level):
    # An MD array is not device-mapper, so it has no entry in `prefixes`; the
    # walk recognizes it by its kernel name and descends into its members.
    md = "/sys/devices/virtual/block/md0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={"/dev/nvme0n1p1": None, "/dev/nvme1n1p1": None},
        slaves={md: ["/dev/nvme0n1p1", "/dev/nvme1n1p1"]},
        start=md,
        levels={md: level},
    )
    assert ais_check.stack_over_nvme("9:0") is True


@pytest.mark.parametrize("level", ["raid4", "raid5", "raid6"])
def test_stack_over_nvme_md_parity_levels_rejected(ais_check, monkeypatch, level):
    # All-NVMe members do not save a parity array.
    md = "/sys/devices/virtual/block/md0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={"/dev/nvme0n1p1": None, "/dev/nvme1n1p1": None},
        slaves={md: ["/dev/nvme0n1p1", "/dev/nvme1n1p1"]},
        start=md,
        levels={md: level},
    )
    assert ais_check.stack_over_nvme("9:0") is False


def test_stack_over_nvme_md_unknown_level_rejected(ais_check, monkeypatch):
    # Anything outside the validated set (a test personality here) is treated
    # the same as a parity array.
    md = "/sys/devices/virtual/block/md0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={"/dev/nvme0n1p1": None},
        slaves={md: ["/dev/nvme0n1p1"]},
        start=md,
        levels={md: "faulty"},
    )
    assert ais_check.stack_over_nvme("9:0") is False


def test_stack_over_nvme_md_unreadable_level_unresolvable(ais_check, monkeypatch):
    md = "/sys/devices/virtual/block/md0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={"/dev/nvme0n1p1": None},
        slaves={md: ["/dev/nvme0n1p1"]},
        start=md,
        levels={},
    )
    assert ais_check.stack_over_nvme("9:0") is None


def test_stack_over_nvme_md_raid_with_non_nvme_member_rejected(ais_check, monkeypatch):
    # Every member has to qualify -- a mirror can serve a read from any copy, a
    # stripe places each region on a particular member -- so one non-NVMe
    # member disqualifies the array.
    md = "/sys/devices/virtual/block/md0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={"/dev/nvme0n1p1": None, "/dev/sda1": None},
        slaves={md: ["/dev/nvme0n1p1", "/dev/sda1"]},
        start=md,
        levels={md: "raid1"},
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
        levels={md: "raid0"},
    )
    monkeypatch.setattr(ais_check.os.path, "exists", lambda p: p == part + "/partition")
    assert ais_check.stack_over_nvme("259:3") is True


def test_stack_over_nvme_md_raid_partition_real_layout(
    ais_check, monkeypatch, tmp_path
):
    # The same step-up against a sysfs-shaped directory tree with the real
    # probes, so the layout the walk assumes -- a partition is a child
    # directory of its disk, carries a "partition" file, and has no slaves or
    # md/level of its own -- is checked rather than stubbed out.
    real_realpath = ais_check.os.path.realpath

    md = tmp_path / "md0"
    (md / "slaves").mkdir(parents=True)
    (md / "md").mkdir()
    (md / "md" / "level").write_text("raid10\n")
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
    lv, md = "/sys/dev/block/253:0", "/dev/md0"
    _fake_topology(
        ais_check,
        monkeypatch,
        prefixes={lv: "lvm", md: None, "/dev/nvme0n1p1": None},
        slaves={lv: [md], md: ["/dev/nvme0n1p1"]},
        levels={md: "raid1"},
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
        levels={md: "raid0"},
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
        levels={md: "raid1"},
    )
    assert ais_check.stack_over_nvme("9:0") is False


def test_stack_over_nvme_md_raid_unresolvable_topology(ais_check, monkeypatch):
    md = "/sys/devices/virtual/block/md0"
    _fake_topology(
        ais_check, monkeypatch, prefixes={}, slaves={}, start=md, levels={md: "raid1"}
    )
    assert ais_check.stack_over_nvme("9:0") is None


def _patch_dmsetup(monkeypatch, ais_check, *, stdout="", returncode=0, exc=None):
    def fake_run(*_args, **_kwargs):
        if exc is not None:
            raise exc
        return _Completed(returncode, stdout)

    monkeypatch.setattr(ais_check.subprocess, "run", fake_run)


def test_dm_raid_segment_types_parses_table(ais_check, monkeypatch):
    _patch_dmsetup(
        monkeypatch,
        ais_check,
        stdout=(
            "0 2097152 raid raid5_ls 3 128 region_size 1024 - 253:2 253:3 253:4 253:5\n"
            "2097152 2097152 raid raid1 3 0 region_size 1024 - 253:6 253:7\n"
        ),
    )
    assert ais_check.dm_raid_segment_types("vg-lv") == {"raid5_ls", "raid1"}


def test_dm_raid_segment_types_empty_for_plain_lv(ais_check, monkeypatch):
    _patch_dmsetup(monkeypatch, ais_check, stdout="")
    assert ais_check.dm_raid_segment_types("vg-lv") == set()


def test_dm_raid_segment_types_unreadable_table(ais_check, monkeypatch):
    _patch_dmsetup(monkeypatch, ais_check, returncode=1, stdout="")
    assert ais_check.dm_raid_segment_types("vg-lv") is None


def test_dm_raid_segment_types_dmsetup_missing(ais_check, monkeypatch):
    _patch_dmsetup(monkeypatch, ais_check, exc=FileNotFoundError())
    assert ais_check.dm_raid_segment_types("vg-lv") is None


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


@pytest.mark.parametrize("fstype", ["nfs", "nfs4"])
@pytest.mark.parametrize(
    "source_options_capable",
    [
        ("server:/export", "rw,proto=tcp", False),
        ("[2001:db8::1]:/export", "ro,proto=tcp6", False),
        ("[2001:db8::1]:/export", "ro", False),
    ],
)
def test_collect_lists_non_rdma_nfs_without_probing(
    ais_check, monkeypatch, capsys, fstype, source_options_capable
):
    def fail_probe(*_args):
        raise AssertionError("NFS listing must not resolve block devices or probe I/O")

    monkeypatch.setattr(ais_check, "backing_storage", fail_probe)
    monkeypatch.setattr(ais_check, "probe_odirect", fail_probe)
    monkeypatch.setattr(ais_check.subprocess, "run", fail_probe)
    source, options, capable = source_options_capable
    mount = ais_check.Mount("0:42", "/mnt/nfs", fstype, source, options)
    monkeypatch.setattr(ais_check, "parse_mountinfo", lambda: [mount])

    rows, any_capable = ais_check.capable_volumes()

    assert rows == [
        {
            "mountpoint": "/mnt/nfs",
            "fstype": fstype,
            "device": source,
            "backing": "nfs",
            "odirect": None,
            "capable": capable,
        }
    ]
    assert any_capable is False
    ais_check.print_volume_table(rows)
    assert capsys.readouterr().out.splitlines()[1].split() == [
        "/mnt/nfs",
        fstype,
        source,
        "nfs",
        "unverified",
        "no" if capable is False else "unverified",
    ]


def _assert_probe_io(ais_check, probe_io, error):
    probe_io.open.assert_called_once()
    path, flags, mode = probe_io.open.call_args.args
    assert path.startswith("/mnt/nfs/.ais-odirect-")
    assert flags == (
        ais_check.os.O_RDWR
        | ais_check.os.O_CREAT
        | ais_check.os.O_EXCL
        | ais_check.os.O_CLOEXEC
        | ais_check.os.O_DIRECT
    )
    assert mode == 0o600
    if error is None:
        probe_io.close.assert_called_once_with(42)
    else:
        probe_io.close.assert_not_called()
    probe_io.unlink.assert_called_once_with(path)


@pytest.mark.parametrize(
    "fstype_proto",
    [(fstype, proto) for fstype in ("nfs", "nfs4") for proto in ("rdma", "rdma6")],
)
@pytest.mark.parametrize(
    "probe_result",
    [
        (None, True, None),
        (errno.EINVAL, False, False),
        (errno.EACCES, None, None),
        (errno.EROFS, None, None),
        (errno.ENOSPC, None, None),
    ],
)
def test_collect_probes_rdma_nfs(
    ais_check, monkeypatch, capsys, fstype_proto, probe_result
):
    fstype, proto = fstype_proto
    error, odirect, capable = probe_result
    mount = ais_check.Mount(
        "0:42", "/mnt/nfs", fstype, "server:/export", f"rw,proto={proto}"
    )
    monkeypatch.setattr(ais_check, "parse_mountinfo", lambda: [mount])

    def fail_resolution(*_args):
        raise AssertionError("NFS probing must not resolve block devices or run tools")

    monkeypatch.setattr(ais_check, "backing_storage", fail_resolution)
    monkeypatch.setattr(ais_check.subprocess, "run", fail_resolution)
    probe_io = Mock()
    probe_io.open.return_value = 42
    probe_io.open.side_effect = OSError(error, "probe failed") if error else None
    monkeypatch.setattr(ais_check.os, "open", probe_io.open)
    monkeypatch.setattr(ais_check.os, "close", probe_io.close)
    monkeypatch.setattr(ais_check.os, "unlink", probe_io.unlink)

    rows, any_capable = ais_check.capable_volumes()

    _assert_probe_io(ais_check, probe_io, error)
    assert rows[0]["odirect"] is odirect
    assert rows[0]["capable"] is capable
    assert any_capable is False
    ais_check.print_volume_table(rows)
    assert capsys.readouterr().out.splitlines()[1].split()[-2:] == [
        {True: "yes", False: "no", None: "unverified"}[odirect],
        "no" if capable is False else "unverified",
    ]


@pytest.mark.parametrize(
    "fstype,source", [("cifs", "//server/share"), ("fuse.sshfs", "server:/export")]
)
def test_collect_skips_other_network_mounts_before_resolution(
    ais_check, monkeypatch, fstype, source
):
    def _fail_backing(devno):
        raise AssertionError("backing_storage must not run on a non-block mount")

    def _fail_probe(mp):
        raise AssertionError("probe_odirect must not run on a non-block mount")

    monkeypatch.setattr(ais_check, "backing_storage", _fail_backing)
    monkeypatch.setattr(ais_check, "probe_odirect", _fail_probe)

    mount = ais_check.Mount("0:42", "/mnt/network", fstype, source, "rw")
    assert ais_check.collect([mount]) == []


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
