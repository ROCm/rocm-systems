"""Tests for the sysfs device-stack walk: _collect_leaf_disks() and helpers.

These build a real fake sysfs tree under tmp_path (directories, files, and
symlinks) and exercise the walk against it, so no os calls are mocked.
"""

# Test names document themselves; the ais_check fixture is injected by name.
# These tests intentionally exercise the script's internal (underscore)
# helpers, so protected-access is expected.
# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument
# pylint: disable=protected-access


def _leaves(ais_check, base):
    flags = {"lvm": False}
    found = ais_check._collect_leaf_disks(str(base), set(), flags)
    return [ais_check.os.path.basename(p) for p in found], flags


def test_plain_physical_disk_is_its_own_leaf(tmp_path, ais_check, make_sysfs_disk):
    base = make_sysfs_disk(tmp_path, "nvme0n1")

    names, flags = _leaves(ais_check, base)

    assert names == ["nvme0n1"]
    assert flags["lvm"] is False


def test_partition_resolves_to_parent_disk(tmp_path, ais_check, make_sysfs_disk):
    disk = make_sysfs_disk(tmp_path, "nvme0n1")
    part = make_sysfs_disk(disk, "nvme0n1p1", partition=True)

    names, _ = _leaves(ais_check, part)

    assert names == ["nvme0n1"]


def test_lvm_layer_sets_flag_and_descends(tmp_path, ais_check, make_sysfs_disk):
    disk = make_sysfs_disk(tmp_path, "nvme0n1")
    lv = make_sysfs_disk(tmp_path, "dm-0", dm_uuid="LVM-abc123", slaves=[disk])

    names, flags = _leaves(ais_check, lv)

    assert names == ["nvme0n1"]
    assert flags["lvm"] is True


def test_non_lvm_dm_layer_does_not_set_flag(tmp_path, ais_check, make_sysfs_disk):
    disk = make_sysfs_disk(tmp_path, "nvme0n1")
    crypt = make_sysfs_disk(
        tmp_path, "dm-0", dm_uuid="CRYPT-LUKS2-deadbeef", slaves=[disk]
    )

    names, flags = _leaves(ais_check, crypt)

    assert names == ["nvme0n1"]
    assert flags["lvm"] is False


def test_partition_reached_through_slaves_resolves_to_disk(
    tmp_path, ais_check, make_sysfs_disk
):
    disk = make_sysfs_disk(tmp_path, "nvme0n1")
    part = make_sysfs_disk(disk, "nvme0n1p1", partition=True)
    lv = make_sysfs_disk(tmp_path, "dm-0", dm_uuid="LVM-x", slaves=[part])

    names, flags = _leaves(ais_check, lv)

    assert names == ["nvme0n1"]
    assert flags["lvm"] is True


def test_multiple_lower_devices_yield_multiple_leaves(
    tmp_path, ais_check, make_sysfs_disk
):
    disk_a = make_sysfs_disk(tmp_path, "nvme0n1")
    disk_b = make_sysfs_disk(tmp_path, "nvme1n1")
    raid = make_sysfs_disk(tmp_path, "md0", slaves=[disk_a, disk_b])

    names, _ = _leaves(ais_check, raid)

    assert sorted(names) == ["nvme0n1", "nvme1n1"]


def test_shared_lower_device_is_not_double_counted(
    tmp_path, ais_check, make_sysfs_disk
):
    # A diamond: two intermediate dm layers over the same physical disk.
    disk = make_sysfs_disk(tmp_path, "nvme0n1")
    mid_a = make_sysfs_disk(tmp_path, "dm-0", slaves=[disk])
    mid_b = make_sysfs_disk(tmp_path, "dm-1", slaves=[disk])
    top = make_sysfs_disk(tmp_path, "dm-2", dm_uuid="LVM-top", slaves=[mid_a, mid_b])

    names, _ = _leaves(ais_check, top)

    assert names == ["nvme0n1"]


def test_whole_disk_sysfs_steps_up_for_partition(tmp_path, ais_check, make_sysfs_disk):
    disk = make_sysfs_disk(tmp_path, "sda")
    part = make_sysfs_disk(disk, "sda1", partition=True)

    assert ais_check._whole_disk_sysfs(str(part)) == str(disk)


def test_whole_disk_sysfs_returns_disk_unchanged(tmp_path, ais_check, make_sysfs_disk):
    disk = make_sysfs_disk(tmp_path, "sda")

    assert ais_check._whole_disk_sysfs(str(disk)) == str(disk)


def test_read_sysfs_strips_value(tmp_path, ais_check):
    f = tmp_path / "attr"
    f.write_text("  value\n", encoding="utf-8")

    assert ais_check._read_sysfs(str(f)) == "value"


def test_read_sysfs_missing_returns_none(tmp_path, ais_check):
    assert ais_check._read_sysfs(str(tmp_path / "nope")) is None
