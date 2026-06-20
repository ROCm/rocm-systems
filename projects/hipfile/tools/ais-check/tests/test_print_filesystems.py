"""Tests for print_filesystems(): the default vs verbose filesystem report."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument


def test_default_lists_only_supported(capsys, ais_check, make_fs_result):
    filesystems = [
        make_fs_result("/data", "/dev/nvme0n1p1", True),
        make_fs_result("/", "/dev/sda2", False, ["not on an NVMe drive"]),
    ]

    ais_check.print_filesystems(filesystems, verbose=False)

    out = capsys.readouterr().out
    assert "Supported filesystems:" in out
    assert "/data (/dev/nvme0n1p1, ext4)" in out
    # Unsupported entry and its reasons must not appear in default mode.
    assert "/dev/sda2" not in out
    assert "not on an NVMe drive" not in out


def test_default_none_when_no_supported(capsys, ais_check, make_fs_result):
    filesystems = [make_fs_result("/", "/dev/sda2", False, ["uses LVM"])]

    ais_check.print_filesystems(filesystems, verbose=False)

    out = capsys.readouterr().out
    assert "Supported filesystems:" in out
    assert "None" in out


def test_verbose_lists_all_with_reasons(capsys, ais_check, make_fs_result):
    filesystems = [
        make_fs_result("/data", "/dev/nvme0n1p1", True),
        make_fs_result("/", "/dev/sda2", False, ["not on an NVMe drive", "uses LVM"]),
    ]

    ais_check.print_filesystems(filesystems, verbose=True)

    out = capsys.readouterr().out
    assert "Filesystems:" in out
    assert "/data (/dev/nvme0n1p1, ext4): supported" in out
    assert "/ (/dev/sda2, ext4): NOT supported" in out
    assert "- not on an NVMe drive" in out
    assert "- uses LVM" in out


def test_verbose_no_filesystems(capsys, ais_check):
    ais_check.print_filesystems([], verbose=True)

    assert "No block-backed filesystems found" in capsys.readouterr().out
