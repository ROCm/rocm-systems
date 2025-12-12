"""Tests for binutils module."""

from pathlib import Path

from rocm_kpack import binutils


def test_toolchain(test_assets_dir: Path, toolchain: binutils.Toolchain):
    """Test basic toolchain functionality."""
    bb = binutils.BundledBinary(
        test_assets_dir / "ccob" / "ccob_gfx942_sample1.co", toolchain=toolchain
    )
    with bb.unbundle() as contents:
        for target, filename in contents.target_list:
            if filename.endswith(".hsaco"):
                assert "gfx942" in target
                assert (contents.dest_dir / filename).exists()
                break
        else:
            raise AssertionError("No target hsaco file")
