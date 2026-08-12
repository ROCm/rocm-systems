"""Tests for binutils and the bulk-unbundle CLI."""

from argparse import Namespace
from pathlib import Path

import pytest

from rocm_kpack import binutils
from rocm_kpack.tools import bulk_unbundle


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


def test_bundled_binary_unbundles_elf_without_external_tools(
    test_assets_dir: Path, tmp_path: Path
):
    """ELF discovery and section extraction use the in-process parser."""
    binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
    unavailable = tmp_path / "not-a-tool"
    toolchain = binutils.Toolchain(readelf=unavailable, objcopy=unavailable)

    with binutils.BundledBinary(binary, toolchain=toolchain).unbundle() as contents:
        assert contents.target_list
        assert all((contents.dest_dir / name).is_file() for name in contents.file_names)


def test_bulk_unbundle_output_dir(
    test_assets_dir: Path, tmp_path: Path
):
    source = test_assets_dir / "ccob" / "ccob_gfx942_sample1.co"
    output_dir = tmp_path / "explicit-output"

    bulk_unbundle.run(
        Namespace(files=[source], output_dir=output_dir, gfx_arch=None)
    )

    outputs = list(output_dir.iterdir())
    assert outputs
    assert all(output.is_file() for output in outputs)


def test_bulk_unbundle_filters_gfx_arch(test_assets_dir: Path, tmp_path: Path):
    source = test_assets_dir / "ccob" / "ccob_gfx942_sample1.co"
    output_dir = tmp_path / "gfx942-only"

    bulk_unbundle.run(
        Namespace(files=[source], output_dir=output_dir, gfx_arch="gfx942")
    )

    outputs = list(output_dir.iterdir())
    assert outputs
    assert all("gfx942" in output.name for output in outputs)


def test_unbundle_gfx_arch_filter_requires_a_match(
    test_assets_dir: Path, tmp_path: Path
):
    source = test_assets_dir / "ccob" / "ccob_gfx942_sample1.co"
    output_dir = tmp_path / "no-matches"

    with pytest.raises(ValueError, match="No code objects for architecture 'gfx1250'"):
        binutils.BundledBinary(source).unbundle(
            dest_dir=output_dir, gfx_arch="gfx1250"
        )

    assert not output_dir.exists()


def test_bulk_unbundle_output_dir_requires_one_input(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        bulk_unbundle.main(
            ["--output-dir", str(tmp_path / "output"), "first.co", "second.co"]
        )

    assert exc.value.code == 2
