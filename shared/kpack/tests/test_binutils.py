"""Tests for binutils and the bulk-unbundle CLI."""

from argparse import Namespace
from pathlib import Path
import struct

import pytest

from elf_test_utils import patch_hip_fatbin_to_nobits
from rocm_kpack import binutils
from rocm_kpack.elf.surgery import ElfSurgery
from rocm_kpack.tools import bulk_unbundle


def _write_synthetic_elf(path: Path, sections: list[tuple[str, int, bytes]]) -> None:
    """Write a minimal ELF64 with the requested named sections."""
    names = bytearray(b"\0")
    name_offsets = []
    for name, _, _ in sections:
        name_offsets.append(len(names))
        names += name.encode() + b"\0"
    shstrtab_name = len(names)
    names += b".shstrtab\0"

    section_count = len(sections) + 2
    section_table_offset = 64
    payload_offset = section_table_offset + section_count * 64
    payload = bytearray()
    headers = [bytes(64)]
    for name_offset, (_, section_type, content) in zip(name_offsets, sections):
        offset = payload_offset + len(payload)
        headers.append(
            struct.pack(
                "<IIQQQQIIQQ",
                name_offset,
                section_type,
                0,
                0,
                offset,
                len(content),
                0,
                0,
                1,
                0,
            )
        )
        if section_type != 8:  # SHT_NOBITS
            payload += content

    names_offset = payload_offset + len(payload)
    headers.append(
        struct.pack(
            "<IIQQQQIIQQ",
            shstrtab_name,
            3,
            0,
            0,
            names_offset,
            len(names),
            0,
            0,
            1,
            0,
        )
    )
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into(
        "<HHIQQQIHHHHHH",
        header,
        16,
        3,
        62,
        1,
        0,
        0,
        section_table_offset,
        0,
        64,
        0,
        0,
        64,
        section_count,
        section_count - 1,
    )
    path.write_bytes(header + b"".join(headers) + payload + names)


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
    test_assets_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """ELF extraction reads only the fatbin rather than loading the host ELF."""
    binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
    unavailable = tmp_path / "not-a-tool"
    toolchain = binutils.Toolchain(readelf=unavailable, objcopy=unavailable)
    monkeypatch.setattr(
        binutils.ElfSurgery,
        "load",
        lambda _path: pytest.fail("full ELF load is not allowed for extraction"),
    )

    with binutils.BundledBinary(binary, toolchain=toolchain).unbundle() as contents:
        assert contents.target_list
        assert all((contents.dest_dir / name).is_file() for name in contents.file_names)


def test_lazy_elf_reader_matches_each_section_name_exactly(tmp_path: Path):
    binary = tmp_path / "substring-name.elf"
    _write_synthetic_elf(
        binary,
        [
            ("x.hip_fatbin", 1, b"wrong"),
            (".hip_fatbin", 1, b"right"),
        ],
    )

    assert binutils.ElfSurgery.read_section(binary, ".hip_fatbin") == b"right"


def test_lazy_elf_reader_rejects_nobits_sections(tmp_path: Path):
    binary = tmp_path / "nobits.elf"
    _write_synthetic_elf(binary, [(".ghost", 8, b"not-file-content")])

    assert binutils.ElfSurgery.read_section(binary, ".ghost") is None


def test_bulk_unbundle_output_dir(test_assets_dir: Path, tmp_path: Path):
    source = test_assets_dir / "ccob" / "ccob_gfx942_sample1.co"
    output_dir = tmp_path / "explicit-output"

    bulk_unbundle.run(Namespace(files=[source], output_dir=output_dir, gfx_arch=None))

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
        binutils.BundledBinary(source).unbundle(dest_dir=output_dir, gfx_arch="gfx1250")

    assert not output_dir.exists()


def test_unbundle_no_match_does_not_create_owned_temporary_directory(
    test_assets_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    source = test_assets_dir / "ccob" / "ccob_gfx942_sample1.co"
    binary = binutils.BundledBinary(source)

    def unexpected_mkdtemp():
        pytest.fail("temporary directory was created before filtering")

    monkeypatch.setattr(binutils.tempfile, "mkdtemp", unexpected_mkdtemp)
    with pytest.raises(ValueError, match="No code objects for architecture 'gfx1250'"):
        binary.unbundle(gfx_arch="gfx1250")


def test_bulk_unbundle_output_dir_requires_one_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(SystemExit) as exc:
        bulk_unbundle.main(
            ["--output-dir", str(tmp_path / "output"), "first.co", "second.co"]
        )

    assert exc.value.code == 2
    assert "--output-dir requires exactly one input file" in capsys.readouterr().err


def test_has_fatbin_section_rejects_nobits(test_assets_dir: Path, tmp_path: Path):
    """ElfSurgery.has_fatbin_section() must ignore a SHT_NOBITS .hip_fatbin.

    Separated debug files (`objcopy --only-keep-debug`) keep the section
    header but hold no section content.
    """
    source = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
    assert ElfSurgery.has_fatbin_section(source) is True

    debug_copy = patch_hip_fatbin_to_nobits(
        source, tmp_path / "test_kernel_single.exe.debug"
    )
    assert ElfSurgery.has_fatbin_section(debug_copy) is False


def test_bundled_binary_nobits_is_standalone(
    test_assets_dir: Path, tmp_path: Path, toolchain: binutils.Toolchain
):
    """A separated debug file must not be classified as BUNDLED.

    Regression test: it previously was, and the subsequent
    `objcopy --dump-section .hip_fatbin=<tmp>/fatbin.o` exited 0 without
    writing anything (GNU objcopy only warns for a section with no contents),
    so unbundling died with `FileNotFoundError: .../fatbin.o`.
    """
    source = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
    assert (
        binutils.BundledBinary(source, toolchain=toolchain).binary_type
        == binutils.BinaryType.BUNDLED
    )

    debug_copy = patch_hip_fatbin_to_nobits(
        source, tmp_path / "test_kernel_single.exe.debug"
    )
    assert (
        binutils.BundledBinary(debug_copy, toolchain=toolchain).binary_type
        == binutils.BinaryType.STANDALONE
    )
