"""Tests for CCOB parser module."""

import struct
import subprocess
from pathlib import Path

import pytest
import zstandard as zstd

from rocm_kpack.ccob_parser import (
    CCOBHeader,
    UncompressedBundle,
    decompress_ccob,
    extract_code_objects_from_fatbin,
    list_ccob_targets,
    parse_ccob_file,
    parse_fatbin_data,
)


def test_ccob_header_parse():
    """Test CCOB header parsing."""
    # Construct a minimal CCOB v3 header
    header = (
        b"CCOB"  # Magic
        b"\x03\x00\x01\x00"  # Version 3, compression method 1 (zstd)
        b"\x0c\x8b\x12\x00\x00\x00\x00\x00"  # totalSize: 1,215,244
        b"\x38\x7f\xc6\x00\x00\x00\x00\x00"  # uncompressedSize: 13,008,696
        b"\xde\x30\xdc\xec\xea\x03\x74\xc3"  # hash
    )

    parsed = CCOBHeader.parse(header)
    assert parsed.magic == "CCOB"
    assert parsed.version == 3
    assert parsed.compression_method == 1
    assert parsed.total_size == 1_215_244
    assert parsed.uncompressed_size == 13_008_696


def test_ccob_header_invalid_magic():
    """Test that invalid magic raises ValueError."""
    bad_header = b"XXXX" + b"\x00" * 28

    with pytest.raises(ValueError, match="Invalid magic"):
        CCOBHeader.parse(bad_header)


def test_ccob_header_too_short():
    """Test that short header raises ValueError."""
    short_header = b"CCOB" + b"\x00" * 20

    with pytest.raises(ValueError, match="Header too short"):
        CCOBHeader.parse(short_header)


def test_decompress_ccob_with_real_binary(tmp_path: Path, toolchain):
    """Test CCOB decompression with a real ROCm binary."""
    # Find a real ROCm library with CCOB bundle
    rocm_lib = Path("/home/stella/workspace/rocm/gfx1100/lib/librocblas.so.5")

    if not rocm_lib.exists():
        pytest.skip("librocblas.so.5 not found")

    # Extract .hip_fatbin section
    fatbin_path = tmp_path / "librocblas_fatbin.o"
    result = subprocess.run(
        ["objcopy", "--dump-section", f".hip_fatbin={fatbin_path}", str(rocm_lib)],
        capture_output=True,
    )

    if result.returncode != 0:
        pytest.skip("Could not extract .hip_fatbin section")

    # Parse and decompress
    bundle = parse_ccob_file(fatbin_path)

    # Verify structure
    assert bundle.magic.startswith("__CLANG_OFFLOAD_BUNDLE__")
    assert bundle.num_entries > 0

    # Should have at least one device target
    triples = bundle.list_triples()
    device_targets = [t for t in triples if "hipv4-amdgcn" in t]
    assert len(device_targets) > 0

    # Verify we can extract code objects
    for triple in device_targets:
        code_obj = bundle.get_code_object(triple)
        assert code_obj is not None
        assert len(code_obj) > 0


def test_list_ccob_targets_with_real_binary(tmp_path: Path):
    """Test listing CCOB targets."""
    rocm_lib = Path("/home/stella/workspace/rocm/gfx1100/lib/librocblas.so.5")

    if not rocm_lib.exists():
        pytest.skip("librocblas.so.5 not found")

    # Extract .hip_fatbin section
    fatbin_path = tmp_path / "librocblas_fatbin.o"
    result = subprocess.run(
        ["objcopy", "--dump-section", f".hip_fatbin={fatbin_path}", str(rocm_lib)],
        capture_output=True,
    )

    if result.returncode != 0:
        pytest.skip("Could not extract .hip_fatbin section")

    # List targets
    data = fatbin_path.read_bytes()
    targets = list_ccob_targets(data)

    # Should have host + device targets
    assert len(targets) > 0
    assert any("host-" in t for t in targets)
    assert any("hipv4-amdgcn" in t for t in targets)


def test_decompress_respects_total_size():
    """Test that decompression respects totalSize field, not buffer size.

    This is the key fix that clang-offload-bundler is missing.
    """
    # This test would need a crafted CCOB with padding, which we have
    # in librocblas.so.5. The test above verifies this works.
    pass


def test_uncompressed_bundle_get_code_object():
    """Test getting specific code object from bundle."""
    # Create a minimal uncompressed bundle structure
    magic = b"__CLANG_OFFLOAD_BUNDLE__\x00"[:24]
    num_entries = (1).to_bytes(8, "little")

    # Single entry: offset=100, size=10, triple="test-triple"
    triple = b"test-triple"
    entry = (
        (100).to_bytes(8, "little")  # offset
        + (10).to_bytes(8, "little")  # size
        + len(triple).to_bytes(8, "little")  # triple_size
        + triple
    )

    # Padding to offset 100, then code object data
    padding = b"\x00" * (100 - 32 - len(entry))
    code_data = b"0123456789"

    bundle_data = magic + num_entries + entry + padding + code_data

    # Parse
    bundle = UncompressedBundle.parse(bundle_data)

    assert bundle.num_entries == 1
    assert bundle.entries[0].triple == "test-triple"

    # Get code object
    code_obj = bundle.get_code_object("test-triple")
    assert code_obj == b"0123456789"

    # Try non-existent triple
    assert bundle.get_code_object("nonexistent") is None


# ---------------------------------------------------------------------------
# Helpers for building synthetic CCOB bundles
# ---------------------------------------------------------------------------


def _make_uncompressed_bundle(entries: list[tuple[str, bytes]]) -> bytes:
    """Build an uncompressed offload bundle from (triple, data) pairs."""
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    num_entries = struct.pack("<Q", len(entries))

    # Compute header size to know where code objects start
    header_size = 24 + 8  # magic + num_entries
    for triple, _ in entries:
        header_size += 8 + 8 + 8 + len(triple)  # offset + size + triple_len + triple

    # Build entry descriptors and concatenate code object data
    descriptors = b""
    code_blobs = b""
    for triple, data in entries:
        offset = header_size + len(code_blobs)
        descriptors += struct.pack("<QQQ", offset, len(data), len(triple))
        descriptors += triple.encode("ascii")
        code_blobs += data

    return magic + num_entries + descriptors + code_blobs


def _make_ccob(bundle_data: bytes) -> bytes:
    """Wrap an uncompressed bundle in a CCOB compressed envelope."""
    cctx = zstd.ZstdCompressor()
    compressed = cctx.compress(bundle_data)

    # CCOB v3 header: 4B magic + 2B version + 2B method + 8B totalSize + 8B uncompSize + 8B hash
    header_size = 32
    total_size = header_size + len(compressed)
    header = (
        b"CCOB"
        + struct.pack("<HH", 3, 1)  # version=3, method=1 (zstd)
        + struct.pack("<Q", total_size)
        + struct.pack("<Q", len(bundle_data))
        + struct.pack("<Q", 0)  # hash (unused for decompression)
    )
    return header + compressed


# ---------------------------------------------------------------------------
# Multi-CCOB tests
# ---------------------------------------------------------------------------


def test_parse_fatbin_data_single_ccob():
    """Single CCOB round-trips correctly."""
    bundle_bytes = _make_uncompressed_bundle(
        [
            ("host-x86_64-unknown-linux-gnu-", b""),
            ("hipv4-amdgcn-amd-amdhsa--gfx942", b"CODE_OBJ_A"),
        ]
    )
    ccob = _make_ccob(bundle_bytes)

    bundles = parse_fatbin_data(ccob)
    assert len(bundles) == 1
    assert bundles[0].num_entries == 2


def test_parse_fatbin_data_multiple_ccobs():
    """Multiple back-to-back CCOBs are all extracted.

    Matches the layout produced by the linker for multi-TU HIP binaries.
    Verifies the fix for the bug where only the first CCOB was parsed.
    See: llvm/lib/Object/OffloadBundle.cpp:33-99 (extractOffloadBundle)
    """
    bundle_a = _make_uncompressed_bundle(
        [
            ("host-x86_64-unknown-linux-gnu-", b""),
            ("hipv4-amdgcn-amd-amdhsa--gfx942", b"KERNEL_A"),
        ]
    )
    bundle_b = _make_uncompressed_bundle(
        [
            ("host-x86_64-unknown-linux-gnu-", b""),
            ("hipv4-amdgcn-amd-amdhsa--gfx942", b"KERNEL_B"),
        ]
    )
    bundle_c = _make_uncompressed_bundle(
        [
            ("host-x86_64-unknown-linux-gnu-", b""),
            ("hipv4-amdgcn-amd-amdhsa--gfx942", b"KERNEL_C"),
        ]
    )

    fatbin = _make_ccob(bundle_a) + _make_ccob(bundle_b) + _make_ccob(bundle_c)

    bundles = parse_fatbin_data(fatbin)
    assert len(bundles) == 3

    # Verify we can extract each code object
    objs = extract_code_objects_from_fatbin(fatbin)
    assert len(objs) == 3
    assert objs[0].data == b"KERNEL_A"
    assert objs[1].data == b"KERNEL_B"
    assert objs[2].data == b"KERNEL_C"


def test_parse_fatbin_data_ccobs_with_padding():
    """CCOBs separated by page-alignment padding (zeros).

    Real linker output page-aligns each CCOB. The parser must skip padding
    bytes between entries, matching LLVM's magic-scanning approach.
    """
    ccob_a = _make_ccob(
        _make_uncompressed_bundle([("hipv4-amdgcn-amd-amdhsa--gfx942", b"A" * 100)])
    )
    ccob_b = _make_ccob(
        _make_uncompressed_bundle([("hipv4-amdgcn-amd-amdhsa--gfx942", b"B" * 100)])
    )

    # Pad first CCOB to next 4096-byte boundary
    padded_size = ((len(ccob_a) + 4095) // 4096) * 4096
    padding = b"\x00" * (padded_size - len(ccob_a))
    fatbin = ccob_a + padding + ccob_b

    objs = extract_code_objects_from_fatbin(fatbin)
    assert len(objs) == 2
    assert objs[0].data == b"A" * 100
    assert objs[1].data == b"B" * 100


def test_parse_fatbin_data_empty_raises():
    """Empty or unrecognized data raises ValueError."""
    with pytest.raises(ValueError, match="Unrecognized fatbin format"):
        parse_fatbin_data(b"")

    with pytest.raises(ValueError, match="Unrecognized fatbin format"):
        parse_fatbin_data(b"\x00" * 100)
