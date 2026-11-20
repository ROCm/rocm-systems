"""Builders for synthetic clang offload bundles used across tests."""

import struct

import zstandard as zstd


def make_uncompressed_bundle(entries: list[tuple[str, bytes]]) -> bytes:
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


def make_ccob(bundle_data: bytes) -> bytes:
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


def make_fatbin(bundles: list[dict[str, bytes]]) -> bytes:
    """Build a multi-TU fatbin: one CCOB per bundle, each mapping arch to code."""
    return b"".join(
        make_ccob(
            make_uncompressed_bundle(
                [("host-x86_64-unknown-linux-gnu-", b"")]
                + [(f"hipv4-amdgcn-amd-amdhsa--{arch}", co) for arch, co in b.items()]
            )
        )
        for b in bundles
    )
