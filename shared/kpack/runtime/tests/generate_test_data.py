#!/usr/bin/env python3
"""Generate test .kpack archives for C++ runtime tests.

This script creates minimal .kpack files with known data for testing the C++
runtime API without depending on live build artifacts.
"""

import sys
from pathlib import Path

# Add Python module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import msgpack

from rocm_kpack.kpack import PackedKernelArchive
from rocm_kpack.compression import NoOpCompressor, ZstdCompressor


def generate_noop_archive(output_dir: Path) -> None:
    """Generate a NoOp (uncompressed) test archive."""
    archive = PackedKernelArchive(
        group_name="test",
        gfx_arch_family="gfx900X",
        gfx_arches=["gfx900", "gfx906"],
        compressor=NoOpCompressor(),
    )

    # Add test kernels with recognizable patterns
    # TOC keys always use indexed format: "binary#N"
    # Runtime constructs lookup_key as kernel_name + "#" + co_index
    #
    # Kernel 1: lib/libtest.so#0 @ gfx900
    kernel1_data = b"KERNEL1_GFX900_DATA" + b"\x00" * 100
    prepared1 = archive.prepare_kernel("lib/libtest.so#0", "gfx900", kernel1_data)
    archive.add_kernel(prepared1)

    # Kernel 2: lib/libtest.so#0 @ gfx906
    kernel2_data = b"KERNEL2_GFX906_DATA" + b"\x00" * 200
    prepared2 = archive.prepare_kernel("lib/libtest.so#0", "gfx906", kernel2_data)
    archive.add_kernel(prepared2)

    # Kernel 3: bin/testapp#0 @ gfx900
    kernel3_data = b"KERNEL3_APP_GFX900" + b"\xFF" * 150
    prepared3 = archive.prepare_kernel("bin/testapp#0", "gfx900", kernel3_data)
    archive.add_kernel(prepared3)

    archive.finalize_archive()
    output_path = output_dir / "test_noop.kpack"
    archive.write(output_path)
    print(f"Generated NoOp archive: {output_path}")
    print(f"  - 2 binaries, 3 kernels (indexed TOC keys)")
    print(
        f"  - lib/libtest.so#0: gfx900 ({len(kernel1_data)} bytes), gfx906 ({len(kernel2_data)} bytes)"
    )
    print(f"  - bin/testapp#0: gfx900 ({len(kernel3_data)} bytes)")


def generate_zstd_archive(output_dir: Path) -> None:
    """Generate a Zstd compressed test archive."""
    archive = PackedKernelArchive(
        group_name="test",
        gfx_arch_family="gfx110X",
        gfx_arches=["gfx1100", "gfx1101"],
        compressor=ZstdCompressor(compression_level=3),
    )

    # Add test kernels with compressible patterns
    # TOC keys always use indexed format: "binary#N"
    #
    # Kernel 1: lib/libhip.so#0 @ gfx1100
    kernel1_data = b"HIP_KERNEL_GFX1100_" + b"A" * 500 + b"B" * 500
    prepared1 = archive.prepare_kernel("lib/libhip.so#0", "gfx1100", kernel1_data)
    archive.add_kernel(prepared1)

    # Kernel 2: lib/libhip.so#0 @ gfx1101
    kernel2_data = b"HIP_KERNEL_GFX1101_" + b"X" * 300 + b"Y" * 300
    prepared2 = archive.prepare_kernel("lib/libhip.so#0", "gfx1101", kernel2_data)
    archive.add_kernel(prepared2)

    # Kernel 3: bin/hiptest#0 @ gfx1100
    kernel3_data = b"TEST_APP_KERNEL___" + b"\x42" * 1000
    prepared3 = archive.prepare_kernel("bin/hiptest#0", "gfx1100", kernel3_data)
    archive.add_kernel(prepared3)

    archive.finalize_archive()
    output_path = output_dir / "test_zstd.kpack"
    archive.write(output_path)
    print(f"Generated Zstd archive: {output_path}")
    print(f"  - 2 binaries, 3 kernels (indexed TOC keys)")
    print(
        f"  - lib/libhip.so#0: gfx1100 ({len(kernel1_data)} bytes), gfx1101 ({len(kernel2_data)} bytes)"
    )
    print(f"  - bin/hiptest#0: gfx1100 ({len(kernel3_data)} bytes)")


def generate_test_manifests(output_dir: Path) -> None:
    """Generate .kpm manifest files for testing manifest resolution."""
    # Manifest for test_noop.kpack (gfx900, gfx906)
    noop_manifest = {
        "format_version": 1,
        "component_name": "test_noop",
        "prefix": "lib",
        "kpack_files": {
            "gfx900": {
                "file": "test_noop.kpack",
                "size": (output_dir / "test_noop.kpack").stat().st_size,
                "kernel_count": 2,
            },
            "gfx906": {
                "file": "test_noop.kpack",
                "size": (output_dir / "test_noop.kpack").stat().st_size,
                "kernel_count": 1,
            },
        },
    }
    noop_path = output_dir / "test_noop.kpm"
    noop_path.write_bytes(msgpack.packb(noop_manifest))
    print(f"Generated manifest: {noop_path}")
    print(f"  - gfx900, gfx906 -> test_noop.kpack")

    # Manifest for test_zstd.kpack (gfx1100, gfx1101)
    zstd_manifest = {
        "format_version": 1,
        "component_name": "test_zstd",
        "prefix": "lib",
        "kpack_files": {
            "gfx1100": {
                "file": "test_zstd.kpack",
                "size": (output_dir / "test_zstd.kpack").stat().st_size,
                "kernel_count": 2,
            },
            "gfx1101": {
                "file": "test_zstd.kpack",
                "size": (output_dir / "test_zstd.kpack").stat().st_size,
                "kernel_count": 1,
            },
        },
    }
    zstd_path = output_dir / "test_zstd.kpm"
    zstd_path.write_bytes(msgpack.packb(zstd_manifest))
    print(f"Generated manifest: {zstd_path}")
    print(f"  - gfx1100, gfx1101 -> test_zstd.kpack")

    # Partial manifest (only gfx900, no gfx906) for negative tests
    partial_manifest = {
        "format_version": 1,
        "component_name": "test_partial",
        "prefix": "lib",
        "kpack_files": {
            "gfx900": {
                "file": "test_noop.kpack",
                "size": (output_dir / "test_noop.kpack").stat().st_size,
                "kernel_count": 2,
            },
        },
    }
    partial_path = output_dir / "test_partial.kpm"
    partial_path.write_bytes(msgpack.packb(partial_manifest))
    print(f"Generated manifest: {partial_path}")
    print(f"  - gfx900 only -> test_noop.kpack")


def main() -> None:
    """Generate all test archives."""
    # Output to runtime/tests/test_assets
    script_dir = Path(__file__).parent
    output_dir = script_dir / "test_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating test kpack archives...")
    print()

    generate_noop_archive(output_dir)
    print()
    generate_zstd_archive(output_dir)
    print()
    generate_test_manifests(output_dir)
    print()
    print("Done!")


if __name__ == "__main__":
    main()
