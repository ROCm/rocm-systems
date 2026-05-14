"""Utility for kpacking fat binaries by removing .hip_fatbin sections.

This tool removes .hip_fatbin sections from ELF binaries and reclaims disk space
by shifting subsequent sections to fill the gap. Unlike objcopy --remove-section,
this tool actually modifies PT_LOAD segments to eliminate device code content.

Usage:
  python -m rocm_kpack.tools.kpack_binary <input_binary> <output_binary>

Example:
  python -m rocm_kpack.tools.kpack_binary librocblas.so.5.1 librocblas_host_only.so
"""

import argparse
from pathlib import Path
import sys

from rocm_kpack.elf import kpack_offload_binary


def run(args: argparse.Namespace):
    input_path = args.input
    output_path = args.output

    print(f"Kpacking: {input_path}")
    print(f"Output: {output_path}")
    print()

    result = kpack_offload_binary(input_path, output_path, verbose=True)

    print()
    print(f"Results:")
    print(
        f"  Original size: {result['original_size']:,} bytes ({result['original_size'] / (1024**2):.2f} MB)"
    )
    print(
        f"  New size: {result['new_size']:,} bytes ({result['new_size'] / (1024**2):.2f} MB)"
    )
    print(
        f"  Removed: {result['removed']:,} bytes ({result['removed'] / (1024**2):.2f} MB)"
    )
    if result["original_size"] > 0:
        reduction_pct = (result["removed"] / result["original_size"]) * 100
        print(f"  Reduction: {reduction_pct:.2f}%")


def main(argv: list[str]):
    p = argparse.ArgumentParser(
        description="Kpack fat binaries by removing .hip_fatbin sections and reclaiming disk space"
    )
    p.add_argument(
        "input", type=Path, help="Input fat binary (ELF executable or shared library)"
    )
    p.add_argument("output", type=Path, help="Output host-only binary")
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
