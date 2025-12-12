#!/usr/bin/env python3
"""
ELF verification CLI tool.

Validates ELF binaries using both internal structural checks and external tools
(readelf, strip, gdb, ldd).

Usage:
    python -m rocm_kpack.tools.verify_elf <binary> [--verbose]
"""

import argparse
import sys
from pathlib import Path

from rocm_kpack.elf import (
    verify_all,
    verify_with_readelf,
    verify_with_strip,
    verify_with_gdb,
    verify_with_ldd,
    ElfVerifier,
    VerificationResult,
)


def verify_binary(binary: Path, verbose: bool = False) -> VerificationResult:
    """Run all verification checks on a binary.

    Args:
        binary: Path to ELF binary
        verbose: Whether to show warnings

    Returns:
        Combined VerificationResult
    """
    print(f"Verifying: {binary}")
    print("-" * 60)

    result = VerificationResult()

    # Internal structural checks
    print("  internal: ", end="")
    internal = ElfVerifier.verify(binary)
    result.merge(internal)
    print("PASS" if internal.passed else "FAIL")
    if internal.errors:
        for e in internal.errors:
            print(f"    ERROR: {e}")
    if verbose and internal.warnings:
        for w in internal.warnings:
            print(f"    WARN: {w}")

    # External tool checks
    tools = [
        ("readelf", verify_with_readelf),
        ("strip", verify_with_strip),
        ("gdb", verify_with_gdb),
        ("ldd", verify_with_ldd),
    ]

    for name, func in tools:
        print(f"  {name}: ", end="")
        if name == "strip":
            tool_result = func(binary, tmp_dir=None)
        else:
            tool_result = func(binary)
        result.merge(tool_result)
        print("PASS" if tool_result.passed else "FAIL")

        if tool_result.errors:
            for e in tool_result.errors:
                print(f"    ERROR: {e}")
        if verbose and tool_result.warnings:
            for w in tool_result.warnings:
                print(f"    WARN: {w}")

    print("-" * 60)
    overall = "PASSED" if result.passed else "FAILED"
    print(f"Overall: {overall}")

    if result.errors:
        print("\nAll errors:")
        for e in result.errors:
            print(f"  {e}")

    if verbose and result.warnings:
        print("\nAll warnings:")
        for w in result.warnings:
            print(f"  {w}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Verify ELF binary integrity using structural checks and external tools"
    )
    parser.add_argument("binary", type=Path, help="Path to ELF binary to verify")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including warnings",
    )
    args = parser.parse_args()

    if not args.binary.exists():
        print(f"Error: {args.binary} does not exist", file=sys.stderr)
        sys.exit(1)

    result = verify_binary(args.binary, args.verbose)
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
