#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify the standalone saturation kernels' compiled GPU resources."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
COUNT_FILE = HERE / "vgpr_saturation_counts.def"
SOURCE_FILE = HERE / "vgpr_saturation.hip"
REQUIRED_ZERO_FIELDS = (
    "agpr_count",
    "private_segment_fixed_size",
    "vgpr_spill_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rocm-path",
        type=Path,
        default=Path(os.environ["ROCM_PATH"]) if "ROCM_PATH" in os.environ else None,
        help="ROCm SDK root (or set ROCM_PATH)",
    )
    parser.add_argument("--compiler", type=Path, help="amdclang++ executable")
    parser.add_argument("--arch", default="gfx942", help="AMDGPU offload architecture")
    return parser.parse_args()


def find_compiler(rocm_path: Path, requested: Path | None) -> Path:
    if requested is not None:
        candidates = [requested]
    else:
        candidates = [
            rocm_path / "lib" / "llvm" / "bin" / "amdclang++",
            rocm_path / "bin" / "amdclang++",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("amdclang++ not found; pass --compiler or a valid --rocm-path")


def expected_counts() -> list[int]:
    counts = [
        int(match)
        for match in re.findall(
            r"^VGPR_SATURATION_COUNT\((\d+)\)$", COUNT_FILE.read_text(), re.M
        )
    ]
    if not counts:
        raise RuntimeError(f"no saturation counts found in {COUNT_FILE}")
    return counts


def compile_assembly(compiler: Path, rocm_path: Path, arch: str, output: Path) -> None:
    command = [
        str(compiler),
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"--offload-arch={arch}",
        f"--rocm-path={rocm_path}",
        "--cuda-device-only",
        "-S",
        str(SOURCE_FILE),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)


def parse_kernel_metadata(assembly: str) -> dict[int, dict[str, int]]:
    marker = "\t.amdgpu_metadata\n"
    if marker not in assembly:
        raise RuntimeError("compiler assembly contains no .amdgpu_metadata section")
    metadata = assembly.split(marker, 1)[1]
    kernel_section_match = re.search(
        r"^amdhsa\.kernels:\s*\n(?P<records>(?:^(?:  |\s*$).*\n?)*)",
        metadata,
        re.M,
    )
    if kernel_section_match is None:
        raise RuntimeError("AMDGPU metadata contains no amdhsa.kernels sequence")

    kernels: dict[int, dict[str, int]] = {}
    for block in re.split(
        r"(?=^  - )", kernel_section_match.group("records"), flags=re.M
    ):
        name_match = re.search(r"^(?:  - |    )\.name:\s+(.+)$", block, re.M)
        if name_match is None:
            continue
        count_match = re.search(r"vgpr_saturation_kernelILi(\d+)E", name_match.group(1))
        if count_match is None:
            continue
        fields = {
            key: int(value)
            for key, value in re.findall(
                r"^(?:  - |    )\.([a-z_]+):\s+(\d+)$", block, re.M
            )
        }
        count = int(count_match.group(1))
        if count in kernels:
            raise RuntimeError(f"duplicate saturation kernel for Count={count}")
        kernels[count] = fields
    return kernels


def verify(kernels: dict[int, dict[str, int]], counts: list[int]) -> None:
    if set(kernels) != set(counts):
        missing = sorted(set(counts) - set(kernels))
        unexpected = sorted(set(kernels) - set(counts))
        raise RuntimeError(
            f"kernel set mismatch: missing={missing}, unexpected={unexpected}"
        )

    print("requested  vgprs  spills  private_bytes  agprs")
    for count in counts:
        fields = kernels[count]
        missing_fields = [
            field
            for field in ("vgpr_count", *REQUIRED_ZERO_FIELDS)
            if field not in fields
        ]
        if missing_fields:
            raise RuntimeError(f"Count={count} metadata lacks {missing_fields}")
        print(
            f"{count:9d}  {fields['vgpr_count']:5d}  {fields['vgpr_spill_count']:6d}"
            f"  {fields['private_segment_fixed_size']:13d}  {fields['agpr_count']:5d}"
        )
        if fields["vgpr_count"] != count + 1:
            raise RuntimeError(
                f"Count={count} uses {fields['vgpr_count']} VGPRs; expected {count + 1}"
            )
        for field in REQUIRED_ZERO_FIELDS:
            if fields[field] != 0:
                raise RuntimeError(
                    f"Count={count} has nonzero {field}: {fields[field]}"
                )


def main() -> None:
    args = parse_args()
    if args.rocm_path is None:
        raise RuntimeError("--rocm-path is required when ROCM_PATH is unset")
    compiler = find_compiler(args.rocm_path, args.compiler)
    with tempfile.TemporaryDirectory(prefix="vgpr-saturation-") as temporary_directory:
        assembly_path = Path(temporary_directory) / "vgpr_saturation.s"
        compile_assembly(compiler, args.rocm_path, args.arch, assembly_path)
        verify(parse_kernel_metadata(assembly_path.read_text()), expected_counts())
    print(f"verified all saturation kernels for {args.arch}")


if __name__ == "__main__":
    main()
