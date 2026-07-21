#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Create a compact, provenance-aware waitcheck corpus manifest.

Inputs may be executables or AMDGPU code objects; with ``--extract-dir`` they
may also be containers with embedded AMDGPU ELFs.  The manifest records the
extraction command instead of copying framework packages into the source tree.
``rj_waitcheck`` supplies authoritative target, code-object, and kernel
identities from the final bytes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import mmap
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys

ELF64_HEADER_SIZE = 64
ELF64_PROGRAM_HEADER_SIZE = 56
ELF64_SECTION_HEADER_SIZE = 64
EM_AMDGPU = 224
SHT_NOBITS = 8


@dataclass(frozen=True)
class Artifact:
    path: Path
    container: Path
    container_offset: int | None
    container_target: str | None = None
    container_code_object_index: int | None = None
    container_member: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--waitcheck", type=Path, required=True)
    parser.add_argument(
        "--project", required=True, choices=("ck", "iree", "pytorch", "triton")
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Package version, build identity, or source revision for the artifacts",
    )
    parser.add_argument(
        "--provenance",
        required=True,
        choices=("llvm", "generated-assembly", "unknown"),
        help="Who selected the encoded waits; clang assembly alone is not LLVM provenance",
    )
    parser.add_argument(
        "--extraction-command",
        help="Exact command used to produce the inputs (defaults to this invocation)",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        help="Extract embedded AMDGPU ELF code objects here before inventorying them",
    )
    parser.add_argument(
        "--target", help="Restrict inventory to one supported GPU target"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel rj_co extraction jobs (default: up to 16)",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recurse through directory inputs"
    )
    parser.add_argument(
        "--output", type=Path, help="Write JSONL here instead of stdout"
    )
    return parser.parse_args()


def input_files(inputs: list[Path], recursive: bool) -> list[Path]:
    files: set[Path] = set()
    for path in inputs:
        if path.is_file():
            files.add(path)
        elif path.is_dir() and recursive:
            files.update(
                candidate for candidate in path.rglob("*") if candidate.is_file()
            )
        else:
            raise ValueError(
                f"input is not a file{'' if recursive else ' (use --recursive for directories)'}: {path}"
            )
    return sorted(files)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedded_elf_size(data: bytes, offset: int) -> int | None:
    """Returns the exact size of a little-endian AMDGPU ELF at offset."""
    if offset + ELF64_HEADER_SIZE > len(data):
        return None
    ident = data[offset : offset + 16]
    if ident[:4] != b"\x7fELF" or ident[4:7] != bytes((2, 1, 1)) or ident[7] != 64:
        return None

    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data, offset)
    if header[2] != EM_AMDGPU:
        return None
    phoff, shoff = header[5], header[6]
    ehsize, phentsize, phnum = header[8:11]
    shentsize, shnum = header[11:13]
    if ehsize < ELF64_HEADER_SIZE:
        return None

    image_end = ehsize
    if phnum:
        if phentsize < ELF64_PROGRAM_HEADER_SIZE:
            return None
        table_end = phoff + phentsize * phnum
        if table_end > len(data) - offset:
            return None
        image_end = max(image_end, table_end)
        for index in range(phnum):
            program_header = struct.unpack_from(
                "<IIQQQQQQ", data, offset + phoff + phentsize * index
            )
            image_end = max(image_end, program_header[2] + program_header[5])

    if shnum:
        if shentsize < ELF64_SECTION_HEADER_SIZE:
            return None
        table_end = shoff + shentsize * shnum
        if table_end > len(data) - offset:
            return None
        image_end = max(image_end, table_end)
        for index in range(shnum):
            section_header = struct.unpack_from(
                "<IIQQQQIIQQ", data, offset + shoff + shentsize * index
            )
            if section_header[1] != SHT_NOBITS:
                image_end = max(image_end, section_header[4] + section_header[5])

    if image_end > len(data) - offset:
        return None
    return image_end


def extract_code_objects(path: Path, output_dir: Path) -> list[Artifact]:
    artifacts: list[Artifact] = []
    if path.stat().st_size == 0:
        return artifacts
    container_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    with path.open("rb") as stream, mmap.mmap(
        stream.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        offset = data.find(b"\x7fELF")
        while offset >= 0:
            image_size = embedded_elf_size(data, offset)
            if image_size is not None:
                output_path = output_dir / (
                    f"{path.stem}-{container_id}-{len(artifacts):04d}-{offset:x}.hsaco"
                )
                output_path.write_bytes(data[offset : offset + image_size])
                artifacts.append(Artifact(output_path, path, offset))
                offset = data.find(b"\x7fELF", offset + image_size)
            else:
                offset = data.find(b"\x7fELF", offset + 4)
    return artifacts


def extract_code_objects_with_rj_co(
    waitcheck: Path,
    path: Path,
    output_dir: Path,
    target: str | None,
    jobs: int,
) -> list[Artifact]:
    """Extracts every selected code object through rj_co's container parser."""
    rj_co = waitcheck.resolve().with_name("rj_co")
    if not rj_co.is_file():
        return []

    listed = subprocess.run(
        [str(rj_co), str(path), "--list-code-objects"],
        check=False,
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        return []

    selections: list[tuple[str, int]] = []
    for line in listed.stdout.splitlines():
        name, separator, count_text = line.partition(":")
        if not separator or (target and name.strip() != target):
            continue
        try:
            count = int(count_text.strip())
        except ValueError:
            continue
        selections.extend((name.strip(), index) for index in range(count))
    if not selections:
        return []

    container_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]

    def extract(selection: tuple[str, int]) -> Artifact:
        selected_target, index = selection
        output_path = output_dir / (
            f"{path.stem}-{container_id}-{selected_target}-{index:04d}.hsaco"
        )
        command = [
            str(rj_co),
            str(path),
            "--target",
            selected_target,
            "--code-object-index",
            str(index),
            "--extract-code-object",
            str(output_path),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (
                result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
            )
            raise RuntimeError(
                f"rj_co extraction failed for {path} {selected_target}[{index}] "
                f"(exit {result.returncode}): {detail}"
            )
        return Artifact(output_path, path, None, selected_target, index)

    with ThreadPoolExecutor(max_workers=max(1, min(jobs, 16))) as executor:
        return list(executor.map(extract, selections))


def extract_code_objects_from_kpack(
    path: Path,
    output_dir: Path,
    target: str | None,
) -> list[Artifact]:
    """Extracts code objects from a TheRock split-wheel kpack archive."""
    try:
        from rocm_kpack.kpack import PackedKernelArchive
    except ImportError as error:
        raise RuntimeError(
            f"inventorying {path} requires the rocm_kpack Python package"
        ) from error

    try:
        archive = PackedKernelArchive.read(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"failed to read kpack archive {path}: {error}") from error

    selections = sorted(
        (binary, arch)
        for binary, arches in archive.toc.items()
        for arch in arches
        if target is None or arch == target
    )
    container_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    artifacts: list[Artifact] = []
    for ordinal, (binary, arch) in enumerate(selections):
        data = archive.get_kernel(binary, arch)
        if data is None:
            raise RuntimeError(f"failed to extract {binary} ({arch}) from {path}")
        output_path = output_dir / (
            f"{path.stem}-{container_id}-{arch}-{ordinal:04d}.hsaco"
        )
        output_path.write_bytes(data)
        index_text = binary.rpartition("#")[2]
        code_object_index = int(index_text) if index_text.isdecimal() else ordinal
        artifacts.append(
            Artifact(output_path, path, None, arch, code_object_index, binary)
        )
    return artifacts


def inventory_file(
    waitcheck: Path, path: Path, target: str | None
) -> list[dict[str, object]]:
    command = [str(waitcheck), str(path), "--list-kernels"]
    if target:
        command.extend(("--target", target))
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (
            result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        )
        raise RuntimeError(
            f"rj_waitcheck inventory failed for {path} (exit {result.returncode}): {detail}"
        )
    rows: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("schema") != "rj-waitcheck-corpus-inventory-v1":
            raise ValueError(f"unexpected rj_waitcheck inventory row: {line}")
        rows.append(row)
    if not rows:
        raise RuntimeError(f"rj_waitcheck returned no kernel inventory rows for {path}")
    return rows


def main() -> int:
    args = parse_args()
    if args.jobs < 1 or args.jobs > 16:
        print("--jobs must be between 1 and 16", file=sys.stderr)
        return 2
    try:
        files = input_files(args.inputs, args.recursive)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    if args.extract_dir:
        args.extract_dir.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for path in files:
            if path.suffix == ".kpack":
                extracted = extract_code_objects_from_kpack(
                    path, args.extract_dir, args.target
                )
            else:
                extracted = extract_code_objects_with_rj_co(
                    args.waitcheck, path, args.extract_dir, args.target, args.jobs
                )
            if not extracted:
                extracted = extract_code_objects(path, args.extract_dir)
            artifacts.extend(extracted)
    else:
        artifacts = [Artifact(path, path, 0) for path in files]
    if not artifacts:
        print("no AMDGPU code objects found in the requested inputs", file=sys.stderr)
        return 1

    extraction_command = args.extraction_command or shlex.join(
        [str(Path(__file__).resolve()), *sys.argv[1:]]
    )
    output = args.output.open("w", encoding="utf-8") if args.output else sys.stdout
    container_metadata: dict[Path, tuple[int, str]] = {}
    try:
        print(
            json.dumps(
                {
                    "schema": "rj-waitcheck-corpus-manifest-v1",
                    "kind": "manifest",
                    "project": args.project,
                    "source": args.source,
                    "provenance": args.provenance,
                    "extraction_command": extraction_command,
                    "artifact_count": len(artifacts),
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=output,
        )
        for artifact_id, artifact in enumerate(artifacts):
            rows = inventory_file(args.waitcheck, artifact.path, args.target)
            artifact_sha256 = sha256(artifact.path)
            artifact_size = artifact.path.stat().st_size
            if artifact.container not in container_metadata:
                container_metadata[artifact.container] = (
                    artifact.container.stat().st_size,
                    sha256(artifact.container),
                )
            container_size, container_sha256 = container_metadata[artifact.container]
            print(
                json.dumps(
                    {
                        "schema": "rj-waitcheck-corpus-manifest-v1",
                        "kind": "artifact",
                        "artifact_id": artifact_id,
                        "artifact": str(artifact.path),
                        "artifact_size": artifact_size,
                        "artifact_sha256": artifact_sha256,
                        "container": str(artifact.container),
                        "container_offset": artifact.container_offset,
                        "container_target": artifact.container_target,
                        "container_code_object_index": artifact.container_code_object_index,
                        "container_member": artifact.container_member,
                        "container_size": container_size,
                        "container_sha256": container_sha256,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                file=output,
            )
            for row in rows:
                row.pop("input", None)
                row["schema"] = "rj-waitcheck-corpus-manifest-v1"
                row["artifact_id"] = artifact_id
                print(
                    json.dumps(row, separators=(",", ":"), sort_keys=True), file=output
                )
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        if output is not sys.stdout:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
