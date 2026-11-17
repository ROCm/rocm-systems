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
import tempfile

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
        default=min(4, os.cpu_count() or 1),
        help="Parallel input-container extraction jobs (default: up to 4; capped at 4)",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recurse through directory inputs"
    )
    parser.add_argument(
        "--output", type=Path, help="Write JSONL here instead of stdout"
    )
    return parser.parse_args()


def input_files(
    inputs: list[Path], recursive: bool, extract_dir: Path | None, output: Path | None
) -> list[Path]:
    extract_root = extract_dir.resolve() if extract_dir else None
    output_file = output.resolve() if output else None
    files: set[Path] = set()
    for path in inputs:
        if path.is_file():
            files.add(path)
        elif path.is_dir() and recursive:
            for directory, subdirs, names in os.walk(path):
                directory_path = Path(directory)
                if extract_root and directory_path.resolve().is_relative_to(extract_root):
                    subdirs.clear()
                    continue
                subdirs[:] = [
                    name for name in subdirs
                    if not extract_root
                    or not (directory_path / name).resolve().is_relative_to(extract_root)
                ]
                files.update(
                    candidate for name in names
                    if (candidate := directory_path / name).is_file()
                    and (output_file is None or candidate.resolve() != output_file)
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


def listed_selections(output: str, target: str | None) -> list[tuple[str, int]]:
    selections = []
    for line in output.splitlines():
        name, separator, count = line.partition(":")
        if separator and count.strip().isdecimal() and (not target or name.strip() == target):
            selections.extend((name.strip(), index) for index in range(int(count)))
    return selections


def extract_code_objects_with_waitcheck(
    waitcheck: Path,
    path: Path,
    output_dir: Path,
    target: str | None,
) -> list[Artifact]:
    """Parse one container once and extract its selected code objects."""
    container_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    container_dir = output_dir / f"{path.stem}-{container_id}"
    command = [
        str(waitcheck), str(path), "--extract-code-objects", str(container_dir),
        "--skip-unsupported",
    ]
    if target:
        command.extend(("--target", target))
    extracted = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if extracted.returncode != 0:
        detail = extracted.stderr.strip() or extracted.stdout.strip() or "no diagnostic output"
        raise RuntimeError(
            f"rj_waitcheck extraction failed for {path} "
            f"(exit {extracted.returncode}): {detail}"
        )
    artifacts: list[Artifact] = []
    seen: set[tuple[str, int]] = set()
    for line in extracted.stdout.splitlines():
        row = json.loads(line)
        if (row.get("schema") != "rj-waitcheck-extraction-v1" or
                row.get("kind") != "code-object"):
            raise ValueError(f"unexpected rj_waitcheck extraction row: {line}")
        selected_target = row["target"]
        index = row["code_object_index"]
        output_path = Path(row["output"])
        if (not isinstance(selected_target, str) or not isinstance(index, int) or
                index < 0 or (target and selected_target != target) or
                (selected_target, index) in seen or
                output_path.parent.resolve() != container_dir.resolve() or
                not output_path.is_file()):
            raise ValueError(f"invalid rj_waitcheck extraction row: {line}")
        seen.add((selected_target, index))
        artifacts.append(Artifact(output_path, path, None, selected_target, index))
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


def artifact_matches_target(waitcheck: Path, artifact: Artifact, target: str) -> bool:
    listed = subprocess.run(
        [str(waitcheck), str(artifact.path), "--list-code-objects"],
        capture_output=True, text=True, check=False,
    )
    if listed.returncode != 0:
        raise RuntimeError(
            f"failed to identify target of embedded code object {artifact.path}: "
            f"{listed.stderr.strip() or listed.stdout.strip()}"
        )
    return bool(listed_selections(listed.stdout, target))


def paths_alias(left: Path, right: Path) -> bool:
    if left.exists() and right.exists() and left.samefile(right):
        return True
    return left.resolve() == right.resolve()


def main() -> int:
    args = parse_args()
    if args.jobs < 1 or args.jobs > 16:
        print("--jobs must be between 1 and 16", file=sys.stderr)
        return 2
    try:
        files = input_files(args.inputs, args.recursive, args.extract_dir, args.output)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    if args.extract_dir:
        args.extract_dir.mkdir(parents=True, exist_ok=True)

        def extract_input(path: Path) -> list[Artifact]:
            extracted = extract_code_objects_with_waitcheck(
                args.waitcheck, path, args.extract_dir, args.target
            )
            if not extracted:
                extracted = extract_code_objects(path, args.extract_dir)
                if args.target:
                    selected = []
                    for artifact in extracted:
                        if artifact_matches_target(args.waitcheck, artifact, args.target):
                            selected.append(artifact)
                    extracted = selected
            return extracted

        # One waitcheck process parses each container; at most four large
        # containers are in memory at once, even when --jobs requests more.
        try:
            with ThreadPoolExecutor(max_workers=min(args.jobs, 4)) as executor:
                artifacts = [artifact for group in executor.map(extract_input, files)
                             for artifact in group]
        except (OSError, RuntimeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 1
    else:
        artifacts = [Artifact(path, path, 0) for path in files]
    if not artifacts:
        print("no AMDGPU code objects found in the requested inputs", file=sys.stderr)
        return 1

    if args.output and any(
        paths_alias(args.output, protected)
        for protected in [*files, *(artifact.path for artifact in artifacts)]
    ):
        print("manifest output aliases an input or selected artifact", file=sys.stderr)
        return 2

    extraction_command = args.extraction_command or shlex.join(
        [str(Path(__file__).resolve()), *sys.argv[1:]]
    )
    temporary_output: Path | None = None
    if args.output:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{args.output.name}.",
            suffix=".tmp", dir=args.output.parent, delete=False,
        ) as stream:
            temporary_output = Path(stream.name)
        output = temporary_output.open("w", encoding="utf-8")
    else:
        output = sys.stdout
    container_metadata: dict[Path, tuple[int, str]] = {}
    complete = False
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
        complete = True
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        if output is not sys.stdout:
            output.close()
        if temporary_output and temporary_output.exists() and args.output:
            # The final manifest appears only after every artifact was inventoried.
            if complete:
                temporary_output.replace(args.output)
            else:
                temporary_output.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
