#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reject unapproved firmware and PCI ROMs from a vfio-user guest image."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable

FIRMWARE_DIRECTORIES = (
    PurePosixPath("lib/firmware/amdgpu"),
    PurePosixPath("lib/firmware/updates/amdgpu"),
)
NEWC_HEADER_BYTES = 110
NEWC_MAGIC = {b"070701", b"070702"}
MAX_ARCHIVE_NAME_BYTES = 4096
ROMFILE_ARGUMENT = re.compile(r"""(?:^|[,\{\s])["']?romfile["']?\s*[:=]""")


class ProvenanceError(ValueError):
    """An input cannot satisfy the guest provenance contract."""


def sha256_stream(stream: BinaryIO, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ProvenanceError("truncated initramfs entry")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream, path.stat().st_size)


def normalized_path(name: str) -> PurePosixPath:
    while name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ProvenanceError(f"unsafe manifest or archive path: {name!r}")
    return path


def is_firmware(path: PurePosixPath) -> bool:
    return any(
        path != directory and path.is_relative_to(directory)
        for directory in FIRMWARE_DIRECTORIES
    )


def read_allowlist(path: Path) -> dict[PurePosixPath, str]:
    allowed: dict[PurePosixPath, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="ascii").splitlines(), 1
    ):
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ProvenanceError(f"{path}:{line_number}: expected SHA256SUMS format")
        digest, name = fields
        try:
            int(digest, 16)
        except ValueError as error:
            raise ProvenanceError(f"{path}:{line_number}: invalid SHA-256") from error
        relative = normalized_path(name[1:] if name.startswith("*") else name)
        if not is_firmware(relative):
            raise ProvenanceError(
                f"{path}:{line_number}: allowlisted path is outside AMD firmware "
                "directories: "
                f"{relative}"
            )
        if relative in allowed:
            raise ProvenanceError(
                f"{path}:{line_number}: duplicate allowlisted path: {relative}"
            )
        allowed[relative] = digest.lower()
    return allowed


def scan_guest_root(root: Path) -> tuple[dict[PurePosixPath, str], list[str]]:
    if not root.is_dir():
        raise ProvenanceError(f"guest root is not a directory: {root}")
    found: dict[PurePosixPath, str] = {}
    errors: list[str] = []
    for relative_directory in FIRMWARE_DIRECTORIES:
        directory = root.joinpath(*relative_directory.parts)
        component = root
        symlink = None
        for part in relative_directory.parts:
            component /= part
            if component.is_symlink():
                symlink = PurePosixPath(component.relative_to(root).as_posix())
                break
        if symlink is not None:
            errors.append(f"firmware symlink is forbidden: {symlink}")
            continue
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            relative = PurePosixPath(path.relative_to(root).as_posix())
            if path.is_symlink():
                errors.append(f"firmware symlink is forbidden: {relative}")
            elif path.is_file():
                found[relative] = sha256_file(path)
    return found, errors


def read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ProvenanceError("truncated initramfs")
    return data


def discard(stream: BinaryIO, size: int) -> None:
    remaining = size
    while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ProvenanceError("truncated initramfs entry")
        remaining -= len(chunk)


def has_nonzero_bytes(stream: BinaryIO) -> bool:
    while chunk := stream.read(1024 * 1024):
        if any(chunk):
            return True
    return False


def scan_initramfs(path: Path) -> tuple[dict[PurePosixPath, str], list[str]]:
    found: dict[PurePosixPath, str] = {}
    errors: list[str] = []
    with gzip.open(path, "rb") as stream:
        while True:
            header = stream.read(NEWC_HEADER_BYTES)
            if not header:
                raise ProvenanceError("initramfs has no TRAILER!!! entry")
            if len(header) != NEWC_HEADER_BYTES or header[:6] not in NEWC_MAGIC:
                raise ProvenanceError(
                    "initramfs is not a valid gzip-compressed newc archive"
                )
            try:
                fields = [
                    int(header[offset : offset + 8], 16) for offset in range(6, 110, 8)
                ]
            except ValueError as error:
                raise ProvenanceError(
                    "initramfs has a malformed newc header"
                ) from error
            mode = fields[1]
            size = fields[6]
            name_size = fields[11]
            if name_size == 0 or name_size > MAX_ARCHIVE_NAME_BYTES:
                raise ProvenanceError("initramfs entry has an invalid name size")
            raw_name = read_exact(stream, name_size)
            if raw_name[-1:] != b"\0":
                raise ProvenanceError("initramfs entry name is not terminated")
            try:
                name = raw_name[:-1].decode("utf-8")
            except UnicodeDecodeError as error:
                raise ProvenanceError("initramfs entry name is not UTF-8") from error
            discard(stream, -(NEWC_HEADER_BYTES + name_size) % 4)
            if name == "TRAILER!!!":
                discard(stream, size)
                if has_nonzero_bytes(stream):
                    raise ProvenanceError("initramfs has nonzero data after TRAILER!!!")
                return found, errors

            relative = normalized_path(name)
            firmware = is_firmware(relative)
            if firmware and stat.S_IFMT(mode) != stat.S_IFREG:
                errors.append(
                    f"initramfs: firmware entry is not a regular file: {relative}"
                )
                discard(stream, size)
            elif firmware:
                if relative in found:
                    errors.append(f"initramfs: duplicate firmware entry: {relative}")
                    discard(stream, size)
                else:
                    found[relative] = sha256_stream(stream, size)
            else:
                discard(stream, size)
            discard(stream, -size % 4)


def compare_firmware(
    label: str, found: dict[PurePosixPath, str], allowed: dict[PurePosixPath, str]
) -> list[str]:
    errors: list[str] = []
    for path in sorted(found.keys() - allowed.keys(), key=str):
        errors.append(f"{label}unmanifested firmware: {path}")
    for path in sorted(allowed.keys() - found.keys(), key=str):
        errors.append(f"{label}manifested firmware is missing: {path}")
    for path in sorted(found.keys() & allowed.keys(), key=str):
        if found[path] != allowed[path]:
            errors.append(f"{label}firmware hash mismatch: {path}")
    return errors


def check_qemu_arguments(arguments: Iterable[str]) -> list[str]:
    errors: list[str] = []
    for argument in arguments:
        if (
            argument == "-option-rom"
            or argument.startswith("-option-rom=")
            or ROMFILE_ARGUMENT.search(argument)
        ):
            errors.append(f"PCI expansion ROM argument is forbidden: {argument}")
    return errors


def parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a vfio-user guest contains no unapproved AMD firmware "
            "or PCI ROM."
        )
    )
    parser.add_argument("--root", type=Path, help="unpacked guest root to inspect")
    parser.add_argument(
        "--initramfs", type=Path, help="gzip-compressed newc initramfs to inspect"
    )
    parser.add_argument(
        "--allowlist", required=True, type=Path, help="SHA256SUMS allowlist"
    )
    parser.add_argument("qemu_arguments", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(arguments)
    if parsed.root is None and parsed.initramfs is None:
        parser.error("at least one of --root or --initramfs is required")
    if parsed.qemu_arguments[:1] == ["--"]:
        parsed.qemu_arguments = parsed.qemu_arguments[1:]
    return parsed


def main(arguments: list[str] | None = None) -> int:
    args = parse_arguments(arguments)
    try:
        allowed = read_allowlist(args.allowlist)
        errors = check_qemu_arguments(args.qemu_arguments)
        if args.root is not None:
            root_firmware, root_errors = scan_guest_root(args.root)
            errors.extend(root_errors)
            errors.extend(compare_firmware("", root_firmware, allowed))
        if args.initramfs is not None:
            archive_firmware, archive_errors = scan_initramfs(args.initramfs)
            errors.extend(archive_errors)
            errors.extend(compare_firmware("initramfs: ", archive_firmware, allowed))
    except (OSError, ProvenanceError) as error:
        print(f"provenance check failed: {error}", file=sys.stderr)
        return 2

    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
