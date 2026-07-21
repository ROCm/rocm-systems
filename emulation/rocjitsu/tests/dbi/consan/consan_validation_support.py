#!/usr/bin/env python3
"""Shared filesystem and source-provenance primitives for ConSan validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


RESULT_SCHEMA_VERSION = 3
SITE_KINDS = ("access", "barrier", "atomic", "fence")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_row_result(root: Path, *parts: str) -> object:
    return read_json(root.joinpath(*parts, "result.json"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    """Writes stable JSON without exposing a partially written result file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def git_output(root: Path, *args: str, check: bool = True) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    ).stdout


def git_identity(root: Path, *, discover_root: bool = False) -> dict[str, object] | None:
    """Returns the common root/HEAD/dirty identity used in validation artifacts."""
    resolved = root.resolve()
    arguments = ("rev-parse", "--show-toplevel", "HEAD") if discover_root else (
        "rev-parse",
        "HEAD",
    )
    completed = subprocess.run(
        ["git", "-C", str(resolved), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        if discover_root:
            return None
        return {"root": str(resolved), "head": None, "dirty": None}
    lines = completed.stdout.splitlines()
    if discover_root:
        if len(lines) != 2:
            return None
        repository_root, head = lines
    else:
        if len(lines) != 1:
            return None
        repository_root, head = str(resolved), lines[0]
    status = subprocess.run(
        ["git", "-C", repository_root, "status", "--porcelain"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    dirty = bool(status.stdout) if status.returncode == 0 else None
    return {"root": repository_root, "head": head, "dirty": dirty}
