#!/usr/bin/env python3
"""Shared filesystem and source-provenance primitives for ConSan validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

RESULT_SCHEMA_VERSION = 3
SITE_KINDS = ("access", "barrier", "atomic", "fence")
FAULT_RESERVATION_SCHEMA_VERSION = 1
FAULT_RESERVATION_OUTCOMES = (
    "reserved",
    "mutation_already_installed",
    "contention_timeout",
    "reentrant_contention",
)
FAULT_RESERVATION_QUALIFIED = "qualified"
FAULT_RESERVATION_EVIDENCE_INVALID = "evidence_invalid"
FAULT_RESERVATION_CONTENDED = "contended"
FAULT_RESERVATION_NOT_APPLIED = "not_applied"


def fault_reservation_shape_error(value: object) -> str | None:
    """Returns why a retained process-reservation summary has an invalid shape."""
    if not isinstance(value, dict):
        return "reservation evidence must be an object"
    for name in ("attempts", "not_requested_records", "unattributed_attempts"):
        if type(value.get(name)) is not int or value[name] < 0:
            return f"{name} must be a non-negative integer"
    outcomes = value.get("outcomes")
    if not isinstance(outcomes, dict):
        return "outcomes must be an object"
    for outcome in FAULT_RESERVATION_OUTCOMES:
        if type(outcomes.get(outcome)) is not int or outcomes[outcome] < 0:
            return f"outcomes.{outcome} must be a non-negative integer"
    expected_attempts = sum(outcomes[outcome] for outcome in FAULT_RESERVATION_OUTCOMES)
    if value["attempts"] != expected_attempts:
        return "attempts must equal the sum of typed outcomes"
    return None


def fault_reservation_qualification(value: object) -> tuple[str, list[str]]:
    """Classifies retained reservation evidence and returns rejection reasons."""
    if not isinstance(value, dict):
        return FAULT_RESERVATION_EVIDENCE_INVALID, ["reservation evidence is missing"]
    if value.get("schema_version") != FAULT_RESERVATION_SCHEMA_VERSION:
        return FAULT_RESERVATION_EVIDENCE_INVALID, [
            "reservation_schema_version="
            f"{value.get('schema_version')}, "
            f"expected={FAULT_RESERVATION_SCHEMA_VERSION}; rerun required"
        ]
    if value.get("evidence_complete") is not True:
        return FAULT_RESERVATION_EVIDENCE_INVALID, [
            f"reservation_evidence_complete={value.get('evidence_complete')}"
        ]
    shape_error = fault_reservation_shape_error(value)
    if shape_error is not None:
        return FAULT_RESERVATION_EVIDENCE_INVALID, [
            f"reservation evidence shape is invalid: {shape_error}"
        ]

    outcomes = value["outcomes"]
    reasons = []
    if outcomes["reserved"] < 1:
        reasons.append(f"reservation_reserved={outcomes['reserved']}")
    for outcome in ("contention_timeout", "reentrant_contention"):
        if outcomes[outcome]:
            reasons.append(f"reservation_{outcome}={outcomes[outcome]}")
    if value["unattributed_attempts"]:
        reasons.append(
            "reservation_unattributed_attempts=" f"{value['unattributed_attempts']}"
        )

    if outcomes["contention_timeout"] or outcomes["reentrant_contention"]:
        return FAULT_RESERVATION_CONTENDED, reasons
    if value["unattributed_attempts"]:
        return FAULT_RESERVATION_EVIDENCE_INVALID, reasons
    if outcomes["reserved"] < 1:
        return FAULT_RESERVATION_NOT_APPLIED, reasons
    return FAULT_RESERVATION_QUALIFIED, reasons


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


def git_identity(
    root: Path, *, discover_root: bool = False
) -> dict[str, object] | None:
    """Returns the common root/HEAD/dirty identity used in validation artifacts."""
    resolved = root.resolve()
    arguments = (
        ("rev-parse", "--show-toplevel", "HEAD")
        if discover_root
        else (
            "rev-parse",
            "HEAD",
        )
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
