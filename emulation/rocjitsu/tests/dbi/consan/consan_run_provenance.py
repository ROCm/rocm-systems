#!/usr/bin/env python3
"""Creates and validates fail-closed provenance contracts for ConSan matrix runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

from consan_validation_support import (
    RESULT_SCHEMA_VERSION,
    atomic_write_json,
    git_output,
    read_row_result,
    sha256_bytes,
)


CONTRACT_NAME = "run-contract.json"


def _file_record(label: str, path: Path, *, canonical_json: bool = False) -> dict:
    data = path.read_bytes()
    record = {
        "label": label,
        "path": str(path.resolve()),
        "sha256": sha256_bytes(data),
        "size": len(data),
    }
    if canonical_json:
        value = json.loads(data)
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        record["canonical_sha256"] = sha256_bytes(canonical)
    return record


def _source_record(root: Path) -> dict:
    resolved = root.resolve()
    head = git_output(resolved, "rev-parse", "HEAD").decode().strip()
    status = git_output(resolved, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    diff = git_output(resolved, "diff", "--binary", "HEAD", "--")
    untracked_payload = bytearray()
    for entry in status.split(b"\0"):
        if not entry.startswith(b"?? "):
            continue
        relative = entry[3:].decode(errors="surrogateescape")
        path = resolved / relative
        if path.is_file():
            untracked_payload.extend(relative.encode(errors="surrogateescape"))
            untracked_payload.extend(b"\0")
            untracked_payload.extend(path.read_bytes())
            untracked_payload.extend(b"\0")
    return {
        "root": str(resolved),
        "head": head,
        "dirty": bool(status),
        "worktree_sha256": sha256_bytes(status + b"\0" + diff + b"\0" + untracked_payload),
    }


def _contract_payload(contract: dict) -> dict:
    return {key: value for key, value in contract.items() if key != "contract_sha256"}


def _contract_sha(contract: dict) -> str:
    data = json.dumps(_contract_payload(contract), sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(data)


def load_contract(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("run contract must be a schema-version-1 object")
    if value.get("contract_sha256") != _contract_sha(value):
        raise ValueError("run contract hash is invalid")
    return value


def validate_current_contract(contract: dict) -> list[str]:
    errors = []
    for record in contract.get("files", []):
        path = Path(record["path"])
        if not path.is_file():
            errors.append(f"missing input {record['label']}: {path}")
            continue
        current = _file_record(
            record["label"], path, canonical_json="canonical_sha256" in record
        )
        if current["sha256"] != record["sha256"]:
            errors.append(f"input hash changed: {record['label']}")
        if current.get("canonical_sha256") != record.get("canonical_sha256"):
            errors.append(f"canonical JSON hash changed: {record['label']}")
    for expected in contract.get("sources", []):
        try:
            current = _source_record(Path(expected["root"]))
        except (OSError, subprocess.CalledProcessError) as error:
            errors.append(f"source revision unavailable: {expected['root']}: {error}")
            continue
        if current["head"] != expected["head"]:
            errors.append(f"source HEAD changed: {expected['root']}")
        if current["dirty"] != expected["dirty"]:
            errors.append(f"source dirty state changed: {expected['root']}")
        if current["worktree_sha256"] != expected["worktree_sha256"]:
            errors.append(f"source worktree fingerprint changed: {expected['root']}")
    return errors


def initialize_contract(
    artifact_root: Path,
    plan_path: Path,
    manifest_path: Path,
    inputs: list[tuple[str, Path]],
    source_roots: list[Path],
) -> dict:
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise ValueError("artifact root is nonempty; refusing foreign or reused evidence")
    labels = [label for label, _ in inputs]
    if "hook" not in labels or "binary" not in labels:
        raise ValueError("run inputs must include hook and binary labels")
    if len(labels) != len(set(labels)):
        raise ValueError("run input labels must be unique")
    if {"plan", "manifest"} & set(labels):
        raise ValueError("plan and manifest are reserved run input labels")
    if not source_roots:
        raise ValueError("run contract requires at least one source revision root")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("plan must be a schema-version-1 object")
    rows = plan.get("rows")
    if not isinstance(rows, list):
        raise ValueError("plan rows must be an array")
    declared_rows = [row.get("name") for row in rows if isinstance(row, dict)]
    if len(declared_rows) != len(rows) or not all(
        isinstance(name, str) and name for name in declared_rows
    ):
        raise ValueError("every plan row needs a name")
    if len(declared_rows) != len(set(declared_rows)):
        raise ValueError("plan row names must be unique")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("manifest must be a schema-version-1 object")
    contract = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "declared_rows": declared_rows,
        "files": [
            _file_record("plan", plan_path, canonical_json=True),
            _file_record("manifest", manifest_path, canonical_json=True),
            *(_file_record(label, path) for label, path in inputs),
        ],
        "sources": [_source_record(root) for root in source_roots],
    }
    contract["contract_sha256"] = _contract_sha(contract)
    artifact_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(artifact_root / CONTRACT_NAME, contract)
    return contract


def summarize_contract(artifact_root: Path) -> dict:
    contract_path = artifact_root / CONTRACT_NAME
    try:
        contract = load_contract(contract_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return {"schema_version": 1, "status": "rejected", "errors": [str(error)], "rows": []}
    errors = validate_current_contract(contract)
    declared = set(contract["declared_rows"])
    result_paths = sorted(artifact_root.glob("*/result.json"))
    actual = {path.parent.name for path in result_paths}
    for name in sorted(declared - actual):
        errors.append(f"missing declared row: {name}")
    for name in sorted(actual - declared):
        errors.append(f"extra undeclared row: {name}")
    accepted_rows = []
    for path in result_paths:
        name = path.parent.name
        if name not in declared:
            continue
        try:
            result = read_row_result(artifact_root, name)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            errors.append(f"row {name}: malformed result: {error}")
            continue
        if (
            not isinstance(result, dict)
            or result.get("schema_version") != RESULT_SCHEMA_VERSION
        ):
            errors.append(
                f"row {name}: result is not schema version {RESULT_SCHEMA_VERSION}"
            )
            continue
        provenance = result.get("run_provenance")
        if not isinstance(provenance, dict):
            errors.append(f"row {name}: missing run provenance")
            continue
        if provenance.get("run_id") != contract["run_id"]:
            errors.append(f"row {name}: run_id mismatch")
        if provenance.get("contract_sha256") != contract["contract_sha256"]:
            errors.append(f"row {name}: contract hash mismatch")
        if provenance.get("plan_canonical_sha256") != contract["files"][0]["canonical_sha256"]:
            errors.append(f"row {name}: plan hash mismatch")
        if provenance.get("manifest_canonical_sha256") != contract["files"][1]["canonical_sha256"]:
            errors.append(f"row {name}: manifest hash mismatch")
        if provenance.get("files") != contract["files"]:
            errors.append(f"row {name}: input file provenance mismatch")
        if provenance.get("sources") != contract["sources"]:
            errors.append(f"row {name}: source revision/fingerprint mismatch")
        if not any(error.startswith(f"row {name}:") for error in errors):
            accepted_rows.append(name)
    return {
        "schema_version": 1,
        "status": "accepted" if not errors else "rejected",
        "run_id": contract["run_id"],
        "contract_sha256": contract["contract_sha256"],
        "declared_rows": contract["declared_rows"],
        "rows": accepted_rows if not errors else [],
        "errors": errors,
    }


def _label_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--artifact-root", type=Path, required=True)
    init.add_argument("--plan", type=Path, required=True)
    init.add_argument("--manifest", type=Path, required=True)
    init.add_argument("--input", type=_label_path, action="append", default=[])
    init.add_argument("--source-root", type=Path, action="append", default=[])
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--artifact-root", type=Path, required=True)
    summary.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "init":
            result = initialize_contract(
                args.artifact_root.resolve(), args.plan.resolve(), args.manifest.resolve(),
                args.input, args.source_root,
            )
        else:
            result = summarize_contract(args.artifact_root.resolve())
            if args.json_out:
                args.json_out.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(args.json_out, result)
    except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.action == "init" or result["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
