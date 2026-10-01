#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Record and compare deterministic gfx1250 B0-to-A0 translation hashes."""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

SHA256_RE = re.compile(r"[0-9a-f]{64}")
# Provenance is later rendered into a trusted PR comment. Keep these values
# single-token and canonical at the artifact boundary.
GIT_SHA_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
SDK_VERSION_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,127}")
SHA256SUM_RE = re.compile(r"(?P<digest>[0-9a-f]{64})  objects/(?P=digest)\.hsaco")
PROFILE = {
    "target": "gfx1250",
    "input_revision": "b0",
    "output_revision": "a0",
    "output_mode": "code-object",
    "verify_idempotence": True,
}
COMPARE_UNCHANGED = 0
ERROR = 1
COMPARE_CHANGED = 2
COMPARE_INCOMPATIBLE = 3


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "collect":
            collect_pairs(args)
            return 0
        if args.command == "finalize":
            finalize_manifest(args)
            return 0
        if args.command == "compare":
            return compare_manifests(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return ERROR
    raise AssertionError(f"unhandled command: {args.command}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    collect_parser = commands.add_parser(
        "collect", help="translate successful corpus inputs and record SHA-256 pairs"
    )
    collect_parser.add_argument("--translator", type=Path, required=True)
    collect_parser.add_argument("--corpus", type=Path, required=True)
    collect_parser.add_argument("--input-manifest", type=Path, required=True)
    collect_parser.add_argument("--expected-failures", type=Path, required=True)
    collect_parser.add_argument("--fragments", type=Path, required=True)
    collect_parser.add_argument("--workers", type=int, required=True)
    collect_parser.add_argument("--timeout", type=float, required=True)

    finalize_parser = commands.add_parser(
        "finalize", help="validate fragments and write one canonical manifest"
    )
    finalize_parser.add_argument("--fragments", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    finalize_parser.add_argument("--input-manifest", type=Path, required=True)
    finalize_parser.add_argument("--expected-failures", type=Path, required=True)
    finalize_parser.add_argument("--expected-rewrites", type=Path, required=True)
    finalize_parser.add_argument("--package-lock", type=Path, required=True)
    finalize_parser.add_argument("--source-commit", required=True)
    finalize_parser.add_argument("--corpus-commit", required=True)
    finalize_parser.add_argument("--rocm-sdk-version", required=True)

    compare_parser = commands.add_parser(
        "compare", help="compare a candidate manifest to a develop manifest"
    )
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--markdown", type=Path, required=True)
    compare_parser.add_argument("--max-details", type=int, default=50)

    return parser.parse_args(argv)


def collect_pairs(args: argparse.Namespace) -> None:
    translator = args.translator.resolve()
    if not translator.is_file() or not os.access(translator, os.X_OK):
        raise ValueError(f"translator is not executable: {args.translator}")
    args.translator = translator
    if args.workers <= 0:
        raise ValueError("--workers must be greater than zero")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
    if args.fragments.exists():
        if not args.fragments.is_dir():
            raise ValueError(
                f"translation fragment path is not a directory: {args.fragments}"
            )
        if any(args.fragments.iterdir()):
            raise ValueError(
                f"translation fragment directory is not empty: {args.fragments}"
            )
    args.fragments.mkdir(parents=True, exist_ok=True)
    corpus_inputs = _load_input_manifest(args.input_manifest)
    expected_failures = _load_expected_failures(args.expected_failures)
    unknown_failures = expected_failures - corpus_inputs
    if unknown_failures:
        raise ValueError(
            f"expected-failure manifest names {len(unknown_failures)} input(s) "
            f"absent from the corpus: {_shown(unknown_failures)}"
        )
    successful_inputs = sorted(corpus_inputs - expected_failures)

    errors = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_translate_and_hash, args, digest): digest
            for digest in successful_inputs
        }
        for future in as_completed(futures):
            digest = futures[future]
            try:
                _record_fragment(args.fragments, future.result())
            except (OSError, ValueError) as error:
                errors.append(f"{digest}: {error}")
            completed += 1
            if completed % 1000 == 0 or completed == len(successful_inputs):
                print(f"collected {completed}/{len(successful_inputs)} SHA-256 pairs")
    if errors:
        shown = "\n".join(errors[:20])
        suffix = f"\n... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(
            f"{len(errors)} translation hash collection(s) failed:\n{shown}{suffix}"
        )


def _translate_and_hash(args: argparse.Namespace, input_sha256: str) -> dict:
    source = args.corpus / "objects" / f"{input_sha256}.hsaco"
    if not source.is_file():
        raise ValueError(f"translation input is not a file: {source}")
    actual_input_sha256, input_size = _hash_and_size(source)
    if actual_input_sha256 != input_sha256:
        raise ValueError(
            f"input SHA-256 mismatch: expected {input_sha256}, got {actual_input_sha256}"
        )
    command = [
        str(args.translator),
        "--verify-idempotence",
        str(source),
        "--input-target",
        PROFILE["target"],
        "--output-target",
        PROFILE["target"],
        "--input-revision",
        PROFILE["input_revision"],
        "--output-revision",
        PROFILE["output_revision"],
        "--output-mode",
        PROFILE["output_mode"],
    ]
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as error_output:
        process = subprocess.Popen(command, stdout=output, stderr=error_output)
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait()
            raise ValueError(
                f"translation timed out after {args.timeout:g} seconds"
            ) from error
        if returncode:
            error_output.seek(0)
            stderr = error_output.read(64 * 1024).decode("utf-8", errors="replace")
            raise ValueError(
                f"translation failed with status {returncode}: {stderr.strip() or '<empty>'}"
            )
        output.seek(0)
        output_sha256 = hashlib.sha256()
        output_size = 0
        while chunk := output.read(1024 * 1024):
            output_sha256.update(chunk)
            output_size += len(chunk)
    if output_size == 0:
        raise ValueError("translation produced an empty output")
    return {
        "input_bytes": input_size,
        "input_sha256": input_sha256,
        "output_bytes": output_size,
        "output_sha256": output_sha256.hexdigest(),
    }


def finalize_manifest(args: argparse.Namespace) -> None:
    _validate_git_sha(args.source_commit, "source commit")
    _validate_git_sha(args.corpus_commit, "corpus commit")
    _validate_sdk_version(args.rocm_sdk_version, "ROCm SDK version")

    pairs = _load_fragments(args.fragments)
    corpus_inputs = _load_input_manifest(args.input_manifest)
    expected_failures = _load_expected_failures(args.expected_failures)
    unknown_failures = expected_failures - corpus_inputs
    if unknown_failures:
        raise ValueError(
            f"expected-failure manifest names {len(unknown_failures)} input(s) "
            f"absent from the corpus: {_shown(unknown_failures)}"
        )

    expected_pairs = corpus_inputs - expected_failures
    actual_pairs = set(pairs)
    missing = expected_pairs - actual_pairs
    extra = actual_pairs - expected_pairs
    if missing or extra:
        raise ValueError(
            "translation fragments do not cover the successful corpus inputs: "
            f"missing={len(missing)} [{_shown(missing)}], "
            f"extra={len(extra)} [{_shown(extra)}]"
        )

    payload = {
        "excluded_inputs": sorted(expected_failures),
        "pairs": [pairs[digest] for digest in sorted(pairs)],
        "profile": PROFILE,
        "provenance": {
            "corpus_commit": args.corpus_commit,
            "expected_failures_sha256": _sha256_file(args.expected_failures),
            "expected_rewrites_sha256": _sha256_file(args.expected_rewrites),
            "input_manifest_sha256": _sha256_file(args.input_manifest),
            "package_lock_sha256": _sha256_file(args.package_lock),
            "rocm_sdk_version": args.rocm_sdk_version,
            "rocm_systems_commit": args.source_commit,
        },
        "schema_version": 1,
    }
    _write_json(args.output, payload)
    print(
        f"wrote {len(pairs)} gfx1250 B0-to-A0 SHA-256 pairs to {args.output} "
        f"({len(expected_failures)} expected failure(s) excluded)"
    )


def compare_manifests(args: argparse.Namespace) -> int:
    if args.max_details <= 0:
        raise ValueError("--max-details must be greater than zero")
    baseline = _load_pair_manifest(args.baseline)
    candidate = _load_pair_manifest(args.candidate)

    compatibility_fields = (
        "corpus_commit",
        "input_manifest_sha256",
        "package_lock_sha256",
        "rocm_sdk_version",
    )
    incompatible = [
        field
        for field in compatibility_fields
        if baseline["provenance"][field] != candidate["provenance"][field]
    ]
    baseline_pairs = _pairs_by_input(baseline)
    candidate_pairs = _pairs_by_input(candidate)
    removed = sorted(set(baseline_pairs) - set(candidate_pairs))
    added = sorted(set(candidate_pairs) - set(baseline_pairs))

    if incompatible or removed or added:
        lines = [
            "## gfx1250 B0-to-A0 translation baseline",
            "",
            "> [!WARNING]",
            "> The PR result is not directly comparable to the latest develop baseline.",
            "",
        ]
        if incompatible:
            lines.extend(
                [
                    "Changed provenance:",
                    "",
                    "| Field | develop | PR |",
                    "| --- | --- | --- |",
                ]
            )
            for field in incompatible:
                lines.append(
                    f"| `{field}` | `{baseline['provenance'][field]}` | "
                    f"`{candidate['provenance'][field]}` |"
                )
            lines.append("")
        if removed or added:
            lines.extend(
                [
                    f"Input-set difference: {len(removed)} removed, {len(added)} added.",
                    "",
                ]
            )
        _write_markdown(args.markdown, lines)
        return COMPARE_INCOMPATIBLE

    changed = [
        digest
        for digest in sorted(baseline_pairs)
        if baseline_pairs[digest]["output_sha256"]
        != candidate_pairs[digest]["output_sha256"]
    ]
    baseline_commit = baseline["provenance"]["rocm_systems_commit"]
    candidate_commit = candidate["provenance"]["rocm_systems_commit"]
    if not changed:
        _write_markdown(
            args.markdown,
            [
                "## gfx1250 B0-to-A0 translation baseline",
                "",
                f"No translated output SHA-256 changes across {len(baseline_pairs)} "
                "successful corpus inputs.",
                "",
                f"- develop source: `{baseline_commit}`",
                f"- PR source: `{candidate_commit}`",
                "",
            ],
        )
        return COMPARE_UNCHANGED

    lines = [
        "## gfx1250 B0-to-A0 translation baseline",
        "",
        "> [!WARNING]",
        f"> This PR changes {len(changed)} translated output SHA-256 value(s).",
        "",
        f"Compared `{candidate_commit}` against develop baseline `{baseline_commit}`.",
        "",
        "| Input SHA-256 | Input bytes | develop output | develop bytes | "
        "PR output | PR bytes |",
        "| --- | ---: | --- | ---: | --- | ---: |",
    ]
    for digest in changed[: args.max_details]:
        lines.append(
            f"| `{digest}` | {baseline_pairs[digest]['input_bytes']:,} | "
            f"`{baseline_pairs[digest]['output_sha256']}` | "
            f"{baseline_pairs[digest]['output_bytes']:,} | "
            f"`{candidate_pairs[digest]['output_sha256']}` | "
            f"{candidate_pairs[digest]['output_bytes']:,} |"
        )
    if len(changed) > args.max_details:
        lines.extend(
            [
                "",
                f"Showing the first {args.max_details} of {len(changed)} changes.",
            ]
        )
    lines.append("")
    _write_markdown(args.markdown, lines)
    return COMPARE_CHANGED


def _record_fragment(directory: Path, record: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{record['input_sha256']}.json"
    fd, temporary_name = tempfile.mkstemp(prefix=".pair-", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(record, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != record:
                raise ValueError(
                    f"conflicting translation outputs for {record['input_sha256']}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _load_fragments(directory: Path) -> dict[str, dict]:
    if not directory.is_dir():
        raise ValueError(f"translation fragment directory does not exist: {directory}")
    pairs = {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        _validate_pair_record(record, str(path))
        digest = record["input_sha256"]
        if path.name != f"{digest}.json":
            raise ValueError(f"{path} name does not match its input SHA-256")
        if digest in pairs:
            raise ValueError(f"duplicate translation fragment for {digest}")
        pairs[digest] = record
    return pairs


def _load_input_manifest(path: Path) -> set[str]:
    inputs = set()
    for line_number, line in enumerate(
        path.read_text(encoding="ascii").splitlines(), start=1
    ):
        match = SHA256SUM_RE.fullmatch(line)
        if match is None:
            raise ValueError(
                f"{path}:{line_number} is not a canonical SHA256SUMS entry"
            )
        digest = match.group("digest")
        if digest in inputs:
            raise ValueError(f"{path}:{line_number} repeats {digest}")
        inputs.add(digest)
    if not inputs:
        raise ValueError(f"{path} must not be empty")
    return inputs


def _load_expected_failures(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = payload.get("expected_failures")
    if not isinstance(failures, dict):
        raise ValueError(f"{path} field 'expected_failures' must be an object")
    invalid = [
        digest
        for digest in failures
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)
    ]
    if invalid:
        raise ValueError(f"{path} contains an invalid expected-failure SHA-256")
    return set(failures)


def _load_pair_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "excluded_inputs",
        "pairs",
        "profile",
        "provenance",
        "schema_version",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"{path} must contain exactly {sorted(required)}")
    if payload["schema_version"] != 1:
        raise ValueError(f"{path} field 'schema_version' must be 1")
    if payload["profile"] != PROFILE:
        raise ValueError(f"{path} does not describe the gfx1250 B0-to-A0 profile")
    provenance_fields = {
        "corpus_commit",
        "expected_failures_sha256",
        "expected_rewrites_sha256",
        "input_manifest_sha256",
        "package_lock_sha256",
        "rocm_sdk_version",
        "rocm_systems_commit",
    }
    provenance = payload["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != provenance_fields:
        raise ValueError(f"{path} has invalid provenance fields")
    for field in (
        "expected_failures_sha256",
        "expected_rewrites_sha256",
        "input_manifest_sha256",
        "package_lock_sha256",
    ):
        if not isinstance(provenance[field], str) or not SHA256_RE.fullmatch(
            provenance[field]
        ):
            raise ValueError(f"{path} provenance field {field!r} is not SHA-256")
    for field in ("corpus_commit", "rocm_systems_commit"):
        _validate_git_sha(provenance[field], f"{path} provenance field {field!r}")
    _validate_sdk_version(
        provenance["rocm_sdk_version"],
        f"{path} provenance field 'rocm_sdk_version'",
    )
    excluded = payload["excluded_inputs"]
    if (
        not isinstance(excluded, list)
        or excluded != sorted(set(excluded))
        or any(
            not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)
            for digest in excluded
        )
    ):
        raise ValueError(f"{path} field 'excluded_inputs' is not canonical")
    pairs = _pairs_by_input(payload)
    if list(pairs) != sorted(pairs):
        raise ValueError(f"{path} field 'pairs' is not sorted by input SHA-256")
    overlap = set(excluded) & set(pairs)
    if overlap:
        raise ValueError(f"{path} both excludes and records {_shown(overlap)}")
    return payload


def _pairs_by_input(manifest: dict) -> dict[str, dict]:
    records = manifest["pairs"]
    if not isinstance(records, list):
        raise ValueError("manifest field 'pairs' must be a list")
    pairs = {}
    for record in records:
        _validate_pair_record(record, "manifest pair")
        digest = record.get("input_sha256")
        if digest in pairs:
            raise ValueError(f"manifest repeats input SHA-256 {digest}")
        pairs[digest] = record
    return pairs


def _validate_pair_record(record: dict, description: str) -> None:
    expected_fields = {
        "input_bytes",
        "input_sha256",
        "output_bytes",
        "output_sha256",
    }
    if not isinstance(record, dict) or set(record) != expected_fields:
        raise ValueError(
            f"{description} must contain exactly {sorted(expected_fields)}"
        )
    for field in ("input_sha256", "output_sha256"):
        if not isinstance(record[field], str) or not SHA256_RE.fullmatch(record[field]):
            raise ValueError(f"{description} field {field!r} must be lowercase SHA-256")
    for field in ("input_bytes", "output_bytes"):
        if (
            isinstance(record[field], bool)
            or not isinstance(record[field], int)
            or record[field] <= 0
        ):
            raise ValueError(
                f"{description} field {field!r} must be a positive integer"
            )


def _hash_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _sha256_file(path: Path) -> str:
    return _hash_and_size(path)[0]


def _validate_git_sha(value: str, description: str) -> None:
    if not isinstance(value, str) or GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a full lowercase Git SHA")


def _validate_sdk_version(value: str, description: str) -> None:
    if not isinstance(value, str) or SDK_VERSION_RE.fullmatch(value) is None:
        raise ValueError(f"{description} contains unsupported characters")


def _shown(digests: set[str]) -> str:
    ordered = sorted(digests)
    shown = ", ".join(ordered[:3])
    return f"{shown}, ..." if len(ordered) > 3 else shown


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
