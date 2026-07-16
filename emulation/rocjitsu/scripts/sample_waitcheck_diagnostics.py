#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Draw a reproducible hash-priority sample from waitcheck diagnostic JSONL."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
from typing import Any

SCHEMA = "rj-waitcheck-diagnostic-v1"
DEFAULT_SEED = "bd-jps.45-gfx942-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def diagnostic_identity(record: dict[str, Any]) -> dict[str, Any]:
    """Mirror waitcheck's static key while retaining per-kernel provenance."""
    return {
        "input": record["input"],
        "target": record["target"],
        "code_object_index": record["code_object_index"],
        "kernel_name": record["kernel_name"],
        "kernel_entry": record["kernel_entry"],
        "counter": record["counter"],
        "access": record["access"],
        "register": record["register"],
        "section": record["section"],
        "section_offset": record["section_offset"],
        "file_offset": record["file_offset"],
        "producer_section_offset": record["producer_section_offset"],
        "producer_file_offset": record["producer_file_offset"],
        "required_count": record["required_count"],
        "instruction": record["instruction"],
        "producer_instruction": record["producer_instruction"],
    }


def read_population(path: pathlib.Path) -> tuple[list[dict[str, Any]], int]:
    unique: dict[str, dict[str, Any]] = {}
    raw_count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if record.get("schema") != SCHEMA:
                raise ValueError(
                    f"{path}:{line_number}: expected schema {SCHEMA!r}, "
                    f"got {record.get('schema')!r}"
                )
            try:
                identity = canonical_json(diagnostic_identity(record))
            except KeyError as error:
                raise ValueError(
                    f"{path}:{line_number}: missing field {error.args[0]!r}"
                ) from error

            # Input order is intentionally irrelevant. If duplicate identities
            # differ only in auxiliary fields, retain the canonical minimum.
            previous = unique.get(identity)
            if previous is None or canonical_json(record) < canonical_json(previous):
                unique[identity] = record
    return [unique[key] for key in sorted(unique)], raw_count


def selection_priority(seed: str, record: dict[str, Any]) -> tuple[str, str]:
    identity = canonical_json(diagnostic_identity(record))
    digest = hashlib.sha256(seed.encode() + b"\0" + identity.encode()).hexdigest()
    return digest, identity


def distribution(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = collections.Counter(str(record[field]) for record in records)
    return dict(sorted(counts.items()))


def code_object_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = collections.Counter(
        f"{record['target']}:{record['code_object_index']}" for record in records
    )
    return dict(sorted(counts.items()))


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=pathlib.Path, help="lossless diagnostic JSONL population"
    )
    parser.add_argument("output", type=pathlib.Path, help="selected diagnostic JSONL")
    parser.add_argument(
        "--count", type=int, default=50, help="sample size (default: 50)"
    )
    parser.add_argument("--seed", default=DEFAULT_SEED, help="recorded selection seed")
    parser.add_argument(
        "--unique-output",
        type=pathlib.Path,
        help="optional canonical JSONL containing the deduplicated population",
    )
    parser.add_argument(
        "--metadata",
        type=pathlib.Path,
        help="metadata JSON path (default: OUTPUT.metadata.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")

    population, raw_count = read_population(args.input)
    if len(population) < args.count:
        raise ValueError(
            f"population has only {len(population)} unique diagnostics; "
            f"cannot select {args.count}"
        )

    if args.unique_output:
        args.unique_output.parent.mkdir(parents=True, exist_ok=True)
        with args.unique_output.open("w", encoding="utf-8") as stream:
            for record in population:
                stream.write(canonical_json(record))
                stream.write("\n")

    ranked = sorted(
        (selection_priority(args.seed, record), record) for record in population
    )
    selected: list[dict[str, Any]] = []
    for rank, ((priority, identity), original) in enumerate(ranked[: args.count], 1):
        record = dict(original)
        record["sample"] = {
            "rank": rank,
            "seed": args.seed,
            "algorithm": "sha256(seed + NUL + canonical diagnostic identity), lowest first",
            "priority": priority,
            "identity_sha256": hashlib.sha256(identity.encode()).hexdigest(),
            "population_size": len(population),
        }
        selected.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in selected:
            stream.write(canonical_json(record))
            stream.write("\n")

    metadata_path = args.metadata or args.output.with_suffix(
        args.output.suffix + ".metadata.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "rj-waitcheck-diagnostic-sample-v1",
        "algorithm": "sha256(seed + NUL + canonical diagnostic identity), lowest first",
        "seed": args.seed,
        "requested_count": args.count,
        "raw_population_count": raw_count,
        "unique_population_count": len(population),
        "duplicate_count": raw_count - len(population),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "unique_output": str(args.unique_output) if args.unique_output else None,
        "unique_output_sha256": (
            sha256_file(args.unique_output) if args.unique_output else None
        ),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "identity_fields": list(diagnostic_identity(population[0])),
        "population_distribution": {
            "counter": distribution(population, "counter"),
            "access": distribution(population, "access"),
            "code_object": code_object_distribution(population),
        },
        "sample_distribution": {
            "counter": distribution(selected, "counter"),
            "access": distribution(selected, "access"),
            "code_object": code_object_distribution(selected),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    print(
        f"sampled {len(selected)} of {len(population)} unique diagnostics "
        f"({raw_count} raw) with seed {args.seed!r}"
    )
    print(f"sample: {args.output}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
