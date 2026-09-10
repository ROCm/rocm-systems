# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish one raw Rocjitsu benchmark run as static dashboard data."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import tempfile
from typing import Any

SCHEMA_VERSION = 1
INDEX_FORMAT = "rocjitsu.dashboard.index"
CATALOG_FORMAT = "rocjitsu.dashboard.catalog"
RUN_FORMAT = "rocjitsu.dashboard.run"
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CATALOG_NAME = re.compile(r"^catalog-([0-9a-f]{12})\.json$")
SHA = re.compile(r"^[0-9a-fA-F]{40}$")


class PublishError(ValueError):
    """A malformed input or conflicting immutable dashboard resource."""


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublishError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise PublishError(f"non-finite JSON value {value}")


def load_json_document(path: str | Path) -> Any:
    """Read exactly one finite, duplicate-key-free JSON document."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
        return json.loads(
            text,
            object_pairs_hook=_object,
            parse_constant=_constant,
            parse_float=lambda value: _finite_float(value),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, PublishError) as error:
        raise PublishError(f"cannot read JSON document {source}: {error}") from error


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        _constant(value)
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublishError(f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublishError(f"{name} must be a non-empty string")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PublishError(f"{name} must be a positive integer")
    return value


def _timestamp(value: Any, name: str) -> str:
    value = _string(value, name)
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PublishError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise PublishError(f"{name} must include a timezone")
    return parsed.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PublishError(f"value is not finite JSON: {error}") from error


def _definition(test: Mapping[str, Any]) -> dict[str, Any]:
    logical_id = _string(test.get("logicalTestId"), "test logicalTestId")
    data_type = _string(test.get("dataType"), f"test {logical_id} dataType")
    if data_type != data_type.lower():
        raise PublishError(f"test {logical_id} dataType must be lowercase")
    problem = test.get("problem")
    if problem is not None and not isinstance(problem, Mapping):
        raise PublishError(f"test {logical_id} problem must be an object or null")
    return {
        "id": logical_id,
        "suite": _string(test.get("suite"), f"test {logical_id} suite"),
        "name": _string(test.get("name"), f"test {logical_id} name"),
        "operation": _string(
            test.get("operation"), f"test {logical_id} operation"
        ),
        "dataType": data_type,
        "problem": dict(problem) if problem is not None else None,
        "execMode": _string(test.get("execMode"), f"test {logical_id} execMode"),
        "numThreads": _positive_integer(
            test.get("numThreads"), f"test {logical_id} numThreads"
        ),
    }


def _catalog_definition(test: Mapping[str, Any]) -> dict[str, Any]:
    logical_id = _string(
        test.get("id", test.get("logicalTestId")), "catalog test ID"
    )
    data_type = _string(test.get("dataType"), f"catalog test {logical_id} dataType")
    if data_type != data_type.lower():
        raise PublishError(f"catalog test {logical_id} dataType must be lowercase")
    problem = test.get("problem")
    if not isinstance(problem, Mapping):
        raise PublishError(f"catalog test {logical_id} has no problem definition")
    return {
        "id": logical_id,
        "suite": _string(test.get("suite"), f"catalog test {logical_id} suite"),
        "name": _string(test.get("name"), f"catalog test {logical_id} name"),
        "operation": _string(
            test.get("operation"), f"catalog test {logical_id} operation"
        ),
        "dataType": data_type,
        "problem": dict(problem),
        "execMode": _string(
            test.get("execMode"), f"catalog test {logical_id} execMode"
        ),
        "numThreads": _positive_integer(
            test.get("numThreads"), f"catalog test {logical_id} numThreads"
        ),
    }


def _details(provenance: Mapping[str, Any], source_ref: str | None) -> list[dict[str, Any]]:
    fields = (
        ("sourceRef", "Source ref", source_ref),
        ("buildType", "Build type", provenance.get("buildType")),
        ("rocmSdkVersion", "ROCm SDK", provenance.get("rocmSdkVersion")),
        ("pythonVersion", "Python", provenance.get("pythonVersion")),
        ("torchVersion", "PyTorch", provenance.get("torchVersion")),
        ("tritonVersion", "Triton", provenance.get("tritonVersion")),
        ("tritonCommitSha", "Triton commit", provenance.get("tritonCommitSha")),
        (
            "tensileLiteCommitSha",
            "TensileLite commit",
            provenance.get("tensileLiteCommitSha"),
        ),
    )
    result = []
    for key, label, value in fields:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise PublishError(f"provenance {key} must be a string or number")
        if isinstance(value, float) and not math.isfinite(value):
            raise PublishError(f"provenance {key} must be finite")
        if isinstance(value, str) and not value.strip():
            raise PublishError(f"provenance {key} must not be empty")
        result.append({"key": key, "label": label, "value": value})
    return result


def _validate_result(test: Mapping[str, Any], definition: Mapping[str, Any]) -> None:
    logical_id = definition["id"]
    target = _string(test.get("target"), f"test {logical_id} target")
    if test.get("testId") != f"{target}:{logical_id}":
        raise PublishError(f"test {logical_id} has an invalid testId")
    observed = _definition(test)
    for field in ("suite", "name", "operation", "dataType", "execMode", "numThreads"):
        if observed[field] != definition[field]:
            raise PublishError(f"test {logical_id} conflicts with catalog field {field}")
    if observed["problem"] is not None and observed["problem"] != definition["problem"]:
        raise PublishError(f"test {logical_id} conflicts with catalog field problem")

    status = test.get("status")
    if status not in ("completed", "failed", "timeout"):
        raise PublishError(f"test {logical_id} has invalid status {status!r}")
    duration = test.get("durationSeconds")
    if status == "completed":
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise PublishError(f"completed test {logical_id} needs a positive duration")
        if test.get("error") is not None:
            raise PublishError(f"completed test {logical_id} cannot have an error")
    elif duration is not None:
        raise PublishError(f"incomplete test {logical_id} must have a null duration")
    if test.get("timedOut") is not (status == "timeout"):
        raise PublishError(f"test {logical_id} has inconsistent timeout metadata")
    exit_code = test.get("exitCode")
    if exit_code is not None and (
        isinstance(exit_code, bool) or not isinstance(exit_code, int)
    ):
        raise PublishError(f"test {logical_id} has an invalid exit code")
    error = test.get("error")
    if error is not None:
        _string(error, f"test {logical_id} error")


def normalize_run(
    raw_run: Mapping[str, Any],
    *,
    run_id: str,
    trigger: str,
    branch: str,
    environment_id: str,
    catalog_tests: Mapping[str, Mapping[str, Any]] | None = None,
    machine_id: str | None = None,
    commit_order: int | None = None,
    commit_message: str | None = None,
    source_ref: str | None = None,
    expected_sha: str | None = None,
) -> dict[str, Any]:
    """Normalize a finalized raw schema-v1 run to dashboard schema v1."""

    raw_run = _mapping(raw_run, "raw run")
    if raw_run.get("schemaVersion") != SCHEMA_VERSION:
        raise PublishError("raw run must use schema version 1")
    if raw_run.get("status") not in ("completed", "failed"):
        raise PublishError("raw run must be finalized")
    if not RUN_ID.fullmatch(run_id):
        raise PublishError(f"unsafe run ID {run_id!r}")
    if trigger not in ("auto", "manual"):
        raise PublishError("trigger must be auto or manual")
    if branch != "develop":
        raise PublishError("dashboard publication requires branch develop")
    environment_id = _string(environment_id, "environment ID")
    if machine_id is not None:
        machine_id = _string(machine_id, "machine ID")
    if commit_message is not None:
        commit_message = _string(commit_message, "commit message")
    if source_ref is not None:
        source_ref = _string(source_ref, "source ref")
    if commit_order is not None and (
        isinstance(commit_order, bool) or not isinstance(commit_order, int) or commit_order < 0
    ):
        raise PublishError("commit order must be a non-negative integer")

    raw_provenance = _mapping(raw_run.get("provenance"), "raw provenance")
    if raw_provenance.get("dirty") is not False:
        raise PublishError("raw run must come from a clean Rocjitsu checkout")
    sha = _string(raw_provenance.get("rocjitsuCommitSha"), "Rocjitsu commit SHA")
    if not SHA.fullmatch(sha):
        raise PublishError("Rocjitsu commit SHA must contain 40 hexadecimal digits")
    if expected_sha is not None:
        expected_sha = _string(expected_sha, "expected SHA")
        if not SHA.fullmatch(expected_sha):
            raise PublishError("expected SHA must contain 40 hexadecimal digits")
        if sha.lower() != expected_sha.lower():
            raise PublishError(
                f"raw Rocjitsu commit SHA {sha} does not match expected SHA {expected_sha}"
            )
    configuration = _mapping(raw_run.get("configuration"), "raw configuration")
    configuration_id = _string(configuration.get("id"), "configuration ID")
    raw_tests = raw_run.get("tests")
    if not isinstance(raw_tests, list) or not raw_tests:
        raise PublishError("raw run must contain benchmark tests")

    definitions = dict(catalog_tests or {})
    candidates: dict[str, dict[str, Any]] = {}
    for position, value in enumerate(raw_tests):
        test = _mapping(value, f"raw test {position}")
        candidate = _definition(test)
        if test.get("status") == "completed" and candidate["problem"] is None:
            raise PublishError(
                f"completed test {candidate['id']} has no problem definition"
            )
        if candidate["problem"] is None:
            continue
        previous = candidates.get(candidate["id"])
        if previous is not None and previous != candidate:
            raise PublishError(f"run has conflicting definitions for {candidate['id']}")
        candidates[candidate["id"]] = candidate
    for logical_id, candidate in candidates.items():
        previous = definitions.get(logical_id)
        if previous is not None and dict(previous) != candidate:
            raise PublishError(f"benchmark definition conflicts for {logical_id}")
        definitions[logical_id] = candidate

    normalized_tests = []
    test_ids: set[str] = set()
    for value in raw_tests:
        test = _mapping(value, "raw test")
        logical_id = _string(test.get("logicalTestId"), "test logicalTestId")
        definition = definitions.get(logical_id)
        if definition is None:
            raise PublishError(
                f"failed test {logical_id} has no existing catalog definition"
            )
        _validate_result(test, definition)
        test_id = str(test["testId"])
        if test_id in test_ids:
            raise PublishError(f"raw run contains duplicate test {test_id}")
        test_ids.add(test_id)
        normalized_tests.append(
            {
                "testId": test_id,
                "logicalTestId": logical_id,
                "suite": definition["suite"],
                "name": definition["name"],
                "target": test["target"],
                "operation": definition["operation"],
                "dataType": definition["dataType"],
                "problem": definition["problem"],
                "execMode": definition["execMode"],
                "numThreads": definition["numThreads"],
                "durationSeconds": test.get("durationSeconds"),
                "status": test.get("status"),
                "exitCode": test.get("exitCode"),
                "timedOut": test.get("timedOut"),
                "error": test.get("error"),
            }
        )

    result: dict[str, Any] = {
        "format": RUN_FORMAT,
        "schemaVersion": SCHEMA_VERSION,
        "runId": run_id,
        "timestamp": _timestamp(raw_run.get("finishedAt"), "raw finishedAt"),
        "commitTimestamp": _timestamp(
            raw_provenance.get("rocjitsuCommitTimestamp"), "commit timestamp"
        ),
        "trigger": trigger,
        "branch": branch,
        "environmentId": environment_id,
        "configurationId": configuration_id,
        "provenance": {
            "rocjitsuCommitSha": sha.lower(),
            "details": _details(raw_provenance, source_ref),
        },
        "tests": normalized_tests,
    }
    if machine_id is not None:
        result["machineId"] = machine_id
    if commit_order is not None:
        result["commitOrder"] = commit_order
    if commit_message is not None:
        result["provenance"]["commitMessage"] = commit_message
    return result


def _validate_catalog(value: Any) -> dict[str, Any]:
    catalog = dict(_mapping(value, "dashboard catalog"))
    if catalog.get("format") != CATALOG_FORMAT or catalog.get("schemaVersion") != 1:
        raise PublishError("dashboard catalog must use schema version 1")
    _string(catalog.get("repository"), "catalog repository")
    if not isinstance(catalog.get("isDemo"), bool):
        raise PublishError("catalog isDemo must be a boolean")
    targets = catalog.get("targets")
    if not isinstance(targets, list) or any(
        not isinstance(target, str) or not target for target in targets
    ):
        raise PublishError("catalog targets must be an array of names")
    if len(set(targets)) != len(targets):
        raise PublishError("catalog targets must be unique")
    tests = catalog.get("testCatalog")
    if not isinstance(tests, list):
        raise PublishError("catalog testCatalog must be an array")
    seen: set[str] = set()
    for test in tests:
        definition = _catalog_definition(_mapping(test, "catalog test"))
        if definition["id"] in seen:
            raise PublishError(f"duplicate catalog test {definition['id']}")
        seen.add(definition["id"])
    return catalog


def _validate_dashboard_run(
    value: Any, expected_run_id: str, catalog: Mapping[str, Any]
) -> None:
    run = _mapping(value, f"dashboard run {expected_run_id}")
    if run.get("format") != RUN_FORMAT or run.get("schemaVersion") != 1:
        raise PublishError(f"dashboard run {expected_run_id} must use schema version 1")
    if run.get("runId") != expected_run_id:
        raise PublishError(f"dashboard run {expected_run_id} has a mismatched runId")
    _timestamp(run.get("timestamp"), f"run {expected_run_id} timestamp")
    _timestamp(run.get("commitTimestamp"), f"run {expected_run_id} commitTimestamp")
    if run.get("branch") != "develop" or run.get("trigger") not in ("auto", "manual"):
        raise PublishError(f"dashboard run {expected_run_id} is not an official develop run")
    _string(run.get("environmentId"), f"run {expected_run_id} environmentId")
    _string(run.get("configurationId"), f"run {expected_run_id} configurationId")
    provenance = _mapping(run.get("provenance"), f"run {expected_run_id} provenance")
    sha = _string(
        provenance.get("rocjitsuCommitSha"), f"run {expected_run_id} commit SHA"
    )
    if not SHA.fullmatch(sha):
        raise PublishError(f"dashboard run {expected_run_id} has an invalid commit SHA")
    definitions = {test["id"]: test for test in catalog["testCatalog"]}
    targets = set(catalog["targets"])
    tests = run.get("tests")
    if not isinstance(tests, list) or not tests:
        raise PublishError(f"dashboard run {expected_run_id} has no tests")
    seen: set[str] = set()
    for value in tests:
        test = _mapping(value, f"run {expected_run_id} test")
        logical_id = _string(test.get("logicalTestId"), "test logicalTestId")
        definition = definitions.get(logical_id)
        if definition is None:
            raise PublishError(
                f"dashboard run {expected_run_id} references unknown test {logical_id}"
            )
        if test.get("target") not in targets:
            raise PublishError(
                f"dashboard run {expected_run_id} references unknown target {test.get('target')!r}"
            )
        if not isinstance(test.get("problem"), Mapping):
            raise PublishError(
                f"dashboard run {expected_run_id} test {logical_id} has no problem definition"
            )
        _validate_result(test, definition)
        test_id = str(test.get("testId"))
        if test_id in seen:
            raise PublishError(f"dashboard run {expected_run_id} has duplicate test {test_id}")
        seen.add(test_id)


def _load_index(data_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path = data_dir / "index.json"
    if not path.exists():
        return None, None
    index = dict(_mapping(load_json_document(path), "dashboard index"))
    if index.get("format") != INDEX_FORMAT or index.get("schemaVersion") != 1:
        raise PublishError("dashboard index must use schema version 1")
    _timestamp(index.get("generatedAt"), "index generatedAt")
    match = CATALOG_NAME.fullmatch(str(index.get("catalog", "")))
    if match is None:
        raise PublishError("index catalog is not content-addressed")
    references = index.get("runs")
    if not isinstance(references, list):
        raise PublishError("index runs must be an array")
    run_ids: set[str] = set()
    paths: set[str] = set()
    for reference in references:
        reference = _mapping(reference, "run reference")
        run_id = _string(reference.get("runId"), "run reference ID")
        run_path = reference.get("path")
        if not RUN_ID.fullmatch(run_id) or run_path != f"runs/{run_id}.json":
            raise PublishError(f"invalid run reference for {run_id}")
        if run_id in run_ids or run_path in paths:
            raise PublishError(f"duplicate run reference for {run_id}")
        run_ids.add(run_id)
        paths.add(run_path)
    catalog_path = data_dir / str(index["catalog"])
    catalog_bytes = catalog_path.read_bytes() if catalog_path.is_file() else b""
    if not catalog_bytes or hashlib.sha256(catalog_bytes).hexdigest()[:12] != match.group(1):
        raise PublishError("indexed catalog is missing or has the wrong content hash")
    catalog = _validate_catalog(load_json_document(catalog_path))
    for reference in references:
        run_path = data_dir / reference["path"]
        if not run_path.is_file():
            raise PublishError(f"indexed dashboard run is missing: {run_path}")
        _validate_dashboard_run(
            load_json_document(run_path), str(reference["runId"]), catalog
        )
    return index, catalog


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _write_immutable(path: Path, value: Any, description: str) -> bool:
    content = _canonical(value)
    if path.exists():
        if load_json_document(path) != value:
            raise PublishError(f"conflicting immutable {description}: {path}")
        return False
    _write_atomic(path, content)
    return True


def publish(
    raw_run: Mapping[str, Any],
    *,
    data_dir: str | Path,
    run_id: str,
    repository: str,
    environment_id: str,
    trigger: str,
    branch: str,
    is_demo: bool = False,
    machine_id: str | None = None,
    commit_order: int | None = None,
    commit_message: str | None = None,
    generated_at: str | None = None,
    source_ref: str | None = None,
    expected_sha: str | None = None,
) -> dict[str, Any]:
    """Publish immutable resources first and replace the discovery index last."""

    root = Path(data_dir).expanduser().resolve()
    repository = _string(repository, "repository")
    if not isinstance(is_demo, bool):
        raise PublishError("is_demo must be a boolean")
    index, old_catalog = _load_index(root)
    if old_catalog is not None and (
        old_catalog["repository"] != repository or old_catalog["isDemo"] is not is_demo
    ):
        raise PublishError("repository or isDemo conflicts with the existing catalog")
    old_definitions = {
        test["id"]: test for test in (old_catalog or {}).get("testCatalog", [])
    }
    run = normalize_run(
        raw_run,
        run_id=run_id,
        trigger=trigger,
        branch=branch,
        environment_id=environment_id,
        catalog_tests=old_definitions,
        machine_id=machine_id,
        commit_order=commit_order,
        commit_message=commit_message,
        source_ref=source_ref,
        expected_sha=expected_sha,
    )

    definitions = dict(old_definitions)
    for test in run["tests"]:
        definition = _catalog_definition(test)
        previous = definitions.get(definition["id"])
        if previous is not None and previous != definition:
            raise PublishError(f"benchmark definition conflicts for {definition['id']}")
        definitions[definition["id"]] = definition
    targets = set((old_catalog or {}).get("targets", []))
    targets.update(test["target"] for test in run["tests"])
    catalog = {
        "format": CATALOG_FORMAT,
        "schemaVersion": SCHEMA_VERSION,
        "repository": repository,
        "isDemo": is_demo,
        "targets": sorted(targets),
        "testCatalog": [definitions[key] for key in sorted(definitions)],
    }
    catalog_content = _canonical(catalog)
    catalog_name = f"catalog-{hashlib.sha256(catalog_content).hexdigest()[:12]}.json"
    run_path = root / "runs" / f"{run_id}.json"
    catalog_path = root / catalog_name

    references = list((index or {}).get("runs", []))
    descriptor = {"runId": run_id, "path": f"runs/{run_id}.json"}
    matching = [item for item in references if item["runId"] == run_id]
    if matching and matching[0] != descriptor:
        raise PublishError(f"run ID {run_id} has a conflicting index descriptor")
    if run_path.exists() and load_json_document(run_path) != run:
        raise PublishError(f"conflicting immutable dashboard run: {run_path}")
    if catalog_path.exists() and catalog_path.read_bytes() != catalog_content:
        raise PublishError(f"conflicting content-addressed catalog: {catalog_path}")

    run_changed = _write_immutable(run_path, run, "dashboard run")
    catalog_changed = _write_immutable(catalog_path, catalog, "dashboard catalog")
    index_changed = not matching or (index is not None and index["catalog"] != catalog_name)
    if index is None or index_changed:
        if not matching:
            references.append(descriptor)
        publication_time = _timestamp(
            generated_at
            or datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "generatedAt",
        )
        new_index = {
            "format": INDEX_FORMAT,
            "schemaVersion": SCHEMA_VERSION,
            "generatedAt": publication_time,
            "catalog": catalog_name,
            "runs": references,
        }
        _write_atomic(root / "index.json", _canonical(new_index))

    return {
        "changed": run_changed or catalog_changed or index_changed or index is None,
        "run": str(run_path),
        "catalog": str(catalog_path),
        "index": str(root / "index.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--trigger", choices=("auto", "manual"), required=True)
    parser.add_argument("--branch", choices=("develop",), required=True)
    parser.add_argument("--is-demo", action="store_true")
    parser.add_argument("--machine-id")
    parser.add_argument("--commit-order", type=int)
    parser.add_argument("--commit-message")
    parser.add_argument("--generated-at")
    parser.add_argument("--source-ref")
    parser.add_argument("--expected-sha", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = publish(
            load_json_document(arguments.raw_run),
            data_dir=arguments.data_dir,
            run_id=arguments.run_id,
            repository=arguments.repository,
            environment_id=arguments.environment_id,
            trigger=arguments.trigger,
            branch=arguments.branch,
            is_demo=arguments.is_demo,
            machine_id=arguments.machine_id,
            commit_order=arguments.commit_order,
            commit_message=arguments.commit_message,
            generated_at=arguments.generated_at,
            source_ref=arguments.source_ref,
            expected_sha=arguments.expected_sha,
        )
    except PublishError as error:
        build_parser().error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
