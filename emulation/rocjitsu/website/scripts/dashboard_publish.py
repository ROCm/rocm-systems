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
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CATALOG_NAME = re.compile(r"^test-catalogs/catalog-([0-9a-f]{12})\.json$")
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


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
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


PLUGIN_NAMES = {
    "none": ("vanilla", "Vanilla"),
    "logging": ("logging", "Logging"),
    "race": ("race", "Race detection"),
    "throughput": ("throughput", "Throughput"),
}


def _identifier(value: Any, name: str) -> str:
    value = _string(value, name)
    if not RUN_ID.fullmatch(value):
        raise PublishError(f"unsafe {name}: {value!r}")
    return value


def _scalar(value: Any, name: str) -> Any:
    if not isinstance(value, (str, int, float, bool)) or (
        isinstance(value, float) and not math.isfinite(value)
    ):
        raise PublishError(f"{name} must be a finite scalar")
    return value


def _definition(test: Mapping[str, Any]) -> dict[str, Any]:
    problem = dict(_mapping(test.get("problem"), "test problem"))
    dtype = _string(test.get("dataType"), "dataType")
    if dtype != dtype.lower():
        raise PublishError("dataType must be lowercase")
    if "dtype" in problem and problem.pop("dtype") != dtype:
        raise PublishError("problem dtype conflicts with dataType")
    for key, value in {
        "operation": _string(test.get("operation"), "operation"),
        "dataType": dtype,
    }.items():
        if key in problem and problem[key] != value:
            raise PublishError(f"problem conflicts with {key}")
        problem[key] = value
    for key, value in problem.items():
        _scalar(value, _string(key, "problem key"))
    return {
        "id": _identifier(test.get("logicalTestId"), "test ID"),
        "suite": _string(test.get("suite"), "suite"),
        "name": _string(test.get("name"), "name"),
        "problem": problem,
    }


def _catalog(
    tests: list[dict[str, Any]], targets: dict[str, list[str]]
) -> dict[str, Any]:
    content = {
        "tests": sorted(tests, key=lambda test: test["id"]),
        "targets": {target: sorted(ids) for target, ids in sorted(targets.items())},
    }
    digest = hashlib.sha256(_canonical(content)).hexdigest()[:12]
    return {"id": f"catalog-{digest}", **content}


def _result(test: Mapping[str, Any], test_id: str) -> dict[str, Any]:
    status = test.get("status")
    if status not in ("completed", "failed", "timeout"):
        raise PublishError(f"test {test_id} has invalid status")
    duration = test.get("durationSeconds")
    if status == "completed":
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise PublishError(f"completed test {test_id} needs a positive duration")
        if test.get("error") is not None:
            raise PublishError(f"completed test {test_id} cannot have an error")
    elif duration is not None:
        raise PublishError(f"incomplete test {test_id} must have a null duration")
    exit_code = test.get("exitCode")
    if exit_code is not None and (
        isinstance(exit_code, bool) or not isinstance(exit_code, int)
    ):
        raise PublishError(f"test {test_id} has an invalid exit code")
    error = test.get("error")
    if error is not None:
        _string(error, "test error")
    return {
        "testId": test_id,
        "durationSeconds": duration,
        "status": status,
        "error": error,
    }


def normalize_run(
    raw_run: Mapping[str, Any],
    *,
    run_id: str,
    trigger: str,
    branch: str,
    environment_id: str,
    comparison_id: str | None = None,
    machine_id: str | None = None,
    commit_message: str | None = None,
    source_ref: str | None = None,
    expected_sha: str | None = None,
    expected_corpus_sha: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the dashboard run and its complete immutable catalog."""
    raw_run = _mapping(raw_run, "raw run")
    if raw_run.get("schemaVersion") != SCHEMA_VERSION or raw_run.get("status") not in (
        "completed",
        "failed",
    ):
        raise PublishError("raw run must be finalized schema version 1")
    if branch != "develop" or trigger not in ("auto", "manual"):
        raise PublishError(
            "publication requires branch develop and trigger auto or manual"
        )
    provenance = _mapping(raw_run.get("provenance"), "provenance")
    for prefix, sha_key, dirty_key, expected in (
        ("Rocjitsu", "rocjitsuCommitSha", "dirty", expected_sha),
        ("corpus", "corpusCommitSha", "corpusDirty", expected_corpus_sha),
    ):
        if provenance.get(dirty_key) is not False:
            raise PublishError(f"raw run must come from a clean {prefix} checkout")
        sha = _string(provenance.get(sha_key), f"{prefix} SHA")
        if not SHA.fullmatch(sha):
            raise PublishError(f"invalid {prefix} SHA")
        if expected is not None and (
            not SHA.fullmatch(expected) or sha.lower() != expected.lower()
        ):
            raise PublishError(f"{prefix} SHA does not match expected SHA")
    configuration = _mapping(raw_run.get("configuration"), "configuration")
    profile = configuration.get("pluginProfile")
    if profile not in PLUGIN_NAMES:
        raise PublishError(f"unsupported plugin profile {profile!r}")
    plugin_id, plugin_name = PLUGIN_NAMES[profile]
    source = {
        "branch": branch,
        "commit": provenance["rocjitsuCommitSha"].lower(),
        "committedAt": _timestamp(
            provenance.get("rocjitsuCommitTimestamp"), "commit timestamp"
        ),
    }
    if commit_message is not None:
        source["message"] = _string(commit_message, "commit message")
    host = _mapping(raw_run.get("environment"), "environment")
    machine = _string(
        machine_id if machine_id is not None else host.get("hostname"), "machine ID"
    )
    details = {"environmentId": _string(environment_id, "environment ID")}
    for key in (
        "corpusCommitSha",
        "corpusCommitTimestamp",
        "buildType",
        "rocmSdkVersion",
        "pythonVersion",
        "torchVersion",
        "tritonVersion",
    ):
        if provenance.get(key) is not None:
            details[key] = provenance[key]
    for name, version in _mapping(
        provenance.get("packages", {}), "provenance packages"
    ).items():
        if version is not None:
            details[f"package.{name}"] = version
    for key in ("cpu", "kernel", "platform"):
        if host.get(key) is not None:
            details[key] = host[key]
    if source_ref is not None:
        details["sourceRef"] = _string(source_ref, "source ref")
    for key, value in _mapping(raw_run.get("measurement"), "measurement").items():
        details[f"measurement.{key}"] = value
    for target, value in _mapping(
        configuration.get("targetConfigSha256"), "target config hashes"
    ).items():
        details[f"target.{target}.configSha256"] = value
    raw_tests = raw_run.get("tests")
    if not isinstance(raw_tests, list) or not raw_tests:
        raise PublishError("raw run must contain tests")
    definitions = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for value in raw_tests:
        test = _mapping(value, "test")
        definition = _definition(test)
        ident = definition["id"]
        if ident in definitions and definitions[ident] != definition:
            raise PublishError(f"conflicting definition for {ident}")
        definitions[ident] = definition
        target = _string(test.get("target"), "target")
        if test.get("testId") != f"{target}:{ident}":
            raise PublishError("invalid raw testId")
        if test.get("timedOut") is not (test.get("status") == "timeout"):
            raise PublishError("inconsistent timeout metadata")
        results = groups.setdefault(target, [])
        if any(result["testId"] == ident for result in results):
            raise PublishError(f"duplicate result {target}:{ident}")
        results.append(_result(test, ident))
        for key, item in (
            ("execMode", _string(test.get("execMode"), "execMode")),
            ("numThreads", _positive_integer(test.get("numThreads"), "numThreads")),
        ):
            detail_key = f"target.{target}.{key}"
            if detail_key in details and details[detail_key] != item:
                raise PublishError(f"conflicting {detail_key}")
            details[detail_key] = item
    targets = raw_run.get("targets")
    if (
        not isinstance(targets, list)
        or any(not isinstance(t, str) for t in targets)
        or len(set(targets)) != len(targets)
        or set(targets) != set(groups)
    ):
        raise PublishError("raw run targets do not match results")
    catalog = _catalog(
        list(definitions.values()),
        {
            target: [test["testId"] for test in tests]
            for target, tests in groups.items()
        },
    )
    run = {
        "id": _identifier(run_id, "run ID"),
        "comparisonId": _identifier(
            comparison_id if comparison_id is not None else run_id, "comparison ID"
        ),
        "testCatalog": f"test-catalogs/{catalog['id']}.json",
        "plugin": {"id": plugin_id, "name": plugin_name},
        "source": source,
        "execution": {
            "completedAt": _timestamp(raw_run.get("finishedAt"), "finishedAt"),
            "trigger": trigger,
            "machine": machine,
        },
        "environment": [
            {"key": key, "label": key, "value": _scalar(value, key)}
            for key, value in sorted(details.items())
        ],
        "targets": [
            {"id": target, "results": sorted(tests, key=lambda t: t["testId"])}
            for target, tests in sorted(groups.items())
        ],
    }
    return run, catalog


def _comparison_identity(run: Mapping[str, Any]) -> Any:
    return (
        run["testCatalog"],
        run["source"],
        run["execution"]["trigger"],
        run["execution"]["machine"],
        sorted(group["id"] for group in run["targets"]),
        sorted((item["key"], item["value"]) for item in run["environment"]),
    )


def _validate_published_run(run: Mapping[str, Any], catalog: Mapping[str, Any]) -> None:
    _identifier(run.get("comparisonId"), "comparison ID")
    source = _mapping(run.get("source"), "source")
    if source.get("branch") != "develop" or not SHA.fullmatch(
        _string(source.get("commit"), "commit")
    ):
        raise PublishError("invalid published source")
    _timestamp(source.get("committedAt"), "committedAt")
    if "message" in source:
        _string(source["message"], "source message")
    execution = _mapping(run.get("execution"), "execution")
    _timestamp(execution.get("completedAt"), "completedAt")
    _string(execution.get("machine"), "machine")
    if execution.get("trigger") not in ("auto", "manual"):
        raise PublishError("invalid trigger")
    plugin = _mapping(run.get("plugin"), "plugin")
    _identifier(plugin.get("id"), "plugin ID")
    _string(plugin.get("name"), "plugin name")
    environment = run.get("environment")
    if not isinstance(environment, list):
        raise PublishError("environment must be an array")
    keys = set()
    for item in environment:
        item = _mapping(item, "environment detail")
        key = _string(item.get("key"), "environment key")
        if key in keys:
            raise PublishError("duplicate environment key")
        keys.add(key)
        _string(item.get("label"), "environment label")
        _scalar(item.get("value"), key)
    groups = run.get("targets")
    if not isinstance(groups, list):
        raise PublishError("targets must be an array")
    seen = set()
    for group in groups:
        group = _mapping(group, "target")
        target = _string(group.get("id"), "target ID")
        if target in seen or target not in catalog["targets"]:
            raise PublishError("duplicate or unknown target")
        seen.add(target)
        results = group.get("results")
        if not isinstance(results, list):
            raise PublishError("results must be an array")
        ids = []
        for result in results:
            result = _mapping(result, "result")
            ident = _identifier(result.get("testId"), "test ID")
            ids.append(ident)
            _result(result, ident)
        if sorted(ids) != sorted(catalog["targets"][target]):
            raise PublishError("results do not match catalog")
    if seen != set(catalog["targets"]):
        raise PublishError("targets do not match catalog")


def _load_dataset(root: Path) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    if not (root / "index.json").exists():
        return [], {}, []
    index = _mapping(load_json_document(root / "index.json"), "index")
    if set(index) != {"generatedAt", "runFiles"}:
        raise PublishError(
            "unsupported dashboard index; use a fresh data directory for the new contract"
        )
    _timestamp(index["generatedAt"], "generatedAt")
    paths = index["runFiles"]
    if (
        not isinstance(paths, list)
        or any(not isinstance(p, str) for p in paths)
        or len(set(paths)) != len(paths)
    ):
        raise PublishError("runFiles must contain unique paths")
    definitions, runs, catalogs = {}, [], {}
    for path in paths:
        if not re.fullmatch(r"runs/[A-Za-z0-9][A-Za-z0-9._-]*\.json", path):
            raise PublishError(f"unsafe run path {path!r}")
        run = _mapping(load_json_document(root / path), "published run")
        if path != f"runs/{run.get('id')}.json":
            raise PublishError("run ID does not match filename")
        catalog_path = run.get("testCatalog")
        if not isinstance(catalog_path, str) or not CATALOG_NAME.fullmatch(
            catalog_path
        ):
            raise PublishError("unsafe test catalog path")
        if catalog_path not in catalogs:
            catalog = _mapping(load_json_document(root / catalog_path), "catalog")
            if (
                catalog != _catalog(catalog["tests"], catalog["targets"])
                or catalog_path != f"test-catalogs/{catalog['id']}.json"
            ):
                raise PublishError("catalog content hash mismatch")
            catalogs[catalog_path] = catalog
            for definition in catalog["tests"]:
                ident = definition["id"]
                if ident in definitions and definitions[ident] != definition:
                    raise PublishError(f"conflicting definition for {ident}")
                definitions[ident] = definition
        _validate_published_run(run, catalogs[catalog_path])
        runs.append(dict(run))
    return paths, definitions, runs


def publish(
    raw_run: Mapping[str, Any],
    *,
    data_dir: str | Path,
    run_id: str,
    repository: str,
    environment_id: str,
    trigger: str,
    branch: str,
    is_beta: bool = False,
    comparison_id: str | None = None,
    machine_id: str | None = None,
    commit_message: str | None = None,
    generated_at: str | None = None,
    source_ref: str | None = None,
    expected_sha: str | None = None,
    expected_corpus_sha: str | None = None,
) -> dict[str, Any]:
    root = Path(data_dir).expanduser().resolve()
    if not isinstance(is_beta, bool):
        raise PublishError("is_beta must be boolean")
    metadata = {
        "schemaVersion": SCHEMA_VERSION,
        "repository": _string(repository, "repository"),
        "isBeta": is_beta,
    }
    if (root / "metadata.json").exists() and load_json_document(
        root / "metadata.json"
    ) != metadata:
        raise PublishError("metadata conflicts with the existing dataset")
    try:
        paths, definitions, old_runs = _load_dataset(root)
    except (KeyError, TypeError) as error:
        raise PublishError(f"malformed existing dataset: {error}") from error
    run, catalog = normalize_run(
        raw_run,
        run_id=run_id,
        trigger=trigger,
        branch=branch,
        environment_id=environment_id,
        comparison_id=comparison_id,
        machine_id=machine_id,
        commit_message=commit_message,
        source_ref=source_ref,
        expected_sha=expected_sha,
        expected_corpus_sha=expected_corpus_sha,
    )
    for definition in catalog["tests"]:
        ident = definition["id"]
        if ident in definitions and definitions[ident] != definition:
            raise PublishError(f"benchmark definition conflicts for {ident}")
    for old in old_runs:
        if (
            old["source"]["commit"] == run["source"]["commit"]
            and old["source"]["committedAt"] != run["source"]["committedAt"]
        ):
            raise PublishError(
                "conflicting commit timestamp for the same source revision"
            )
        if old["id"] == run["id"]:
            if old != run:
                raise PublishError("conflicting immutable dashboard run")
        elif old["comparisonId"] == run["comparisonId"]:
            if _comparison_identity(old) != _comparison_identity(run):
                raise PublishError("incompatible comparison executions")
            if old["plugin"]["id"] == run["plugin"]["id"]:
                raise PublishError("comparison repeats a plugin")
    if any(
        old["execution"]["machine"] != run["execution"]["machine"] for old in old_runs
    ):
        raise PublishError("all dashboard runs must use the same machine")
    if run["plugin"]["id"] != "vanilla" and not any(
        old["comparisonId"] == run["comparisonId"] and old["plugin"]["id"] == "vanilla"
        for old in old_runs
    ):
        raise PublishError(
            "publish the comparison's Vanilla baseline before plugin runs"
        )
    run_path = root / "runs" / f"{run_id}.json"
    catalog_path = root / run["testCatalog"]
    for path, value in ((run_path, run), (catalog_path, catalog)):
        if path.exists() and load_json_document(path) != value:
            raise PublishError(f"conflicting immutable resource: {path}")
    publication_time = _timestamp(
        generated_at or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generatedAt",
    )
    changed = _write_immutable(root / "metadata.json", metadata, "metadata")
    changed = _write_immutable(catalog_path, catalog, "catalog") or changed
    changed = _write_immutable(run_path, run, "run") or changed
    relative = f"runs/{run_id}.json"
    if relative not in paths:
        _write_atomic(
            root / "index.json",
            _canonical(
                {"generatedAt": publication_time, "runFiles": [*paths, relative]}
            ),
        )
        changed = True
    return {
        "changed": changed,
        "run": str(run_path),
        "catalog": str(catalog_path),
        "index": str(root / "index.json"),
        "metadata": str(root / "metadata.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    for flag in (
        "run-id",
        "repository",
        "environment-id",
        "expected-sha",
        "expected-corpus-sha",
    ):
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--trigger", choices=("auto", "manual"), required=True)
    parser.add_argument("--branch", choices=("develop",), required=True)
    parser.add_argument("--is-beta", action="store_true")
    for flag in (
        "machine-id",
        "comparison-id",
        "commit-message",
        "generated-at",
        "source-ref",
    ):
        parser.add_argument(f"--{flag}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = vars(parser.parse_args(argv))
    try:
        raw = load_json_document(arguments.pop("raw_run"))
        result = publish(raw, **arguments)
    except (PublishError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
