# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "dashboard_publish.py"
SPEC = importlib.util.spec_from_file_location("dashboard_publish", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


@pytest.fixture
def publisher_context(tmp_path):
    return SimpleNamespace(root=tmp_path, data=tmp_path / "public" / "data")


def _test(
    logical_id: str = "triton.copy_fp32_32m",
    *,
    target: str = "gfx950",
    problem: dict[str, object] | None = None,
    status: str = "completed",
) -> dict[str, object]:
    timed_out = status == "timeout"
    return {
        "testId": f"{target}:{logical_id}",
        "logicalTestId": logical_id,
        "suite": "Triton",
        "name": "32 MiB contiguous FP32 copy",
        "target": target,
        "operation": "Copy",
        "dataType": "fp32",
        "problem": {"elements": 8388608} if problem is None else problem,
        "execMode": "functional",
        "numThreads": 1,
        "durationSeconds": 0.00125 if status == "completed" else None,
        "timing": {
            "unit": "ns",
            "samples": [1200000, 1250000, 1300000],
            "minimum": 1200000,
            "median": 1250000,
            "maximum": 1300000,
        },
        "status": status,
        "exitCode": 0 if status == "completed" else None,
        "timedOut": timed_out,
        "error": None if status == "completed" else f"{status} fixture",
        "artifacts": {},
    }


def _raw(
    publisher_context, tests: list[dict[str, object]] | None = None
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "timestamp": "2026-09-01T20:00:00Z",
        "finishedAt": "2026-09-01T20:01:00+00:00",
        "status": "completed",
        "wallTimeSeconds": 60.0,
        "benchmarkSuite": "nightly",
        "targets": sorted({t["target"] for t in tests or [_test()]}),
        "measurement": {"warmups": 3, "samples": 3, "timeoutSeconds": 300},
        "configuration": {
            "id": "plugins-none-v1",
            "pluginProfile": "none",
            "plugins": [],
            "targetConfigSha256": {"gfx950": "b" * 64},
        },
        "provenance": {
            "rocjitsuCommitSha": "a" * 40,
            "rocjitsuCommitTimestamp": "2026-09-01T19:42:10-07:00",
            "dirty": False,
            "corpusCommitSha": "e" * 40,
            "corpusCommitTimestamp": "2026-09-01T00:00:00Z",
            "corpusDirty": False,
            "buildType": "Release",
            "rocmSdkPath": "/opt/rocm",
            "rocmSdkVersion": "7.2.0",
            "pythonVersion": "3.12.0",
            "torchVersion": "2.10.0",
            "tritonVersion": "3.6.0",
            "tritonCommitSha": None,
            "tensileLiteCommitSha": "c" * 12,
            "packages": {
                "rocm-sdk-devel": "7.2.0",
                "rocm-sdk-libraries": "7.2.0",
                "rocm-sdk-device-gfx950": "7.2.0",
            },
        },
        "environment": {"hostname": "benchmark-host"},
        "tests": tests or [_test()],
    }


def _publish(publisher_context, raw: dict[str, object] | None = None, **overrides):
    options = {
        "data_dir": publisher_context.data,
        "run_id": "github-123-attempt-1",
        "repository": "https://github.com/ROCm/rocm-systems",
        "environment_id": "rocm-7.2-sjc-01",
        "trigger": "auto",
        "branch": "develop",
        "machine_id": "sjc-rocjitsu-perf-01",
        "commit_message": "Improve dispatch",
        "generated_at": "2026-09-01T20:02:00Z",
    }
    options.update(overrides)
    return publisher.publish(raw or _raw(publisher_context), **options)


def test_contract_and_idempotent_publication(publisher_context):
    result = _publish(publisher_context)
    metadata = publisher.load_json_document(result["metadata"])
    assert metadata == {
        "schemaVersion": 1,
        "repository": "https://github.com/ROCm/rocm-systems",
        "isBeta": False,
    }
    index = publisher.load_json_document(result["index"])
    assert set(index) == {"generatedAt", "runFiles"}
    assert index["runFiles"] == ["runs/github-123-attempt-1.json"]
    run = publisher.load_json_document(result["run"])
    assert set(run) == {
        "id",
        "comparisonId",
        "testCatalog",
        "plugin",
        "source",
        "execution",
        "environment",
        "targets",
    }
    assert run["plugin"] == {"id": "vanilla", "name": "Vanilla"}
    assert run["source"]["committedAt"] == "2026-09-02T02:42:10Z"
    assert run["execution"]["machine"] == "sjc-rocjitsu-perf-01"
    catalog = publisher.load_json_document(result["catalog"])
    assert catalog["targets"] == {"gfx950": ["triton.copy_fp32_32m"]}
    assert catalog["tests"][0]["problem"] == {
        "operation": "Copy",
        "dataType": "fp32",
        "elements": 8388608,
    }
    assert run["targets"][0]["results"][0]["testId"] == "triton.copy_fp32_32m"
    snapshot = {p: p.read_bytes() for p in publisher_context.data.rglob("*.json")}
    assert not _publish(publisher_context)["changed"]
    assert snapshot == {
        p: p.read_bytes() for p in publisher_context.data.rglob("*.json")
    }


def test_subset_catalogs_are_snapshots_and_preserve_history(publisher_context):
    first = _publish(publisher_context)
    original = Path(first["catalog"]).read_bytes()
    tests = [_test(), _test("other", target="gfx1250")]
    second = _publish(
        publisher_context, _raw(publisher_context, tests), run_id="second"
    )
    assert first["catalog"] != second["catalog"]
    assert Path(first["catalog"]).read_bytes() == original
    catalog = publisher.load_json_document(second["catalog"])
    assert catalog["targets"] == {
        "gfx950": ["triton.copy_fp32_32m"],
        "gfx1250": ["other"],
    }
    reordered = _publish(
        publisher_context,
        _raw(publisher_context, list(reversed(tests))),
        run_id="third",
    )
    assert second["catalog"] == reordered["catalog"]


@pytest.mark.parametrize("branch", ["develop", "users/ianwood2/topic"])
def test_manual_publication_preserves_mixed_history(publisher_context, branch):
    first = _publish(publisher_context)
    original = Path(first["run"]).read_bytes()
    options = {"run_id": "manual", "trigger": "manual", "branch": branch}
    manual = _publish(publisher_context, **options)
    run = publisher.load_json_document(manual["run"])
    assert run["source"]["branch"] == branch
    assert run["execution"]["trigger"] == "manual"
    snapshot = {p: p.read_bytes() for p in publisher_context.data.rglob("*.json")}
    assert not _publish(publisher_context, **options)["changed"]
    assert snapshot == {
        p: p.read_bytes() for p in publisher_context.data.rglob("*.json")
    }
    _publish(publisher_context, run_id="next-auto")
    assert Path(first["run"]).read_bytes() == original
    assert publisher.load_json_document(manual["index"])["runFiles"] == [
        "runs/github-123-attempt-1.json",
        "runs/manual.json",
        "runs/next-auto.json",
    ]
    raw = _raw(publisher_context)
    raw["tests"][0]["durationSeconds"] = 42
    with pytest.raises(publisher.PublishError, match="immutable"):
        _publish(publisher_context, raw, **options)


@pytest.mark.parametrize(
    "trigger,branch,error",
    [
        ("auto", "topic", "requires branch develop"),
        ("manual", "", "branch must be a non-empty string"),
        ("manual", "  ", "branch must be a non-empty string"),
        ("manual", None, "branch must be a non-empty string"),
        ("manual", 42, "branch must be a non-empty string"),
        ("unknown", "develop", "invalid trigger"),
    ],
)
def test_invalid_publication_sources_rejected_before_writes(
    publisher_context, trigger, branch, error
):
    with pytest.raises(publisher.PublishError, match=error):
        _publish(publisher_context, trigger=trigger, branch=branch)
    assert not publisher_context.data.exists()


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("branch", "", "branch must be a non-empty string"),
        ("branch", "  ", "branch must be a non-empty string"),
        ("trigger", "auto", "requires branch develop"),
        ("trigger", "unknown", "invalid trigger"),
        ("commit", "invalid", "invalid published source"),
    ],
)
def test_invalid_existing_manual_source_rejected_before_writes(
    publisher_context, field, value, error
):
    result = _publish(publisher_context, trigger="manual", branch="topic")
    run = publisher.load_json_document(result["run"])
    run["execution" if field == "trigger" else "source"][field] = value
    Path(result["run"]).write_text(json.dumps(run))
    before = {p: p.read_bytes() for p in publisher_context.data.rglob("*.json")}
    with pytest.raises(publisher.PublishError, match=error):
        _publish(publisher_context, run_id="next")
    assert before == {p: p.read_bytes() for p in publisher_context.data.rglob("*.json")}


@pytest.mark.parametrize("option", ["expected_sha", "expected_corpus_sha"])
def test_manual_publication_requires_matching_shas(publisher_context, option):
    with pytest.raises(publisher.PublishError, match="expected SHA"):
        _publish(
            publisher_context, trigger="manual", branch="topic", **{option: "f" * 40}
        )
    assert not publisher_context.data.exists()


def test_first_failure_timeout_and_recovery(publisher_context):
    raw = _raw(
        publisher_context,
        [_test(status="failed"), _test(target="gfx1250", status="timeout")],
    )
    raw["status"] = "failed"
    first = _publish(publisher_context, raw)
    run = publisher.load_json_document(first["run"])
    assert [g["results"][0]["status"] for g in run["targets"]] == ["timeout", "failed"]
    recovered = _publish(
        publisher_context,
        _raw(publisher_context, [_test(), _test(target="gfx1250")]),
        run_id="recovered",
    )
    assert first["catalog"] == recovered["catalog"]


def test_definition_conflicts_fail_before_writes(publisher_context):
    _publish(publisher_context)
    before = (publisher_context.data / "index.json").read_bytes()
    for field, value in (
        ("name", "renamed"),
        ("suite", "another"),
        ("problem", {"elements": 17}),
    ):
        test = _test()
        test[field] = value
        with pytest.raises(publisher.PublishError, match="definition conflicts"):
            _publish(
                publisher_context, _raw(publisher_context, [test]), run_id="conflict"
            )
    assert (publisher_context.data / "index.json").read_bytes() == before
    assert not (publisher_context.data / "runs/conflict.json").exists()


def test_source_and_corpus_provenance(publisher_context):
    result = _publish(
        publisher_context, expected_sha="A" * 40, expected_corpus_sha="E" * 40
    )
    run = publisher.load_json_document(result["run"])
    details = {item["key"]: item["value"] for item in run["environment"]}
    assert details["corpusCommitSha"] == "e" * 40
    assert details["target.gfx950.numThreads"] == 1
    for field, value in (
        ("dirty", True),
        ("corpusDirty", True),
        ("rocjitsuCommitSha", "bad"),
        ("corpusCommitSha", "bad"),
    ):
        raw = _raw(publisher_context)
        raw["provenance"][field] = value
        with pytest.raises(publisher.PublishError):
            _publish(publisher_context, raw, run_id="bad")
    for option in ("expected_sha", "expected_corpus_sha"):
        with pytest.raises(publisher.PublishError, match="expected SHA"):
            _publish(publisher_context, **{option: "f" * 40})


def test_same_source_requires_consistent_commit_timestamp(publisher_context):
    _publish(publisher_context)
    before = (publisher_context.data / "index.json").read_bytes()
    raw = _raw(publisher_context)
    raw["provenance"]["rocjitsuCommitTimestamp"] = "2026-09-03T00:00:00Z"
    with pytest.raises(publisher.PublishError, match="commit timestamp"):
        _publish(publisher_context, raw, run_id="different-timestamp")
    assert (publisher_context.data / "index.json").read_bytes() == before


def test_published_environment_preserves_package_versions(publisher_context):
    raw = _raw(publisher_context)
    raw["provenance"]["packages"].update(
        {"additional-package": "1.2.3", "uninstalled-package": None}
    )
    result = _publish(publisher_context, raw)
    run = publisher.load_json_document(result["run"])
    packages = {
        detail["key"]: detail["value"]
        for detail in run["environment"]
        if detail["key"].startswith("package.")
    }
    assert packages == {
        "package.additional-package": "1.2.3",
        "package.rocm-sdk-devel": "7.2.0",
        "package.rocm-sdk-device-gfx950": "7.2.0",
        "package.rocm-sdk-libraries": "7.2.0",
    }


@pytest.mark.parametrize("packages", ["missing", {}, {"uninstalled-package": None}])
def test_unavailable_package_versions_are_omitted(publisher_context, packages):
    raw = _raw(publisher_context)
    if packages == "missing":
        del raw["provenance"]["packages"]
    else:
        raw["provenance"]["packages"] = packages
    result = _publish(publisher_context, raw)
    run = publisher.load_json_document(result["run"])
    assert not any(
        detail["key"].startswith("package.") for detail in run["environment"]
    )


@pytest.mark.parametrize(
    "packages",
    [None, [], {"package": []}, {"package": {}}, {"package": float("inf")}],
)
def test_malformed_package_versions_rejected_before_writes(publisher_context, packages):
    raw = _raw(publisher_context)
    raw["provenance"]["packages"] = packages
    with pytest.raises(publisher.PublishError):
        _publish(publisher_context, raw)
    assert not publisher_context.data.exists()


def test_compatible_plugins_share_comparison(publisher_context):
    _publish(publisher_context, comparison_id="experiment")
    for profile in ("logging", "race", "throughput"):
        raw = _raw(publisher_context)
        raw["configuration"].update(
            id=f"plugins-{profile}-v1", pluginProfile=profile, plugins=[profile]
        )
        result = _publish(
            publisher_context, raw, run_id=profile, comparison_id="experiment"
        )
        run = publisher.load_json_document(result["run"])
        assert run["plugin"]["id"] == profile
        assert "pluginProfile" not in {e["key"] for e in run["environment"]}
    with pytest.raises(publisher.PublishError, match="repeats a plugin"):
        _publish(
            publisher_context, run_id="duplicate-vanilla", comparison_id="experiment"
        )


def test_incompatible_comparisons_rejected(publisher_context):
    _publish(publisher_context, comparison_id="experiment")
    for option, value in (
        ("machine_id", "other"),
        ("environment_id", "other"),
        ("trigger", "manual"),
        ("commit_message", "other"),
    ):
        raw = _raw(publisher_context)
        raw["configuration"]["pluginProfile"] = "logging"
        with pytest.raises(publisher.PublishError, match="incompatible comparison"):
            _publish(
                publisher_context,
                raw,
                run_id="logging",
                comparison_id="experiment",
                **{option: value},
            )
    for change in ("source", "catalog", "environment"):
        raw = _raw(publisher_context)
        raw["configuration"]["pluginProfile"] = "logging"
        if change == "source":
            raw["provenance"]["rocjitsuCommitSha"] = "b" * 40
        elif change == "catalog":
            raw["tests"][0]["logicalTestId"] = "new"
            raw["tests"][0]["testId"] = "gfx950:new"
        else:
            raw["measurement"]["samples"] = 21
        with pytest.raises(publisher.PublishError, match="incompatible comparison"):
            _publish(
                publisher_context, raw, run_id="logging", comparison_id="experiment"
            )


@pytest.mark.parametrize("package", ["rocm-sdk-libraries", "rocm-sdk-device-gfx950"])
@pytest.mark.parametrize(
    "before_version,after_version",
    [("7.2.0", "7.2.1"), ("7.2.0", None), (None, "7.2.0")],
)
def test_package_version_changes_reject_comparison_before_writes(
    publisher_context, package, before_version, after_version
):
    baseline = _raw(publisher_context)
    if before_version is None:
        del baseline["provenance"]["packages"][package]
    else:
        baseline["provenance"]["packages"][package] = before_version
    _publish(publisher_context, baseline, comparison_id="experiment")
    before = {p: p.read_bytes() for p in publisher_context.data.rglob("*.json")}
    raw = _raw(publisher_context)
    raw["configuration"]["pluginProfile"] = "logging"
    if after_version is None:
        del raw["provenance"]["packages"][package]
    else:
        raw["provenance"]["packages"][package] = after_version
    assert (
        raw["provenance"]["rocmSdkVersion"] == baseline["provenance"]["rocmSdkVersion"]
    )
    with pytest.raises(publisher.PublishError, match="incompatible comparison"):
        _publish(publisher_context, raw, run_id="logging", comparison_id="experiment")
    assert before == {p: p.read_bytes() for p in publisher_context.data.rglob("*.json")}


def test_malformed_results_and_problems_rejected(publisher_context):
    for field, value in (
        ("durationSeconds", 0),
        ("durationSeconds", float("inf")),
        ("durationSeconds", True),
        ("exitCode", {}),
        ("error", {}),
        ("problem", None),
        ("problem", {"nested": {"x": 1}}),
        ("problem", {"list": [1]}),
        ("timedOut", True),
        ("status", "unknown"),
    ):
        test = _test()
        test[field] = value
        with pytest.raises(publisher.PublishError):
            _publish(publisher_context, _raw(publisher_context, [test]))
    assert not publisher_context.data.exists()


def test_duplicate_results_and_target_mismatches_rejected(publisher_context):
    raw = _raw(publisher_context, [_test(), _test()])
    with pytest.raises(publisher.PublishError, match="duplicate result"):
        _publish(publisher_context, raw)
    raw = _raw(publisher_context)
    raw["targets"].append("gfx1250")
    with pytest.raises(publisher.PublishError, match="targets do not match"):
        _publish(publisher_context, raw)


def test_running_run_and_unsafe_ids_rejected(publisher_context):
    raw = _raw(publisher_context)
    raw["status"] = "running"
    with pytest.raises(publisher.PublishError, match="finalized"):
        _publish(publisher_context, raw)
    for name in ("../escape", "", "/absolute"):
        with pytest.raises(publisher.PublishError):
            _publish(publisher_context, run_id=name)


def test_conflicting_run_id_rejected(publisher_context):
    _publish(publisher_context)
    raw = _raw(publisher_context)
    raw["tests"][0]["durationSeconds"] = 42
    with pytest.raises(publisher.PublishError, match="immutable"):
        _publish(publisher_context, raw)


def test_legacy_dataset_rejected_without_changes(publisher_context):
    publisher_context.data.mkdir(parents=True)
    old = '{"schemaVersion": 1, "runs": [], "catalog": "old.json"}'
    (publisher_context.data / "index.json").write_text(old)
    with pytest.raises(publisher.PublishError, match="fresh data directory"):
        _publish(publisher_context)
    assert (publisher_context.data / "index.json").read_text() == old


def test_missing_or_corrupt_indexed_resources_rejected(publisher_context):
    result = _publish(publisher_context)
    run_path = Path(result["run"])
    original = run_path.read_bytes()
    run_path.unlink()
    with pytest.raises(publisher.PublishError):
        _publish(publisher_context, run_id="next")
    run_path.write_bytes(original)
    run = json.loads(original)
    run["targets"][0]["results"] = []
    run_path.write_text(json.dumps(run))
    with pytest.raises(publisher.PublishError, match="catalog"):
        _publish(publisher_context, run_id="next")
    run_path.write_bytes(original)
    catalog = publisher.load_json_document(result["catalog"])
    catalog["tests"][0]["name"] = "changed"
    Path(result["catalog"]).write_text(json.dumps(catalog))
    with pytest.raises(publisher.PublishError, match="hash"):
        _publish(publisher_context, run_id="next")
    assert not (publisher_context.data / "runs/next.json").exists()


def test_metadata_conflict(publisher_context):
    _publish(publisher_context)
    for options in (
        {"is_beta": True},
        {"repository": "https://github.com/example/fork"},
    ):
        with pytest.raises(publisher.PublishError, match="metadata conflicts"):
            _publish(publisher_context, **options)


def test_index_written_after_catalog_and_run(publisher_context):
    writes = []
    write = publisher._write_atomic

    def record(path, content):
        writes.append(path)
        write(path, content)

    with mock.patch.object(publisher, "_write_atomic", side_effect=record):
        result = _publish(publisher_context)
    assert writes.index(Path(result["catalog"])) < writes.index(Path(result["run"]))
    assert writes[-1] == publisher_context.data / "index.json"


def test_strict_json(publisher_context):
    for value in ("{} {}", '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
        path = publisher_context.root / "bad.json"
        path.write_text(value)
        with pytest.raises(publisher.PublishError):
            publisher.load_json_document(path)


@pytest.mark.parametrize("branch", ["develop", "users/ianwood2/topic"])
def test_cli(publisher_context, branch):
    raw_path = publisher_context.root / "raw.json"
    raw_path.write_text(json.dumps(_raw(publisher_context)))
    with contextlib.redirect_stdout(io.StringIO()):
        assert (
            publisher.main(
                [
                    "--raw-run",
                    str(raw_path),
                    "--data-dir",
                    str(publisher_context.data),
                    "--run-id",
                    "cli",
                    "--repository",
                    "https://github.com/ROCm/rocm-systems",
                    "--environment-id",
                    "local",
                    "--trigger",
                    "manual",
                    "--branch",
                    branch,
                    "--expected-sha",
                    "a" * 40,
                    "--expected-corpus-sha",
                    "e" * 40,
                    "--is-beta",
                    "--comparison-id",
                    "experiment",
                ]
            )
            == 0
        )
    run = publisher.load_json_document(publisher_context.data / "runs/cli.json")
    assert run["source"]["branch"] == branch
    assert run["execution"]["trigger"] == "manual"
    assert run["execution"]["machine"] == "benchmark-host"
    assert run["comparisonId"] == "experiment"
    assert publisher.load_json_document(publisher_context.data / "metadata.json")[
        "isBeta"
    ]


@pytest.mark.parametrize("trigger,branch", [("auto", "topic"), ("manual", " ")])
def test_cli_rejects_invalid_branch(publisher_context, trigger, branch):
    raw_path = publisher_context.root / "raw.json"
    raw_path.write_text(json.dumps(_raw(publisher_context)))
    with pytest.raises(SystemExit) as error:
        publisher.main(
            [
                "--raw-run",
                str(raw_path),
                "--data-dir",
                str(publisher_context.data),
                "--run-id",
                "cli",
                "--repository",
                "https://github.com/ROCm/rocm-systems",
                "--environment-id",
                "local",
                "--trigger",
                trigger,
                "--branch",
                branch,
                "--expected-sha",
                "a" * 40,
                "--expected-corpus-sha",
                "e" * 40,
            ]
        )
    assert error.value.code == 2
    assert not publisher_context.data.exists()


@pytest.mark.parametrize("status", ["completed", "failed", "timeout"])
def test_published_results_omit_raw_exit_code(publisher_context, status):
    result = _publish(
        publisher_context, _raw(publisher_context, [_test(status=status)])
    )
    run = publisher.load_json_document(result["run"])
    assert set(run["targets"][0]["results"][0]) == {
        "testId",
        "durationSeconds",
        "status",
        "error",
    }


def test_independent_runs_require_same_machine(publisher_context):
    _publish(publisher_context)
    before = (publisher_context.data / "index.json").read_bytes()
    with pytest.raises(publisher.PublishError, match="same machine"):
        _publish(publisher_context, run_id="other-machine", machine_id="other")
    assert (publisher_context.data / "index.json").read_bytes() == before


@pytest.mark.parametrize("unrelated_baseline", [False, True])
def test_plugin_requires_its_own_vanilla_baseline(
    publisher_context, unrelated_baseline
):
    if unrelated_baseline:
        _publish(publisher_context)
    raw = _raw(publisher_context)
    raw["configuration"]["pluginProfile"] = "logging"
    with pytest.raises(publisher.PublishError, match="Vanilla baseline"):
        _publish(publisher_context, raw, run_id="logging")
    assert not (publisher_context.data / "runs/logging.json").exists()
