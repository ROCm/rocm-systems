"""ConSan diagnostic, coverage, timing, and retained-code-object parsing."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess

from consan_coverage_gate import CoverageParseError, parse_coverage_evidence
from consan_validation_catalog import (
    LLVM_READELF_ENV,
    ValidationError,
)
from consan_validation_support import SITE_KINDS, sha256_file

_COVERAGE_DUMP_NAME = re.compile(
    r"rj-dbi-(?P<dump_id>[0-9]{6,})-reader-(?P<reader>[0-9]+)-"
    r"(?P<kind>original|patched)\.hsaco"
)


def _retained_relative_path(row_dir: Path, path: Path) -> str:
    row = row_dir.resolve()
    workload_root = row.parents[1]
    resolved = path.resolve()
    if not resolved.is_relative_to(workload_root):
        raise ValidationError(
            f"retained artifact path escapes workload artifacts: {resolved}"
        )
    return os.path.relpath(resolved, row)


class _DiagnosticFieldsError(ValueError):
    pass


def _log_fields(payload: str, context: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in payload.split():
        if "=" not in token:
            raise _DiagnosticFieldsError(
                f"malformed {context}: malformed field {token!r}"
            )
        key, value = token.split("=", 1)
        if re.fullmatch(r"[a-z_]+", key) is None or not value:
            raise _DiagnosticFieldsError(
                f"malformed {context}: malformed field {token!r}"
            )
        if key in fields:
            raise _DiagnosticFieldsError(
                f"malformed {context}: duplicate field {key!r}"
            )
        fields[key] = value
    return fields


def _parse_log_fields(
    line: str, marker: str, context: str, reasons: list[str]
) -> dict[str, str] | None:
    try:
        return _log_fields(line.split(marker, 1)[1], context)
    except (IndexError, _DiagnosticFieldsError) as error:
        reasons.append(str(error) or f"malformed {context}")
        return None


def _unsigned(fields: dict[str, str], name: str) -> int | None:
    try:
        value = int(fields[name], 0)
    except (KeyError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _boolean(fields: dict[str, str], name: str) -> bool | None:
    value = fields.get(name)
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _coverage_summary(
    log_text: str,
    profile: str | None = None,
) -> dict:
    try:
        evidence = parse_coverage_evidence(log_text)
    except CoverageParseError as error:
        rejection_prefix = "[rocjitsu-dbi-hooks] ConSan load rejection "
        rejection_lines = [
            line[len(rejection_prefix) :]
            for line in log_text.splitlines()
            if line.startswith(rejection_prefix)
        ]
        if rejection_lines:
            fields = dict(re.findall(r"([a-z_]+)=([^ ]+)", rejection_lines[-1]))
            return {
                "accepted": False,
                "error": "ConSan rejected a code object before execution",
                "load_rejection": fields,
            }
        return {"accepted": False, "error": str(error)}
    verdict = evidence.verdict
    reasons = []
    if not verdict.applicable:
        reasons.append("no applicable code object")
    if not verdict.analysis_complete:
        reasons.append("analysis incomplete")
    elif not verdict.static_complete:
        reasons.append("static coverage incomplete")
    if not verdict.dynamic_complete:
        reasons.append("dynamic coverage incomplete")
    if verdict.counts["dynamic_incomplete"] != 0:
        reasons.append(f"dynamic_incomplete={verdict.counts['dynamic_incomplete']}")
    for kind in SITE_KINDS:
        patched, supported = verdict.patched_supported[kind]
        if patched != supported:
            reasons.append(f"{kind}={patched}/{supported}")
    if any(record.expert_limit for record in evidence.coverage):
        reasons.append("expert patch limit enabled")
    summary = {
        "accepted": not reasons,
        "reasons": reasons,
        "analysis_complete": verdict.analysis_complete,
        "static_complete": verdict.static_complete,
        "dynamic_complete": verdict.dynamic_complete,
        "patched_supported": {
            kind: list(verdict.patched_supported[kind]) for kind in SITE_KINDS
        },
        "dynamic_incomplete": verdict.counts["dynamic_incomplete"],
    }
    return summary


def _benchmark_samples(path: Path) -> list[float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("benchmarks", [])
    iterations = [
        row
        for row in rows
        if row.get("run_type") == "iteration"
        and str(row.get("name", "")).startswith("BM_main/")
    ]
    selected = iterations
    if not selected:
        selected = [
            row
            for row in rows
            if row.get("aggregate_name") == "median"
            and str(row.get("name", "")).startswith("BM_main/")
        ]
    if not selected:
        raise ValidationError(f"expected Qwen benchmark timing rows in {path}")
    benchmark_names = {str(row.get("name", "")) for row in selected}
    if len(benchmark_names) != 1:
        raise ValidationError(
            f"expected one Qwen benchmark identity in {path}, found "
            f"{sorted(benchmark_names)}"
        )
    scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
    try:
        return [float(row["real_time"]) * scale[row["time_unit"]] for row in selected]
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationError(f"malformed Qwen benchmark timing in {path}") from error


def _benchmark_median(path: Path) -> float:
    return statistics.median(_benchmark_samples(path))


def _json_timing_samples(log_text: str, workload_kind: str) -> dict[str, list[float]]:
    documents = []
    for line in log_text.splitlines():
        if line.startswith("{"):
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if len(documents) != 1:
        raise ValidationError(f"expected one {workload_kind} JSON result")
    document = documents[0]
    timings = {}
    for key, value in document.items():
        if not isinstance(value, dict):
            continue
        if "median_ms" in value:
            samples = value.get("samples_ms")
            if samples is None:
                samples = [value["median_ms"]]
            if (
                not isinstance(samples, list)
                or not samples
                or any(
                    not isinstance(sample, (int, float))
                    or not math.isfinite(float(sample))
                    or sample <= 0
                    for sample in samples
                )
            ):
                raise ValidationError(
                    f"invalid {workload_kind} timing samples for {key}"
                )
            timings[key] = [float(sample) for sample in samples]
        device_samples = value.get("device_samples_ms")
        if device_samples is None and "device_median_ms" in value:
            device_samples = [value["device_median_ms"]]
        if device_samples is not None:
            if (
                not isinstance(device_samples, list)
                or not device_samples
                or any(
                    not isinstance(sample, (int, float))
                    or not math.isfinite(float(sample))
                    or sample <= 0
                    for sample in device_samples
                )
            ):
                raise ValidationError(
                    f"invalid {workload_kind} device timing samples for {key}"
                )
            timings[f"{key}:device"] = [float(sample) for sample in device_samples]
    return timings


def _json_measurements(log_text: str, workload_kind: str) -> dict[str, dict]:
    documents = []
    for line in log_text.splitlines():
        if line.startswith("{"):
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if len(documents) != 1:
        raise ValidationError(f"expected one {workload_kind} JSON result")
    measurements = {
        key: value
        for key, value in documents[0].items()
        if isinstance(key, str) and isinstance(value, dict)
    }
    if not measurements:
        raise ValidationError(f"expected {workload_kind} measurement rows")
    return measurements


def _json_medians(log_text: str, workload_kind: str) -> dict[str, float]:
    return {
        key: statistics.median(values)
        for key, values in _json_timing_samples(log_text, workload_kind).items()
    }


def _sharktank_medians(log_text: str) -> dict[str, float]:
    return _json_medians(log_text, "Sharktank")


def _gtest_timing_samples(log_texts: list[str]) -> dict[str, list[float]]:
    pattern = re.compile(r"\[==========\].*\(([0-9]+) ms total\)")
    values = []
    for log_text in log_texts:
        matches = pattern.findall(log_text)
        if not matches:
            raise ValidationError("missing GTest total latency")
        values.append(float(matches[-1]))
    return {"process": values}


def _gtest_median(log_texts: list[str]) -> dict[str, float]:
    return {
        mode: statistics.median(values)
        for mode, values in _gtest_timing_samples(log_texts).items()
    }


def _gtest_device_measurement(
    log_text: str, benchmark: str, expected_iterations: int
) -> tuple[float, dict[str, object]]:
    number = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
    matches = re.findall(
        rf"^hip_moi_gpu_timing benchmark={re.escape(benchmark)} timer=hip-event "
        rf"aggregate_ms=({number}) iterations=([0-9]+) "
        rf"per_iteration_ms=({number})$",
        log_text,
        re.MULTILINE,
    )
    if len(matches) != 1:
        raise ValidationError(
            f"expected one {benchmark} GPU timing row, found {len(matches)}"
        )
    aggregate_ms = float(matches[0][0])
    iterations = int(matches[0][1])
    per_iteration_ms = float(matches[0][2])
    if iterations != expected_iterations:
        raise ValidationError(
            f"{benchmark} GPU timing iteration mismatch: "
            f"{iterations} != {expected_iterations}"
        )
    if (
        not math.isfinite(aggregate_ms)
        or not math.isfinite(per_iteration_ms)
        or aggregate_ms <= 0.0
        or per_iteration_ms <= 0.0
        or not math.isclose(aggregate_ms / iterations, per_iteration_ms, rel_tol=2.0e-6)
    ):
        raise ValidationError(f"{benchmark} GPU timing row is inconsistent")
    return per_iteration_ms, {
        "benchmark_iterations": iterations,
        "timed_aggregate_ms": aggregate_ms,
        "timing_source": "hip-event",
    }


def _discard_first_sample_per_process(
    per_run: list[dict[str, list[float]]],
) -> list[dict[str, list[float]]]:
    discarded = []
    for item in per_run:
        if any(len(values) < 2 for values in item.values()):
            raise ValidationError(
                "cannot discard one warmup sample from a process with fewer than "
                "two timing samples"
            )
        discarded.append({key: values[1:] for key, values in item.items()})
    return discarded


def _nonnegative_float(fields: dict[str, str], name: str) -> float | None:
    try:
        value = float(fields[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


def _empirical_structural_metrics(log_text: str) -> dict[str, object]:
    readers: dict[int, dict[str, object]] = {}
    process_memory: dict[str, int] = {}
    reasons = []

    def parse(line: str, marker: str, context: str) -> dict[str, str] | None:
        return _parse_log_fields(line, marker, context, reasons)

    for line in log_text.splitlines():
        fields = None
        if "ConSan waitcheck timing reader=" in line:
            fields = parse(line, "ConSan waitcheck timing ", "waitcheck timing")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                elapsed = _nonnegative_float(fields, "elapsed_ms")
                if reader is None or elapsed is None:
                    reasons.append("malformed waitcheck timing")
                else:
                    readers.setdefault(reader, {})["waitcheck_ms"] = elapsed
        elif "ConSan inventory end reader=" in line:
            fields = parse(line, "ConSan inventory end ", "ConSan inventory timing")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                elapsed = _nonnegative_float(fields, "elapsed_ms")
                if reader is None or elapsed is None:
                    reasons.append("malformed ConSan inventory timing")
                else:
                    readers.setdefault(reader, {})["inventory_ms"] = elapsed
        elif "ConSan patch begin reader=" in line:
            fields = parse(line, "ConSan patch begin ", "patch begin")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                byte_count = _unsigned(fields, "bytes")
                if reader is None or byte_count is None:
                    reasons.append("malformed patch begin")
                else:
                    readers.setdefault(reader, {})["original_bytes"] = byte_count
        elif "ConSan patch end reader=" in line:
            fields = parse(line, "ConSan patch end ", "patch end")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                elapsed = _nonnegative_float(fields, "patch_ms")
                patches = _unsigned(fields, "patches")
                if reader is None or elapsed is None or patches is None:
                    reasons.append("patch end lacks empirical timing fields")
                else:
                    record = readers.setdefault(reader, {})
                    record["patch_ms"] = elapsed
                    record["patches"] = patches
                    record["modified"] = _boolean(fields, "modified")
                    record["outcome"] = fields.get("outcome")
        elif "ConSan replacement reader=" in line:
            fields = parse(line, "ConSan replacement ", "replacement image")
            if fields is not None:
                reader = _unsigned(fields, "original_reader")
                byte_count = _unsigned(fields, "bytes")
                if reader is None or byte_count is None:
                    reasons.append("malformed replacement image")
                else:
                    readers.setdefault(reader, {})["patched_bytes"] = byte_count
        elif "ConSan resources reader=" in line:
            fields = parse(line, "ConSan resources ", "ConSan resources")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                if reader is None:
                    reasons.append("malformed ConSan resources")
                    continue
                resource_names = (
                    "explicit",
                    "dead",
                    "descriptor_growth",
                    "spill",
                    "unsupported",
                    "planned_spill_slot_bytes",
                    "emitted_spill_patches",
                    "emitted_spill_slot_bytes",
                    "alternative_attempts",
                    "alternative_selected",
                    "alternative_rejected",
                    "alternative_superseded",
                    "alternative_contributed",
                    "alternative_vetoed",
                )
                resources = {name: _unsigned(fields, name) for name in resource_names}
                if any(value is None for value in resources.values()):
                    reasons.append("malformed ConSan resources")
                else:
                    readers.setdefault(reader, {})["resources"] = resources
        elif "ConSan report memory required_bytes=" in line:
            fields = parse(line, "ConSan report memory ", "report memory")
            if fields is not None:
                for source, destination in (
                    ("required_bytes", "report_required_bytes"),
                    ("allocated_bytes", "report_allocated_bytes"),
                    ("peak_live_bytes", "report_peak_live_bytes"),
                    ("allocation_failures", "report_allocation_failures"),
                    ("capacity_failures", "report_capacity_failures"),
                    ("cleanup_failures", "report_cleanup_failures"),
                ):
                    value = _unsigned(fields, source)
                    if value is None:
                        reasons.append(f"malformed report memory field {source}")
                    else:
                        process_memory[destination] = value
        elif "ConSan transform admission memory live_bytes=" in line:
            fields = parse(
                line,
                "ConSan transform admission memory ",
                "transform admission memory",
            )
            if fields is not None:
                value = _unsigned(fields, "peak_reserved_bytes")
                if value is None:
                    reasons.append("malformed transform admission memory")
                else:
                    process_memory["transform_peak_reserved_bytes"] = value
        elif "ConSan patched-image memory live_bytes=" in line:
            fields = parse(line, "ConSan patched-image memory ", "patched image memory")
            if fields is not None:
                value = _unsigned(fields, "peak_image_bytes")
                if value is None:
                    reasons.append("malformed patched image memory")
                else:
                    process_memory["patched_image_peak_bytes"] = value
        elif "ConSan patched-image growth memory live_bytes=" in line:
            fields = parse(
                line,
                "ConSan patched-image growth memory ",
                "patched image growth memory",
            )
            if fields is not None:
                value = _unsigned(fields, "peak_growth_bytes")
                if value is None:
                    reasons.append("malformed patched image growth memory")
                else:
                    process_memory["patched_image_peak_growth_bytes"] = value

    code_objects = []
    for reader, record in sorted(readers.items()):
        original = record.get("original_bytes")
        patched = record.get("patched_bytes", original)
        record["reader"] = reader
        record["patched_bytes"] = patched
        if isinstance(original, int) and isinstance(patched, int):
            record["growth_bytes"] = patched - original
            record["growth_ratio"] = patched / original if original else None
        code_objects.append(record)
    patched = [
        record["patch_ms"]
        for record in code_objects
        if isinstance(record.get("patch_ms"), float)
    ]
    return {
        "accepted": not reasons and bool(patched),
        "reasons": reasons,
        "code_objects": code_objects,
        "total_patch_ms": sum(patched),
        "process_memory": process_memory,
    }


def _empirical_structural_totals(result: dict) -> dict[str, float]:
    if result.get("accepted") is not True:
        raise ValidationError("empirical row was rejected")
    runs = result.get("structural_metrics_runs")
    if not isinstance(runs, list) or not runs:
        raise ValidationError("empirical row has no structural metrics")
    if any(run.get("accepted") is not True for run in runs):
        raise ValidationError("empirical row has rejected structural metrics")
    totals = {"patch_ms": 0.0, "waitcheck_ms": 0.0, "inventory_ms": 0.0}
    for run in runs:
        totals["patch_ms"] += float(run["total_patch_ms"])
        for record in run["code_objects"]:
            for source, destination in (
                ("waitcheck_ms", "waitcheck_ms"),
                ("inventory_ms", "inventory_ms"),
            ):
                value = record.get(source)
                if isinstance(value, (int, float)):
                    totals[destination] += float(value)
    return totals


def _parse_amdgpu_kernel_metadata(text: str) -> dict[str, object]:
    integer_fields = {
        "group_segment_fixed_size",
        "private_segment_fixed_size",
        "sgpr_count",
        "sgpr_spill_count",
        "vgpr_count",
        "vgpr_spill_count",
        "agpr_count",
        "accum_offset",
        "wavefront_size",
        "max_flat_workgroup_size",
    }
    kernels = []
    current: dict[str, object] | None = None
    for line in text.splitlines():
        start = re.match(r"^  - \.([a-z_]+):\s*(.*?)\s*$", line)
        if start is not None:
            if current:
                kernels.append(current)
            current = {}
            name, value = start.groups()
            if name in integer_fields and value:
                try:
                    current[name] = int(value, 0)
                except ValueError:
                    current[name] = None
        if current is None:
            continue
        match = re.match(r"^    \.([a-z_]+):\s*(.*?)\s*$", line)
        if match is None:
            continue
        name, value = match.groups()
        if name == "name":
            current[name] = value
        elif name in integer_fields:
            try:
                current[name] = int(value, 0)
            except ValueError:
                current[name] = None
    if current:
        kernels.append(current)
    return {
        "kernels": kernels,
        "kernel_count": len(kernels),
        "fields": sorted(
            {name for kernel in kernels for name in kernel if name != "name"}
        ),
    }


def _llvm_readelf() -> Path | None:
    configured = os.environ.get(LLVM_READELF_ENV)
    if configured:
        return Path(os.path.abspath(Path(configured).expanduser()))
    discovered = shutil.which("llvm-readelf")
    return Path(discovered) if discovered else None


def _amdgpu_kernel_metadata(path: Path) -> dict[str, object]:
    tool = _llvm_readelf()
    if tool is None:
        return {
            "accepted": False,
            "reason": "llvm-readelf is unavailable",
            "tool": None,
        }
    try:
        completed = subprocess.run(
            [str(tool), "--notes", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"accepted": False, "reason": str(error), "tool": str(tool)}
    if completed.returncode != 0:
        return {
            "accepted": False,
            "reason": completed.stderr.strip() or "llvm-readelf failed",
            "tool": str(tool),
        }
    metadata = _parse_amdgpu_kernel_metadata(completed.stdout)
    return {
        "accepted": metadata["kernel_count"] > 0,
        "reason": None if metadata["kernel_count"] > 0 else "no AMDGPU kernels found",
        "tool": str(tool),
        **metadata,
    }


def _retained_code_object_inventory(row_dir: Path) -> dict[str, object]:
    records: dict[tuple[str, str], dict[str, object]] = {}
    for path in sorted(row_dir.glob("code-objects-*/*.hsaco")):
        match = _COVERAGE_DUMP_NAME.fullmatch(path.name)
        if match is None:
            continue
        key = (match.group("dump_id"), match.group("reader"))
        record = records.setdefault(
            key,
            {"dump_id": key[0], "reader": int(key[1])},
        )
        record[match.group("kind")] = {
            "path": _retained_relative_path(row_dir, path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "metadata": _amdgpu_kernel_metadata(path),
        }
    pairs = []
    for record in records.values():
        original = record.get("original")
        patched = record.get("patched")
        if isinstance(original, dict) and isinstance(patched, dict):
            record["growth_bytes"] = patched["size"] - original["size"]
            record["growth_ratio"] = (
                patched["size"] / original["size"] if original["size"] else None
            )
            original_kernels = {
                kernel.get("name"): kernel
                for kernel in original["metadata"].get("kernels", [])
                if isinstance(kernel.get("name"), str)
            }
            patched_kernels = {
                kernel.get("name"): kernel
                for kernel in patched["metadata"].get("kernels", [])
                if isinstance(kernel.get("name"), str)
            }
            original_names = sorted(original_kernels)
            patched_names = sorted(patched_kernels)
            resource_fields = (
                "group_segment_fixed_size",
                "private_segment_fixed_size",
                "sgpr_count",
                "sgpr_spill_count",
                "vgpr_count",
                "vgpr_spill_count",
                "agpr_count",
            )
            record["kernel_metadata_delta"] = {
                "original_names": original_names,
                "patched_names": patched_names,
                "name_sets_match": original_names == patched_names,
                "kernels": {
                    name: {
                        field: patched_kernels[name][field]
                        - original_kernels[name][field]
                        for field in resource_fields
                        if isinstance(original_kernels[name].get(field), int)
                        and isinstance(patched_kernels[name].get(field), int)
                    }
                    for name in sorted(set(original_kernels) & set(patched_kernels))
                },
            }
        pairs.append(record)
    return {
        "pairs": pairs,
        "complete_pairs": sum("growth_bytes" in row for row in pairs),
        "metadata_complete_pairs": sum(
            "growth_bytes" in row
            and row["original"]["metadata"]["accepted"]
            and row["patched"]["metadata"]["accepted"]
            for row in pairs
        ),
    }


def _gtest_test_count(log_text: str) -> int | None:
    matches = re.findall(
        r"\[==========\]\s+Running\s+([0-9]+)\s+tests?\s+from",
        log_text,
    )
    return int(matches[-1]) if matches else None
