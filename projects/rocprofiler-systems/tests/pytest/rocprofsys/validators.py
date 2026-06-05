# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Output validators for rocprofiler-systems test results.

All validation runs in-process via the vendored ``rocprofsys_validator`` framework:
Perfetto / RocPD / timemory-JSON / causal / unified-memory readers act as the
load/query layer (RocPD rule sets live in ``rocprofsys.rocpd_rules``). No
validation shells out to standalone scripts anymore.

We also provide the following validators:
- validate_file_exists
"""

from __future__ import annotations
import contextlib
import io
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes:
        is_valid: Whether the validation passed
        message: Description of result or error
        details: Additional details (e.g., query results)
        stdout: Standard output from validation script
        stderr: Standard error from validation script
        command: The command that was executed
    """

    is_valid: bool
    message: str
    details: Optional[dict[str, Any]] = None
    stdout: str = ""
    stderr: str = ""
    command: str = ""


ROCPROFSYS_ABORT_FAIL_REGEX = [
    r"### ERROR ###",
    r"unknown-hash=",
    r"address of faulting memory reference",
    r"exiting with non-zero exit code",
    r"terminate called after throwing an instance",
    r"calling abort\.\. in ",
    r"Exit code: [1-9]",
]

from rocprofsys.runners import TestResult


def _validate_regex(
    text: str,
    pass_regex: Optional[list[str]] = None,
    fail_regex: Optional[list[str]] = None,
    use_abort_fail_regex: bool = False,
) -> ValidationResult:
    """Validate the regex patterns in some given text.

    Args:
        text: Text to validate
        pass_regex: Optional list of regex patterns that must be found for success
        fail_regex: Optional list of regex patterns that must NOT be found
        use_abort_fail_regex: Whether to validate against ROCPROFSYS_ABORT_FAIL_REGEX (default: True)

    Returns:
        ValidationResult with is_valid=True if all patterns pass, False otherwise
    """
    # Build fail regex list
    fail_patterns: list[str] = []
    if fail_regex:
        fail_patterns.extend(fail_regex)
    if use_abort_fail_regex:
        fail_patterns.extend(ROCPROFSYS_ABORT_FAIL_REGEX)

    # Build combined regex with named groups
    all_patterns: list[str] = []
    fail_indices: set[str] = set()
    pass_indices: set[str] = set()

    if fail_patterns:
        for i, pattern in enumerate(fail_patterns):
            all_patterns.append(f"(?P<f{i}>{pattern})")
            fail_indices.add(f"f{i}")

    if pass_regex:
        for i, pattern in enumerate(pass_regex):
            all_patterns.append(f"(?P<p{i}>{pattern})")
            pass_indices.add(f"p{i}")

    if not all_patterns:
        return ValidationResult(is_valid=True, message="No patterns to validate")

    # Use re.DOTALL so '.' matches newlines (like CMake regex behavior)
    combined_regex = re.compile("|".join(all_patterns), re.DOTALL)
    found_pass: set[str] = set()

    for match in combined_regex.finditer(text):
        matched_group = match.lastgroup

        if matched_group in fail_indices:
            original_idx = int(matched_group[1:])
            return ValidationResult(
                is_valid=False,
                message=f"Fail pattern matched: {fail_patterns[original_idx]}",
            )

        if matched_group in pass_indices:
            found_pass.add(matched_group)

    # Check if all pass patterns were found
    if pass_regex:
        missing = pass_indices - found_pass
        if missing:
            missing_idx = int(next(iter(missing))[1:])
            return ValidationResult(
                is_valid=False,
                message=f"Pass pattern not found: {pass_regex[missing_idx]}",
            )

    return ValidationResult(is_valid=True, message="All patterns validated successfully")


def validate_regex(
    test_result: TestResult,
    pass_regex: Optional[list[str]] = None,
    fail_regex: Optional[list[str]] = None,
    use_abort_fail_regex: bool = True,
) -> ValidationResult:
    return _validate_regex(
        test_result.test_output, pass_regex, fail_regex, use_abort_fail_regex
    )


def validate_file_regex(
    file_path: Path,
    pass_regex: Optional[list[str]] = None,
    fail_regex: Optional[list[str]] = None,
    use_abort_fail_regex: bool = True,
) -> ValidationResult:
    if not file_path.exists():
        return ValidationResult(False, f"File not found: {file_path}")
    with open(file_path, "r") as f:
        text = f.read()
    return _validate_regex(text, pass_regex, fail_regex, use_abort_fail_regex)


def validate_file_exists(path: Path, description: str = "File") -> ValidationResult:
    """Validate that a file exists and is non-empty.

    Args:
        path: Path to check
        description: Description for error messages

    Returns:
        ValidationResult
    """

    if not path.exists():
        return ValidationResult(False, f"{description} not found: {path}")

    if path.stat().st_size == 0:
        return ValidationResult(False, f"{description} is empty: {path}")

    return ValidationResult(True, f"{description} exists: {path}")


# ============================================================================
# Perfetto Validation - in-process via rocprofsys_validator.PerfettoReader
# ============================================================================


def _resolve_trace_processor_bin(trace_processor_path: Optional[Path]) -> Optional[str]:
    """Resolve the trace_processor_shell binary path (env var wins, like the script).

    ``ROCPROFSYS_TRACE_PROC_SHELL`` overrides the caller-supplied path (used to run
    perfetto validation against an older GLIBC). A path that does not point at an
    existing file falls back to the perfetto package's bundled binary (None).
    """
    env_path = os.environ.get("ROCPROFSYS_TRACE_PROC_SHELL")
    bin_path = env_path or (str(trace_processor_path) if trace_processor_path else None)
    if bin_path and not os.path.isfile(bin_path):
        bin_path = None
    return bin_path


def _load_perfetto_reader(trace: str, tp_bin: Optional[str], max_tries: int = 5,
                          retry_wait: int = 1):
    """Build a PerfettoReader, retrying transient trace-processor connection errors.

    Mirrors the retry behaviour of the old ``load_trace`` helper so spurious HTTP
    errors from the trace processor subprocess do not flake the test suite.
    """
    from rocprofsys_validator import PerfettoReader

    last_exc: Optional[BaseException] = None
    for attempt in range(max_tries + 1):
        try:
            return PerfettoReader(trace, tp_bin=tp_bin)
        except Exception as ex:  # noqa: BLE001 - surfaced to caller on final attempt
            last_exc = ex
            sys.stderr.write(f"{ex}\n")
            sys.stderr.flush()
            if attempt >= max_tries:
                raise
            time.sleep(retry_wait)
    assert last_exc is not None  # pragma: no cover - loop always returns or raises
    raise last_exc


def _pf_validate_positional(data, labels, counts, depths, use_substrings=False):
    """Positional (label, count, depth) row-by-row validation (ported from the script)."""
    if not data and labels:
        raise RuntimeError("Data is empty but labels are not")

    if len(labels) != len(counts) or len(labels) != len(depths):
        raise RuntimeError(
            "labels, counts, and depths must have the same length "
            f"(got {len(labels)}, {len(counts)}, {len(depths)})"
        )

    expected = [[litr, citr, ditr] for litr, citr, ditr in zip(labels, counts, depths)]
    for ditr, eitr in zip(data, expected):
        _label = ditr["label"]
        _count = ditr["count"]
        _depth = ditr["depth"]
        if use_substrings:
            if eitr[0] not in _label:
                raise RuntimeError(
                    f"Mismatched label (substring): {_label!r} does not contain {eitr[0]!r}"
                )
        elif _label != eitr[0]:
            raise RuntimeError(
                f"Mismatched label (exact): {_label!r} vs expected {eitr[0]!r}"
            )
        if _count != eitr[1]:
            raise RuntimeError(f"Mismatched count: {_count} vs. {eitr[1]}")
        if _depth != eitr[2]:
            raise RuntimeError(f"Mismatched depth: {_depth} vs. {eitr[2]}")


def _pf_validate_by_label(data, labels, counts, use_substrings=False):
    """Aggregate-by-name validation summing counts across depths (ported from script)."""
    from collections import defaultdict

    presence_only = len(counts) == 0
    if not presence_only and len(counts) != len(labels):
        raise RuntimeError(
            "counts must have one entry per label, or be omitted for presence-only mode"
        )

    totals_by_slice_name: dict[str, int] = defaultdict(int)
    for srow in data:
        totals_by_slice_name[srow["label"]] += srow["count"]

    for i, litr in enumerate(labels):
        if use_substrings:
            total = sum(c for name, c in totals_by_slice_name.items() if litr in name)
        else:
            total = totals_by_slice_name.get(litr, 0)

        if presence_only:
            if total < 1:
                raise RuntimeError(f"No slice found for expected label '{litr}'")
            continue

        if total != counts[i]:
            raise RuntimeError(
                f"Mismatched count for expected label '{litr}': "
                f"got {total}, expected {counts[i]}"
            )


def _perfetto_validate_in_process(
    reader,
    *,
    input_name: str,
    categories: list[str],
    labels: list[str],
    label_substrings: list[str],
    counts: list[int],
    depths: list[int],
    print_output: bool,
    key_names: list[str],
    key_counts: list[int],
    counter_names: list[str],
    check_counter_pairing: bool,
) -> tuple[int, str]:
    """Reproduce validate-perfetto-proto.py exactly, in-process.

    Returns ``(ret, stdout)``. ``ret == 0`` means validated. All ``print`` output is
    captured into the returned string so callers (and conftest ``pass_regex`` /
    ``fail_regex``) see the same text the standalone script emitted.
    """
    if labels and label_substrings:
        raise RuntimeError(
            "Cannot specify both expected labels and expected label substrings"
        )

    expected_labels = labels if labels else label_substrings
    aggregate_by_name = not depths
    use_substrings = bool(label_substrings)

    if expected_labels:
        if aggregate_by_name:
            if counts and len(counts) != len(expected_labels):
                raise RuntimeError(
                    "With -d omitted, provide no -c (presence-only) or one count per label"
                )
        elif len(expected_labels) != len(counts) or len(expected_labels) != len(depths):
            raise RuntimeError(
                "The same number of labels, counts, and depths must be specified "
                "when -d is provided"
            )

    if (key_names or key_counts) and len(key_names) != len(key_counts):
        raise RuntimeError(
            "--key-names and --key-counts must have the same number of entries"
        )

    buf = io.StringIO()
    ret = 0
    with contextlib.redirect_stdout(buf):
        # Build the per-(name, depth) call-count table, filtered by category.
        slice_df = reader.execute_sql("SELECT name, depth, category FROM slice")
        pdata: dict = {}
        for name, depth, category in zip(
            slice_df["name"].tolist(),
            slice_df["depth"].tolist(),
            slice_df["category"].tolist(),
        ):
            if categories and category not in categories:
                continue
            pdata.setdefault(name, {})
            d = int(depth)
            pdata[name][d] = pdata[name].get(d, 0) + 1

        perfetto_data = [
            {"label": name, "count": count, "depth": depth}
            for name, by_depth in pdata.items()
            for depth, count in by_depth.items()
        ]

        if print_output:
            print(f"Printing Perfetto Data {categories}")
            for itr in perfetto_data:
                n = 0 if itr["depth"] < 2 else itr["depth"] - 1
                lbl = "{}{}{}".format(
                    "  " * n, "|_" if itr["depth"] > 0 else "", itr["label"]
                )
                print("| {:40} | {:6} | {:6} |".format(lbl, itr["count"], itr["depth"]))

        try:
            if expected_labels:
                if aggregate_by_name:
                    _pf_validate_by_label(
                        perfetto_data, expected_labels, counts,
                        use_substrings=use_substrings,
                    )
                else:
                    _pf_validate_positional(
                        perfetto_data, expected_labels, counts, depths,
                        use_substrings=use_substrings,
                    )
        except RuntimeError as e:
            print(f"Fail: {e}")
            ret = 1

        for key_name, key_count in zip(key_names, key_counts):
            slice_args = reader.execute_sql(
                "select * from slice join args using (arg_set_id) "
                f"where key='debug.{key_name}'"
            )
            count = len(slice_args)
            if print_output:
                print(f"{key_name} (expected: {key_count}):")
                for rec in slice_args.to_dict(orient="records"):
                    for key, val in rec.items():
                        print(f"  - {key:20} :: {val}")
            print(f"Number of entries with {key_name} = {count} (expected: {key_count})")
            if key_count != count:
                ret = 1

        if counter_names and print_output:
            all_counter_tracks = reader.execute_sql(
                "SELECT DISTINCT name FROM counter_track ORDER BY name"
            )
            track_names = [n for n in all_counter_tracks["name"].tolist()]
            print(f"Available counter tracks ({len(track_names)}):")
            for name in track_names:
                print(f"  - {name}")

        for counter_name in counter_names:
            if print_output:
                matching_tracks = reader.execute_sql(
                    "SELECT counter_track.name, COUNT(counter.id) AS num_entries, "
                    "SUM(counter.value) AS sum_value, MIN(counter.value) AS min_value, "
                    "MAX(counter.value) AS max_value "
                    "FROM counter_track JOIN counter ON counter.track_id = counter_track.id "
                    f"WHERE counter_track.name LIKE '%{counter_name}%' "
                    "GROUP BY counter_track.name ORDER BY counter_track.name"
                )
                if len(matching_tracks) == 0:
                    print(f"  No counter tracks matching '%{counter_name}%' found in trace")
                for rec in matching_tracks.to_dict(orient="records"):
                    print(
                        f"  Track: {rec['name']} | entries={rec['num_entries']} "
                        f"sum={rec['sum_value']} min={rec['min_value']} max={rec['max_value']}"
                    )

            sum_df = reader.execute_sql(
                "SELECT SUM(counter.value) AS total_value FROM counter_track "
                "JOIN counter ON counter.track_id = counter_track.id "
                f"WHERE counter_track.name LIKE '%{counter_name}%'"
            )
            total_value = 0
            if len(sum_df) > 0:
                raw = sum_df["total_value"].iloc[0]
                total_value = -1 if raw is None or _is_nan(raw) else raw

            if print_output:
                print(f"Total value of {counter_name} is {total_value}")

            if total_value <= 0:
                print(f"Fail: Counter {counter_name} is not found in the traces")
                ret = 1

        if check_counter_pairing and counter_names:
            for counter_name in counter_names:
                tracks = reader.execute_sql(
                    "SELECT counter_track.id, counter_track.name, "
                    "COUNT(counter.id) AS num_entries "
                    "FROM counter_track JOIN counter ON counter.track_id = counter_track.id "
                    f"WHERE counter_track.name LIKE '%{counter_name}%' "
                    "GROUP BY counter_track.id"
                )
                for rec in tracks.to_dict(orient="records"):
                    if rec["num_entries"] % 2 != 0:
                        print(
                            f"Fail: Counter track '{rec['name']}' has {rec['num_entries']} "
                            "entries (expected even number for paired start/end)"
                        )
                        ret = 1
                    else:
                        last_df = reader.execute_sql(
                            "SELECT counter.value FROM counter "
                            f"WHERE counter.track_id = {rec['id']} "
                            "ORDER BY counter.ts DESC LIMIT 1"
                        )
                        if len(last_df) > 0 and last_df["value"].iloc[0] != 0:
                            print(
                                f"Fail: Counter track '{rec['name']}' last value is "
                                f"{last_df['value'].iloc[0]} (expected 0 for end marker)"
                            )
                            ret = 1

        if ret == 0:
            print(f"{input_name} validated")
        else:
            print(f"Failure validating {input_name}")

    return ret, buf.getvalue()


def _is_nan(value: Any) -> bool:
    """True only for NaN floats (NULL SQL aggregates surface as NaN via pandas)."""
    return isinstance(value, float) and value != value


def validate_perfetto_trace(
    trace_path: Path,
    tests_dir: Path,
    categories: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
    counts: Optional[list[int]] = None,
    depths: Optional[list[int]] = None,
    label_substrings: Optional[list[str]] = None,
    counter_names: Optional[list[str]] = None,
    key_names: Optional[list[str]] = None,
    key_counts: Optional[list[int]] = None,
    trace_processor_path: Optional[Path] = None,
    print_output: bool = False,
    check_counter_pairing: bool = False,
    timeout: int = 120,
) -> ValidationResult:
    """Validate a Perfetto trace in-process via rocprofsys_validator.PerfettoReader.

    Slice validation mode is inferred: pass ``depths`` (-d) for positional row-by-row
    checks; omit ``depths`` for aggregate-by-name (sum counts across depths). Omit
    ``counts`` (-c) in aggregate mode for presence-only checks.

    Args:
        trace_path: Path to perfetto-trace.proto file
        tests_dir: Unused (kept for signature compatibility)
        categories: List of categories to filter by
        labels: Expected labels (exact match)
        counts: Expected counts
        depths: Expected depths; omit for aggregate-by-name validation
        label_substrings: Expected label substrings (substring match)
        counter_names: Counter names to validate
        key_names: Debug annotation key names to check
        key_counts: Expected counts for debug annotation keys
        trace_processor_path: Path to trace_processor_shell
        print_output: Whether to render the slice/counter tables into stdout
        check_counter_pairing: Verify counter tracks have paired start/end entries
        timeout: Accepted for compatibility; in-process validation has no subprocess

    Returns:
        ValidationResult with validation status
    """
    if not trace_path.exists():
        return ValidationResult(False, f"Trace file not found: {trace_path}")

    categories = categories or []
    labels = labels or []
    counts = counts or []
    depths = depths or []
    label_substrings = label_substrings or []
    counter_names = counter_names or []
    key_names = key_names or []
    key_counts = key_counts or []

    cmd_str = (
        f"validate_perfetto_trace(in-process) input={trace_path} "
        f"categories={categories} labels={labels} substrings={label_substrings} "
        f"counts={counts} depths={depths} counter_names={counter_names} "
        f"check_counter_pairing={check_counter_pairing} "
        f"key_names={key_names} key_counts={key_counts}"
    )

    try:
        from rocprofsys_validator import PerfettoReader  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return ValidationResult(
            False, f"rocprofsys_validator import failed: {e}", command=cmd_str
        )

    tp_bin = _resolve_trace_processor_bin(trace_processor_path)

    reader = None
    try:
        reader = _load_perfetto_reader(str(trace_path), tp_bin)
    except Exception as e:  # noqa: BLE001
        return ValidationResult(
            False, f"Failed to load trace {trace_path}: {e}", command=cmd_str
        )

    try:
        ret, output = _perfetto_validate_in_process(
            reader,
            input_name=str(trace_path),
            categories=categories,
            labels=labels,
            label_substrings=label_substrings,
            counts=counts,
            depths=depths,
            print_output=print_output,
            key_names=key_names,
            key_counts=key_counts,
            counter_names=counter_names,
            check_counter_pairing=check_counter_pairing,
        )
    except RuntimeError as e:
        return ValidationResult(False, str(e), command=cmd_str)
    except Exception as e:  # noqa: BLE001
        return ValidationResult(False, f"Validation error: {e}", command=cmd_str)
    finally:
        with contextlib.suppress(Exception):
            reader.close()

    return ValidationResult(
        is_valid=(ret == 0),
        message=output.strip(),
        stdout=output,
        command=cmd_str,
    )


# ============================================================================
# ROCpd Database Validation - in-process via rocprofsys_validator.RocpdReader
# ============================================================================


def _detect_available_metrics(tests_dir: Path) -> Optional[set]:
    """Detect available GPU metrics for ``requires``-gated rules (via amd-smi).

    Imports check_amd_smi_metrics from the tests dir and unions the metric names
    across all GPUs. Returns None on any failure so gated rules run unconditionally.
    """
    try:
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))
        from check_amd_smi_metrics import get_available_metrics_set

        return get_available_metrics_set()
    except Exception:  # noqa: BLE001 - any failure => run all queries
        return None


def validate_rocpd_database(
    db_path: Path,
    tests_dir: Path,
    rule_sets: Optional[list] = None,
    timeout: int = 60,
) -> ValidationResult:
    """Validate a ROCpd database file in-process via native rule sets.

    Args:
        db_path: Path to rocpd.db file
        tests_dir: Path to directory containing the check_amd_smi_metrics helper
        rule_sets: Native rule-set callables (see rocprofsys.rocpd_rules)
        timeout: Accepted for compatibility; in-process validation has no subprocess

    Returns:
        ValidationResult with validation status
    """
    if not db_path.exists():
        return ValidationResult(False, f"Database not found: {db_path}")

    rule_sets = rule_sets or []
    names = [getattr(rs, "__name__", str(rs)) for rs in rule_sets]
    cmd_str = f"validate_rocpd_database(in-process) db={db_path} rule_sets={names}"

    try:
        from rocprofsys_validator import RocpdReader
        from rocprofsys.rocpd_rules import run_rule_sets
    except Exception as e:  # noqa: BLE001
        return ValidationResult(False, f"rocprofsys_validator import failed: {e}", command=cmd_str)

    available_metrics = _detect_available_metrics(tests_dir)

    try:
        reader = RocpdReader(str(db_path))
    except Exception as e:  # noqa: BLE001
        return ValidationResult(False, f"Failed to open database {db_path}: {e}", command=cmd_str)

    try:
        is_valid, message = run_rule_sets(reader, rule_sets, available_metrics)
    except Exception as e:  # noqa: BLE001
        return ValidationResult(False, f"Validation error: {e}", command=cmd_str)
    finally:
        with contextlib.suppress(Exception):
            reader.close()

    return ValidationResult(is_valid=is_valid, message=message, stdout=message, command=cmd_str)


# ============================================================================
# Timemory JSON Validation - in-process via rocprofsys_validator.TimemoryJsonReader
# ============================================================================


def _tm_validate_json(data, labels, counts, depths):
    """Positional prefix/laps/depth validation (ported from validate-timemory-json.py).

    The ``>>>`` prefix strip is reproduced verbatim, including the legacy behaviour
    where a missing ``>>>`` (find returns -1) slices ``prefix[3:]``.
    """
    expected = []
    for litr, citr, ditr in zip(labels, counts, depths):
        _label = litr
        if ditr > 0:
            _label = "{}|_{}".format("  " * (ditr - 1), litr)
        expected.append([_label, citr, ditr])

    for ditr, eitr in zip(data, expected):
        _prefix = ditr["prefix"]
        _depth = ditr["depth"]
        _count = ditr["entry"]["laps"]
        _idx = _prefix.find(">>>")
        _prefix = _prefix[(_idx + 4):]
        if _prefix != eitr[0]:
            raise RuntimeError(f"Mismatched prefix: {_prefix} vs. {eitr[0]}")
        if _count != eitr[1]:
            raise RuntimeError(f"Mismatched count for {_prefix}: {_count} vs. {eitr[1]}")
        if _depth != eitr[2]:
            raise RuntimeError(f"Mismatched depth for {_prefix}: {_depth} vs. {eitr[2]}")


def validate_timemory_json(
    json_path: Path,
    tests_dir: Path,
    metric: str,
    labels: Optional[list[str]] = None,
    counts: Optional[list[int]] = None,
    depths: Optional[list[int]] = None,
    print_output: bool = False,
    timeout: int = 60,
) -> ValidationResult:
    """Validate a timemory JSON output file in-process via TimemoryJsonReader.

    Args:
        json_path: Path to JSON file
        metric: Metric name (JSON key under data["timemory"])
        tests_dir: Unused (kept for signature compatibility)
        labels: Expected labels
        counts: Expected counts (laps)
        depths: Expected depths
        print_output: Whether to render the graph table into stdout
        timeout: Accepted for compatibility; in-process validation has no subprocess

    Returns:
        ValidationResult with validation status
    """
    if not json_path.exists():
        return ValidationResult(False, f"JSON file not found: {json_path}")

    labels = labels or []
    counts = counts or []
    depths = depths or []

    cmd_str = (
        f"validate_timemory_json(in-process) input={json_path} metric={metric} "
        f"labels={labels} counts={counts} depths={depths}"
    )

    if len(labels) != len(counts) or len(labels) != len(depths):
        return ValidationResult(
            False,
            "The same number of labels, counts, and depths must be specified",
            command=cmd_str,
        )

    try:
        from rocprofsys_validator import TimemoryJsonReader
    except Exception as e:  # noqa: BLE001
        return ValidationResult(
            False, f"rocprofsys_validator import failed: {e}", command=cmd_str
        )

    try:
        reader = TimemoryJsonReader(str(json_path))
    except Exception as e:  # noqa: BLE001
        return ValidationResult(
            False, f"Failed to load {json_path}: {e}", command=cmd_str
        )

    buf = io.StringIO()
    ret = 0
    try:
        with contextlib.redirect_stdout(buf):
            # Reach the graph the same way the script did so a missing metric/graph
            # raises (parity with the script's uncaught KeyError -> non-zero exit).
            graph = reader._data["timemory"][metric]["ranks"][0]["graph"]

            if print_output:
                for itr in graph:
                    _prefix = itr["prefix"]
                    _depth = itr["depth"]
                    _count = itr["entry"]["laps"]
                    _idx = _prefix.find(">>>")
                    _prefix = _prefix[(_idx + 4):]
                    print("| {:40} | {:6} | {:6} |".format(_prefix, _count, _depth))

            try:
                _tm_validate_json(graph, labels, counts, depths)
            except RuntimeError as e:
                print(f"{e}")
                ret = 1

            if ret == 0:
                print(f"{json_path} validated")
    except Exception as e:  # noqa: BLE001 - parity with the script's traceback->failure
        return ValidationResult(
            False, f"{type(e).__name__}: {e}", stdout=buf.getvalue(), command=cmd_str
        )
    finally:
        with contextlib.suppress(Exception):
            reader.close()

    output = buf.getvalue()
    return ValidationResult(
        is_valid=(ret == 0),
        message=output.strip(),
        stdout=output,
        command=cmd_str,
    )


# ============================================================================
# Causal JSON Validation - in-process via rocprofsys_validator causal CLI
# ============================================================================


def validate_causal_json(
    json_path: Path,
    tests_dir: Path,
    ci_mode: bool = False,
    additional_args: Optional[list[str]] = None,
    timeout: int = 60,
) -> ValidationResult:
    """Validate a causal profiling JSON file in-process via the framework causal CLI.

    Args:
        json_path: Path to causal JSON file
        tests_dir: Unused (kept for signature compatibility)
        ci_mode: Whether running in CI mode (--ci flag)
        additional_args: Additional CLI arguments (same surface as the old script)
        timeout: Accepted for compatibility; in-process validation has no subprocess

    Returns:
        ValidationResult with validation status
    """
    if not json_path.exists():
        return ValidationResult(False, f"JSON file not found: {json_path}")

    additional_args = additional_args or []
    argv = ["-i", str(json_path)]
    if ci_mode:
        argv.append("--ci")
    argv.extend(additional_args)
    cmd_str = "validate_causal_json(in-process) " + " ".join(argv)

    try:
        from rocprofsys_validator.readers._causal_validator import main as causal_main
    except Exception as e:  # noqa: BLE001
        return ValidationResult(
            False, f"rocprofsys_validator import failed: {e}", command=cmd_str
        )

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            rc = causal_main(argv)
    except SystemExit as e:  # argparse error / explicit exit
        rc = e.code if isinstance(e.code, int) else 1
    except Exception as e:  # noqa: BLE001
        return ValidationResult(
            False, f"Validation error: {e}", stdout=out_buf.getvalue(), command=cmd_str
        )

    out = out_buf.getvalue()
    err = err_buf.getvalue()
    if rc == 0:
        message = out.strip()
    else:
        message = err.strip() or out.strip() or f"Exit code: {rc}"

    return ValidationResult(
        is_valid=(rc == 0), message=message, stdout=out, stderr=err, command=cmd_str
    )


def validate_unified_memory_outputs(
    output_dir: Path,
    tests_dir: Path,
    timeout: int = 60,
) -> ValidationResult:
    """Validate unified-memory text and JSON outputs in-process via UnifiedMemoryReader."""
    txt_matches = sorted(output_dir.rglob("unified_memory*.txt"))
    json_matches = sorted(output_dir.rglob("unified_memory*.json"))

    if not txt_matches:
        return ValidationResult(False, f"No unified_memory*.txt found under {output_dir}")
    if not json_matches:
        return ValidationResult(
            False, f"No unified_memory*.json found under {output_dir}"
        )

    txt_file = txt_matches[0]
    json_file = json_matches[0]

    if txt_file.parent != json_file.parent:
        return ValidationResult(
            False,
            "Unified-memory outputs landed in different directories: "
            f"{txt_file.parent} vs {json_file.parent}",
        )

    target_dir = txt_file.parent
    cmd_str = f"validate_unified_memory_outputs(in-process) output_dir={target_dir}"

    try:
        from rocprofsys_validator.readers.unified_memory import UnifiedMemoryReader
    except Exception as e:  # noqa: BLE001
        return ValidationResult(
            False, f"rocprofsys_validator import failed: {e}", command=cmd_str
        )

    try:
        is_valid, text = UnifiedMemoryReader(target_dir).validate_outputs()
    except Exception as e:  # noqa: BLE001
        return ValidationResult(False, f"Validation error: {e}", command=cmd_str)

    return ValidationResult(
        is_valid=is_valid, message=text.strip(), stdout=text, command=cmd_str
    )
