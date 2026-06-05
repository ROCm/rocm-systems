# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native RocPD validation rules.

These are the in-process replacement for the ``rocpd-validation-rules/*.json`` tree.
Each rule set is a plain Python function that receives a :class:`RocpdValidator`
(bound to a ``rocprofsys_validator.RocpdReader``) and expresses its checks as direct
framework ``execute_sql`` calls — table presence, required columns, minimum row
counts, and SQL count queries with comparison operators and optional GPU-metric
gating (``requires``).

Tests reference these functions by importing them (or via :data:`RULE_SETS`) and
passing them to ``assert_rocpd(rule_sets=[...])`` instead of JSON file paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

# A rule set runs its checks against a RocpdValidator, recording failures on it.
RuleSet = Callable[["RocpdValidator"], None]


def _compare(actual, op: str, expected, expected_max=None) -> bool:
    """Apply a comparison operator (mirrors validate-rocpd.py validate_query)."""
    if actual is None:
        # A missing scalar can never satisfy a numeric comparison; treat as failure
        # (the standalone script would raise and exit non-zero — same verdict).
        return op == "equals" and expected is None
    if op == "equals":
        return actual == expected
    if op == "not_equals":
        return actual != expected
    if op == "greater_than":
        return actual > expected
    if op == "less_than":
        return actual < expected
    if op == "greater_than_or_equal":
        return actual >= expected
    if op == "less_than_or_equal":
        return actual <= expected
    if op == "between_inclusive":
        if expected_max is None:
            raise ValueError("between_inclusive requires expected_max")
        return expected <= actual <= expected_max
    raise ValueError(f"Unknown comparison operator: {op}")


@dataclass(frozen=True)
class Q:
    """A single SQL count/scalar validation query.

    ``query`` may contain the ``{table_name}`` placeholder (substituted with each
    matching table name, exactly like the JSON engine). ``requires`` names a GPU
    metric that must be available for the query to run; when absent it is skipped.
    """

    op: str
    expected: object
    query: str
    error: str
    expected_max: object = None
    requires: Optional[str] = None

    def summary(self) -> str:
        if self.op == "between_inclusive":
            return f"between_inclusive [{self.expected}, {self.expected_max}]"
        return f"{self.op} {self.expected}"


class RocpdValidator:
    """Runs native RocPD rules against a ``rocprofsys_validator.RocpdReader``.

    Structural checks (required columns, minimum row count) delegate to the reader's
    high-level ``assert_columns_exist`` / ``assert_min_row_count`` (and the semantic
    ``hip_api`` / ``kernel_dispatches`` / ``memory_copies`` / ``pmc_events`` /
    ``agent_info`` helpers). Arbitrary SQL count rules use the :class:`Q` helper via
    ``execute_sql``. This module imports nothing from the framework — it operates on
    the ``reader`` passed in and reads the returned ``CheckResult`` objects — so it
    stays cheap to import (no perfetto/pandas) in tests that only reference rule sets.
    """

    def __init__(self, reader, available_metrics: Optional[set] = None) -> None:
        self._reader = reader
        self._available = available_metrics
        self.failures: list[str] = []
        self.checks_run = 0

    # -- low-level helpers ----------------------------------------------------
    def _scalar(self, query: str):
        rows = self._reader.execute_sql(query)
        if not rows:
            return None
        row = rows[0]
        keys = row.keys()
        return row["count"] if "count" in keys else row[0]

    def _matching_tables(self, name: Optional[str], prefix: Optional[str]) -> list[str]:
        rows = self._reader.execute_sql(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
        names = [r["name"] for r in rows]
        if name is not None:
            return [name] if name in names else []
        return [n for n in names if n.startswith(prefix)]

    def _record(self, result) -> bool:
        """Record a framework CheckResult; return whether it passed."""
        self.checks_run += 1
        if not result.passed:
            self.failures.append(result.message)
        return result.passed

    def _run_queries(self, table: str, queries: tuple[Q, ...]) -> None:
        for q in queries:
            if (
                q.requires
                and self._available is not None
                and q.requires not in self._available
            ):
                continue
            self.checks_run += 1
            actual = self._scalar(q.query.replace("{table_name}", table))
            if not _compare(actual, q.op, q.expected, q.expected_max):
                self.failures.append(
                    f"{q.error} (Table: '{table}'); expected {q.summary()}, got {actual}"
                )

    # -- rule API -------------------------------------------------------------
    def table(
        self,
        *,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        columns: list[str],
        min_rows: int = 1,
        queries: tuple[Q, ...] = (),
    ) -> None:
        """Validate a table (exact ``name`` or all tables matching ``prefix``).

        For each matching table: required columns must be present (delegated to
        ``assert_columns_exist``), the row count must be >= ``min_rows`` (delegated
        to ``assert_min_row_count``), and each query must pass. A table that fails
        the column or row-count gate has its queries skipped (parity with the script).
        """
        if (name is None) == (prefix is None):
            raise ValueError("specify exactly one of name= or prefix=")

        matched = self._matching_tables(name, prefix)
        if not matched:
            ident = name if name is not None else f"{prefix}*"
            self.failures.append(f"Required table '{ident}' not found in database")
            return

        for table in matched:
            if not self._record(self._reader.assert_columns_exist(table, columns)):
                continue
            if not self._record(self._reader.assert_min_row_count(table, min_rows)):
                continue
            self._run_queries(table, queries)

    def queries_on(self, table: str, *queries: Q) -> None:
        """Run arbitrary SQL queries against an already-present table."""
        self._run_queries(table, queries)

    # -- semantic high-level delegators (use where a rule maps cleanly) -------
    def hip_api(self, function=None, min_count=1, category="rocm_hip_api") -> None:
        self._record(
            self._reader.assert_hip_api_calls_present(function, min_count, category)
        )

    def kernel_dispatches(self, min_count=1) -> None:
        self._record(self._reader.assert_kernel_dispatches_valid(min_count))

    def memory_copies(self, direction=None, min_count=1) -> None:
        self._record(self._reader.assert_memory_copies_present(direction, min_count))

    def pmc_events(self, min_count=1) -> None:
        self._record(self._reader.assert_pmc_events_present(min_count))

    def agent_info(self, expected_type="GPU", expected_vendor="AMD") -> None:
        self._record(self._reader.assert_agent_info(expected_type, expected_vendor))


def run_rule_sets(
    reader,
    rule_sets: list[RuleSet],
    available_metrics: Optional[set] = None,
) -> tuple[bool, str]:
    """Run every rule set against ``reader``; return (is_valid, message)."""
    validator = RocpdValidator(reader, available_metrics)
    for rule_set in rule_sets:
        rule_set(validator)
    if validator.failures:
        return False, "RocPD validation failed:\n  - " + "\n  - ".join(validator.failures)
    return True, f"RocPD validated ({validator.checks_run} checks passed)"


# ===========================================================================
# Rule sets (one per former JSON file). Add new apps here as they are ported.
# ===========================================================================

_KERNELS_COLUMNS = ["id", "category", "name", "start", "end", "queue", "stream"]
_REGIONS_COLUMNS = ["tid", "start", "end", "name"]


def default_rules(v: RocpdValidator) -> None:
    """Baseline checks (former default-rules.json)."""
    v.table(
        name="top_kernels",
        columns=["name", "total_calls", "total_duration", "average", "percentage"],
        queries=(
            Q("equals", 0, "SELECT COUNT(*) FROM top_kernels WHERE name IS NULL",
              "Found API calls with null function names"),
        ),
    )
    v.table(
        name="kernels",
        columns=_KERNELS_COLUMNS,
        queries=(
            Q("equals", 0, "SELECT COUNT(*) as count FROM kernels WHERE name IS NULL",
              "Found kernels with null function names"),
            Q("equals", 0, "SELECT COUNT(*) as count FROM kernels WHERE (end - start) = 0",
              "Kernels with no active execution times found"),
        ),
    )
    v.table(name="threads", columns=["tid", "start", "end", "name"], min_rows=0)
    v.table(
        name="regions",
        columns=_REGIONS_COLUMNS,
        queries=(
            Q("equals", 0, "SELECT COUNT(*) as count FROM regions WHERE start = 0",
              "Found entries with zero start times in table"),
        ),
    )


def recursion_rules(v: RocpdValidator) -> None:
    """Minimal recursion checks (former minimal/recursion-rules.json)."""
    v.table(
        name="regions",
        columns=["name"],
        queries=(
            Q("equals", 101, "SELECT COUNT(*) FROM regions WHERE name = 'recurse'",
              "Expected 101 'recurse' rows (recurse(100) + 100 descents); fewer rows "
              "indicate the per-thread region cache is collapsing nested same-name calls"),
        ),
    )


def shmem_rules(v: RocpdValidator) -> None:
    """OpenSHMEM checks (former shmem/validation-rules.json)."""
    v.table(
        prefix="rocpd_string",
        columns=["id", "string"],
        queries=(
            Q("greater_than", 0,
              "SELECT COUNT(*) FROM {table_name} WHERE string LIKE 'shmem%';",
              "'shmem' string not found in the table rocpd_string"),
        ),
    )
    v.table(
        name="regions",
        columns=["tid", "start", "end", "name", "category"],
        queries=(
            Q("greater_than", 0,
              "SELECT COUNT(*) FROM {table_name} WHERE category = 'shmem';",
              "'shmem' category entries are fewer than expected in regions"),
        ),
    )


def ucx_rules(v: RocpdValidator) -> None:
    """UCX checks (former ucx/validation-rules.json)."""
    v.table(
        prefix="rocpd_string",
        columns=["id", "string"],
        queries=(
            Q("equals", 2,
              "SELECT COUNT(*) FROM {table_name} WHERE string LIKE '%UCX Comm%';",
              "'UCX Comm' string not found in the table rocpd_string"),
        ),
    )
    v.table(
        name="regions",
        columns=["tid", "start", "end", "name", "category"],
        queries=(
            Q("equals", 0,
              "SELECT COUNT(*) as count FROM {table_name} WHERE start = 0",
              "Found entries with zero start times in table"),
            Q("greater_than", 100,
              "SELECT COUNT(*) FROM {table_name} WHERE category = 'ucx';",
              "'ucx' category entries are fewer than expected in regions"),
        ),
    )


# Shared column lists reused across apps.
_TOP_KERNELS_COLS = ["name", "total_calls", "total_duration", "average", "percentage"]
_THREADS_COLS = ["tid", "start", "end", "name"]
_REGIONS_GUID_COLS = ["id", "guid", "category", "name"]
_EVENTS_ARGS_COLS = ["event_id", "category", "stack_id", "parent_stack_id", "correlation_id"]
_ROCPD_STRING_COLS = ["id", "guid", "string"]
_PMC_INFO_COLS = ["agent_id", "target_arch", "name", "symbol", "description", "units", "value_type"]
_PMC_EVENT_COLS = ["event_id", "pmc_id", "value"]


def gpu_connect_validation_rules(v: RocpdValidator) -> None:
    """gpu-connect HIP API + agent checks (former gpu-connect/validation-rules.json)."""
    v.table(
        name="events_args",
        columns=_EVENTS_ARGS_COLS,
        queries=(
            Q("greater_than", 100, "SELECT COUNT(*) FROM events_args WHERE category = 'rocm_hip_api';",
              "'rocm_hip_api' category entries are fewer than expected in events_args"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM events_args WHERE category IS NULL OR TRIM(category) = '';",
              "Empty or NULL category entries found in events_args"),
        ),
    )
    v.table(
        name="regions",
        columns=_REGIONS_GUID_COLS,
        queries=(
            Q("greater_than", 50, "SELECT COUNT(*) FROM regions WHERE category = 'rocm_hip_api';",
              "'rocm_hip_api' category entries are fewer than expected in regions"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM regions WHERE category = 'rocm_hip_api' AND duration = 0;",
              "Found HIP API captures where duration is 0"),
            Q("equals", 0, "SELECT COUNT(*) FROM regions WHERE name IS NULL;",
              "NULL entries found in the name column of regions"),
        ),
    )
    v.table(
        name="rocpd_info_agent",
        columns=["id", "guid", "nid", "pid", "type", "name"],
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM rocpd_info_agent WHERE type = 'GPU'",
              "No GPU agents found"),
            Q("equals", 0, "SELECT COUNT(*) as count FROM rocpd_info_agent WHERE name IS NULL",
              "Found agents with NULL names"),
        ),
    )


def gpu_connect_amd_smi_rules(v: RocpdValidator) -> None:
    """gpu-connect XGMI/PCIe amd-smi checks (former gpu-connect/amd-smi-rules.json)."""
    v.table(
        prefix="rocpd_info_pmc",
        columns=_PMC_INFO_COLS,
        min_rows=1,
        queries=(
            Q("greater_than", 1, "SELECT COUNT(*) as count FROM {table_name} WHERE symbol LIKE 'Xgmi%'",
              "Did not find Xgmi data in amd-smi metrics", requires="xgmi"),
            Q("greater_than", 1, "SELECT COUNT(*) as count FROM {table_name} WHERE symbol LIKE 'Pcie%'",
              "Did not find Pcie data in amd-smi metrics", requires="pcie"),
        ),
    )
    _join = (
        "SELECT COUNT(*) as count FROM {table_name} event "
        "JOIN rocpd_info_pmc info ON event.pmc_id = info.id WHERE "
    )
    v.table(
        prefix="rocpd_pmc_event",
        columns=_PMC_EVENT_COLS,
        min_rows=500,
        queries=(
            Q("greater_than", 100, _join + "info.name = 'device_xgmi_link_speed'",
              "Less than expected number of captured amd-smi xgmi link speed samples!", requires="xgmi"),
            Q("greater_than", 100, _join + "info.name = 'device_xgmi_link_width'",
              "Less than expected number of captured amd-smi xgmi link width samples!", requires="xgmi"),
            Q("greater_than", 100, _join + "info.name LIKE 'device_xgmi_read_data%'",
              "Less than expected number of captured amd-smi xgmi read data samples!", requires="xgmi"),
            Q("greater_than", 100, _join + "info.name LIKE 'device_xgmi_write_data%'",
              "Less than expected number of captured amd-smi xgmi write data samples!", requires="xgmi"),
            Q("greater_than", 100, _join + "info.name = 'device_pcie_bandwidth_inst'",
              "Less than expected number of captured amd-smi pcie bandwidth instantaneous samples!", requires="pcie"),
            Q("greater_than", 100, _join + "info.name = 'device_pcie_bandwidth_acc'",
              "Less than expected number of captured amd-smi pcie bandwidth accumulated samples!", requires="pcie"),
            Q("greater_than", 100, _join + "info.name = 'device_pcie_link_speed'",
              "Less than expected number of captured amd-smi pcie link speed samples!", requires="pcie"),
            Q("greater_than", 100, _join + "info.name = 'device_pcie_link_width'",
              "Less than expected number of captured amd-smi pcie link width samples!", requires="pcie"),
        ),
    )


def jpeg_validation_rules(v: RocpdValidator) -> None:
    """jpeg-decode kernel/thread checks (former jpeg-decode/validation-rules.json)."""
    v.table(
        name="top_kernels",
        columns=_TOP_KERNELS_COLS,
        queries=(
            Q("equals", 0, "SELECT COUNT(*) FROM top_kernels WHERE name IS NULL",
              "Found API calls with null function names"),
            Q("greater_than", 0, "SELECT COUNT(*) FROM top_kernels",
              "No kernel calls found in summary"),
        ),
    )
    v.table(
        name="kernels",
        columns=_KERNELS_COLUMNS,
        min_rows=50,
        queries=(
            Q("equals", 0, "SELECT COUNT(*) as count FROM kernels WHERE name IS NULL",
              "Found kernels with null function names"),
            Q("greater_than", 50, "SELECT COUNT(*) as count FROM kernels",
              "No kernel entries found"),
            Q("equals", 0, "SELECT COUNT(*) as count FROM kernels WHERE (end - start) = 0",
              "Kernels with no active execution times found"),
        ),
    )
    v.table(name="threads", columns=_THREADS_COLS, min_rows=2)


def jpeg_sdk_metrics_rules(v: RocpdValidator) -> None:
    """jpeg-decode rocJPEG + HIP API checks (former jpeg-decode/sdk-metrics-rules.json)."""
    v.table(
        name="regions",
        columns=_REGIONS_GUID_COLS,
        queries=(
            Q("greater_than", 100, "SELECT COUNT(*) FROM regions WHERE category = 'rocm_rocjpeg_api';",
              "'rocm_rocjpeg_api' category entries are fewer than expected in regions"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM regions WHERE category = 'rocm_rocjpeg_api' AND duration = 0;",
              "Found rocJPEG API captures where duration is 0"),
            Q("equals", 0, "SELECT COUNT(*) FROM regions WHERE name IS NULL;",
              "NULL entries found in the name column of regions"),
        ),
    )
    v.table(
        name="rocpd_string",
        columns=_ROCPD_STRING_COLS,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%rocm_rocjpeg_api%';",
              "'rocm_rocjpeg_api' string not found in the table rocpd_string"),
        ),
    )
    v.table(
        name="events_args",
        columns=_EVENTS_ARGS_COLS,
        queries=(
            Q("greater_than", 100, "SELECT COUNT(*) FROM events_args WHERE category = 'rocm_hip_api';",
              "'rocm_hip_api' category entries are fewer than expected in events_args"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM events_args WHERE category IS NULL OR TRIM(category) = '';",
              "Empty or NULL category entries found in events_args"),
        ),
    )
    v.table(
        name="regions",
        columns=_REGIONS_GUID_COLS,
        queries=(
            Q("greater_than", 50, "SELECT COUNT(*) FROM regions WHERE category = 'rocm_hip_api';",
              "'rocm_hip_api' category entries are fewer than expected in regions"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM regions WHERE category = 'rocm_hip_api' AND duration = 0;",
              "Found HIP API captures where duration is 0"),
            Q("equals", 0, "SELECT COUNT(*) FROM regions WHERE name IS NULL;",
              "NULL entries found in the name column of regions"),
        ),
    )
    v.table(
        name="rocpd_string",
        columns=_ROCPD_STRING_COLS,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%rocm_hip_api%';",
              "'rocm_hip_api' string not found in the table rocpd_string"),
        ),
    )


def jpeg_amd_smi_rules(v: RocpdValidator) -> None:
    """jpeg-decode amd-smi PMC checks (former jpeg-decode/amd-smi-rules.json)."""
    v.table(
        prefix="rocpd_info_pmc_",
        columns=_PMC_INFO_COLS,
        min_rows=1,
        queries=(
            Q("greater_than", 1, "SELECT COUNT(*) as count FROM {table_name} WHERE symbol LIKE 'device_jpeg_activity_%'",
              "Did not find jpeg_activity in amd-smi metrics", requires="jpeg_activity"),
        ),
    )
    _join = (
        "SELECT COUNT(*) as count FROM {table_name} event "
        "JOIN rocpd_info_pmc info ON event.pmc_id = info.id WHERE "
    )
    v.table(
        prefix="rocpd_pmc_event_",
        columns=_PMC_EVENT_COLS,
        min_rows=100,
        queries=(
            Q("greater_than", 10, _join + "info.name = 'device_busy_gfx'",
              "Less than expected number of captured amd-smi gfx-busy samples!", requires="gfx_activity"),
            Q("greater_than", 10, _join + "info.name = 'device_busy_umc'",
              "Less than expected number of captured amd-smi umc-busy samples!", requires="umc_activity"),
            Q("greater_than", 10, _join + "info.name = 'device_busy_mm'",
              "Less than expected number of captured amd-smi mm-busy samples!", requires="mm_activity"),
            Q("greater_than", 10, _join + "info.name = 'device_memory_usage'",
              "Less than expected number of captured amd-smi memory-usage samples!", requires="mem_usage"),
            Q("greater_than", 10, _join + "info.name LIKE 'device_jpeg_activity_%' and event.value > 0",
              "Less than expected activity in amd-smi jpeg-activity samples!", requires="jpeg_activity"),
            Q("greater_than", 10, _join + "info.name = 'device_gfx_clock'",
              "Less than expected number of captured amd-smi gfx-clock samples!", requires="gfx_clock"),
            Q("greater_than", 10, _join + "info.name = 'device_mem_clock'",
              "Less than expected number of captured amd-smi mem-clock samples!", requires="mem_clock"),
        ),
    )


_MEMORY_COPIES_COLS = ["id", "category", "name", "start", "end"]
_PMC_EVENT_EXT_COLS = ["event_id", "pmc_id", "value", "extdata"]
_TOP_COLS = ["total_calls", "total_duration", "average", "percentage"]


def hpc_matrix_exponential_rules(v: RocpdValidator) -> None:
    """hpc matrix-exponential checks (former hpc/matrix-exponential/sdk-metrics-rules.json)."""
    v.table(
        name="regions",
        columns=_REGIONS_GUID_COLS,
        queries=(
            Q("equals", 171,
              "SELECT COUNT(*) FROM regions WHERE category = 'rocm_marker_api' "
              "AND name LIKE '%rocblas_dgemm%'",
              "Expecting 171 rocblas_dgemm roctx marker regions"),
        ),
    )
    v.table(
        name="kernels",
        columns=_KERNELS_COLUMNS,
        queries=(
            Q("equals", 171, "SELECT COUNT(*) FROM kernels WHERE name LIKE '%Cijk_Ailk_Bljk%'",
              "Expecting 171 rocBLAS GEMM kernel dispatches"),
        ),
    )
    v.table(
        name="regions",
        columns=_REGIONS_GUID_COLS,
        queries=(
            Q("equals", 4,
              "SELECT COUNT(*) FROM regions WHERE category = 'rocm_ompt_api' AND name = 'omp_work'",
              "Expecting 4 omp_work regions"),
        ),
    )


def hpc_split_copy_compute_rules(v: RocpdValidator) -> None:
    """hpc split-copy-compute checks (former hpc/split-copy-compute/sdk-metrics-rules.json)."""
    v.table(
        name="memory_copies",
        columns=_MEMORY_COPIES_COLS,
        queries=(
            Q("equals", 4, "SELECT COUNT(*) FROM memory_copies WHERE name = 'MEMORY_COPY_DEVICE_TO_HOST'",
              "Expecting 4 MEMORY_COPY_DEVICE_TO_HOST memory copies"),
            Q("equals", 4, "SELECT COUNT(*) FROM memory_copies WHERE name = 'MEMORY_COPY_HOST_TO_DEVICE'",
              "Expecting 4 MEMORY_COPY_HOST_TO_DEVICE memory copies"),
        ),
    )
    v.table(
        name="kernels",
        columns=_KERNELS_COLUMNS,
        queries=(
            Q("equals", 4, "SELECT COUNT(*) FROM kernels WHERE name LIKE '%cube%'",
              "Expecting 4 cube kernel dispatches"),
        ),
    )


def openmp_kernel_rules(v: RocpdValidator) -> None:
    """openmp-target kernel checks (former openmp-target/kernel-rules.json)."""
    v.table(
        name="top_kernels",
        columns=_TOP_KERNELS_COLS,
        queries=(
            Q("equals", 0, "SELECT COUNT(*) FROM top_kernels WHERE name IS NULL",
              "Found API calls with null function names"),
            Q("between_inclusive", 3, "SELECT COUNT(*) FROM top_kernels",
              "Expecting between 3 and 5 unique kernels in top_kernels", expected_max=5),
        ),
    )
    v.table(
        name="kernels",
        columns=_KERNELS_COLUMNS,
        min_rows=12,
        queries=(
            Q("equals", 0, "SELECT COUNT(*) as count FROM kernels WHERE name IS NULL",
              "Found kernels with null function names"),
            Q("greater_than_or_equal", 12, "SELECT COUNT(*) as count FROM kernels",
              "Expecting at least 12 kernel dispatches"),
            Q("equals", 0, "SELECT COUNT(*) as count FROM kernels WHERE (end - start) = 0",
              "Kernels with no active execution times found"),
            Q("equals", 4,
              "SELECT COUNT(*) as count FROM kernels WHERE name LIKE '__omp_offloading_%Z4vmulIiEvPT_S1_S1_i_l51.kd'",
              "Unexpected %Z4vmulIiEvPT_S1_S1_i_l51.kd kernel dispatches"),
            Q("equals", 4,
              "SELECT COUNT(*) as count FROM kernels WHERE name LIKE '__omp_offloading_%Z4vmulIfEvPT_S1_S1_i_l51.kd'",
              "Unexpected %Z4vmulIfEvPT_S1_S1_i_l51.kd kernel dispatches"),
            Q("equals", 4,
              "SELECT COUNT(*) as count FROM kernels WHERE name LIKE '__omp_offloading_%Z4vmulIdEvPT_S1_S1_i_l51.kd'",
              "Unexpected %Z4vmulIdEvPT_S1_S1_i_l51.kd kernel dispatches"),
        ),
    )


def openmp_sdk_metrics_rules(v: RocpdValidator) -> None:
    """openmp-target OMPT/HSA checks (former openmp-target/sdk-metrics-rules.json)."""
    v.table(
        name="regions",
        columns=_REGIONS_GUID_COLS,
        queries=(
            Q("greater_than", 100, "SELECT COUNT(*) FROM regions WHERE category = 'rocm_ompt_api';",
              "'rocm_ompt_api' category entries are fewer than expected in regions"),
            Q("equals", 0, "SELECT COUNT(*) FROM regions WHERE name IS NULL;",
              "NULL entries found in the name column of regions"),
        ),
    )
    v.table(
        name="rocpd_string",
        columns=_ROCPD_STRING_COLS,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%rocm_ompt_api%';",
              "'rocm_ompt_api' string not found in the table rocpd_string"),
        ),
    )
    v.table(
        name="regions",
        columns=_REGIONS_GUID_COLS,
        queries=(
            Q("greater_than", 500, "SELECT COUNT(*) FROM regions WHERE category = 'rocm_hsa_api';",
              "'rocm_hsa_api' category entries are fewer than expected in regions"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM regions WHERE category = 'rocm_hsa_api' AND duration = 0;",
              "Found HSA API captures where duration is 0"),
            Q("equals", 0, "SELECT COUNT(*) FROM regions WHERE name IS NULL;",
              "NULL entries found in the name column of regions"),
        ),
    )
    v.table(
        name="rocpd_string",
        columns=_ROCPD_STRING_COLS,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%rocm_hsa_api%';",
              "'rocm_hsa_api' string not found in the table rocpd_string"),
        ),
    )


def python_builtin_rules(v: RocpdValidator) -> None:
    """python builtin-instrumentation checks (former python/python-builtin-rules.json)."""
    v.table(
        name="top",
        columns=_TOP_COLS,
        queries=(
            Q("equals", 5, "SELECT total_calls FROM top WHERE name = '[run][builtin.py:31]';",
              "'[run][builtin.py:31]' does not have 5 total calls, as expected."),
            Q("equals", 5, "SELECT total_calls FROM top WHERE name = '[inefficient][builtin.py:17]';",
              "'[inefficient][builtin.py:17]' does not have 5 total calls, as expected."),
            Q("equals", 885, "SELECT total_calls FROM top WHERE name = '[fib][builtin.py:13]';",
              "'[fib][builtin.py:13]' does not have 885 total calls, as expected."),
            Q("equals", 100, "SELECT ROUND(SUM(percentage), 1) FROM top;",
              "Percentages do not sum to 100."),
        ),
    )
    v.table(
        name="regions",
        columns=_REGIONS_GUID_COLS,
        queries=(
            Q("equals", 5, "SELECT COUNT(*) FROM regions WHERE name = '[run][builtin.py:31]';",
              "'[run][builtin.py:31]' does not have 5 total calls, as expected."),
            Q("equals", 5, "SELECT COUNT(*) FROM regions WHERE name = '[inefficient][builtin.py:17]';",
              "'[inefficient][builtin.py:17]' does not have 5 total calls, as expected."),
            Q("equals", 885, "SELECT COUNT(*) FROM regions WHERE name = '[fib][builtin.py:13]';",
              "'[fib][builtin.py:13]' does not have 885 total calls, as expected."),
        ),
    )


def python_source_rules(v: RocpdValidator) -> None:
    """python source-instrumentation checks (former python/python-source-rules.json)."""
    v.table(
        name="top",
        columns=_TOP_COLS,
        queries=(
            Q("equals", 5, "SELECT total_calls FROM top WHERE name = 'main_loop';",
              "'main_loop' does not have 5 total calls, as expected."),
            Q("equals", 3, "SELECT total_calls FROM top WHERE name = 'run';",
              "'run' does not have 3 total calls, as expected."),
            Q("equals", 3, "SELECT total_calls FROM top WHERE name = 'inefficient';",
              "'inefficient' does not have 3 total calls, as expected."),
            Q("equals", 3, "SELECT total_calls FROM top WHERE name = '_sum';",
              "'_sum' does not have 3 total calls, as expected."),
            Q("equals", 45, "SELECT total_calls FROM top WHERE name = 'fib';",
              "'fib' does not have 45 total calls, as expected."),
            Q("equals", 100, "SELECT ROUND(SUM(percentage), 1) FROM top;",
              "Percentages do not sum to 100."),
        ),
    )
    v.table(
        name="regions",
        columns=_REGIONS_GUID_COLS,
        queries=(
            Q("equals", 5, "SELECT COUNT(*) FROM regions WHERE name = 'main_loop';",
              "'main_loop' does not have 5 total calls, as expected."),
            Q("equals", 3, "SELECT COUNT(*) FROM regions WHERE name = 'run';",
              "'run' does not have 3 total calls, as expected."),
            Q("equals", 3, "SELECT COUNT(*) FROM regions WHERE name = 'inefficient';",
              "'inefficient' does not have 3 total calls, as expected."),
            Q("equals", 3, "SELECT COUNT(*) FROM regions WHERE name = '_sum';",
              "'_sum' does not have 3 total calls, as expected."),
            Q("equals", 45, "SELECT COUNT(*) FROM regions WHERE name = 'fib';",
              "'fib' does not have 45 total calls, as expected."),
        ),
    )


def rccl_comm_rules(v: RocpdValidator) -> None:
    """RCCL comm PMC checks (former rccl/rccl-comm-rules.json)."""
    v.table(
        prefix="rocpd_info_pmc",
        columns=_PMC_INFO_COLS,
        min_rows=1,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM {table_name} WHERE name LIKE 'RCCL Comm%'",
              "Did not find any RCCL Comm PMC descriptions"),
            Q("greater_than", 0,
              "SELECT COUNT(*) as count FROM {table_name} WHERE name LIKE 'RCCL Comm%' AND agent_id IS NOT NULL",
              "Found RCCL PMCs with NULL agent_id"),
        ),
    )
    _join = (
        "SELECT COUNT(*) as count FROM {table_name} event "
        "JOIN rocpd_info_pmc info ON event.pmc_id = info.id WHERE info.name LIKE 'RCCL Comm%'"
    )
    v.table(
        prefix="rocpd_pmc_event",
        columns=_PMC_EVENT_EXT_COLS,
        min_rows=1,
        queries=(
            Q("greater_than", 0, _join, "No RCCL Comm events found"),
            Q("greater_than", 0, _join + " AND event.extdata LIKE '%transfer_bytes%'",
              "RCCL events missing transfer_bytes in extdata"),
            Q("greater_than", 0, _join + " AND event.value > 0",
              "RCCL events have no positive cumulative values"),
        ),
    )


def scratch_memory_rules(v: RocpdValidator) -> None:
    """scratch-memory checks (former scratch-memory/sdk-metrics-rules.json)."""
    v.table(
        name="scratch_memory",
        columns=["id", "guid", "category", "operation", "agent_name", "size", "start", "end"],
        min_rows=1,
        queries=(
            Q("equals", 0,
              "SELECT COUNT(*) FROM scratch_memory WHERE category = 'rocm_scratch_memory' AND id IS NULL;",
              "NULL entries found in the id column of scratch_memory for scratch memory"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM scratch_memory WHERE operation = 'ALLOC' AND (size IS NULL OR size = 0);",
              "Found scratch memory allocations where size is NULL or 0"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM scratch_memory WHERE start IS NULL OR end IS NULL OR start = 0 OR end = 0;",
              "Found scratch memory allocations where start or end is NULL"),
            Q("equals", 0, "SELECT COUNT(*) FROM scratch_memory WHERE start >= end;",
              "Found scratch memory allocations where start is greater than or equal to end"),
        ),
    )
    v.table(
        name="rocpd_string",
        columns=_ROCPD_STRING_COLS,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%rocm_scratch_memory%';",
              "'rocm_scratch_memory' string not found in the table rocpd_string"),
        ),
    )


def _amd_smi_pmc_event_queries(threshold: int) -> tuple[Q, ...]:
    """The common gfx/umc/mm/temp/power/mem/clock amd-smi sample checks."""
    join = (
        "SELECT COUNT(*) as count FROM {table_name} event "
        "JOIN rocpd_info_pmc info ON event.pmc_id = info.id WHERE "
    )
    return (
        Q("greater_than", threshold, join + "info.name = 'device_busy_gfx'",
          "Less than expected number of captured amd-smi gfx-busy samples!", requires="gfx_activity"),
        Q("greater_than", threshold, join + "info.name = 'device_busy_umc'",
          "Less than expected number of captured amd-smi umc-busy samples!", requires="umc_activity"),
        Q("greater_than", threshold, join + "info.name = 'device_busy_mm'",
          "Less than expected number of captured amd-smi mm-busy samples!", requires="mm_activity"),
        Q("greater_than", threshold, join + "info.name = 'device_temp'",
          "Less than expected number of captured amd-smi-temperature samples!", requires="temp"),
        Q("greater_than", threshold, join + "info.name = 'device_power'",
          "Less than expected number of captured amd-smi-power samples!", requires="power"),
        Q("greater_than", threshold, join + "info.name = 'device_memory_usage'",
          "Less than expected number of captured amd-smi-memory-usage samples!", requires="mem_usage"),
        Q("greater_than", threshold, join + "info.name = 'device_gfx_clock'",
          "Less than expected number of captured amd-smi gfx-clock samples!", requires="gfx_clock"),
        Q("greater_than", threshold, join + "info.name = 'device_mem_clock'",
          "Less than expected number of captured amd-smi mem-clock samples!", requires="mem_clock"),
    )


def roctx_validation_rules(v: RocpdValidator) -> None:
    """roctx kernel/thread checks (former roctx/validation-rules.json)."""
    v.table(
        name="top_kernels",
        columns=_TOP_KERNELS_COLS,
        queries=(
            Q("equals", 0, "SELECT COUNT(*) FROM top_kernels WHERE name IS NULL",
              "Found API calls with null function names"),
            Q("greater_than", 0, "SELECT COUNT(*) FROM top_kernels",
              "No kernel calls found in summary"),
            Q("equals", 1, "SELECT COUNT(*) FROM top_kernels WHERE name LIKE 'hipKernelLaunch%'",
              "'hipKernelLaunch' not found in top_kernels"),
            Q("greater_than", 0, "SELECT total_calls FROM top_kernels WHERE name LIKE 'hipKernelLaunch%'",
              "No kernel calls found in summary"),
        ),
    )
    v.table(
        name="kernels",
        columns=_KERNELS_COLUMNS,
        queries=(
            Q("equals", 0, "SELECT COUNT(*) as count FROM kernels WHERE name IS NULL",
              "Found kernels with null function names"),
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM kernels",
              "No kernel entries found"),
            Q("equals", 0, "SELECT COUNT(*) as count FROM kernels WHERE (end - start) = 0",
              "Kernels with no active execution times found"),
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM kernels WHERE name LIKE 'hipKernelLaunch%'",
              "No kernel entries found"),
        ),
    )
    v.table(name="threads", columns=_THREADS_COLS, min_rows=1)


def roctx_sdk_metrics_rules(v: RocpdValidator) -> None:
    """roctx marker-API checks (former roctx/sdk-metrics-rules.json)."""
    v.table(
        name="events_args",
        columns=_EVENTS_ARGS_COLS,
        queries=(
            Q("greater_than", 3, "SELECT COUNT(*) FROM events_args WHERE category = 'rocm_marker_api';",
              "'rocm_marker_api' category entries are fewer than expected in events_args"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM events_args WHERE category IS NULL OR TRIM(category) = '';",
              "Empty or NULL category entries found in events_args"),
        ),
    )
    v.table(
        name="regions",
        columns=_REGIONS_GUID_COLS,
        queries=(
            Q("greater_than_or_equal", 6, "SELECT COUNT(*) FROM regions WHERE category = 'rocm_marker_api';",
              "Less than 6 'rocm_marker_api' entries in the 'regions' table"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM regions WHERE category = 'rocm_marker_api' AND duration = 0;",
              "Found rocTX API captures where duration is 0"),
            Q("equals", 0, "SELECT COUNT(*) FROM regions WHERE name IS NULL;",
              "NULL entries found in the name column of regions"),
            Q("equals", 4,
              "SELECT COUNT(*) FROM regions WHERE category = 'rocm_marker_api' AND name LIKE 'roctxMark_%';",
              "Expected 4 roctxMark marker entries in `regions` table"),
            Q("equals", 3,
              "SELECT COUNT(*) FROM regions WHERE category = 'rocm_marker_api' AND name LIKE 'roctxRangePush_%';",
              "Expected 3 roctxRangePush marker entries in `regions` table"),
            Q("equals", 2,
              "SELECT COUNT(*) FROM regions WHERE category = 'rocm_marker_api' AND name LIKE 'roctxRangeStart_%';",
              "Expected 2 roctxRangeStart marker entries in `regions` table"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM regions WHERE category = 'rocm_marker_api' AND name LIKE 'roctxRangePop%';",
              "Found unexpected roctxRangePop marker entries in `regions` table"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM regions WHERE category = 'rocm_marker_api' AND name LIKE 'roctxRangeStop%';",
              "Found unexpected roctxRangeStop marker entries in `regions` table"),
        ),
    )
    v.table(
        name="rocpd_string",
        columns=_ROCPD_STRING_COLS,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%rocm_marker_api%';",
              "'rocm_marker_api' string not found in the table rocpd_string"),
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%roctx%';",
              "'roctx' string not found in the table rocpd_string"),
        ),
    )


def roctx_amd_smi_rules(v: RocpdValidator) -> None:
    """roctx amd-smi PMC checks (former roctx/amd-smi-rules.json)."""
    v.table(
        prefix="rocpd_info_pmc_",
        columns=_PMC_INFO_COLS,
        min_rows=4,
        queries=(
            Q("greater_than", 4, "SELECT COUNT(*) as count FROM {table_name} WHERE target_arch is 'GPU'",
              "Found none of the amd-smi categories"),
        ),
    )
    v.table(
        prefix="rocpd_pmc_event_",
        columns=_PMC_EVENT_COLS,
        min_rows=20,
        queries=_amd_smi_pmc_event_queries(5),
    )


def video_validation_rules(v: RocpdValidator) -> None:
    """video-decode thread checks (former video-decode/validation-rules.json)."""
    v.table(name="threads", columns=_THREADS_COLS, min_rows=10)


def video_sdk_metrics_rules(v: RocpdValidator) -> None:
    """video-decode rocDecode + HIP API checks (former video-decode/sdk-metrics-rules.json)."""
    v.table(
        name="events_args",
        columns=_EVENTS_ARGS_COLS,
        queries=(
            Q("greater_than", 1500, "SELECT COUNT(*) FROM events_args WHERE category = 'rocm_rocdecode_api';",
              "'rocm_rocdecode_api' category entries are fewer than expected in events_args"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM events_args WHERE category IS NULL OR TRIM(category) = '';",
              "Empty or NULL category entries found in events_args"),
        ),
    )
    v.table(
        name="regions",
        columns=_REGIONS_GUID_COLS,
        queries=(
            Q("greater_than", 500, "SELECT COUNT(*) FROM regions WHERE category = 'rocm_rocdecode_api';",
              "'rocm_rocdecode_api' category entries are fewer than expected in regions"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM regions WHERE category = 'rocm_rocdecode_api' AND duration = 0;",
              "Found rocDecode API captures where duration is 0"),
            Q("equals", 0, "SELECT COUNT(*) FROM regions WHERE name IS NULL;",
              "NULL entries found in the name column of regions"),
        ),
    )
    v.table(
        name="rocpd_string",
        columns=_ROCPD_STRING_COLS,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%rocm_rocdecode_api%';",
              "'rocm_rocdecode_api' string not found in the table rocpd_string"),
        ),
    )
    v.table(
        name="events_args",
        columns=_EVENTS_ARGS_COLS,
        queries=(
            Q("greater_than", 100, "SELECT COUNT(*) FROM events_args WHERE category = 'rocm_hip_api';",
              "'rocm_hip_api' category entries are fewer than expected in events_args"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM events_args WHERE category IS NULL OR TRIM(category) = '';",
              "Empty or NULL category entries found in events_args"),
        ),
    )
    v.table(
        name="regions",
        columns=_REGIONS_GUID_COLS,
        queries=(
            Q("greater_than", 50, "SELECT COUNT(*) FROM regions WHERE category = 'rocm_hip_api';",
              "'rocm_hip_api' category entries are fewer than expected in regions"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM regions WHERE category = 'rocm_hip_api' AND duration = 0;",
              "Found HIP API captures where duration is 0"),
            Q("equals", 0, "SELECT COUNT(*) FROM regions WHERE name IS NULL;",
              "NULL entries found in the name column of regions"),
        ),
    )
    v.table(
        name="rocpd_string",
        columns=_ROCPD_STRING_COLS,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%rocm_hip_api%';",
              "'rocm_hip_api' string not found in the table rocpd_string"),
        ),
    )


def video_amd_smi_rules(v: RocpdValidator) -> None:
    """video-decode amd-smi PMC checks (former video-decode/amd-smi-rules.json)."""
    v.table(
        prefix="rocpd_info_pmc_",
        columns=_PMC_INFO_COLS,
        min_rows=1,
        queries=(
            Q("greater_than", 1, "SELECT COUNT(*) as count FROM {table_name} WHERE symbol LIKE 'device_vcn_activity_%'",
              "Did not find vcn_activity in amd-smi metrics", requires="vcn_activity"),
        ),
    )
    _join = (
        "SELECT COUNT(*) as count FROM {table_name} event "
        "JOIN rocpd_info_pmc info ON event.pmc_id = info.id WHERE "
    )
    v.table(
        prefix="rocpd_pmc_event_",
        columns=_PMC_EVENT_COLS,
        min_rows=100,
        queries=(
            Q("greater_than", 10, _join + "info.name = 'device_busy_gfx'",
              "Less than expected number of captured amd-smi gfx-busy samples!", requires="gfx_activity"),
            Q("greater_than", 10, _join + "info.name = 'device_busy_umc'",
              "Less than expected number of captured amd-smi umc-busy samples!", requires="umc_activity"),
            Q("greater_than", 10, _join + "info.name = 'device_busy_mm'",
              "Less than expected number of captured amd-smi mm-busy samples!", requires="mm_activity"),
            Q("greater_than", 10, _join + "info.name = 'device_memory_usage'",
              "Less than expected number of captured amd-smi memory-usage samples!", requires="mem_usage"),
            Q("greater_than", 10, _join + "info.name LIKE 'device_vcn_activity_%' and event.value > 0",
              "Less than expected activity in amd-smi vcn-activity samples!", requires="vcn_activity"),
            Q("greater_than", 10, _join + "info.name = 'device_gfx_clock'",
              "Less than expected number of captured amd-smi gfx-clock samples!", requires="gfx_clock"),
            Q("greater_than", 10, _join + "info.name = 'device_mem_clock'",
              "Less than expected number of captured amd-smi mem-clock samples!", requires="mem_clock"),
        ),
    )


def kfd_rules(v: RocpdValidator) -> None:
    """KFD page-fault/migrate checks (former kfd/kfd-rules.json)."""
    v.table(
        name="regions",
        columns=["id", "guid", "category", "name", "start", "end"],
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM regions WHERE category = 'rocm_kfd_page_fault';",
              "'rocm_kfd_page_fault' category entries not found in regions"),
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM regions WHERE category = 'rocm_kfd_page_migrate';",
              "'rocm_kfd_page_migrate' category entries not found in regions"),
            Q("greater_than", 0,
              "SELECT COUNT(*) as count FROM regions WHERE category = 'rocm_kfd_event_unmap_from_gpu';",
              "'rocm_kfd_event_unmap_from_gpu' category entries not found in regions"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM regions WHERE category LIKE 'rocm_kfd_%' AND name IS NULL;",
              "NULL entries found in the name column of KFD regions"),
        ),
    )
    v.table(
        name="rocpd_string",
        columns=_ROCPD_STRING_COLS,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM rocpd_string WHERE string = 'rocm_kfd_page_fault';",
              "'rocm_kfd_page_fault' string not found in rocpd_string"),
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM rocpd_string WHERE string = 'rocm_kfd_page_migrate';",
              "'rocm_kfd_page_migrate' string not found in rocpd_string"),
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM rocpd_string WHERE string = 'rocm_kfd_queue';",
              "'rocm_kfd_queue' string not found in rocpd_string"),
            Q("greater_than", 0,
              "SELECT COUNT(*) as count FROM rocpd_string WHERE string = 'rocm_kfd_event_unmap_from_gpu';",
              "'rocm_kfd_event_unmap_from_gpu' string not found in rocpd_string"),
        ),
    )
    v.table(
        name="events_args",
        columns=["event_id", "category", "arg_name", "arg_value"],
        queries=(
            Q("greater_than", 0,
              "SELECT COUNT(*) as count FROM events_args WHERE category = 'rocm_kfd_page_fault' AND arg_name = 'address';",
              "No 'address' argument found for kfd_page_fault events"),
            Q("greater_than", 0,
              "SELECT COUNT(*) as count FROM events_args WHERE category = 'rocm_kfd_page_migrate' AND arg_name = 'src_agent';",
              "No 'src_agent' argument found for kfd_page_migrate events"),
            Q("greater_than", 0,
              "SELECT COUNT(*) as count FROM events_args WHERE category = 'rocm_kfd_page_migrate' AND arg_name = 'dst_agent';",
              "No 'dst_agent' argument found for kfd_page_migrate events"),
            Q("greater_than", 0,
              "SELECT CASE WHEN (SELECT COUNT(*) FROM regions WHERE category = 'rocm_kfd_queue') = 0 "
              "THEN 1 ELSE CASE WHEN (SELECT COUNT(*) FROM events_args WHERE category = 'rocm_kfd_queue' "
              "AND arg_name = 'agent') > 0 THEN 1 ELSE 0 END END as count;",
              "kfd_queue events exist but 'agent' argument is missing in events_args"),
            Q("greater_than", 0,
              "SELECT COUNT(*) as count FROM events_args WHERE category = 'rocm_kfd_event_unmap_from_gpu' AND arg_name = 'start_address';",
              "No 'start_address' argument found for kfd_event_unmap_from_gpu events"),
            Q("greater_than", 0,
              "SELECT COUNT(*) as count FROM events_args WHERE category = 'rocm_kfd_event_unmap_from_gpu' AND arg_name = 'end_address';",
              "No 'end_address' argument found for kfd_event_unmap_from_gpu events"),
        ),
    )
    v.table(
        prefix="rocpd_info_pmc_",
        columns=_PMC_INFO_COLS,
        min_rows=1,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM {table_name} WHERE name = 'rocm_kfd_page_fault'",
              "No PMC descriptor found for 'rocm_kfd_page_fault'"),
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM {table_name} WHERE name = 'rocm_kfd_page_migrate'",
              "No PMC descriptor found for 'rocm_kfd_page_migrate'"),
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM {table_name} WHERE name = 'rocm_kfd_queue'",
              "No PMC descriptor found for 'rocm_kfd_queue'"),
            Q("greater_than", 0,
              "SELECT COUNT(*) as count FROM {table_name} WHERE name = 'rocm_kfd_event_unmap_from_gpu'",
              "No PMC descriptor found for 'rocm_kfd_event_unmap_from_gpu'"),
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM {table_name} WHERE block = 'KFD'",
              "KFD PMC descriptors with block='KFD' not found"),
        ),
    )
    v.table(
        prefix="rocpd_pmc_event_",
        columns=_PMC_EVENT_COLS,
        min_rows=1,
        queries=(
            Q("greater_than", 0,
              "SELECT COUNT(*) as count FROM {table_name} event JOIN rocpd_info_pmc info "
              "ON event.pmc_id = info.id WHERE info.name = 'rocm_kfd_page_fault'",
              "No PMC event samples found for 'rocm_kfd_page_fault'"),
            Q("greater_than", 0,
              "SELECT COUNT(*) as count FROM {table_name} event JOIN rocpd_info_pmc info "
              "ON event.pmc_id = info.id WHERE info.name = 'rocm_kfd_page_migrate'",
              "No PMC event samples found for 'rocm_kfd_page_migrate'"),
            Q("greater_than", 0,
              "SELECT CASE WHEN (SELECT COUNT(*) FROM regions WHERE category = 'rocm_kfd_queue') = 0 "
              "THEN 1 ELSE CASE WHEN (SELECT COUNT(*) FROM {table_name} event JOIN rocpd_info_pmc info "
              "ON event.pmc_id = info.id WHERE info.name = 'rocm_kfd_queue') > 0 THEN 1 ELSE 0 END END as count",
              "kfd_queue events exist but no corresponding PMC event samples found"),
            Q("greater_than", 0,
              "SELECT COUNT(*) as count FROM {table_name} event JOIN rocpd_info_pmc info "
              "ON event.pmc_id = info.id WHERE info.name = 'rocm_kfd_event_unmap_from_gpu'",
              "No PMC event samples found for 'rocm_kfd_event_unmap_from_gpu'"),
        ),
    )


def transpose_validation_rules(v: RocpdValidator) -> None:
    """transpose kernel/thread checks (former transpose/validation-rules.json)."""
    v.table(
        name="top_kernels",
        columns=_TOP_KERNELS_COLS,
        queries=(
            Q("equals", 0, "SELECT COUNT(*) FROM top_kernels WHERE name IS NULL",
              "Found API calls with null function names"),
            Q("greater_than", 0, "SELECT COUNT(*) FROM top_kernels",
              "No kernel calls found in summary"),
            Q("equals", 1, "SELECT COUNT(*) FROM top_kernels WHERE name LIKE 'transpose%'",
              "transpose kernel not found in top_kernels"),
            Q("greater_than_or_equal", 1000, "SELECT total_calls FROM top_kernels WHERE name LIKE 'transpose%'",
              "Fewer than 1000 transpose kernel calls found in summary"),
        ),
    )
    v.table(
        name="kernels",
        columns=_KERNELS_COLUMNS,
        min_rows=1000,
        queries=(
            Q("equals", 0, "SELECT COUNT(*) as count FROM kernels WHERE name IS NULL",
              "Found kernels with null function names"),
            Q("greater_than", 1000, "SELECT COUNT(*) as count FROM kernels",
              "No kernel entries found"),
            Q("equals", 0, "SELECT COUNT(*) as count FROM kernels WHERE (end - start) = 0",
              "Kernels with no active execution times found"),
            Q("greater_than_or_equal", 1000, "SELECT COUNT(*) as count FROM kernels WHERE name LIKE 'transpose%'",
              "Fewer than 1000 transpose kernel entries found"),
        ),
    )
    v.table(name="threads", columns=_THREADS_COLS, min_rows=2)


def transpose_sdk_metrics_rules(v: RocpdValidator) -> None:
    """transpose HIP/HSA/memory checks (former transpose/sdk-metrics-rules.json)."""
    v.table(
        name="events_args", columns=_EVENTS_ARGS_COLS,
        queries=(
            Q("greater_than", 1500, "SELECT COUNT(*) FROM events_args WHERE category = 'rocm_hip_api';",
              "'rocm_hip_api' category entries are fewer than expected in events_args"),
            Q("equals", 0, "SELECT COUNT(*) FROM events_args WHERE category IS NULL OR TRIM(category) = '';",
              "Empty or NULL category entries found in events_args"),
        ),
    )
    v.table(
        name="regions", columns=_REGIONS_GUID_COLS,
        queries=(
            Q("greater_than", 500, "SELECT COUNT(*) FROM regions WHERE category = 'rocm_hip_api';",
              "'rocm_hip_api' category entries are fewer than expected in regions"),
            Q("equals", 0, "SELECT COUNT(*) FROM regions WHERE category = 'rocm_hip_api' AND duration = 0;",
              "Found HIP API captures where duration is 0"),
            Q("equals", 0, "SELECT COUNT(*) FROM regions WHERE name IS NULL;",
              "NULL entries found in the name column of regions"),
        ),
    )
    v.table(
        name="rocpd_string", columns=_ROCPD_STRING_COLS,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%rocm_hip_api%';",
              "'rocm_hip_api' string not found in the table rocpd_string"),
        ),
    )
    v.table(
        name="events_args", columns=_EVENTS_ARGS_COLS,
        queries=(
            Q("greater_than", 1000, "SELECT COUNT(*) FROM events_args WHERE category = 'rocm_hsa_api';",
              "'rocm_hsa_api' category entries are fewer than expected in events_args"),
        ),
    )
    v.table(
        name="regions", columns=_REGIONS_GUID_COLS,
        queries=(
            Q("greater_than", 500, "SELECT COUNT(*) FROM regions WHERE category = 'rocm_hsa_api';",
              "'rocm_hsa_api' category entries are fewer than expected in regions"),
            Q("equals", 0, "SELECT COUNT(*) FROM regions WHERE category = 'rocm_hsa_api' AND duration = 0;",
              "Found HSA API captures where duration is 0"),
            Q("equals", 0, "SELECT COUNT(*) FROM regions WHERE name IS NULL;",
              "NULL entries found in the name column of regions"),
        ),
    )
    v.table(
        name="rocpd_string", columns=_ROCPD_STRING_COLS,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%rocm_hsa_api%';",
              "'rocm_hsa_api' string not found in the table rocpd_string"),
        ),
    )
    v.table(
        name="memory_allocations",
        columns=["id", "guid", "category", "nid", "pid", "tid", "start", "end",
                 "duration", "type", "level", "agent_name"],
        queries=(
            Q("equals", 0, "SELECT COUNT(*) FROM memory_allocations WHERE id IS NULL;",
              "NULL entries found in the id column of memory_allocations"),
            Q("equals", 0, "SELECT COUNT(*) FROM memory_allocations WHERE size IS NULL or 0;",
              "Entries found where allocated size is 0 or NULL"),
            Q("greater_than", 1, "SELECT COUNT(*) FROM memory_allocations WHERE category LIKE '%rocm_memory_allocate%';",
              "'rocm_memory_allocate' string appears fewer than 1 times in category column of memory_allocations"),
        ),
    )
    v.table(
        name="rocpd_string", columns=_ROCPD_STRING_COLS,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%rocm_memory_allocate%';",
              "'rocm_memory_allocate' string not found in the string column of rocpd_string"),
        ),
    )
    v.table(
        name="rocpd_string", columns=_ROCPD_STRING_COLS,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%rocm_memory_copy%';",
              "'rocm_memory_copy' string not found in the string column of rocpd_string"),
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%MEMORY_COPY_DEVICE_TO_HOST%';",
              "'MEMORY_COPY_DEVICE_TO_HOST' string not found in the string column of rocpd_string"),
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%MEMORY_COPY_HOST_TO_DEVICE%';",
              "'MEMORY_COPY_HOST_TO_DEVICE' string not found in the string column of rocpd_string"),
        ),
    )
    v.table(
        name="memory_copies",
        columns=["id", "guid", "category", "nid", "pid", "tid", "start", "end"],
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) FROM memory_copies WHERE category LIKE '%rocm_memory_copy%';",
              "'rocm_memory_copy' string appears fewer than 10 times in category column of memory_copies"),
            Q("equals", 0, "SELECT COUNT(*) FROM memory_copies WHERE id IS NULL;",
              "NULL entries found in the id column of memory_copies"),
            Q("equals", 0, "SELECT COUNT(*) FROM memory_copies WHERE size IS NULL or 0;",
              "NULL entries found where copied size is 0 or non-existing field"),
            Q("equals", 0, "SELECT COUNT(*) FROM memory_copies WHERE size IS NULL or 0;",
              "NULL entries found where copied size is 0 or non-existing field"),
        ),
    )
    v.table(
        name="memory_copies",
        columns=["id", "guid", "dst_agent_abs_index", "src_agent_abs_index"],
        queries=(
            Q("equals", 0,
              "SELECT COUNT(*) FROM memory_copies mc LEFT JOIN rocpd_info_agent ag "
              "ON mc.dst_agent_abs_index = ag.absolute_index AND mc.guid = ag.guid "
              "WHERE mc.dst_agent_abs_index IS NOT NULL AND ag.absolute_index IS NULL;",
              "Found dst_agent_abs_index values in memory_copies that do not exist in rocpd_info_agent table"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM memory_copies mc LEFT JOIN rocpd_info_agent ag "
              "ON mc.src_agent_abs_index = ag.absolute_index AND mc.guid = ag.guid "
              "WHERE mc.src_agent_abs_index IS NOT NULL AND ag.absolute_index IS NULL;",
              "Found src_agent_abs_index values in memory_copies that do not exist in rocpd_info_agent table"),
            Q("equals", 0, "SELECT COUNT(*) FROM memory_copies WHERE dst_agent_abs_index IS NULL;",
              "Found NULL dst_agent_abs_index values in memory_copies"),
            Q("equals", 0, "SELECT COUNT(*) FROM memory_copies WHERE src_agent_abs_index IS NULL;",
              "Found NULL src_agent_abs_index values in memory_copies"),
        ),
    )


def transpose_cpu_metrics_rules(v: RocpdValidator) -> None:
    """transpose CPU PMC integrity checks (former transpose/cpu-metrics-rules.json)."""
    v.table(
        name="pmc_info",
        columns=["id", "guid", "nid", "pid", "agent_abs_index", "is_constant",
                 "is_derived", "name", "description", "block", "expression"],
        queries=(
            Q("equals", 0, "SELECT COUNT(*) FROM rocpd_info_pmc WHERE name IS NULL OR name = ''",
              "Found PMC info entries with NULL/empty name"),
            Q("equals", 0, "SELECT COUNT(*) FROM rocpd_info_agent WHERE absolute_index < 0",
              "Found agents with negative absolute_index"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM rocpd_info_pmc WHERE is_derived = 1 AND (expression IS NULL OR expression = '')",
              "Found derived PMCs without an expression"),
            Q("equals", 0, "SELECT COUNT(*) FROM rocpd_info_pmc WHERE description IS NULL OR description = ''",
              "Found PMC info entries with NULL/empty description"),
        ),
    )
    v.table(
        name="rocpd_pmc_event",
        columns=["id", "guid", "event_id", "pmc_id", "value", "extdata"],
        queries=(
            Q("equals", 0, "SELECT COUNT(*) FROM rocpd_pmc_event WHERE id IS NULL",
              "Found rocpd_pmc_event rows with NULL id"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM rocpd_pmc_event WHERE guid IS NULL AND "
              "(SELECT COUNT(*) FROM rocpd_pmc_event WHERE guid = rocpd_pmc_event.guid) > 3000",
              "Found rocpd_pmc_event rows with NULL guid"),
            Q("equals", 0, "SELECT COUNT(*) FROM rocpd_pmc_event WHERE event_id IS NULL",
              "Found rocpd_pmc_event rows with NULL event_id"),
            Q("equals", 0, "SELECT COUNT(*) FROM rocpd_pmc_event WHERE pmc_id IS NULL",
              "Found rocpd_pmc_event rows with NULL pmc_id"),
        ),
    )
    v.table(
        name="rocpd_sample",
        columns=["id", "guid", "track_id", "timestamp", "event_id"],
        queries=(
            Q("equals", 0, "SELECT COUNT(*) FROM rocpd_sample WHERE id IS NULL",
              "Found rocpd_sample rows with NULL id"),
            Q("equals", 0,
              "SELECT COUNT(*) FROM rocpd_sample WHERE guid IS NULL AND "
              "(SELECT COUNT(*) FROM rocpd_sample WHERE guid = rocpd_sample.guid) > 3000",
              "Found rocpd_sample rows with NULL guid"),
            Q("equals", 0, "SELECT COUNT(*) FROM rocpd_sample WHERE track_id IS NULL",
              "Found rocpd_sample rows with NULL track_id"),
            Q("equals", 0, "SELECT COUNT(*) FROM rocpd_sample WHERE timestamp IS NULL",
              "Found rocpd_sample rows with NULL timestamp"),
            Q("equals", 0, "SELECT COUNT(*) FROM rocpd_sample WHERE event_id IS NULL",
              "Found rocpd_sample rows with NULL event_id"),
        ),
    )


def transpose_gpu_perf_counter_rules(v: RocpdValidator) -> None:
    """transpose GPU perf-counter checks (former transpose/gpu-perf-counter-rules.json)."""
    v.table(
        prefix="rocpd_info_pmc_", columns=_PMC_INFO_COLS, min_rows=1,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM {table_name} WHERE target_arch = 'GPU'",
              "No GPU perf-counter info entries found"),
            Q("greater_than", 0, "SELECT COUNT(*) as count FROM {table_name} WHERE name LIKE 'SQ_WAVES%'",
              "No SQ_WAVES perf-counter descriptor found"),
        ),
    )
    v.table(
        prefix="rocpd_pmc_event_", columns=_PMC_EVENT_COLS, min_rows=100,
        queries=(
            Q("greater_than", 100, "SELECT COUNT(*) as count FROM {table_name}",
              "Fewer than expected GPU perf-counter events"),
            Q("greater_than", 0,
              "SELECT COUNT(*) as count FROM {table_name} event JOIN rocpd_info_pmc info "
              "ON event.pmc_id = info.id WHERE info.name LIKE 'SQ_WAVES%' AND event.value > 0",
              "No SQ_WAVES events with non-zero value"),
        ),
    )


def transpose_hw_counter_rules(v: RocpdValidator) -> None:
    """transpose HW-counter checks (former transpose/hw-counter-rules.json)."""
    v.table(
        prefix="rocpd_info_pmc_",
        columns=["id", "agent_id", "target_arch", "name", "description", "units", "value_type"],
        min_rows=1,
        queries=(
            Q("greater_than", 0,
              "SELECT COUNT(*) FROM {table_name} WHERE target_arch = 'GPU' AND name LIKE 'GPU %'",
              "No GPU HW-counter metadata registered"),
        ),
    )
    _j = ("SELECT COUNT(*) FROM {table_name} event JOIN rocpd_info_pmc info "
          "ON event.pmc_id = info.id WHERE info.name LIKE 'GPU %'")
    v.table(
        prefix="rocpd_pmc_event_", columns=_PMC_EVENT_COLS, min_rows=1,
        queries=(
            Q("greater_than", 0, _j, "No GPU HW-counter events exist"),
            Q("greater_than", 0, _j + " AND event.value = 0.0", "No GPU HW-counter end-of-event zero markers exist"),
            Q("greater_than", 0, _j + " AND event.value > 0.0", "No GPU HW-counter non-zero values exist"),
            Q("equals", 0,
              "SELECT (SELECT COUNT(*) FROM {table_name} event JOIN rocpd_info_pmc info "
              "ON event.pmc_id = info.id WHERE info.name LIKE 'GPU %' AND event.value > 0.0) - "
              "(SELECT COUNT(*) FROM {table_name} event JOIN rocpd_info_pmc info "
              "ON event.pmc_id = info.id WHERE info.name LIKE 'GPU %' AND event.value = 0.0) as count",
              "Non-zero GPU HW counters do not each have a matching zero marker"),
        ),
    )


def transpose_timer_sampling_rules(v: RocpdValidator) -> None:
    """transpose timer-sampling checks (former transpose/timer-sampling-rules.json)."""
    v.table(
        name="rocpd_string", columns=_ROCPD_STRING_COLS,
        queries=(
            Q("greater_than", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string LIKE '%timer_sampling%';",
              "'timer_sampling' string not found in rocpd_string"),
            Q("equals", 0, "SELECT COUNT(*) FROM rocpd_string WHERE string IS NULL OR TRIM(string) = '';",
              "Found NULL/empty strings in rocpd_string"),
        ),
    )
    v.table(
        name="rocpd_sample", columns=["id", "guid", "track_id", "timestamp", "event_id"],
        queries=(
            Q("greater_than", 1000,
              "SELECT COUNT(*) FROM rocpd_sample R INNER JOIN rocpd_event E "
              "ON E.id = R.event_id AND E.guid = R.guid INNER JOIN rocpd_string S "
              "ON S.id = E.category_id AND S.guid = E.guid WHERE S.string = 'timer_sampling';",
              "Fewer than expected timer_sampling samples"),
            Q("equals", 0, "SELECT COUNT(*) FROM rocpd_sample WHERE guid IS NULL OR TRIM(guid) = '';",
              "Found rocpd_sample rows with NULL/empty guid"),
        ),
    )


def transpose_amd_smi_rules(v: RocpdValidator) -> None:
    """transpose amd-smi PMC checks (former transpose/amd-smi-rules.json)."""
    v.table(
        prefix="rocpd_info_pmc_", columns=_PMC_INFO_COLS, min_rows=4,
        queries=(
            Q("greater_than", 4, "SELECT COUNT(*) as count FROM {table_name} WHERE target_arch is 'GPU'",
              "Found none of the amd-smi categories"),
        ),
    )
    v.table(
        prefix="rocpd_pmc_event_", columns=_PMC_EVENT_COLS, min_rows=2000,
        queries=_amd_smi_pmc_event_queries(100),
    )


# Stable name -> rule set, for callers that prefer lookup by key.
RULE_SETS: dict[str, RuleSet] = {
    "default": default_rules,
    "minimal/recursion": recursion_rules,
    "shmem": shmem_rules,
    "ucx": ucx_rules,
    "transpose/validation": transpose_validation_rules,
    "transpose/sdk-metrics": transpose_sdk_metrics_rules,
    "transpose/cpu-metrics": transpose_cpu_metrics_rules,
    "transpose/gpu-perf-counter": transpose_gpu_perf_counter_rules,
    "transpose/hw-counter": transpose_hw_counter_rules,
    "transpose/timer-sampling": transpose_timer_sampling_rules,
    "transpose/amd-smi": transpose_amd_smi_rules,
    "gpu-connect/validation": gpu_connect_validation_rules,
    "gpu-connect/amd-smi": gpu_connect_amd_smi_rules,
    "jpeg-decode/validation": jpeg_validation_rules,
    "jpeg-decode/sdk-metrics": jpeg_sdk_metrics_rules,
    "jpeg-decode/amd-smi": jpeg_amd_smi_rules,
    "hpc/matrix-exponential": hpc_matrix_exponential_rules,
    "hpc/split-copy-compute": hpc_split_copy_compute_rules,
    "openmp-target/kernel": openmp_kernel_rules,
    "openmp-target/sdk-metrics": openmp_sdk_metrics_rules,
    "python/builtin": python_builtin_rules,
    "python/source": python_source_rules,
    "rccl/comm": rccl_comm_rules,
    "scratch-memory/sdk-metrics": scratch_memory_rules,
    "roctx/validation": roctx_validation_rules,
    "roctx/sdk-metrics": roctx_sdk_metrics_rules,
    "roctx/amd-smi": roctx_amd_smi_rules,
    "video-decode/validation": video_validation_rules,
    "video-decode/sdk-metrics": video_sdk_metrics_rules,
    "video-decode/amd-smi": video_amd_smi_rules,
    "kfd": kfd_rules,
}
