# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fluent assertion builder API for the rocprofsys-validator framework.

Provides:
- ``ExpectBuilder``: Abstract base class for all fluent builders (marker only).
- ``PerfettoExpectBuilder``: 16 fluent methods + terminal ``sql()`` for Perfetto traces.
- ``RocpdExpectBuilder``: 12 fluent methods + terminal ``sql()`` for RocPD databases.
- ``TimemoryExpectBuilder``: 6 fluent methods for timemory output.
- ``SoftExpectBuilder``: Context manager that accumulates all failures before raising.
- ``expect()``: Factory — returns the correct builder for the given reader type.
- ``expect_all()``: Factory — returns a ``SoftExpectBuilder`` context manager.

Typical usage::

    # Eager assertion — stops (via __bool__) when any method fails
    assert expect(perfetto_reader).has_track("HIP").with_slice_count("HIP", min=1)

    # Soft assertion — collects ALL failures before raising
    with expect_all(rocpd_reader) as e:
        e.has_valid_schema()
        e.has_hip_api_calls()
        e.has_kernel_dispatches()
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rocprofsys_validator.core import AssertionBase, CheckResult

if TYPE_CHECKING:
    import pandas as pd

    from rocprofsys_validator.readers.causal import CausalReader
    from rocprofsys_validator.readers.perfetto import PerfettoReader
    from rocprofsys_validator.readers.rocpd import RocpdReader
    from rocprofsys_validator.readers.timemory import TimemoryReader
    from rocprofsys_validator.readers.timemory_json import TimemoryJsonReader

# ---------------------------------------------------------------------------
# Abstract marker base
# ---------------------------------------------------------------------------

class ExpectBuilder(AssertionBase):
    """Marker base class for all fluent assertion builders.

    Subclasses implement fluent methods that delegate to Phase 2 readers.
    ``__eq__`` is inherited from ``AssertionBase`` (raises ``TypeError``) — do not
    override in subclasses. Each concrete subclass MUST override ``__bool__`` to
    raise ``AssertionError`` (not ``TypeError``) with ``CheckResult._format_message()``
    output so that ``assert expect(reader)...`` produces pytest FAILED (not ERROR).
    """

    def satisfies(
        self,
        name: str,
        predicate,
        message: str | None = None,
    ) -> "ExpectBuilder":
        """Assert an arbitrary predicate over the reader (generic escape hatch).

        For checks no fluent method covers, without dropping out of the builder
        (and its CheckResult formatting / soft-assert accumulation):

            expect(r).satisfies("at least 3 streams",
                                lambda rdr: len(rdr.execute_sql("...")) >= 3)

        Args:
            name: Short label for the check (appears in the result/validator name).
            predicate: Callable taking the reader and returning a truthy value
                       when the check passes.
            message: Optional custom message; a default is generated otherwise.

        Returns:
            self, for chaining. The predicate raising is treated as a failure.
        """
        try:
            passed = bool(predicate(self._reader))  # type: ignore[attr-defined]
            msg = message or (
                f"predicate {name!r} satisfied"
                if passed
                else f"predicate {name!r} not satisfied"
            )
        except Exception as exc:  # noqa: BLE001 — predicate errors are check failures
            passed = False
            msg = message or f"predicate {name!r} raised {type(exc).__name__}: {exc}"
        self._results.append(  # type: ignore[attr-defined]
            CheckResult(
                passed=passed,
                validator_name=f"satisfies[{name}]",
                message=msg,
            )
        )
        return self

# ---------------------------------------------------------------------------
# PerfettoExpectBuilder
# ---------------------------------------------------------------------------

class PerfettoExpectBuilder(ExpectBuilder):
    """Fluent builder for Perfetto trace assertions.

    Wraps all ``PerfettoReader.assert_*`` methods with user-friendly names.
    Every fluent method returns ``self`` to enable chaining. ``sql()`` is a
    terminal method — it returns a raw ``pd.DataFrame`` and does not return ``self``.
    """

    def __init__(self, reader: "PerfettoReader") -> None:
        self._reader = reader
        self._results: list[CheckResult] = []

    def __bool__(self) -> bool:
        """Evaluate all accumulated results.

        Returns:
            True if all results passed.

        Raises:
            AssertionError: With formatted messages for every failed result.
        """
        failed = [r for r in self._results if not r.passed]
        if failed:
            msgs = "\n".join(r._format_message() for r in failed)
            raise AssertionError(msgs)
        return True

    # ------------------------------------------------------------------
    # Fluent methods — each delegates to reader, appends result, returns self
    # ------------------------------------------------------------------

    def has_track(self, pattern: str, match: str = "exact") -> "PerfettoExpectBuilder":
        """Assert a track with the given name pattern exists."""
        self._results.append(self._reader.assert_track_exists(pattern, match))
        return self

    def has_no_track(
        self, pattern: str, match: str = "exact"
    ) -> "PerfettoExpectBuilder":
        """Assert NO track matching the pattern exists (negative test)."""
        self._results.append(self._reader.assert_track_absent(pattern, match))
        return self

    def has_process_track(
        self, pattern: str, match: str = "substring"
    ) -> "PerfettoExpectBuilder":
        """Assert a process-scoped track exists."""
        self._results.append(self._reader.assert_process_track_exists(pattern, match))
        return self

    def has_thread_track(
        self, pattern: str, match: str = "substring"
    ) -> "PerfettoExpectBuilder":
        """Assert a thread-scoped track exists."""
        self._results.append(self._reader.assert_thread_track_exists(pattern, match))
        return self

    def has_slices(
        self,
        track: str | None = None,
        slice_name: str | None = None,
        category: str | None = None,
    ) -> "PerfettoExpectBuilder":
        """Assert slices exist matching track, category, and/or slice name."""
        self._results.append(
            self._reader.assert_slices_exist(track, slice_name, category)
        )
        return self

    def with_slice_count(
        self,
        track: str | None = None,
        min: int | None = None,
        max: int | None = None,
        category: str | None = None,
    ) -> "PerfettoExpectBuilder":
        """Assert slice count on the given track and/or category is within [min, max]."""
        self._results.append(self._reader.assert_slice_count(track, min, max, category))
        return self

    def with_slice_category_count(
        self,
        category: str,
        min: int | None = None,
        max: int | None = None,
    ) -> "PerfettoExpectBuilder":
        """Assert the number of slices of the given category is within [min, max]."""
        self._results.append(
            self._reader.assert_slice_category_count(category, min, max)
        )
        return self

    def with_max_depth(self, track: str, depth: int) -> "PerfettoExpectBuilder":
        """Assert the maximum nesting depth on the given track does not exceed depth."""
        self._results.append(self._reader.assert_max_nesting_depth(track, depth))
        return self

    def non_overlapping(self, track: str) -> "PerfettoExpectBuilder":
        """Assert that slices on the given track do not overlap."""
        self._results.append(self._reader.assert_non_overlapping_slices(track))
        return self

    def has_counter_track(
        self, pattern: str, match: str = "substring"
    ) -> "PerfettoExpectBuilder":
        """Assert a counter track exists matching the given pattern."""
        self._results.append(self._reader.assert_counter_track_exists(pattern, match))
        return self

    def with_counter_aggregate(
        self,
        track: str,
        metric: str,
        min: float | None = None,
        max: float | None = None,
    ) -> "PerfettoExpectBuilder":
        """Assert a counter aggregate (sum/min/max) is within [min, max]."""
        self._results.append(
            self._reader.assert_counter_aggregate(track, metric, min, max)
        )
        return self

    def counter_monotonic(
        self, track: str, direction: str
    ) -> "PerfettoExpectBuilder":
        """Assert the counter on the given track is monotonically increasing or decreasing."""
        self._results.append(self._reader.assert_counter_monotonic(track, direction))
        return self

    def has_debug_annotations(
        self, track: str, slice: str, keys: list[str]
    ) -> "PerfettoExpectBuilder":
        """Assert debug annotation keys exist on slices matching track and slice patterns."""
        self._results.append(
            self._reader.assert_debug_annotations(track, slice, keys)
        )
        return self

    def has_flow_events(
        self,
        *,
        require: bool = False,
        min_count: int = 1,
        cross_track: bool = False,
    ) -> "PerfettoExpectBuilder":
        """Assert flow events are present.

        Soft probe by default (warns and passes when absent). Pass
        ``require=True`` to fail when fewer than ``min_count`` flow events exist,
        and ``cross_track=True`` to require flows whose endpoints lie on
        different tracks.
        """
        self._results.append(
            self._reader.assert_flow_events_connect_tracks(
                require=require, min_count=min_count, cross_track=cross_track
            )
        )
        return self

    def flow_between(
        self,
        from_track: str,
        to_track: str,
        *,
        match: str = "substring",
        min_count: int = 1,
        directional: bool = True,
    ) -> "PerfettoExpectBuilder":
        """Assert flow events link a source track to a destination track.

        Counts flows whose source slice is on ``from_track`` and destination
        slice is on ``to_track`` (set ``directional=False`` for either way).
        """
        self._results.append(
            self._reader.assert_flow_between_tracks(
                from_track, to_track, match=match,
                min_count=min_count, directional=directional,
            )
        )
        return self

    def slice_order(
        self,
        track: str,
        *steps: "object",
        match: str = "exact",
        track_match: str = "substring",
        depth: int | None = None,
    ) -> "PerfettoExpectBuilder":
        """Assert slices on a track occur in a given chronological order.

        Each step is ``"name"``, ``["name", count]``, or ``...`` / ``ANYTHING``
        (any run of don't-care slices). See
        ``PerfettoReader.assert_slice_order`` for the full contract.
        """
        self._results.append(
            self._reader.assert_slice_order(
                track, *steps, match=match, track_match=track_match, depth=depth
            )
        )
        return self

    # --- temporal / concurrency ---------------------------------------------

    def gpu_utilization(
        self, track: str, min_pct: float, *, track_match: str = "substring",
        depth: "int | None" = 0,
    ) -> "PerfettoExpectBuilder":
        """Assert a track is busy at least ``min_pct``% of its active span."""
        self._results.append(
            self._reader.assert_gpu_utilization(
                track, min_pct, track_match=track_match, depth=depth
            )
        )
        return self

    def max_idle_gap(
        self, track: str, max_ns: int, *, track_match: str = "substring",
        depth: "int | None" = 0,
    ) -> "PerfettoExpectBuilder":
        """Assert no idle gap between consecutive slices exceeds ``max_ns``."""
        self._results.append(
            self._reader.assert_max_idle_gap(
                track, max_ns, track_match=track_match, depth=depth
            )
        )
        return self

    def overlaps(
        self, name_a: str, name_b: str, *, min_overlap_pct: "float | None" = None,
        max_overlap_pct: "float | None" = None, match: str = "substring",
    ) -> "PerfettoExpectBuilder":
        """Assert two slice groups overlap (or don't) within a percentage band."""
        self._results.append(
            self._reader.assert_overlap(
                name_a, name_b, min_overlap_pct=min_overlap_pct,
                max_overlap_pct=max_overlap_pct, match=match,
            )
        )
        return self

    def serial_on_stream(
        self, track: str, *, track_match: str = "substring", depth: "int | None" = 0,
    ) -> "PerfettoExpectBuilder":
        """Assert slices on a track never overlap (strictly serial)."""
        self._results.append(
            self._reader.assert_serial_on_track(
                track, track_match=track_match, depth=depth
            )
        )
        return self

    def flow_latency(
        self, from_track: str, to_track: str, max_ns: int, *,
        pctile: float = 99.0, match: str = "substring",
    ) -> "PerfettoExpectBuilder":
        """Assert flow latency (dest start - source start) stays within ``max_ns``."""
        self._results.append(
            self._reader.assert_flow_latency(
                from_track, to_track, max_ns, pctile=pctile, match=match
            )
        )
        return self

    def iterations(
        self, name: str, *, count: "int | None" = None, max_cv: "float | None" = None,
        no_upward_trend: bool = False, trend_tol: float = 0.05, match: str = "exact",
        track_pattern: "str | None" = None, track_match: str = "substring",
    ) -> "PerfettoExpectBuilder":
        """Assert repeated slices form a stable iteration (count / CV / trend)."""
        self._results.append(
            self._reader.assert_iteration_consistency(
                name, count=count, max_cv=max_cv, no_upward_trend=no_upward_trend,
                trend_tol=trend_tol, match=match, track_pattern=track_pattern,
                track_match=track_match,
            )
        )
        return self

    # --- statistical / distributional ---------------------------------------

    def slice_duration(
        self, name: str, *, p50_range: "tuple[float, float] | None" = None,
        p95_max_ns: "float | None" = None, p99_max_ns: "float | None" = None,
        min_ns: "float | None" = None, max_ns: "float | None" = None,
        match: str = "exact", track_pattern: "str | None" = None,
        track_match: str = "substring",
    ) -> "PerfettoExpectBuilder":
        """Assert percentile bounds on the duration distribution of matching slices."""
        self._results.append(
            self._reader.assert_slice_duration_distribution(
                name, p50_range=p50_range, p95_max_ns=p95_max_ns, p99_max_ns=p99_max_ns,
                min_ns=min_ns, max_ns=max_ns, match=match,
                track_pattern=track_pattern, track_match=track_match,
            )
        )
        return self

    def no_duration_outliers(
        self, name: str, *, sigma: float = 4.0, max_outliers: int = 0,
        match: str = "exact", track_pattern: "str | None" = None,
        track_match: str = "substring",
    ) -> "PerfettoExpectBuilder":
        """Assert at most ``max_outliers`` durations lie beyond ``sigma`` SD of the mean."""
        self._results.append(
            self._reader.assert_no_duration_outliers(
                name, sigma=sigma, max_outliers=max_outliers, match=match,
                track_pattern=track_pattern, track_match=track_match,
            )
        )
        return self

    def counter_in_range(
        self, track: str, *, min_val: "float | None" = None,
        max_val: "float | None" = None, track_match: str = "substring",
    ) -> "PerfettoExpectBuilder":
        """Assert all values of a counter track stay within [min_val, max_val]."""
        self._results.append(
            self._reader.assert_counter_in_range(
                track, min_val=min_val, max_val=max_val, track_match=track_match
            )
        )
        return self

    def counter_rate(
        self, track: str, *, max_per_sec: "float | None" = None,
        min_per_sec: "float | None" = None, track_match: str = "substring",
    ) -> "PerfettoExpectBuilder":
        """Assert the per-second rate of change of a counter stays within bounds."""
        self._results.append(
            self._reader.assert_counter_rate(
                track, max_per_sec=max_per_sec, min_per_sec=min_per_sec,
                track_match=track_match,
            )
        )
        return self

    # --- structural ----------------------------------------------------------

    def call_tree(
        self, parent: str, *, contains: "list[str] | None" = None,
        max_depth: "int | None" = None, no_recursion: bool = False,
        match: str = "exact", track_pattern: "str | None" = None,
        track_match: str = "substring",
    ) -> "PerfettoExpectBuilder":
        """Assert structural properties of a slice subtree rooted at a parent."""
        self._results.append(
            self._reader.assert_call_tree(
                parent, contains=contains, max_depth=max_depth,
                no_recursion=no_recursion, match=match,
                track_pattern=track_pattern, track_match=track_match,
            )
        )
        return self

    def slice_args(
        self, name: str, require: "dict[str, type]", *, match: str = "exact",
    ) -> "PerfettoExpectBuilder":
        """Assert every slice matching ``name`` carries the required arg keys/types."""
        self._results.append(
            self._reader.assert_slice_args(name, require, match=match)
        )
        return self

    # --- anti-patterns -------------------------------------------------------

    def no_anti_patterns(
        self, *, negative_durations: bool = True, duplicate_slices: bool = True,
        orphan_slices: bool = True, giant_slice: bool = True, giant_pct: float = 99.0,
        zero_duration_category: "str | None" = None,
    ) -> "PerfettoExpectBuilder":
        """Assert a curated bundle of timeline anti-patterns is absent."""
        self._results.append(
            self._reader.assert_no_anti_patterns(
                negative_durations=negative_durations, duplicate_slices=duplicate_slices,
                orphan_slices=orphan_slices, giant_slice=giant_slice, giant_pct=giant_pct,
                zero_duration_category=zero_duration_category,
            )
        )
        return self

    def with_sampling_frequency(
        self, track: str, hz: float, tol: float = 20.0
    ) -> "PerfettoExpectBuilder":
        """Assert the sampling frequency on the given track is close to hz (within tol %)."""
        self._results.append(
            self._reader.assert_sampling_frequency(track, hz, tol)
        )
        return self

    def with_slice_sampling_frequency(
        self, track: str, hz: float, tol: float = 20.0
    ) -> "PerfettoExpectBuilder":
        """Assert the slice-derived sampling frequency on the track is close to hz."""
        self._results.append(
            self._reader.assert_slice_sampling_frequency(track, hz, tol)
        )
        return self

    def has_categories(self, cats: list[str]) -> "PerfettoExpectBuilder":
        """Assert the given slice categories are present in the trace."""
        self._results.append(self._reader.assert_categories_present(cats))
        return self

    def has_process_name(
        self, pattern: str, match: str = "substring"
    ) -> "PerfettoExpectBuilder":
        """Assert a process with the given name pattern exists."""
        self._results.append(self._reader.assert_process_name(pattern, match))
        return self

    def sql(self, query: str) -> "pd.DataFrame":
        """Execute raw PerfettoSQL and return the result as a DataFrame.

        Terminal method — returns raw DataFrame. Does not return self; chaining ends here.

        Args:
            query: PerfettoSQL query string.

        Returns:
            pd.DataFrame: Query results.
        """
        return self._reader.execute_sql(query)

# ---------------------------------------------------------------------------
# RocpdExpectBuilder
# ---------------------------------------------------------------------------

class RocpdExpectBuilder(ExpectBuilder):
    """Fluent builder for RocPD SQLite database assertions.

    Wraps all ``RocpdReader.assert_*`` methods with user-friendly names.
    Every fluent method returns ``self`` to enable chaining. ``sql()`` is a
    terminal method — it returns ``list[sqlite3.Row]`` and does not return ``self``.
    """

    def __init__(self, reader: "RocpdReader") -> None:
        self._reader = reader
        self._results: list[CheckResult] = []

    def __bool__(self) -> bool:
        """Evaluate all accumulated results.

        Returns:
            True if all results passed.

        Raises:
            AssertionError: With formatted messages for every failed result.
        """
        failed = [r for r in self._results if not r.passed]
        if failed:
            msgs = "\n".join(r._format_message() for r in failed)
            raise AssertionError(msgs)
        return True

    # ------------------------------------------------------------------
    # Fluent methods
    # ------------------------------------------------------------------

    def has_valid_schema(self) -> "RocpdExpectBuilder":
        """Assert the RocPD schema is valid (all expected views present)."""
        self._results.append(self._reader.assert_schema_valid())
        return self

    def has_columns(self, view: str, cols: list[str]) -> "RocpdExpectBuilder":
        """Assert the given view contains all expected column names."""
        self._results.append(self._reader.assert_columns_exist(view, cols))
        return self

    def has_min_rows(self, view: str, n: int) -> "RocpdExpectBuilder":
        """Assert the given view has at least n rows."""
        self._results.append(self._reader.assert_min_row_count(view, n))
        return self

    def has_no_rows(self, view: str) -> "RocpdExpectBuilder":
        """Assert the given view has zero rows (negative test)."""
        self._results.append(self._reader.assert_no_rows(view))
        return self

    def has_hip_api_calls(
        self, fn: str | None = None, min: int = 1
    ) -> "RocpdExpectBuilder":
        """Assert HIP API calls are present, optionally filtered by function name."""
        self._results.append(self._reader.assert_hip_api_calls_present(fn, min))
        return self

    def has_kernel_dispatches(self, min: int = 1) -> "RocpdExpectBuilder":
        """Assert at least min kernel dispatches are recorded."""
        self._results.append(self._reader.assert_kernel_dispatches_valid(min))
        return self

    def has_samples(
        self, min: int = 1, category: str | None = None
    ) -> "RocpdExpectBuilder":
        """Assert at least min point-samples are recorded (optionally by category)."""
        self._results.append(self._reader.assert_samples_present(min, category))
        return self

    def has_memory_copies(
        self, direction: str | None = None, min: int = 1
    ) -> "RocpdExpectBuilder":
        """Assert memory copies are present, optionally filtered by direction."""
        self._results.append(self._reader.assert_memory_copies_present(direction, min))
        return self

    def has_pmc_events(self, min: int = 1) -> "RocpdExpectBuilder":
        """Assert at least min PMC (hardware counter) events are present."""
        self._results.append(self._reader.assert_pmc_events_present(min))
        return self

    def has_agent_info(
        self, agent_type: str = "GPU", vendor: str = "AMD"
    ) -> "RocpdExpectBuilder":
        """Assert agent info matches the expected device type and vendor."""
        self._results.append(self._reader.assert_agent_info(agent_type, vendor))
        return self

    def has_gpu_counter(
        self, name: str, profile: object = None
    ) -> "RocpdExpectBuilder":
        """Assert a GPU counter with the given name is present (conditional on GPU profile)."""
        self._results.append(self._reader.assert_gpu_conditional(name, profile))
        return self

    def has_region_args(
        self, region: str, arg: str, val: str | None = None
    ) -> "RocpdExpectBuilder":
        """Assert a region has the given argument, optionally with an expected value."""
        self._results.append(self._reader.assert_region_args(region, arg, val))
        return self

    def has_schema_version(self, min: int) -> "RocpdExpectBuilder":
        """Assert the RocPD schema version is at least min."""
        self._results.append(self._reader.assert_schema_version(min))
        return self

    # --- timeline / statistical / structural mirror -------------------------

    def gpu_utilization(
        self, min_pct: float, *, view: str = "kernels", name: "str | None" = None,
        match: str = "exact", category: "str | None" = None, stream: "str | None" = None,
    ) -> "RocpdExpectBuilder":
        """Assert records in a view cover at least ``min_pct``% of their active span."""
        self._results.append(self._reader.assert_gpu_utilization(
            min_pct, view=view, name=name, match=match, category=category, stream=stream))
        return self

    def max_idle_gap(
        self, max_ns: int, *, view: str = "kernels", name: "str | None" = None,
        match: str = "exact", category: "str | None" = None, stream: "str | None" = None,
    ) -> "RocpdExpectBuilder":
        """Assert no idle gap between consecutive records exceeds ``max_ns``."""
        self._results.append(self._reader.assert_max_idle_gap(
            max_ns, view=view, name=name, match=match, category=category, stream=stream))
        return self

    def overlaps(
        self, name_a: str, name_b: str, *, view: str = "kernels", view_b: "str | None" = None,
        min_overlap_pct: "float | None" = None, max_overlap_pct: "float | None" = None,
        match: str = "exact",
    ) -> "RocpdExpectBuilder":
        """Assert two record groups overlap (or don't) within a percentage band."""
        self._results.append(self._reader.assert_overlap(
            name_a, name_b, view=view, view_b=view_b, min_overlap_pct=min_overlap_pct,
            max_overlap_pct=max_overlap_pct, match=match))
        return self

    def serial_on_stream(self, stream: str, *, view: str = "kernels") -> "RocpdExpectBuilder":
        """Assert records on a stream never overlap (strictly serial)."""
        self._results.append(self._reader.assert_serial_on_stream(stream, view=view))
        return self

    def flow_latency(
        self, max_ns: int, *, from_category: str = "rocm_hip_api", pctile: float = 99.0,
    ) -> "RocpdExpectBuilder":
        """Assert API-call → kernel-dispatch latency (via corr_id) stays within ``max_ns``."""
        self._results.append(self._reader.assert_flow_latency(
            max_ns, from_category=from_category, pctile=pctile))
        return self

    def iterations(
        self, name: str, *, view: str = "kernels", count: "int | None" = None,
        max_cv: "float | None" = None, no_upward_trend: bool = False,
        trend_tol: float = 0.05, match: str = "exact",
    ) -> "RocpdExpectBuilder":
        """Assert repeated records form a stable iteration (count / CV / trend)."""
        self._results.append(self._reader.assert_iteration_consistency(
            name, view=view, count=count, max_cv=max_cv, no_upward_trend=no_upward_trend,
            trend_tol=trend_tol, match=match))
        return self

    def duration_distribution(
        self, name: str, *, view: str = "kernels",
        p50_range: "tuple[float, float] | None" = None, p95_max_ns: "float | None" = None,
        p99_max_ns: "float | None" = None, min_ns: "float | None" = None,
        max_ns: "float | None" = None, match: str = "exact",
    ) -> "RocpdExpectBuilder":
        """Assert percentile bounds on the duration distribution of matching records."""
        self._results.append(self._reader.assert_duration_distribution(
            name, view=view, p50_range=p50_range, p95_max_ns=p95_max_ns,
            p99_max_ns=p99_max_ns, min_ns=min_ns, max_ns=max_ns, match=match))
        return self

    def no_duration_outliers(
        self, name: str, *, view: str = "kernels", sigma: float = 4.0,
        max_outliers: int = 0, match: str = "exact",
    ) -> "RocpdExpectBuilder":
        """Assert at most ``max_outliers`` durations lie beyond ``sigma`` SD of the mean."""
        self._results.append(self._reader.assert_no_duration_outliers(
            name, view=view, sigma=sigma, max_outliers=max_outliers, match=match))
        return self

    def counter_in_range(
        self, counter_name: str, *, min_val: "float | None" = None,
        max_val: "float | None" = None,
    ) -> "RocpdExpectBuilder":
        """Assert all values of a PMC counter stay within [min_val, max_val]."""
        self._results.append(self._reader.assert_counter_in_range(
            counter_name, min_val=min_val, max_val=max_val))
        return self

    def counter_rate(
        self, counter_name: str, *, max_per_sec: "float | None" = None,
        min_per_sec: "float | None" = None,
    ) -> "RocpdExpectBuilder":
        """Assert the per-second rate of change of a PMC counter stays within bounds."""
        self._results.append(self._reader.assert_counter_rate(
            counter_name, max_per_sec=max_per_sec, min_per_sec=min_per_sec))
        return self

    def record_order(
        self, view: str, *steps, match: str = "exact",
        category: "str | None" = None, stream: "str | None" = None,
    ) -> "RocpdExpectBuilder":
        """Assert records in a view occur in a given order (mirror of slice_order)."""
        self._results.append(self._reader.assert_record_order(
            view, *steps, match=match, category=category, stream=stream))
        return self

    def call_tree(
        self, parent: str, *, contains: "list[str] | None" = None,
        max_depth: "int | None" = None, no_recursion: bool = False, match: str = "exact",
    ) -> "RocpdExpectBuilder":
        """Assert structural properties of the region subtree rooted at a parent."""
        self._results.append(self._reader.assert_call_tree(
            parent, contains=contains, max_depth=max_depth,
            no_recursion=no_recursion, match=match))
        return self

    def region_args_present(
        self, region: str, require: "list[str]", *, types: "dict[str, str] | None" = None,
    ) -> "RocpdExpectBuilder":
        """Assert every instance of a region carries the required arg names."""
        self._results.append(self._reader.assert_region_args_present(
            region, require, types=types))
        return self

    def no_anti_patterns(
        self, *, negative_durations: bool = True, duplicate_records: bool = True,
        zero_duration: bool = False, giant_record: bool = True, giant_pct: float = 99.0,
        views: "tuple[str, ...]" = ("regions", "kernels"),
    ) -> "RocpdExpectBuilder":
        """Assert a curated bundle of timeline anti-patterns is absent across views."""
        self._results.append(self._reader.assert_no_anti_patterns(
            negative_durations=negative_durations, duplicate_records=duplicate_records,
            zero_duration=zero_duration, giant_record=giant_record, giant_pct=giant_pct,
            views=views))
        return self

    def sql(self, query: str) -> list:
        """Execute raw SQL and return the result as a list of sqlite3.Row objects.

        Terminal method — returns list[sqlite3.Row]. Does not return self; chaining ends here.

        Args:
            query: SQL query string.

        Returns:
            list[sqlite3.Row]: Query results.
        """
        return self._reader.execute_sql(query)

# ---------------------------------------------------------------------------
# TimemoryExpectBuilder
# ---------------------------------------------------------------------------

class TimemoryExpectBuilder(ExpectBuilder):
    """Fluent builder for timemory output assertions.

    Wraps all ``TimemoryReader.assert_*`` methods with user-friendly names.
    Every fluent method returns ``self`` to enable chaining.
    """

    def __init__(self, reader: "TimemoryReader") -> None:
        self._reader = reader
        self._results: list[CheckResult] = []

    def __bool__(self) -> bool:
        """Evaluate all accumulated results.

        Returns:
            True if all results passed.

        Raises:
            AssertionError: With formatted messages for every failed result.
        """
        failed = [r for r in self._results if not r.passed]
        if failed:
            msgs = "\n".join(r._format_message() for r in failed)
            raise AssertionError(msgs)
        return True

    # ------------------------------------------------------------------
    # Fluent methods
    # ------------------------------------------------------------------

    def has_files(
        self, stems: list[str] | None = None
    ) -> "TimemoryExpectBuilder":
        """Assert expected timemory output files are present."""
        self._results.append(self._reader.assert_files_present(stems))
        return self

    def has_label(
        self, stem: str, pattern: str, match: str = "auto"
    ) -> "TimemoryExpectBuilder":
        """Assert a label matching pattern exists in the given stem file."""
        self._results.append(self._reader.assert_label_exists(stem, pattern, match))
        return self

    def without_label(
        self, stem: str, pattern: str, match: str = "auto"
    ) -> "TimemoryExpectBuilder":
        """Assert NO label matching pattern exists in the given stem file (negative test)."""
        self._results.append(self._reader.assert_label_absent(stem, pattern, match))
        return self

    def with_count_gte(
        self, stem: str, pattern: str, n: int
    ) -> "TimemoryExpectBuilder":
        """Assert label count in the given stem file is at least n."""
        self._results.append(self._reader.assert_count(stem, pattern, n))
        return self

    def with_metric_range(
        self,
        stem: str,
        label: str,
        col: str,
        min: float | None = None,
        max: float | None = None,
        tol: float = 0.0,
    ) -> "TimemoryExpectBuilder":
        """Assert a metric value is within [min, max] (with optional tolerance %)."""
        self._results.append(
            self._reader.assert_metric_range(stem, label, col, min, max, tol)
        )
        return self

    def with_cpu_sampling(
        self,
        label: str,
        col: str = "SUM",
        min: float | None = None,
        max: float | None = None,
    ) -> "TimemoryExpectBuilder":
        """Assert CPU sampling data for the given label is within [min, max]."""
        self._results.append(
            self._reader.assert_cpu_sampling(label, col, min, max)
        )
        return self

    def with_pct_self(
        self,
        stem: str,
        label: str,
        min: float | None = None,
        max: float | None = None,
    ) -> "TimemoryExpectBuilder":
        """Assert the % self metric for the given label is within [min, max]."""
        self._results.append(self._reader.assert_pct_self(stem, label, min, max))
        return self

    # --- structural / aggregate-stat mirror ---------------------------------

    def call_tree(
        self, stem: str, parent: str, *, contains: "list[str] | None" = None,
        max_depth: "int | None" = None, no_recursion: bool = False, match: str = "auto",
    ) -> "TimemoryExpectBuilder":
        """Assert structural properties of the call subtree rooted at a label."""
        self._results.append(self._reader.assert_call_tree(
            stem, parent, contains=contains, max_depth=max_depth,
            no_recursion=no_recursion, match=match))
        return self

    def cv(self, stem: str, label: str, max_cv: float, *, match: str = "auto") -> "TimemoryExpectBuilder":
        """Assert the coefficient of variation (STDDEV/MEAN) is within ``max_cv``."""
        self._results.append(self._reader.assert_cv(stem, label, max_cv, match=match))
        return self

    def iterations(
        self, stem: str, label: str, *, count: "int | None" = None,
        max_cv: "float | None" = None, match: str = "auto",
    ) -> "TimemoryExpectBuilder":
        """Assert a label's invocation count and/or CV (no trend)."""
        self._results.append(self._reader.assert_iteration_consistency(
            stem, label, count=count, max_cv=max_cv, match=match))
        return self

    def no_anti_patterns(
        self, stem: str, *, negative_metrics: bool = True,
        metric_columns: "tuple[str, ...]" = ("SUM", "MEAN", "MIN", "MAX"),
        zero_count: bool = True,
    ) -> "TimemoryExpectBuilder":
        """Assert a label table has no negative metrics or zero-count labels."""
        self._results.append(self._reader.assert_no_anti_patterns(
            stem, negative_metrics=negative_metrics, metric_columns=metric_columns,
            zero_count=zero_count))
        return self

# ---------------------------------------------------------------------------
# CausalExpectBuilder
# ---------------------------------------------------------------------------

class CausalExpectBuilder(ExpectBuilder):
    """Fluent builder for CausalReader assertions.

    Wraps all ``CausalReader.assert_*`` methods with user-friendly names.
    Every fluent method returns ``self`` to enable chaining.
    """

    def __init__(self, reader: "CausalReader") -> None:
        self._reader = reader
        self._results: list[CheckResult] = []

    def __bool__(self) -> bool:
        """Evaluate all accumulated results.

        Returns:
            True if all results passed.

        Raises:
            AssertionError: With formatted messages for every failed result.
        """
        failed = [r for r in self._results if not r.passed]
        if failed:
            msgs = "\n".join(r._format_message() for r in failed)
            raise AssertionError(msgs)
        return True

    # ------------------------------------------------------------------
    # Fluent methods
    # ------------------------------------------------------------------

    def has_experiments(self) -> "CausalExpectBuilder":
        """Assert that at least one causal experiment record was loaded."""
        self._results.append(self._reader.assert_has_experiments())
        return self

    def has_progress_points(self, pattern: str = ".*") -> "CausalExpectBuilder":
        """Assert that progress points matching the given regex pattern exist."""
        self._results.append(self._reader.assert_has_progress_points(pattern))
        return self

    def has_min_data_points(self, n: int) -> "CausalExpectBuilder":
        """Assert that at least n total observations exist across all data points."""
        self._results.append(self._reader.assert_min_data_points(n))
        return self

    def has_speedup(
        self,
        experiment: str,
        progress_point: str,
        virtual_speedup: int,
        expected: float,
        tolerance: float,
        ci_mode: bool = False,
    ) -> "CausalExpectBuilder":
        """Assert that the measured speedup for an experiment/progress-point matches expected."""
        self._results.append(
            self._reader.assert_speedup(
                experiment, progress_point, virtual_speedup, expected, tolerance, ci_mode
            )
        )
        return self

# ---------------------------------------------------------------------------
# TimemoryJsonExpectBuilder
# ---------------------------------------------------------------------------

class TimemoryJsonExpectBuilder(ExpectBuilder):
    """Fluent builder for TimemoryJsonReader assertions. No has_files method — JSON reader is single-file.

    Wraps all ``TimemoryJsonReader.assert_*`` methods with user-friendly names.
    Every fluent method returns ``self`` to enable chaining.
    """

    def __init__(self, reader: "TimemoryJsonReader") -> None:
        self._reader = reader
        self._results: list[CheckResult] = []

    def __bool__(self) -> bool:
        """Evaluate all accumulated results.

        Returns:
            True if all results passed.

        Raises:
            AssertionError: With formatted messages for every failed result.
        """
        failed = [r for r in self._results if not r.passed]
        if failed:
            msgs = "\n".join(r._format_message() for r in failed)
            raise AssertionError(msgs)
        return True

    # ------------------------------------------------------------------
    # Fluent methods
    # ------------------------------------------------------------------

    def has_label(
        self, stem: str, pattern: str, match: str = "auto"
    ) -> "TimemoryJsonExpectBuilder":
        """Assert a label matching pattern exists in the given stem metric."""
        self._results.append(self._reader.assert_label_exists(stem, pattern, match))
        return self

    def with_count_gte(
        self, stem: str, pattern: str, n: int
    ) -> "TimemoryJsonExpectBuilder":
        """Assert label laps (count) in the given stem metric is at least n."""
        self._results.append(self._reader.assert_count(stem, pattern, n))
        return self

    def with_metric_range(
        self,
        stem: str,
        label: str,
        col: str,
        min: float | None = None,
        max: float | None = None,
        tol: float = 0.0,
    ) -> "TimemoryJsonExpectBuilder":
        """Assert a metric value is within [min, max] (with optional tolerance %)."""
        self._results.append(
            self._reader.assert_metric_range(stem, label, col, min, max, tol)
        )
        return self

    def with_pct_self(
        self,
        stem: str,
        label: str,
        min: float | None = None,
        max: float | None = None,
    ) -> "TimemoryJsonExpectBuilder":
        """Assert the % self metric for the given label is within [min, max]."""
        self._results.append(self._reader.assert_pct_self(stem, label, min, max))
        return self

    def without_label(
        self, stem: str, pattern: str, match: str = "auto"
    ) -> "TimemoryJsonExpectBuilder":
        """Assert no label matching pattern exists in the given stem metric.

        Replaces fail_regex semantics from the old assert_timemory fixture.
        Passes when no node label matches the given pattern; fails when a match
        is found.
        """
        self._results.append(self._reader.assert_label_absent(stem, pattern, match))
        return self

    def with_depth(
        self, stem: str, label: str, depth: int
    ) -> "TimemoryJsonExpectBuilder":
        """Assert the first matched label is at the expected graph depth (0 = root).

        For labels appearing multiple times at different depths, only the first
        matched node's depth is checked.
        """
        self._results.append(self._reader.assert_depth(stem, label, depth))
        return self

    # --- structural / aggregate-stat mirror ---------------------------------

    def call_tree(
        self, stem: str, parent: str, *, contains: "list[str] | None" = None,
        max_depth: "int | None" = None, no_recursion: bool = False, match: str = "auto",
    ) -> "TimemoryJsonExpectBuilder":
        """Assert structural properties of the call subtree rooted at a label."""
        self._results.append(self._reader.assert_call_tree(
            stem, parent, contains=contains, max_depth=max_depth,
            no_recursion=no_recursion, match=match))
        return self

    def cv(self, stem: str, label: str, max_cv: float, *, match: str = "auto") -> "TimemoryJsonExpectBuilder":
        """Assert the coefficient of variation (stddev/mean) is within ``max_cv``."""
        self._results.append(self._reader.assert_cv(stem, label, max_cv, match=match))
        return self

    def iterations(
        self, stem: str, label: str, *, count: "int | None" = None,
        max_cv: "float | None" = None, match: str = "auto",
    ) -> "TimemoryJsonExpectBuilder":
        """Assert a label's lap count and/or CV (no trend)."""
        self._results.append(self._reader.assert_iteration_consistency(
            stem, label, count=count, max_cv=max_cv, match=match))
        return self

    def no_anti_patterns(
        self, stem: str, *, negative_value: bool = True, value_key: str = "value",
        zero_laps: bool = True,
    ) -> "TimemoryJsonExpectBuilder":
        """Assert no graph node has a negative metric value or zero laps."""
        self._results.append(self._reader.assert_no_anti_patterns(
            stem, negative_value=negative_value, value_key=value_key, zero_laps=zero_laps))
        return self

# ---------------------------------------------------------------------------
# ExtensionExpectBuilder — generic builder for registered third-party readers
# ---------------------------------------------------------------------------

class ExtensionExpectBuilder(ExpectBuilder):
    """Generic fluent builder for third-party FormatReader extensions.

    Used by expect() when the reader is not one of the 3 built-in types but
    is a FormatReader subclass registered via @register_validator. Delegates
    to reader.validate() and surfaces failures through __bool__.
    """

    def __init__(self, reader: object) -> None:
        from rocprofsys_validator.core import FormatReader

        if not isinstance(reader, FormatReader):
            raise TypeError(f"ExtensionExpectBuilder requires a FormatReader, got {type(reader).__name__}")
        self._reader = reader
        self._results: list = []

    def validate(self) -> list:
        """Run all validations and return results."""
        return self._reader.validate()

    def __bool__(self) -> bool:
        results = self.validate()
        failed = [r for r in results if not r.passed]
        if failed:
            from rocprofsys_validator.core import CheckResult  # noqa: F401
            msg = "\n".join(r._format_message() for r in failed)
            raise AssertionError(msg)
        return True

    def __eq__(self, other: object) -> bool:
        raise TypeError(
            "Use .validate() to evaluate assertions — "
            "don't compare or bool() the builder"
        )

# ---------------------------------------------------------------------------
# SoftExpectBuilder — context manager for soft assertions
# ---------------------------------------------------------------------------

class SoftExpectBuilder:
    """Context manager that accumulates all validation failures before raising.

    Usage::

        with expect_all(reader) as e:
            e.has_track("HIP")          # failure collected, execution continues
            e.has_categories(["host"])  # also collected
        # AssertionError raised here with ALL failures combined

    Non-``AssertionError`` exceptions raised inside the block propagate unchanged
    (``__exit__`` returns ``False``) so programming errors are not swallowed.

    Implemented as a thin proxy over the eager builder for the reader (see
    ``_make_builder``). Fluent calls are forwarded to that builder, so there is a
    single definition of every assertion method; any method added to the eager
    builders — or contributed by a registered extension builder — is available
    here automatically, with no duplicated method table to keep in sync.
    """

    def __init__(self, reader: object) -> None:
        self._reader = reader
        self._builder = _make_builder(reader)

    def __enter__(self) -> "SoftExpectBuilder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        # Non-assertion exceptions are bugs in test code — let them propagate unchanged
        if exc_type is not None and not issubclass(exc_type, AssertionError):
            return False

        failed = [r for r in self._builder._results if not r.passed]
        if failed:
            count = len(failed)
            msgs = "\n".join(r._format_message() for r in failed)
            parts = [f"Soft assertion: {count} failure(s) collected:\n{msgs}"]
            if exc_type is AssertionError and exc_val is not None:
                parts.append(f"\nIn-block AssertionError: {exc_val}")
            raise AssertionError("\n".join(parts)) from exc_val

        # In-block AssertionError but no accumulated failures: let it propagate.
        if exc_type is AssertionError:
            return False

        return True

    @property
    def _results(self) -> list:
        """Accumulated results live on the wrapped eager builder."""
        return self._builder._results

    def __getattr__(self, name: str):
        # Reached only for names not found normally (the fluent methods). Forward
        # them to the wrapped builder, re-wrapping a fluent return as self so the
        # chain stays on the soft proxy; terminals (e.g. sql()) pass through.
        if name in ("_builder", "_reader"):
            raise AttributeError(name)
        attr = getattr(self._builder, name)
        if callable(attr):
            def _wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                return self if result is self._builder else result

            return _wrapper
        return attr

# ---------------------------------------------------------------------------
# Builder registry + dispatch
# ---------------------------------------------------------------------------

# Maps a reader class to the ExpectBuilder subclass that should serve it. Unlike
# the built-in isinstance ladder, this registry IS consulted at dispatch time, so
# third-party readers can register a fluent builder and get first-class treatment
# (chainable methods + soft-assert support) instead of the validate()-only fallback.
_BUILDER_REGISTRY: dict[type, type] = {}

def register_builder(reader_cls: type, builder_cls: type) -> None:
    """Register a fluent ExpectBuilder for an extension reader class.

    After registering, ``expect(reader)`` and ``expect_all(reader)`` return the
    given builder for instances of ``reader_cls`` (or its subclasses), so the
    extension's chainable methods work through both the eager and soft APIs.

    Args:
        reader_cls: The FormatReader subclass the builder serves.
        builder_cls: An ExpectBuilder subclass taking the reader in its __init__.
    """
    _BUILDER_REGISTRY[reader_cls] = builder_cls

def _make_builder(reader: object) -> "ExpectBuilder":
    """Return the appropriate ExpectBuilder for a reader (shared by expect/expect_all).

    Built-in readers use the dedicated builders; registered extension readers use
    their registered builder (matched along the MRO); any other FormatReader falls
    back to the validate()-only ExtensionExpectBuilder.
    """
    from rocprofsys_validator.core import FormatReader
    from rocprofsys_validator.readers.causal import CausalReader
    from rocprofsys_validator.readers.perfetto import PerfettoReader
    from rocprofsys_validator.readers.rocpd import RocpdReader
    from rocprofsys_validator.readers.timemory import TimemoryReader
    from rocprofsys_validator.readers.timemory_json import TimemoryJsonReader

    if not isinstance(reader, FormatReader):
        raise TypeError(
            f"expect() requires a FormatReader subclass, "
            f"got {type(reader).__name__}"
        )
    from rocprofsys_validator.registry import discover_validators

    discover_validators(_stacklevel=5)
    if isinstance(reader, PerfettoReader):
        return PerfettoExpectBuilder(reader)
    elif isinstance(reader, RocpdReader):
        return RocpdExpectBuilder(reader)
    elif isinstance(reader, TimemoryReader):
        return TimemoryExpectBuilder(reader)
    elif isinstance(reader, CausalReader):
        return CausalExpectBuilder(reader)
    elif isinstance(reader, TimemoryJsonReader):
        return TimemoryJsonExpectBuilder(reader)
    # Registered extension builders (match along the MRO so subclasses work too).
    for cls in type(reader).__mro__:
        if cls in _BUILDER_REGISTRY:
            return _BUILDER_REGISTRY[cls](reader)
    return ExtensionExpectBuilder(reader)

# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def expect(reader: object) -> ExpectBuilder:
    """Return the appropriate ExpectBuilder for the given reader type.

    Built-in readers get their dedicated fluent builder; extension readers get a
    builder registered via ``register_builder`` (or the validate()-only fallback).

    Args:
        reader: A pre-constructed reader instance (built-in or extension).

    Returns:
        The correct ``ExpectBuilder`` for the reader.

    Raises:
        TypeError: If reader is not a FormatReader subclass.

    Example::

        assert expect(perfetto_reader).has_track("HIP").with_slice_count("HIP", min=1)
    """
    return _make_builder(reader)

def expect_all(reader: object) -> SoftExpectBuilder:
    """Return a SoftExpectBuilder context manager for the given reader.

    Lazy-imports reader classes inside the function body to avoid circular imports.

    Args:
        reader: A pre-constructed ``PerfettoReader``, ``RocpdReader``, or
                ``TimemoryReader`` instance.

    Returns:
        A ``SoftExpectBuilder`` context manager.

    Raises:
        TypeError: If reader is not one of the three supported reader types.

    Example::

        with expect_all(rocpd_reader) as e:
            e.has_valid_schema()
            e.has_hip_api_calls()
            e.has_kernel_dispatches()
    """
    from rocprofsys_validator.core import FormatReader

    # Type check: must be a FormatReader subclass (built-in or registered extension)
    if not isinstance(reader, FormatReader):
        raise TypeError(
            f"expect_all() requires a FormatReader subclass, "
            f"got {type(reader).__name__}"
        )
    from rocprofsys_validator.registry import discover_validators

    discover_validators(_stacklevel=4)
    return SoftExpectBuilder(reader)
