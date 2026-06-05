# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CausalReader — FormatReader for rocprof-sys causal profiling JSON output.

Design decisions:
- Lazy loading: _ensure_loaded() is called on the first assert_* invocation.
  Construction is always O(1) — no I/O in __init__.
- Multi-file merging: process_data() is called once per path, accumulating
  all experiment records into a single _raw_data dict.
- Delegation: all speedup math is delegated to _causal_validator helpers
  (process_data, compute_speedups, validation). CausalReader only handles
  the CheckResult protocol and pattern matching.
- Security (T-01-01): json.loads() errors in _ensure_loaded() are caught and
  stored as CheckResult(passed=False) — never raised to the caller.
- Security (T-01-02): Paths come from trusted test fixtures; no sandboxing
  needed — documented here per the threat register.
- min_experiments default=1: allows single-experiment test fixtures to work
  without setting this parameter explicitly.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

from rocprofsys_validator.core import CheckResult, FormatReader
from rocprofsys_validator.readers import _causal_validator as _cv
from rocprofsys_validator.registry import reader

# ---------------------------------------------------------------------------
# CausalReader
# ---------------------------------------------------------------------------

@reader("causal")
class CausalReader(FormatReader):
    """Format reader and validator for causal profiling JSON output.

    Loads one or more causal JSON files produced by rocprof-sys, merges their
    experiment records using the process_data() helper from causal_validator.py,
    and exposes four assert_* methods for programmatic validation.

    Usage::

        with CausalReader(["causal/experiments.json"]) as r:
            r.assert_has_experiments()
            r.assert_has_progress_points("my_progress_point")
            r.assert_min_data_points(5)
            r.assert_speedup("my_func.cpp:42", "my_progress", 10, expected=15.0, tolerance=30.0)
            results = r.validate()
            assert all(res.passed for res in results)

    Trust model:
        All paths come from test fixtures (trusted). No user-supplied file
        paths flow through this class in production use (T-01-02 accepted).
    """

    def __init__(self, paths: list[Path | str], min_experiments: int = 1) -> None:
        """Initialize without loading any files.

        Args:
            paths: List of paths to causal JSON files produced by rocprof-sys.
                   May be empty — all assert_* methods handle the empty case.
            min_experiments: Minimum number of experiments per speedup entry
                             required by compute_speedups(). Default=1 for
                             single-experiment test fixture ergonomics.
        """
        self._paths: list[Path] = [Path(p) for p in paths]
        self._min_experiments: int = min_experiments
        self._raw_data: dict[str, Any] = {}
        self._results: list[CheckResult] = []
        self._loaded: bool = False
        self._load_error: CheckResult | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load all JSON files on first call. Idempotent on subsequent calls.

        Reads each path with json.loads(), then calls process_data() to merge
        the experiment records into self._raw_data.

        On a load failure — missing/unreadable file (OSError), malformed JSON
        (JSONDecodeError/ValueError), or unexpected schema (KeyError/TypeError
        from process_data) — records the error in self._load_error, appends a
        failed CheckResult to self._results, and sets self._loaded = True (to
        avoid infinite retry loops). Does NOT raise; the stored error is
        surfaced by every assert_* method so the real cause is never hidden.
        """
        if self._loaded:
            return

        try:
            for path in self._paths:
                inp = json.loads(path.read_text())
                self._raw_data = _cv.process_data(self._raw_data, inp, ".*", ".*")
        except (json.JSONDecodeError, ValueError, OSError, KeyError, TypeError) as exc:
            self._load_error = CheckResult(
                passed=False,
                validator_name="CausalReader._ensure_loaded",
                message=f"Failed to load causal JSON: {exc}",
                details={"error": str(exc), "paths": [str(p) for p in self._paths]},
            )
            self._results.append(self._load_error)
        finally:
            self._loaded = True

    def _load_error_result(self, validator_name: str) -> CheckResult:
        """Surface a recorded load error as the failure for ``validator_name``.

        Returns a copy of the stored load-error CheckResult re-labelled with the
        calling validator's name so the assertion reports the actual root cause
        (e.g. file-not-found) rather than a misleading "no records" message.
        """
        assert self._load_error is not None
        result = CheckResult(
            passed=False,
            validator_name=validator_name,
            message=self._load_error.message,
            details=dict(self._load_error.details),
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # assert_has_experiments
    # -------------------------------------------------------------------------

    def assert_has_experiments(self) -> CheckResult:
        """Assert that at least one causal experiment record was loaded.

        Returns:
            CheckResult: passed=True if _raw_data is non-empty.
        """
        self._ensure_loaded()
        if self._load_error is not None:
            return self._load_error_result("assert_has_experiments")
        passed = bool(self._raw_data)
        result = CheckResult(
            passed=passed,
            validator_name="assert_has_experiments",
            message=(
                "Experiment records found"
                if passed
                else "No causal experiment records found"
            ),
            details={"num_experiments": len(self._raw_data)},
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # assert_has_progress_points
    # -------------------------------------------------------------------------

    def assert_has_progress_points(self, pattern: str = ".*") -> CheckResult:
        """Assert that progress points matching the given regex pattern exist.

        Iterates self._raw_data ({exp: {pp: {speedup: point}}}) and collects
        all unique progress point names, then uses re.search(pattern, pp_name)
        to find matches.

        Args:
            pattern: Regex pattern to match against progress point names.
                     Default ".*" matches all progress points.

        Returns:
            CheckResult: passed=True if at least one progress point matches pattern.
        """
        self._ensure_loaded()
        if self._load_error is not None:
            return self._load_error_result("assert_has_progress_points")
        all_pp: set[str] = set()
        for _exp, pp_dict in self._raw_data.items():
            for pp_name in pp_dict.keys():
                all_pp.add(pp_name)

        matches = sorted(pp for pp in all_pp if re.search(pattern, pp))
        passed = bool(matches)
        result = CheckResult(
            passed=passed,
            validator_name="assert_has_progress_points",
            message=(
                f"Found {len(matches)} progress point(s) matching {pattern!r}"
                if passed
                else f"No progress points matching {pattern!r} found"
            ),
            details={
                "pattern": pattern,
                "found": matches,
            },
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # assert_min_data_points
    # -------------------------------------------------------------------------

    def assert_min_data_points(self, n: int) -> CheckResult:
        """Assert that at least n total observations exist across all throughput/latency points.

        Counts len(point) for each (exp, pp, speedup, point) entry in _raw_data.
        Both throughput_point and latency_point implement __len__ returning the
        number of recorded samples.

        Args:
            n: Minimum number of observations required.

        Returns:
            CheckResult: passed=True if total observations >= n.
        """
        self._ensure_loaded()
        if self._load_error is not None:
            return self._load_error_result("assert_min_data_points")
        total = 0
        for _exp, pp_dict in self._raw_data.items():
            for _pp, speedup_dict in pp_dict.items():
                for _speedup, point in speedup_dict.items():
                    total += len(point)

        passed = total >= n
        result = CheckResult(
            passed=passed,
            validator_name="assert_min_data_points",
            message=(
                f"Found {total} total observation(s) (required >= {n})"
                if passed
                else f"Insufficient data points: found {total}, required >= {n}"
            ),
            details={
                "total_observations": total,
                "required": n,
            },
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # assert_speedup
    # -------------------------------------------------------------------------

    def assert_speedup(
        self,
        experiment: str,
        progress_point: str,
        virtual_speedup: int,
        expected: float,
        tolerance: float,
        ci_mode: bool = False,
    ) -> CheckResult:
        """Assert that the measured speedup for an experiment/progress-point matches expected.

        Calls compute_speedups() to extract line_speedup objects, then finds the
        entry matching the experiment and progress_point regex patterns at the
        requested virtual_speedup value. Delegates the tolerance check to
        causal_validator.validation.validate().

        Args:
            experiment: Regex pattern for the experiment name (e.g., "test.cpp:10").
            progress_point: Regex pattern for the progress point name.
            virtual_speedup: The integer virtual speedup slot to check (e.g., 10 for 10%).
            expected: Expected program speedup percentage.
            tolerance: Allowed deviation from expected (percentage points).
            ci_mode: If True, passes ci=True to validation.validate(), which
                     increases tolerance based on observed standard deviations.
                     This accommodates artificially deflated speedup on CI systems.

        Returns:
            CheckResult: passed=True if tolerance check passes.
                         passed=False if no matching data or tolerance check fails.
        """
        self._ensure_loaded()
        if self._load_error is not None:
            return self._load_error_result("assert_speedup")

        # Build the base details dict (always present in return value)
        base_details: dict[str, Any] = {
            "experiment": experiment,
            "progress_point": progress_point,
            "virtual_speedup": virtual_speedup,
            "ci_mode": ci_mode,
        }

        # Compute experiment_progress groups — search all virtual speedups to find baseline
        ep_list = _cv.compute_speedups(self._raw_data, [], self._min_experiments)

        # Search for an experiment_progress whose name/prog match our patterns
        found_ep = None
        found_ls = None
        for ep in ep_list:
            if not ep.data:
                continue
            # Representative name/prog comes from data[0]
            rep = ep.data[0]
            if not re.search(experiment, rep.get_name()):
                continue
            if not re.search(progress_point, rep.prog):
                continue
            # Found the right experiment_progress group — now find the specific speedup slot
            for ls in ep.data:
                if int(ls.virtual_speedup()) == virtual_speedup:
                    found_ep = ep
                    found_ls = ls
                    break
            if found_ls is not None:
                break

        if found_ls is None:
            result = CheckResult(
                passed=False,
                validator_name="assert_speedup",
                message=(
                    f"No data for experiment={experiment!r}, "
                    f"pp={progress_point!r}, vspeedup={virtual_speedup}"
                ),
                details=base_details,
            )
            self._results.append(result)
            return result

        # Delegate tolerance math to causal_validator.validation. Degenerate
        # experiment data (e.g. zero-sum throughput deltas) makes the speedup
        # math divide by zero; convert that into a failed CheckResult rather
        # than letting it crash the caller.
        try:
            _exp_name = found_ls.get_name()
            _pp_name = found_ls.prog
            vs = found_ls.virtual_speedup()
            prog_speedup = found_ls.compute_speedup()
            prog_stddev = found_ls.compute_speedup_stddev()
            base_stddev = found_ep.data[0].compute_speedup_stddev()

            stderr_buf = io.StringIO()
            val = _cv.validation(experiment, progress_point, virtual_speedup, expected, tolerance)
            v = val.validate(
                _exp_name,
                _pp_name,
                vs,
                prog_speedup,
                prog_stddev,
                base_stddev,
                ci_mode,
                stderr_buf=stderr_buf,
            )
        except ZeroDivisionError as exc:
            result = CheckResult(
                passed=False,
                validator_name="assert_speedup",
                message=(
                    f"Speedup math failed on degenerate causal data "
                    f"(division by zero): {exc}"
                ),
                details={**base_details, "error": str(exc)},
            )
            self._results.append(result)
            return result

        # v is True (pass), False (fail), or None (filter didn't match — treat as pass)
        passed = v is True or v is None

        result = CheckResult(
            passed=passed,
            validator_name="assert_speedup",
            message=(
                f"Speedup check passed: [{_exp_name}][{_pp_name}][{virtual_speedup}] "
                f"actual={prog_speedup:.2f}, expected={expected} +/- {tolerance}"
                if passed
                else f"Speedup check failed: [{_exp_name}][{_pp_name}][{virtual_speedup}] "
                f"actual={prog_speedup:.2f} not in {expected} +/- {tolerance}"
            ),
            expected=f"{expected} +/- {tolerance}",
            actual=prog_speedup,
            details={
                **base_details,
                "computed_speedup": prog_speedup,
                "speedup_stddev": prog_stddev,
                "validation_notes": stderr_buf.getvalue().strip() or None,
            },
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # FormatReader ABC methods
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """No-op — CausalReader holds no file handles or subprocesses."""
        pass

    def validate(self) -> list[CheckResult]:
        """Return a copy of all accumulated CheckResult objects.

        Returns:
            list[CheckResult]: Results from all assert_* calls made so far,
                               plus any load-error results from _ensure_loaded().
        """
        return list(self._results)
