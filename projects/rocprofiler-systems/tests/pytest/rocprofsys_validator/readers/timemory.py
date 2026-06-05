# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TimemoryReader — pipe-delimited text parser for timemory metric files.

Design decisions:
- Manual line-by-line parser — pandas.read_table() FAILS on this format.
  Banner lines (|---|) and variable column counts per metric type break the parser.
- Column count derived from the header line at runtime — never hardcoded.
  wall_clock/sampling_*_clock: 12 cols; sampling_percent: 6 cols; trip_count: 5 cols.
- Labels contain embedded '|' and '>>>' characters from the timemory hierarchy format.
  After parsing, labels have the form '|0>>> label_text' or '|0>>>|_child_text'.
- '% SELF' column (with space) is present in 12-column files only.
  assert_pct_self gracefully handles files where the column is absent.
- Results are cached in self._cache by stem name — subsequent calls return same object.

Pitfalls (from RESEARCH.md):
- Pitfall 1: pandas.read_table() raises ParserError on banner lines.
- Pitfall 5: Label spaces are stripped by parser; use substring match not raw file text.
- Pitfall 6: Column counts vary — NEVER hardcode n_numeric = 11.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from rocprofsys_validator.core import FormatReader, CheckResult
from rocprofsys_validator.registry import reader

# ---------------------------------------------------------------------------
# Shared helpers for the call-tree / statistics validators (text + json).
# timemory output is aggregated statistics + an explicit call hierarchy — there
# is no timeline, so only the structural and aggregate-stat validators mirror
# across from Perfetto/RocPD.
# ---------------------------------------------------------------------------

def _to_float(v) -> "float | None":
    """Best-effort float conversion; None on non-numeric/absent input."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def _clean_label(raw) -> str:
    """Strip the timemory hierarchy header (``|N>>>``) and ``|_`` markers.

    ``"|0>>> parallel-overhead"``      -> ``"parallel-overhead"``
    ``"|3>>>   |_pthread_barrier_wait"`` -> ``"pthread_barrier_wait"``
    """
    s = str(raw)
    idx = s.find(">>>")
    if idx >= 0:
        s = s[idx + 3:]
    s = s.strip()
    while s.startswith("|_"):
        s = s[2:].strip()
    return s

def _tm_name_match(actual, expected: str, match: str) -> bool:
    """Match a (clean) label against a pattern under exact/substring/regex/auto."""
    if actual is None:
        return False
    a = str(actual)
    if match == "exact":
        return a == expected
    if match == "substring":
        return expected in a
    if match == "regex":
        return re.search(expected, a) is not None
    if match == "auto":
        return a == expected or expected in a or re.search(expected, a) is not None
    raise ValueError(f"invalid match type: {match!r}")

def _calltree_eval(
    rows: list[tuple[str, int]],
    parent_pattern: str,
    contains: "list[str] | None",
    max_depth: "int | None",
    no_recursion: bool,
    match: str,
) -> dict:
    """Evaluate call-tree properties over a pre-order (clean_label, depth) list.

    timemory rows are a pre-order DFS dump with an explicit depth, so a parent's
    descendants are the contiguous following rows with strictly greater depth.
    """
    parents = [i for i, (lbl, _) in enumerate(rows) if _tm_name_match(lbl, parent_pattern, match)]
    if not parents:
        return {"parent_found": False, "issues": ["parent not found"], "found": [], "max_rel": 0}
    found: set[str] = set()
    max_rel = 0
    recursion = False
    n = len(rows)
    for i in parents:
        plbl, pdepth = rows[i]
        j = i + 1
        while j < n and rows[j][1] > pdepth:
            clbl, cd = rows[j]
            found.add(clbl)
            max_rel = max(max_rel, cd - pdepth)
            if no_recursion and clbl == plbl:
                recursion = True
            j += 1
    issues: list[str] = []
    for c in (contains or []):
        if not any(_tm_name_match(fc, c, match) for fc in found):
            issues.append(f"missing child {c!r}")
    if max_depth is not None and max_rel > max_depth:
        issues.append(f"depth {max_rel} > {max_depth}")
    if no_recursion and recursion:
        issues.append("unexpected recursion (descendant shares parent name)")
    return {"parent_found": True, "issues": issues, "found": sorted(found), "max_rel": max_rel}

# Default set of expected metric file stems (TIM-07).
EXPECTED_STEMS: list[str] = [
    "wall_clock",
    "sampling_cpu_clock",
    "sampling_wall_clock",
    "sampling_percent",
    "trip_count",
]

@reader("timemory")
class TimemoryReader(FormatReader):
    """Reader and validator for timemory pipe-delimited metric text files.

    Accepts a directory path containing .txt metric files produced by rocprof-sys.
    Parses each file with a manual line-by-line parser that handles variable column
    counts and embedded pipe characters in labels.

    Usage::

        with TimemoryReader("output_dir/") as t:
            t.assert_files_present(["wall_clock", "sampling_cpu_clock"])
            t.assert_label_exists("wall_clock", label_pattern="main")
            t.assert_metric_range("wall_clock", label="main", column="SUM",
                                  min_val=0.0, max_val=100.0)
            results = t.validate()
    """

    def __init__(self, dir_path: str | Path) -> None:
        """Initialize with a directory path containing .txt metric files.

        Args:
            dir_path: Path to directory containing timemory .txt metric files.
        """
        self._dir = Path(dir_path)
        self._results: list[CheckResult] = []
        self._cache: dict[str, pd.DataFrame] = {}

    def close(self) -> None:
        """No-op — timemory reader holds no persistent resources."""

    def validate(self) -> list[CheckResult]:
        """Return all accumulated validation results.

        Returns:
            list[CheckResult]: All validation results accumulated so far.
        """
        return list(self._results)

    def _parse_file(self, stem: str) -> pd.DataFrame:
        """Parse a pipe-delimited timemory text file into a DataFrame.

        Column count varies by metric type:
        - wall_clock, sampling_cpu_clock, sampling_wall_clock: 12 cols
          (LABEL COUNT DEPTH METRIC UNITS SUM MEAN MIN MAX VAR STDDEV "% SELF")
        - sampling_percent: 6 cols (LABEL COUNT DEPTH METRIC UNITS SUM)
        - trip_count: 5 cols (LABEL COUNT DEPTH METRIC SUM)

        PITFALL: pandas.read_table() FAILS on this format — banner lines break it.
        PITFALL: Column count is NOT hardcoded — derived from actual header line.

        Results are cached in self._cache; subsequent calls return the same object.

        Args:
            stem: File stem name (e.g., "wall_clock", "trip_count") without ".txt".

        Returns:
            pd.DataFrame: Parsed data with columns from the header line.
                          Empty DataFrame if file cannot be parsed.
        """
        if stem in self._cache:
            return self._cache[stem]

        path = self._dir / f"{stem}.txt"
        header_cols: list[str] | None = None
        data_rows: list[list[str]] = []
        n_numeric: int | None = None

        try:
            text = path.read_text()
        except FileNotFoundError:
            self._cache[stem] = pd.DataFrame()
            return self._cache[stem]

        for line in text.splitlines():
            stripped = line.strip()

            # Skip empty lines and banner/separator lines (only '-' and '|' chars)
            if not stripped or all(c in "-|" for c in stripped):
                continue

            # Header line detection: contains both 'LABEL' and '| COUNT'
            if "LABEL" in stripped and "| COUNT" in stripped:
                cols = [c.strip() for c in stripped.split("|")]
                header_cols = [c for c in cols if c]
                n_numeric = len(header_cols) - 1  # LABEL is not numeric
                continue

            # Data row detection: contains '>>>' and header has been found
            if ">>>" in stripped and n_numeric is not None:
                parts = [p.strip() for p in stripped.split("|")]
                # Drop leading/trailing empty strings from outer pipe delimiters
                non_empty = parts[1:-1]
                if len(non_empty) >= n_numeric + 1:
                    # Last n_numeric parts are the numeric columns
                    numeric = non_empty[-n_numeric:]
                    # Everything before the numeric columns is the label
                    label_parts = non_empty[:-n_numeric]
                    # Rejoin with '|' to reconstruct the hierarchical label
                    # (empty first element + label text = "|0>>> label_text")
                    label = "|".join(label_parts).strip()
                    data_rows.append([label] + numeric)

        if not header_cols or not data_rows:
            result = pd.DataFrame()
        else:
            result = pd.DataFrame(data_rows, columns=header_cols)

        self._cache[stem] = result
        return result

    def _match_rows(
        self,
        df: pd.DataFrame,
        label_pattern: str,
        match_type: str = "auto",
    ) -> pd.DataFrame:
        """Return rows from df where the LABEL column matches label_pattern.

        Match modes:
        - "exact": exact string equality
        - "substring": case-sensitive substring search
        - "regex": Python regular expression search
        - "auto": try exact; if no match try substring; if no match try regex

        Args:
            df: DataFrame with a LABEL column to search.
            label_pattern: Pattern to match against LABEL values.
            match_type: One of "exact", "substring", "regex", or "auto".

        Returns:
            pd.DataFrame: Subset of rows where LABEL matches the pattern.
        """
        if df.empty or "LABEL" not in df.columns:
            return df
        if match_type == "exact":
            mask = df["LABEL"] == label_pattern
            return df[mask]
        elif match_type == "substring":
            mask = df["LABEL"].str.contains(label_pattern, regex=False, na=False)
            return df[mask]
        elif match_type == "regex":
            mask = df["LABEL"].str.contains(label_pattern, regex=True, na=False)
            return df[mask]
        else:  # auto
            # Try exact first
            mask = df["LABEL"] == label_pattern
            if mask.any():
                return df[mask]
            # Try substring
            mask = df["LABEL"].str.contains(label_pattern, regex=False, na=False)
            if mask.any():
                return df[mask]
            # Try regex
            mask = df["LABEL"].str.contains(label_pattern, regex=True, na=False)
            return df[mask]

    def assert_files_present(
        self,
        expected_stems: list[str] | None = None,
    ) -> CheckResult:
        """Assert that all expected metric file stems are present in the directory.

        TIM-07: Discover which metric files are present.

        Args:
            expected_stems: List of file stem names to check (e.g., ["wall_clock"]).
                            If None, uses EXPECTED_STEMS (all five standard files).

        Returns:
            CheckResult: passed=True if all stems found; passed=False with
                              details["missing"] listing absent stems.
        """
        stems = expected_stems if expected_stems is not None else EXPECTED_STEMS
        found = {p.stem for p in self._dir.glob("*.txt")}
        missing = [s for s in stems if s not in found]
        passed = not missing
        result = CheckResult(
            passed=passed,
            validator_name="assert_files_present",
            message=(
                "All expected files present"
                if passed
                else f"Missing timemory files: {missing}"
            ),
            expected=stems,
            actual=sorted(found),
            details={
                "missing": missing,
                "found": sorted(found),
                "directory": str(self._dir),
            },
        )
        self._results.append(result)
        return result

    def assert_label_exists(
        self,
        stem: str,
        label_pattern: str,
        match_type: str = "auto",
    ) -> CheckResult:
        """Assert that at least one label matching the pattern exists in the file.

        TIM-02: Assert named label exists.

        Args:
            stem: File stem name (e.g., "wall_clock").
            label_pattern: Pattern to match against LABEL column values.
            match_type: One of "exact", "substring", "regex", or "auto".

        Returns:
            CheckResult: passed=True if at least one matching label found.
        """
        df = self._parse_file(stem)
        rows = self._match_rows(df, label_pattern, match_type)
        passed = not rows.empty
        result = CheckResult(
            passed=passed,
            validator_name="assert_label_exists",
            message=(
                f"Label matching {label_pattern!r} found in {stem}.txt"
                if passed
                else f"No label matching {label_pattern!r} found in {stem}.txt"
            ),
            expected=label_pattern if not passed else None,
            actual="(none)" if not passed else None,
            details=(
                {}
                if passed
                else {
                    "stem": stem,
                    "label_pattern": label_pattern,
                    "match_type": match_type,
                }
            ),
        )
        self._results.append(result)
        return result

    def assert_label_absent(
        self,
        stem: str,
        label_pattern: str,
        match_type: str = "auto",
    ) -> CheckResult:
        """Assert that NO label matching the pattern exists in the file (negative test).

        The complement of assert_label_exists — gives the text reader the same
        absence verb the JSON reader already exposes (without_label).

        Args:
            stem: File stem name (e.g., "wall_clock").
            label_pattern: Pattern to match against LABEL column values.
            match_type: One of "exact", "substring", "regex", or "auto".

        Returns:
            CheckResult: passed=True when no matching label is found.
        """
        df = self._parse_file(stem)
        rows = self._match_rows(df, label_pattern, match_type)
        passed = rows.empty
        result = CheckResult(
            passed=passed,
            validator_name="assert_label_absent",
            message=(
                f"No label matching {label_pattern!r} present in {stem}.txt (as expected)"
                if passed
                else f"Label matching {label_pattern!r} unexpectedly present in {stem}.txt"
            ),
            expected=f"no label matching {label_pattern!r}" if not passed else None,
            actual=f"{len(rows)} matching label(s)" if not passed else None,
            details={
                "stem": stem,
                "label_pattern": label_pattern,
                "match_type": match_type,
            },
        )
        self._results.append(result)
        return result

    def assert_count(
        self,
        stem: str,
        label_pattern: str,
        min_count: int = 1,
    ) -> CheckResult:
        """Assert that the COUNT column value for a label meets a minimum threshold.

        TIM-03: Assert COUNT column value >= minimum.

        Args:
            stem: File stem name (e.g., "wall_clock").
            label_pattern: Pattern to match against LABEL column values.
            min_count: Minimum expected COUNT value (default: 1).

        Returns:
            CheckResult: passed=True if COUNT >= min_count.
        """
        df = self._parse_file(stem)
        rows = self._match_rows(df, label_pattern)
        if rows.empty:
            result = CheckResult(
                passed=False,
                validator_name="assert_count",
                message=f"Label matching {label_pattern!r} not found in {stem}.txt",
                expected=f">= {min_count}",
                actual="(label not found)",
                details={"stem": stem, "label_pattern": label_pattern},
            )
            self._results.append(result)
            return result

        raw_count = rows["COUNT"].iloc[0]
        try:
            count_val = int(raw_count)
        except (ValueError, TypeError) as exc:
            result = CheckResult(
                passed=False,
                validator_name="assert_count",
                message=(
                    f"Non-numeric COUNT {raw_count!r} for label matching "
                    f"{label_pattern!r} in {stem}.txt: {exc}"
                ),
                details={"stem": stem, "label_pattern": label_pattern,
                         "raw_value": str(raw_count)},
            )
            self._results.append(result)
            return result
        passed = count_val >= min_count
        result = CheckResult(
            passed=passed,
            validator_name="assert_count",
            message=(
                f"COUNT={count_val} >= {min_count} for label matching {label_pattern!r}"
                if passed
                else f"COUNT={count_val} < {min_count} for label matching {label_pattern!r}"
            ),
            expected=f">= {min_count}",
            actual=count_val,
            details={"stem": stem, "label_pattern": label_pattern},
        )
        self._results.append(result)
        return result

    def assert_metric_range(
        self,
        stem: str,
        label_pattern: str,
        column: str,
        min_val: float | None = None,
        max_val: float | None = None,
        tolerance_pct: float = 0.0,
    ) -> CheckResult:
        """Assert that a metric column value for a label falls within a range.

        TIM-04: Assert SUM/MEAN/MIN/MAX within ±% tolerance.

        If the column does not exist in the DataFrame (e.g., "% SELF" in trip_count.txt),
        returns passed=False with a descriptive message — does NOT raise KeyError.

        Args:
            stem: File stem name (e.g., "wall_clock").
            label_pattern: Pattern to match against LABEL column values.
            column: Column name to check (e.g., "SUM", "MEAN", "% SELF").
            min_val: Minimum acceptable value (inclusive). None means no lower bound.
            max_val: Maximum acceptable value (inclusive). None means no upper bound.
            tolerance_pct: Expand bounds by this percentage (0.0 = no tolerance).

        Returns:
            CheckResult: passed=True if column value within [min_val, max_val].
                              passed=False if column absent, label not found, or out of range.
        """
        df = self._parse_file(stem)

        # Guard: column must exist in this file type
        if column not in df.columns:
            result = CheckResult(
                passed=False,
                validator_name="assert_metric_range",
                message=(
                    f"Column {column!r} not found in {stem}.txt "
                    f"(available: {list(df.columns)})"
                ),
                details={"stem": stem, "column": column, "available": list(df.columns)},
            )
            self._results.append(result)
            return result

        rows = self._match_rows(df, label_pattern)
        if rows.empty:
            result = CheckResult(
                passed=False,
                validator_name="assert_metric_range",
                message=f"Label matching {label_pattern!r} not found in {stem}.txt",
                expected=label_pattern,
                actual="(label not found)",
                details={"stem": stem, "column": column},
            )
            self._results.append(result)
            return result

        raw_value = rows[column].iloc[0]
        try:
            value = float(raw_value)
        except (ValueError, TypeError) as exc:
            result = CheckResult(
                passed=False,
                validator_name="assert_metric_range",
                message=(
                    f"Non-numeric {column!r} value {raw_value!r} for label "
                    f"matching {label_pattern!r} in {stem}.txt: {exc}"
                ),
                details={"stem": stem, "column": column, "raw_value": str(raw_value)},
            )
            self._results.append(result)
            return result

        # Apply tolerance expansion
        eff_min = min_val
        eff_max = max_val
        if tolerance_pct > 0.0:
            if eff_min is not None:
                eff_min = eff_min * (1.0 - tolerance_pct / 100.0)
            if eff_max is not None:
                eff_max = eff_max * (1.0 + tolerance_pct / 100.0)

        passed = (eff_min is None or value >= eff_min) and (
            eff_max is None or value <= eff_max
        )

        # Build expected string
        parts = []
        if min_val is not None:
            parts.append(f"{column} >= {min_val}")
        if max_val is not None:
            parts.append(f"{column} <= {max_val}")
        expected_str = " and ".join(parts) if parts else f"{column} in any range"

        result = CheckResult(
            passed=passed,
            validator_name="assert_metric_range",
            message=(
                f"{column}={value} in range [{eff_min}, {eff_max}]"
                if passed
                else f"{column}={value} out of range [{eff_min}, {eff_max}]"
            ),
            expected=expected_str,
            actual=value,
            details={"stem": stem, "column": column, "label_pattern": label_pattern},
        )
        self._results.append(result)
        return result

    def assert_cpu_sampling(
        self,
        label_pattern: str,
        column: str = "SUM",
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> CheckResult:
        """Assert that a CPU sampling metric value falls within a range.

        TIM-05: Assert CPU sampling values within range.
        Delegates to assert_metric_range using "sampling_cpu_clock" as the stem.

        Args:
            label_pattern: Pattern to match against LABEL column values.
            column: Column name to check (default: "SUM").
            min_val: Minimum acceptable value (inclusive).
            max_val: Maximum acceptable value (inclusive).

        Returns:
            CheckResult: passed=True if column value within [min_val, max_val].
        """
        result = self.assert_metric_range(
            stem="sampling_cpu_clock",
            label_pattern=label_pattern,
            column=column,
            min_val=min_val,
            max_val=max_val,
        )
        # Override validator_name to reflect the actual method called
        # Re-create result with correct validator_name (CheckResult is a dataclass)
        corrected = CheckResult(
            passed=result.passed,
            validator_name="assert_cpu_sampling",
            message=result.message,
            expected=result.expected,
            actual=result.actual,
            details=result.details,
        )
        # Replace the last appended result with the corrected one
        if self._results and self._results[-1] is result:
            self._results[-1] = corrected
        return corrected

    def assert_pct_self(
        self,
        stem: str,
        label_pattern: str,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> CheckResult:
        """Assert that the '% SELF' column value for a label falls within a range.

        TIM-06: Assert '% SELF' value for label.
        The '% SELF' column is present only in files with 12 columns (wall_clock,
        sampling_cpu_clock, sampling_wall_clock). For trip_count and sampling_percent,
        this method returns passed=False with a "column not found" message gracefully.

        Args:
            stem: File stem name (e.g., "wall_clock").
            label_pattern: Pattern to match against LABEL column values.
            min_val: Minimum acceptable '% SELF' value.
            max_val: Maximum acceptable '% SELF' value.

        Returns:
            CheckResult: passed=True if '% SELF' value within range;
                              passed=False with message if column absent.
        """
        result = self.assert_metric_range(
            stem=stem,
            label_pattern=label_pattern,
            column="% SELF",
            min_val=min_val,
            max_val=max_val,
        )
        # Override validator_name to reflect the actual method called
        corrected = CheckResult(
            passed=result.passed,
            validator_name="assert_pct_self",
            message=result.message,
            expected=result.expected,
            actual=result.actual,
            details=result.details,
        )
        # Replace the last appended result with the corrected one
        if self._results and self._results[-1] is result:
            self._results[-1] = corrected
        return corrected

    # =====================================================================
    # Structural / aggregate-stat validators (mirror of the timeline readers,
    # restricted to what aggregated timemory output can honestly express).
    # =====================================================================

    def assert_call_tree(
        self,
        stem: str,
        parent_pattern: str,
        *,
        contains: list[str] | None = None,
        max_depth: int | None = None,
        no_recursion: bool = False,
        match: str = "auto",
    ) -> CheckResult:
        """Assert structural properties of the call subtree rooted at a label.

        Uses timemory's native hierarchy (the ``DEPTH`` column in pre-order).
        """
        df = self._parse_file(stem)
        if df.empty or "DEPTH" not in df.columns or "LABEL" not in df.columns:
            result = CheckResult(
                passed=False, validator_name="assert_call_tree",
                message=f"no call-tree data in stem {stem!r}",
                details={"stem": stem}, actual="(no data)",
            )
            self._results.append(result)
            return result
        rows = []
        for lbl, dep in zip(df["LABEL"].tolist(), df["DEPTH"].tolist()):
            d = _to_float(dep)
            rows.append((_clean_label(lbl), int(d) if d is not None else 0))
        ev = _calltree_eval(rows, parent_pattern, contains, max_depth, no_recursion, match)
        if not ev["parent_found"]:
            result = CheckResult(
                passed=False, validator_name="assert_call_tree",
                message=f"no label matched {parent_pattern!r} in {stem!r}",
                actual="(parent not found)", details={"stem": stem, "parent": parent_pattern},
            )
            self._results.append(result)
            return result
        passed = not ev["issues"]
        result = CheckResult(
            passed=passed, validator_name="assert_call_tree",
            message=(f"call tree under {parent_pattern!r} ok (depth {ev['max_rel']})" if passed
                     else f"call tree under {parent_pattern!r} issues: " + "; ".join(ev["issues"])),
            expected=None if passed else "; ".join(ev["issues"]),
            actual=None if passed else ev["found"],
            details={"stem": stem, "parent": parent_pattern,
                     "children_found": ev["found"], "max_rel_depth": ev["max_rel"]},
        )
        self._results.append(result)
        return result

    def assert_cv(
        self,
        stem: str,
        label_pattern: str,
        max_cv: float,
        *,
        match: str = "auto",
    ) -> CheckResult:
        """Assert the coefficient of variation (STDDEV/MEAN) is within ``max_cv``."""
        df = self._parse_file(stem)
        rows = self._match_rows(df, label_pattern, match)
        if rows.empty:
            result = CheckResult(
                passed=False, validator_name="assert_cv",
                message=f"label {label_pattern!r} not found in {stem}.txt",
                actual="(label not found)", details={"stem": stem, "label_pattern": label_pattern},
            )
            self._results.append(result)
            return result
        if "MEAN" not in df.columns or "STDDEV" not in df.columns:
            result = CheckResult(
                passed=False, validator_name="assert_cv",
                message=f"MEAN/STDDEV columns unavailable in {stem}.txt",
                details={"stem": stem, "available": list(df.columns)},
            )
            self._results.append(result)
            return result
        mean = _to_float(rows.iloc[0]["MEAN"])
        std = _to_float(rows.iloc[0]["STDDEV"])
        if mean is None or std is None:
            result = CheckResult(
                passed=False, validator_name="assert_cv",
                message=f"non-numeric MEAN/STDDEV for {label_pattern!r} in {stem}.txt",
                details={"stem": stem, "label_pattern": label_pattern},
            )
            self._results.append(result)
            return result
        cv = 0.0 if std == 0 else (std / mean if mean else float("inf"))
        passed = cv <= max_cv
        result = CheckResult(
            passed=passed, validator_name="assert_cv",
            message=f"CV for {label_pattern!r} = {cv:.4f}",
            expected=None if passed else f"<= {max_cv}",
            actual=None if passed else f"{cv:.4f}",
            details={"stem": stem, "label_pattern": label_pattern,
                     "cv": cv, "mean": mean, "stddev": std},
        )
        self._results.append(result)
        return result

    def assert_iteration_consistency(
        self,
        stem: str,
        label_pattern: str,
        *,
        count: int | None = None,
        max_cv: float | None = None,
        match: str = "auto",
    ) -> CheckResult:
        """Assert a label's invocation count and/or CV (no trend — no time series)."""
        df = self._parse_file(stem)
        rows = self._match_rows(df, label_pattern, match)
        if rows.empty:
            result = CheckResult(
                passed=False, validator_name="assert_iteration_consistency",
                message=f"label {label_pattern!r} not found in {stem}.txt",
                actual="(label not found)", details={"stem": stem, "label_pattern": label_pattern},
            )
            self._results.append(result)
            return result
        issues: list[str] = []
        n = None
        if count is not None:
            n = _to_float(rows.iloc[0]["COUNT"]) if "COUNT" in df.columns else None
            if n is None or int(n) != count:
                issues.append(f"count {None if n is None else int(n)} != {count}")
        cv = None
        if max_cv is not None:
            mean = _to_float(rows.iloc[0]["MEAN"]) if "MEAN" in df.columns else None
            std = _to_float(rows.iloc[0]["STDDEV"]) if "STDDEV" in df.columns else None
            if mean is None or std is None:
                issues.append("MEAN/STDDEV unavailable for CV")
            else:
                cv = 0.0 if std == 0 else (std / mean if mean else float("inf"))
                if cv > max_cv:
                    issues.append(f"CV {cv:.4f} > {max_cv}")
        passed = not issues
        result = CheckResult(
            passed=passed, validator_name="assert_iteration_consistency",
            message=(f"{label_pattern!r} iteration stats consistent" if passed
                     else f"{label_pattern!r} issues: " + "; ".join(issues)),
            expected=None if passed else "stable iterations",
            actual=None if passed else "; ".join(issues),
            details={"stem": stem, "label_pattern": label_pattern,
                     "count": None if n is None else int(n), "cv": cv},
        )
        self._results.append(result)
        return result

    def assert_no_anti_patterns(
        self,
        stem: str,
        *,
        negative_metrics: bool = True,
        metric_columns: tuple[str, ...] = ("SUM", "MEAN", "MIN", "MAX"),
        zero_count: bool = True,
    ) -> CheckResult:
        """Assert a label table has no negative metrics or zero-count labels."""
        df = self._parse_file(stem)
        found: dict[str, int] = {}
        if not df.empty:
            if negative_metrics:
                for col in metric_columns:
                    if col in df.columns:
                        neg = sum(1 for v in df[col].tolist()
                                  if (_to_float(v) is not None and _to_float(v) < 0))
                        if neg:
                            found[f"negative_{col}"] = neg
            if zero_count and "COUNT" in df.columns:
                zc = sum(1 for v in df["COUNT"].tolist() if _to_float(v) == 0)
                if zc:
                    found["zero_count"] = zc
        passed = not found
        result = CheckResult(
            passed=passed, validator_name="assert_no_anti_patterns",
            message=("no anti-patterns detected" if passed
                     else "anti-patterns detected: " + ", ".join(f"{k}={v}" for k, v in found.items())),
            expected=None if passed else "no anti-patterns",
            actual=None if passed else found,
            details={"stem": stem, "found": found},
        )
        self._results.append(result)
        return result
