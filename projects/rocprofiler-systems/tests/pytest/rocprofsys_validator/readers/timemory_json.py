# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TimemoryJsonReader — in-process loader for timemory JSON metric files.

Design decisions:
- Eager load at construction time (single JSON file read on __init__).
  FileNotFoundError and json.JSONDecodeError propagate immediately — no lazy loading.
- stem parameter maps to a JSON metric key, NOT a filename.
  Data path: data["timemory"][stem]["ranks"][0]["graph"]
- assert_pct_self returns passed=False gracefully when percent_self field absent.
  The timemory JSON format does not always include percent_self; callers must not
  assume the field exists.
- No assert_files_present / has_files (D-06). JSON reader is single-file;
  file existence is implicitly checked at construction time.
- NOT a subclass of TimemoryReader (D-04). These are entirely independent readers
  sharing only the FormatReader ABC and CheckResult return type.

Pitfalls:
- KeyError/IndexError in _load_nodes is silently caught and returns [].
  All assert_* methods fail gracefully when a stem is absent.
- Label extraction: idx = prefix.find(">>>"); label = prefix[idx + 4:] strips the
  "|0>>> " prefix. If no ">>>" found, the prefix string is returned unchanged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from rocprofsys_validator.core import CheckResult, FormatReader
from rocprofsys_validator.registry import reader
from rocprofsys_validator.readers.timemory import (
    _calltree_eval,
    _clean_label,
    _to_float,
)

@reader("timemory_json")
class TimemoryJsonReader(FormatReader):
    """Reader and validator for timemory JSON metric files.

    Accepts a single file path to a timemory JSON output. The JSON is loaded
    eagerly at construction time. Validation results are accumulated in
    ``self._results`` and returned by ``validate()``.

    The ``stem`` parameter used in all assert_* methods maps to the metric key
    in the JSON schema::

        data["timemory"][stem]["ranks"][0]["graph"]

    Usage::

        with TimemoryJsonReader("timemory-output.json") as t:
            t.assert_label_exists("wall_clock", "main")
            t.assert_count("wall_clock", "child_func", min_count=5)
            t.assert_metric_range("wall_clock", "main", "laps", min_val=0, max_val=10)
            results = t.validate()
    """

    def __init__(self, path: str | Path) -> None:
        """Load timemory JSON file at construction time (eager, not lazy).

        Args:
            path: Path to the timemory JSON metric file.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file contains malformed JSON.
        """
        self._path = Path(path)
        self._data: dict = json.loads(self._path.read_text())
        self._results: list[CheckResult] = []

    def close(self) -> None:
        """No-op — JSON reader holds no persistent resources."""

    def validate(self) -> list[CheckResult]:
        """Return all accumulated validation results.

        Returns:
            list[CheckResult]: Copy of all validation results accumulated so far.
        """
        return list(self._results)

    def _extract_label(self, prefix: str) -> str:
        """Extract the label from a timemory node prefix string.

        Strips the ``|N>>> `` header produced by the timemory hierarchy format.
        For example::

            "|0>>> main"         -> "main"
            "|0>>> |_child_func" -> "|_child_func"
            "no arrows here"     -> "no arrows here"

        Args:
            prefix: Raw prefix string from a timemory JSON graph node.

        Returns:
            str: The extracted label (prefix after '>>> ', or unchanged if absent).
        """
        idx = prefix.find(">>>")
        if idx >= 0:
            return prefix[idx + 4:]
        return prefix

    def _load_nodes(self, stem: str) -> list[dict]:
        """Load the graph node list for a given metric stem.

        Accesses ``data["timemory"][stem]["ranks"][0]["graph"]``.
        Returns an empty list if any key or index is missing.

        Args:
            stem: Metric key name (e.g., "wall_clock", "trip_count").

        Returns:
            list[dict]: Graph node list, or [] if stem is absent or malformed.
        """
        try:
            return self._data["timemory"][stem]["ranks"][0]["graph"]
        except (KeyError, IndexError):
            return []

    def _match_nodes(
        self,
        nodes: list[dict],
        pattern: str,
        match_type: str = "auto",
    ) -> list[dict]:
        """Return nodes whose extracted label matches the given pattern.

        Mirrors TimemoryReader._match_rows() logic exactly.

        Match modes:
        - ``"exact"``: exact string equality
        - ``"substring"``: case-sensitive substring search (``pattern in label``)
        - ``"regex"``: Python ``re.search()``
        - ``"auto"``: try exact; if empty try substring; if empty try regex

        Args:
            nodes: List of timemory graph node dicts to search.
            pattern: Pattern to match against extracted labels.
            match_type: One of "exact", "substring", "regex", or "auto".

        Returns:
            list[dict]: Matching nodes (may be empty).
        """
        if not nodes:
            return []

        if match_type == "exact":
            return [n for n in nodes if self._extract_label(n.get("prefix", "")) == pattern]

        elif match_type == "substring":
            return [n for n in nodes if pattern in self._extract_label(n.get("prefix", ""))]

        elif match_type == "regex":
            return [
                n for n in nodes
                if re.search(pattern, self._extract_label(n.get("prefix", "")))
            ]

        else:  # auto
            # Try exact first
            matched = [n for n in nodes if self._extract_label(n.get("prefix", "")) == pattern]
            if matched:
                return matched
            # Try substring
            matched = [n for n in nodes if pattern in self._extract_label(n.get("prefix", ""))]
            if matched:
                return matched
            # Try regex
            return [
                n for n in nodes
                if re.search(pattern, self._extract_label(n.get("prefix", "")))
            ]

    def assert_label_exists(
        self,
        stem: str,
        label_pattern: str,
        match_type: str = "auto",
    ) -> CheckResult:
        """Assert that at least one label matching the pattern exists in the graph.

        Args:
            stem: Metric key name (e.g., "wall_clock").
            label_pattern: Pattern to match against extracted node labels.
            match_type: One of "exact", "substring", "regex", or "auto".

        Returns:
            CheckResult: passed=True if at least one matching label found.
        """
        nodes = self._load_nodes(stem)
        matched = self._match_nodes(nodes, label_pattern, match_type)
        passed = bool(matched)
        result = CheckResult(
            passed=passed,
            validator_name="assert_label_exists",
            message=(
                f"Label matching {label_pattern!r} found in stem {stem!r}"
                if passed
                else f"No label matching {label_pattern!r} found in stem {stem!r}"
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
        """Assert that NO label matching the pattern exists in the graph.

        Replaces fail_regex semantics from the old assert_timemory fixture.
        Inverts the presence check of assert_label_exists: passes when no node
        matches the pattern, fails when a matching node is found.

        An empty node list (e.g., a missing stem) means no label can match,
        so the absence assertion is satisfied and passed=True is returned.

        Args:
            stem: Metric key name (e.g., "wall_clock").
            label_pattern: Pattern to match against extracted node labels.
            match_type: One of "exact", "substring", "regex", or "auto".

        Returns:
            CheckResult: passed=True if no matching label found;
                         passed=False if a matching label was found.
        """
        nodes = self._load_nodes(stem)
        matched = self._match_nodes(nodes, label_pattern, match_type)
        passed = not bool(matched)
        result = CheckResult(
            passed=passed,
            validator_name="assert_label_absent",
            message=(
                f"No label matching {label_pattern!r} found in stem {stem!r} (correct)"
                if passed
                else f"Label matching {label_pattern!r} should be absent but was found in stem {stem!r}"
            ),
            expected=None if passed else f"no match for {label_pattern!r}",
            actual=None if passed else self._extract_label(matched[0].get("prefix", "")),
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

    def assert_depth(
        self,
        stem: str,
        label_pattern: str,
        expected_depth: int,
    ) -> CheckResult:
        """Assert that the first matched label is at the expected graph depth.

        Depth is the top-level 'depth' field in each graph node dict, NOT a
        field inside 'entry'. Value 0 = root, 1 = first child, etc.

        For labels appearing multiple times (duplicate names at different depths),
        only the first matched node's depth is checked.

        Args:
            stem: Metric key name (e.g., "wall_clock").
            label_pattern: Pattern to match against extracted node labels.
            expected_depth: Expected integer depth value (0 = root).

        Returns:
            CheckResult: passed=True if first matched node's depth == expected_depth;
                         passed=False if label not found or depth does not match.
        """
        nodes = self._load_nodes(stem)
        matched = self._match_nodes(nodes, label_pattern)
        if not matched:
            result = CheckResult(
                passed=False,
                validator_name="assert_depth",
                message=f"Label matching {label_pattern!r} not found in stem {stem!r}",
                expected=expected_depth,
                actual="(label not found)",
                details={"stem": stem, "label_pattern": label_pattern},
            )
            self._results.append(result)
            return result

        actual_depth = matched[0]["depth"]
        passed = actual_depth == expected_depth
        result = CheckResult(
            passed=passed,
            validator_name="assert_depth",
            message=(
                f"depth={actual_depth} == {expected_depth} for label matching {label_pattern!r}"
                if passed
                else f"depth={actual_depth} != {expected_depth} for label matching {label_pattern!r}"
            ),
            expected=expected_depth,
            actual=actual_depth,
            details={"stem": stem, "label_pattern": label_pattern},
        )
        self._results.append(result)
        return result

    def assert_count(
        self,
        stem: str,
        label_pattern: str,
        min_count: int = 1,
    ) -> CheckResult:
        """Assert that entry.laps for a matching label meets a minimum threshold.

        The ``laps`` field of the first matched node's ``entry`` dict is used
        as the count value (mirrors the COUNT column in text format).

        Args:
            stem: Metric key name (e.g., "wall_clock").
            label_pattern: Pattern to match against extracted node labels.
            min_count: Minimum expected laps value (default: 1).

        Returns:
            CheckResult: passed=True if laps >= min_count.
                         passed=False with actual="(label not found)" if no match.
        """
        nodes = self._load_nodes(stem)
        matched = self._match_nodes(nodes, label_pattern)
        if not matched:
            result = CheckResult(
                passed=False,
                validator_name="assert_count",
                message=f"Label matching {label_pattern!r} not found in stem {stem!r}",
                expected=f">= {min_count}",
                actual="(label not found)",
                details={"stem": stem, "label_pattern": label_pattern},
            )
            self._results.append(result)
            return result

        raw_count = matched[0].get("entry", {}).get("laps")
        try:
            count_val = int(raw_count)
        except (ValueError, TypeError) as exc:
            result = CheckResult(
                passed=False,
                validator_name="assert_count",
                message=(
                    f"Non-numeric/absent laps {raw_count!r} for label matching "
                    f"{label_pattern!r} in stem {stem!r}: {exc}"
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
                f"laps={count_val} >= {min_count} for label matching {label_pattern!r}"
                if passed
                else f"laps={count_val} < {min_count} for label matching {label_pattern!r}"
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
        """Assert that entry[column] for a matching label falls within a range.

        If the column does not exist in the node's entry dict (e.g., "percent_self"
        is not always present), returns passed=False with a descriptive message —
        does NOT raise KeyError.

        Mirrors TimemoryReader.assert_metric_range tolerance logic exactly.

        Args:
            stem: Metric key name (e.g., "wall_clock").
            label_pattern: Pattern to match against extracted node labels.
            column: Entry field name to check (e.g., "laps", "value", "accum").
            min_val: Minimum acceptable value (inclusive). None means no lower bound.
            max_val: Maximum acceptable value (inclusive). None means no upper bound.
            tolerance_pct: Expand bounds by this percentage (0.0 = no tolerance).

        Returns:
            CheckResult: passed=True if entry[column] within [min_val, max_val].
                         passed=False if column absent, label not found, or out of range.
        """
        nodes = self._load_nodes(stem)
        matched = self._match_nodes(nodes, label_pattern)
        if not matched:
            result = CheckResult(
                passed=False,
                validator_name="assert_metric_range",
                message=f"Label {label_pattern!r} not found in stem {stem!r}",
                expected=label_pattern,
                actual="(label not found)",
                details={"stem": stem, "column": column},
            )
            self._results.append(result)
            return result

        node = matched[0]
        entry = node.get("entry", {})

        if column not in entry:
            result = CheckResult(
                passed=False,
                validator_name="assert_metric_range",
                message=(
                    f"Column {column!r} not found in entry for stem {stem!r} "
                    f"(available: {list(entry.keys())})"
                ),
                details={
                    "stem": stem,
                    "column": column,
                    "available": list(entry.keys()),
                },
            )
            self._results.append(result)
            return result

        raw_value = entry[column]
        try:
            value = float(raw_value)
        except (ValueError, TypeError) as exc:
            result = CheckResult(
                passed=False,
                validator_name="assert_metric_range",
                message=(
                    f"Non-numeric {column!r} value {raw_value!r} for label "
                    f"{label_pattern!r} in stem {stem!r}: {exc}"
                ),
                details={"stem": stem, "column": column, "raw_value": str(raw_value)},
            )
            self._results.append(result)
            return result

        # Apply tolerance expansion (same as TimemoryReader.assert_metric_range)
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

    def assert_pct_self(
        self,
        stem: str,
        label_pattern: str,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> CheckResult:
        """Assert that percent_self for a matching label falls within a range.

        If the matched node's entry dict does not contain a "percent_self" field
        (common in timemory JSON output), returns passed=False with a message
        indicating the field is not available — does NOT raise an exception.

        When percent_self is present, delegates to assert_metric_range and
        overrides validator_name to "assert_pct_self".

        Args:
            stem: Metric key name (e.g., "wall_clock").
            label_pattern: Pattern to match against extracted node labels.
            min_val: Minimum acceptable percent_self value.
            max_val: Maximum acceptable percent_self value.

        Returns:
            CheckResult: passed=True if percent_self within range;
                         passed=False with "not available" message if field absent.
        """
        nodes = self._load_nodes(stem)
        matched = self._match_nodes(nodes, label_pattern)

        if matched and matched[0].get("entry", {}).get("percent_self") is not None:
            # Delegate to assert_metric_range and override validator_name
            inner = self.assert_metric_range(
                stem, label_pattern, "percent_self", min_val, max_val, 0.0
            )
            corrected = CheckResult(
                passed=inner.passed,
                validator_name="assert_pct_self",
                message=inner.message,
                expected=inner.expected,
                actual=inner.actual,
                details=inner.details,
            )
            # Replace the last appended result (the one assert_metric_range added)
            if self._results and self._results[-1] is inner:
                self._results[-1] = corrected
            return corrected

        # percent_self not available in this node
        result = CheckResult(
            passed=False,
            validator_name="assert_pct_self",
            message="% self not available in JSON format for this node",
            details={"stem": stem, "label_pattern": label_pattern},
        )
        self._results.append(result)
        return result

    # =====================================================================
    # Structural / aggregate-stat validators (mirror; JSON graph variant).
    # =====================================================================

    def _entry_mean_std(self, entry: dict) -> tuple["float | None", "float | None"]:
        """Best-effort (mean, stddev) from a graph entry across timemory versions."""
        mean = entry.get("mean")
        if mean is None:
            value, laps = entry.get("value"), entry.get("laps")
            mv, lv = _to_float(value), _to_float(laps)
            mean = (mv / lv) if (mv is not None and lv) else None
        else:
            mean = _to_float(mean)
        std = entry.get("stddev", entry.get("std", entry.get("std_dev")))
        return mean, _to_float(std)

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
        """Assert structural properties of the call subtree rooted at a label."""
        nodes = self._load_nodes(stem)
        if not nodes:
            result = CheckResult(
                passed=False, validator_name="assert_call_tree",
                message=f"no graph nodes for stem {stem!r}",
                actual="(no data)", details={"stem": stem},
            )
            self._results.append(result)
            return result
        rows = [(_clean_label(n.get("prefix", "")), int(n.get("depth", 0) or 0)) for n in nodes]
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
        """Assert the coefficient of variation (stddev/mean) is within ``max_cv``."""
        nodes = self._load_nodes(stem)
        matched = self._match_nodes(nodes, label_pattern, match)
        if not matched:
            result = CheckResult(
                passed=False, validator_name="assert_cv",
                message=f"label {label_pattern!r} not found in stem {stem!r}",
                actual="(label not found)", details={"stem": stem, "label_pattern": label_pattern},
            )
            self._results.append(result)
            return result
        mean, std = self._entry_mean_std(matched[0].get("entry", {}))
        if mean is None or std is None:
            result = CheckResult(
                passed=False, validator_name="assert_cv",
                message=f"mean/stddev unavailable in JSON entry for {label_pattern!r}",
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
        """Assert a label's lap count and/or CV (no trend — no time series)."""
        nodes = self._load_nodes(stem)
        matched = self._match_nodes(nodes, label_pattern, match)
        if not matched:
            result = CheckResult(
                passed=False, validator_name="assert_iteration_consistency",
                message=f"label {label_pattern!r} not found in stem {stem!r}",
                actual="(label not found)", details={"stem": stem, "label_pattern": label_pattern},
            )
            self._results.append(result)
            return result
        entry = matched[0].get("entry", {})
        issues: list[str] = []
        n = None
        if count is not None:
            n = _to_float(entry.get("laps"))
            if n is None or int(n) != count:
                issues.append(f"laps {None if n is None else int(n)} != {count}")
        cv = None
        if max_cv is not None:
            mean, std = self._entry_mean_std(entry)
            if mean is None or std is None:
                issues.append("mean/stddev unavailable for CV")
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
                     "laps": None if n is None else int(n), "cv": cv},
        )
        self._results.append(result)
        return result

    def assert_no_anti_patterns(
        self,
        stem: str,
        *,
        negative_value: bool = True,
        value_key: str = "value",
        zero_laps: bool = True,
    ) -> CheckResult:
        """Assert no graph node has a negative metric value or zero laps."""
        nodes = self._load_nodes(stem)
        found: dict[str, int] = {}
        for n in nodes:
            entry = n.get("entry", {})
            if negative_value:
                v = _to_float(entry.get(value_key))
                if v is not None and v < 0:
                    found["negative_value"] = found.get("negative_value", 0) + 1
            if zero_laps and _to_float(entry.get("laps")) == 0:
                found["zero_laps"] = found.get("zero_laps", 0) + 1
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
