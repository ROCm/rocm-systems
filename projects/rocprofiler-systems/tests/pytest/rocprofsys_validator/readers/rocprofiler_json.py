# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RocprofilerJsonReader — ground-truth record counts from rocprofiler-sdk JSON.

The rocprofiler-sdk JSON output is the canonical reference for cross-format
count-equality checks. Its records are grouped under::

    json["rocprofiler-sdk-tool"]["buffer_records"][<category>]   -> list
    json["rocprofiler-sdk-tool"]["callback_records"][<category>] -> list
    json["rocprofiler-sdk-tool"]["agents"]                       -> list

so the count of a given record kind is simply the length of its list. This
reader exposes those counts so they can be compared against Perfetto slice
counts and RocPD view counts via correlation.assert_record_count_parity.

Validated via stdlib ``json`` only — no rocprofiler/timemory Python bindings.
"""
from __future__ import annotations

import json
from pathlib import Path

from rocprofsys_validator.core import CheckResult, FormatReader
from rocprofsys_validator.registry import reader

_ROOT_KEY = "rocprofiler-sdk-tool"

@reader("rocprofiler_json")
class RocprofilerJsonReader(FormatReader):
    """Reads record counts from a rocprofiler-sdk JSON document."""

    def __init__(self, path: str | Path) -> None:
        """Load and parse the JSON document.

        Args:
            path: Path to the rocprofiler-sdk JSON output file.
        """
        self._path = str(path)
        with open(self._path, "r", encoding="utf-8") as fh:
            self._data = json.load(fh)
        self._root = self._data.get(_ROOT_KEY, {})
        self._results: list[CheckResult] = []

    def validate(self) -> list[CheckResult]:
        """Return all accumulated validation results."""
        return list(self._results)

    def record_count(self, category: str, kind: str = "buffer_records") -> int:
        """Return the number of records for a category.

        Args:
            category: Record category (e.g. 'kernel_dispatch', 'memory_copy').
            kind: Section to read — 'buffer_records' (default), 'callback_records',
                  or 'agents' (category is ignored for 'agents').

        Returns:
            The record count, or 0 when the section/category is absent.
        """
        if kind == "agents":
            section = self._root.get("agents", [])
            return len(section) if isinstance(section, list) else 0
        section = self._root.get(kind, {})
        records = section.get(category, []) if isinstance(section, dict) else []
        return len(records) if isinstance(records, list) else 0

    def assert_record_count(
        self,
        category: str,
        min_count: int = 1,
        kind: str = "buffer_records",
    ) -> CheckResult:
        """Assert a category has at least min_count records.

        Args:
            category: Record category to count.
            min_count: Minimum expected record count (default: 1).
            kind: Record section (see record_count).

        Returns:
            CheckResult: passed=True if count >= min_count.
        """
        count = self.record_count(category, kind)
        passed = count >= min_count
        result = CheckResult(
            passed=passed,
            validator_name="assert_record_count",
            message=(
                f"{kind}[{category!r}] has {count} record(s)"
                if passed
                else f"Expected >= {min_count} {kind}[{category!r}] record(s), found {count}"
            ),
            expected=f">= {min_count}",
            actual=count,
            details={"category": category, "kind": kind, "count": count},
        )
        self._results.append(result)
        return result
