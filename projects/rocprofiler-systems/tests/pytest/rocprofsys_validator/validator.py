# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ProfilerOutputValidator — a facade over all per-format validators.

Discovers the output files of a rocprof-sys run, instantiates the matching
``FormatReader`` for each, and translates *semantic capability* calls to the
specific readers that support them. Formats that cannot answer a given capability
are reported as SKIPPED rather than failing or raising.

Design:
- SQL never crosses the facade. Each reader compiles a capability to its own
  medium (PerfettoSQL / SQLite / pandas / dict walk); the facade only speaks the
  semantic vocabulary and aggregates ``CheckResult`` objects.
- Capabilities are modelled as runtime-checkable Protocols. A reader "supports" a
  capability iff it implements the protocol's methods — so adding a format or a
  capability needs no change here (the project's extensibility constraint).
- The raw query escape hatch stays per-format and explicit (``v.perfetto.sql(...)``,
  ``v.rocpd.sql(...)``). There is deliberately no unified ``v.sql(...)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from rocprofsys_validator.core import CheckResult, FormatReader

# ---------------------------------------------------------------------------
# Capability protocols — a reader supports a capability iff it has these methods.
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsTimeline(Protocol):
    """Timestamped events: utilization, gaps, overlap, serial, flow latency."""

    def assert_gpu_utilization(self, *args, **kwargs) -> CheckResult: ...
    def assert_max_idle_gap(self, *args, **kwargs) -> CheckResult: ...
    def assert_serial_on_stream(self, *args, **kwargs) -> CheckResult: ...
    def assert_overlap(self, *args, **kwargs) -> CheckResult: ...
    def assert_flow_latency(self, *args, **kwargs) -> CheckResult: ...

@runtime_checkable
class SupportsCounters(Protocol):
    """Counter/PMC time series: range and rate-of-change checks."""

    def assert_counter_in_range(self, *args, **kwargs) -> CheckResult: ...
    def assert_counter_rate(self, *args, **kwargs) -> CheckResult: ...

@runtime_checkable
class SupportsCallTree(Protocol):
    """A call/region hierarchy: containment, depth, recursion checks."""

    def assert_call_tree(self, *args, **kwargs) -> CheckResult: ...

@runtime_checkable
class SupportsAntiPatterns(Protocol):
    """A curated "this should never appear" bundle."""

    def assert_no_anti_patterns(self, *args, **kwargs) -> CheckResult: ...

_TIMEMORY_FORMATS = {"timemory", "timemory_json"}

class ProfilerOutputValidator:
    """Facade that fans semantic checks across all present profiler outputs."""

    def __init__(self, readers: dict[str, FormatReader],
                 load_errors: dict[str, str] | None = None) -> None:
        self.readers = readers
        self.load_errors = load_errors or {}
        self._results: list[CheckResult] = []

    # ------------------------------------------------------------------
    # Construction / discovery
    # ------------------------------------------------------------------

    @classmethod
    def from_files(cls, **readers: FormatReader) -> "ProfilerOutputValidator":
        """Build from already-constructed readers, keyed by format name.

        e.g. ``ProfilerOutputValidator.from_files(rocpd=RocpdReader(db),
        timemory=TimemoryReader(dir))``.
        """
        return cls({k: v for k, v in readers.items() if v is not None})

    @classmethod
    def from_directory(cls, path: str | Path) -> "ProfilerOutputValidator":
        """Discover and open whatever profiler outputs exist under ``path``.

        Heuristics (first match wins per format):
        - perfetto       : ``*.pftrace`` / ``perfetto*.proto`` / ``*.proto``
        - rocpd          : ``rocpd*.db`` / ``*.db``
        - timemory (text): the directory itself if it holds ``*.txt`` metric files
        - timemory_json  : ``timemory*.json``
        - causal         : ``*causal*.json``

        Readers that fail to open are recorded in ``load_errors`` and skipped.
        """
        from rocprofsys_validator.readers.causal import CausalReader
        from rocprofsys_validator.readers.perfetto import PerfettoReader
        from rocprofsys_validator.readers.rocpd import RocpdReader
        from rocprofsys_validator.readers.timemory import TimemoryReader
        from rocprofsys_validator.readers.timemory_json import TimemoryJsonReader

        d = Path(path)
        readers: dict[str, FormatReader] = {}
        errors: dict[str, str] = {}

        def _first(*patterns: str) -> Path | None:
            for pat in patterns:
                hits = sorted(d.glob(pat))
                if hits:
                    return hits[0]
            return None

        def _try(fmt: str, factory) -> None:
            try:
                rdr = factory()
                if rdr is not None:
                    readers[fmt] = rdr
            except Exception as exc:  # discovery must not abort on one bad file
                errors[fmt] = f"{type(exc).__name__}: {exc}"

        perfetto_file = _first("*.pftrace", "perfetto*.proto", "*.proto")
        if perfetto_file is not None:
            _try("perfetto", lambda: PerfettoReader(perfetto_file))

        rocpd_file = _first("rocpd*.db", "*.db")
        if rocpd_file is not None:
            _try("rocpd", lambda: RocpdReader(rocpd_file))

        tm_json = _first("timemory*.json")
        if tm_json is not None:
            _try("timemory_json", lambda: TimemoryJsonReader(tm_json))

        causal_file = _first("*causal*.json")
        if causal_file is not None:
            _try("causal", lambda: CausalReader([causal_file]))

        if any(d.glob("*.txt")):
            _try("timemory", lambda: TimemoryReader(d))

        return cls(readers, errors)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def formats(self) -> list[str]:
        return sorted(self.readers)

    def supporting(self, capability: type) -> list[str]:
        """Format names whose reader implements the given capability protocol."""
        return sorted(f for f, r in self.readers.items() if isinstance(r, capability))

    # raw escape hatches — explicit and per-format; no unified .sql()
    @property
    def perfetto(self): return self.readers.get("perfetto")
    @property
    def rocpd(self): return self.readers.get("rocpd")
    @property
    def timemory(self): return self.readers.get("timemory")
    @property
    def timemory_json(self): return self.readers.get("timemory_json")
    @property
    def causal(self): return self.readers.get("causal")

    # ------------------------------------------------------------------
    # Fan-out
    # ------------------------------------------------------------------

    @staticmethod
    def _skipped(fmt: str, what: str, reason: str = "") -> CheckResult:
        tail = f" ({reason})" if reason else ""
        return CheckResult(
            passed=True, validator_name=what,
            message=f"SKIPPED: {fmt} does not support {what}{tail}",
            details={"skipped": True, "format": fmt},
        )

    def for_each(self, capability: type, method: str, *args, **kwargs) -> list[CheckResult]:
        """Run ``method`` on every reader implementing ``capability``.

        Args are passed through verbatim, so this is uniform across readers that
        share a signature (Perfetto and RocPD do, by design). Readers lacking the
        capability yield a SKIPPED result. Results are also accumulated for
        ``assert_ok()``.
        """
        out: list[CheckResult] = []
        for fmt, rdr in self.readers.items():
            if isinstance(rdr, capability) and hasattr(rdr, method):
                out.append(getattr(rdr, method)(*args, **kwargs))
            else:
                out.append(self._skipped(fmt, method))
        self._results.extend(out)
        return out

    def call_tree(
        self,
        parent: str,
        *,
        stem: str | None = None,
        track_pattern: str | None = None,
        contains: list[str] | None = None,
        max_depth: int | None = None,
        no_recursion: bool = False,
        match: str = "exact",
    ) -> list[CheckResult]:
        """Normalized call-tree check across every format that has a hierarchy.

        Adapts the one capability whose native signature differs by family:
        timemory needs a ``stem``; Perfetto takes an optional ``track_pattern``.
        Formats without a hierarchy (or, for timemory, without a ``stem``) are
        reported SKIPPED.
        """
        out: list[CheckResult] = []
        for fmt, rdr in self.readers.items():
            if not isinstance(rdr, SupportsCallTree):
                out.append(self._skipped(fmt, "call_tree"))
                continue
            if fmt in _TIMEMORY_FORMATS:
                if stem is None:
                    out.append(self._skipped(fmt, "call_tree", "stem required"))
                    continue
                tm_match = "auto" if match == "exact" else match
                out.append(rdr.assert_call_tree(
                    stem, parent, contains=contains, max_depth=max_depth,
                    no_recursion=no_recursion, match=tm_match))
            elif fmt == "perfetto":
                out.append(rdr.assert_call_tree(
                    parent, contains=contains, max_depth=max_depth,
                    no_recursion=no_recursion, match=match, track_pattern=track_pattern))
            else:  # rocpd and any other timeline reader
                out.append(rdr.assert_call_tree(
                    parent, contains=contains, max_depth=max_depth,
                    no_recursion=no_recursion, match=match))
        self._results.extend(out)
        return out

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def results(self) -> list[CheckResult]:
        """All non-skipped CheckResults accumulated so far."""
        return [r for r in self._results if not r.details.get("skipped")]

    def summary(self) -> dict:
        real = self.results()
        skipped = [r for r in self._results if r.details.get("skipped")]
        return {
            "formats": self.formats,
            "load_errors": self.load_errors,
            "checks": len(real),
            "passed": sum(1 for r in real if r.passed),
            "failed": sum(1 for r in real if not r.passed),
            "skipped": len(skipped),
        }

    def assert_ok(self, results: list[CheckResult] | None = None) -> bool:
        """Raise AssertionError if any (non-skipped) result failed; else True."""
        pool = results if results is not None else self._results
        failed = [r for r in pool if not r.passed and not r.details.get("skipped")]
        if failed:
            raise AssertionError("\n".join(r._format_message() for r in failed))
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        for rdr in self.readers.values():
            try:
                rdr.close()
            except Exception:
                pass

    def __enter__(self) -> "ProfilerOutputValidator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
