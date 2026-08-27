# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Declarative pytest marker engine for rocprofiler-systems tests.

This module is the reusable machinery for declaring test markers in a single
place. A :class:`MarkerRegistry` holds the declarations; the actual ``register_*``
calls live in ``conftest.py`` (bound to a single conftest-owned registry) so
functional and non-functional markers sit side by side.

Markers are declared with :meth:`MarkerRegistry.register_marker` (non-functional
labels / behavior) and :meth:`MarkerRegistry.register_functional_marker` (markers
that carry a skip condition); dependencies between markers are declared with
:meth:`MarkerRegistry.add_marker_dependency_if`. Given those declarations, the
registry handles:

* registration (``addinivalue_line``)      -> :meth:`MarkerRegistry.register_markers_with_pytest`
* dependency / optional-marker injection   -> :meth:`MarkerRegistry.resolve_dependencies`
* capability-based skipping                -> :meth:`MarkerRegistry.apply_skip_conditions`
* CTest label export policy                -> :meth:`MarkerRegistry.ctest_labels_for_marker`

A condition callable receives a :class:`MarkerCtx` and returns ``None`` when the
requirement is met (test runs) or a ``str`` skip reason otherwise.
Minimum-version markers can be built with :meth:`MarkerRegistry.min_version`.

The engine keeps no module-level state: ``conftest.py`` owns one registry and
tests can construct their own, so unit tests need not save/restore globals. The
name->:class:`Marker` index and the functional-marker list are built once and
rebuilt only when a new marker is registered. (Each CTest test spawns a fresh
pytest process that re-imports ``conftest`` and rebuilds the registry once, so
there is no cross-process caching to exploit -- building once per process is the
best available.)

CTest Export Policy = None, Name, Args, All (default = Name)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    # Type-only references to the test helper package; no runtime dependency.
    from rocprofsys import GPUInfo, RocprofsysConfig
    from rocprofsys.capabilities import SystemCapabilities


class CTestExport(Enum):
    """How a marker is emitted as a CTest label during ctest generation.

    * ``NONE`` - not emitted as a label
    * ``NAME`` - emit the marker name only (arguments hidden)
    * ``ARGS`` - emit each argument as a label (name hidden)
    * ``ALL``  - emit the name, plus ``name[args]`` when the marker has arguments
    """

    NONE = "none"
    NAME = "name"
    ARGS = "args"
    ALL = "all"


class OnUnmet(Enum):
    """What to do when a dependency's ``when`` condition is not met."""

    SKIP = "skip"  # skip the test with the returned reason
    IGNORE = "ignore"  # add nothing; the test still runs


# A condition returns None (requirement met) or a single skip-reason string.
# Both a functional marker's ``skip_if`` and a dependency's ``when`` use it.
Condition = Callable[["MarkerCtx"], Optional[str]]
SkipCondition = Condition


@dataclass
class MarkerCtx:
    """Plain data passed to marker condition callables.

    Built per marker evaluation so conditions stay small and testable. ``args``
    holds the arguments of the marker currently being evaluated.
    """

    config: "RocprofsysConfig"
    gpu_info: "GPUInfo"
    args: "tuple[Any, ...]" = ()


def caps(ctx: "MarkerCtx") -> "SystemCapabilities":
    """Return the detected system capabilities for ``ctx``."""
    return ctx.config.capabilities


@dataclass
class Marker:
    """A declarative test marker.

    ``skip_if`` runs when the marker is present and returns a skip reason (or
    ``None``); markers without a ``skip_if`` are "non-functional" (labels or
    behavior only). ``arg_hint`` documents the marker's argument in the pytest
    registration line, and ``ctest`` controls how the marker is exported as a
    CTest label. Dependencies between markers are declared separately with
    :meth:`MarkerRegistry.add_marker_dependency_if`.
    """

    name: str
    description: Optional[str] = None
    arg_hint: Optional[str] = None
    skip_if: Optional[SkipCondition] = None
    ctest: CTestExport = CTestExport.NAME


@dataclass
class Dependency:
    """Declares that the ``add`` marker is added when ``trigger`` is present.

    ``when`` (optional) is evaluated with the trigger marker's :class:`MarkerCtx`.
    When it returns a reason string, ``on_unmet`` (an :class:`OnUnmet`) decides
    the outcome: ``SKIP`` skips the test with that reason, ``IGNORE`` adds
    nothing (the test still runs).

    When ``when`` is ``None`` the ``add`` marker is always added if ``trigger``
    is present.
    """

    add: str
    trigger: str
    when: Optional[Condition] = None
    on_unmet: OnUnmet = OnUnmet.SKIP


def registration_line(marker: "Marker") -> str:
    """Return a marker's ``addinivalue_line`` text, e.g. ``name(arg): description``."""
    signature = (
        marker.name if marker.arg_hint is None else f"{marker.name}({marker.arg_hint})"
    )
    description = marker.description or f"label test as {marker.name}"
    return f"{signature}: {description}"


# Factory used by conftest to build a MarkerCtx per item/marker.
CtxFactory = Callable[["pytest.Item", "tuple[Any, ...]"], "MarkerCtx"]

# Pytest built-in / internal markers that never become CTest labels.
PYTEST_BUILTINS = frozenset(
    {"parametrize", "usefixtures", "filterwarnings", "skipif", "skip", "xfail"}
)


def requires(predicate: Callable[["MarkerCtx"], object], reason: str) -> SkipCondition:
    """Build a ``skip_if`` from a predicate and a fixed skip message.

    The test runs when ``predicate(ctx)`` is truthy; otherwise it is skipped with
    ``reason``. This lets simple capability markers declare their check and
    message inline in the ``register_functional_marker(...)`` call instead of
    defining a one-line ``*_reason`` function.
    """

    def skip_if(ctx: "MarkerCtx") -> Optional[str]:
        return None if predicate(ctx) else reason

    return skip_if


class MarkerRegistry:
    """Holds marker + dependency declarations and the engine that consumes them.

    ``conftest.py`` owns a single instance and binds its ``register_*`` methods
    to module-level names so the declarations read naturally. Tests can build an
    independent registry, keeping the engine free of module-level state.
    """

    def __init__(self) -> None:
        self.markers: "List[Marker]" = []
        self.dependencies: "List[Dependency]" = []
        # Built lazily and cached; invalidated whenever a marker is registered.
        self._index: "Optional[Dict[str, Marker]]" = None
        self._functional: "Optional[List[Marker]]" = None

    # -- declaration -------------------------------------------------------
    def register_marker(
        self,
        name: str,
        description: Optional[str] = None,
        *,
        arg_hint: Optional[str] = None,
        ctest: CTestExport = CTestExport.NAME,
    ) -> "Marker":
        """Register a non-functional marker (a label / behavior with no skip check)."""
        return self._add(
            Marker(name=name, description=description, arg_hint=arg_hint, ctest=ctest)
        )

    def register_functional_marker(
        self,
        name: str,
        description: Optional[str] = None,
        *,
        arg_hint: Optional[str] = None,
        skip_if: Optional[SkipCondition] = None,
        ctest: CTestExport = CTestExport.NAME,
    ) -> "Marker":
        """Register a functional marker whose ``skip_if`` is evaluated at collection."""
        return self._add(
            Marker(
                name=name,
                description=description,
                arg_hint=arg_hint,
                skip_if=skip_if,
                ctest=ctest,
            )
        )

    def min_version(
        self,
        name: str,
        get_version: Callable[["MarkerCtx"], Optional[Sequence[int]]],
        *,
        parts: int,
        label: Optional[str] = None,
        not_found_msg: Optional[str] = None,
        too_old_msg: Optional[str] = None,
        description: Optional[str] = None,
        arg_hint: str = "version",
        ctest: CTestExport = CTestExport.NONE,
    ) -> "Marker":
        """Template registering a "minimum version" functional marker.

        ``get_version`` returns the detected version tuple (or ``None``). The
        marker takes a single dotted version-string argument (e.g. ``"7.0"``) and
        skips the test when the detected version is missing or older than
        required.

        Skip messages can be customized with ``label`` (used to derive the
        defaults ``"{label} version not found"`` and
        ``"{label} {found} < required {req}"``) or by passing ``not_found_msg`` /
        ``too_old_msg`` directly. Both message templates may reference ``{found}``
        and ``{req}``. Returns the registered :class:`Marker`.
        """
        reason_not_found = not_found_msg or f"{label} version not found"
        reason_too_old = too_old_msg or f"{label} {{found}} < required {{req}}"

        def skip_if(ctx: "MarkerCtx") -> Optional[str]:
            req = str(ctx.args[0])
            detected = get_version(ctx)
            padded = req.split(".") + ["0"] * parts
            required = tuple(int(p) for p in padded[:parts])
            if detected is not None and tuple(detected) >= required:
                return None
            found = (
                ".".join(map(str, detected)) if detected is not None else "not found"
            )
            template = reason_too_old if detected is not None else reason_not_found
            return template.format(found=found, req=req)

        return self.register_functional_marker(
            name,
            description=description,
            arg_hint=arg_hint,
            skip_if=skip_if,
            ctest=ctest,
        )

    def add_marker_dependency_if(
        self,
        marker: str,
        *,
        when_present: str,
        when: Optional[Condition] = None,
        on_unmet: OnUnmet = OnUnmet.SKIP,
    ) -> None:
        """Declare that ``marker`` is added to a test when ``when_present`` is present.

        Declarative successor to the old ``add_marker_if`` helper: ``marker`` is
        the marker to add and ``when_present`` is the trigger marker. When
        ``when`` is given it is evaluated with the trigger's :class:`MarkerCtx`; a
        returned reason string means the requirement is unmet and ``on_unmet``
        decides whether to skip the test (``OnUnmet.SKIP``) or leave it as-is
        (``OnUnmet.IGNORE``).
        """
        self.dependencies.append(
            Dependency(
                add=marker, trigger=when_present, when=when, on_unmet=on_unmet
            )
        )

    def _add(self, marker: "Marker") -> "Marker":
        self.markers.append(marker)
        self._index = None
        self._functional = None
        return marker

    # -- indexing ----------------------------------------------------------
    def _ensure_index(self) -> None:
        """Build the name->Marker index and functional-marker list once (cached)."""
        if self._index is not None:
            return
        index: "Dict[str, Marker]" = {}
        for marker in self.markers:
            if marker.name in index:
                raise pytest.UsageError(
                    f"duplicate marker declaration: '{marker.name}'"
                )
            index[marker.name] = marker
        self._index = index
        # Preserve declaration order so the surfaced skip reason is stable when a
        # test has several failing functional markers (pytest reports the first).
        self._functional = [m for m in self.markers if m.skip_if is not None]

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        """Check that every dependency references declared markers.

        With ``--strict-markers`` a bad ``add`` name already fails when the marker
        is applied, but a bad ``when_present`` (trigger) would silently never
        match; validating both here surfaces typos at configure time.
        """
        self._ensure_index()
        assert self._index is not None
        for dep in self.dependencies:
            if dep.add not in self._index:
                raise pytest.UsageError(
                    f"marker dependency error: '{dep.add}' (added when "
                    f"'{dep.trigger}' is present) is not a registered marker"
                )
            if dep.trigger not in self._index:
                raise pytest.UsageError(
                    f"marker dependency error: trigger '{dep.trigger}' (for "
                    f"marker '{dep.add}') is not a registered marker"
                )

    # -- pytest wiring -----------------------------------------------------
    def register_markers_with_pytest(self, config: "pytest.Config") -> None:
        """Register every declared marker with pytest and validate dependencies."""
        for marker in self.markers:
            config.addinivalue_line("markers", registration_line(marker))
        self.validate()

    def resolve_dependencies(self, ctx_for: CtxFactory, item: "pytest.Item") -> None:
        """Add dependency markers declared with :meth:`add_marker_dependency_if`.

        For each declared dependency whose trigger marker is present on ``item``,
        the dependent marker is added (or, when a ``when`` condition is unmet, the
        test is skipped or left as-is per ``on_unmet``).

        Runs in all modes, including CTest generation, because added markers
        become CTest labels.
        """
        for dep in self.dependencies:
            trigger = item.get_closest_marker(dep.trigger)
            if trigger is None:
                continue
            if dep.when is None:
                item.add_marker(getattr(pytest.mark, dep.add))
                continue
            reason = dep.when(ctx_for(item, trigger.args))
            if reason is None:
                item.add_marker(getattr(pytest.mark, dep.add))
            elif dep.on_unmet is OnUnmet.SKIP:
                item.add_marker(pytest.mark.skip(reason=reason))
            # OnUnmet.IGNORE: leave the test as-is

    def apply_skip_conditions(self, ctx_for: CtxFactory, item: "pytest.Item") -> None:
        """Evaluate each present functional marker's ``skip_if`` and skip if needed.

        Functional markers are evaluated in declaration order; when several apply,
        pytest surfaces the first skip marker, so declaration order determines the
        reported reason.
        """
        self._ensure_index()
        assert self._functional is not None
        for marker in self._functional:
            present = item.get_closest_marker(marker.name)
            if present is None:
                continue
            assert marker.skip_if is not None
            reason = marker.skip_if(ctx_for(item, present.args))
            if reason:
                item.add_marker(pytest.mark.skip(reason=reason))

    # -- CTest export ------------------------------------------------------
    def ctest_labels_for_marker(self, name: str, args: "tuple[Any, ...]") -> set:
        """Return the CTest labels contributed by a single marker per its policy."""
        if name in PYTEST_BUILTINS:
            return set()
        policy = self.policy_for(name)
        if policy is CTestExport.NONE:
            return set()
        if policy is CTestExport.ARGS:
            return {str(a) for a in args}
        if policy is CTestExport.NAME or not args:
            return {name}
        args_str = ", ".join(str(a) for a in args)
        return {f"{name}[{args_str}]"}

    def policy_for(self, name: str) -> CTestExport:
        """Return the CTest export policy for ``name`` (O(1) via the index)."""
        self._ensure_index()
        assert self._index is not None
        marker = self._index.get(name)
        if marker is not None:
            return marker.ctest
        # Defensive fallback only: conftest enables --strict-markers, so any
        # non-builtin marker reaching here is guaranteed to be registered. If this
        # is ever hit, fall back to the historical default (name, plus name[args]).
        return CTestExport.ALL
