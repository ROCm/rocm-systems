"""Tests for `perfxpert.cli._backend.protocol` (Task 1).

Asserts:

* A no-op adapter that implements every Protocol member satisfies
  `runtime_checkable` via `isinstance`.
* All report + plan dataclasses are frozen (assignment raises).
* Every exception in the taxonomy is importable and subclasses
  `PerfxpertBackendError`.
* `tool_name_template` and `spawn_strategy` are exposed on the
  Protocol surface (B1 / I1).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import pytest

from perfxpert.cli._backend import (
    BackendAdapter,
    BackendAdapterError,
    BackendNotFound,
    ConfigClobber,
    ConsentDenied,
    GateHookUnsupported,
    InstallReport,
    LiveCheckReport,
    PartialInstall,
    PerfxpertBackendError,
    Plan,
    SchemaUnknown,
    TrustRequired,
    UninstallReport,
    VersionTooOld,
)


class _NoopAdapter:
    """Minimal adapter that satisfies `BackendAdapter` without doing anything."""

    name = "noop"
    binary_name = "noop"
    install_hint = "(no-op adapter)"
    min_version: str | None = None
    known_schema_versions: tuple[str, ...] = ("1.x",)
    tool_name_template = "mcp__perfxpert__{tool}"
    spawn_strategy: Literal["execvpe", "subprocess"] = "execvpe"

    def check_available(self) -> tuple[bool, str]:
        return True, "/bin/true"

    def plan(self, cwd, scope="project", dry_run=True):
        return Plan(backend=self.name)

    def install(
        self,
        cwd,
        scope="project",
        allow_agents_md_append=False,
        dry_run=False,
        quiet=False,
    ):
        return InstallReport(backend=self.name)

    def verify_mcp_live(self, cwd, telemetry=False):
        return LiveCheckReport(backend=self.name, mcp_listed=True, mcp_healthy=True)

    def uninstall(self, cwd, scope="project"):
        return UninstallReport(backend=self.name)

    def spawn(self, argv, env, cwd):
        return 0


def test_noop_adapter_satisfies_backend_protocol() -> None:
    """runtime_checkable Protocol should accept the no-op implementation."""
    assert isinstance(_NoopAdapter(), BackendAdapter)


def test_protocol_declares_tool_name_template_and_spawn_strategy() -> None:
    """B1 + I1 — tool-name rewrite template and spawn strategy on the Protocol."""
    # Accessing via the class-level dict avoids Protocol instance semantics.
    ann = BackendAdapter.__annotations__
    assert "tool_name_template" in ann
    assert "spawn_strategy" in ann


def test_install_report_is_frozen() -> None:
    r = InstallReport(backend="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.backend = "y"  # type: ignore[misc]


def test_uninstall_report_is_frozen() -> None:
    r = UninstallReport(backend="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.backend = "y"  # type: ignore[misc]


def test_plan_is_frozen() -> None:
    p = Plan(backend="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.backend = "y"  # type: ignore[misc]


def test_live_check_report_is_frozen() -> None:
    r = LiveCheckReport(backend="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.backend = "y"  # type: ignore[misc]


def test_live_check_report_gate_hook_tristate() -> None:
    """I-N1: gate_hook_installed supports None / False / True."""
    none_r = LiveCheckReport(backend="x")
    false_r = LiveCheckReport(backend="x", gate_hook_installed=False)
    true_r = LiveCheckReport(backend="x", gate_hook_installed=True)
    assert none_r.gate_hook_installed is None
    assert false_r.gate_hook_installed is False
    assert true_r.gate_hook_installed is True


def test_exception_taxonomy_all_subclass_base() -> None:
    base = PerfxpertBackendError
    for cls in (
        BackendNotFound,
        VersionTooOld,
        ConfigClobber,
        ConsentDenied,
        PartialInstall,
        SchemaUnknown,
        TrustRequired,
        GateHookUnsupported,
    ):
        assert issubclass(cls, base), cls


def test_backend_adapter_error_is_alias() -> None:
    """The plan uses both names interchangeably."""
    assert BackendAdapterError is PerfxpertBackendError


def test_adapter_report_paths_accept_paths() -> None:
    p = Path("/tmp/x")
    ir = InstallReport(backend="x", paths_written=(p,))
    ur = UninstallReport(backend="x", paths_removed=(p,))
    assert ir.paths_written == (p,)
    assert ur.paths_removed == (p,)


def test_gate_hook_unsupported_is_distinct_class() -> None:
    """I-N1: ensure the taxonomy distinguishes documented-known-limit."""
    with pytest.raises(GateHookUnsupported):
        raise GateHookUnsupported("test")
