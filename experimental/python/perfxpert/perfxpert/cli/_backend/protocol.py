"""Backend adapter protocol + report dataclasses + error taxonomy.

Task 1 of the multi-backend plan. Defines the contract every backend
adapter (Claude / Codex / Gemini) must satisfy. The Protocol is locked
from day one (cycle-2 I2): every method and kwarg that tasks 4b/4c/5/10
consume is present here so those tasks never need to mutate the
Protocol retroactively.

The `tool_name_template` class attribute (B1) and `spawn_strategy`
literal (I1) are part of the public Protocol so `_prompt_adapter`
(Task 4a) and the dispatcher (Task 6) can parameterize themselves
without type-narrowing per adapter.

The report dataclasses (`InstallReport`, `UninstallReport`, `Plan`,
`LiveCheckReport`) are frozen so the dispatcher cannot mutate an
adapter's return value after it has been logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Error taxonomy (cycle-2 Task 1 / I-N1 added GateHookUnsupported)
# ---------------------------------------------------------------------------


class PerfxpertBackendError(Exception):
    """Base class for every backend-adapter exception.

    Also aliased as `BackendAdapterError` per the cycle-2 plan Task 1.
    """


# Alias kept for stylistic parity with the plan document.
BackendAdapterError = PerfxpertBackendError


class BackendNotFound(PerfxpertBackendError):
    """Backend binary is not installed / not on PATH."""


class VersionTooOld(PerfxpertBackendError):
    """Backend binary is present but below `min_version`."""


class ConfigClobber(PerfxpertBackendError):
    """An existing config entry conflicts and `--force` was not set."""


class ConsentDenied(PerfxpertBackendError):
    """User declined the consent prompt; no writes were attempted."""


class PartialInstall(PerfxpertBackendError):
    """One or more install steps succeeded; at least one failed."""


class SchemaUnknown(PerfxpertBackendError):
    """Config schema version is not in `known_schema_versions`."""


class TrustRequired(PerfxpertBackendError):
    """Codex project is not trusted; cannot register project-scope config."""


class GateHookUnsupported(PerfxpertBackendError):
    """Backend exposes no suitable pre-tool-call surface (I-N1).

    Raised by a gate-hook installer BEFORE MCP registration runs, so
    no partial install state is left behind. `verify_mcp_live()`
    treats the resulting `LiveCheckReport.gate_hook_installed=False`
    as a documented-known-limit, not as a failure.
    """


# ---------------------------------------------------------------------------
# Report + Plan dataclasses (frozen — returned values are immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstallReport:
    """Result of a successful `BackendAdapter.install()` call."""

    backend: str
    actions: tuple[str, ...] = field(default_factory=tuple)
    paths_written: tuple[Path, ...] = field(default_factory=tuple)
    duration_s: float = 0.0


@dataclass(frozen=True)
class UninstallReport:
    """Result of a successful `BackendAdapter.uninstall()` call."""

    backend: str
    actions: tuple[str, ...] = field(default_factory=tuple)
    paths_removed: tuple[Path, ...] = field(default_factory=tuple)
    skipped_due_to_drift: tuple[Path, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Plan:
    """Dry-run plan returned by `BackendAdapter.plan()` before any writes."""

    backend: str
    actions: tuple[str, ...] = field(default_factory=tuple)
    targets: tuple[Path, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LiveCheckReport:
    """Result of `BackendAdapter.verify_mcp_live()`.

    `gate_hook_installed` encodes three states (I-N1):

    * `None` — gate-hook probe skipped (e.g. `PERFXPERT_GATE_HOOK=0`).
    * `False` — surface unsupported / documented-known-limit (not a
      failure).
    * `True` — installed AND effective (verified via gate probe).
    """

    backend: str
    mcp_listed: bool = False
    mcp_healthy: bool = False
    observed_tool_names: tuple[str, ...] = field(default_factory=tuple)
    gate_hook_installed: bool | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# BackendAdapter Protocol (full lifecycle, day-1 locked — I2)
# ---------------------------------------------------------------------------


@runtime_checkable
class BackendAdapter(Protocol):
    """Every backend adapter implements this Protocol.

    Class attributes (the adapter declares them; they are part of the
    public Protocol surface):

    * `name`: short identifier used in logs + dispatcher routing.
    * `binary_name`: name passed to `shutil.which`.
    * `install_hint`: one-line user-facing "how do I install this?"
      message, shown when `check_available()` returns
      `(False, ...)`.
    * `min_version`: optional SemVer floor; `None` disables the check.
    * `known_schema_versions`: tuple of config schema versions this
      adapter can parse without raising `SchemaUnknown`.
    * `tool_name_template` (B1): f-string-style template for
      rendering `perfxpert_<X>` tool names into the backend's
      wire format. Exactly one `{tool}` placeholder.
    * `spawn_strategy` (I1): either `"execvpe"` (replace the Python
      process with the backend TUI — default for interactive backends)
      or `"subprocess"` (shell out and wait).
    """

    name: str
    binary_name: str
    install_hint: str
    min_version: str | None
    known_schema_versions: tuple[str, ...]
    tool_name_template: str
    spawn_strategy: Literal["execvpe", "subprocess"]

    # Lifecycle --------------------------------------------------------------

    def check_available(self) -> tuple[bool, str]:
        """Return (found, path-or-reason).

        If the binary is found AND at/above `min_version`, returns
        `(True, /absolute/path/to/binary)`. Otherwise returns
        `(False, "human-readable reason")`. Callers should surface
        `install_hint` alongside the reason.
        """
        ...

    def plan(
        self,
        cwd: Path,
        scope: Literal["project", "user"] = "project",
        dry_run: bool = True,
    ) -> Plan:
        """Return a `Plan` describing every file + entry that `install()`
        would touch. Pure (no writes), safe to call on every invocation.
        """
        ...

    def install(
        self,
        cwd: Path,
        scope: Literal["project", "user"] = "project",
        allow_agents_md_append: bool = False,
        dry_run: bool = False,
        quiet: bool = False,
    ) -> InstallReport:
        """Idempotently register perfxpert MCP + stage prompt + install
        gate hook. On failure raises one of the taxonomy exceptions
        (never returns a partial report). A successful return MUST
        leave the user's config in a working state (all steps
        complete OR none performed).
        """
        ...

    def verify_mcp_live(
        self,
        cwd: Path,
        telemetry: bool = False,
    ) -> LiveCheckReport:
        """Spawn the backend non-interactively, list MCP tools,
        compare to the rendered prompt's tool-name template. On
        success returns a populated `LiveCheckReport`; on failure
        raises `PartialInstall` with diagnostic context.
        """
        ...

    def uninstall(
        self,
        cwd: Path,
        scope: Literal["project", "user"] = "project",
    ) -> UninstallReport:
        """Reverse `install()`. Refuses on marker-block drift
        (content edited inside our managed block) and records those
        paths in `UninstallReport.skipped_due_to_drift`.
        """
        ...

    def spawn(
        self,
        argv: list[str],
        env: dict[str, str],
        cwd: Path,
    ) -> int:
        """Dispatch the backend binary per `spawn_strategy`.

        * `"execvpe"`: replaces the Python process via `os.execvpe`;
          the returned int is notional — control never returns on
          success.
        * `"subprocess"`: runs the binary, waits, returns the exit
          code.
        """
        ...
