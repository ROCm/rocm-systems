"""perfxpert.cli._backend — backend-adapter machinery for perfxpert-code.

Exports the `BackendAdapter` Protocol, the report + plan dataclasses, and
the full error taxonomy. Individual adapters (Claude, Gemini, Codex) live
in sibling modules and each implement the Protocol.

This package is internal to `perfxpert.cli`; consumers outside the CLI
should not depend on symbols here directly.
"""

from __future__ import annotations

from perfxpert.cli._backend.protocol import (
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

__all__ = [
    "BackendAdapter",
    "BackendAdapterError",
    "BackendNotFound",
    "ConfigClobber",
    "ConsentDenied",
    "GateHookUnsupported",
    "InstallReport",
    "LiveCheckReport",
    "PartialInstall",
    "PerfxpertBackendError",
    "Plan",
    "SchemaUnknown",
    "TrustRequired",
    "UninstallReport",
    "VersionTooOld",
]
