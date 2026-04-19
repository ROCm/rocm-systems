"""Gemini gate hook — `allowedTools` restriction + runtime state (Task 4.6).

Gemini CLI doesn't expose a native PreToolUse hook, so we use the
`allowedTools` list in `~/.gemini/settings.json` as a static policy
surface and flip it to "unrestricted" via a sidecar state file once
`intent_classify` is observed.

Session-state location (I-N3):
`~/.gemini/runtime/perfxpert-gate-<session_id>.json`. A NEW session
starts with the gate engaged (different session_id; fresh state
file required).

Install writes:

1. `allowedTools: ["mcp_perfxpert_*"]` into `~/.gemini/settings.json`
   (gate engaged — only perfxpert tools may fire at session start).
2. A companion config file
   `~/.gemini/runtime/perfxpert-gate-config.json` describing the
   lift trigger (event: `mcp_perfxpert_intent_classify` returned).

Because Gemini's `allowedTools` is static per-session, the "lift"
actually happens when a wrapper script detects the state file and
rewrites `allowedTools` for the remainder of the session. That
wrapper lives in the launcher (`perfxpert-code gemini ...`) and
polls the state file every few seconds.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from perfxpert.cli._backend import _prompt_adapter as pa
from perfxpert.cli._backend.protocol import GateHookUnsupported
from perfxpert.cli._gate_hooks import (
    GATE_HOOK_DISABLED_ENV,
    GATE_STATE_LIFTED_SENTINEL,
)


__all__ = [
    "GeminiGateHook",
    "GeminiGateInstallResult",
    "evaluate_gate_state",
]


@dataclass(frozen=True)
class GeminiGateInstallResult:
    settings_path: Path
    runtime_config_path: Path


class GeminiGateHook:
    """Installer for the Gemini `allowedTools` gate."""

    _RESTRICTED_ALLOWEDTOOLS = ["mcp_perfxpert_*"]

    def install(
        self,
        cwd: Path,
        *,
        classify_tool: str = "mcp_perfxpert_intent_classify",
        home: Path | None = None,
    ) -> GeminiGateInstallResult:
        """Write `allowedTools` + runtime config atomically.

        Raises `GateHookUnsupported` if the user explicitly disabled
        the gate (`PERFXPERT_GATE_HOOK=0`) or if the settings file
        cannot be safely merged.
        """
        if os.environ.get(GATE_HOOK_DISABLED_ENV, "").strip() == "0":
            raise GateHookUnsupported(
                f"{GATE_HOOK_DISABLED_ENV}=0 — Gemini gate hook install skipped"
            )

        home = home or Path.home()
        settings = home / ".gemini" / "settings.json"
        runtime_cfg = home / ".gemini" / "runtime" / "perfxpert-gate-config.json"

        # Merge settings.json.
        data: dict[str, Any] = {}
        if settings.is_file():
            try:
                data = json.loads(settings.read_text())
            except json.JSONDecodeError as exc:
                raise GateHookUnsupported(
                    f"{settings} is not valid JSON: {exc}"
                )
            if not isinstance(data, dict):
                raise GateHookUnsupported(
                    f"{settings} top-level must be an object"
                )
        # Preserve any existing allowedTools the user had; merge in
        # our required entries.
        existing = data.get("allowedTools") or []
        if not isinstance(existing, list):
            raise GateHookUnsupported(
                f"{settings}['allowedTools'] must be a list"
            )
        merged = list(self._RESTRICTED_ALLOWEDTOOLS)
        for item in existing:
            if item not in merged:
                merged.append(item)
        data["allowedTools"] = merged

        pa.atomic_write(settings, json.dumps(data, indent=2) + "\n")

        # Write companion runtime config.
        runtime_payload = {
            "classify_tool": classify_tool,
            "sentinel": GATE_STATE_LIFTED_SENTINEL,
            "description": (
                "perfxpert-code Gemini gate: launcher polls "
                f"~/.gemini/runtime/perfxpert-gate-<session_id>.json; when a "
                "file containing the sentinel appears, the launcher clears "
                "allowedTools for the remainder of the session."
            ),
        }
        pa.atomic_write(runtime_cfg, json.dumps(runtime_payload, indent=2) + "\n")

        return GeminiGateInstallResult(
            settings_path=settings, runtime_config_path=runtime_cfg
        )

    def uninstall(self, *, home: Path | None = None) -> None:
        home = home or Path.home()
        settings = home / ".gemini" / "settings.json"
        runtime_cfg = home / ".gemini" / "runtime" / "perfxpert-gate-config.json"
        if settings.is_file():
            try:
                data = json.loads(settings.read_text())
            except json.JSONDecodeError:
                return
            allowed = data.get("allowedTools") or []
            data["allowedTools"] = [
                t for t in allowed if t != "mcp_perfxpert_*"
            ]
            if not data["allowedTools"]:
                data.pop("allowedTools")
            pa.atomic_write(settings, json.dumps(data, indent=2) + "\n")
        if runtime_cfg.is_file():
            try:
                runtime_cfg.unlink()
            except OSError:
                pass


def evaluate_gate_state(
    tool_name: str,
    *,
    intent_classify_observed: bool,
    classify_tool: str = "mcp_perfxpert_intent_classify",
) -> dict[str, Any]:
    """Mirror of the allowedTools logic for unit testing.

    Returns `{"allowed": True}` / `{"allowed": False, "reason": ...}`
    per the event-based rule (B-N3).
    """
    if tool_name.startswith("mcp_perfxpert_"):
        return {"allowed": True}
    if intent_classify_observed:
        return {"allowed": True}
    return {
        "allowed": False,
        "reason": (
            f"Gemini gate engaged: call {classify_tool} first. "
            "After it returns, any tool is permitted."
        ),
    }
