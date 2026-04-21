"""GeminiAdapter — perfxpert MCP install for Gemini CLI (Task 5).

Gemini's config lives in `~/.gemini/settings.json`; the adapter
registers the perfxpert MCP server there and list-appends the
staged `AGENTS.md` cache into `context.fileName`. **Never touches
`GEMINI.md`** — that's the point of Gemini over Claude's rebrand
(I3).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

from perfxpert.cli._backend import _prompt_adapter as pa
from perfxpert.cli._backend.protocol import (
    BackendAdapter,
    ConfigClobber,
    ConsentDenied,
    InstallReport,
    LiveCheckReport,
    PartialInstall,
    Plan,
    UninstallReport,
)
from perfxpert.cli._consent import (
    CONSENT_ASSUME_ENV,
    file_set_hash,
    grant_consent,
    has_consent,
    prompt_consent_interactive,
    revoke_consent,
)


__all__ = ["GeminiAdapter"]


_LOG = logging.getLogger("perfxpert.backend.gemini")


# Reuse the claude-module tool registry. Kept in sync manually so
# neither adapter needs a cross-import of the other.
_KNOWN_TOOLS: tuple[str, ...] = (
    "intent_classify",
    "next_step",
    "report",
    "analyze",
    "classify",
    "workflow_next_step",
    # Agent tools — perfxpert/tools/agents/*.py, one per agent in the
    # Root → Analysis → Recommendation → Correctness + 3 specialists
    # hierarchy. Each is READ_ONLY and routes through build_session so
    # the airgap + provider + fallback-chain semantics apply.
    "agent_root",
    "agent_analysis",
    "agent_recommendation",
    "agent_correctness",
    "agent_compute_specialist",
    "agent_memory_specialist",
    "agent_latency_specialist",
    "agent_diff_specialist",
)


class GeminiAdapter:
    """Adapter for the `gemini` CLI."""

    name: str = "gemini"
    binary_name: str = "gemini"
    install_hint: str = (
        "Install via https://github.com/google-gemini/gemini-cli"
    )
    min_version: str | None = "0.2.0"
    known_schema_versions: tuple[str, ...] = ("1.x",)
    tool_name_template: str = "mcp_perfxpert_{tool}"
    spawn_strategy: Literal["execvpe", "subprocess"] = "execvpe"

    # Gemini settings + per-project staging.
    _SETTINGS_REL = ".gemini/settings.json"
    _PERFXPERT_DIR = ".perfxpert"
    _AGENTS_FILE = "AGENTS.md"

    # ------------------------------------------------------------------
    # check_available
    # ------------------------------------------------------------------

    def check_available(self) -> tuple[bool, str]:
        path = shutil.which(self.binary_name)
        if not path:
            return False, f"{self.binary_name!r} not found on PATH. {self.install_hint}"
        return True, path

    # ------------------------------------------------------------------
    # plan
    # ------------------------------------------------------------------

    def _settings_path(self, home: Path | None = None) -> Path:
        return (home or Path.home()) / self._SETTINGS_REL

    def plan(
        self,
        cwd: Path,
        scope: Literal["project", "user"] = "project",
        dry_run: bool = True,
    ) -> Plan:
        cwd = Path(cwd).expanduser().resolve()
        settings = self._settings_path()
        agents = cwd / self._PERFXPERT_DIR / self._AGENTS_FILE
        actions = [
            f"Register perfxpert MCP in {settings}",
            f"Stage rendered prompt at {agents.relative_to(cwd)}",
            f"List-append {agents.relative_to(cwd)} to context.fileName (preserve user entries)",
            "Write allowedTools gate restriction (event-based lift)",
        ]
        return Plan(
            backend=self.name,
            actions=tuple(actions),
            targets=(settings, agents),
        )

    # ------------------------------------------------------------------
    # install
    # ------------------------------------------------------------------

    def install(
        self,
        cwd: Path,
        scope: Literal["project", "user"] = "project",
        allow_agents_md_append: bool = False,
        dry_run: bool = False,
        quiet: bool = False,
    ) -> InstallReport:
        start = time.monotonic()
        cwd = Path(cwd).expanduser().resolve()
        settings = self._settings_path()
        agents = cwd / self._PERFXPERT_DIR / self._AGENTS_FILE

        # Consent.
        fset = file_set_hash(
            (
                (settings, settings.exists(), False),
                (agents, agents.exists(), False),
            )
        )
        if not has_consent(self.name, cwd, fset):
            plan_lines = [
                f"Register perfxpert MCP in {settings}",
                f"Stage prompt at {agents}",
                "List-append cache path to context.fileName (never touches GEMINI.md)",
            ]
            if not prompt_consent_interactive(self.name, cwd, plan_lines):
                raise ConsentDenied(
                    f"user declined perfxpert install for gemini in {cwd}. "
                    f"Re-run with {CONSENT_ASSUME_ENV}=1 to bypass prompts."
                )

        if dry_run:
            self._log_step(quiet, "[dry-run] would install perfxpert for gemini")
            return InstallReport(
                backend=self.name,
                actions=tuple(self.plan(cwd).actions),
                paths_written=(),
                duration_s=time.monotonic() - start,
            )

        actions: list[str] = []
        written: list[Path] = []

        # Step 1/4: Render + stage AGENTS.md cache.
        self._log_step(quiet, "[1/4] Staging rendered prompt ...")
        rendered = self._render_prompt_for_gemini()
        pa.stage_cache_file(agents, agents, rendered)
        actions.append("staged AGENTS.md cache")
        written.append(agents)

        # Step 2/4: Merge settings.json — mcpServers + context.fileName.
        self._log_step(quiet, "[2/4] Registering perfxpert MCP in ~/.gemini/settings.json ...")
        self._merge_settings(settings, agents, cwd)
        actions.append("merged .gemini/settings.json")
        written.append(settings)

        # Step 3/4: Gate hook — allowedTools restriction.
        self._log_step(quiet, "[3/4] Installing gate hook (allowedTools restriction) ...")
        try:
            from perfxpert.cli._gate_hooks.gemini import GeminiGateHook

            GeminiGateHook().install(cwd)
            actions.append("installed gate hook")
        except Exception as exc:  # GateHookUnsupported OR filesystem error
            _LOG.warning("gemini gate hook install failed: %s", exc)
            actions.append(f"gate hook skipped: {exc}")

        # Step 4/4: Verify.
        if os.environ.get("PERFXPERT_SKIP_LIVE_CHECK", "").strip() not in {
            "1",
            "true",
            "yes",
        }:
            self._log_step(quiet, "[4/4] Verifying perfxpert MCP is live ...")
            report = self.verify_mcp_live(cwd)
            if not report.mcp_healthy:
                raise PartialInstall(
                    f"gemini MCP registered but live-check failed: {report.error}"
                )
        else:
            self._log_step(quiet, "[4/4] SKIPPED (PERFXPERT_SKIP_LIVE_CHECK=1)")

        grant_consent(self.name, cwd, fset)
        return InstallReport(
            backend=self.name,
            actions=tuple(actions),
            paths_written=tuple(written),
            duration_s=time.monotonic() - start,
        )

    # ------------------------------------------------------------------
    # verify_mcp_live
    # ------------------------------------------------------------------

    def verify_mcp_live(
        self, cwd: Path, telemetry: bool = False
    ) -> LiveCheckReport:
        """Best-effort live probe.

        Gemini CLI doesn't guarantee an `mcp list --json` subcommand
        across versions, so the default probe reads the on-disk
        settings.json and confirms the perfxpert entry was written.
        A real telemetry probe via the CLI binary is a future
        enhancement.
        """
        settings = self._settings_path()
        if not settings.is_file():
            return LiveCheckReport(
                backend=self.name,
                mcp_listed=False,
                mcp_healthy=False,
                gate_hook_installed=None,
                error=f"{settings} not present",
            )

        try:
            data = json.loads(settings.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return LiveCheckReport(
                backend=self.name,
                mcp_listed=False,
                mcp_healthy=False,
                error=f"failed to read {settings}: {exc}",
            )

        servers = (data.get("mcpServers") or {})
        perfxpert_entry = servers.get("perfxpert")
        listed = perfxpert_entry is not None
        gate = self._probe_gate_hook_installed(data)
        return LiveCheckReport(
            backend=self.name,
            mcp_listed=listed,
            mcp_healthy=listed,
            observed_tool_names=(),
            gate_hook_installed=gate,
            error=None if listed else "perfxpert entry missing from settings.json",
        )

    def _probe_gate_hook_installed(self, data: dict) -> bool | None:
        if os.environ.get("PERFXPERT_GATE_HOOK", "").strip() == "0":
            return None
        allowed = data.get("allowedTools")
        if not isinstance(allowed, list):
            return False
        return "mcp_perfxpert_*" in allowed

    # ------------------------------------------------------------------
    # spawn
    # ------------------------------------------------------------------

    def spawn(self, argv: list[str], env: dict[str, str], cwd: Path) -> int:
        os.chdir(str(cwd))
        os.execvpe(self.binary_name, [self.binary_name, *argv], env)
        return 127  # pragma: no cover

    # ------------------------------------------------------------------
    # uninstall
    # ------------------------------------------------------------------

    def uninstall(
        self, cwd: Path, scope: Literal["project", "user"] = "project"
    ) -> UninstallReport:
        cwd = Path(cwd).expanduser().resolve()
        settings = self._settings_path()
        agents = cwd / self._PERFXPERT_DIR / self._AGENTS_FILE
        actions: list[str] = []
        removed: list[Path] = []

        if settings.is_file():
            try:
                data = json.loads(settings.read_text())
            except (OSError, json.JSONDecodeError):
                data = None
            if isinstance(data, dict):
                # Remove our MCP entry.
                servers = data.get("mcpServers")
                if isinstance(servers, dict) and "perfxpert" in servers:
                    servers.pop("perfxpert")
                    if not servers:
                        data.pop("mcpServers", None)
                    actions.append("removed mcpServers.perfxpert")
                # Remove our context.fileName entry.
                context = data.get("context")
                if isinstance(context, dict):
                    filenames = context.get("fileName")
                    if isinstance(filenames, list):
                        filtered = [
                            f for f in filenames
                            if self._AGENTS_FILE not in str(f)
                            or self._PERFXPERT_DIR not in str(f)
                        ]
                        if len(filtered) != len(filenames):
                            context["fileName"] = filtered
                            actions.append("list-removed context.fileName entry")
                pa.atomic_write(settings, json.dumps(data, indent=2) + "\n")

            # Also clear the allowedTools gate restriction.
            try:
                from perfxpert.cli._gate_hooks.gemini import GeminiGateHook

                GeminiGateHook().uninstall()
                actions.append("removed gate-hook allowedTools entry")
            except Exception as exc:
                actions.append(f"gate-hook uninstall failed: {exc}")

        if agents.exists():
            try:
                agents.unlink()
                removed.append(agents)
                actions.append(f"removed {agents}")
            except OSError as exc:
                actions.append(f"failed to remove {agents}: {exc}")

        perfxpert_dir = cwd / self._PERFXPERT_DIR
        if perfxpert_dir.is_dir():
            try:
                perfxpert_dir.rmdir()
            except OSError:
                pass

        revoke_consent(self.name, cwd)
        actions.append("revoked consent")

        return UninstallReport(
            backend=self.name,
            actions=tuple(actions),
            paths_removed=tuple(removed),
            skipped_due_to_drift=(),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_step(self, quiet: bool, msg: str) -> None:
        if not quiet:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        _LOG.debug(msg)

    def _render_prompt_for_gemini(self) -> str:
        bundled_source = _find_bundled_agents_md()
        if bundled_source is None:
            source = (
                "Always call `perfxpert_intent_classify` first. After that "
                "tool returns, any other tool is permitted.\n"
            )
            return pa.render_prompt(
                source,
                backend=self.name,
                tool_name_template=self.tool_name_template,
                known_tools=_KNOWN_TOOLS,
                reject_language=True,
            )
        return pa.render_prompt(
            bundled_source,
            backend=self.name,
            tool_name_template=self.tool_name_template,
            known_tools=_KNOWN_TOOLS,
            reject_language=True,
        )

    def _merge_settings(
        self, settings: Path, agents: Path, cwd: Path
    ) -> None:
        """Atomically merge mcpServers.perfxpert + context.fileName.

        Preserves every existing key. Raises `ConfigClobber` if a
        different `perfxpert` entry is already present.
        """
        data: dict = {}
        if settings.is_file():
            try:
                data = json.loads(settings.read_text())
            except json.JSONDecodeError as exc:
                raise PartialInstall(
                    f"{settings} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise PartialInstall(
                    f"{settings} top-level must be an object"
                )

        # mcpServers.perfxpert.
        servers = data.setdefault("mcpServers", {})
        if not isinstance(servers, dict):
            raise PartialInstall(
                f"{settings}['mcpServers'] must be an object"
            )
        existing = servers.get("perfxpert")
        if existing and existing.get("command") not in (None, "perfxpert-mcp"):
            raise ConfigClobber(
                f"{settings} already has a perfxpert MCP entry with "
                f"command {existing.get('command')!r}; refuse to overwrite."
            )
        servers["perfxpert"] = {"command": "perfxpert-mcp", "args": []}

        # context.fileName (list-append).
        context = data.setdefault("context", {})
        if not isinstance(context, dict):
            raise PartialInstall(
                f"{settings}['context'] must be an object"
            )
        existing_files = context.setdefault("fileName", [])
        if not isinstance(existing_files, list):
            raise PartialInstall(
                f"{settings}['context']['fileName'] must be a list"
            )
        new_entry = str(agents)
        if new_entry not in existing_files:
            existing_files.append(new_entry)

        pa.atomic_write(settings, json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Protocol conformance.
# ---------------------------------------------------------------------------


assert isinstance(GeminiAdapter(), BackendAdapter)


def _find_bundled_agents_md() -> Path | None:
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "_bundled" / "opencode_config" / "AGENTS.md",
        here.parent.parent.parent / "_bundled" / "AGENTS.md",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None
