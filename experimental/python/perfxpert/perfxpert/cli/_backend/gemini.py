"""GeminiAdapter — perfxpert MCP install for Gemini CLI (Task 5).

Gemini supports project-scoped configuration in `<cwd>/.gemini/settings.json`.
The adapter registers the perfxpert MCP server there, list-appends the
staged `AGENTS.md` cache into `context.fileName`, and installs project-local
Gemini hooks for the event-based tool-priority gate. **Never touches
`GEMINI.md`**.
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
    "arch_lookup_peaks",
    "counters_lookup_info",
    "regression_compare_runs",
    "sol_sanity_check",
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

    # Gemini workspace settings + per-project staging.
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

    def _settings_path(self, cwd: Path) -> Path:
        cwd = Path(cwd).expanduser().resolve()
        return cwd / self._SETTINGS_REL

    def plan(
        self,
        cwd: Path,
        scope: Literal["project", "user"] = "project",
        dry_run: bool = True,
    ) -> Plan:
        cwd = Path(cwd).expanduser().resolve()
        settings = self._settings_path(cwd)
        agents = cwd / self._PERFXPERT_DIR / self._AGENTS_FILE
        actions = [
            f"Register perfxpert MCP in {settings.relative_to(cwd)}",
            f"Stage rendered prompt at {agents.relative_to(cwd)}",
            f"List-append {agents.relative_to(cwd)} to context.fileName (preserve user entries)",
            "Install Gemini BeforeTool/AfterTool gate hooks in .gemini/settings.json",
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
        settings = self._settings_path(cwd)
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
                "Install Gemini BeforeTool/AfterTool gate hooks",
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

        # Step 1/5: Gate hook — native Gemini hooks. This MUST succeed before
        # any MCP or prompt writes so Gemini never comes up ungated.
        self._log_step(
            quiet,
            "[1/5] Installing gate hook (BeforeTool/AfterTool hooks) ...",
        )
        from perfxpert.cli._gate_hooks.gemini import GeminiGateHook

        GeminiGateHook().install(cwd)
        actions.append("installed gate hook")

        try:
            # Step 2/5: Render + stage AGENTS.md cache.
            self._log_step(quiet, "[2/5] Staging rendered prompt ...")
            rendered = self._render_prompt_for_gemini()
            pa.stage_cache_file(agents, agents, rendered)
            actions.append("staged AGENTS.md cache")
            written.append(agents)

            # Step 3/5: Merge workspace settings.json — mcpServers + context.fileName.
            self._log_step(
                quiet,
                "[3/5] Registering perfxpert MCP in .gemini/settings.json ...",
            )
            self._merge_settings(settings, agents, cwd)
            actions.append("merged .gemini/settings.json")
            written.append(settings)
        except Exception:
            # Roll back project-local writes if we fail after installing the
            # Gemini hook surface. Legacy user-scope migration has not run yet.
            self._rollback_project_state(cwd)
            raise

        # Step 4/5: Verify.
        if os.environ.get("PERFXPERT_SKIP_LIVE_CHECK", "").strip() not in {
            "1",
            "true",
            "yes",
        }:
            self._log_step(quiet, "[4/5] Verifying perfxpert MCP is live ...")
            report = self.verify_mcp_live(cwd)
            if not report.mcp_healthy:
                raise PartialInstall(
                    f"gemini MCP registered but live-check failed: {report.error}"
                )
            if report.gate_hook_installed is not True:
                raise PartialInstall(
                    "gemini gate hook is missing after install; refusing to "
                    "leave the backend partially installed"
                )
        else:
            self._log_step(quiet, "[4/5] SKIPPED (PERFXPERT_SKIP_LIVE_CHECK=1)")
            report = self.verify_mcp_live(cwd)
            if report.gate_hook_installed is not True:
                raise PartialInstall(
                    "gemini gate hook is missing after install; refusing to "
                    "leave the backend partially installed"
                )

        # Step 5/5: Done. Gemini installs are project-local only.
        self._log_step(quiet, "[5/5] Project-local Gemini install complete")

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
        """Probe the workspace MCP registration and Gemini's own view of it."""
        cwd = Path(cwd).expanduser().resolve()
        settings = self._settings_path(cwd)
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

        servers = data.get("mcpServers") or {}
        perfxpert_entry = servers.get("perfxpert")
        listed = (
            isinstance(perfxpert_entry, dict)
            and perfxpert_entry.get("command") == "perfxpert-mcp"
        )
        gate = self._probe_gate_hook_installed(cwd, data)
        if not listed:
            return LiveCheckReport(
                backend=self.name,
                mcp_listed=False,
                mcp_healthy=False,
                gate_hook_installed=gate,
                error="perfxpert entry missing or malformed in .gemini/settings.json",
            )

        probe_error: str | None = None
        try:
            line = pa.retry_mcp_handshake(
                lambda: self._probe_mcp_list_line(cwd),
            )
            healthy = "Connected" in line
            listed = True
            if not healthy:
                probe_error = f"gemini mcp list reported: {line}"
        except Exception as exc:
            healthy = True
            listed = False
            probe_error = (
                "advisory: local .gemini/settings.json is valid, but "
                f"gemini mcp list could not confirm project MCP visibility: {exc}"
            )
        return LiveCheckReport(
            backend=self.name,
            mcp_listed=listed,
            mcp_healthy=healthy,
            observed_tool_names=(),
            gate_hook_installed=gate,
            error=probe_error,
        )

    def _probe_gate_hook_installed(self, cwd: Path, data: dict) -> bool | None:
        if os.environ.get("PERFXPERT_GATE_HOOK", "").strip() == "0":
            return None
        hooks = data.get("hooks")
        if not isinstance(hooks, dict):
            return False
        serialized = json.dumps(hooks)
        if "perfxpert-gate" in serialized:
            return True
        return all(
            path.exists()
            for path in (
                cwd / ".gemini" / "hooks" / "perfxpert-gate.sh",
                cwd / ".gemini" / "hooks" / "perfxpert-gate-post.sh",
            )
        )

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
        settings = self._settings_path(cwd)
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
                        filtered = [f for f in filenames if str(f) != str(agents)]
                        if len(filtered) != len(filenames):
                            context["fileName"] = filtered
                            actions.append("list-removed context.fileName entry")
                pa.atomic_write(settings, json.dumps(data, indent=2) + "\n")

            # Also clear the Gemini hook entries.
            try:
                from perfxpert.cli._gate_hooks.gemini import GeminiGateHook

                if GeminiGateHook().uninstall(cwd):
                    actions.append("removed gate-hook hook entries")
                else:
                    actions.append("gate-hook uninstall skipped due to drift")
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

    def _rollback_project_state(self, cwd: Path) -> None:
        """Best-effort rollback for a failed project-local Gemini install."""
        try:
            self.uninstall(cwd)
        except Exception as exc:  # pragma: no cover - defensive cleanup
            _LOG.warning("gemini rollback failed: %s", exc)

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
        self, settings: Path, agents: Path, _cwd: Path
    ) -> None:
        """Atomically merge workspace mcpServers.perfxpert + context.fileName.

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
        if existing is not None and not isinstance(existing, dict):
            raise ConfigClobber(
                f"{settings} already has a non-object perfxpert MCP entry; "
                "refuse to overwrite."
            )
        if existing and existing.get("command") not in (None, "perfxpert-mcp"):
            raise ConfigClobber(
                f"{settings} already has a perfxpert MCP entry with "
                f"command {existing.get('command')!r}; refuse to overwrite."
            )
        merged_entry = dict(existing or {})
        merged_entry.setdefault("command", "perfxpert-mcp")
        if merged_entry["command"] != "perfxpert-mcp":
            raise ConfigClobber(
                f"{settings} already has a perfxpert MCP entry with "
                f"command {merged_entry['command']!r}; refuse to overwrite."
            )
        merged_entry.setdefault("args", [])
        servers["perfxpert"] = merged_entry

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

    def _probe_mcp_list_line(self, cwd: Path) -> str:
        result = subprocess.run(
            [self.binary_name, "mcp", "list"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(detail)
        for line in result.stdout.splitlines():
            if "perfxpert" in line:
                return line.strip()
        raise RuntimeError("perfxpert not listed by gemini mcp list")


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
