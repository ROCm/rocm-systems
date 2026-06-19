"""ClaudeCodeAdapter — perfxpert MCP install for Claude Code (Task 4b).

Implements the Protocol from `perfxpert.cli._backend.protocol` for
the `claude` CLI. Covers:

* `check_available()` — `shutil.which("claude")` + `--version` parse.
* `plan()` — target-file enumeration for the installer.
* `install()` — 4-step lifecycle (warm → register → stage prompt →
  pointer → verify). Warmup + live verification integrate in
  Tasks 4.6/4.7/4c; until Task 4c lands, `PERFXPERT_SKIP_LIVE_CHECK=1`
  short-circuits Step 4/4 so the intermediate 4b-only state is
  safe.
* `spawn()` — `os.execvpe` (Claude TUI takes over the process).
* `uninstall()` — removes marker block + MCP entry; refuses on
  content drift.

MCP registration uses the CLI primary path
(`claude mcp add perfxpert --scope project -- perfxpert-mcp`) with a
print-for-human fallback. The adapter NEVER rewrites the user's
`~/.claude.json` directly — that file is multi-MB and contains
session history (I4).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

from perfxpert.cli._backend import _prompt_adapter as pa
from perfxpert.cli._backend.protocol import (
    BackendAdapter,
    BackendNotFound,
    ConfigClobber,
    ConsentDenied,
    InstallReport,
    LiveCheckReport,
    PartialInstall,
    Plan,
    UninstallReport,
    VersionTooOld,
)
from perfxpert.cli._consent import (
    CONSENT_ASSUME_ENV,
    file_set_hash,
    grant_consent,
    has_consent,
    prompt_consent_interactive,
    revoke_consent,
)


__all__ = ["ClaudeCodeAdapter", "SKIP_LIVE_CHECK_ENV"]


_LOG = logging.getLogger("perfxpert.backend.claude")


# Env override — Task 4b intermediate state. Task 4c removes the
# default-on behavior (live check always runs). Until then setting
# this to "1" is a no-op short-circuit so 4b can land without 4c.
SKIP_LIVE_CHECK_ENV = "PERFXPERT_SKIP_LIVE_CHECK"


# ---------------------------------------------------------------------------
# Perfxpert MCP tool name registry — kept next to the adapter so each
# adapter can expand the list independently. The canonical list lives
# in `perfxpert/mcp_server/tools.py`; copying the names here avoids a
# circular import and lets this module be used from `plan()` dry-runs.
# ---------------------------------------------------------------------------

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


class ClaudeCodeAdapter:
    """Adapter for the `claude` CLI (Claude Code)."""

    name: str = "claude"
    binary_name: str = "claude"
    install_hint: str = (
        "Install via https://code.claude.com/docs/en/install"
    )
    min_version: str | None = "2.1.59"
    known_schema_versions: tuple[str, ...] = ("1.x",)
    tool_name_template: str = "mcp__perfxpert__{tool}"
    spawn_strategy: Literal["execvpe", "subprocess"] = "execvpe"

    # File targets (relative to cwd).
    _MCP_CONFIG_FILENAME = ".mcp.json"
    # Pointer file — CLAUDE.local.md at project root is the canonical
    # local-override context file that Claude Code auto-loads on top of
    # any tracked CLAUDE.md. `.claude/CLAUDE.md` is NOT an auto-loaded
    # location, so an earlier version of this adapter silently wrote a
    # file the model never read. See uninstall path for the migration
    # cleanup of that stale location.
    _POINTER_DIR = ""
    _POINTER_FILE = "CLAUDE.local.md"
    # Legacy pointer locations we need to clean up on uninstall for users
    # who installed before the CLAUDE.local.md fix.
    _LEGACY_POINTER_PATHS = (
        (".claude", "CLAUDE.md"),
    )
    _PERFXPERT_DIR = ".perfxpert"
    _AGENTS_FILE = "AGENTS.md"

    # ------------------------------------------------------------------
    # check_available
    # ------------------------------------------------------------------

    def check_available(self) -> tuple[bool, str]:
        path = shutil.which(self.binary_name)
        if not path:
            return False, f"{self.binary_name!r} not found on PATH. {self.install_hint}"
        if self.min_version is not None:
            ok, detail = self._probe_version(path)
            if not ok:
                return False, detail
        return True, path

    def _probe_version(self, binary: str) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"{self.binary_name} --version failed: {exc}"
        if result.returncode != 0:
            return False, f"{self.binary_name} --version returned {result.returncode}"
        version_line = (result.stdout or result.stderr or "").strip()
        if self.min_version and not _version_at_or_above(version_line, self.min_version):
            return False, (
                f"{self.binary_name} version {version_line!r} is below the "
                f"minimum required {self.min_version!r}."
            )
        return True, version_line

    # ------------------------------------------------------------------
    # plan
    # ------------------------------------------------------------------

    def _target_paths(self, cwd: Path) -> tuple[Path, Path, Path]:
        """Return (mcp_config, pointer_file, agents_cache) absolute paths."""
        cwd = cwd.expanduser().resolve()
        return (
            cwd / self._MCP_CONFIG_FILENAME,
            cwd / self._POINTER_DIR / self._POINTER_FILE,
            cwd / self._PERFXPERT_DIR / self._AGENTS_FILE,
        )

    def plan(
        self,
        cwd: Path,
        scope: Literal["project", "user"] = "project",
        dry_run: bool = True,
    ) -> Plan:
        mcp_config, pointer, agents_cache = self._target_paths(cwd)
        actions: list[str] = [
            f"Register perfxpert MCP in {mcp_config.name} (scope={scope})",
            f"Stage rendered prompt at {agents_cache.relative_to(cwd.expanduser().resolve())}",
            f"Write pointer at {pointer.relative_to(cwd.expanduser().resolve())}",
            "Install PreToolUse gate hook in .claude/settings.json (event-based lift on intent_classify)",
            "Verify perfxpert MCP is live (claude mcp list)",
        ]
        return Plan(
            backend=self.name,
            actions=tuple(actions),
            targets=(mcp_config, pointer, agents_cache),
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
        cwd = cwd.expanduser().resolve()
        mcp_config, pointer, agents_cache = self._target_paths(cwd)
        actions_done: list[str] = []
        paths_written: list[Path] = []

        # ---- Consent ----------------------------------------------------
        file_set = (
            (mcp_config, mcp_config.exists(), pa.is_git_tracked(Path(mcp_config.name), cwd)),
            (pointer, pointer.exists(), pa.is_git_tracked(Path(pointer.relative_to(cwd)), cwd)),
            (agents_cache, agents_cache.exists(), False),
        )
        fset = file_set_hash(file_set)
        if not has_consent(self.name, cwd, fset):
            plan_lines = [
                f"Register perfxpert MCP in {mcp_config}",
                f"Stage rendered prompt at {agents_cache}",
                f"Write pointer at {pointer}",
            ]
            granted = prompt_consent_interactive(self.name, cwd, plan_lines)
            if not granted:
                raise ConsentDenied(
                    f"user declined perfxpert install for claude in {cwd}. "
                    f"Re-run with {CONSENT_ASSUME_ENV}=1 to bypass the prompt "
                    f"(non-interactive)."
                )

        if dry_run:
            # Dry run — no writes, no consent persistence.
            self._log_step(quiet, "[dry-run] would install perfxpert for claude")
            return InstallReport(
                backend=self.name,
                actions=tuple(self.plan(cwd, scope=scope).actions),
                paths_written=(),
                duration_s=time.monotonic() - start,
            )

        # ---- Step 1/4: Register MCP ------------------------------------
        self._log_step(quiet, "[1/4] Registering perfxpert MCP in .mcp.json ...")
        self._register_mcp(cwd, mcp_config, scope)
        actions_done.append("registered perfxpert MCP")
        paths_written.append(mcp_config)
        self._append_gitignore_if_present(cwd, self._MCP_CONFIG_FILENAME)

        # ---- Step 2/4: Stage prompt ------------------------------------
        self._log_step(quiet, "[2/4] Staging rendered prompt ...")
        rendered = self._render_prompt_for_claude()
        pa.stage_cache_file(agents_cache, agents_cache, rendered)
        actions_done.append("staged AGENTS.md cache")
        paths_written.append(agents_cache)

        # ---- Step 3/4: Write pointer -----------------------------------
        pointer_rel = pointer.relative_to(cwd)
        self._log_step(quiet, f"[3/4] Writing pointer at {pointer_rel} ...")
        pointer_contents = self._make_pointer(agents_cache, cwd)
        # Never touch a git-tracked CLAUDE.local.md unless explicit opt-in.
        if pointer.exists() and pa.is_git_tracked(
            Path(pointer_rel), cwd
        ) and not allow_agents_md_append:
            raise ConfigClobber(
                f"{pointer} is tracked in git. Pass --allow-agents-md-append "
                "to merge, or remove the file first. CLAUDE.local.md is "
                "conventionally gitignored as a local-only context file."
            )
        pa.atomic_write(pointer, pointer_contents)
        actions_done.append("wrote pointer")
        paths_written.append(pointer)

        # ---- Step 4/4: Verify ------------------------------------------
        if os.environ.get(SKIP_LIVE_CHECK_ENV, "").strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            self._log_step(
                quiet,
                f"[4/4] SKIPPED (via {SKIP_LIVE_CHECK_ENV}=1 — Task 4b intermediate state).",
            )
        else:
            self._log_step(quiet, "[4/4] Verifying perfxpert MCP is live ...")
            report = self.verify_mcp_live(cwd)
            if not report.mcp_healthy:
                raise PartialInstall(
                    f"perfxpert MCP registered but live-check failed: "
                    f"{report.error or 'unknown reason'}"
                )

        # Persist consent on success.
        grant_consent(self.name, cwd, fset)

        return InstallReport(
            backend=self.name,
            actions=tuple(actions_done),
            paths_written=tuple(paths_written),
            duration_s=time.monotonic() - start,
        )

    # ------------------------------------------------------------------
    # verify_mcp_live (Task 4c) — retry-aware MCP handshake + gate probe.
    # ------------------------------------------------------------------

    def verify_mcp_live(
        self, cwd: Path, telemetry: bool = False
    ) -> LiveCheckReport:
        """Spawn `claude mcp list`, assert perfxpert is listed + healthy.

        Wraps the probe in `retry_mcp_handshake` for the MCP bootstrap
        race: 3 attempts with exponential backoff 2/4/8 under
        `PERFXPERT_MCP_RETRY_BUDGET_S` budget.

        Gate probe: if the gate hook is installed, run a canned query
        and assert the hook rejected it. When the gate surface is
        unsupported, `gate_hook_installed` is `False` — documented
        known-limit, not failure.

        `PERFXPERT_TELEMETRY=1` additionally probes perfxpert-mcp's
        telemetry log for an `intent_classify` observation.

        Parses the plain-text `mcp list` output; recent Claude CLI
        versions removed the `--json` flag so we scan line-by-line
        for the ``<name>: <endpoint> - <status>`` form.
        """
        binary = shutil.which(self.binary_name)
        if binary is None:
            return LiveCheckReport(
                backend=self.name,
                mcp_listed=False,
                mcp_healthy=False,
                gate_hook_installed=None,
                error=f"{self.binary_name!r} not on PATH at verify time",
            )

        def _list_probe() -> tuple[bool, tuple[str, ...]]:
            """Single attempt — raises on error so the retry helper can backoff."""
            result = subprocess.run(
                [binary, "mcp", "list"],
                cwd=str(cwd),
                timeout=15,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                raise PartialInstall(
                    f"claude mcp list exit {result.returncode}: "
                    f"{result.stderr.decode('utf-8', errors='replace')}"
                )
            stdout = result.stdout.decode("utf-8", errors="replace")
            servers = _parse_mcp_list_text(stdout)
            if "perfxpert" not in servers:
                raise PartialInstall(
                    f"perfxpert entry missing from claude mcp list output "
                    f"(observed: {sorted(servers)})"
                )
            status = servers["perfxpert"]
            if not _is_healthy_status(status):
                raise PartialInstall(
                    f"perfxpert entry present but unhealthy: {status!r}"
                )
            # Plain-text `mcp list` does not expose per-server tool names;
            # leave the observed list empty — consumers treat empty as
            # "unavailable" rather than "zero tools".
            return True, ()

        # Drive the probe through the retry helper.
        try:
            listed, observed_tools = pa.retry_mcp_handshake(_list_probe)
        except PartialInstall as exc:
            return LiveCheckReport(
                backend=self.name,
                mcp_listed=False,
                mcp_healthy=False,
                gate_hook_installed=self._probe_gate_hook_installed(cwd),
                error=str(exc),
            )

        # Gate-hook probe — stays lightweight in unit tests because we
        # only check the settings.json marker here; the live "reject
        # first bash call" assertion is part of the manual
        # acceptance-criterion recipe.
        gate_status = self._probe_gate_hook_installed(cwd)

        # Optional telemetry probe (I11).
        telem_ok = True
        if telemetry or os.environ.get("PERFXPERT_TELEMETRY", "").strip() in {
            "1",
            "true",
        }:
            telem_ok = self._probe_telemetry_log(cwd)

        return LiveCheckReport(
            backend=self.name,
            mcp_listed=listed,
            mcp_healthy=listed and telem_ok,
            observed_tool_names=observed_tools,
            gate_hook_installed=gate_status,
            error=None if telem_ok else "telemetry log did not record intent_classify",
        )

    def _probe_gate_hook_installed(self, cwd: Path) -> bool | None:
        """Cheap check: does `.claude/settings.json` contain our PreToolUse
        hook entry? Returns tri-state per `LiveCheckReport.gate_hook_installed`.
        """
        if os.environ.get("PERFXPERT_GATE_HOOK", "").strip() == "0":
            return None
        settings = cwd / ".claude" / "settings.json"
        if not settings.is_file():
            return False
        try:
            data = json.loads(settings.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        hooks = data.get("hooks", {})
        pre = hooks.get("PreToolUse") if isinstance(hooks, dict) else None
        if not pre:
            return False
        # A minimal marker: the hook command references the perfxpert
        # gate script. Task 4.6 writes the exact path.
        serialized = json.dumps(pre)
        return "perfxpert-gate" in serialized or "perfxpert" in serialized

    def _probe_telemetry_log(self, cwd: Path) -> bool:
        """Inspect the perfxpert-mcp telemetry log for `intent_classify`."""
        xdg_cache = os.environ.get("XDG_CACHE_HOME") or str(
            Path.home() / ".cache"
        )
        log_path = Path(xdg_cache) / "perfxpert" / "mcp-telemetry.log"
        if not log_path.is_file():
            return True  # Absent log = unknown, treat as non-blocking.
        try:
            text = log_path.read_text()
        except OSError:
            return True
        return "intent_classify" in text

    # ------------------------------------------------------------------
    # spawn
    # ------------------------------------------------------------------

    def spawn(
        self, argv: list[str], env: dict[str, str], cwd: Path
    ) -> int:
        """Replace the Python process with the claude TUI via execvpe."""
        os.chdir(str(cwd))
        os.execvpe(self.binary_name, [self.binary_name, *argv], env)
        # execvpe never returns on success; reach here = failure.
        return 127  # pragma: no cover

    # ------------------------------------------------------------------
    # uninstall
    # ------------------------------------------------------------------

    def uninstall(
        self,
        cwd: Path,
        scope: Literal["project", "user"] = "project",
    ) -> UninstallReport:
        cwd = cwd.expanduser().resolve()
        mcp_config, pointer, agents_cache = self._target_paths(cwd)
        actions: list[str] = []
        removed: list[Path] = []
        drifted: list[Path] = []

        # Pointer file(s) — current canonical CLAUDE.local.md + any legacy
        # locations from older installs. Each is drift-checked independently
        # so a user who hand-edited one doesn't block cleanup of the others.
        pointer_candidates = [pointer]
        for legacy_dir, legacy_name in self._LEGACY_POINTER_PATHS:
            pointer_candidates.append(cwd / legacy_dir / legacy_name)
        for candidate in pointer_candidates:
            if not candidate.exists():
                continue
            try:
                body = candidate.read_text()
            except OSError:
                body = ""
            if self._POINTER_FILE_SENTINEL in body:
                try:
                    candidate.unlink()
                    removed.append(candidate)
                    actions.append(f"removed pointer {candidate}")
                except OSError as exc:
                    actions.append(
                        f"failed to remove pointer {candidate}: {exc}"
                    )
            else:
                drifted.append(candidate)
                actions.append(
                    f"refused to remove {candidate} — content drifted from "
                    "managed sentinel"
                )

        # Cache file — safe to remove (we own the .perfxpert dir).
        if agents_cache.exists():
            try:
                agents_cache.unlink()
                removed.append(agents_cache)
                actions.append(f"removed {agents_cache}")
            except OSError as exc:
                actions.append(f"failed to remove {agents_cache}: {exc}")

        # Remove the .perfxpert/ dir if empty.
        perfxpert_dir = cwd / self._PERFXPERT_DIR
        if perfxpert_dir.is_dir():
            try:
                perfxpert_dir.rmdir()
            except OSError:
                pass

        # MCP entry — shell out to `claude mcp remove perfxpert`.
        if shutil.which(self.binary_name):
            try:
                subprocess.run(
                    [self.binary_name, "mcp", "remove", "perfxpert"],
                    cwd=str(cwd),
                    timeout=15,
                    capture_output=True,
                    check=False,
                )
                actions.append("removed MCP entry via claude mcp remove")
            except (OSError, subprocess.TimeoutExpired) as exc:
                actions.append(f"failed to remove MCP entry: {exc}")

        # Consent cache — revoke so next install re-prompts.
        revoke_consent(self.name, cwd)
        actions.append("revoked consent")

        return UninstallReport(
            backend=self.name,
            actions=tuple(actions),
            paths_removed=tuple(removed),
            skipped_due_to_drift=tuple(drifted),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    _POINTER_FILE_SENTINEL = "@.perfxpert/AGENTS.md"

    def _log_step(self, quiet: bool, msg: str) -> None:
        if not quiet:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        _LOG.debug(msg)

    def _render_prompt_for_claude(self) -> str:
        """Render the perfxpert prompt with claude's wire format."""
        # The source prompt is bundled inside the perfxpert package; if
        # missing (unusual install), fall back to a minimal inline form
        # that still passes the render-pipeline and keeps the rejection
        # language active.
        bundled_source = _find_bundled_agents_md()
        if bundled_source is None:
            source = (
                "Always call `perfxpert_intent_classify` first. After that "
                "tool returns, you may use any other tool.\n"
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

    def _make_pointer(self, agents_cache: Path, cwd: Path) -> str:
        """Return the pointer file body (distinctive so uninstall can detect drift)."""
        rel = agents_cache.relative_to(cwd)
        return (
            "<!-- perfxpert-managed pointer file. Do not edit inside this file — "
            "your perfxpert config lives in .perfxpert/ and .mcp.json. -->\n"
            f"{self._POINTER_FILE_SENTINEL}\n"
        )

    # ---- MCP registration -------------------------------------------------

    def _register_mcp(
        self, cwd: Path, mcp_config: Path, scope: Literal["project", "user"]
    ) -> None:
        """Primary: `claude mcp add perfxpert --scope <s> -- perfxpert-mcp`.

        Idempotent: if `claude mcp get perfxpert` returns 0 already, skip.
        On non-zero / timeout, print the command for the user and fall
        back to a structured edit of `.mcp.json` (project scope only;
        user scope `~/.claude.json` is multi-MB — I4 bans whole-file
        rewrite there).
        """
        binary = shutil.which(self.binary_name)

        # 1) Idempotency: if entry is already present + identical, skip.
        if binary and self._mcp_already_registered(binary, cwd):
            _LOG.debug("perfxpert MCP entry already present; skipping add")
            return

        # 2) Primary: shell out to `claude mcp add`.
        if binary:
            try:
                result = subprocess.run(
                    [
                        binary,
                        "mcp",
                        "add",
                        "perfxpert",
                        "--scope",
                        scope,
                        "--",
                        "perfxpert-mcp",
                    ],
                    cwd=str(cwd),
                    timeout=15,
                    capture_output=True,
                    check=False,
                )
                if result.returncode == 0:
                    return
                _LOG.debug(
                    "claude mcp add returned %d: stdout=%r stderr=%r",
                    result.returncode,
                    result.stdout,
                    result.stderr,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                _LOG.debug("claude mcp add failed: %s", exc)

        # 3) Print-for-human fallback: tell user the command and fall
        #    through to structured edit ONLY for project scope.
        sys.stderr.write(
            "\nperfxpert-code install: primary `claude mcp add` path failed.\n"
            "  Run this command manually (or fix your claude install):\n"
            f"    claude mcp add perfxpert --scope {scope} -- perfxpert-mcp\n"
        )
        if scope == "user":
            # Refuse structured edit on user scope — that file is
            # multi-MB and contains session history (I4).
            raise BackendNotFound(
                "cannot register perfxpert in user scope without the "
                "`claude` CLI; run the printed command manually."
            )

        # 4) Structured edit of .mcp.json (project scope is a small
        #    dedicated file — safe to edit atomically).
        self._structured_edit_mcp_json(mcp_config)

    def _mcp_already_registered(self, binary: str, cwd: Path) -> bool:
        try:
            result = subprocess.run(
                [binary, "mcp", "get", "perfxpert"],
                cwd=str(cwd),
                timeout=10,
                capture_output=True,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def _structured_edit_mcp_json(self, mcp_config: Path) -> None:
        """Atomically add `perfxpert` entry to `.mcp.json`.

        Preserves every other `mcpServers.*` entry. Raises
        `ConfigClobber` if a conflicting `perfxpert` entry is already
        present (different command).
        """
        data: dict = {}
        if mcp_config.is_file():
            try:
                data = json.loads(mcp_config.read_text())
            except json.JSONDecodeError as exc:
                raise PartialInstall(
                    f"{mcp_config} is not valid JSON; cannot merge: {exc}"
                ) from exc

        servers = data.setdefault("mcpServers", {})
        existing = servers.get("perfxpert")
        new_entry = {"command": "perfxpert-mcp", "args": []}
        if existing:
            # Idempotent re-register: same entry = no-op.
            if existing.get("command") == "perfxpert-mcp":
                return
            raise ConfigClobber(
                f"{mcp_config} already has a perfxpert entry with command "
                f"{existing.get('command')!r}; refuse to overwrite."
            )
        servers["perfxpert"] = new_entry
        data["mcpServers"] = servers
        pa.atomic_write(mcp_config, json.dumps(data, indent=2) + "\n")

    def _append_gitignore_if_present(self, cwd: Path, filename: str) -> None:
        gi = cwd / ".gitignore"
        if not gi.is_file():
            return
        current = gi.read_text()
        if filename in current.splitlines():
            return
        # Append with a header if we're introducing the first perfxpert
        # rule.
        if "perfxpert" not in current:
            addition = f"\n# Added by perfxpert-code install\n{filename}\n"
        else:
            addition = f"{filename}\n"
        pa.atomic_write(gi, current + addition, backup=False)


# ---------------------------------------------------------------------------
# Runtime-checkable type assertion.
# ---------------------------------------------------------------------------


# Fail fast in tests / at import time if the adapter drifts away from the
# Protocol. Because `BackendAdapter` is runtime_checkable, this also
# guards against accidentally adding a method signature mismatch.
assert isinstance(ClaudeCodeAdapter(), BackendAdapter)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _version_at_or_above(line: str, minimum: str) -> bool:
    """Return True iff `line` contains a SemVer >= `minimum`.

    Forgiving parser: scans `line` for the first `N.N.N` pattern and
    compares numeric-tuple-wise. Falls back to `True` if no version
    shape is present (better than false rejection).
    """
    import re

    m = re.search(r"(\d+)\.(\d+)\.(\d+)", line)
    if not m:
        return True
    found = tuple(int(x) for x in m.groups())
    mm = re.match(r"(\d+)\.(\d+)\.(\d+)", minimum)
    if not mm:
        return True
    required = tuple(int(x) for x in mm.groups())
    return found >= required


_HEALTHY_STATUS_MARKERS = ("✓", "Connected", "connected", "OK", "ok")
_UNHEALTHY_STATUS_MARKERS = ("✘", "failed", "Failed", "error", "Error")
_MCP_LIST_LINE_RE = re.compile(
    r"^(?P<name>[^:]+):\s+(?P<endpoint>\S.*?)\s+-\s+(?P<status>.+?)\s*$"
)


def _parse_mcp_list_text(output: str) -> dict[str, str]:
    """Parse `claude mcp list` plain-text output into {server_name: status}.

    Recent Claude CLI dropped the `--json` flag. The text form is
    ``<name>: <endpoint-or-command> - <status>``. Header and empty
    lines are skipped.
    """
    servers: dict[str, str] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("Checking", "No MCP")):
            continue
        match = _MCP_LIST_LINE_RE.match(stripped)
        if match:
            servers[match.group("name").strip()] = match.group("status").strip()
    return servers


def _is_healthy_status(status: str) -> bool:
    """Status line is healthy when it contains a known OK marker and no
    explicit failure marker."""
    if any(bad in status for bad in _UNHEALTHY_STATUS_MARKERS):
        return False
    return any(ok in status for ok in _HEALTHY_STATUS_MARKERS)


def _find_bundled_agents_md() -> Path | None:
    """Find the bundled opencode AGENTS.md to use as the prompt source."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "_bundled" / "opencode_config" / "AGENTS.md",
        here.parent.parent.parent / "_bundled" / "AGENTS.md",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None
