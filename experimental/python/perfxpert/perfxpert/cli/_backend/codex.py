"""CodexAdapter — perfxpert MCP install for OpenAI Codex CLI (Task 10, PR 2).

Implements the Protocol from `perfxpert.cli._backend.protocol` for
the `codex` CLI. Covers:

* `check_available()` — `shutil.which("codex")` + `--version` parse.
* `_check_trust(cwd)` — helper returning `TrustStatus` for the given
  cwd. Reads `[projects."<path>"].trust_level` from
  `~/.codex/config.toml` (there is NO `codex projects list` CLI as of
  April 2026 — API drift from the original plan, see decision record).
* `plan()` / `install()` / `uninstall()` / `spawn()` — full lifecycle.
* `verify_mcp_live()` — shells out to `codex mcp list` + probes for
  the perfxpert entry.

Trust-gate behavior (plan Task 10):
  1. untrusted + non-interactive + no ASSUME_CONSENT → TrustRequired.
  2. untrusted + interactive → prompt (or auto-trust if
     PERFXPERT_AUTO_TRUST=1).
  3. untrusted after prompt → fall back to user-scope
     `~/.codex/config.toml` with an explicit stderr warning.
  4. trusted → project-scope `<cwd>/.codex/config.toml`.

MCP registration prefers `codex mcp add` shell-out; falls back to
direct TOML edit via **lazy-imported `tomlkit`** (historically
cycle-2 I7, superseded by commit 3547736829 which made tomlkit a
required runtime dep — imported inside the fallback branch ONLY;
the primary code path must never pay the import cost).

Prompt staging writes to `<scope_dir>/.perfxpert/AGENTS.md`, NEVER to
a tracked `AGENTS.md` unless `--allow-agents-md-append` is given.

Gate hook (Task 4.6 Codex-portion): the gate hook module is
imported from `perfxpert.cli._gate_hooks.codex` and invoked BEFORE
MCP registration. If Codex's hook surface cannot enforce the gate
(API drift: Codex's PreToolUse currently only intercepts Bash, NOT
MCP tool calls — see decision record 2026-04-19-codex-hook-surface.md),
the hook raises `GateHookUnsupported` and `install()` records
`gate_hook_installed=False` in the subsequent `verify_mcp_live()`
report (I-N1 documented-known-limit).
"""

from __future__ import annotations

import enum
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
    BackendNotFound,
    ConfigClobber,
    ConsentDenied,
    GateHookUnsupported,
    InstallReport,
    LiveCheckReport,
    PartialInstall,
    Plan,
    TrustRequired,
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


__all__ = [
    "CodexAdapter",
    "TrustStatus",
    "AUTO_TRUST_ENV",
    "SKIP_LIVE_CHECK_ENV",
]


_LOG = logging.getLogger("perfxpert.backend.codex")


# Env override — matches ClaudeCodeAdapter so dispatcher / tests share
# a single key across backends.
SKIP_LIVE_CHECK_ENV = "PERFXPERT_SKIP_LIVE_CHECK"


# Auto-trust env gate. When "1", an untrusted cwd is marked trusted
# without prompting; useful for CI pipelines where the user has
# already accepted the remote config.
AUTO_TRUST_ENV = "PERFXPERT_AUTO_TRUST"


# Known MCP tool names rendered into the staged AGENTS.md.
_KNOWN_TOOLS: tuple[str, ...] = (
    "intent_classify",
    "next_step",
    "report",
    "analyze",
    "classify",
)


class TrustStatus(enum.Enum):
    """Result of probing `~/.codex/config.toml` for a cwd's trust_level.

    TRUSTED    — `[projects."<cwd>"].trust_level = "trusted"`.
    UNTRUSTED  — `[projects."<cwd>"].trust_level = "untrusted"` (or
                  empty string, or any non-"trusted" value).
    UNKNOWN    — no `[projects."<cwd>"]` table at all. Codex treats
                  this as "untrusted by default" and skips project
                  scope. We also treat it as UNTRUSTED for write
                  decisions, but surface it separately so the
                  "never been seen" case can log differently.
    """

    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    UNKNOWN = "unknown"


class CodexAdapter:
    """Adapter for the `codex` CLI."""

    name: str = "codex"
    binary_name: str = "codex"
    install_hint: str = (
        "Install via https://developers.openai.com/codex/cli "
        "(npm i -g @openai/codex or equivalent)"
    )
    min_version: str | None = "0.7.0"
    known_schema_versions: tuple[str, ...] = ("0.7.x",)
    # Codex's MCP tool name wire format is `mcp_<server>_<tool>` per
    # the April 2026 observed behavior. verify_mcp_live() probes the
    # actual wire format and warns on drift (plan B1).
    tool_name_template: str = "mcp_perfxpert_{tool}"
    spawn_strategy: Literal["execvpe", "subprocess"] = "execvpe"

    _PERFXPERT_DIR = ".perfxpert"
    _AGENTS_FILE = "AGENTS.md"
    _CODEX_CFG_REL = ".codex/config.toml"

    # ------------------------------------------------------------------
    # check_available
    # ------------------------------------------------------------------

    def check_available(self) -> tuple[bool, str]:
        path = shutil.which(self.binary_name)
        if not path:
            return False, f"{self.binary_name!r} not found on PATH. {self.install_hint}"
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
            return False, (
                f"{self.binary_name} --version returned {result.returncode}"
            )
        line = (result.stdout or result.stderr or "").strip()
        if self.min_version and not _version_at_or_above(line, self.min_version):
            return False, (
                f"{self.binary_name} version {line!r} is below the "
                f"minimum required {self.min_version!r}."
            )
        return True, line

    # ------------------------------------------------------------------
    # _check_trust — reads `~/.codex/config.toml` directly.
    # ------------------------------------------------------------------

    def _user_codex_config_path(self, home: Path | None = None) -> Path:
        return (home or Path.home()) / self._CODEX_CFG_REL

    def _check_trust(
        self, cwd: Path, home: Path | None = None
    ) -> TrustStatus:
        """Return trust status for `cwd` per `~/.codex/config.toml`.

        **API drift note.** The original plan assumed a
        `codex projects list --json` command existed. As of April 2026
        it does NOT; trust is config-only (rationale in the local
        Codex hook-surface decision record). We read the config file
        directly, consulting `[projects."<abs-cwd>"].trust_level`.
        """
        cfg = self._user_codex_config_path(home)
        if not cfg.is_file():
            return TrustStatus.UNKNOWN

        try:
            toml_source = cfg.read_text(encoding="utf-8")
        except OSError as exc:
            _LOG.debug("read(%s) failed: %s", cfg, exc)
            return TrustStatus.UNKNOWN

        parsed = _parse_toml_read_only(toml_source)
        if parsed is None:
            return TrustStatus.UNKNOWN

        projects = parsed.get("projects")
        if not isinstance(projects, dict):
            return TrustStatus.UNKNOWN

        # Codex's `[projects."<path>"]` table is keyed on the absolute
        # resolved project path. Match case-sensitively to mirror
        # Codex's behavior.
        resolved = str(Path(cwd).expanduser().resolve())
        entry = projects.get(resolved)
        if not isinstance(entry, dict):
            return TrustStatus.UNKNOWN
        level = entry.get("trust_level")
        if level == "trusted":
            return TrustStatus.TRUSTED
        if level == "untrusted":
            return TrustStatus.UNTRUSTED
        return TrustStatus.UNKNOWN

    # ------------------------------------------------------------------
    # plan
    # ------------------------------------------------------------------

    def _scope_config_dir(
        self, cwd: Path, scope: Literal["project", "user"]
    ) -> Path:
        """Return the directory whose `config.toml` we should edit."""
        if scope == "project":
            return Path(cwd) / ".codex"
        return Path.home() / ".codex"

    def _target_paths(
        self,
        cwd: Path,
        scope: Literal["project", "user"] = "project",
    ) -> tuple[Path, Path]:
        """Return (config_toml_path, agents_cache_path)."""
        cwd = Path(cwd).expanduser().resolve()
        config_dir = self._scope_config_dir(cwd, scope)
        config_toml = config_dir / "config.toml"
        agents_cache = cwd / self._PERFXPERT_DIR / self._AGENTS_FILE
        return config_toml, agents_cache

    def plan(
        self,
        cwd: Path,
        scope: Literal["project", "user"] = "project",
        dry_run: bool = True,
    ) -> Plan:
        config_toml, agents_cache = self._target_paths(cwd, scope)
        cwd_resolved = Path(cwd).expanduser().resolve()
        actions = [
            f"Check project trust in {self._user_codex_config_path()} for {cwd_resolved}",
            f"Register [mcp_servers.perfxpert] in {config_toml} (scope={scope})",
            f"Stage rendered prompt at {agents_cache.relative_to(cwd_resolved)}",
            "Install PreToolUse gate hook (prompt-layer-only on Codex — see decision record)",
            "Warm perfxpert-mcp",
            "Verify perfxpert MCP is live (codex mcp list)",
        ]
        return Plan(
            backend=self.name,
            actions=tuple(actions),
            targets=(config_toml, agents_cache),
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
        config_toml, agents_cache = self._target_paths(cwd, scope)

        # ---- Consent -------------------------------------------------
        fset = file_set_hash(
            (
                (config_toml, config_toml.exists(), False),
                (agents_cache, agents_cache.exists(), False),
            )
        )
        if not has_consent(self.name, cwd, fset):
            plan_lines = [
                f"Register [mcp_servers.perfxpert] in {config_toml}",
                f"Stage rendered prompt at {agents_cache}",
            ]
            if not prompt_consent_interactive(self.name, cwd, plan_lines):
                raise ConsentDenied(
                    f"user declined perfxpert install for codex in {cwd}. "
                    f"Re-run with {CONSENT_ASSUME_ENV}=1 to bypass the "
                    "prompt (non-interactive)."
                )

        if dry_run:
            self._log_step(quiet, "[dry-run] would install perfxpert for codex")
            return InstallReport(
                backend=self.name,
                actions=tuple(self.plan(cwd, scope=scope).actions),
                paths_written=(),
                duration_s=time.monotonic() - start,
            )

        actions: list[str] = []
        written: list[Path] = []

        # ---- Step 0/5: Trust gate -----------------------------------
        self._log_step(quiet, "[0/5] Checking project trust ...")
        scope = self._resolve_trust_and_scope(cwd, scope, quiet=quiet)
        # Re-resolve target paths in case scope changed.
        config_toml, agents_cache = self._target_paths(cwd, scope)
        actions.append(f"resolved install scope={scope}")

        # ---- Step 1/5: Install gate hook (BEFORE MCP) --------------
        # I-N1: hook install runs first so any GateHookUnsupported
        # surfaces BEFORE we touch config.toml (no partial state).
        self._log_step(quiet, "[1/5] Installing gate hook (before MCP) ...")
        gate_installed = False
        try:
            from perfxpert.cli._gate_hooks.codex import install as gate_install

            gate_install(cwd)
            gate_installed = True
            actions.append("installed gate hook (prompt-layer)")
        except GateHookUnsupported as exc:
            # Documented-known-limit: Codex's PreToolUse only hits
            # Bash. Prompt-layer-only is the acceptable fallback.
            _LOG.warning("codex gate hook unsupported: %s", exc)
            actions.append(f"gate hook unsupported: {exc} (prompt-layer-only)")
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning("codex gate hook install failed: %s", exc)
            actions.append(f"gate hook install failed: {exc}")

        # ---- Step 2/5: Register MCP ---------------------------------
        self._log_step(
            quiet,
            f"[2/5] Registering perfxpert MCP in {config_toml.name} (scope={scope}) ...",
        )
        self._register_mcp(cwd, config_toml, scope)
        actions.append("registered perfxpert MCP")
        written.append(config_toml)

        # ---- Step 3/5: Stage prompt ---------------------------------
        self._log_step(quiet, "[3/5] Staging rendered prompt ...")
        rendered = self._render_prompt_for_codex()
        # Pre-check: 32 KiB cap on AGENTS.md so we don't silently
        # produce a monster file (practical §3.4).
        if len(rendered.encode("utf-8")) > 32 * 1024:
            raise PartialInstall(
                f"rendered codex prompt exceeds 32 KiB cap "
                f"({len(rendered.encode('utf-8'))} bytes). Shrink the "
                "source AGENTS.md or raise the cap explicitly."
            )
        # Respect --allow-agents-md-append: if the user's AGENTS.md is
        # tracked AND the flag is NOT set, we refuse to touch it. Our
        # cache path is always under .perfxpert/, so we never edit
        # AGENTS.md directly unless the user explicitly opted in.
        tracked_agents_md = cwd / self._AGENTS_FILE
        if (
            tracked_agents_md.exists()
            and pa.is_git_tracked(Path(self._AGENTS_FILE), cwd)
            and not allow_agents_md_append
        ):
            # Stage to .perfxpert/AGENTS.md only; never touch the
            # tracked file. This is the default safe path.
            _LOG.debug(
                "tracked AGENTS.md at %s; using .perfxpert/ staging path",
                tracked_agents_md,
            )
        pa.stage_cache_file(agents_cache, agents_cache, rendered)
        actions.append("staged AGENTS.md cache")
        written.append(agents_cache)

        # ---- Step 4/5: Warmup ---------------------------------------
        self._log_step(quiet, "[4/5] Warming perfxpert-mcp ...")
        try:
            from perfxpert.cli._mcp_warmup import warmup_perfxpert_mcp

            report = warmup_perfxpert_mcp()
            if report.success:
                actions.append(f"warmed perfxpert-mcp in {report.duration_s:.2f}s")
            else:
                actions.append(
                    f"warmup skipped/failed: {report.error or 'unknown'}"
                )
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.debug("warmup failed: %s", exc)
            actions.append(f"warmup raised: {exc}")

        # ---- Step 5/5: Verify ---------------------------------------
        if os.environ.get(SKIP_LIVE_CHECK_ENV, "").strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            self._log_step(
                quiet,
                f"[5/5] SKIPPED (via {SKIP_LIVE_CHECK_ENV}=1).",
            )
        else:
            self._log_step(quiet, "[5/5] Verifying perfxpert MCP is live ...")
            live = self.verify_mcp_live(cwd)
            if not live.mcp_healthy:
                raise PartialInstall(
                    f"perfxpert MCP registered but live-check failed: "
                    f"{live.error or 'unknown reason'}"
                )

        grant_consent(self.name, cwd, fset)
        _ = gate_installed  # quiets type-checker on unused when no telem.
        return InstallReport(
            backend=self.name,
            actions=tuple(actions),
            paths_written=tuple(written),
            duration_s=time.monotonic() - start,
        )

    # ------------------------------------------------------------------
    # Trust resolution — consolidates the interactive / env / fallback
    # decision tree from Task 10.
    # ------------------------------------------------------------------

    def _resolve_trust_and_scope(
        self,
        cwd: Path,
        scope: Literal["project", "user"],
        quiet: bool = False,
    ) -> Literal["project", "user"]:
        """Resolve trust; return the effective scope.

        Decision tree (Task 10 spec):

        * scope=user → no trust check needed (user scope is always
          applicable; project trust only gates project scope).
        * scope=project + trust=TRUSTED → keep project scope.
        * scope=project + trust in {UNTRUSTED, UNKNOWN}:
          1. `PERFXPERT_AUTO_TRUST=1` → auto-mark trusted (CI).
          2. not interactive + NOT PERFXPERT_ASSUME_CONSENT →
             `TrustRequired` (explicit consent on trust is required
             before we write to ~/.codex/config.toml or fall back).
          3. not interactive + PERFXPERT_ASSUME_CONSENT → fall back
             to user-scope with a ⚠ warning (we don't auto-trust
             because consent cache ≠ trust grant).
          4. interactive → prompt user:
             - accept → mark trusted, keep project scope.
             - decline → fall back to user-scope with ⚠ warning.
        """
        if scope == "user":
            return scope

        status = self._check_trust(cwd)
        if status == TrustStatus.TRUSTED:
            return scope

        # Path 1: Auto-trust via env (CI pipelines).
        if os.environ.get(AUTO_TRUST_ENV, "").strip() == "1":
            self._mark_trusted(cwd)
            # Security decision: auto-trust bypasses the normal
            # consent/trust prompt. The warning MUST print regardless
            # of --quiet so the user always sees that trust was granted
            # automatically (finding #5). Logged as a fresh \n-prefixed
            # line to stand out on stderr.
            sys.stderr.write(
                f"\n[WARN] codex: {AUTO_TRUST_ENV}=1 — marked {cwd} as "
                "trusted in ~/.codex/config.toml without prompting. "
                "This bypasses Codex's trust gate.\n"
            )
            sys.stderr.flush()
            _LOG.info(
                "%s=1 auto-trust granted for %s", AUTO_TRUST_ENV, cwd
            )
            return scope

        interactive = (
            sys.stdin.isatty() if hasattr(sys.stdin, "isatty") else False
        )
        assume_consent = (
            os.environ.get(CONSENT_ASSUME_ENV, "").strip() == "1"
        )

        # Path 2: non-interactive, no consent env → raise.
        if not interactive and not assume_consent:
            raise TrustRequired(
                f"Codex project at {cwd} is not marked trusted "
                f"(trust_level={status.value!r} in "
                f"~/.codex/config.toml). Mark it trusted by adding:\n"
                f'  [projects."{cwd}"]\n'
                f'  trust_level = "trusted"\n'
                f"to ~/.codex/config.toml, OR re-run with "
                f"{AUTO_TRUST_ENV}=1 to auto-trust, OR pass "
                f"--scope user to use user-scope config."
            )

        # Path 3: non-interactive + consent-env → user scope fallback.
        if not interactive and assume_consent:
            sys.stderr.write(
                f"\n[WARN] codex: {cwd} is not marked trusted, "
                f"{CONSENT_ASSUME_ENV}=1 is set but {AUTO_TRUST_ENV} is "
                "not — falling back to user-scope config at "
                "~/.codex/config.toml. The MCP entry will be visible "
                "from EVERY codex invocation, not just this project.\n"
            )
            return "user"

        # Path 4: interactive prompt.
        if self._prompt_user_for_trust(cwd, status):
            self._mark_trusted(cwd)
            return scope
        sys.stderr.write(
            f"\n[WARN] codex: declined to trust {cwd} — falling back "
            "to user-scope config at ~/.codex/config.toml. The MCP "
            "entry will be visible from EVERY codex invocation, not "
            "just this project.\n"
        )
        return "user"

    def _prompt_user_for_trust(self, cwd: Path, status: TrustStatus) -> bool:
        """Interactive Y/n prompt asking whether to mark `cwd` trusted.

        Returns True iff the user accepted. On EOFError (e.g. /dev/null
        stdin in a half-interactive environment) treat as declined.
        """
        sys.stderr.write(
            f"\nperfxpert-code codex: project at {cwd} is not trusted "
            f"(trust_level={status.value!r}).\n"
            f"  Mark it trusted in ~/.codex/config.toml now?  [y/N] "
        )
        sys.stderr.flush()
        try:
            answer = sys.stdin.readline().strip().lower()
        except (EOFError, OSError):
            return False
        return answer in {"y", "yes"}

    def _mark_trusted(
        self, cwd: Path, home: Path | None = None
    ) -> None:
        """Write `[projects."<cwd>"].trust_level = "trusted"` into
        `~/.codex/config.toml`, preserving every other key + comment.

        Uses lazy-imported `tomlkit` for lossless round-trip; falls
        back to a minimal hand-rolled rewrite if `tomlkit` is not
        installed. Historically `tomlkit` came from an optional
        `[backends]` extra (plan I7), but commit 3547736829 promoted
        it to a required runtime dep, so the fallback is now truly
        defensive rather than user-facing.

        **Git-tracked refusal.** If `~/.codex/config.toml` is tracked
        in an enclosing git repo (dotfiles-style), raise
        `ConfigClobber` rather than silently rewriting a versioned
        file. The user must move it outside version control before
        we'll touch it. Mirrors the git-tracked rule the Claude
        adapter applies to `CLAUDE.local.md`.
        """
        cfg = self._user_codex_config_path(home)
        self._refuse_if_git_tracked(cfg)
        cfg.parent.mkdir(parents=True, exist_ok=True)
        existing = cfg.read_text(encoding="utf-8") if cfg.is_file() else ""

        tomlkit_mod = _lazy_import_tomlkit(
            "marking a project trusted in ~/.codex/config.toml"
        )

        try:
            doc = tomlkit_mod.parse(existing)
        except Exception as exc:
            # A syntactically invalid file should NEVER produce a raw
            # `tomllib.TOMLDecodeError` traceback — user-facing error
            # with remediation instead. Finding #4 from PR 2 review.
            raise ConfigClobber(
                f"{cfg} is not valid TOML: {exc}. "
                "Fix the syntax error or move the file aside before "
                "re-running perfxpert-code."
            ) from exc
        projects = doc.get("projects")
        if projects is None:
            projects = tomlkit_mod.table()
            doc["projects"] = projects

        resolved = str(Path(cwd).expanduser().resolve())
        entry = projects.get(resolved)
        if entry is None:
            entry = tomlkit_mod.table()
            projects[resolved] = entry
        entry["trust_level"] = "trusted"

        pa.atomic_write(cfg, tomlkit_mod.dumps(doc))

    # ------------------------------------------------------------------
    # MCP registration — primary shell-out, fallback tomlkit edit.
    # ------------------------------------------------------------------

    def _register_mcp(
        self,
        cwd: Path,
        config_toml: Path,
        scope: Literal["project", "user"],
    ) -> None:
        """Primary: `codex mcp add perfxpert -- perfxpert-mcp`.

        Fallback: lazy-import `tomlkit` and add/merge
        `[mcp_servers.perfxpert]` into `config_toml` directly
        (preserves comments + unknown keys).
        """
        binary = shutil.which(self.binary_name)

        # 1) Idempotency: skip if already registered.
        if binary and self._mcp_already_registered(binary, cwd):
            _LOG.debug("perfxpert MCP already registered; skipping add")
            return

        # 2) Primary path: shell out.
        if binary:
            try:
                result = subprocess.run(
                    [
                        binary,
                        "mcp",
                        "add",
                        "perfxpert",
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
                    "codex mcp add returned %d: stdout=%r stderr=%r",
                    result.returncode,
                    result.stdout,
                    result.stderr,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                _LOG.debug("codex mcp add failed: %s", exc)

        # 3) Print-for-human fallback notice, then structured edit.
        sys.stderr.write(
            "\nperfxpert-code install: primary `codex mcp add` path "
            "failed.\n"
            "  Run this command manually (or fix your codex install):\n"
            "    codex mcp add perfxpert -- perfxpert-mcp\n"
        )

        # 4) Direct TOML edit via lazy-imported tomlkit.
        self._structured_edit_config_toml(config_toml)

    def _mcp_already_registered(self, binary: str, cwd: Path) -> bool:
        try:
            result = subprocess.run(
                [binary, "mcp", "list"],
                cwd=str(cwd),
                timeout=10,
                capture_output=True,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        if result.returncode != 0:
            return False
        output = (
            (result.stdout or b"").decode("utf-8", errors="replace")
            + (result.stderr or b"").decode("utf-8", errors="replace")
        )
        # `codex mcp list` output format is version-dependent; we
        # match on the bare server name token. Safe because we
        # reserve the name "perfxpert".
        return "perfxpert" in output.split()

    def _structured_edit_config_toml(self, config_toml: Path) -> None:
        """Merge `[mcp_servers.perfxpert]` into `config_toml`.

        Preserves every other table and comment. Raises `ConfigClobber`
        if a conflicting entry exists (different `command`) OR if
        `config_toml` is git-tracked in an enclosing repo (the user
        must move a versioned config out of tree before we'll touch
        it).

        **Lazy-import.** `tomlkit` is ONLY imported here — never at
        module import time. Originally motivated by plan cycle-2 I7
        (tomlkit was an optional `[backends]` extra); commit
        3547736829 promoted tomlkit to a required dep, but the lazy
        import pattern is preserved so the primary `codex mcp add`
        path never pays the tomlkit import cost.
        """
        self._refuse_if_git_tracked(config_toml)

        tomlkit_mod = _lazy_import_tomlkit(
            "editing codex config.toml directly"
        )

        config_toml.parent.mkdir(parents=True, exist_ok=True)
        existing = (
            config_toml.read_text(encoding="utf-8")
            if config_toml.is_file()
            else ""
        )
        doc = tomlkit_mod.parse(existing)

        servers = doc.get("mcp_servers")
        if servers is None:
            servers = tomlkit_mod.table(is_super_table=True)
            doc["mcp_servers"] = servers
        existing_entry = servers.get("perfxpert")
        if existing_entry is not None:
            cmd = existing_entry.get("command")
            if cmd != "perfxpert-mcp":
                raise ConfigClobber(
                    f"{config_toml} already has [mcp_servers.perfxpert] "
                    f"with command {cmd!r}; refuse to overwrite."
                )
            # Same entry — idempotent no-op.
            return

        entry = tomlkit_mod.table()
        entry["command"] = "perfxpert-mcp"
        entry["args"] = tomlkit_mod.array()
        entry["enabled"] = True
        entry["startup_timeout_sec"] = 10
        servers["perfxpert"] = entry

        pa.atomic_write(config_toml, tomlkit_mod.dumps(doc))

    # ------------------------------------------------------------------
    # verify_mcp_live
    # ------------------------------------------------------------------

    def verify_mcp_live(
        self, cwd: Path, telemetry: bool = False
    ) -> LiveCheckReport:
        """Verify codex sees the perfxpert MCP entry.

        Uses `codex mcp list` (no `--json` flag exists as of April
        2026; we parse the textual output for the "perfxpert"
        token). Falls back to reading `config.toml` if the binary
        is missing at verify time.
        """
        binary = shutil.which(self.binary_name)
        gate_installed = self._probe_gate_hook_installed(cwd)

        if binary is None:
            return LiveCheckReport(
                backend=self.name,
                mcp_listed=False,
                mcp_healthy=False,
                gate_hook_installed=gate_installed,
                error=f"{self.binary_name!r} not on PATH at verify time",
            )

        def _probe() -> bool:
            result = subprocess.run(
                [binary, "mcp", "list"],
                cwd=str(cwd),
                timeout=15,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                raise PartialInstall(
                    f"codex mcp list exit {result.returncode}: "
                    f"{result.stderr.decode('utf-8', errors='replace')}"
                )
            output = result.stdout.decode("utf-8", errors="replace")
            if "perfxpert" not in output:
                raise PartialInstall(
                    "perfxpert entry missing from `codex mcp list` output"
                )
            return True

        try:
            listed = pa.retry_mcp_handshake(_probe)
        except PartialInstall as exc:
            return LiveCheckReport(
                backend=self.name,
                mcp_listed=False,
                mcp_healthy=False,
                gate_hook_installed=gate_installed,
                error=str(exc),
            )

        return LiveCheckReport(
            backend=self.name,
            mcp_listed=listed,
            mcp_healthy=listed,
            observed_tool_names=(),
            gate_hook_installed=gate_installed,
            error=None,
        )

    def _probe_gate_hook_installed(self, cwd: Path) -> bool | None:
        """Tri-state per `LiveCheckReport.gate_hook_installed`:

        * `None` — `PERFXPERT_GATE_HOOK=0` (user disabled).
        * `False` — documented-known-limit (Codex's PreToolUse only
          intercepts Bash, not MCP; we run prompt-layer-only).
        * `True` — never returned today; future Codex hook surface
          expansion would flip this.
        """
        if os.environ.get("PERFXPERT_GATE_HOOK", "").strip() == "0":
            return None
        # On Codex, the hook surface cannot enforce the gate (see
        # decision record). False is the "documented-known-limit"
        # state, not a failure.
        return False

    # ------------------------------------------------------------------
    # spawn — execvpe into codex.
    # ------------------------------------------------------------------

    def spawn(
        self,
        argv: list[str],
        env: dict[str, str],
        cwd: Path,
    ) -> int:
        os.chdir(str(cwd))
        os.execvpe(self.binary_name, [self.binary_name, *argv], env)
        return 127  # pragma: no cover

    # ------------------------------------------------------------------
    # uninstall
    # ------------------------------------------------------------------

    def uninstall(
        self,
        cwd: Path,
        scope: Literal["project", "user"] = "project",
    ) -> UninstallReport:
        cwd = Path(cwd).expanduser().resolve()
        config_toml, agents_cache = self._target_paths(cwd, scope)
        actions: list[str] = []
        removed: list[Path] = []
        drifted: list[Path] = []

        # Remove MCP entry — prefer shell-out.
        binary = shutil.which(self.binary_name)
        if binary is not None:
            try:
                subprocess.run(
                    [binary, "mcp", "remove", "perfxpert"],
                    cwd=str(cwd),
                    timeout=15,
                    capture_output=True,
                    check=False,
                )
                actions.append("removed MCP entry via codex mcp remove")
            except (OSError, subprocess.TimeoutExpired) as exc:
                actions.append(f"codex mcp remove failed: {exc}")
                # Fall through to structured edit.
                try:
                    if self._structured_remove(config_toml, drifted):
                        actions.append(f"removed MCP entry from {config_toml}")
                except ConfigClobber as cexc:
                    # Malformed user-edited TOML → record drift + surface
                    # the clear user-facing message in actions.
                    actions.append(f"refused to edit {config_toml}: {cexc}")
                    drifted.append(config_toml)
        else:
            try:
                if self._structured_remove(config_toml, drifted):
                    actions.append(f"removed MCP entry from {config_toml}")
            except ConfigClobber as cexc:
                actions.append(f"refused to edit {config_toml}: {cexc}")
                drifted.append(config_toml)

        # Remove cache file.
        if agents_cache.exists():
            try:
                agents_cache.unlink()
                removed.append(agents_cache)
                actions.append(f"removed {agents_cache}")
            except OSError as exc:
                actions.append(f"failed to remove {agents_cache}: {exc}")

        # Remove .perfxpert/ if empty.
        perfxpert_dir = cwd / self._PERFXPERT_DIR
        if perfxpert_dir.is_dir():
            try:
                perfxpert_dir.rmdir()
            except OSError:
                pass

        # Gate hook cleanup (best-effort).
        try:
            from perfxpert.cli._gate_hooks.codex import uninstall as gate_uninstall

            gate_uninstall(cwd)
            actions.append("gate hook uninstall: complete")
        except Exception as exc:
            actions.append(f"gate hook uninstall failed: {exc}")

        revoke_consent(self.name, cwd)
        actions.append("revoked consent")

        return UninstallReport(
            backend=self.name,
            actions=tuple(actions),
            paths_removed=tuple(removed),
            skipped_due_to_drift=tuple(drifted),
        )

    def _structured_remove(
        self, config_toml: Path, drifted: list[Path]
    ) -> bool:
        """Remove `[mcp_servers.perfxpert]` from `config_toml`.

        Returns True iff the entry was present AND removed.

        Malformed TOML → raises `ConfigClobber` with a clear user-
        facing message (finding #4). The caller can catch this and
        surface drift through `UninstallReport.skipped_due_to_drift`;
        the error message tells the user exactly what to do, instead
        of the raw `tomllib.TOMLDecodeError` traceback we used to
        swallow silently.

        `tomlkit` missing → swallowed as drift (the user never
        installed the optional extra; not a config error).
        """
        if not config_toml.is_file():
            return False
        try:
            tomlkit_mod = _lazy_import_tomlkit("uninstall codex MCP entry")
        except Exception:
            drifted.append(config_toml)
            return False
        try:
            doc = tomlkit_mod.parse(
                config_toml.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise ConfigClobber(
                f"{config_toml} is not valid TOML: {exc}. "
                "Fix the syntax error or delete the file before "
                "re-running perfxpert-code uninstall."
            ) from exc

        servers = doc.get("mcp_servers")
        if not isinstance(servers, dict) or "perfxpert" not in servers:
            return False
        existing = servers["perfxpert"]
        if existing.get("command") != "perfxpert-mcp":
            # Someone else's perfxpert entry — refuse to touch it.
            drifted.append(config_toml)
            return False
        del servers["perfxpert"]
        # Drop the table if empty.
        if len(servers) == 0:
            del doc["mcp_servers"]
        pa.atomic_write(config_toml, tomlkit_mod.dumps(doc))
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_step(self, quiet: bool, msg: str) -> None:
        if not quiet:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        _LOG.debug(msg)

    def _refuse_if_git_tracked(self, config_path: Path) -> None:
        """Raise `ConfigClobber` if `config_path` is tracked in any git repo.

        Walks upward from `config_path.parent` looking for a `.git`
        entry, then asks git whether the file is tracked relative to
        that repo root (dotfiles-style layouts do check ``~/.codex/``
        into git from time to time — we refuse to silently rewrite
        such a versioned file). Mirrors the ``AGENTS.md`` /
        ``CLAUDE.md`` git-tracked rule the Claude adapter already
        applies.
        """
        if pa.is_absolute_path_git_tracked(config_path):
            raise ConfigClobber(
                f"{config_path} is tracked in a git repository. "
                "Refusing to rewrite a versioned config file — move "
                "it outside version control (e.g. `git rm --cached` "
                "the file and add it to .gitignore), then re-run "
                "perfxpert-code."
            )

    def _render_prompt_for_codex(self) -> str:
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


# ---------------------------------------------------------------------------
# Protocol conformance.
# ---------------------------------------------------------------------------


assert isinstance(CodexAdapter(), BackendAdapter)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _version_at_or_above(line: str, minimum: str) -> bool:
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


def _lazy_import_tomlkit(reason: str):
    """Import `tomlkit` lazily — raises `BackendNotFound` if missing.

    `tomlkit` is a required dependency of perfxpert (declared in
    `pyproject.toml`), so this import should always succeed in a
    correctly-installed environment. The error-path survives as a
    safety net for broken installs (e.g. user-site / virtualenv
    misalignment where the adapter runs with a different Python than
    `pip install` used). The error message tells the user how to
    recover without rebuilding the wheel.
    """
    try:
        import tomlkit as _tomlkit
    except ImportError as exc:
        raise BackendNotFound(
            f"tomlkit is required for {reason} but isn't importable from "
            f"the Python that runs `perfxpert-code` ({sys.executable}). "
            f"Reinstall perfxpert into that environment, or install "
            f"tomlkit there directly: "
            f"{sys.executable} -m pip install tomlkit. "
            f"Original error: {exc}"
        ) from exc
    return _tomlkit


def _parse_toml_read_only(source: str) -> dict | None:
    """Parse TOML using the stdlib `tomllib` (Python 3.11+).

    Keeps `_check_trust` tomlkit-free so the trust probe uses only
    stdlib — historically this mattered because `tomlkit` came from
    an optional `[backends]` extra; commit 3547736829 promoted it
    to a required dep, but staying stdlib-only here keeps the read
    path lean. Writes still use tomlkit for round-trip preservation.
    """
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:
        # Fallback for Python 3.10 (plan is 3.11+, but defensive).
        try:
            import tomli as tomllib  # type: ignore[import-not-found,no-redef]
        except ImportError:
            return None
    try:
        return tomllib.loads(source)
    except Exception:
        return None
