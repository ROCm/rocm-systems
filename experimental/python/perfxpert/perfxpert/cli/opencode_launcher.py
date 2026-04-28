"""opencode_launcher — `perfxpert-code` entry point.

Launches the locally built or packaged bundled opencode binary with the
AMD-themed config directory.

In source/editable checkouts, `perfxpert-code install-patches` can build the
patched binary from the pinned `opencode` submodule. In packaged installs, the
launcher requires the bundled binary produced from that same pinned submodule.
Users who intentionally want their own upstream opencode binary can run
`perfxpert-code opencode ...`, which bypasses the PerfXpert bundle.

PERFXPERT_OPENCODE_PATH is only honored by the explicit
`perfxpert-code opencode ...` escape hatch. The default PerfXpert TUI path
does not use arbitrary opencode binaries.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Iterable

from perfxpert.tools._tooldep import require_tool

__all__ = [
    "main",
    "resolve_opencode_binary",
    "resolve_config_dir",
    "print_banner",
    "route_subcommand",
]


_BRANDING_NAME = "AMD ROCm PerfXpert"
_BRANDING_VERSION = "0.2.0"

# Known opencode subcommands (v1.4.x) — a single bare positional that matches
# one of these MUST be forwarded as a subcommand, not treated as the CWD.
# Derived from opencode/packages/opencode/src/cli/cmd/*.ts.
_OPENCODE_SUBCOMMANDS = frozenset({
    "account",
    "acp",
    "agent",
    "auth",      # legacy alias for account
    "config",
    "db",
    "debug",
    "export",
    "generate",
    "github",
    "import",
    "mcp",
    "models",
    "pr",
    "plug",
    "plugin",
    "providers",
    "run",
    "serve",
    "session",
    "stats",
    "tui",
    "web",
})

# Subcommands the perfxpert launcher handles itself (does NOT exec opencode).
# Kept in sync with `perfxpert/__main__.py`; descriptions are surfaced by
# `_print_perfxpert_help()` for bare `perfxpert-code --help` discovery.
_PERFXPERT_SUBCOMMANDS: "dict[str, str]" = {
    "analyze": "Analyze a rocprofiler-sdk trace database for GPU bottlenecks",
    "init": "First-run wizard: detect GPU, write config, and suggest rocprofv3 commands",
    "diff": "Compare baseline vs new trace databases and emit a diff report",
    "ci": "CI wrapper over diff; returns non-zero on regressions above a threshold",
    "config": "Show or set perfxpert configuration (~/.config/perfxpert/config.yaml)",
    "doctor": "Health check: verify MCP server, LLM providers, and dependencies",
    "install-patches": "(deprecated) Build the patched opencode submodule into perfxpert/_bundled/opencode",
    "opencode": "Run a user-owned upstream opencode binary explicitly",
    "providers": "List LLM providers and configuration status",
    "uninstall": "Reverse `perfxpert-code <backend>` install: remove MCP + AGENTS + gate hook",
}

# Subcommand names the launcher itself dispatches to `python -m perfxpert`
# or handles inline. These must short-circuit BEFORE resolve_opencode_binary()
# so they work on a fresh install without opencode on disk.
_PERFXPERT_DISPATCH_SUBCOMMANDS = frozenset(
    {"init", "diff", "ci", "doctor", "install-patches", "uninstall"}
)


# Cycle-2 Task 2: third-party agent backends the launcher dispatches to via
# `perfxpert.cli._backend.claude|gemini|codex` adapters. Codex is listed
# here so the dispatcher routes correctly, but the adapter itself ships
# in PR 2 — the PR-1 dispatcher raises NotImplementedError for codex.
_BACKEND_SUBCOMMANDS: frozenset[str] = frozenset({"claude", "codex", "gemini"})


def _perfxpert_version() -> str:
    try:
        import perfxpert

        return getattr(perfxpert, "__version__", _BRANDING_VERSION)
    except ImportError:
        return _BRANDING_VERSION


def _wellknown_opencode_paths() -> "list[Path]":
    """Canonical well-known install locations for a user-owned `opencode` binary.

    These are used only by `perfxpert-code opencode ...`. The default
    PerfXpert TUI path requires the bundled binary built from the pinned
    submodule.
    """
    home = Path.home()
    return [
        home / ".opencode" / "bin" / "opencode",
        home / ".local" / "bin" / "opencode",
        Path("/usr/local/bin/opencode"),
        Path("/opt/opencode/bin/opencode"),
    ]


def _repo_local_patched_opencode_paths() -> "list[Path]":
    """Candidate binaries from a locally built repo checkout.

    In editable/source installs the pinned ``experimental/python/perfxpert/opencode``
    submodule is the canonical patched source. After ``install-patches`` (or
    a direct ``build-bundled-opencode.sh`` run), prefer the built binary from
    that checkout before any packaged bundle or upstream install on disk.
    """
    package_root = Path(__file__).resolve().parents[2]
    submodule = package_root / "opencode" / "packages" / "opencode" / "dist"
    return [
        submodule / "opencode-linux-x64" / "bin" / "opencode",
        submodule / "opencode-linux-arm64" / "bin" / "opencode",
        submodule / "opencode-darwin-x64" / "bin" / "opencode",
        submodule / "opencode-darwin-arm64" / "bin" / "opencode",
        submodule / "opencode" / "bin" / "opencode",
    ]


def resolve_opencode_binary() -> Path:
    """Locate the PerfXpert-managed bundled opencode binary.

    Priority:

    1. locally built repo checkout under ``experimental/python/perfxpert/opencode``.
    2. ``perfxpert/_bundled/opencode`` built from the pinned submodule.
    3. Final actionable error with install command hint.
    """
    # Locally built patched repo checkout (editable/source installs).
    for candidate in _repo_local_patched_opencode_paths():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    # Bundled binary built during pip install.
    try:
        with resources.as_file(resources.files("perfxpert") / "_bundled" / "opencode") as p:
            if p.is_file() and os.access(p, os.X_OK):
                return p
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    raise FileNotFoundError(
        "bundled patched opencode binary not found. Reinstall perfxpert with the "
        "GitHub wrapper so the pinned opencode submodule is built:\n"
        '  REF=develop; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"\n'
        "For a user-owned upstream opencode binary, run: perfxpert-code opencode [args]"
    )


def resolve_user_opencode_binary() -> Path:
    """Locate a user-owned upstream opencode binary for `perfxpert-code opencode`."""
    override = os.environ.get("PERFXPERT_OPENCODE_PATH")
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(
                f"PERFXPERT_OPENCODE_PATH={override!r} does not exist. "
                "Correct the path or unset the variable to use normal opencode lookup."
            )
        if not os.access(p, os.X_OK):
            raise FileNotFoundError(
                f"PERFXPERT_OPENCODE_PATH={override!r} is not executable. "
                "Fix permissions or unset the variable to use normal opencode lookup."
            )
        return p

    for candidate in _wellknown_opencode_paths():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    try:
        require_tool("opencode", allow_install=True)
        on_path = shutil.which("opencode")
        if on_path:
            return Path(on_path)
    except Exception:
        pass

    raise FileNotFoundError(
        "user-owned opencode binary not found. Install it with:\n"
        "  curl -fsSL https://opencode.ai/install.sh | bash\n"
        "or set PERFXPERT_OPENCODE_PATH to point at an existing binary."
    )


def resolve_config_dir() -> Path:
    """Return the bundled AMD-themed opencode config directory."""
    try:
        with resources.as_file(resources.files("perfxpert") / "_bundled" / "opencode_config") as p:
            return Path(p)
    except (ModuleNotFoundError, FileNotFoundError) as e:
        raise FileNotFoundError("perfxpert/_bundled/opencode_config not found. " "Reinstall perfxpert.") from e


def print_banner(stream=sys.stderr) -> None:
    """Print the AMD PerfXpert banner."""
    version = _perfxpert_version()
    banner = f"""
\033[38;5;196m╔══════════════════════════════════════════════════════════╗
║                                                          ║
║                  AMD ROCm PerfXpert                      ║
║                     version {version:<10s}                   ║
║                                                          ║
║   Interactive GPU performance analysis and optimization  ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝\033[0m
"""
    stream.write(banner)


def _handle_version_flag(argv: Iterable[str]) -> bool:
    """If --version / -V appears, print AMD-branded version and exit."""
    for a in argv:
        if a in {"--version", "-V"}:
            version = _perfxpert_version()
            print(f"{_BRANDING_NAME} {version} (opencode wrapper)")
            return True
    return False


def _print_perfxpert_help(stream=None) -> None:
    """Print the AMD PerfXpert help banner listing perfxpert-owned subcommands.

    Called when `perfxpert-code --help` (or `-h`) appears BEFORE any
    recognized opencode subcommand. After this banner, control falls
    through to `opencode --help` for generic flags.

    ``stream`` resolves to the live ``sys.stdout`` on each call so pytest's
    capsys captures the banner correctly.
    """
    if stream is None:
        stream = sys.stdout
    version = _perfxpert_version()
    lines = [
        f"{_BRANDING_NAME} {version} — opencode wrapper",
        "",
        "Usage:",
        "  perfxpert-code [opencode-args]      Launch the AMD-branded opencode TUI",
        "  perfxpert-code opencode [args]      Run a user-owned upstream opencode binary",
        "  perfxpert-code --version | -V       Print perfxpert version and exit",
        "  perfxpert-code --help | -h          Show this banner, then opencode help",
        "",
        "perfxpert-owned subcommands (invoke via `perfxpert <subcommand>`):",
    ]
    for name, desc in _PERFXPERT_SUBCOMMANDS.items():
        lines.append(f"  {name:<10s}  {desc}")
    lines.append("")
    lines.append("Pass-through subcommands are routed to the bundled opencode binary.")
    lines.append("For opencode-native flags (stats / run / auth / models / debug), see below:")
    lines.append("")
    stream.write("\n".join(lines) + "\n")
    stream.flush()


def _help_flag_precedes_subcommand(argv: Iterable[str]) -> bool:
    """Return True when `--help` / `-h` appears with no prior positional token.

    A help flag AFTER a positional arg (e.g. `perfxpert-code run --help`) is
    a request for opencode's subcommand-specific help; we pass it through
    verbatim. Only the bare `perfxpert-code --help` case should print our
    banner.
    """
    for tok in argv:
        if tok in {"--help", "-h"}:
            return True
        if not tok.startswith("-"):
            return False
    return False


def _handle_help_flag(argv: Iterable[str]) -> bool:
    """If `--help` / `-h` is the first non-flag token, print the perfxpert
    banner. Return True so the caller can decide whether to fall through to
    `opencode --help` for generic flag details.
    """
    if _help_flag_precedes_subcommand(argv):
        _print_perfxpert_help()
        return True
    return False


def route_subcommand(argv: list[str]) -> tuple[str, list[str]]:
    """Classify argv for dispatch.

    Returns a tuple ``(kind, argv_out)`` where ``kind`` is one of:

      - ``"perfxpert"``: the first positional is a perfxpert-handled
        subcommand (``doctor``); ``argv_out`` is forwarded to
        ``python -m perfxpert``.
      - ``"backend"`` (cycle-2 Task 2): the first positional is a
        third-party agent backend (``claude`` / ``codex`` / ``gemini``);
        ``argv_out`` is forwarded to
        ``perfxpert.cli._backend_dispatch._exec_backend``.
      - ``"user_opencode"``: the first positional is ``opencode``;
        ``argv_out`` is forwarded to a user-owned upstream opencode binary.
      - ``"opencode_subcommand"``: the first positional is a known opencode
        subcommand; ``argv_out`` is forwarded verbatim to opencode.
      - ``"opencode_default"``: no recognized subcommand; ``argv_out`` is
        forwarded to opencode for interactive-TUI-from-CWD behavior.

    The distinction matters because opencode treats a single unrecognized
    positional as a CWD override, which produced the
    "Failed to change directory to ...doctor" bug reported in session
    ses_25e1.
    """
    # Find the first non-flag positional token (flags start with '-').
    first_positional: str | None = None
    for a in argv:
        if not a.startswith("-"):
            first_positional = a
            break

    if first_positional is None:
        return ("opencode_default", list(argv))

    if first_positional in _PERFXPERT_DISPATCH_SUBCOMMANDS:
        return ("perfxpert", list(argv))

    # Cycle-2 Task 2: route claude/codex/gemini to the backend dispatcher.
    # Checked BEFORE the opencode-subcommand table so a name collision
    # (if upstream opencode ever added one) still routes to the adapter.
    if first_positional in _BACKEND_SUBCOMMANDS:
        return ("backend", list(argv))

    if first_positional == "opencode":
        out = list(argv)
        for idx, token in enumerate(out):
            if token == "opencode":
                del out[idx]
                break
        return ("user_opencode", out)

    if first_positional in _OPENCODE_SUBCOMMANDS:
        return ("opencode_subcommand", list(argv))

    return ("opencode_default", list(argv))


def _run_install_patches(argv: list[str]) -> int:
    """Run scripts/build-bundled-opencode.sh.

    The script lives alongside the perfxpert source tree; from a wheel
    install it is reachable because pyproject includes the scripts/
    directory via the `perfxpert` package. We look for it relative to
    this module first, then walk up to the repo root for editable
    installs.
    """
    # Path discovery — search candidate locations for the build script.
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "scripts" / "build-bundled-opencode.sh",  # editable install
        Path.cwd() / "scripts" / "build-bundled-opencode.sh",                 # dev cwd
    ]
    script: Path | None = None
    for c in candidates:
        if c.is_file():
            script = c
            break

    # Task 6: deprecation notice — `install-patches` is an alias kept
    # for backward-compat; the first-class surface is
    # `perfxpert-code install --backend=<name>` (arrives with Task 7).
    if not os.environ.get("PERFXPERT_SILENCE_DEPRECATION"):
        sys.stderr.write(
            "\033[33mperfxpert-code install-patches: this command is a "
            "deprecated alias for the opencode bundle-build step.\n"
            "  Future releases will route `install` through per-backend "
            "adapters (claude/codex/gemini).\n"
            "  Silence: export PERFXPERT_SILENCE_DEPRECATION=1\033[0m\n"
        )

    if script is None:
        sys.stderr.write(
            "\033[31mperfxpert-code install-patches: build-bundled-opencode.sh not found.\n"
            "  Expected alongside perfxpert source at experimental/python/perfxpert/scripts/.\n"
            "  This command is for editable/source installs; wheel users should already\n"
            "  have a bundled binary — if not, reinstall perfxpert from source.\033[0m\n"
        )
        return 2

    # Forward remaining argv (e.g. --skip-install) to the build script.
    cmd = ["bash", str(script), *argv[1:]]
    print(f"perfxpert-code install-patches: running {script.name}")
    try:
        proc = subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        return 130
    return proc.returncode


def _inject_perfxpert_agent_for_run(argv_out: list[str]) -> list[str]:
    """Force ``--agent perfxpert`` on ``opencode run`` when the user did
    not pass ``--agent`` explicitly.

    opencode's ``run`` subcommand otherwise defaults to ``agent=build``,
    which ignores our bundled AGENTS.md (tool-priority gate + MCP wiring).
    We only inject for ``run``; other subcommands (``stats``, ``auth``,
    etc.) don't take an ``--agent`` flag.

    Users can still override by passing ``--agent <something>``
    explicitly. They can opt out of the auto-inject with
    ``PERFXPERT_NO_AUTO_AGENT=1``.
    """
    if os.environ.get("PERFXPERT_NO_AUTO_AGENT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return argv_out

    # First non-flag positional must be "run".
    run_idx: int | None = None
    for i, tok in enumerate(argv_out):
        if tok.startswith("-"):
            continue
        if tok == "run":
            run_idx = i
        break  # stop at first positional either way

    if run_idx is None:
        return argv_out

    # User already specified --agent somewhere?
    if any(tok == "--agent" or tok.startswith("--agent=") for tok in argv_out):
        return argv_out

    # Insert after `run`. argv is [maybe-flags..., 'run', message_tokens...].
    new_argv = list(argv_out)
    new_argv.insert(run_idx + 1, "--agent")
    new_argv.insert(run_idx + 2, "perfxpert")
    return new_argv


def _exec_perfxpert_subcommand(argv: list[str]) -> int:
    """Dispatch a perfxpert-owned subcommand.

    ``install-patches`` is handled inline (invokes the bundled build
    script); ``uninstall <backend>`` routes into the backend
    adapter's uninstall(); all other dispatch-subcommands shell out
    to ``python -m perfxpert <argv...>``.
    """
    if argv and argv[0] == "install-patches":
        return _run_install_patches(argv)
    if argv and argv[0] == "uninstall":
        return _run_uninstall(argv[1:])

    cmd = [sys.executable, "-m", "perfxpert", *argv]
    try:
        proc = subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        return 130
    return proc.returncode


def _run_uninstall(remaining_argv: list[str]) -> int:
    """Handle `perfxpert-code uninstall <backend>` (Task 8).

    Backends supported: claude, gemini, codex. Each backend's adapter
    is imported lazily so unrelated subcommands don't pay the import
    cost.
    """
    # Parse dispatcher flags (--yes / -y / --quiet).
    assume_yes = False
    quiet = False
    idx = 0
    while idx < len(remaining_argv):
        a = remaining_argv[idx]
        if a in ("--yes", "-y"):
            assume_yes = True
        elif a == "--quiet":
            quiet = True
        else:
            break
        idx += 1
    backend = remaining_argv[idx] if idx < len(remaining_argv) else None

    if backend is None:
        sys.stderr.write(
            "perfxpert-code uninstall: which backend?\n"
            "  Usage: perfxpert-code uninstall {claude,gemini,codex}\n"
        )
        return 2

    # Lazy import: avoid loading adapter modules for unrelated subcommands.
    if backend == "claude":
        from perfxpert.cli._backend.claude import ClaudeCodeAdapter

        adapter = ClaudeCodeAdapter()
    elif backend == "gemini":
        from perfxpert.cli._backend.gemini import GeminiAdapter

        adapter = GeminiAdapter()
    elif backend == "codex":
        from perfxpert.cli._backend.codex import CodexAdapter

        adapter = CodexAdapter()
    else:
        sys.stderr.write(
            f"perfxpert-code uninstall: unknown backend {backend!r}. "
            "Expected one of: claude, gemini, codex.\n"
        )
        return 2

    from pathlib import Path

    cwd = Path.cwd()

    # Dry-run preview + confirmation.
    plan = adapter.plan(cwd)
    if not quiet:
        sys.stderr.write(
            f"perfxpert-code uninstall {backend}: will remove\n"
        )
        for action in plan.actions:
            sys.stderr.write(f"    - (reverse) {action}\n")

    if not assume_yes and os.environ.get("PERFXPERT_ASSUME_CONSENT", "") not in {
        "1",
        "true",
        "yes",
    }:
        if not sys.stdin.isatty():
            sys.stderr.write(
                "perfxpert-code uninstall: non-interactive stdin; pass --yes or "
                "export PERFXPERT_ASSUME_CONSENT=1 to confirm.\n"
            )
            return 2
        sys.stderr.write(f"\nProceed? [y/N] ")
        sys.stderr.flush()
        try:
            answer = input().strip().lower()
        except EOFError:
            answer = ""
        if answer not in {"y", "yes"}:
            sys.stderr.write("aborted.\n")
            return 1

    report = adapter.uninstall(cwd)
    if not quiet:
        for action in report.actions:
            sys.stderr.write(f"  {action}\n")
        if report.skipped_due_to_drift:
            sys.stderr.write(
                "\nRefused to remove these files (marker drift):\n"
            )
            for p in report.skipped_due_to_drift:
                sys.stderr.write(f"  - {p}\n")
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for `perfxpert-code`.

    Dispatch order:
    1. ``--version`` / ``-V`` short-circuit (print + exit 0).
    2. ``--help`` / ``-h`` BEFORE any positional → print perfxpert banner,
       then fall through to ``opencode --help`` for generic flag details.
    3. First positional is a perfxpert-dispatch subcommand (``doctor``):
       exec ``python -m perfxpert`` directly, BEFORE resolving the
       opencode binary (so health checks work without opencode installed).
    4. First positional is a known opencode subcommand: forward verbatim
       to opencode, preserving the user's CWD.
    5. Otherwise: stage the bundled config dir and launch opencode TUI
       from there.
    """
    if argv is None:
        argv = sys.argv[1:]

    if _handle_version_flag(argv):
        return 0

    # Print our banner when `--help`/`-h` precedes any opencode subcommand.
    # We still fall through to `opencode --help` afterwards so the user sees
    # the generic opencode flag reference appended to our perfxpert summary.
    _printed_perfxpert_help = _handle_help_flag(argv)

    kind, argv_out = route_subcommand(argv)

    # perfxpert-owned subcommands (doctor) short-circuit before opencode is
    # resolved so that `perfxpert-code doctor` works even if opencode is not
    # installed yet.
    if kind == "perfxpert":
        return _exec_perfxpert_subcommand(argv_out)

    # Cycle-2 Task 2 — route to a third-party agent adapter
    # (claude / codex / gemini). The dispatcher handles recursion-guard,
    # help passthrough, and the stub "not implemented" responses until
    # Tasks 4b/5/10 register the real adapters.
    if kind == "backend":
        # argv_out[0] is the backend name (route_subcommand gated it).
        backend_name = argv_out[0]
        remaining = argv_out[1:]
        from perfxpert.cli._backend_dispatch import _exec_backend

        return _exec_backend(backend_name, remaining)

    if kind == "user_opencode":
        if _printed_perfxpert_help:
            return 0
        try:
            binary = resolve_user_opencode_binary()
        except FileNotFoundError as e:
            print(f"\033[31mperfxpert-code opencode: {e}\033[0m", file=sys.stderr)
            return 1
        try:
            proc = subprocess.run([str(binary), *argv_out], env=dict(os.environ), check=False)
        except KeyboardInterrupt:
            return 130
        return proc.returncode

    try:
        binary = resolve_opencode_binary()
        config_dir = resolve_config_dir()
    except FileNotFoundError as e:
        # If we already printed perfxpert help, it's the user's expected
        # happy path for discovery; don't surface a red-text error just
        # because opencode isn't on disk yet.
        if _printed_perfxpert_help:
            return 0
        print(f"\033[31mperfxpert-code: {e}\033[0m", file=sys.stderr)
        return 1

    # Banner to stderr so it doesn't pollute piped stdout
    if os.environ.get("PERFXPERT_CODE_NO_BANNER", "").strip().lower() not in {"1", "true", "yes"}:
        print_banner()

    # opencode 1.4.x discovers `opencode.json` from cwd (no `--config` flag).
    # Copy the bundled config into a stable per-user dir and `cd` there before exec,
    # so MCP wiring + AGENTS.md instructions apply without polluting the user's cwd.
    runtime_cfg_dir = _prepare_runtime_config_dir(config_dir)

    # Force `--agent perfxpert` on `run` when the user did
    # not override. opencode's `run` subcommand otherwise defaults to
    # `agent=build`, which loads the stock prompt; `perfxpert` loads
    # AGENTS.md with the tool-priority gate.  MUST run BEFORE `cmd` is
    # constructed from argv_out.
    argv_out = _inject_perfxpert_agent_for_run(argv_out)

    cmd = [str(binary), *argv_out]

    # Pass through most of the user env; opencode needs LLM API keys and rocprofv3 envs.
    # We do NOT use the EXECUTION-tool env whitelist here because opencode is the
    # user's interactive session and they explicitly consent to it.
    env = dict(os.environ)
    # Recursion guard marker (spec §5.8 / R10)
    env["PERFXPERT_IN_OPENCODE_SESSION"] = "1"
    # Disable opencode's auto-update check — it prompts with upstream branding
    # and, if confirmed, would replace our patched bundle with upstream.
    env.setdefault("OPENCODE_DISABLE_AUTOUPDATE", "1")

    # Point opencode at our bundled config regardless of cwd.
    # Previously `perfxpert-code run ...` preserved the user's CWD (so they
    # could reference files in their project), which meant opencode never
    # picked up the bundled opencode.json → users got the default `build`
    # agent instead of our `perfxpert` agent, and the tool-priority gate in
    # AGENTS.md never loaded. OPENCODE_CONFIG=<file> force-loads our
    # bundled opencode.json globally; the user's own opencode.json (if any)
    # still merges on top.
    bundled_cfg_file = runtime_cfg_dir / "opencode.json"
    if bundled_cfg_file.is_file() and "OPENCODE_CONFIG" not in env:
        env["OPENCODE_CONFIG"] = str(bundled_cfg_file)

    # Subcommand dispatch:
    #   - Known opencode subcommands (stats/run/auth/…) are NOT executed from
    #     the runtime-config dir because they operate on the user's project
    #     and must see the user's cwd. Pass them through verbatim.
    #   - Default/interactive launches stage into runtime_cfg_dir so opencode
    #     picks up the bundled opencode.json + MCP wiring.
    exec_cwd = None if kind == "opencode_subcommand" else str(runtime_cfg_dir)

    try:
        proc = subprocess.run(cmd, env=env, cwd=exec_cwd, check=False)
    except KeyboardInterrupt:
        return 130
    return proc.returncode


def _prepare_runtime_config_dir(src_config_dir: Path) -> Path:
    """Stage a read-only copy of the bundled config where opencode will pick it up.

    opencode 1.4.x has no `--config <path>` flag; it auto-discovers `opencode.json`
    from the current directory. We create a dedicated runtime dir so we can point
    opencode at our bundled config without forcing the user to run from a specific
    directory or clobber their own `opencode.json`.
    """
    import shutil
    import tempfile

    def _stage_into(runtime_dir: Path) -> Path:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        for f in src_config_dir.iterdir():
            if not f.is_file():
                continue
            target = runtime_dir / f.name
            if not target.exists() or target.read_bytes() != f.read_bytes():
                shutil.copy2(f, target)
        return runtime_dir

    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    ).expanduser()
    try:
        return _stage_into(cache_root / "perfxpert" / "opencode")
    except OSError:
        # Some test and sandbox environments expose a read-only HOME cache.
        # Fall back to a private temp dir so the launcher remains runnable.
        return _stage_into(Path(tempfile.mkdtemp(prefix="perfxpert-opencode-")))
