"""Safe inspection for TUI-only external workflow adapters.

External workflow adapters are intentionally advisory. They can contribute
capability hints, knowledge links, and MCP server descriptors to the active TUI
session, but import never installs packages, runs adapter scripts, imports
adapter modules, starts MCP servers, or overrides PerfXpert's gate, MCP, or
correctness policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse


MAX_CANDIDATE_FILES = 120
MAX_DEFERRED_FILES = MAX_CANDIDATE_FILES
MAX_SCANNED_DIRS = 512
MAX_SCAN_DEPTH = 8
MAX_CHILDREN_PER_DIR = 1024
MAX_TEXT_BYTES = 256 * 1024
_CLONE_COMPLETE_SENTINEL = ".perfxpert-adapter-clone-complete"
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "test",
    "tests",
    "build",
    "dist",
    "node_modules",
    "venv",
    ".venv",
}
_TEXT_SUFFIXES = {
    ".md",
    ".rst",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".py",
    ".sh",
}
_SPECIAL_NAMES = {
    ".mcp.json",
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "SKILL.md",
    "README",
    "README.md",
    "README.rst",
    "mcp.json",
    "opencode.json",
    "package.json",
    "pyproject.toml",
}
_PRIORITY_DIR_NAMES = {"docs", "doc", "knowledge", "skills", "agents"}
_SPARSE_TREE_PATTERNS = tuple(sorted(_SPECIAL_NAMES | _PRIORITY_DIR_NAMES))
_GIT_ENV_PASSTHROUGH = {
    "ALL_PROXY",
    "CURL_CA_BUNDLE",
    "GIT_SSL_CAINFO",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "no_proxy",
}
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_SECRET_KV_RE = re.compile(
    r"(?i)\b([A-Za-z0-9_.-]*(?:token|api[_-]?key|password|passwd|secret|credential|authorization)"
    r"[A-Za-z0-9_.-]*)=([^&\s#]+)"
)
_SECRET_HEADER_RE = re.compile(
    r"(?im)\b((?:proxy-)?authorization|[A-Za-z0-9_.-]*(?:token|api[_-]?key|secret)[A-Za-z0-9_.-]*)" r"\s*:\s*([^\r\n]+)"
)
_URL_USERINFO_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@")


class ExternalWorkflowError(RuntimeError):
    """Raised when an external workflow source cannot be safely inspected."""


class ExternalWorkflowUsageError(ExternalWorkflowError):
    """Raised for invalid user input or missing TUI consent."""


class ExternalWorkflowRuntimeError(ExternalWorkflowError):
    """Raised for IO, clone, cache, or manifest failures."""


@dataclass(frozen=True)
class ExternalWorkflowCapability:
    """A capability discovered in an external workflow source."""

    kind: str
    name: str
    source: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalWorkflowKnowledgeLink:
    """A candidate knowledge file the TUI agent may read as advisory context."""

    path: str
    reason: str


@dataclass(frozen=True)
class ExternalWorkflowMcpServer:
    """A discovered MCP server descriptor. Registration still needs consent."""

    name: str
    command: str | None
    args: list[str]
    source: str
    arg_count: int = 0
    args_redacted: bool = True
    activation: str = (
        "requires explicit active-session TUI consent before registration; "
        "persistent or global MCP config changes require a separate persistent-install request"
    )


@dataclass
class ExternalWorkflowAdapterPlan:
    """Serializable adapter plan handed back to the active TUI session."""

    adapter_id: str
    source: str
    source_kind: str
    materialized_path: str | None
    interactive_only: bool = True
    execution_allowed: bool = False
    authority: str = "advisory; PerfXpert MCP gate and correctness checks remain authoritative"
    target_host_scope: str = (
        "inspect advisory metadata on the workload host; any target-dependent tool execution "
        "requires a separate explicit approval"
    )
    capabilities: list[ExternalWorkflowCapability] = field(default_factory=list)
    workflow_hints: list[str] = field(default_factory=list)
    knowledge_links: list[ExternalWorkflowKnowledgeLink] = field(default_factory=list)
    mcp_servers: list[ExternalWorkflowMcpServer] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    inspected_files: list[str] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)
    manifest_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


def inspect_external_workflow(
    source: str,
    *,
    interactive: bool,
    allow_network: bool = False,
    cache_root: Path | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Inspect an external workflow source and return an advisory adapter plan.

    The function performs bounded metadata inspection only. It does not import
    Python modules from the source, run scripts, install packages, or register
    discovered MCP servers.
    """

    if not interactive:
        raise ExternalWorkflowUsageError(
            "external workflow import is TUI-interactive only; start perfxpert-code and ask inside the TUI"
        )
    if not _has_active_tui_session():
        raise ExternalWorkflowUsageError(
            "external workflow import is available only from an active perfxpert-code TUI session"
        )

    root = _default_cache_root() if cache_root is None else Path(cache_root).expanduser()
    materialized, source_kind, display_source, source_warnings = _materialize_source(
        source,
        cache_root=root,
        interactive=interactive,
        allow_network=allow_network,
    )
    adapter_id = _adapter_id(display_source)
    plan = ExternalWorkflowAdapterPlan(
        adapter_id=adapter_id,
        source=display_source,
        source_kind=source_kind,
        materialized_path=str(materialized),
        warnings=[
            "External workflow adapters are advisory TUI context only.",
            "PerfXpert does not execute install commands, scripts, or MCP servers during import.",
            *source_warnings,
        ],
    )
    plan.provenance.update(_read_git_provenance(materialized))

    files, scan_warnings = _candidate_files(materialized)
    plan.warnings.extend(scan_warnings)
    texts: list[tuple[Path, str]] = []
    for path in files:
        rel = _relpath(path, materialized)
        text = _read_text_limited(path)
        if text is None:
            continue
        plan.inspected_files.append(rel)
        texts.append((path, text))

    _detect_capabilities(plan, materialized, texts)
    _detect_mcp_servers(plan, materialized, texts)
    _detect_knowledge_links(plan, materialized, files)
    _detect_risky_install_hints(plan, texts)
    _derive_workflow_hints(plan)

    if persist:
        try:
            plan.manifest_path = str(_write_manifest(root, plan))
        except OSError as exc:
            raise ExternalWorkflowRuntimeError(f"failed to write adapter manifest: {exc}") from exc

    return plan.to_dict()


def _default_cache_root() -> Path:
    return _workflow_base_dir() / ".perfxpert" / "external-workflows"


def _workflow_base_dir() -> Path:
    return Path(os.environ.get("PERFXPERT_WORKLOAD_CWD", Path.cwd())).expanduser().resolve()


def _materialize_source(
    source: str,
    *,
    cache_root: Path,
    interactive: bool,
    allow_network: bool,
) -> tuple[Path, str, str, list[str]]:
    root = _safe_cache_root(cache_root)
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        if parsed.scheme != "https":
            raise ExternalWorkflowUsageError("external workflow URLs must use https://")
        if not parsed.hostname:
            raise ExternalWorkflowUsageError("external workflow URLs must include a hostname")
        if any(ord(ch) < 32 or ch.isspace() for ch in source):
            raise ExternalWorkflowUsageError("external workflow URLs must not contain whitespace or control characters")
        if parsed.username or parsed.password:
            raise ExternalWorkflowUsageError("external workflow URLs must not contain embedded credentials")
        if parsed.query or parsed.fragment:
            raise ExternalWorkflowUsageError("external workflow URLs must not contain query strings or fragments")
        if not allow_network:
            raise ExternalWorkflowUsageError("URL inspection requires explicit network consent from the active TUI")
        if not interactive:
            raise ExternalWorkflowUsageError("URL inspection is TUI-interactive only")
        display_source = _redact_url(source)
        dest = _safe_cache_child(root, "sources", _adapter_id(display_source))
        if _clone_is_complete(dest):
            _verify_cached_remote(dest, display_source)
            return dest, "https-url", display_source, [f"Reused cached source checkout at {dest}."]
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run_git_clone(source, display_source, dest)
        return dest, "https-url", display_source, [f"Cloned external source into {dest} for bounded inspection."]

    path = Path(source).expanduser()
    if not path.is_absolute():
        path = _workflow_base_dir() / path
    if not path.exists():
        raise ExternalWorkflowUsageError(f"external workflow source not found: {source}")
    _reject_symlink_ancestry(path)
    if path.is_symlink():
        raise ExternalWorkflowUsageError("external workflow source must not be a symlink")
    if path.is_file():
        raise ExternalWorkflowUsageError(
            "external workflow source must be a directory; pass the adapter root instead of a file"
        )
    elif not path.is_dir():
        raise ExternalWorkflowUsageError("external workflow source must be a directory")
    return path.resolve(), "local-path", str(path.resolve()), []


def _clone_is_complete(dest: Path) -> bool:
    return (
        dest.is_dir()
        and not dest.is_symlink()
        and (dest / ".git").is_dir()
        and (dest / _CLONE_COMPLETE_SENTINEL).is_file()
    )


def _verify_cached_remote(dest: Path, display_source: str) -> None:
    cmd = ["git", "-C", str(dest), "remote", "get-url", "origin"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
            env=_git_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExternalWorkflowRuntimeError(f"failed to verify cached external workflow source: {exc}") from exc
    actual = _redact_url((proc.stdout or "").strip())
    if proc.returncode != 0 or actual != display_source:
        raise ExternalWorkflowRuntimeError("cached external workflow source does not match requested URL")


def _run_git_clone(source: str, display_source: str, dest: Path) -> None:
    if dest.exists():
        try:
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        except OSError as exc:
            raise ExternalWorkflowRuntimeError(f"failed to replace cached external workflow source: {exc}") from exc
    with tempfile.TemporaryDirectory(prefix=f"{dest.name}-", dir=str(dest.parent)) as tmp:
        tmp_dest = Path(tmp) / "checkout"
        _run_git_clone_to_tmp(source, display_source, tmp_dest)
        try:
            (tmp_dest / _CLONE_COMPLETE_SENTINEL).write_text("ok\n", encoding="utf-8")
            tmp_dest.rename(dest)
        except FileExistsError:
            if _clone_is_complete(dest):
                _verify_cached_remote(dest, display_source)
                return
            raise ExternalWorkflowRuntimeError("cached external workflow source changed during clone")
        except OSError as exc:
            raise ExternalWorkflowRuntimeError(f"failed to finalize cached external workflow source: {exc}") from exc


def _run_git_clone_to_tmp(source: str, display_source: str, dest: Path) -> None:
    cmd = _git_command_prefix() + [
        "clone",
        "--depth",
        "1",
        "--filter",
        "blob:none",
        "--no-checkout",
        source,
        str(dest),
    ]
    _run_git_command(cmd, source, display_source, "clone")
    sparse_paths = _select_sparse_checkout_paths(dest, source, display_source)
    if not sparse_paths:
        return
    _run_git_command(
        _git_command_prefix()
        + [
            "-C",
            str(dest),
            "sparse-checkout",
            "set",
            "--no-cone",
            *sparse_paths,
        ],
        source,
        display_source,
        "configure sparse checkout",
    )
    _run_git_command(
        _git_command_prefix() + ["-C", str(dest), "checkout", "--quiet"],
        source,
        display_source,
        "checkout sparse source",
    )


def _select_sparse_checkout_paths(dest: Path, source: str, display_source: str) -> list[str]:
    cmd = _git_command_prefix() + [
        "-C",
        str(dest),
        "ls-tree",
        "-r",
        "--name-only",
        "HEAD",
        "--",
        *_SPARSE_TREE_PATTERNS,
    ]
    output = _run_git_command_output(cmd, source, display_source, "list source tree")
    candidates: list[str] = []
    for raw in output.splitlines():
        rel = raw.strip()
        if not _is_safe_repo_relative_path(rel):
            continue
        if not _is_candidate_repo_text_path(rel):
            continue
        candidates.append(rel)
    return _top_candidate_relative_paths(candidates)


def _run_git_command(cmd: list[str], source: str, display_source: str, action: str) -> None:
    _run_git_command_output(cmd, source, display_source, action)


def _run_git_command_output(cmd: list[str], source: str, display_source: str, action: str) -> str:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
            env=_git_env(),
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExternalWorkflowRuntimeError(f"failed to {action} external workflow source: {exc}") from exc
    if proc.returncode != 0:
        detail = _sanitize_clone_output((proc.stderr or proc.stdout or "").strip(), source, display_source)
        raise ExternalWorkflowRuntimeError(f"git {action} failed for external workflow source: {detail}")
    return proc.stdout or ""


def _git_command_prefix() -> list[str]:
    return [
        "git",
        "-c",
        "credential.helper=",
        "-c",
        "core.askPass=",
        "-c",
        "http.extraHeader=",
    ]


def _git_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "PATH",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TERM",
            "TMPDIR",
            *_GIT_ENV_PASSTHROUGH,
        }
    }
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "false",
            "SSH_ASKPASS": "false",
            "GCM_INTERACTIVE": "never",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "HOME": tempfile.gettempdir(),
            "XDG_CONFIG_HOME": tempfile.gettempdir(),
        }
    )
    return env


def _candidate_files(root: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    deferred_files: list[Path] = []
    deferred_capped = False
    warnings: list[str] = []
    stack = [(root, 0)]
    scanned_dirs = 0
    while stack and len(files) < MAX_CANDIDATE_FILES:
        current, depth = stack.pop()
        scanned_dirs += 1
        if scanned_dirs > MAX_SCANNED_DIRS:
            warnings.append(f"Stopped scanning after {MAX_SCANNED_DIRS} directories; adapter inspection is partial.")
            break
        children, child_warnings = _bounded_children(root, current)
        warnings.extend(child_warnings)
        next_dirs: list[Path] = []
        next_files: list[Path] = []
        for child in children:
            if child.is_symlink():
                continue
            if child.is_dir():
                if child.name in _SKIP_DIRS:
                    continue
                if depth >= MAX_SCAN_DEPTH:
                    warnings.append(
                        f"Skipped directories deeper than {MAX_SCAN_DEPTH} levels; adapter inspection is partial."
                    )
                    continue
                next_dirs.append(child)
                continue
            if _is_candidate_text_file(child):
                next_files.append(child)
        for child in sorted(next_files, key=lambda path: _candidate_priority(path, root)):
            priority, _ = _candidate_priority(child, root)
            if priority <= 2:
                files.append(child)
                if len(files) >= MAX_CANDIDATE_FILES:
                    break
            else:
                if len(deferred_files) < MAX_DEFERRED_FILES:
                    deferred_files.append(child)
                else:
                    deferred_capped = True
        for child in sorted(next_dirs, key=lambda path: _candidate_priority(path, root), reverse=True):
            stack.append((child, depth + 1))

    for child in sorted(deferred_files, key=lambda path: _candidate_priority(path, root)):
        if len(files) >= MAX_CANDIDATE_FILES:
            break
        files.append(child)

    if deferred_capped:
        warnings.append(
            f"Stopped collecting low-priority files after {MAX_DEFERRED_FILES}; adapter inspection is partial."
        )

    if len(files) >= MAX_CANDIDATE_FILES:
        warnings.append(f"Stopped scanning after {MAX_CANDIDATE_FILES} candidate files; adapter inspection is partial.")

    return files, sorted(set(warnings))


def _bounded_children(root: Path, current: Path) -> tuple[list[Path], list[str]]:
    warnings: list[str] = []
    priority_children = []
    seen: set[Path] = set()
    for name in sorted(_SPECIAL_NAMES | _PRIORITY_DIR_NAMES):
        child = current / name
        if child.exists() and not child.is_symlink():
            priority_children.append(child)
            seen.add(child)

    capped_children: list[Path] = []
    try:
        for child in current.iterdir():
            if child in seen:
                continue
            if len(capped_children) >= MAX_CHILDREN_PER_DIR:
                warnings.append(
                    f"Stopped reading {current} after {MAX_CHILDREN_PER_DIR} children; "
                    "adapter inspection is partial."
                )
                break
            capped_children.append(child)
    except OSError as exc:
        warnings.append(f"Could not read {current}: {exc}; adapter inspection is partial.")
        return [], warnings

    children = priority_children + capped_children
    return sorted(children, key=lambda path: _candidate_priority(path, root)), warnings


def _candidate_priority(path: Path, root: Path | None = None) -> tuple[int, str]:
    rel = _relative_candidate_path(path, root)
    parts = {part.lower() for part in rel.parts}
    if path.name in _SPECIAL_NAMES:
        return (0, rel.as_posix())
    if parts & _PRIORITY_DIR_NAMES:
        return (1, rel.as_posix())
    if path.suffix.lower() in {".json", ".toml", ".yaml", ".yml"}:
        return (2, rel.as_posix())
    return (3, rel.as_posix())


def _is_candidate_text_file(path: Path) -> bool:
    try:
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode):
            return False
        if st.st_size > MAX_TEXT_BYTES:
            return False
    except OSError:
        return False
    return path.name in _SPECIAL_NAMES or path.suffix.lower() in _TEXT_SUFFIXES


def _read_text_limited(path: Path) -> str | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = None
    try:
        fd = os.open(path, flags)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_TEXT_BYTES:
            return None
        data = os.read(fd, MAX_TEXT_BYTES + 1)
    except OSError:
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
    if len(data) > MAX_TEXT_BYTES:
        return None
    if b"\x00" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")


def _detect_capabilities(
    plan: ExternalWorkflowAdapterPlan,
    root: Path,
    texts: list[tuple[Path, str]],
) -> None:
    combined = "\n".join(f"{_relpath(path, root)}\n{text}" for path, text in texts).lower()
    detectors = [
        (
            "profiling",
            "profiling and counter collection",
            ("profile", "profiler", "rocprof", "rocprof-compute", "pmc", "counter"),
        ),
        (
            "source_correlation",
            "source-line performance attribution",
            ("source line", "source mapping", "line attribution", "pc sampling"),
        ),
        (
            "trace_inspection",
            "packet and trace inspection",
            ("packet", "trace inspection", "hsa packet", "aql", "assembly"),
        ),
        (
            "kernel_isolation",
            "kernel extraction and replay",
            ("kernel extract", "extract kernel", "replay kernel", "kernel replay"),
        ),
        (
            "correctness",
            "optimization validation workflow",
            ("correctness", "validate", "validation", "regression"),
        ),
        ("mcp_tools", "external MCP tools", ("mcpservers", "fastmcp", "mcp server", "-mcp")),
        ("agent_skill", "agent skill or prompt package", ("skill.md", "agents.md", "agent workflow")),
    ]
    for kind, name, needles in detectors:
        if any(needle in combined for needle in needles):
            plan.capabilities.append(
                ExternalWorkflowCapability(
                    kind=kind,
                    name=name,
                    source="text-scan",
                    details={"matched_keywords": [n for n in needles if n in combined]},
                )
            )

    for path, text in texts:
        if path.name == "pyproject.toml":
            scripts = _parse_toml_scripts(text)
            for script in scripts:
                lowered = script.lower()
                if "mcp" in lowered or "perf" in lowered or "prof" in lowered:
                    plan.capabilities.append(
                        ExternalWorkflowCapability(
                            kind="entry_point",
                            name=script,
                            source=_relpath(path, root),
                            details={"type": "project script"},
                        )
                    )
        if path.name == "package.json":
            for script in _parse_package_bins(text):
                plan.capabilities.append(
                    ExternalWorkflowCapability(
                        kind="entry_point",
                        name=script,
                        source=_relpath(path, root),
                        details={"type": "package bin"},
                    )
                )


def _detect_mcp_servers(
    plan: ExternalWorkflowAdapterPlan,
    root: Path,
    texts: list[tuple[Path, str]],
) -> None:
    for path, text in texts:
        if path.name not in {".mcp.json", "mcp.json", "opencode.json"}:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        servers = _extract_mcp_servers(data, path.name)
        if not isinstance(servers, dict):
            continue
        for name, raw in sorted(servers.items()):
            if not isinstance(raw, dict):
                continue
            command, arg_count = _mcp_command_and_arg_count(raw)
            if command is not None and not _is_safe_mcp_command(command):
                plan.warnings.append(f"Ignored unsafe MCP command for {name!r}; review manually before registration.")
                command = None
            if arg_count:
                plan.warnings.append(
                    f"Redacted {arg_count} MCP argument(s) for {name!r}; review source before registration."
                )
            plan.mcp_servers.append(
                ExternalWorkflowMcpServer(
                    name=str(name),
                    command=command,
                    args=[],
                    arg_count=arg_count,
                    args_redacted=arg_count > 0,
                    source=_relpath(path, root),
                )
            )


def _extract_mcp_servers(data: object, filename: str) -> object:
    if not isinstance(data, dict):
        return None
    if filename == "opencode.json" and "mcp" in data:
        return data.get("mcp")
    return data.get("mcpServers")


def _mcp_command_and_arg_count(raw: dict[str, object]) -> tuple[str | None, int]:
    command = raw.get("command")
    args = raw.get("args", [])
    arg_count = 0
    command_parts: list[str] = []
    arg_parts = [arg for arg in args if isinstance(arg, str)] if isinstance(args, list) else []
    if isinstance(command, list):
        command_parts = [part for part in command if isinstance(part, str)]
    elif isinstance(command, str):
        try:
            command_parts = shlex.split(command)
        except ValueError:
            command_parts = []
    command = None
    if command_parts and command_parts[0] == "env":
        env_command, env_arg_count = _mcp_env_wrapped_command(command_parts + arg_parts)
        return env_command, env_arg_count
    for idx, token in enumerate(command_parts):
        if _is_env_assignment_token(token) or _token_contains_secret_kv(token):
            arg_count += 1
            continue
        command = token
        arg_count += max(0, len(command_parts) - idx - 1)
        break
    arg_count += len(arg_parts)
    return command if isinstance(command, str) else None, arg_count


def _mcp_env_wrapped_command(command_parts: list[str]) -> tuple[str | None, int]:
    arg_count = 1
    idx = 1
    while idx < len(command_parts):
        token = command_parts[idx]
        if token == "--":
            arg_count += 1
            idx += 1
            break
        if token in {"-i", "--ignore-environment"}:
            arg_count += 1
            idx += 1
            continue
        if token in {"-u", "--unset"}:
            arg_count += 1
            idx += 1
            if idx < len(command_parts):
                arg_count += 1
                idx += 1
            continue
        if _is_env_assignment_token(token) or _token_contains_secret_kv(token):
            arg_count += 1
            idx += 1
            continue
        if token.startswith("-"):
            return None, len(command_parts)
        return token, arg_count + max(0, len(command_parts) - idx - 1)
    return None, arg_count


def _detect_knowledge_links(
    plan: ExternalWorkflowAdapterPlan,
    root: Path,
    files: list[Path],
) -> None:
    for path in files:
        rel = _relpath(path, root)
        rel_lower = rel.lower()
        rel_parts = set(rel_lower.split("/"))
        if rel_parts & {"doc", "docs", "knowledge"}:
            plan.knowledge_links.append(
                ExternalWorkflowKnowledgeLink(
                    path=rel,
                    reason="candidate advisory knowledge for the active optimization session",
                )
            )
        elif path.name in {"SKILL.md", "AGENTS.md"}:
            plan.knowledge_links.append(
                ExternalWorkflowKnowledgeLink(
                    path=rel,
                    reason="agent instructions that may describe reusable workflow logic",
                )
            )


def _detect_risky_install_hints(
    plan: ExternalWorkflowAdapterPlan,
    texts: list[tuple[Path, str]],
) -> None:
    risky = (
        "pip install",
        "uv tool install",
        "npm install",
        "curl -",
        "| bash",
        "bash <(",
        "docker run",
    )
    hits: list[str] = []
    for path, text in texts:
        lowered = text.lower()
        if any(token in lowered for token in risky):
            hits.append(path.name)
    if hits:
        plan.warnings.append(
            "Install or execution snippets were found in "
            + ", ".join(sorted(set(hits)))
            + "; do not run them until the user explicitly approves."
        )


def _derive_workflow_hints(plan: ExternalWorkflowAdapterPlan) -> None:
    kinds = {cap.kind for cap in plan.capabilities}
    if "profiling" in kinds:
        plan.workflow_hints.append(
            "Map discovered profiling logic to PerfXpert's profile/analyze/reprofile phases; "
            "do not replace rocprofv3 validation."
        )
    if "source_correlation" in kinds:
        plan.workflow_hints.append(
            "Use source-correlation hints to prioritize files and kernels for inspection after MCP analysis."
        )
    if "kernel_isolation" in kinds:
        plan.workflow_hints.append(
            "Treat kernel replay or extraction tools as optional execution steps that require user consent."
        )
    if "correctness" in kinds:
        plan.workflow_hints.append(
            "Fold external validation ideas into the PerfXpert correctness gate; "
            "the gate verdict remains authoritative."
        )
    if plan.mcp_servers:
        plan.workflow_hints.append(
            "Discovered MCP servers can be added to the active TUI backend only after explicit registration consent."
        )
    if plan.knowledge_links:
        plan.workflow_hints.append(
            "Read knowledge links as supplemental context; they must not override measured counters or GPU specs."
        )
    if not plan.workflow_hints:
        plan.workflow_hints.append(
            "No specific optimization workflow hooks were detected; use inspected files as advisory context only."
        )


def _parse_toml_scripts(text: str) -> list[str]:
    scripts: list[str] = []
    in_scripts = False
    header_re = re.compile(r"^\s*\[([^\]]+)\]\s*$")
    entry_re = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*=\s*[\"']")
    for line in text.splitlines():
        header = header_re.match(line)
        if header:
            in_scripts = header.group(1) in {"project.scripts", "tool.poetry.scripts"}
            continue
        if not in_scripts:
            continue
        match = entry_re.match(line)
        if match:
            scripts.append(match.group(1))
    return scripts


def _parse_package_bins(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    raw = data.get("bin") if isinstance(data, dict) else None
    if isinstance(raw, str):
        return [Path(raw).name]
    if isinstance(raw, dict):
        return sorted(str(name) for name in raw)
    return []


def _is_safe_mcp_command(command: object) -> bool:
    if not isinstance(command, str):
        return False
    if not command.strip():
        return False
    return not any(ch in command for ch in "|&;<>()`$\n\r")


def _write_manifest(cache_root: Path, plan: ExternalWorkflowAdapterPlan) -> Path:
    root = _safe_cache_root(cache_root)
    manifest_dir = _safe_cache_child(root, "adapters")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = _safe_cache_child(root, "adapters", f"{plan.adapter_id}.json")
    if path.is_symlink():
        raise ExternalWorkflowRuntimeError(f"manifest path must not be a symlink: {path}")
    plan.manifest_path = str(path)
    payload = json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n"
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{plan.adapter_id}.", suffix=".tmp", dir=str(manifest_dir))
    tmp_path = Path(raw_tmp)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(tmp_path, path)
    except OSError:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    return path


def _safe_cache_root(cache_root: Path) -> Path:
    root = Path(cache_root).expanduser()
    for candidate in _existing_cache_paths(root):
        if candidate.is_symlink():
            raise ExternalWorkflowRuntimeError(f"cache path must not be a symlink: {candidate}")
    return root


def _safe_cache_child(cache_root: Path, *parts: str) -> Path:
    root_abs = cache_root.resolve(strict=False)
    child = cache_root.joinpath(*parts)
    for existing in [child, child.parent]:
        if existing.exists() and existing.is_symlink():
            raise ExternalWorkflowRuntimeError(f"cache path must not be a symlink: {existing}")
    child_abs = child.resolve(strict=False)
    try:
        child_abs.relative_to(root_abs)
    except ValueError as exc:
        raise ExternalWorkflowRuntimeError("cache path escapes adapter cache root") from exc
    return child


def _existing_cache_paths(path: Path) -> list[Path]:
    paths: list[Path] = []
    current = path
    while current != current.parent:
        if current.exists():
            paths.append(current)
        current = current.parent
    return paths


def _adapter_id(source: str) -> str:
    parsed = urlparse(source)
    base = Path(parsed.path if parsed.scheme else source).name or "workflow"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", base).strip("-._").lower() or "workflow"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}"


def _redact_url(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme and parsed.netloc:
        host = parsed.hostname or "<redacted-host>"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port = parsed.port
        except ValueError:
            port = None
        netloc = f"{host}:{port}" if port else host
        source = urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))
    return _redact_secret_kv(source)


def _sanitize_clone_output(detail: str, source: str, display_source: str) -> str:
    if not detail:
        return "no diagnostic output"
    sanitized = detail.replace(source, display_source)
    sanitized = _redact_url_userinfo(sanitized)
    sanitized = _redact_secret_headers(sanitized)
    return _redact_secret_kv(sanitized)


def _read_git_provenance(root: Path) -> dict[str, str]:
    if not (root / ".git").exists():
        return {}
    provenance: dict[str, str] = {}
    for key, cmd in {
        "commit": ["git", "-C", str(root), "rev-parse", "HEAD"],
        "remote": ["git", "-C", str(root), "remote", "get-url", "origin"],
    }.items():
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
                env=_git_env(),
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            value = proc.stdout.strip()
            provenance[key] = _redact_url(value) if key == "remote" else value
    return provenance


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _has_active_tui_session() -> bool:
    try:
        from perfxpert.cli._tui_session import has_active_tui_session
    except ImportError:
        return False
    return has_active_tui_session(os.environ)


def _reject_symlink_ancestry(path: Path) -> None:
    for candidate in [*reversed(path.parents), path]:
        if not candidate.exists():
            continue
        try:
            if candidate.is_symlink():
                raise ExternalWorkflowUsageError("external workflow source path must not contain symlinks")
        except OSError as exc:
            raise ExternalWorkflowUsageError(f"could not inspect external workflow source path: {exc}") from exc


def _relative_candidate_path(path: Path, root: Path | None) -> Path:
    if root is None:
        return path
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _is_safe_repo_relative_path(path: str) -> bool:
    if not path or path.startswith("/") or "\x00" in path:
        return False
    if any(ch in path for ch in "*?[\\"):
        return False
    parts = path.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _is_candidate_repo_text_path(path: str) -> bool:
    pure = Path(path)
    return pure.name in _SPECIAL_NAMES or pure.suffix.lower() in _TEXT_SUFFIXES


def _top_candidate_relative_paths(paths: list[str]) -> list[str]:
    unique = sorted(set(paths), key=_relative_path_priority)
    return unique[:MAX_CANDIDATE_FILES]


def _relative_path_priority(path: str) -> tuple[int, str]:
    pure = Path(path)
    parts = {part.lower() for part in pure.parts}
    if pure.name in _SPECIAL_NAMES:
        return (0, path)
    if parts & _PRIORITY_DIR_NAMES:
        return (1, path)
    if pure.suffix.lower() in {".json", ".toml", ".yaml", ".yml"}:
        return (2, path)
    return (3, path)


def _is_env_assignment_token(token: str) -> bool:
    return bool(_ENV_ASSIGNMENT_RE.match(token))


def _token_contains_secret_kv(token: str) -> bool:
    return bool(_SECRET_KV_RE.search(token))


def _redact_secret_kv(text: str) -> str:
    return _SECRET_KV_RE.sub(r"\1=<redacted>", text)


def _redact_secret_headers(text: str) -> str:
    return _SECRET_HEADER_RE.sub(r"\1: <redacted>", text)


def _redact_url_userinfo(text: str) -> str:
    return _URL_USERINFO_RE.sub(r"\1<redacted>@", text)
