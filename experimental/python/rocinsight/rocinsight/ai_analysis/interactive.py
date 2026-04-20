"""Interactive session for rocinsight analyze --interactive."""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import tempfile
import warnings
import dataclasses
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from .llm_conversation import LLMConversation
from .llm_analyzer import load_reference_guide

# ── Command safety ───────────────────────────────────────────────────────────
_SHELL_META = set(";|&`$(){}!")


def _has_shell_meta(cmd: str) -> bool:
    """Return True if *cmd* contains shell metacharacters that could enable injection."""
    return any(c in cmd for c in _SHELL_META)


# ── Plateau detection constants ──────────────────────────────────────────────
_PLATEAU_THRESHOLD_PCT = 2.0  # <2% change = plateau
_PLATEAU_MIN_ITERATIONS = 2   # need 2+ sub-threshold iterations to declare plateau
_MAX_SEEN_REC_HASHES = 200    # cap to prevent unbounded growth

# ── Session data ─────────────────────────────────────────────────────────────


@dataclass
class PersistentMenuItem:
    """A recommendation promoted to the main menu from a previous analysis."""

    id: str
    title: str
    priority: str  # "HIGH" | "MEDIUM" | "LOW"
    source: str  # "profiling_analysis" | "code_change_analysis"
    added_at: str  # ISO-8601
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoryEntry:
    type: str  # "profiling_run" | "code_change"
    timestamp: str
    db_path: str = ""
    files_modified: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class SessionContext:
    """Accumulated analysis context stored inside a SessionData.

    Tracks iteration count, per-run analysis summaries, suggestions given to
    the user, and profiling commands that have been executed.  Serialized as a
    plain dict inside ``SessionData.context`` for backward compatibility with
    sessions written before this field existed.
    """

    iteration: int = 0
    analyses: List[Dict[str, Any]] = field(default_factory=list)
    suggestions_given: List[str] = field(default_factory=list)
    commands_run: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SessionData:
    session_id: str
    source_dir: str
    created_at: str
    last_updated: str
    history: List[HistoryEntry] = field(default_factory=list)
    persistent_menu_items: List[PersistentMenuItem] = field(default_factory=list)
    conversation: Optional[Dict[str, Any]] = None  # serialized LLMConversation
    sent_source_files: List[str] = field(
        default_factory=list
    )  # files already sent to LLM
    context: Optional[Dict[str, Any]] = None  # serialized SessionContext

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionData":
        history = [HistoryEntry(**h) for h in d.get("history", [])]
        items = [PersistentMenuItem(**m) for m in d.get("persistent_menu_items", [])]
        return cls(
            session_id=d["session_id"],
            source_dir=d["source_dir"],
            created_at=d["created_at"],
            last_updated=d["last_updated"],
            history=history,
            persistent_menu_items=items,
            conversation=d.get("conversation"),  # None if key absent (backward compat)
            sent_source_files=d.get(
                "sent_source_files", []
            ),  # empty list for old sessions
            context=d.get("context"),  # None for sessions written before this field
        )


# ── SessionStore ──────────────────────────────────────────────────────────────

_DEFAULT_SESSIONS_DIR = pathlib.Path.home() / ".rocinsight" / "sessions"


class SessionStore:
    """Handles session file I/O under sessions_dir."""

    def __init__(self, sessions_dir: Optional[Union[str, pathlib.Path]] = None) -> None:
        self._dir = pathlib.Path(sessions_dir) if sessions_dir else _DEFAULT_SESSIONS_DIR

    def _path_for(self, session_id: str) -> pathlib.Path:
        return self._dir / f"{session_id}.json"

    def save(self, data: SessionData) -> pathlib.Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        p = self._path_for(data.session_id)
        p.write_text(json.dumps(data.to_dict(), indent=2))
        return p

    def load(self, id_or_path: str) -> Optional[SessionData]:
        """Load by session ID or by absolute/relative file path."""
        try:
            candidate = pathlib.Path(id_or_path)
            if candidate.exists():
                raw = json.loads(candidate.read_text())
                return SessionData.from_dict(raw)
            p = self._path_for(id_or_path)
            if p.exists():
                raw = json.loads(p.read_text())
                return SessionData.from_dict(raw)
            return None
        except Exception as exc:
            warnings.warn(f"Failed to load session {id_or_path!r}: {exc}", stacklevel=2)
            return None

    def find_by_source_dir(self, source_dir: str) -> List[SessionData]:
        """Return all sessions whose source_dir matches, newest first."""
        if not self._dir.exists():
            return []
        results: List[SessionData] = []
        for f in self._dir.glob("*.json"):
            try:
                raw = json.loads(f.read_text())
                if raw.get("source_dir") == source_dir:
                    results.append(SessionData.from_dict(raw))
            except Exception:
                pass

        def _safe_dt(s):
            try:
                dt = datetime.fromisoformat(s.created_at)
                # Ensure timezone-aware so comparison with the fallback doesn't fail
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                return datetime.min.replace(tzinfo=timezone.utc)

        return sorted(results, key=_safe_dt, reverse=True)

    @staticmethod
    def make_session_id(source_dir: str) -> str:
        slug = pathlib.Path(source_dir).name.replace(" ", "_")[:24] or "session"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        return f"{ts}_{slug}"


# ── Rendering helpers ─────────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.panel import Panel

    _RICH = True
except ImportError:
    _RICH = False

_console = Console() if _RICH else None

# AMD brand palette
_AMD_RED = "#E0001A"           # exact AMD red
_AMD_RED_STYLE = f"bold {_AMD_RED}"
_PRI_STYLE = {"HIGH": "bold red", "MEDIUM": "yellow", "LOW": "green", "INFO": "blue"}

# ANSI constants for non-Rich fallback
_A_RED    = "\033[91m"   # bright red ≈ AMD red
_A_BOLD   = "\033[1m"
_A_DIM    = "\033[2m"
_A_WHITE  = "\033[97m"
_A_RESET  = "\033[0m"


def _print(msg: str = "", style: str = "") -> None:
    """Print a line with a uniform style.  markup=False so user-content brackets
    like kernel names or file paths are never accidentally parsed as Rich tags."""
    if _RICH and _console:
        _console.print(msg, style=style or None, markup=False)
    else:
        print(msg)


def _print_m(msg: str = "") -> None:
    """Print a line that may contain Rich markup (for static, trusted strings only)."""
    if _RICH and _console:
        _console.print(msg)
    else:
        # Strip Rich markup tags for plain-text fallback using a simple regex
        import re
        plain = re.sub(r"\[/?[^\]]+\]", "", msg)
        print(plain)


def _menu_opt(key: str, desc: str, dim_suffix: str = "") -> None:
    """Print a colored menu option:  [key]  desc  dim_suffix.

    In Rich markup ``\\[`` renders as a literal ``[``, so ``\\[key]``
    produces the styled ``[key]`` bracket pair.
    """
    suffix = f"  [dim]{dim_suffix}[/dim]" if dim_suffix else ""
    _print_m(
        f"  [bold {_AMD_RED}]\\[{key}][/bold {_AMD_RED}]"
        f"  {desc}{suffix}"
    )


def _print_diff_line(line: str) -> None:
    """Print one unified-diff line with appropriate color."""
    if not _RICH or not _console:
        print(line)
        return
    if line.startswith("+++") or line.startswith("---"):
        _console.print(line, style="bold white", markup=False)
    elif line.startswith("@@"):
        _console.print(line, style=f"bold {_AMD_RED}", markup=False)
    elif line.startswith("+"):
        _console.print(line, style="bold green", markup=False)
    elif line.startswith("-"):
        _console.print(line, style="bold red", markup=False)
    else:
        _console.print(line, style="dim", markup=False)


def _priority_badge(pri: str) -> str:
    """Return a Rich-markup colored priority badge string."""
    colors = {"HIGH": "bold red", "MEDIUM": "bold yellow", "LOW": "bold green", "INFO": "bold blue"}
    style = colors.get(pri.upper(), "white")
    # \\[ renders as literal [ in Rich markup so [HIGH] etc. are not consumed as tags
    return f"[{style}]\\[{pri}][/{style}]"


def _input(prompt: str) -> str:
    return input(prompt)


def _print_token(t: str) -> None:
    """Stream a single LLM token to stdout without newline."""
    print(t, end="", flush=True)


class _Spinner:
    """Context manager: show a Rich spinner or a plain 'working…' line."""

    def __init__(self, msg: str) -> None:
        self._msg = msg
        self._status = None

    def __enter__(self):
        if _RICH and _console:
            self._status = _console.status(self._msg, spinner="dots")
            self._status.__enter__()
        else:
            print(self._msg, flush=True)
        return self

    def __exit__(self, *args):
        if self._status is not None:
            self._status.__exit__(*args)


# ── AMD ROCm logo banner ──────────────────────────────────────────────────────


def _render_logo_halfblock(width: int = 66, threshold: int = 70) -> Optional[str]:
    """Convert the AMD ROCm logo PNG to half-block ANSI art (2 px per char row).

    Uses the white-variant PNG bundled in share/.  Alpha channel encodes logo
    density; each pixel pair (top/bottom) maps to ▀ / ▄ / █ / space.
    All logo pixels are rendered in AMD red (\033[31m).
    Returns None if PIL is unavailable or the logo file is missing.
    """
    try:
        from PIL import Image  # type: ignore[import]

        share_dir = pathlib.Path(__file__).parent / "share"
        logo_path = share_dir / "amd_rocm_logo.png"
        if not logo_path.exists():
            return None

        img = Image.open(str(logo_path)).convert("RGBA")

        # Scale to requested width; account for char cell ~2:1 height:width ratio
        height_px = max(8, int(img.height / img.width * width))
        # Make even so each pair of rows maps cleanly to one character row
        if height_px % 2:
            height_px += 1
        img = img.resize((width, height_px), Image.LANCZOS)

        RED = "\033[31m"
        RESET = "\033[0m"
        lines: List[str] = []

        for y_char in range(height_px // 2):
            row = "  "  # leading indent
            for x in range(width):
                top_a = img.getpixel((x, y_char * 2))[3]
                bot_a = img.getpixel((x, y_char * 2 + 1))[3]
                top = top_a > threshold
                bot = bot_a > threshold
                if top and bot:
                    row += f"{RED}█{RESET}"
                elif top:
                    row += f"{RED}▀{RESET}"
                elif bot:
                    row += f"{RED}▄{RESET}"
                else:
                    row += " "
            if row.strip():
                lines.append(row)

        return "\n".join(lines) if lines else None

    except Exception:
        return None


def _strip_code_preamble(text: str) -> str:
    """Strip any non-code preamble that an LLM may have prefixed before the source.

    Some models prepend an explanation such as "Here's the complete optimized
    file." before the actual code.  This helper finds the first line that looks
    like the start of a C/C++/HIP source file and discards everything before it.

    A fenced code block (```...```) is also unwrapped if present.
    """
    import re as _re

    # Unwrap a fenced code block first (```cpp ... ``` or ``` ... ```)
    fence_match = _re.search(r"```[a-zA-Z]*\n(.*?)```", text, _re.DOTALL)
    if fence_match:
        return fence_match.group(1).rstrip()

    # Scan line by line: return from the first line that looks like source code.
    # Valid code starts: preprocessor directives, C/C++ comments, shebangs, or
    # leading keywords/identifiers.
    _CODE_LINE = _re.compile(
        r"^[ \t]*(#\s*(include|define|pragma|ifndef|ifdef|endif|if\s)|"  # preprocessor
        r"//|/\*|"                                                        # C/C++ comments
        r"#!|"                                                            # shebang
        r"(namespace|using|class|struct|typedef|int|void|float|double"
        r"|extern|static|inline|template|__global__|__device__|__host__)\b)",
    )
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if _CODE_LINE.match(line):
            if i == 0:
                return text  # already clean — no preamble
            return "".join(lines[i:])

    return text


def _apply_search_replace_blocks(original: str, llm_output: str) -> Optional[str]:
    """Parse <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks and apply them.

    Returns the modified file content, or None if parsing fails or a search
    block is not found in the original.
    """
    import re as _re

    # Strip any markdown fencing the LLM might have added
    fence_match = _re.search(r"```[a-zA-Z]*\n(.*?)```", llm_output, _re.DOTALL)
    if fence_match:
        llm_output = fence_match.group(1)

    blocks = _re.split(r"^<<<<<<< SEARCH\s*$", llm_output, flags=_re.MULTILINE)
    if len(blocks) < 2:
        return None  # No search/replace blocks found — not in chunk format

    result = original
    applied = 0
    for block in blocks[1:]:  # skip text before first SEARCH marker
        parts = _re.split(r"^=======\s*$", block, maxsplit=1, flags=_re.MULTILINE)
        if len(parts) != 2:
            continue
        search_text = parts[0]
        replace_parts = _re.split(
            r"^>>>>>>> REPLACE\s*$", parts[1], maxsplit=1, flags=_re.MULTILINE
        )
        if not replace_parts:
            continue
        replace_text = replace_parts[0]

        # Strip exactly one leading and one trailing newline from each block
        # (the markers themselves produce these)
        if search_text.startswith("\n"):
            search_text = search_text[1:]
        if search_text.endswith("\n"):
            search_text = search_text[:-1]
        if replace_text.startswith("\n"):
            replace_text = replace_text[1:]
        if replace_text.endswith("\n"):
            replace_text = replace_text[:-1]

        if not search_text:
            continue

        if search_text not in result:
            # Try with stripped whitespace on each line (fuzzy match)
            _stripped_search = "\n".join(l.rstrip() for l in search_text.splitlines())
            _stripped_result = "\n".join(l.rstrip() for l in result.splitlines())
            if _stripped_search in _stripped_result:
                # Find the position in the stripped version and map back
                idx = _stripped_result.index(_stripped_search)
                # Count lines up to that position
                _pre = _stripped_result[:idx]
                start_line = _pre.count("\n")
                search_lines = search_text.splitlines()
                orig_lines = result.splitlines(keepends=True)
                end_line = start_line + len(search_lines)
                # Replace those lines
                replace_lines = replace_text.splitlines(keepends=True)
                if replace_lines and not replace_lines[-1].endswith("\n"):
                    replace_lines[-1] += "\n"
                orig_lines[start_line:end_line] = replace_lines
                result = "".join(orig_lines)
                applied += 1
                continue
            _print(
                f"  ⚠  SEARCH block not found in file (block {applied + 1}).",
                style="yellow",
            )
            # Show first 80 chars of the search block for debugging
            _preview = search_text[:80].replace("\n", "\\n")
            _print(f"     Looking for: {_preview}...", style="dim")
            continue

        result = result.replace(search_text, replace_text, 1)
        applied += 1

    if applied == 0:
        return None
    _print(f"  Applied {applied} search/replace block(s).", style="dim")
    return result


def _replace_output_dir(cmd: str, new_dir: str) -> str:
    """Replace the -d <dir> argument in a rocprofv3 command with new_dir."""
    import shlex as _shlex
    import re as _re

    # Replace -d <value> token pair
    try:
        parts = _shlex.split(cmd)
    except ValueError:
        parts = cmd.split()
    out = []
    i = 0
    replaced = False
    while i < len(parts):
        if parts[i] in ("-d", "--output-path") and i + 1 < len(parts):
            out.extend([parts[i], new_dir])
            i += 2
            replaced = True
        else:
            out.append(parts[i])
            i += 1
    result = " ".join(_shlex.quote(p) if " " in p else p for p in out)
    if not replaced:
        # Append -d before the -- separator if present
        result = _re.sub(r"\s+--\s+", f" -d {new_dir} -- ", result, count=1)
    return result


def _print_startup_banner() -> None:
    """Print the AMD ROCm logo + session title once at interactive startup."""
    art = _render_logo_halfblock()
    if art:
        print()
        print(art)
        print()
        # Subtitle under the logo
        if _RICH and _console:
            _console.print(
                f"  [bold white]ROCInsight[/bold white]"
                f"  [dim]·[/dim]"
                f"  [bold {_AMD_RED}]AMD ROCm[/bold {_AMD_RED}] AI Analysis"
                f"  [dim]·  Interactive Session[/dim]"
            )
        else:
            print(
                f"  {_A_BOLD}{_A_WHITE}ROCInsight{_A_RESET}"
                f"  {_A_DIM}·{_A_RESET}"
                f"  {_A_BOLD}{_A_RED}AMD ROCm{_A_RESET} AI Analysis"
                f"  {_A_DIM}·  Interactive Session{_A_RESET}"
            )
        print()
    else:
        # Fallback: styled text banner (no logo image)
        _w = 68
        if _RICH and _console:
            _console.print(f"\n  [bold {_AMD_RED}]{'─' * _w}[/bold {_AMD_RED}]")
            _console.print(
                f"  [bold {_AMD_RED}]▌[/bold {_AMD_RED}]"
                f"  [bold white]ROCInsight[/bold white]"
                f"  [dim]AMD ROCm AI Analysis  ·  Interactive Session[/dim]"
            )
            _console.print(f"  [bold {_AMD_RED}]{'─' * _w}[/bold {_AMD_RED}]\n")
        else:
            print(f"\n  {_A_BOLD}{_A_RED}{'─' * _w}{_A_RESET}")
            print(
                f"  {_A_BOLD}{_A_RED}▌{_A_RESET}"
                f"  {_A_BOLD}{_A_WHITE}ROCInsight{_A_RESET}"
                f"  {_A_DIM}AMD ROCm AI Analysis  ·  Interactive Session{_A_RESET}"
            )
            print(f"  {_A_BOLD}{_A_RED}{'─' * _w}{_A_RESET}\n")


# ── InteractiveSession ────────────────────────────────────────────────────────


class InteractiveSession:
    """Top-level interactive menu for rocinsight analyze --interactive."""

    def __init__(
        self,
        source_dir: str,
        tier0_result: Optional[Any],
        recommendations: List[Dict[str, Any]],
        database_path: str,
        llm_provider: Optional[str],
        llm_api_key: Optional[str],
        llm_model: Optional[str],
        llm_local: Optional[str] = None,
        llm_local_model: Optional[str] = None,
        session_store: Optional[SessionStore] = None,
        resume_session_id: Optional[str] = None,
        compact_every: int = 10,
    ) -> None:
        self._source_dir = source_dir
        self._tier0 = tier0_result
        self._recs = recommendations or []
        self._db_path = database_path
        self._llm_provider = llm_provider
        self._llm_api_key = llm_api_key
        self._llm_model = llm_model
        self._llm_local = llm_local
        self._llm_local_model = llm_local_model
        self._store = session_store or SessionStore()
        self._compact_every = compact_every
        self._conv: Optional[LLMConversation] = None
        self._sent_source_files: set = set()  # filenames already sent to _conv
        self._session = self._init_session(resume_session_id)
        # Restore or create a SessionContext from the session's context dict
        raw_ctx = self._session.context or {}
        self._ctx = SessionContext(**raw_ctx) if raw_ctx else SessionContext()

    @property
    def session(self) -> SessionData:
        return self._session

    def _init_session(self, resume_id: Optional[str]) -> SessionData:
        # Explicit resume
        if resume_id:
            loaded = self._store.load(resume_id)
            if loaded:
                self._conv = self._restore_or_create_conv(loaded)
                self._sent_source_files = set(loaded.sent_source_files)
                return loaded

        # Auto-detect previous session for this source dir
        existing = self._store.find_by_source_dir(self._source_dir)
        if existing:
            chosen = self._prompt_resume(existing)
            if chosen:
                self._conv = self._restore_or_create_conv(chosen)
                self._sent_source_files = set(chosen.sent_source_files)
                return chosen

        # New session
        now = datetime.now(timezone.utc).isoformat()
        new_session = SessionData(
            session_id=SessionStore.make_session_id(self._source_dir),
            source_dir=self._source_dir,
            created_at=now,
            last_updated=now,
        )
        self._conv = self._make_fresh_conv(new_session.session_id)
        return new_session

    def _restore_or_create_conv(self, loaded: SessionData) -> Optional["LLMConversation"]:
        """Restore _conv from a loaded session, or create fresh if absent."""
        if not self._llm_provider:
            return None
        raw_conv = loaded.conversation
        if raw_conv:
            return LLMConversation.from_dict(
                raw_conv, api_key=self._llm_api_key, model=self._llm_model
            )
        return self._make_fresh_conv(loaded.session_id)

    def _make_fresh_conv(self, session_id: str) -> Optional["LLMConversation"]:
        """Create a new LLMConversation for a session, or None if no LLM configured."""
        if not self._llm_provider:
            return None
        hp = self._store._dir / f"{session_id}_history.jsonl"
        conv = LLMConversation(
            provider=self._llm_provider,
            api_key=self._llm_api_key,
            model=self._llm_model,
            compact_every=self._compact_every,
            history_path=hp,
        )
        try:
            fence = load_reference_guide()
        except Exception as e:
            warnings.warn(
                f"[LLMConversation] Could not load reference guide: {e}", stacklevel=3
            )
            fence = ""
        conv.initialize(
            "You are an expert AMD GPU performance engineer "
            "helping optimize a HIP/ROCm application.\n\n" + fence
        )
        return conv

    def _prompt_resume(self, existing: List[SessionData]) -> Optional[SessionData]:
        _print()
        _print_m(
            f"  [bold {_AMD_RED}]── Resume a previous session? [/bold {_AMD_RED}]"
            f"[dim]{'─' * 36}[/dim]"
        )
        _print(
            f"  Found {len(existing)} saved session(s) for {self._source_dir}:",
            style="dim",
        )
        _print()
        for i, s in enumerate(existing, 1):
            n_runs = sum(1 for h in s.history if h.type == "profiling_run")
            n_change = sum(1 for h in s.history if h.type == "code_change")
            n_items = len(s.persistent_menu_items)
            detail = (
                f"{n_runs} profiling run(s), {n_change} code change(s), "
                f"{n_items} saved recommendation(s)"
            )
            _menu_opt(str(i), s.session_id, detail)
        _print()
        _menu_opt("n", "Start new session  (or press Enter)")
        _print()
        choice = _input("  > ").strip().lower()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(existing):
                return existing[idx]
            _print("  Invalid selection — starting new session.", style="dim")
        elif choice not in ("n", ""):
            _print("  Unrecognized input — starting new session.", style="dim")
        return None

    def _render_main_menu(self) -> None:
        src_label = (
            pathlib.Path(self._source_dir).name if self._source_dir else "(no source)"
        )
        n_runs = sum(1 for h in self._session.history if h.type == "profiling_run")
        db_label = f"  db: {pathlib.Path(self._db_path).name}" if self._db_path else ""
        runs_label = f"  runs: {n_runs}" if n_runs else ""
        status_line = (
            f"[dim]{db_label}{runs_label}  \\[s] save  \\[q] quit[/dim]"
            if _RICH
            else f"{db_label}{runs_label}  [s] save  [q] quit"
        )
        if _RICH and _console:
            _console.print(
                Panel(
                    f"[bold]Source:[/bold] {src_label}   "
                    f"[bold]Session:[/bold] {self._session.session_id}   " + status_line,
                    title=f"[bold {_AMD_RED}]▌[/bold {_AMD_RED}] [bold white]ROCInsight[/bold white]  [dim]AMD ROCm AI Analysis[/dim]",
                    border_style=_AMD_RED,
                )
            )
        else:
            w = 70
            print(f"  {_A_BOLD}{_A_RED}{'─' * w}{_A_RESET}")
            print(
                f"  {_A_BOLD}{_A_RED}▌{_A_RESET}"
                f"  {_A_BOLD}{_A_WHITE}ROCInsight{_A_RESET}"
                f"  {_A_DIM}AMD ROCm AI Analysis{_A_RESET}"
                f"  {_A_DIM}|  {src_label}{_A_RESET}"
            )
            print(
                f"  {_A_DIM}Session: {self._session.session_id}"
                f"  {db_label}  [s] save  [q] quit{_A_RESET}"
            )
            print(f"  {_A_BOLD}{_A_RED}{'─' * w}{_A_RESET}")

        _print()
        _menu_opt("p", "Profile app  — run rocprofv3, collect .db")
        _menu_opt("a", "Analyze .db  — load existing trace and find bottlenecks")
        _menu_opt("o", "Optimize     — AI code optimization suggestions")
        _print()
        _menu_opt("s", "Save session")
        _menu_opt("q", "Quit")

        if self._session.persistent_menu_items:
            _print()
            if _RICH and _console:
                _console.print(
                    f"  [bold {_AMD_RED}]── Findings from this session [/bold {_AMD_RED}]"
                    + f"[dim]{'─' * 33}[/dim]"
                )
            else:
                print(
                    f"  {_A_BOLD}{_A_RED}── Findings from this session {_A_RESET}"
                    + f"{_A_DIM}{'─' * 33}{_A_RESET}"
                )
            for i, item in enumerate(self._session.persistent_menu_items, 1):
                pri = item.priority.upper()
                pri_style = _PRI_STYLE.get(pri, "white")
                src_tag = (
                    "  [code change]" if item.source == "code_change_analysis" else ""
                )
                if _RICH and _console:
                    _console.print(
                        f"  [bold {_AMD_RED}]\\[{i}][/bold {_AMD_RED}]  "
                        f"[{pri_style}][{pri}][/{pri_style}]  "
                        f"{item.title}{src_tag}"
                    )
                else:
                    print(
                        f"  {_A_BOLD}{_A_RED}[{i}]{_A_RESET}"
                        f"  [{pri}]  {item.title}{src_tag}"
                    )
        _print()

    def _path_analyze_db(self) -> None:
        """Prompt for a .db file, run Tier 1/2 analysis, show summary and add recommendations."""
        _print()
        if self._db_path:
            _print(f"  Current .db: {self._db_path}", style="dim")
            _print(
                "  Enter a .db path to analyze, or press Enter to re-analyze current:",
                style=_AMD_RED,
            )
        else:
            _print("  Enter path to a .db trace file:", style=_AMD_RED)
        try:
            db_input = _input("  > ").strip()
        except EOFError:
            return
        if not db_input and self._db_path:
            db_path = pathlib.Path(self._db_path)
        elif db_input:
            db_path = pathlib.Path(db_input).expanduser()
        else:
            return
        if not db_path.exists():
            _print(f"  File not found: {db_path}", style="red")
            return
        self._db_path = str(db_path)
        _print(f"  Running Tier 1/2 analysis on {db_path.name}...", style="dim")
        new_recs, breakdown = self._run_tier1_analysis(str(db_path))
        if new_recs:
            self._show_analysis_summary(new_recs)
        added = self._ingest_recommendations(new_recs)
        now = datetime.now(timezone.utc).isoformat()
        self._session.history.append(
            HistoryEntry(type="profiling_run", timestamp=now, db_path=str(db_path))
        )
        _print(f"  ✓ {added} new finding(s) added to menu.", style="green")

    def _show_analysis_summary(self, recs: List[Dict[str, Any]]) -> None:
        """Print a brief summary of findings after Tier 1/2 analysis."""
        if not recs:
            _print("  No significant bottlenecks found.", style="green")
            return
        high = [r for r in recs if r.get("priority", "").upper() == "HIGH"]
        med = [r for r in recs if r.get("priority", "").upper() == "MEDIUM"]
        _print()
        _print_m(f"  [bold {_AMD_RED}]── Analysis Summary ────────────────────────────────────[/bold {_AMD_RED}]")
        for r in high:
            issue = r.get("issue", r.get("title", ""))
            _print_m(f"  {_priority_badge('HIGH')}  [white]{issue}[/white]")
        for r in med:
            issue = r.get("issue", r.get("title", ""))
            _print_m(f"  {_priority_badge('MEDIUM')}  [white]{issue}[/white]")
        _print()

    def run(self) -> None:
        """Main event loop."""
        _print_startup_banner()
        try:
            self._run_loop()
        except KeyboardInterrupt:
            _print()
            _print("  Interrupted — saving session.", style="yellow")
            self._save_and_quit()

    def _run_loop(self) -> None:
        while True:
            self._render_main_menu()
            try:
                choice = _input("  Enter choice [p/a/o/s/q]: ").strip().lower()
            except EOFError:
                self._save_and_quit()
                break

            if choice == "q":
                self._save_and_quit()
                break
            elif choice == "s":
                if self._conv:
                    self._session.conversation = self._conv.to_dict()
                self._session.sent_source_files = list(self._sent_source_files)
                self._store.save(self._session)
                _print("  Session saved.", style="green")
            elif choice == "p":
                self._path_profiling()
            elif choice == "a":
                self._path_analyze_db()
            elif choice == "o":
                self._path_optimize()
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(self._session.persistent_menu_items):
                    self._pursue_recommendation(self._session.persistent_menu_items[idx])
            else:
                _print("  Unknown choice. Enter p, a, o, s, q, or a number.", style="dim")

    def _save_and_quit(self) -> None:
        self._session.last_updated = datetime.now(timezone.utc).isoformat()
        if self._conv:
            self._session.conversation = self._conv.to_dict()
        self._session.sent_source_files = list(self._sent_source_files)
        self._store.save(self._session)
        _print("  Session saved. Goodbye.", style=_AMD_RED)

    def _ingest_recommendations(
        self, new_recs: List[Dict[str, Any]], source: str = "profiling_analysis"
    ) -> int:
        """Add unique recommendations to persistent_menu_items. Returns count added."""
        now = datetime.now(timezone.utc).isoformat()
        existing_ids = {m.id for m in self._session.persistent_menu_items}
        added = 0
        for rec in new_recs:
            rid = rec.get("id", rec.get("category", ""))
            if rid and rid not in existing_ids:
                self._session.persistent_menu_items.append(
                    PersistentMenuItem(
                        id=rid,
                        title=rec.get("issue", rec.get("category", rid)),
                        priority=rec.get("priority", "INFO"),
                        source=source,
                        added_at=now,
                        detail=rec,
                    )
                )
                existing_ids.add(rid)
                added += 1
        return added

    def _path_profiling(self, _source: str = "profiling_analysis") -> None:
        """Show profiling commands; let user pick one to run; auto-ingest output .db."""
        _print()
        _print("  ── Profiling Commands ──────────────────────────────────", style=_AMD_RED)
        _print()

        cmds = self._collect_profiling_commands()

        # Optional LLM annotation on tier0 metadata (no source text uploaded)
        if self._llm_provider and self._tier0:
            cmds = self._llm_annotate_profiling_plan(cmds)

        if not cmds:
            _print("  (no profiling commands available)", style="dim")
            return

        for i, (label, cmd) in enumerate(cmds, 1):
            _menu_opt(str(i), label)
            _print(f"       $ {cmd}", style="dim")
            _print()

        _print("  Enter command number to run it, or Enter to skip:", style=_AMD_RED)
        try:
            choice = _input("  > ").strip()
        except EOFError:
            return

        if not choice:
            return

        if not choice.isdigit() or not (1 <= int(choice) <= len(cmds)):
            _print("  Invalid selection.", style="dim")
            return

        _, selected_cmd = cmds[int(choice) - 1]

        # If the command has '-- ./app', ask the user what their app invocation is
        if "-- ./app" in selected_cmd:
            auto = self._resolve_app_placeholder(selected_cmd)
            auto_app = auto.split("-- ", 1)[1] if "-- " in auto else ""
            hint = f" (default: {auto_app})" if auto_app and auto_app != "./app" else ""
            _print(f"  Enter application to profile{hint}:", style=_AMD_RED)
            _print(
                "  (e.g.  ./my_app --arg1 val1   or press Enter to use default)",
                style="dim",
            )
            try:
                app_input = _input("  > ").strip()
            except EOFError:
                return
            if app_input:
                selected_cmd = selected_cmd.replace("-- ./app", f"-- {app_input}")
            elif auto_app and auto_app != "./app":
                selected_cmd = auto
            # else leave as-is (./app stays in command; user will see it)

        if _has_shell_meta(selected_cmd):
            _print("  Command contains shell metacharacters — rejected for safety.", style="red")
            return

        _print()
        _print(f"  Running: $ {selected_cmd}", style=_AMD_RED)
        _print()

        proc = subprocess.run(shlex.split(selected_cmd))
        _print()
        if proc.returncode != 0:
            _print(f"  Command exited with code {proc.returncode}.", style="yellow")

        # Try to find the output .db automatically from the command flags
        db_path = self._find_output_db(selected_cmd)
        if db_path:
            _print(f"  Found output: {db_path}", style="green")
        else:
            _print(
                "  Enter path to the output .db file (or Enter to skip):", style=_AMD_RED
            )
            try:
                db_input = _input("  > ").strip()
            except EOFError:
                return
            if not db_input:
                return
            db_path = pathlib.Path(db_input).expanduser()
            if not db_path.exists():
                _print(f"  File not found: {db_path}", style="red")
                return

        _print("  Running Tier 1/2 analysis...", style="dim")
        new_recs, breakdown = self._run_tier1_analysis(str(db_path))
        if new_recs:
            self._show_analysis_summary(new_recs)
        added = self._ingest_recommendations(new_recs, source=_source)
        now = datetime.now(timezone.utc).isoformat()
        self._session.history.append(
            HistoryEntry(
                type="profiling_run",
                timestamp=now,
                db_path=str(db_path),
            )
        )
        self._db_path = str(db_path)
        _print(f"  ✓ {added} finding(s) added to menu.", style="green")

    def _resolve_app_placeholder(self, cmd: str) -> str:
        """Replace '-- ./app' placeholder with an actual binary found near source_dir."""
        if "-- ./app" not in cmd:
            return cmd
        # Look for any executable in source_dir (non-script, non-dot files)
        base = pathlib.Path(self._source_dir)
        for candidate in sorted(base.iterdir()):
            if (
                candidate.is_file()
                and os.access(str(candidate), os.X_OK)
                and not candidate.name.startswith(".")
                and candidate.suffix
                not in {
                    ".sh",
                    ".py",
                    ".md",
                    ".txt",
                    ".cpp",
                    ".hip",
                    ".cu",
                    ".h",
                    ".hpp",
                }
            ):
                return cmd.replace("-- ./app", f"-- {candidate}")
        return cmd  # leave as-is if nothing found

    def _find_output_db(self, cmd: str) -> Optional[pathlib.Path]:
        """Parse -d <dir> -o <base> from a rocprofv3 command and find the resulting .db."""
        import shlex

        try:
            parts = shlex.split(cmd)
        except ValueError:
            parts = cmd.split()

        out_dir = "."
        out_base = None
        for i, p in enumerate(parts):
            if p in ("-d", "--output-path") and i + 1 < len(parts):
                out_dir = parts[i + 1]
            elif p in ("-o", "--output-file") and i + 1 < len(parts):
                out_base = parts[i + 1]

        if out_base is None:
            return None

        # rocprofv3 creates <out_base>_results.db inside out_dir
        candidates = [
            pathlib.Path(out_dir) / f"{out_base}_results.db",
            pathlib.Path(out_dir) / f"{out_base}.db",
        ]
        for c in candidates:
            if c.exists():
                return c
        # Glob fallback
        import glob

        matches = sorted(glob.glob(str(pathlib.Path(out_dir) / f"{out_base}*.db")))
        if matches:
            return pathlib.Path(matches[0])
        return None

    def _collect_profiling_commands(self) -> List[tuple]:
        """Collect (label, full_command) pairs from tier0 and existing recommendations."""
        cmds: List[tuple] = []
        seen: set = set()

        def _add(label: str, cmd: str) -> None:
            if cmd and cmd not in seen:
                seen.add(cmd)
                cmds.append((label, cmd))

        if self._tier0:
            fc = getattr(self._tier0, "suggested_first_command", None)
            if fc:
                _add("Start Here — suggested first profiling command", fc)

        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
        for rec in sorted(
            self._recs, key=lambda r: priority_order.get(r.get("priority", "INFO"), 4)
        ):
            for cmd in rec.get("commands", []):
                fc = cmd.get("full_command", "")
                label = (
                    f"[{rec.get('priority', 'INFO')}] {rec.get('category', '')} — "
                    f"{cmd.get('tool', '')}: {cmd.get('description', '')}"
                )
                _add(label, fc)

        return cmds

    def _llm_annotate_profiling_plan(self, cmds: List[tuple]) -> List[tuple]:
        """Send tier0 metadata to LLM for annotation via persistent conversation."""
        if self._conv is None:
            return cmds
        try:
            plan = self._tier0
            if plan is None:
                return cmds
            patterns = getattr(plan, "detected_patterns", [])
            import json as _json

            metadata = {
                "programming_model": getattr(plan, "programming_model", "HIP"),
                "kernel_count": getattr(plan, "kernel_count", 0),
                "suggested_counters": getattr(plan, "suggested_counters", []),
                "risk_areas": getattr(plan, "risk_areas", []),
                "detected_patterns": [
                    {
                        "id": (
                            p.get("pattern_id")
                            if isinstance(p, dict)
                            else getattr(p, "pattern_id", "")
                        ),
                        "severity": (
                            p.get("severity")
                            if isinstance(p, dict)
                            else getattr(p, "severity", "")
                        ),
                        "description": (
                            p.get("description")
                            if isinstance(p, dict)
                            else getattr(p, "description", "")
                        ),
                    }
                    for p in patterns
                ],
                "suggested_commands": [cmd for _, cmd in cmds],
            }
            user_msg = (
                f"Annotate this profiling plan (max 200 words, plain text only — no markdown): "
                f"{_json.dumps(metadata)}"
            )
            _print()
            _print("  ── LLM Profiling Advice ────────────────────────────", style=_AMD_RED)
            self._conv.send(user_msg, on_token=_print_token)
            _print()
        except Exception as exc:
            _print(f"  (LLM annotation skipped: {exc})", style="dim")
        return cmds

    # ── SessionContext helpers ────────────────────────────────────────────────

    def _update_ctx_analysis(
        self,
        recs: List[Dict[str, Any]],
        breakdown: Optional[Dict[str, Any]],
    ) -> None:
        """Record a completed analysis run in the session context."""
        self._ctx.iteration += 1
        top_rec = recs[0] if recs else {}
        entry: Dict[str, Any] = {
            "db": self._db_path or "",
            "kernel_pct": (breakdown or {}).get("kernel_time_pct", 0.0),
            "memcpy_pct": (breakdown or {}).get("memcpy_time_pct", 0.0),
            "idle_pct": (breakdown or {}).get("idle_time_pct", 0.0),
            "top_issue": top_rec.get("issue", ""),
            "top_priority": top_rec.get("priority", ""),
        }
        self._ctx.analyses.append(entry)
        if len(self._ctx.analyses) > 5:
            self._ctx.analyses = self._ctx.analyses[-5:]

    def _update_ctx_suggestion(self, suggestion: str) -> None:
        """Record a suggestion shown to the user (capped at 3, truncated at 120 chars)."""
        self._ctx.suggestions_given.append(suggestion[:120])
        if len(self._ctx.suggestions_given) > 3:
            self._ctx.suggestions_given = self._ctx.suggestions_given[-3:]

    def _update_ctx_command(self, cmd: str, exit_code: int) -> None:
        """Record a profiling command that was executed (capped at 5)."""
        self._ctx.commands_run.append({"cmd": cmd, "exit_code": exit_code})
        if len(self._ctx.commands_run) > 5:
            self._ctx.commands_run = self._ctx.commands_run[-5:]

    def _format_context_block(self) -> str:
        """Return a compact text block summarising prior session activity for LLM prompts.

        Returns an empty string when there is no accumulated context yet.
        Kept under ~1500 chars to avoid bloating LLM token budgets.
        """
        ctx = self._ctx
        if not ctx.iteration and not ctx.analyses and not ctx.commands_run:
            return ""

        lines: List[str] = [f"=== Session Context (iteration {ctx.iteration}) ==="]

        if ctx.analyses:
            lines.append("Prior analyses:")
            for a in ctx.analyses[-3:]:  # show at most 3 most-recent
                db_label = f" db={a.get('db', '')}" if a.get("db") else ""
                priority = a.get("top_priority", "")
                prio_label = f" [{priority}]" if priority else ""
                lines.append(
                    f"  run:{db_label} kernel={a.get('kernel_pct', 0):.1f}% "
                    f"idle={a.get('idle_pct', 0):.1f}% "
                    f"top_issue={a.get('top_issue', '')!r}{prio_label}"
                )

        if ctx.suggestions_given:
            lines.append("Suggestions given:")
            for s in ctx.suggestions_given:
                lines.append(f"  - {s[:80]}")

        if ctx.commands_run:
            lines.append("Commands run:")
            for c in ctx.commands_run[-3:]:
                lines.append(f"  $ {c.get('cmd', '')}  [exit {c.get('exit_code', '')}]")

        return "\n".join(lines)

    # ── Analysis helpers ──────────────────────────────────────────────────────

    def _run_tier1_analysis(self, db_path: str):
        """Run Tier 1/2 analysis on db_path; return (recs, breakdown) tuple.

        recs     — List[Dict] of recommendations
        breakdown — Dict with kernel_time_pct, memcpy_time_pct, api_overhead_pct,
                    idle_time_pct, total_runtime_ns; None if analysis fails
        """
        try:
            from rocinsight.ai_analysis.api import analyze_database

            result = analyze_database(pathlib.Path(db_path))
            recs: List[Dict[str, Any]] = []
            for r in (
                result.recommendations.high_priority
                + result.recommendations.medium_priority
                + result.recommendations.low_priority
            ):
                recs.append(
                    {
                        "id": r.id,
                        "priority": r.priority,
                        "category": r.category,
                        "issue": r.title,
                        "suggestion": r.description,
                        "actions": r.next_steps,
                        "commands": [],
                    }
                )
            breakdown: Optional[Dict[str, Any]] = None
            eb = result.execution_breakdown
            if eb is not None:
                breakdown = {
                    "kernel_time_pct": eb.kernel_time_pct,
                    "memcpy_time_pct": eb.memcpy_time_pct,
                    "api_overhead_pct": eb.api_overhead_pct,
                    "idle_time_pct": eb.idle_time_pct,
                    "total_runtime_ns": result.profiling_info.total_duration_ns,
                }
            return recs, breakdown
        except Exception as exc:
            _print(f"  (Tier 1 analysis failed: {exc})", style="red")
            return [], None

    def _extract_ai_commands(self, text: str, structured_cmds: List[str]) -> List[str]:
        """Extract rocprofv3 commands from LLM text + structured recommendation list.

        Free-form matches come first; deduplicates; returns at most 5.
        """
        free_form = re.findall(r"rocprofv3\s+[^\n]+", text)
        # Strip trailing punctuation / markdown from free-form matches
        free_form = [c.rstrip("`.,'\"") for c in free_form]
        seen: set = set()
        result: List[str] = []
        for cmd in free_form + list(structured_cmds):
            cmd = cmd.strip()
            if cmd and cmd not in seen:
                seen.add(cmd)
                result.append(cmd)
            if len(result) >= 5:
                break
        return result

    def _offer_run_ai_commands(self, commands: List[str]) -> None:
        """Prompt the user to run an AI-suggested profiling command; run + re-analyze if chosen."""
        if not commands:
            return
        _print()
        _print(
            "  ── AI-suggested profiling commands ───────────────────────", style=_AMD_RED
        )
        for i, cmd in enumerate(commands, 1):
            _print_m(f"  [bold {_AMD_RED}][{i}][/bold {_AMD_RED}]  [dim]$ {cmd}[/dim]")
        _print()
        prompt_opts = "/".join(str(i) for i in range(1, len(commands) + 1)) + "/n"
        try:
            choice = _input(f"  Run one of these now? [{prompt_opts}]:  ").strip()
        except EOFError:
            return
        if not choice.isdigit() or not (1 <= int(choice) <= len(commands)):
            return

        cmd = commands[int(choice) - 1]
        if "-- ./app" in cmd:
            auto = self._resolve_app_placeholder(cmd)
            _print("  Enter application to profile:", style=_AMD_RED)
            try:
                app_input = _input("  > ").strip()
            except EOFError:
                return
            if app_input:
                cmd = cmd.replace("-- ./app", f"-- {app_input}")
            elif "-- ./app" not in auto:
                cmd = auto

        if _has_shell_meta(cmd):
            _print("  Command contains shell metacharacters — rejected for safety.", style="red")
            return

        _print()
        _print(f"  Running: $ {cmd}", style=_AMD_RED)
        _print()
        proc = subprocess.run(shlex.split(cmd))
        _print()
        if proc.returncode != 0:
            _print(f"  Command exited with code {proc.returncode}.", style="yellow")

        db_path = self._find_output_db(cmd)
        if not db_path:
            _print(
                "  Enter path to the output .db file (or Enter to skip):", style=_AMD_RED
            )
            try:
                db_input = _input("  > ").strip()
            except EOFError:
                return
            if not db_input:
                return
            db_path = pathlib.Path(db_input).expanduser()
            if not db_path.exists():
                _print(f"  File not found: {db_path}", style="red")
                return

        self._db_path = str(db_path)
        _print("  Running Tier 1/2 analysis on new trace...", style="dim")
        new_recs, breakdown = self._run_tier1_analysis(str(db_path))
        if new_recs:
            self._show_analysis_summary(new_recs)
        added = self._ingest_recommendations(new_recs)
        now = datetime.now(timezone.utc).isoformat()
        self._session.history.append(
            HistoryEntry(type="profiling_run", timestamp=now, db_path=str(db_path))
        )
        self._session.last_updated = now
        if self._conv:
            self._session.conversation = self._conv.to_dict()
        self._session.sent_source_files = list(self._sent_source_files)
        self._store.save(self._session)
        _print(f"  ✓ {added} finding(s) added to menu.", style="green")

    _TOKEN_BUDGET = 60_000  # characters (approximate token proxy)

    # Subdirectory names that look like backup/archive copies — skip them so
    # we don't send the same source file twice (e.g. original_code/).
    _SKIP_SUBDIRS = frozenset(
        {
            "original_code",
            "original",
            "backup",
            "bak",
            "old",
            "archive",
            "reference",
            "orig",
            "before",
        }
    )

    def _select_hot_files(self, budget: int = _TOKEN_BUDGET) -> List[tuple]:
        """Return [(abs_path, content)] for files with detected kernels, within budget.

        Deduplicates by basename so that backup copies in subdirectories (e.g.
        original_code/foo.cpp when foo.cpp already exists at the root) are skipped.
        """
        if not self._tier0:
            return []
        # Support both SourceAnalysisResult (detected_kernels directly on tier0)
        # and any future wrapper that exposes a .profiling_plan child object.
        plan = getattr(self._tier0, "profiling_plan", None) or self._tier0

        rel_paths: List[str] = []
        seen_paths: set = set()
        seen_names: set = set()  # deduplicate by basename
        base = pathlib.Path(self._source_dir)
        for k in getattr(plan, "detected_kernels", []):
            # kernels may be dicts {"file": ...} or dataclass objects with .file
            rp = k.get("file", "") if isinstance(k, dict) else getattr(k, "file", "")
            if not rp or rp in seen_paths:
                continue
            # Skip files that live inside known backup subdirectories
            parts = pathlib.Path(rp).parts
            if any(p in self._SKIP_SUBDIRS for p in parts[:-1]):
                continue
            name = pathlib.Path(rp).name
            if name in seen_names:
                continue  # skip duplicate basenames (same file in different subdir)
            seen_paths.add(rp)
            seen_names.add(name)
            rel_paths.append(rp)

        result: List[tuple] = []
        used = 0
        for rp in rel_paths:
            full = base / rp
            if not full.exists():
                continue
            content = full.read_text(errors="replace")
            if used + len(content) > budget:
                content = content[: budget - used]
                result.append((str(full), content))
                break
            result.append((str(full), content))
            used += len(content)

        return result

    def _path_optimize(self) -> None:
        """Get AI optimization suggestions for detected GPU source patterns."""
        # Determine LLM provider
        llm_provider = self._llm_provider
        if not llm_provider:
            llm_provider = self._autodetect_llm()

        if not llm_provider:
            _print(
                "  No LLM configured. Add --llm anthropic or --llm openai to get "
                "AI-generated code suggestions. Showing rule-based hints instead:",
                style="yellow",
            )
            _print()
            self._show_rulebased_suggestions()
            return

        # Fast path: use compact tier0 metadata when available (same speed as [p])
        if self._tier0:
            self._optimize_via_tier0(llm_provider)
            return

        # Fallback: send raw source files (slower — only used when --source-dir
        # was not given, so tier0 was never run)
        hot_files = self._select_hot_files()
        if not hot_files:
            _print(
                "  No kernel-containing files detected. "
                "Run with --source-dir pointing at your source.",
                style="yellow",
            )
            return

        _print()
        _print(f"  Analyzing {len(hot_files)} file(s):", style=_AMD_RED)
        for path, _ in hot_files:
            try:
                label = pathlib.Path(path).relative_to(pathlib.Path(self._source_dir))
            except ValueError:
                label = pathlib.Path(path).name
            _print(f"    · {label}", style="dim")
        _print()

        summaries = [(pathlib.Path(p).name, c) for p, c in hot_files]
        raw = self._request_optimization_suggestions(summaries, llm_provider)
        if not raw:
            return
        # Display first file's suggestion directly
        first_text = next(iter(raw.values()), "")
        if first_text:
            _print()
            _print(
                "  ── Optimization Suggestions ─────────────────────────", style=_AMD_RED
            )
            _print(first_text[:3000] + ("…" if len(first_text) > 3000 else ""))
            _print()

        # Offer to run any profiling commands found in the LLM response
        all_text = "\n".join(raw.values())
        structured = [
            c.get("full_command", "")
            for rec in self._recs
            for c in rec.get("commands", [])
            if c.get("full_command")
        ]
        ai_cmds = self._extract_ai_commands(all_text, structured)
        self._offer_run_ai_commands(ai_cmds)

        # Apply changes file by file (legacy path)
        modified: List[str] = []
        for path, original_content in hot_files:
            name = pathlib.Path(path).name
            file_sugg = raw.get(name)
            if not file_sugg:
                continue
            modified_content = self._present_and_apply(path, original_content, file_sugg)
            if modified_content is not None:
                p = pathlib.Path(path)
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=p.parent, delete=False, suffix=".tmp"
                ) as tmp:
                    tmp.write(modified_content)
                os.replace(tmp.name, str(p))
                modified.append(name)

        if modified:
            now = datetime.now(timezone.utc).isoformat()
            self._session.history.append(
                HistoryEntry(
                    type="code_change",
                    timestamp=now,
                    files_modified=modified,
                    summary=f"Optimized {len(modified)} file(s) via LLM suggestions",
                )
            )
            _print(f"  ✓ Modified: {', '.join(modified)}", style="green")
            _print()
            try:
                ans = _input("  Run profiling commands now? [y/N]  ").strip().lower()
            except EOFError:
                ans = ""
            if ans == "y":
                self._path_profiling(_source="code_change_analysis")

    def _optimize_via_tier0(self, llm_provider: str) -> None:
        """Fast optimization path: send compact tier0 metadata to LLM (not raw source)."""
        _print()
        _print(
            "  Requesting optimization suggestions (based on detected patterns)...",
            style="dim",
        )
        import json as _json

        # Build compact metadata from tier0 — same approach as annotate_profiling_plan
        plan = self._tier0
        patterns = getattr(plan, "detected_patterns", [])
        kernels = getattr(plan, "detected_kernels", [])[:5]
        metadata = {
            "programming_model": getattr(plan, "programming_model", "HIP"),
            "kernel_count": getattr(plan, "kernel_count", 0),
            "risk_areas": getattr(plan, "risk_areas", []),
            "detected_patterns": [
                {
                    "id": (
                        p.get("pattern_id")
                        if isinstance(p, dict)
                        else getattr(p, "pattern_id", "")
                    ),
                    "severity": (
                        p.get("severity")
                        if isinstance(p, dict)
                        else getattr(p, "severity", "")
                    ),
                    "description": (
                        p.get("description")
                        if isinstance(p, dict)
                        else getattr(p, "description", "")
                    ),
                    "count": (
                        p.get("count", 1)
                        if isinstance(p, dict)
                        else getattr(p, "count", 1)
                    ),
                }
                for p in patterns
            ],
            "detected_kernels": [
                {
                    "name": ("[KERNEL]" if isinstance(k, dict) else "[KERNEL]"),
                    "launch_type": (
                        k.get("launch_type", "")
                        if isinstance(k, dict)
                        else getattr(k, "launch_type", "")
                    ),
                }
                for k in kernels
            ],
        }

        if self._conv is None:
            _print("  (No LLM configured — skipping AI optimization)", style="dim")
            return

        user_msg = (
            "Based on these detected GPU source patterns, provide concrete "
            "optimization recommendations (max 300 words, plain text only — no markdown headers):\n"
            + _json.dumps(metadata, indent=2)
        )
        _print()
        _print("  ── AI Optimization Suggestions ──────────────────────", style=_AMD_RED)
        try:
            note = self._conv.send(user_msg, on_token=_print_token)
            _print()
        except Exception as exc:
            _print(f"\n  (LLM optimization failed: {exc})", style="red")
            return
        if note:
            self._offer_apply_suggestions(note, self._llm_provider)
            structured = [
                c.get("full_command", "")
                for rec in self._recs
                for c in rec.get("commands", [])
                if c.get("full_command")
            ]
            ai_cmds = self._extract_ai_commands(note, structured)
            self._offer_run_ai_commands(ai_cmds)
        else:
            _print("  (LLM returned no suggestions)", style="yellow")

    def _offer_apply_suggestions(
        self, suggestions: str, llm_provider: Optional[str] = None
    ) -> None:
        """Ask user whether to save the suggestions or let the LLM edit source code directly."""
        _print("  Apply these suggestions to your source files?", style=_AMD_RED)
        _menu_opt("s", "Save suggestions to a file")
        _menu_opt("e", "Edit code with AI  (LLM rewrites a source file)")
        _menu_opt("n", "No, return to menu (default)")
        try:
            ans = _input("  > ").strip().lower()
        except EOFError:
            return

        if ans == "s":
            out_path = pathlib.Path(self._source_dir) / "ai_optimization_suggestions.txt"
            try:
                out_path.write_text(suggestions + "\n")
                _print(f"  Suggestions saved to: {out_path}", style="green")
            except OSError as e:
                _print(f"  (Could not save file: {e})", style="red")

        elif ans == "e":
            # Always use local LLM for code edits — preserves privacy and avoids
            # cloud token limits that truncate large source files.
            self._apply_suggestions_via_llm(suggestions, "local")

    def _pick_source_file(self) -> Optional[pathlib.Path]:
        """Present a numbered list of source files and return the chosen one."""
        exts = {".hip", ".cpp", ".cu", ".cl", ".h", ".hpp", ".py"}
        src_files: List[pathlib.Path] = []
        try:
            src_files = [
                p
                for p in sorted(pathlib.Path(self._source_dir).rglob("*"))
                if p.suffix in exts and p.is_file()
            ]
        except OSError:
            pass

        if not src_files:
            _print("  (No source files found in source directory)", style="yellow")
            return None

        _print()
        _print("  Choose a file to edit:", style=_AMD_RED)
        for i, p in enumerate(src_files[:15]):
            try:
                label = p.relative_to(pathlib.Path(self._source_dir))
            except ValueError:
                label = p.name
            _menu_opt(str(i + 1), str(label))
        try:
            choice = _input("  > ").strip()
        except EOFError:
            return None
        try:
            idx = int(choice) - 1
            if not (0 <= idx < len(src_files)):
                raise ValueError
        except ValueError:
            _print("  (Invalid choice — skipping)", style="yellow")
            return None
        return src_files[idx]

    def _apply_suggestions_via_llm(
        self, suggestions: str, llm_provider: Optional[str]
    ) -> None:
        """Use the LLM to rewrite a source file applying the optimization suggestions.

        Workflow:
          1. User picks a source file.
          2. LLM receives: original file + suggestions → returns complete rewritten file.
          3. Unified diff is shown.
          4. User confirms before the file is overwritten.
          5. Original is backed up as <file>.orig.
        """
        # For local provider: ensure a local LLM backend is actually running.
        # If not, fall back to the configured online provider (anthropic/openai),
        # or surface a helpful error if nothing is available.
        if llm_provider == "local" and not self._llm_local:
            detected = self._autodetect_llm()
            if not detected:
                fallback = (
                    self._llm_provider
                    if self._llm_provider and self._llm_provider != "local"
                    else None
                )
                if fallback:
                    _print(
                        f"  Local LLM not detected — falling back to {fallback} for code edit.",
                        style="yellow",
                    )
                    llm_provider = fallback
                else:
                    _print("  No LLM available for code editing.", style="yellow")
                    _print("  Options:", style="dim")
                    _print("    • Start a local model:  ollama run llama3", style="dim")
                    _print(
                        "    • Or pass --llm anthropic / --llm openai when launching rocinsight.",
                        style="dim",
                    )
                    return

        if not llm_provider:
            _print(
                "  No LLM configured. Pass --llm local/anthropic/openai to enable AI code edits.",
                style="yellow",
            )
            return

        chosen = self._pick_source_file()
        if chosen is None:
            return

        try:
            original = chosen.read_text()
        except OSError as e:
            _print(f"  (Cannot read {chosen.name}: {e})", style="red")
            return

        # Build LLM prompt
        system = (
            "You are an expert AMD GPU performance engineer and C++/HIP developer. "
            "You will be given a source file and a list of optimization suggestions. "
            "Rewrite the file applying the suggestions. "
            "Return ONLY the complete rewritten source file — no explanation, no markdown fences, "
            "no commentary before or after the code. "
            "Preserve all existing functionality. Make the minimum changes needed to apply the "
            "optimizations. Add a short inline comment on each changed line explaining why."
        )
        user = (
            f"=== OPTIMIZATION SUGGESTIONS ===\n{suggestions}\n\n"
            f"=== SOURCE FILE: {chosen.name} ===\n{original}"
        )

        _print()
        from rocinsight.ai_analysis.llm_analyzer import LLMAnalyzer

        model = self._llm_local_model if llm_provider == "local" else self._llm_model
        analyzer = LLMAnalyzer(
            provider=llm_provider,
            api_key=self._llm_api_key,
            model=model,
        )

        try:
            with _Spinner(f"  {llm_provider} LLM is rewriting {chosen.name}..."):
                if llm_provider == "openai":
                    rewritten = analyzer._call_openai(system, user, max_tokens=16384)
                elif llm_provider == "anthropic":
                    rewritten = analyzer._call_anthropic(system, user)
                elif llm_provider == "private":
                    rewritten = analyzer._call_private(system, user)
                elif llm_provider == "claude-code":
                    rewritten = analyzer._call_claude_code(system, user)
                else:
                    rewritten = analyzer._call_local(system, user)
        except Exception as exc:
            _print(f"  (LLM edit failed: {exc})", style="red")
            return

        if not rewritten or not rewritten.strip():
            _print("  (LLM returned an empty file — aborting)", style="yellow")
            return

        # Show unified diff
        import difflib

        diff_lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                rewritten.splitlines(keepends=True),
                fromfile=f"{chosen.name} (original)",
                tofile=f"{chosen.name} (AI-edited)",
                n=3,
            )
        )

        _print()
        _print_m(f"  [bold {_AMD_RED}]── Proposed changes ─────────────────────────────────[/bold {_AMD_RED}]")
        if diff_lines:
            for line in diff_lines[:120]:  # cap at 120 diff lines for readability
                _print_diff_line(line.rstrip("\n"))
            if len(diff_lines) > 120:
                _print(
                    f"  ... ({len(diff_lines) - 120} more diff lines omitted)",
                    style="dim",
                )
        else:
            _print(
                "  (No changes — rewritten file is identical to original)",
                style="yellow",
            )
            return

        _print()
        try:
            confirm = _input("  Apply these changes? [y/N]  ").strip().lower()
        except EOFError:
            return

        if confirm != "y":
            _print("  Changes discarded — original file unchanged.", style="dim")
            return

        # Back up original then write
        backup = chosen.with_suffix(chosen.suffix + ".orig")
        try:
            backup.write_text(original)
            chosen.write_text(rewritten)
            _print(f"  Original backed up to: {backup.name}", style="dim")
            _print(f"  File updated: {chosen}", style="green")
        except OSError as e:
            _print(f"  (Write failed: {e})", style="red")

        # Notify the persistent conversation about the rewrite
        if self._conv:
            try:
                self._conv.send(
                    f"File `{chosen.name}` was rewritten applying the above optimizations. "
                    f"Compilation: pending.",
                    on_token=None,
                )
            except Exception:
                pass  # post-rewrite summary is advisory; never crash here

    def _autodetect_llm(self) -> Optional[str]:
        """Try to detect a running local LLM (ollama). Returns provider name or None."""
        try:
            import urllib.request

            url = os.environ.get("ROCINSIGHT_LLM_LOCAL_URL", "http://localhost:11434")
            req = urllib.request.urlopen(f"{url}/api/tags", timeout=1)
            if req.status == 200:
                _print(f"  Auto-detected ollama at {url} — using local LLM.", style="dim")
                self._llm_local = "ollama"
                return "local"
        except Exception:
            pass
        return None

    def _show_rulebased_suggestions(self) -> None:
        """Display Tier 0 rule-based optimization hints when no LLM is available."""
        recs = getattr(self._tier0, "recommendations", None) or self._recs
        if not recs:
            _print("  No rule-based suggestions available.", style="dim")
            return
        shown = 0
        for rec in recs:
            pri = (
                rec.get("priority", "INFO")
                if isinstance(rec, dict)
                else getattr(rec, "priority", "INFO")
            )
            if pri in ("HIGH", "MEDIUM"):
                issue = (
                    rec.get("issue", rec.get("category", ""))
                    if isinstance(rec, dict)
                    else getattr(rec, "issue", "")
                )
                suggest = (
                    rec.get("suggestion", "")
                    if isinstance(rec, dict)
                    else getattr(rec, "suggestion", "")
                )
                actions = (
                    rec.get("actions", [])
                    if isinstance(rec, dict)
                    else getattr(rec, "actions", [])
                )
                _print_m(f"  {_priority_badge(pri)}  {issue}")
                if suggest:
                    _print(f"    → {suggest}", style="dim")
                for act in actions[:3]:
                    _print(f"      • {act}", style="dim")
                _print()
                shown += 1
        if shown == 0:
            _print("  No HIGH/MEDIUM priority suggestions found.", style="dim")
        _print(
            "  To apply AI-generated code patches: re-run with --llm anthropic or --llm openai.",
            style="dim",
        )

    def _request_optimization_suggestions(
        self, summaries: List[tuple], llm_provider: Optional[str] = None
    ) -> Dict[str, str]:
        """Send source file summaries to LLM; return {filename: suggestion_text}."""
        if self._conv is None:
            return {}
        try:
            current_files = {name for name, _ in summaries}
            already_sent = current_files.issubset(self._sent_source_files)
            if already_sent:
                # Source content already in conversation history — ask for new suggestions only
                file_list = ", ".join(sorted(current_files))
                user_msg = (
                    f"Based on the source files already shared ({file_list}), provide "
                    f"additional concrete optimization suggestions we haven't covered yet. "
                    f"Plain text only — no markdown headers. "
                    f"Start each file section with exactly: FILE: <filename>"
                )
            else:
                new_files = current_files - self._sent_source_files
                combined = "\n\n".join(
                    f"=== {name} ===\n{content}"
                    for name, content in summaries
                    if name in new_files
                )
                user_msg = (
                    f"Analyze these AMD GPU source files and provide concrete, actionable "
                    f"optimization suggestions. Focus on: memory coalescing, wave occupancy, "
                    f"unnecessary hipDeviceSynchronize, blocking hipMemcpy, MFMA usage, LDS "
                    f"utilization, loop structure, kernel launch parameters. Be specific — "
                    f"reference actual patterns visible in the code. Use plain text only — "
                    f"no markdown headers. Start each file section with exactly: FILE: <filename>\n\n"
                    f"{combined}"
                )
                self._sent_source_files.update(new_files)
            _print()
            _print(
                "  ── AI Optimization Suggestions ──────────────────────", style=_AMD_RED
            )
            raw = self._conv.send(user_msg, on_token=_print_token)
            _print()

            result: Dict[str, str] = {}
            if raw and raw.lstrip().startswith("FILE:"):
                raw = "\n" + raw.lstrip()
            for block in re.split(r"\nFILE:\s*", raw or ""):
                block = block.strip()
                if not block:
                    continue
                lines = block.split("\n", 1)
                if len(lines) == 2:
                    result[lines[0].strip()] = lines[1].strip()
            if not result and raw and raw.strip():
                first_name = summaries[0][0] if summaries else "response"
                result[first_name] = raw.strip()
            return result
        except Exception as exc:
            _print(f"  (LLM optimization failed: {exc})", style="red")
            return {}

    def _present_and_apply(
        self, path: str, original: str, suggestion: str
    ) -> Optional[str]:
        """Show suggestion, optionally show diff, ask for confirmation. Return new content or None."""
        name = pathlib.Path(path).name
        _print()
        _print(
            f"  ── Suggestions for {name} ──────────────────────────────", style=_AMD_RED
        )
        _print(suggestion[:2000] + ("…" if len(suggestion) > 2000 else ""))
        _print()
        try:
            ans = (
                _input(f"  Append LLM suggestions as comments to {name}? [y/N/diff]  ")
                .strip()
                .lower()
            )
        except EOFError:
            return None
        if ans == "diff":
            _print(
                "  (Diff view: LLM suggestions are advisory — showing suggestion text)",
                style="dim",
            )
            _print(suggestion, style="dim")
            try:
                ans = (
                    _input(f"  Append LLM suggestions as comments to {name}? [y/N]  ")
                    .strip()
                    .lower()
                )
            except EOFError:
                return None
        if ans == "y":
            separator = "\n" + "=" * 72 + "\n"
            return (
                original
                + separator
                + "// LLM OPTIMIZATION SUGGESTIONS:\n// "
                + "\n// ".join(suggestion.splitlines())
                + "\n"
            )
        return None

    def _pursue_recommendation(self, item: PersistentMenuItem) -> None:
        """Show full recommendation and sub-menu: [r] run command, [m] back to main menu."""
        _print()
        _print(
            f"  ── {item.title} [{item.priority}] ──────────────────────────────",
            style=_AMD_RED,
        )
        detail = item.detail
        if detail.get("issue"):
            _print(f"  Issue:  {detail['issue']}")
        if detail.get("suggestion"):
            _print(f"  Why:    {detail['suggestion']}")
        if detail.get("estimated_impact"):
            _print(f"  Impact: {detail['estimated_impact']}")
        actions = detail.get("actions", [])
        if actions:
            _print()
            _print("  Next steps:", style=_AMD_RED)
            for act in actions:
                _print(f"    • {act}", style="dim")

        cmds = [
            c.get("full_command", "")
            for c in detail.get("commands", [])
            if c.get("full_command")
        ]
        if cmds:
            _print()
            _print("  Suggested commands:", style=_AMD_RED)
            for i, cmd in enumerate(cmds, 1):
                _print(f"    [{i}]  $ {cmd}", style="dim")

        _print()
        if cmds:
            _menu_opt("r", "Run suggested command")
        else:
            _menu_opt("r", "Run a profiling command")
        _menu_opt("m", "Back to main menu")
        _print()
        try:
            choice = _input("  > ").strip().lower()
        except EOFError:
            return

        if choice == "r" and not cmds:
            # No specific commands → fall back to the full profiling path
            self._path_profiling()
        elif choice == "r" and cmds:
            cmd = self._resolve_app_placeholder(cmds[0])
            # Ask for app if placeholder not resolved
            if "-- ./app" in cmd:
                _print("  Enter application to profile:", style=_AMD_RED)
                try:
                    app_input = _input("  > ").strip()
                except EOFError:
                    return
                if app_input:
                    cmd = cmd.replace("-- ./app", f"-- {app_input}")

            _print()
            _print(f"  Running: $ {cmd}", style=_AMD_RED)
            _print()
            proc = subprocess.run(shlex.split(cmd), check=False)
            _print()
            if proc.returncode != 0:
                _print(f"  Command exited with code {proc.returncode}.", style="yellow")

            # Auto-detect output .db
            db_path = self._find_output_db(cmd)
            if db_path:
                _print(f"  Found output: {db_path}", style="green")
            else:
                try:
                    db_input = _input(
                        "  Enter path to .db file from this run (or Enter to skip): "
                    ).strip()
                except EOFError:
                    db_input = ""
                if db_input:
                    db_path = pathlib.Path(db_input).expanduser()
                    if not db_path.exists():
                        _print(f"  File not found: {db_path}", style="red")
                        db_path = None

            if db_path:
                _print("  Running Tier 1/2 analysis...", style="dim")
                new_recs, breakdown = self._run_tier1_analysis(str(db_path))
                added = self._ingest_recommendations(new_recs)
                now = datetime.now(timezone.utc).isoformat()
                self._session.history.append(
                    HistoryEntry(
                        type="profiling_run", timestamp=now, db_path=str(db_path)
                    )
                )
                _print(f"  ✓ {added} new recommendation(s) added.", style="green")
        # [m] or any other input → return to main menu (item stays in list)


# ── WorkflowSession (7-phase interactive workflow) ───────────────────────────


@dataclass
class _TraceRun:
    """Record of a single profiling run."""

    timestamp: str
    command: str
    db_path: str
    trace_files: List[str] = field(default_factory=list)


@dataclass
class _AnalysisSnapshot:
    """Snapshot of one analysis iteration."""

    timestamp: str
    iteration: int
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    execution_breakdown: Optional[Dict[str, Any]] = None
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    ai_recommended_command: Optional[str] = None
    plateau_detected: bool = False


@dataclass
class _EditRecord:
    """Record of an AI-applied edit."""

    timestamp: str
    file_path: str
    backup_path: str
    checkpoint_id: Optional[int] = None  # cp_id of matching CheckpointRecord


class CheckpointError(Exception):
    """Raised when a git checkpoint operation fails."""


@dataclass
class CheckpointRecord:
    """One git-worktree checkpoint created after an AI edit batch."""

    cp_id: int  # 0-based index
    commit_hash: str  # git commit hash
    ref_name: str  # refs/rocinsight/<session_id>/cp-N
    worktree_path: str  # ~/.rocinsight/sessions/<id>/cp-N
    timestamp: str  # ISO-8601
    files_modified: List[str]  # repo-relative paths touched
    edit_summary: str  # human-readable description of AI changes
    file_snapshots: Dict[str, str]  # {repo_relative_path: full_file_contents}
    run_index: Optional[int] = None  # index into trace_history; None = no run yet
    performance_delta_pct: Optional[float] = None  # +10 = improvement, -67 = regression
    blacklisted: bool = False
    blacklist_category: str = ""  # taken from edit_summary
    blacklist_description: str = ""  # injected into LLM prompts


class GitCheckpointManager:
    """Wraps all git subprocess calls for checkpoint management.

    All git calls pass identity overrides (-c user.email/name) so the feature
    works in HPC environments where git identity is not configured.
    Raises CheckpointError on any git failure.
    """

    # Base args prepended to every git call
    _ID = ["-c", "user.email=rocinsight@local", "-c", "user.name=rocinsight"]

    def __init__(
        self,
        repo_root: str,
        session_id: str,
        sessions_dir: str,
    ) -> None:
        self._repo_root = repo_root
        self._session_id = session_id
        self._sessions_dir = sessions_dir

    def _git(self, *args: str, capture: bool = True) -> "subprocess.CompletedProcess":
        """Run a git command rooted at repo_root with identity overrides."""
        cmd = ["git", "-C", self._repo_root] + list(self._ID) + list(args)
        return subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
        )

    def detect_repo(self, source_path: str) -> str:
        """Return git repo root for source_path, or raise CheckpointError."""
        result = subprocess.run(
            ["git", "-C", source_path, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise CheckpointError(
                f"'{source_path}' is not inside a git repository. "
                "Checkpoints require git."
            )
        return result.stdout.strip()

    def get_head(self) -> str:
        """Return current HEAD commit hash."""
        result = self._git("rev-parse", "HEAD")
        if result.returncode != 0:
            raise CheckpointError("Could not read HEAD commit hash.")
        return result.stdout.strip()

    def create_checkpoint_commit(self, files: List[str], message: str) -> str:
        """Stage files and create a commit; return commit hash.

        Uses --no-verify to skip pre-commit hooks (rocinsight checkpoints are
        tooling artifacts, not user commits).
        """
        # Stage only the specified files
        add_result = self._git("add", "--", *files)
        if add_result.returncode != 0:
            raise CheckpointError(f"git add failed: {add_result.stderr.strip()}")
        commit_result = self._git("commit", "--no-verify", "-m", message)
        if commit_result.returncode != 0:
            raise CheckpointError(f"git commit failed: {commit_result.stderr.strip()}")
        # Get hash of the commit we just created
        hash_result = self._git("rev-parse", "HEAD")
        if hash_result.returncode != 0:
            raise CheckpointError("Could not read new commit hash.")
        return hash_result.stdout.strip()

    def tag_checkpoint(self, cp_id: int, commit_hash: str) -> str:
        """Create a named ref (not a branch) pinning commit_hash from GC.

        Returns the ref name: refs/rocinsight/<session_id>/cp-N
        """
        ref_name = f"refs/rocinsight/{self._session_id}/cp-{cp_id}"
        result = self._git("update-ref", ref_name, commit_hash)
        if result.returncode != 0:
            raise CheckpointError(f"git update-ref failed: {result.stderr.strip()}")
        return ref_name

    def add_worktree(self, cp_id: int, commit_hash: str) -> str:
        """Create a detached-HEAD worktree at sessions_dir/<session_id>/cp-N.

        If the target path already exists (stale from a crashed session),
        it is removed before creating the worktree.
        Returns the worktree path.
        """
        import shutil as _shutil

        worktree_path = str(
            pathlib.Path(self._sessions_dir) / self._session_id / f"cp-{cp_id}"
        )
        if os.path.exists(worktree_path):
            _shutil.rmtree(worktree_path, ignore_errors=True)

        result = self._git("worktree", "add", "--detach", worktree_path, commit_hash)
        if result.returncode != 0:
            raise CheckpointError(f"git worktree add failed: {result.stderr.strip()}")
        return worktree_path

    def restore_files_from_commit(self, commit_hash: str, files: List[str]) -> None:
        """Restore files to their state at commit_hash in the working directory."""
        ls_result = self._git("ls-tree", "-r", "--name-only", commit_hash)
        files_in_commit = set(ls_result.stdout.splitlines())

        for file in files:
            if file in files_in_commit:
                result = self._git("checkout", commit_hash, "--", file)
                if result.returncode != 0:
                    raise CheckpointError(
                        f"git checkout {commit_hash} -- {file} failed: "
                        f"{result.stderr.strip()}"
                    )

    def remove_worktree(self, worktree_path: str) -> None:
        """Remove a worktree directory. Silently skips if path does not exist."""
        if not worktree_path:
            return
        if not os.path.exists(worktree_path):
            return
        self._git("worktree", "remove", worktree_path, "--force")
        # Non-fatal — log but don't raise (exit path must not crash)

    def delete_ref(self, ref_name: str) -> None:
        """Delete a named ref. Silently skips if ref was already gone."""
        self._git("update-ref", "-d", ref_name)

    def commit_reachable(self, commit_hash: str) -> bool:
        """Return True if commit_hash exists as a git object."""
        result = self._git("cat-file", "-e", commit_hash)
        return result.returncode == 0

    def files_in_commit(self, commit_hash: str) -> List[str]:
        """Return repo-relative file paths present in commit_hash tree."""
        result = self._git("ls-tree", "-r", "--name-only", commit_hash)
        return result.stdout.splitlines()

    def list_worktrees(self) -> List[str]:
        """Return list of worktree paths registered in the repo."""
        result = self._git("worktree", "list", "--porcelain")
        paths = []
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                paths.append(line[len("worktree ") :])
        return paths


@dataclass
class WorkflowState:
    """Persistent state for the 7-phase interactive workflow session."""

    app_command: str
    source_paths: List[str] = field(default_factory=list)
    profiling_command: str = ""
    trace_history: List[_TraceRun] = field(default_factory=list)
    analysis_history: List[_AnalysisSnapshot] = field(default_factory=list)
    edit_history: List[_EditRecord] = field(default_factory=list)
    iteration_count: int = 0
    # Checkpoint fields
    repo_root: str = ""
    baseline_commit: str = ""
    checkpoints: List[CheckpointRecord] = field(default_factory=list)
    active_checkpoint: Optional[int] = None
    blacklisted_approaches: List[str] = field(default_factory=list)
    plateau_iteration_count: int = 0
    seen_recommendation_hashes: List[str] = field(default_factory=list)
    conversation: Optional[Dict[str, Any]] = None


def _edit_summary_from_suggestions(suggestions: str) -> str:
    """Extract a short summary from the LLM suggestions string."""
    for line in suggestions.splitlines():
        line = line.strip(" #-*")
        if line:
            return line[:80]
    return "AI code edit"


class WorkflowSession:
    """7-phase interactive profiling + optimization workflow.

    Triggered by: rocinsight analyze --interactive "<app_command>"
    """

    _DEFAULT_TRACE_DIR = "/tmp/rocinsight_trace"

    def __init__(
        self,
        app_command: str,
        source_paths: Optional[List[str]] = None,
        llm_provider: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        trace_dir: Optional[str] = None,
        resume_session: Optional[str] = None,
    ) -> None:
        self._llm_provider = llm_provider
        self._llm_api_key = llm_api_key
        self._llm_model = llm_model
        self._trace_dir = trace_dir or self._DEFAULT_TRACE_DIR
        self._sessions_dir = pathlib.Path.home() / ".rocinsight" / "sessions"
        # Checkpoint manager — set after _init_checkpoints() called from run()
        self._gcm: Optional["GitCheckpointManager"] = None
        self._conv: Optional["LLMConversation"] = None
        self._resumed = False

        if resume_session:
            loaded = self._load_session(resume_session)
            if loaded is not None:
                self._state, self._session_id = loaded
                self._session_file = self._sessions_dir / f"{self._session_id}.json"
                # Merge any new source paths provided on the command line
                for sp in source_paths or []:
                    if sp not in self._state.source_paths:
                        self._state.source_paths.append(sp)
                self._resumed = True
                return

        # Fresh session
        self._state = WorkflowState(
            app_command=app_command,
            source_paths=list(source_paths or []),
        )
        _ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        try:
            _slug = re.sub(r"[^\w-]", "_", shlex.split(app_command)[0])[:24]
        except (ValueError, IndexError):
            _slug = "app"
        self._session_id = f"workflow_{_ts}_{_slug}"
        self._session_file = self._sessions_dir / f"{self._session_id}.json"

    def _ensure_conv(self) -> Optional["LLMConversation"]:
        """Lazily create or restore the persistent LLMConversation."""
        if not self._llm_provider:
            return None
        if self._conv is not None:
            return self._conv
        try:
            if self._state.conversation:
                # Restore from saved session
                self._conv = LLMConversation.from_dict(
                    self._state.conversation,
                    api_key=self._llm_api_key,
                    model=self._llm_model,
                )
            else:
                # Create fresh conversation
                fence = ""
                try:
                    fence = load_reference_guide()
                except Exception:
                    pass
                system_prompt = (
                    "You are an expert AMD GPU performance engineer in an iterative "
                    "optimization workflow. You will see profiling results, source code, "
                    "and edit history across multiple messages. Remember what was already "
                    "tried — do NOT repeat suggestions that failed or showed no impact.\n\n"
                    + fence
                )
                self._conv = LLMConversation(
                    provider=self._llm_provider,
                    api_key=self._llm_api_key,
                    model=self._llm_model,
                    compact_every=10,
                    keep_recent_turns=6,
                )
                self._conv.initialize(system_prompt)
            return self._conv
        except Exception as exc:
            _print(f"  (Conversation init failed: {exc})", style="dim")
            return None

    def _load_session(
        self, path_or_id: str
    ) -> Optional[tuple]:  # -> (WorkflowState, session_id) or None
        """Load a saved WorkflowSession from disk.

        Accepts:
          - An absolute path to a .json file
          - A session ID (looked up in ~/.rocinsight/sessions/<id>.json)
          - "latest" — picks the most-recently-modified workflow session file
        """
        p = pathlib.Path(path_or_id)
        if not p.is_absolute():
            # Try as session ID under sessions dir
            candidate = self._sessions_dir / f"{path_or_id}.json"
            if candidate.exists():
                p = candidate
            elif path_or_id == "latest":
                wf_files = sorted(
                    self._sessions_dir.glob("workflow_*.json"),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )
                if not wf_files:
                    _print("  No saved workflow sessions found.", style="yellow")
                    return None
                p = wf_files[0]
            else:
                _print(
                    f"  Session not found: {path_or_id!r}", style="yellow"
                )
                return None

        if not p.exists():
            _print(f"  Session file not found: {p}", style="yellow")
            return None

        try:
            payload = json.loads(p.read_text())
        except Exception as exc:
            _print(f"  Could not load session: {exc}", style="yellow")
            return None

        if payload.get("type") != "workflow":
            _print(
                "  Session file is not a workflow session — use "
                "--resume-session with a workflow_*.json file.",
                style="yellow",
            )
            return None

        session_id: str = payload.get("session_id", p.stem)
        raw_state: dict = payload.get("state", {})
        try:
            state = self._restore_state(raw_state)
        except Exception as exc:
            _print(f"  Session state could not be restored: {exc}", style="yellow")
            return None

        return state, session_id

    @staticmethod
    def _restore_state(raw: dict) -> "WorkflowState":
        """Reconstruct a WorkflowState dataclass from an asdict() dict."""
        trace_history = [
            _TraceRun(**r) for r in raw.get("trace_history", [])
        ]
        analysis_history = []
        for s in raw.get("analysis_history", []):
            s = dict(s)
            # Remove any unknown keys that may have been added in later versions
            known = {f.name for f in dataclasses.fields(_AnalysisSnapshot)}
            analysis_history.append(_AnalysisSnapshot(**{k: v for k, v in s.items() if k in known}))
        edit_history = [
            _EditRecord(**e) for e in raw.get("edit_history", [])
        ]
        checkpoints = []
        for c in raw.get("checkpoints", []):
            c = dict(c)
            known = {f.name for f in dataclasses.fields(CheckpointRecord)}
            checkpoints.append(CheckpointRecord(**{k: v for k, v in c.items() if k in known}))
        return WorkflowState(
            app_command=raw.get("app_command", ""),
            source_paths=raw.get("source_paths", []),
            profiling_command=raw.get("profiling_command", ""),
            trace_history=trace_history,
            analysis_history=analysis_history,
            edit_history=edit_history,
            iteration_count=raw.get("iteration_count", 0),
            repo_root=raw.get("repo_root", ""),
            baseline_commit=raw.get("baseline_commit", ""),
            checkpoints=checkpoints,
            active_checkpoint=raw.get("active_checkpoint"),
            blacklisted_approaches=raw.get("blacklisted_approaches", []),
            plateau_iteration_count=raw.get("plateau_iteration_count", 0),
            seen_recommendation_hashes=raw.get("seen_recommendation_hashes", []),
            conversation=raw.get("conversation"),
        )

    def _save_session(self) -> None:
        """Serialize WorkflowState to ~/.rocinsight/sessions/workflow_<id>.json."""
        from dataclasses import asdict as _asdict

        # Flush conversation state before serializing
        if self._conv is not None:
            try:
                self._state.conversation = self._conv.to_dict()
            except Exception:
                pass

        # Warn if conversation has grown very large
        if self._state.conversation:
            _msg_count = len(self._state.conversation.get("messages", []))
            if _msg_count > 1000:
                import warnings
                warnings.warn(
                    f"Session has {_msg_count} messages. Consider starting a "
                    f"fresh session to reduce memory usage.",
                    stacklevel=2,
                )

        try:
            self._sessions_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "session_id": self._session_id,
                "type": "workflow",
                "app_command": self._state.app_command,
                "state": _asdict(self._state),
            }
            self._session_file.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            _print(f"  (Session save failed: {exc})", style="dim")

    def _init_checkpoints(self) -> None:
        """Detect git repo and record baseline commit.

        Sets self._state.repo_root and self._state.baseline_commit.
        If source is not in a git repo: leaves repo_root="" and checkpoints
        are silently disabled. Dirty working tree is not a problem — checkpoints
        only commit the specific files modified by each AI edit, leaving all
        other working tree changes untouched.
        """
        if not self._state.source_paths:
            return  # No source paths — checkpoints require a source location

        source = self._state.source_paths[0]
        try:
            gcm = GitCheckpointManager(
                repo_root="",  # Will be set by detect_repo
                session_id=self._session_id,
                sessions_dir=str(self._sessions_dir),
            )
            repo_root = gcm.detect_repo(source)
        except CheckpointError as exc:
            _print(
                f"  Note: checkpoints disabled — {exc}",
                style="dim",
            )
            return

        # Re-create with correct repo_root
        self._gcm = GitCheckpointManager(
            repo_root=repo_root,
            session_id=self._session_id,
            sessions_dir=str(self._sessions_dir),
        )

        self._state.repo_root = repo_root
        try:
            self._state.baseline_commit = self._gcm.get_head()
        except CheckpointError as exc:
            _print(
                f"  Note: checkpoints disabled — {exc}",
                style="dim",
            )
            self._state.repo_root = ""
            self._gcm = None
            return

    def _create_checkpoint(
        self,
        files_modified: List[str],
        edit_summary: str,
        file_snapshots: Dict[str, str],
    ) -> None:
        """Create a git commit + worktree checkpoint after an AI edit batch.

        Silently skips if checkpoints are disabled (self._gcm is None) or if
        any git operation fails (non-fatal — session continues without this cp).
        """
        if self._gcm is None:
            return

        cp_id = len(self._state.checkpoints)
        message = f"rocinsight: cp-{cp_id} — {edit_summary}"

        try:
            commit_hash = self._gcm.create_checkpoint_commit(files_modified, message)
            ref_name = self._gcm.tag_checkpoint(cp_id, commit_hash)
            worktree_path = self._gcm.add_worktree(cp_id, commit_hash)
        except CheckpointError as exc:
            _print(f"  (Checkpoint cp-{cp_id} skipped: {exc})", style="dim")
            return

        cp = CheckpointRecord(
            cp_id=cp_id,
            commit_hash=commit_hash,
            ref_name=ref_name,
            worktree_path=worktree_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
            files_modified=files_modified,
            edit_summary=edit_summary,
            file_snapshots=file_snapshots,
        )
        self._state.checkpoints.append(cp)

        # Link the most recent edit record to this checkpoint
        if self._state.edit_history:
            self._state.edit_history[-1].checkpoint_id = cp_id

    def _update_checkpoint_with_run(self) -> None:
        """Set run_index on the most recent CheckpointRecord that does not yet
        have a run attached.

        Called from Phase 3 immediately after trace_history is appended.
        Delta computation is deferred to _update_checkpoint_delta() which is
        called from Phase 4 after analysis_history is updated.
        """
        if not self._state.checkpoints:
            return

        # Find the most recent checkpoint without a run
        for cp in reversed(self._state.checkpoints):
            if cp.run_index is None:
                cp.run_index = len(self._state.trace_history) - 1
                break

    def _update_checkpoint_delta(self) -> None:
        """Compute performance_delta_pct for the most recently-run checkpoint.

        Called from Phase 4 after _record_analysis() appends to analysis_history,
        so analysis_history[-1] is the current run and analysis_history[-2] is
        the previous run.  Requires at least 2 entries in analysis_history.
        """
        if not self._state.checkpoints or len(self._state.analysis_history) < 2:
            return

        # Find the most recent checkpoint that has a run but no delta yet
        target = None
        for cp in reversed(self._state.checkpoints):
            if cp.run_index is not None and cp.performance_delta_pct is None:
                target = cp
                break
        if target is None:
            return

        prev_ns = (self._state.analysis_history[-2].execution_breakdown or {}).get(
            "total_runtime_ns", 0
        )
        curr_ns = (self._state.analysis_history[-1].execution_breakdown or {}).get(
            "total_runtime_ns", 0
        )
        if prev_ns > 0:
            target.performance_delta_pct = round(((prev_ns - curr_ns) / prev_ns) * 100, 1)

    def _restore_from_snapshots(self, files: set, snapshots: Dict[str, str]) -> None:
        """Write file contents from snapshots dict; delete files not in snapshots."""
        for f in files:
            _path = (
                pathlib.Path(self._state.repo_root) / f
                if self._state.repo_root
                else pathlib.Path(f)
            )
            if f in snapshots:
                _path.parent.mkdir(parents=True, exist_ok=True)
                _path.write_text(snapshots[f])
            else:
                if _path.exists():
                    _path.unlink()

    def _rollback_to_checkpoint(self, target_cp_id: int) -> None:
        """Restore working directory to cp target_cp_id state.

        target_cp_id == -1 means baseline (before any AI edits).
        Removes all checkpoints after target, truncates trace/analysis history,
        and sets active_checkpoint. Uses git fast path if commit is reachable,
        falls back to file_snapshots otherwise.
        """
        if target_cp_id == -1:
            target_hash = self._state.baseline_commit
            modified_after = set()
            for cp in self._state.checkpoints:
                modified_after.update(cp.files_modified)
            target_snapshots: Dict[str, str] = {}
        else:
            matches = [cp for cp in self._state.checkpoints if cp.cp_id == target_cp_id]
            if not matches:
                _print(f"  Checkpoint {target_cp_id} not found.", style="dim")
                return
            target = matches[0]
            target_hash = target.commit_hash
            modified_after = set()
            for cp in self._state.checkpoints:
                if cp.cp_id > target_cp_id:
                    modified_after.update(cp.files_modified)
            target_snapshots = target.file_snapshots

        if not modified_after:
            pass  # Nothing to restore
        elif self._gcm and self._gcm.commit_reachable(target_hash):
            try:
                self._gcm.restore_files_from_commit(target_hash, list(modified_after))
            except CheckpointError as exc:
                _print(
                    f"  Git restore failed: {exc}. Using file snapshots.", style="yellow"
                )
                self._restore_from_snapshots(modified_after, target_snapshots)
        else:
            # Fallback: write file snapshots directly
            if target_cp_id == -1:
                _print(
                    "  \u2717 Cannot restore baseline: git unavailable and no file "
                    "snapshots exist for the baseline state.",
                    style="red",
                )
                pass  # fall through to cleanup: truncate checkpoints/history, _save_session
            else:
                self._restore_from_snapshots(modified_after, target_snapshots)
            _print(
                "  Note: restored from session file snapshot (git unavailable).",
                style="dim",
            )

        # Remove stale worktrees
        if self._gcm:
            for cp in self._state.checkpoints:
                if cp.cp_id > target_cp_id:
                    self._gcm.remove_worktree(cp.worktree_path)

        # Truncate checkpoints list
        if target_cp_id == -1:
            self._state.checkpoints = []
        else:
            self._state.checkpoints = self._state.checkpoints[: target_cp_id + 1]

        # Truncate trace/analysis history
        if target_cp_id == -1:
            self._state.trace_history = []
            self._state.analysis_history = []
            self._state.iteration_count = 0
            self._state.active_checkpoint = None
        else:
            run_idx = target.run_index
            if run_idx is not None:
                self._state.trace_history = self._state.trace_history[: run_idx + 1]
                self._state.analysis_history = self._state.analysis_history[: run_idx + 1]
                self._state.iteration_count = run_idx + 1
            else:
                self._state.trace_history = []
                self._state.analysis_history = []
                self._state.iteration_count = 0
            self._state.active_checkpoint = target_cp_id

        self._save_session()

    def _blacklist_checkpoint(self, cp_id: int) -> None:
        """Mark a checkpoint as blacklisted using its edit_summary as category."""
        matches = [cp for cp in self._state.checkpoints if cp.cp_id == cp_id]
        if not matches:
            return
        cp = matches[0]
        delta_str = (
            f"{cp.performance_delta_pct:.1f}%"
            if cp.performance_delta_pct is not None
            else "unknown regression"
        )
        cp.blacklisted = True
        cp.blacklist_category = cp.edit_summary
        cp.blacklist_description = (
            f"'{cp.edit_summary}' caused {delta_str} performance regression. "
            "Do not suggest this approach again."
        )
        self._state.blacklisted_approaches.append(cp.blacklist_description)
        self._save_session()

    def _build_blacklist_block(self) -> str:
        """Return LLM prompt block for all blacklisted checkpoints.

        Uses the persistent blacklisted_approaches list so entries survive rollback.
        Deduplicates. Returns empty string when none blacklisted.
        """
        approaches = self._state.blacklisted_approaches
        if not approaches:
            return ""
        seen: set = set()
        lines = ["# Blacklisted approaches (do NOT use these):", ""]
        for desc in approaches:
            if desc and desc not in seen:
                seen.add(desc)
                lines.append(f"- {desc}")
        if len(lines) <= 2:
            return ""
        return "\n".join(lines) + "\n"

    def _show_checkpoint_picker(self) -> Optional[int]:
        """Display checkpoint table; prompt for target and optional blacklist.

        Returns target cp_id (-1 = baseline) or None if cancelled.
        """
        _print("\n┌─ Checkpoints " + "─" * 53 + "┐")
        _print("│  [base]  baseline (before any AI edits)" + " " * 31 + "│")
        for cp in self._state.checkpoints:
            if cp.performance_delta_pct is not None:
                if cp.performance_delta_pct > 0:
                    delta_str = f"+{cp.performance_delta_pct:.1f}% ✓"
                else:
                    delta_str = f"{cp.performance_delta_pct:.1f}% ✗"
            else:
                delta_str = "no run yet"
            summary = cp.edit_summary[:38].ljust(38)
            run_str = (
                f"Run {cp.run_index + 1}: {delta_str}"
                if cp.run_index is not None
                else "no run yet"
            )
            _print(f"│  [{cp.cp_id}]  {summary}  {run_str:20s}│")
        _print("└" + "─" * 69 + "┘\n")

        valid = ["base"] + [str(cp.cp_id) for cp in self._state.checkpoints]
        raw = _input(f"  Restore to [{'/'.join(valid)}] or [c] cancel: ").strip().lower()
        if raw == "c":
            return None
        if raw == "base":
            target_cp_id = -1
        elif raw.isdigit() and int(raw) in {cp.cp_id for cp in self._state.checkpoints}:
            target_cp_id = int(raw)
        else:
            _print("  Invalid choice.", style="yellow")
            return None

        # Identify regressions BEFORE rollback removes them
        if target_cp_id == -1:
            # Rolling back to baseline: show all checkpoints with regressions
            regression_cps = [
                cp
                for cp in self._state.checkpoints
                if cp.performance_delta_pct is not None and cp.performance_delta_pct < 0
            ]
        else:
            regression_cps = [
                cp
                for cp in self._state.checkpoints
                if cp.cp_id > target_cp_id
                and cp.performance_delta_pct is not None
                and cp.performance_delta_pct < 0
            ]

        # Prompt for blacklist (before rollback so checkpoints are still in list)
        blacklist_ids: List[int] = []
        if regression_cps:
            _print("\n  Blacklist approaches that caused regressions?")
            for idx, cp in enumerate(regression_cps, 1):
                delta_str = f"{cp.performance_delta_pct:.1f}%"
                _print_m(f"    [bold {_AMD_RED}][{idx}][/bold {_AMD_RED}]  [dim]cp-{cp.cp_id}: {cp.edit_summary} ({delta_str})[/dim]")
            bl_raw = (
                _input("  Enter numbers to blacklist (space-separated), or [n] skip: ")
                .strip()
                .lower()
            )
            if bl_raw != "n":
                for tok in bl_raw.split():
                    if tok.isdigit():
                        idx = int(tok) - 1
                        if 0 <= idx < len(regression_cps):
                            blacklist_ids.append(regression_cps[idx].cp_id)

        # Apply blacklists before rollback removes checkpoints
        for cp_id in blacklist_ids:
            self._blacklist_checkpoint(cp_id)

        # Now perform rollback
        self._rollback_to_checkpoint(target_cp_id=target_cp_id)

        return target_cp_id

    def _teardown_checkpoints(self) -> None:
        """Remove all checkpoint worktrees on session exit.

        Refs (refs/rocinsight/…) are kept so commits survive GC until
        the user runs 'rocinsight sessions --cleanup'.
        """
        if self._gcm is None:
            return
        for cp in self._state.checkpoints:
            try:
                self._gcm.remove_worktree(cp.worktree_path)
            except Exception:
                pass

    def _prune_stale_worktrees(self) -> None:
        """Remove worktrees under ~/.rocinsight/sessions/ with no matching session JSON.

        Called at session start, after git repo is detected.
        """
        if self._gcm is None:
            return
        sessions_dir = str(self._sessions_dir)
        try:
            worktree_paths = self._gcm.list_worktrees()
        except Exception:
            return

        for path in worktree_paths:
            if not path.startswith(sessions_dir + os.sep):
                continue
            # Extract session_id: first path component after sessions_dir
            rel = path[len(sessions_dir) + 1 :]
            parts = rel.split(os.sep)
            if not parts:
                continue
            session_id = parts[0]
            # Never prune the current session's own worktrees
            if session_id == self._session_id:
                continue
            # Check for matching session JSON
            json_candidates = [
                self._sessions_dir / f"workflow_{session_id}.json",
                self._sessions_dir / f"{session_id}.json",
            ]
            if not any(p.exists() for p in json_candidates):
                self._gcm.remove_worktree(path)
                _print(f"  Pruned stale checkpoint worktree: {path}", style="dim")

    # ── Phase 1b: Quick workload analysis ──────────────────────────────────────

    @staticmethod
    def _classify_app_command(app_cmd: str) -> Dict[str, Any]:
        """Inspect the app command for workload-type hints.

        Returns a dict with:
          workload_type  – one of: "python_ml", "python_generic", "hip_compute",
                           "llm_inference", "mpi_multi", "unknown"
          hints          – list of human-readable detection notes
          extra_flags    – additional rocprofv3 flags to add beyond the default set
          warnings       – list of capture-limitation warnings to show
          uses_fork      – True when the app spawns child processes via fork/exec
                           so the profiling command should use --process-sync and
                           per-process output filenames (%nid%)
          mpi_wrap       – True when the launcher is mpirun/mpiexec/srun; the
                           profiling command must be restructured as
                           ``mpirun <args> rocprofv3 <flags> -- <binary>`` so
                           each rank gets its own profiler instance
        """
        try:
            tokens = shlex.split(app_cmd)
        except ValueError:
            tokens = app_cmd.split()
        if not tokens:
            return {
                "workload_type": "unknown",
                "hints": [],
                "extra_flags": [],
                "warnings": [],
                "uses_fork": False,
                "mpi_wrap": False,
            }

        # Strip leading KEY=VALUE env-var tokens (same logic as _phase3_run_profiler)
        while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
            tokens.pop(0)
        if not tokens:
            return {
                "workload_type": "unknown",
                "hints": [],
                "extra_flags": [],
                "warnings": [],
                "uses_fork": False,
                "mpi_wrap": False,
            }

        binary = tokens[0].lower()
        all_lower = " ".join(tokens).lower()

        hints: List[str] = []
        extra_flags: List[str] = []
        warnings: List[str] = []

        # ── Multi-process launchers ──────────────────────────────────────────
        is_mpi = any(
            kw in binary for kw in ("mpirun", "mpiexec", "srun", "jsrun", "orterun")
        )
        if is_mpi:
            # MPI: rocprofv3 must be placed *inside* mpirun so each rank gets its
            # own profiler instance: ``mpirun <args> rocprofv3 <flags> -- <binary>``
            # --process-sync (LD_PRELOAD) does NOT work with MPI because OpenMPI
            # strips the preloaded library from child processes.
            mpi_prefix, mpi_app = WorkflowSession._split_mpi_command(app_cmd)
            hints.append("MPI/Slurm launcher — rocprofv3 will wrap each rank individually")
            warnings.append(
                "MPI launcher detected. The profiler will be placed inside mpirun "
                "so each rank gets its own trace: "
                "mpirun <args> rocprofv3 <flags> -- <binary>"
            )
            return {
                "workload_type": "mpi_multi",
                "hints": hints,
                "extra_flags": [],
                "warnings": warnings,
                "uses_fork": False,
                "mpi_wrap": True,
                "mpi_prefix": mpi_prefix,
                "mpi_app": mpi_app,
            }

        # ── Python workloads ─────────────────────────────────────────────────
        is_python = "python" in binary or binary.endswith(".py")

        ml_keywords = (
            "torch",
            "pytorch",
            "tensorflow",
            "jax",
            "paddle",
            "mxnet",
            "onnx",
            "triton",
            "megatron",
            "deepspeed",
        )
        is_ml = any(kw in all_lower for kw in ml_keywords)

        llm_keywords = (
            "vllm",
            "llm",
            "llama",
            "mistral",
            "falcon",
            "gpt",
            "bert",
            "transformer",
            "inference",
            "generate",
            "decode",
        )
        is_llm = any(kw in all_lower for kw in llm_keywords)

        multiproc_keywords = (
            "torchrun",
            "torch.distributed",
            "accelerate",
            "deepspeed",
            "nccl",
            "ddp",
        )
        is_multiproc = any(kw in all_lower for kw in multiproc_keywords)

        if is_python:
            extra_flags.append("--hip-trace")  # Python HIP API overhead is significant
            if is_multiproc:
                # Fork-based distributed training — --process-sync follows child processes,
                # %nid% in output name gives each worker its own DB file.
                extra_flags.append("--process-sync")
                hints.append(
                    "distributed/multi-process training (--process-sync enabled)"
                )
                warnings.append(
                    "Distributed/multi-process training detected (torchrun/DDP/DeepSpeed). "
                    "Using --process-sync and per-process output naming (%nid%) so each "
                    "worker's GPU activity is captured and merged automatically."
                )
            if is_llm:
                hints.append("Python + LLM inference framework")
                return {
                    "workload_type": "llm_inference",
                    "hints": hints,
                    "extra_flags": extra_flags,
                    "warnings": warnings,
                    "uses_fork": is_multiproc,
                }
            if is_ml:
                hints.append("Python + ML framework (PyTorch / JAX / TF)")
                return {
                    "workload_type": "python_ml",
                    "hints": hints,
                    "extra_flags": extra_flags,
                    "warnings": warnings,
                    "uses_fork": is_multiproc,
                }
            hints.append("Python workload")
            return {
                "workload_type": "python_generic",
                "hints": hints,
                "extra_flags": extra_flags,
                "warnings": warnings,
                "uses_fork": is_multiproc,
            }

        # ── Compiled HIP / ROCm binary ───────────────────────────────────────
        hip_keywords = (
            "hip",
            "rocm",
            "roc",
            "hsa",
            "blas",
            "lapack",
            "fft",
            "conv",
            "gemm",
            "matmul",
        )
        if any(kw in binary for kw in hip_keywords):
            hints.append(f"HIP/ROCm binary ({tokens[0]})")
        else:
            hints.append(f"Compiled binary ({tokens[0]})")

        return {
            "workload_type": "hip_compute",
            "hints": hints,
            "extra_flags": extra_flags,
            "warnings": warnings,
            "uses_fork": False,
        }

    @staticmethod
    def _split_mpi_command(app_cmd: str) -> tuple:
        """Split an MPI command into (mpi_prefix, app_binary_with_args).

        E.g. ``mpirun -n 2 ./multi_gpu_demo arg1``
          → (``mpirun -n 2``, ``./multi_gpu_demo arg1``)

        The split happens at the first token that does not look like an
        mpirun flag or its value.  Any ``--`` separator is removed.
        Returns (app_cmd, "") when no launcher is detected.
        """
        try:
            tokens = shlex.split(app_cmd)
        except ValueError:
            tokens = app_cmd.split()
        if not tokens:
            return (app_cmd, "")

        launcher_keywords = ("mpirun", "mpiexec", "srun", "jsrun", "orterun")
        if not any(tokens[0].endswith(kw) for kw in launcher_keywords):
            return (app_cmd, "")

        # Flags that consume the next token as a value
        value_flags = {
            "-n", "--n", "-np", "--np", "-N", "--N",
            "-H", "--host", "--hosts", "-hostfile", "--hostfile",
            "--machinefile", "-machinefile",
            "-x", "--map-by", "--bind-to", "--rank-by",
            "--mca", "-mca",
            "--ntasks", "--ntasks-per-node", "--nodes",
            "--partition", "--job-name",
        }
        prefix_tokens: List[str] = [tokens[0]]
        i = 1
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--":
                i += 1  # skip separator; rest is the app
                break
            if tok in value_flags:
                # flag + its value
                prefix_tokens.append(tok)
                i += 1
                if i < len(tokens):
                    prefix_tokens.append(tokens[i])
                    i += 1
            elif tok.startswith("-"):
                prefix_tokens.append(tok)
                i += 1
            else:
                # First non-flag, non-value token → start of the app binary
                break

        mpi_prefix = " ".join(prefix_tokens)
        mpi_app = " ".join(tokens[i:])
        if not mpi_app:
            return (app_cmd, "")
        return (mpi_prefix, mpi_app)

    def _phase1b_quick_workload_analysis(self) -> Optional[str]:
        """Analyze the workload before Phase 2 to suggest the best starter command.

        - If source paths provided: runs Tier 0 SourceAnalyzer and uses its
          highest-priority recommendation flags.
        - Always runs app-command heuristics for extra flags / warnings.
        - Falls back to default --sys-trace --kernel-trace --memory-copy-trace --stats
          when nothing more specific can be determined.

        Returns the full suggested rocprofv3 command, or None to use the default.
        """
        _print()
        _print(
            "  ── Quick Workload Analysis ─────────────────────────────────",
            style=_AMD_RED,
        )

        # App-command heuristics
        app_info = self._classify_app_command(self._state.app_command)
        for warn in app_info.get("warnings", []):
            _print(f"  ⚠  {warn}", style="yellow")
        for hint in app_info.get("hints", []):
            _print(f"  Detected: {hint}", style="dim")

        source_cmd_flags: Optional[str] = None

        # Tier 0 source analysis
        if self._state.source_paths:
            try:
                from .source_analyzer import SourceAnalyzer

                plan = SourceAnalyzer(pathlib.Path(self._state.source_paths[0])).analyze()
                _print(
                    f"  Source scan: {plan.files_scanned} files, "
                    f"{plan.kernel_count} kernels, "
                    f"model={plan.programming_model}",
                    style="dim",
                )
                # Extract just the flags from the suggested first command
                # (strip the `-- <app>` part; we'll add the real app below)
                raw = plan.suggested_first_command
                sep_idx = raw.find(" -- ")
                if sep_idx != -1:
                    source_cmd_flags = raw[:sep_idx]
                else:
                    source_cmd_flags = raw
                _print(f"  Source analysis suggests: {source_cmd_flags}", style="dim")
            except Exception as exc:
                _print(f"  Source analysis skipped: {exc}", style="dim")

        # Build the final command
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = f"{self._trace_dir}/run_{run_id}"
        mpi_wrap = app_info.get("mpi_wrap", False)
        uses_fork = app_info.get("uses_fork", False)
        out_name = "results_%nid%" if uses_fork else "results"

        if mpi_wrap:
            # MPI: rocprofv3 goes inside mpirun so each rank gets its own profiler.
            # Structure: ``mpirun <args> rocprofv3 <flags> -- <binary>``
            mpi_prefix = app_info.get("mpi_prefix", "mpirun")
            mpi_app = app_info.get("mpi_app", self._state.app_command)
            base_flags = "--sys-trace --kernel-trace --memory-copy-trace --stats"
            extra = " ".join(app_info.get("extra_flags", []))
            if extra:
                base_flags = f"{base_flags} {extra}"
            cmd = (
                f"{mpi_prefix} rocprofv3 {base_flags} "
                f"-d {out_dir} -o results_%nid% "
                f"-- {mpi_app}"
            )
            reason = "MPI wrap — rocprofv3 inside mpirun"
        elif source_cmd_flags:
            # Source-derived flags — replace any `-d <dir>` and `-o <name>` with fresh values.
            # When the app forks, override -o with per-process naming regardless of what the
            # source analyzer suggested.
            flags = re.sub(r"-d\s+\S+", f"-d {out_dir}", source_cmd_flags)
            flags = re.sub(r"-o\s+\S+", f"-o {out_name}", flags)
            if "-d " not in flags:
                flags = f"{flags} -d {out_dir}"
            if "-o " not in flags:
                flags = f"{flags} -o {out_name}"
            # Append any extra flags from app-command heuristics that aren't already present
            for ef in app_info.get("extra_flags", []):
                if ef not in flags:
                    if "rocprofv3 " in flags:
                        flags = flags.replace("rocprofv3 ", f"rocprofv3 {ef} ", 1)
                    else:
                        flags = f"rocprofv3 {ef} {flags}"
            cmd = f"{flags} -- {self._state.app_command}"
            reason = "source analysis"
        else:
            # Pure heuristics path — build on top of the safe default flag set
            base_flags = "--sys-trace --kernel-trace --memory-copy-trace --stats"
            extra = " ".join(app_info.get("extra_flags", []))
            if extra:
                base_flags = f"{base_flags} {extra}"
            extra_info = (
                f" + heuristics ({', '.join(app_info['hints'])})"
                if app_info.get("hints")
                else ""
            )
            reason = f"default flags{extra_info}"
            cmd = (
                f"rocprofv3 {base_flags} "
                f"-d {out_dir} -o {out_name} "
                f"-- {self._state.app_command}"
            )

        _print(f"  Starter command basis: {reason}", style="dim")
        _print()
        return cmd

    # ── Phase 2: Profiling command generation ─────────────────────────────────

    def _build_profiling_command(self, app_info: Optional[Dict[str, Any]] = None) -> str:
        """Build a default rocprofv3 profiling command wrapping the user's app.

        When app_info indicates mpi_wrap=True, restructures the command as
        ``mpirun <args> rocprofv3 <flags> -- <binary>`` so each rank gets its
        own profiler instance.

        When app_info indicates uses_fork=True (Python DDP/torchrun), adds
        --process-sync and uses %nid% in the output filename.
        """
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = f"{self._trace_dir}/run_{run_id}"
        info = app_info or {}
        mpi_wrap = info.get("mpi_wrap", False)
        uses_fork = info.get("uses_fork", False)
        base = "--sys-trace --kernel-trace --memory-copy-trace --stats"

        if mpi_wrap:
            mpi_prefix = info.get("mpi_prefix", "mpirun")
            mpi_app = info.get("mpi_app", self._state.app_command)
            return (
                f"{mpi_prefix} rocprofv3 {base} "
                f"-d {out_dir} -o results_%nid% "
                f"-- {mpi_app}"
            )

        proc_sync = " --process-sync" if uses_fork else ""
        out_name = "results_%nid%" if uses_fork else "results"
        return (
            f"rocprofv3 {base}{proc_sync} "
            f"-d {out_dir} -o {out_name} "
            f"-- {self._state.app_command}"
        )

    def _phase2_show_command(self, cmd: str) -> bool:
        """Display boxed profiling command; return True if user approves."""
        _print()
        width = max(66, len(cmd) + 8)
        border = "─" * (width - 2)
        _print(f"╭{border}╮", style=_AMD_RED)
        _print("│  Profiling Command" + " " * (width - 21) + "│", style=_AMD_RED)
        _print("│" + " " * (width - 2) + "│", style=_AMD_RED)
        indent = "│  "
        tail = "  │"
        avail = width - len(indent) - len(tail)
        # Word-wrap command
        words = cmd.split()
        line = ""
        for word in words:
            if line and len(line) + 1 + len(word) > avail:
                _print(f"{indent}{line:<{avail}}{tail}", style=_AMD_RED)
                line = word
            else:
                line = f"{line} {word}".lstrip()
        if line:
            _print(f"{indent}{line:<{avail}}{tail}", style=_AMD_RED)
        _print("│" + " " * (width - 2) + "│", style=_AMD_RED)
        _print(f"╰{border}╯", style=_AMD_RED)
        _print()
        try:
            ans = (
                _input(
                    "  Would you like the interactive tool to run this command? [Y/n]  "
                )
                .strip()
                .lower()
            )
        except EOFError:
            return False
        if ans in ("n", "no"):
            _print()
            _print("  Command not run. Copy it to run manually:", style="dim")
            _print(f"  $ {cmd}", style="dim")
            return False
        return True

    # ── Revert helper ──────────────────────────────────────────────────────────

    def _post_revert_menu(self, show_retry: bool) -> str:
        """Ask the user what to do after a revert + LLM analysis.

        show_retry — True when the user can immediately ask the AI for another
                     code fix (Phase 6 context, no alternative applied yet).
                     False when "retry" doesn't apply (Phase 3 context, or an
                     alternative was already applied in this cycle).

        Returns one of: "retry" | "continue" | "exit"
        """
        _print()
        _print("  ── What would you like to do next? ─────────────────────", style=_AMD_RED)
        if show_retry:
            _menu_opt("f", "Try a different fix  — let the AI attempt another approach")
        _menu_opt("p", "Continue to re-profiling  (skip code changes this round)")
        _menu_opt("s", "Save session")
        _menu_opt("q", "Exit session")
        _print()
        prompt = "  [f/p/s/q]: " if show_retry else "  [p/s/q]: "
        try:
            choice = _input(prompt).strip().lower()
        except EOFError:
            return "continue"
        if choice in ("f", "fix", "retry") and show_retry:
            return "retry"
        if choice == "s":
            self._save_session()
            _print("  Session saved.", style="green")
            return "continue"
        if choice in ("q", "quit", "exit"):
            return "exit"
        return "continue"

    def _revert_last_edit(
        self, failure_reason: str = "", allow_retry: bool = True
    ) -> "tuple[bool, str]":
        """Restore the most recently AI-modified file from its .bak backup.

        After restoring, calls the LLM to analyze the failure and propose a
        concrete alternative, then shows a what-next menu.

        allow_retry — when True (Phase 6 context), the what-next menu includes
                      [f] Try a different fix.  When False (Phase 3 context,
                      where a new edit cannot be applied immediately), [f] is
                      hidden and "retry" is never returned.

        Returns (success, next_action) where next_action is one of:
          "retry"    — user wants to try another AI fix (only when allow_retry=True)
          "continue" — user wants to skip code changes and proceed to re-profiling
          "exit"     — user wants to end the session
        """
        if not self._state.edit_history:
            _print("  No AI edits to revert.", style="yellow")
            return False, "continue"
        record = self._state.edit_history[-1]
        bak = pathlib.Path(record.backup_path)
        dst = pathlib.Path(record.file_path)
        if not bak.exists():
            _print(f"  Backup not found: {bak}", style="red")
            return False, "continue"

        # Capture the failed edit content BEFORE overwriting it — we'll send it
        # to the LLM so it can see exactly what went wrong.
        try:
            failed_content = dst.read_text()
        except OSError:
            failed_content = ""

        try:
            original_content = bak.read_text()
        except OSError as exc:
            _print(f"  Backup read failed: {exc}", style="red")
            return False, "continue"

        try:
            dst.write_text(original_content)
            self._state.edit_history.pop()
            self._save_session()
            _print(
                f"  ✓ Reverted: {dst.name}  (backup kept at {bak.name})", style="green"
            )
        except OSError as exc:
            _print(f"  Revert failed: {exc}", style="red")
            return False, "continue"

        # LLM analysis + what-next menu.
        action = self._post_revert_llm_analysis(
            dst,
            original_content,
            failed_content,
            failure_reason,
            allow_retry=allow_retry,
        )
        return True, action

    def _post_revert_llm_analysis(
        self,
        file_path: pathlib.Path,
        original_content: str,
        failed_content: str,
        failure_reason: str,
        allow_retry: bool = True,
    ) -> str:
        """Call LLM to analyze the failed edit and propose a concrete alternative.

        Returns the user's next-action choice from _post_revert_menu:
        "retry" | "continue" | "exit"

        If failure_reason is empty the user is asked to describe the error
        before the LLM is called, so the analysis has useful context.
        """
        # Ask for error context before burning an LLM call if none was provided.
        if not failure_reason.strip():
            _print()
            _print(
                "  What went wrong? Paste the error output or briefly describe the issue.",
                style=_AMD_RED,
            )
            _print(
                "  (Press Enter to skip and proceed without error context)", style="dim"
            )
            lines: List[str] = []
            try:
                while True:
                    line = _input("  > ").strip()
                    if not line:
                        break
                    lines.append(line)
            except EOFError:
                pass
            failure_reason = "\n".join(lines)

        if not self._llm_provider:
            _print("  (No LLM configured — cannot auto-analyze failure)", style="dim")
            return self._post_revert_menu(show_retry=False)

        _print()
        _print("  ── Analyzing failure with LLM ──────────────────────────", style=_AMD_RED)

        error_block = (
            f"\n=== COMPILATION / RUNTIME ERRORS ===\n{failure_reason[:1500]}"
            if failure_reason
            else ""
        )
        failed_block = (
            f"\n=== FAILED EDIT ===\n{failed_content}" if failed_content else ""
        )

        # --- Conversation-based analysis (if available) ---
        analysis = None
        conv = self._ensure_conv()
        if conv is not None:
            try:
                _revert_msg = (
                    f"The edit to {file_path.name} was reverted because it caused errors."
                    f"{error_block}"
                    f"\n=== ORIGINAL CODE (now restored) ===\n{original_content}"
                    f"{failed_block}"
                    "\n\nAnalyze what went wrong. Format response as:\n"
                    "ANALYSIS: (root cause)\n"
                    "ALTERNATIVE: (corrected approach with specific code)"
                )
                _tokens: List[str] = []

                def _collect(tok: str) -> None:
                    _tokens.append(tok)

                with _Spinner(f"  {self._llm_provider} analyzing failure..."):
                    conv.send(_revert_msg, on_token=_collect, max_tokens=4096)
                _result = "".join(_tokens)
                if _result and _result.strip():
                    analysis = _result
            except Exception as exc:
                _print(
                    f"  (Conversation analysis failed: {exc}; falling back to one-shot)",
                    style="dim",
                )

        # --- Fallback: one-shot LLMAnalyzer (existing code) ---
        if analysis is None:
            system = (
                "You are an expert AMD GPU performance engineer. "
                "A code edit you suggested was reverted because it caused errors. "
                "Analyze what went wrong and propose a SPECIFIC corrected alternative.\n\n"
                "Format your response with exactly two sections:\n"
                "ANALYSIS: (root cause — what was wrong in the failed edit and why)\n"
                "ALTERNATIVE: (the corrected optimization approach with specific code changes — "
                "be concrete, not generic)"
            )
            user = (
                f"The edit to {file_path.name} was reverted because it failed."
                f"{error_block}"
                f"\n=== ORIGINAL CODE (now restored) ===\n{original_content}"
                f"{failed_block}"
            )

            try:
                from rocinsight.ai_analysis.llm_analyzer import LLMAnalyzer  # type: ignore[import]

                analyzer = LLMAnalyzer(
                    provider=self._llm_provider,
                    api_key=self._llm_api_key,
                    model=self._llm_model,
                )
                with _Spinner(f"  {self._llm_provider} analyzing failure..."):
                    if self._llm_provider == "openai":
                        analysis = analyzer._call_openai(system, user, timeout=120)
                    elif self._llm_provider == "anthropic":
                        analysis = analyzer._call_anthropic(system, user, timeout=120)
                    elif self._llm_provider == "private":
                        analysis = analyzer._call_private(system, user)
                    elif self._llm_provider == "claude-code":
                        analysis = analyzer._call_claude_code(system, user)
                    else:
                        analysis = analyzer._call_local(system, user)
            except Exception as exc:
                _print(f"  (LLM analysis failed: {exc})", style="red")
                return self._post_revert_menu(show_retry=allow_retry)

        if not analysis:
            return self._post_revert_menu(show_retry=allow_retry)

        _print()
        _print(analysis, style="white")
        _print()

        # Offer to apply the alternative right now.
        applied = False
        try:
            apply = (
                _input("  Apply this alternative approach now? [y/N]  ").strip().lower()
            )
        except EOFError:
            return self._post_revert_menu(show_retry=allow_retry)

        if apply == "y":
            # Extract the ALTERNATIVE section as the suggestion for _llm_rewrite_file.
            alt_section = analysis
            if "ALTERNATIVE:" in analysis:
                alt_section = analysis.split("ALTERNATIVE:", 1)[1].strip()

            rewritten = self._llm_rewrite_file(file_path, alt_section)
            if not rewritten or not rewritten.strip():
                _print("  (LLM did not produce a rewrite)", style="yellow")
            else:
                import difflib

                original_lines = original_content.splitlines(keepends=True)
                rewritten_lines = rewritten.splitlines(keepends=True)
                diff_lines = list(
                    difflib.unified_diff(
                        original_lines,
                        rewritten_lines,
                        fromfile=f"{file_path.name} (original)",
                        tofile=f"{file_path.name} (alternative)",
                        n=3,
                    )
                )
                if not diff_lines:
                    _print(
                        "  (Alternative is identical to original — no changes)",
                        style="yellow",
                    )
                else:
                    _print()
                    _print_m(f"  [bold {_AMD_RED}]── Proposed alternative ──────────────────────────[/bold {_AMD_RED}]")
                    for line in diff_lines[:120]:
                        _print_diff_line(line.rstrip("\n"))
                    if len(diff_lines) > 120:
                        _print(f"  ... ({len(diff_lines) - 120} more lines)", style="dim")
                    _print()
                    try:
                        confirm = (
                            _input("  Apply this corrected version? [y/N]  ")
                            .strip()
                            .lower()
                        )
                    except EOFError:
                        confirm = "n"
                    if confirm == "y":
                        bak2 = file_path.with_suffix(file_path.suffix + ".bak")
                        try:
                            bak2.write_text(original_content)
                            file_path.write_text(rewritten)
                            self._state.edit_history.append(
                                _EditRecord(
                                    timestamp=datetime.now(timezone.utc).isoformat(),
                                    file_path=str(file_path),
                                    backup_path=str(bak2),
                                )
                            )
                            self._save_session()
                            _print(
                                f"  ✓ Alternative applied: {file_path.name}",
                                style="green",
                            )
                            applied = True
                            # Run the compile-wait loop for the newly applied
                            # alternative, same as _phase6_apply_direct does
                            # after its initial LLM rewrite.
                            _print()
                            _print(
                                "  Changes applied. Please recompile your application.",
                                style=_AMD_RED,
                            )
                            _print(
                                "  Type 'done' when compiled, 'revert' to undo,",
                                style="dim",
                            )
                            _print(
                                "  'abort' to exit, or paste compilation errors.",
                                style="dim",
                            )
                            _alt_errors: List[str] = []
                            _alt_action: str = "continue"
                            while True:
                                try:
                                    _resp = _input("  > ").strip()
                                except EOFError:
                                    break
                                _rl = _resp.lower()
                                if _rl in ("done", "compiled", "ok", "yes", "y", ""):
                                    _print(
                                        "  Great — ready to re-profile.",
                                        style="green",
                                    )
                                    break
                                if _rl in ("revert", "undo", "rollback", "v", "r"):
                                    # Undo the alternative: restore original_content
                                    try:
                                        file_path.write_text(original_content)
                                        if self._state.edit_history:
                                            self._state.edit_history.pop()
                                        self._save_session()
                                        _print(
                                            f"  ✓ Reverted: {file_path.name}",
                                            style="yellow",
                                        )
                                    except OSError as _exc:
                                        _print(
                                            f"  Revert failed: {_exc}", style="red"
                                        )
                                    if _alt_errors:
                                        _err_ctx = "\n".join(_alt_errors)
                                        _print(f"  Errors from this attempt:\n{_err_ctx}", style="dim")
                                    _alt_action = self._post_revert_menu(
                                        show_retry=allow_retry
                                    )
                                    break
                                if _rl in ("abort", "cancel", "quit", "exit"):
                                    _alt_action = "exit"
                                    break
                                _alt_errors.append(_resp)
                                _print(
                                    "  Error noted. Type 'done' or 'revert'.",
                                    style="yellow",
                                )
                            return _alt_action
                        except OSError as exc:
                            _print(f"  (Write failed: {exc})", style="red")
                    else:
                        _print(
                            "  Alternative discarded. File remains at original.",
                            style="dim",
                        )

        # [f] only offered when no alternative was applied and allow_retry is True
        return self._post_revert_menu(show_retry=allow_retry and not applied)

    # ── Phase 3: Trace collection ──────────────────────────────────────────────

    def _merge_per_process_dbs(self, db_files: List[str], out_dir: str) -> Optional[str]:
        """Merge per-process DB files into a single database and return its path.

        Uses rocpd.merge.merge_sqlite_dbs() (the same engine behind `rocpd merge`).
        Returns None if merge fails — caller falls back to the first DB file.
        """
        if len(db_files) <= 1:
            return None
        merged_path = str(pathlib.Path(out_dir) / "merged_processes.db")
        # Remove stale merged file from prior iteration to avoid conflicts
        try:
            pathlib.Path(merged_path).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            from rocinsight.connection import merge_sqlite_dbs  # type: ignore[import]

            _print(f"  Merging {len(db_files)} per-process databases…", style=_AMD_RED)
            merge_sqlite_dbs(db_files, merged_path)
            _print(f"  ✓ Merged → {merged_path}", style="green")
            return merged_path
        except Exception as exc:
            _print(
                f"  ⚠  DB merge failed ({exc}); using first DB for analysis.",
                style="yellow",
            )
            return None

    def _find_trace_files(self, cmd: str) -> List[str]:
        """Parse -d <dir> from cmd; return .db/.csv/.json files found there."""
        import glob as _glob
        import shlex as _shlex

        try:
            parts = _shlex.split(cmd)
        except ValueError:
            parts = cmd.split()
        out_dir = "."
        for i, p in enumerate(parts):
            if p in ("-d", "--output-path") and i + 1 < len(parts):
                out_dir = parts[i + 1]
        found = []
        for ext in ("*.db", "*.csv", "*.json"):
            found.extend(_glob.glob(f"{out_dir}/**/{ext}", recursive=True))
            found.extend(_glob.glob(f"{out_dir}/{ext}"))
        return sorted(set(found))

    def _phase3_run_profiler(self, cmd: str) -> bool:
        """Run profiling command with real-time stdout streaming.

        On success (exit 0 + trace files found): records TraceRun, returns True.
        On failure: ask retry / edit command / revert-AI-edit / abort.

        Leading KEY=VALUE tokens (e.g. ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1) are
        extracted and injected into the child process environment automatically.
        """
        import shlex as _shlex

        while True:
            _print(f"  Running: $ {cmd}", style=_AMD_RED)
            _print()

            # Separate leading ENV=value tokens from the executable + args.
            # This lets callers build commands like:
            #   ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1 rocprofv3 --pc-sampling ...
            # without needing shell=True.
            _tokens = _shlex.split(cmd)
            _env_overrides: Dict[str, str] = {}
            while _tokens and "=" in _tokens[0] and not _tokens[0].startswith("-"):
                _key, _, _val = _tokens.pop(0).partition("=")
                _env_overrides[_key] = _val
            _run_env = {**os.environ, **_env_overrides} if _env_overrides else None

            if not _tokens:
                _print(
                    "  [error] Command is empty after stripping env vars.", style="red"
                )
                _print(f"  Command was: {cmd}", style="dim")

                class _FakeProc:
                    returncode = 127

                proc = _FakeProc()  # type: ignore[assignment]
            else:
                try:
                    proc = subprocess.Popen(
                        _tokens,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=_run_env,
                    )
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                    proc.wait()
                except FileNotFoundError as exc:
                    _print(f"  [error] Command not found: {exc}", style="red")

                    class _FakeProc:  # type: ignore[no-redef]
                        returncode = 127

                    proc = _FakeProc()  # type: ignore[assignment]

            _print()
            if proc.returncode == 0:
                trace_files = self._find_trace_files(cmd)
                if trace_files:
                    db_files = [f for f in trace_files if f.endswith(".db")]
                    _print(
                        f"  ✓ Trace collected: {len(trace_files)} file(s)"
                        + (f" ({len(db_files)} DB)" if db_files else ""),
                        style="green",
                    )
                    for tf in trace_files[:5]:
                        _print(f"    · {tf}", style="dim")
                    if len(trace_files) > 5:
                        _print(f"    … and {len(trace_files) - 5} more", style="dim")

                    # When multiple DB files are present (one per forked process),
                    # merge them into a single DB for analysis.
                    out_dir = "."
                    try:
                        import shlex as _sl

                        _parts = _sl.split(cmd)
                    except ValueError:
                        _parts = cmd.split()
                    for _i, _p in enumerate(_parts):
                        if _p in ("-d", "--output-path") and _i + 1 < len(_parts):
                            out_dir = _parts[_i + 1]

                    # Exclude any prior merged_processes.db from the input list
                    # (dir may be reused across iterations)
                    db_files = [
                        f for f in db_files
                        if not f.endswith("merged_processes.db")
                    ]
                    if len(db_files) > 1:
                        merged = self._merge_per_process_dbs(db_files, out_dir)
                        db_path = merged if merged else db_files[0]
                    else:
                        db_path = db_files[0] if db_files else trace_files[0]

                    self._state.trace_history.append(
                        _TraceRun(
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            command=cmd,
                            db_path=db_path,
                            trace_files=trace_files,
                        )
                    )
                    self._save_session()
                    self._update_checkpoint_with_run()
                    return True
                # Ran OK but no files found — ask user for path
                _print("  Profiler completed but no trace files found.", style="yellow")
                try:
                    db_input = _input(
                        "  Enter path to .db file (or Enter to abort): "
                    ).strip()
                except EOFError:
                    return False
                if db_input and pathlib.Path(db_input).exists():
                    self._state.trace_history.append(
                        _TraceRun(
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            command=cmd,
                            db_path=db_input,
                            trace_files=[db_input],
                        )
                    )
                    self._save_session()
                    self._update_checkpoint_with_run()
                    return True
                return False
            else:
                _print(
                    f"  Profiling command failed (exit code {proc.returncode}).",
                    style="red",
                )
                _menu_opt("r", "Retry same command")
                _menu_opt("e", "Edit the command and retry")
                if self._state.edit_history:
                    _menu_opt("v", "Revert last AI edit and retry")
                _menu_opt("a", "Abort")
                try:
                    choice = _input("  > ").strip().lower()
                except EOFError:
                    return False
                if choice == "r":
                    continue
                elif choice == "e":
                    try:
                        new_cmd = _input(f"  Edit command:\n  {cmd}\n  > ").strip()
                        if new_cmd:
                            if _has_shell_meta(new_cmd):
                                _print(
                                    "  Command contains shell metacharacters — rejected for safety.",
                                    style="red",
                                )
                            else:
                                cmd = new_cmd
                    except EOFError:
                        return False
                    continue
                elif choice == "v" and self._state.edit_history:
                    reverted, action = self._revert_last_edit(
                        failure_reason=f"Profiling command failed with exit code {proc.returncode}.",
                        allow_retry=False,
                    )
                    if action == "exit":
                        return False
                    if reverted and action != "retry":
                        _print("  Please recompile, then retry.", style=_AMD_RED)
                    continue
                else:
                    return False

    # ── Plateau detection helpers ─────────────────────────────────────────────

    def _compute_plateau_status(
        self, new_breakdown: Optional[Dict[str, Any]]
    ) -> tuple:
        """Check whether optimization has plateaued.

        Returns (is_plateaued: bool, consecutive_count: int, pct_change: float).
        """
        if new_breakdown is None or not self._state.analysis_history:
            return (False, 0, 0.0)

        prev = self._state.analysis_history[-1]
        prev_bd = prev.execution_breakdown or {}
        prev_ns = prev_bd.get("total_runtime_ns", 0)
        new_ns = new_breakdown.get("total_runtime_ns", 0)

        if prev_ns == 0:
            return (False, 0, 0.0)

        abs_pct_change = abs((new_ns - prev_ns) / prev_ns * 100)

        if abs_pct_change < _PLATEAU_THRESHOLD_PCT:
            self._state.plateau_iteration_count += 1
        else:
            self._state.plateau_iteration_count = 0

        is_plateaued = self._state.plateau_iteration_count >= _PLATEAU_MIN_ITERATIONS
        return (is_plateaued, self._state.plateau_iteration_count, abs_pct_change)

    def _filter_seen_recommendations(
        self, recs: List[Dict[str, Any]]
    ) -> tuple:
        """Filter out recommendations whose hash has been seen before.

        Returns (filtered_recs, suppressed_count).
        Does NOT update seen_recommendation_hashes (caller does that).
        """
        seen = set(self._state.seen_recommendation_hashes)
        filtered = []
        suppressed = 0
        for r in recs:
            _hash = f"{r.get('category', '')}:{r.get('issue', '')[:40]}"
            if _hash in seen:
                suppressed += 1
            else:
                filtered.append(r)
        return (filtered, suppressed)

    # ── Phase 4: AI trace analysis ─────────────────────────────────────────────

    def _record_analysis(
        self,
        recs: List[Dict[str, Any]],
        execution_breakdown: Optional[Dict[str, Any]],
        hotspots: List[Dict[str, Any]],
        ai_recommended_command: Optional[str] = None,
        plateau_detected: bool = False,
    ) -> _AnalysisSnapshot:
        snap = _AnalysisSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            iteration=self._state.iteration_count,
            recommendations=recs,
            execution_breakdown=execution_breakdown,
            hotspots=hotspots,
            ai_recommended_command=ai_recommended_command,
            plateau_detected=plateau_detected,
        )
        self._state.analysis_history.append(snap)
        self._state.iteration_count += 1
        return snap

    def _print_comparison(
        self,
        new_breakdown: Optional[Dict[str, Any]],
    ) -> None:
        # Called before the current snapshot is appended to analysis_history,
        # so [-1] is the most-recent *previous* run.
        if len(self._state.analysis_history) < 1:
            return
        prev = self._state.analysis_history[-1]
        pb = prev.execution_breakdown or {}
        nb = new_breakdown or {}
        prev_s = pb.get("total_runtime_ns", 0) / 1e9
        new_s = nb.get("total_runtime_ns", 0) / 1e9
        if prev_s == 0:
            return
        pct = (new_s - prev_s) / prev_s * 100
        arrow = "▼" if pct < 0 else "▲"
        _print()
        _print("  ── Performance Comparison ──────────────────────────────", style=_AMD_RED)
        _print(f"  {'Metric':<28}  {'Before':>8}  {'After':>8}  Change", style="bold")
        _print(
            f"  {'Total GPU time':<28}  {prev_s:>7.2f}s  {new_s:>7.2f}s  "
            f"{arrow} {abs(pct):.0f}%",
            style="green" if pct < 0 else "yellow",
        )
        for key, label in [
            ("kernel_time_pct", "Kernel %"),
            ("memcpy_time_pct", "MemCopy %"),
            ("api_overhead_pct", "API overhead %"),
        ]:
            pv = pb.get(key, 0)
            nv = nb.get(key, 0)
            diff = nv - pv
            _print(
                f"  {label:<28}  {pv:>7.1f}%  {nv:>7.1f}%  "
                f"{'▼' if diff < 0 else '▲'} {abs(diff):.1f}pp",
                style="green" if diff < 0 else "yellow",
            )
        _print()

    def _phase4_analyze(self, db_path: str) -> _AnalysisSnapshot:
        """Run Tier 1/2 analysis; print structured report; return snapshot."""
        iteration = len(self._state.analysis_history) + 1
        _print()
        if iteration == 1:
            header = "  ══ AI Trace Analysis Report " + "═" * 44
        else:
            header = f"  ══ AI Trace Analysis Report  (Run #{iteration}) " + "═" * 35
        _print(header, style=_AMD_RED_STYLE)
        _print()

        recs: List[Dict[str, Any]] = []
        breakdown: Optional[Dict[str, Any]] = None
        hotspots: List[Dict[str, Any]] = []

        try:
            from rocinsight.ai_analysis.api import analyze_database  # type: ignore[import]

            # Auto-detect ATT stats_*.csv files in the same directory as the db.
            # Present when user ran rocprofv3 --att (decoder writes CSVs alongside .db).
            _db_dir = str(pathlib.Path(db_path).parent)
            _att_csvs = list(pathlib.Path(_db_dir).glob("stats_*.csv"))
            _att_dir = _db_dir if _att_csvs else None

            result = analyze_database(
                pathlib.Path(db_path),
                enable_llm=bool(self._llm_provider),
                llm_provider=self._llm_provider or None,
                llm_api_key=self._llm_api_key or None,
                att_dir=_att_dir,
            )
            if _att_dir:
                _print(
                    f"  [ATT] Found {len(_att_csvs)} stats_*.csv file(s) — Tier 3 ATT analysis active",
                    style=_AMD_RED,
                )

            eb = result.execution_breakdown
            if eb:
                breakdown = {
                    "kernel_time_pct": eb.kernel_time_pct,
                    "memcpy_time_pct": eb.memcpy_time_pct,
                    "api_overhead_pct": eb.api_overhead_pct,
                    "idle_time_pct": eb.idle_time_pct,
                    "total_runtime_ns": result.profiling_info.total_duration_ns,
                }
                total_s = result.profiling_info.total_duration_ns / 1e9
                _print("  Summary:", style="white")
                _print(f"    Total GPU active time : {total_s:.3f}s", style="dim")
                _print(
                    f"    Kernel  {eb.kernel_time_pct:.1f}%  "
                    f"MemCopy {eb.memcpy_time_pct:.1f}%  "
                    f"Overhead {eb.api_overhead_pct:.1f}%",
                    style="dim",
                )
                _print()

            # Warn when GPU time is zero but profiling ran — likely multiprocessing.
            # Skip this warning when ATT data is present: --att alone produces an empty
            # kernel-dispatch DB (the trace is in stats_*.csv, not the DB).
            total_ns = result.profiling_info.total_duration_ns if eb else 0
            if total_ns == 0 and self._state.trace_history and not _att_dir:
                last_cmd = self._state.trace_history[-1].command
                # MPI: command has mpirun <args> rocprofv3 ... -- <binary>
                is_mpi_cmd = any(
                    kw in last_cmd.split()[0].lower()
                    for kw in ("mpirun", "mpiexec", "srun", "jsrun", "orterun")
                    if last_cmd.split()
                )
                already_has_sync = "--process-sync" in last_cmd
                _print("  ⚠  No GPU kernel activity captured.", style="yellow")
                if is_mpi_cmd:
                    _print(
                        "     MPI command run but no GPU activity detected.",
                        style="yellow",
                    )
                    _print(
                        "     Ensure the binary and GPU are accessible from all ranks.",
                        style="yellow",
                    )
                    _print(
                        "     Expected command structure:",
                        style="yellow",
                    )
                    _print(
                        "       mpirun -n <N> rocprofv3 --sys-trace -d <dir> -o results -- <binary>",
                        style="yellow",
                    )
                elif already_has_sync:
                    _print(
                        "     --process-sync is active but the DB is still empty.",
                        style="yellow",
                    )
                    _print(
                        "     The app may use Python multiprocessing 'spawn' (not fork).",
                        style="yellow",
                    )
                    _print(
                        "     Try:  rocprof-sys --trace -- <app>  (spawn-aware)",
                        style="yellow",
                    )
                    _print(
                        "     or:   profile a specific worker with --pid <worker_pid>",
                        style="yellow",
                    )
                else:
                    _print(
                        "     If your app spawns GPU work in child processes (fork/exec,",
                        style="yellow",
                    )
                    _print(
                        "     torchrun, DDP) add --process-sync to the profiling",
                        style="yellow",
                    )
                    _print(
                        "     command so rocprofv3 follows child processes, and use",
                        style="yellow",
                    )
                    _print(
                        "     -o results_%nid% so each process writes its own DB.",
                        style="yellow",
                    )
                    _print(
                        "     For MPI apps, use: mpirun -n N rocprofv3 ... -- <binary>",
                        style="yellow",
                    )
                    _print(
                        "     Alternatively: rocprof-sys --trace -- <app>",
                        style="yellow",
                    )
                _print()

            all_recs = (
                result.recommendations.high_priority
                + result.recommendations.medium_priority
                + result.recommendations.low_priority
            )

            # Get raw recs (which carry the structured `commands` list with
            # full_command strings) so we can surface them in the re-profiling menu.
            # Match by index — raw_recs and all_recs are in the same order;
            # raw recs have no stable id (the dataclass assigns "rec_001" etc. in api.py).
            raw_recs: List[Dict[str, Any]] = getattr(result, "_raw", {}).get(
                "recommendations_raw", []
            )

            for idx, r in enumerate(all_recs):
                raw_rec = raw_recs[idx] if idx < len(raw_recs) else {}
                recs.append(
                    {
                        "id": r.id,
                        "priority": r.priority,
                        "category": r.category,
                        "issue": r.title,
                        "suggestion": r.description,
                        "estimated_impact": r.estimated_impact,
                        "actions": r.next_steps,
                        "commands": raw_rec.get("commands", []),
                    }
                )

        except Exception as exc:
            _print(f"  (Analysis failed: {exc})", style="red")
            raw_recs = []

        # ── LLM-refined recommendations ──────────────────────────────────
        # When a conversation is available, ask the LLM to refine the
        # rule-based recommendations using the full fence context (GPU specs,
        # optimization techniques), edit history, and prior iteration results.
        # This prevents repetitive suggestions for already-applied optimizations.
        conv = self._ensure_conv()
        if conv is not None and recs and breakdown:
            try:
                _edit_summary = ""
                if self._state.edit_history:
                    _edits = [
                        pathlib.Path(e.file_path).name
                        for e in self._state.edit_history[-5:]
                    ]
                    _edit_summary = (
                        f"\n\nEdits already applied in this session: "
                        + ", ".join(_edits)
                    )

                _prior_recs = ""
                if self._state.seen_recommendation_hashes:
                    _prior_recs = (
                        "\n\nPrior recommendations already shown to user: "
                        + "; ".join(self._state.seen_recommendation_hashes[-10:])
                    )

                _refine_msg = (
                    f"Here are the rule-based recommendations from this iteration's trace analysis.\n"
                    f"Runtime: {breakdown.get('total_runtime_ns', 0)/1e9:.3f}s, "
                    f"kernel={breakdown.get('kernel_time_pct', 0):.1f}%, "
                    f"memcpy={breakdown.get('memcpy_time_pct', 0):.1f}%, "
                    f"api_overhead={breakdown.get('api_overhead_pct', 0):.1f}%, idle={breakdown.get('idle_time_pct', 0):.1f}%.\n\n"
                    f"Rule-based recommendations:\n"
                )
                for r in recs:
                    _refine_msg += (
                        f"- [{r.get('priority','')}] {r.get('issue','')}: "
                        f"{r.get('suggestion','')}\n"
                    )
                _refine_msg += (
                    f"{_edit_summary}{_prior_recs}\n\n"
                    "Based on your knowledge of AMD GPU optimization (from the reference guide), "
                    "the edit history, and prior recommendations:\n"
                    "1. Remove any recommendations for optimizations already applied in the edits\n"
                    "2. Adjust suggestion text to be specific to the current state (not generic)\n"
                    "3. Add any NEW recommendations the rules missed but the fence document covers\n"
                    "4. If the workload is init-dominated (<1s total, <5% kernel), say so\n\n"
                    "Output ONLY a numbered list of refined recommendations in this format:\n"
                    "[PRIORITY] Issue title\n"
                    "  Suggestion: specific actionable text\n"
                    "  Actions: bullet list\n\n"
                    "If a rule-based rec is still valid, keep it. If not, drop it."
                )
                _tokens: List[str] = []
                with _Spinner("  Refining recommendations with AI context..."):
                    conv.send(
                        _refine_msg,
                        on_token=lambda t: _tokens.append(t),
                        max_tokens=2048,
                    )
                _refined = "".join(_tokens).strip()
                if _refined:
                    # Parse refined recommendations and replace the rule-based ones
                    # Display the AI-refined output as the primary recommendation text
                    _print()
                    _print(
                        "  ── AI-Refined Analysis (context-aware) ────────────────",
                        style=_AMD_RED,
                    )
                    _print()
                    _print(_refined, style="white")
                    _print()
            except Exception as exc:
                pass  # Silently fall through to rule-based display

        # Source correlation note
        if self._state.source_paths:
            _print(
                f"  (Source paths provided: "
                f"{', '.join(pathlib.Path(p).name for p in self._state.source_paths[:3])})",
                style="dim",
            )
            _print()

        # Print each finding; show recommended commands beneath each issue
        for i, rec in enumerate(recs, 1):
            pri = rec.get("priority", "INFO")
            style = _PRI_STYLE.get(pri, "white")
            _print(f"  ─── Issue #{i}: {rec.get('issue', '')[:70]} ───", style=_AMD_RED)
            _print(f"  Severity   : {pri}", style=style)
            if rec.get("suggestion"):
                _print(f"  Root Cause : {rec['suggestion']}", style="dim")
            if rec.get("estimated_impact"):
                _print(f"  Impact     : {rec['estimated_impact']}", style="dim")
            for act in rec.get("actions", [])[:3]:
                _print(f"    • {act}", style="dim")
            cmds = rec.get("commands", [])
            if cmds:
                _print("  Suggested next commands:", style="dim")
                for cmd_obj in cmds[:3]:
                    fc = cmd_obj.get("full_command", "")
                    desc = cmd_obj.get("description", "")
                    if fc:
                        _print(f"    $ {fc}", style=_AMD_RED)
                    if desc:
                        _print(f"      ({desc})", style="dim")
            _print()

        if not recs:
            _print("  No significant bottlenecks detected.", style="green")
            _print()

        # Comparison with previous run
        if self._state.analysis_history:
            self._print_comparison(breakdown)

        # Plateau detection — warn the user and filter duplicate recommendations
        _all_recs_for_hashing = list(recs)  # snapshot before any filtering
        is_plateaued, consec_count, last_pct = self._compute_plateau_status(breakdown)
        if is_plateaued:
            _print(
                f"  \u26a0  Optimization plateau detected: <{last_pct:.1f}% change over "
                f"{consec_count} consecutive iterations",
                style="yellow",
            )
            _print(
                "     Consider deeper analysis: ATT (instruction stalls) or "
                "rocprof-compute (roofline model)",
                style="yellow",
            )
            filtered_recs, suppressed = self._filter_seen_recommendations(recs)
            if suppressed > 0:
                _print(
                    f"     ({suppressed} previously-seen recommendation(s) suppressed)",
                    style="dim",
                )
            recs = filtered_recs

        # Update seen recommendation hashes from the ORIGINAL recs (before
        # filtering), so suppressed recs are permanently marked as seen and
        # don't reappear on subsequent plateau iterations.
        for r in _all_recs_for_hashing:
            _hash = f"{r.get('category', '')}:{r.get('issue', '')[:40]}"
            if _hash not in self._state.seen_recommendation_hashes:
                self._state.seen_recommendation_hashes.append(_hash)
        # Cap the list
        if len(self._state.seen_recommendation_hashes) > _MAX_SEEN_REC_HASHES:
            self._state.seen_recommendation_hashes = self._state.seen_recommendation_hashes[-_MAX_SEEN_REC_HASHES:]

        # Derive AI-recommended re-profiling command from the first rocprofv3
        # command found in any recommendation, replacing the generic placeholder
        # with the actual application being profiled.
        ai_rec_cmd: Optional[str] = None
        app_cmd = self._state.app_command
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        new_out_dir = f"{self._trace_dir}/run_{run_id}"
        # Detect MPI app to restructure the command correctly
        _app_info = self._classify_app_command(app_cmd)
        _is_mpi = _app_info.get("mpi_wrap", False)
        for rec in recs:
            for cmd_obj in rec.get("commands", []):
                if cmd_obj.get("tool") == "rocprofv3":
                    fc = cmd_obj.get("full_command", "")
                    if fc and "-- ./app" in fc:
                        if _is_mpi:
                            # MPI: restructure as mpirun <args> rocprofv3 <flags> -- <binary>
                            mpi_prefix = _app_info.get("mpi_prefix", "mpirun")
                            mpi_app = _app_info.get("mpi_app", app_cmd)
                            # Extract rocprofv3 flags (everything between rocprofv3 and -- ./app)
                            _rp_idx = fc.find("rocprofv3 ")
                            _app_idx = fc.find("-- ./app")
                            if _rp_idx >= 0 and _app_idx > _rp_idx:
                                _rp_flags = fc[_rp_idx + len("rocprofv3 "):_app_idx].strip()
                                # Strip any existing -o <name> — we'll add our own
                                _rp_flags = re.sub(r"-o\s+\S+", "", _rp_flags).strip()
                                _rp_flags = re.sub(r" {2,}", " ", _rp_flags)
                                fc = f"{mpi_prefix} rocprofv3 {_rp_flags} -o results_%nid% -- {mpi_app}"
                            else:
                                fc = fc.replace("-- ./app", f"-- {app_cmd}")
                        else:
                            fc = fc.replace("-- ./app", f"-- {app_cmd}")
                        fc = _replace_output_dir(fc, new_out_dir)
                        # Strip flags not accepted by rocprofv3 CLI.
                        # The LLM fence documents valid flags, but LLMs still
                        # hallucinate non-existent names — strip defensively.
                        # (a) --hip-api-trace: invalid; correct flag is --hip-trace:
                        fc = re.sub(r"\s*--hip-api-trace\b", "", fc)
                        # (b) --kernel-names <value> — value-taking invalid flag:
                        fc = re.sub(
                            r"--kernel-names\s+(?:'[^']*'|\"[^\"]*\"|\S+)",
                            "",
                            fc,
                        )
                        fc = fc.strip()
                        fc = re.sub(r" {2,}", " ", fc)  # collapse extra spaces
                        ai_rec_cmd = fc
                        break
            if ai_rec_cmd:
                break

        # Don't re-suggest a collection profile whose data has already been gathered.
        # This prevents cycling between two INFO recommendations (e.g. "add --pmc
        # counters" → "add --sys-trace" → "add --pmc counters" → ...).
        # Strategy: fingerprint both PMC counters AND named trace flags, then compare
        # the suggested command against the UNION of everything collected across ALL
        # previous runs (not just the last one).
        if ai_rec_cmd and self._state.trace_history:
            _TRACE_FLAGS = frozenset(
                {
                    "--sys-trace",
                    "--hip-trace",
                    "--kernel-trace",
                    "--memory-copy-trace",
                    "--hsa-trace",
                    "--stats",
                }
            )

            def _collection_fingerprint(cmd: str) -> frozenset:
                items: set = set()
                # Individual PMC counters
                for m in re.finditer(
                    r"--pmc\s+((?:[A-Z_][A-Z0-9_]*(?:\s+|$))+)", cmd, re.IGNORECASE
                ):
                    items.update(f"pmc:{c}" for c in m.group(1).split())
                # Named trace collection flags
                for flag in _TRACE_FLAGS:
                    if flag in cmd:
                        items.add(flag)
                return frozenset(items)

            suggested_fp = _collection_fingerprint(ai_rec_cmd)
            # Union of everything collected across all previous runs
            already_fp = frozenset().union(
                *(_collection_fingerprint(t.command) for t in self._state.trace_history)
            )
            if suggested_fp and suggested_fp.issubset(already_fp):
                ai_rec_cmd = None  # every suggested collection already performed

        snap = self._record_analysis(
            recs, breakdown, hotspots, ai_recommended_command=ai_rec_cmd,
            plateau_detected=is_plateaued,
        )
        # Compute performance delta now that analysis_history has been updated
        self._update_checkpoint_delta()

        # Note: conversation context is seeded by the LLM-refined-recommendations
        # block above (which sends breakdown + recs to the LLM). No separate
        # context-injection call is needed — that would double API usage.

        return snap

    # ── Phase 5: Recommendations menu ─────────────────────────────────────────

    def _phase5_rec_menu(self, snap: _AnalysisSnapshot) -> Optional[tuple]:
        """Show recommendations as a numbered menu.

        Returns (mode, selected_recs) where mode='direct'|'diff', or None if skipped.
        Returns None when recommendations are profiling-guidance only (INFO priority),
        since those require re-profiling rather than source code changes.
        """
        recs = snap.recommendations
        if not recs:
            _print("  No recommendations to act on.", style="dim")
            return None

        # Determine if all recommendations are INFO-level profiling guidance
        # (i.e. "collect more data") with no actionable source code changes.
        all_info = all(r.get("priority", "INFO").upper() == "INFO" for r in recs)

        # Detect "already re-profiled, still no progress" — don't loop indefinitely.
        # This happens when: all INFO, iteration > 0, and no fresh AI command available.
        already_reprofiled = (
            all_info and snap.iteration > 0 and snap.ai_recommended_command is None
        )

        while True:
            _print()
            _print(
                "  ── Recommendations ─────────────────────────────────────",
                style=_AMD_RED_STYLE,
            )
            for i, rec in enumerate(recs, 1):
                pri = rec.get("priority", "INFO")
                issue = rec.get("issue", "")[:70]
                _print_m(f"  [bold {_AMD_RED}]\\[{i}][/bold {_AMD_RED}]  {_priority_badge(pri)}  {issue}")
            _print()
            has_real_data = bool(self._state.trace_history)
            _att_already_run = any(
                "--att" in tr.command for tr in self._state.trace_history
            )
            if all_info and already_reprofiled:
                # Re-profiling already attempted; nothing new to suggest at Tier 1/2.
                if has_real_data:
                    # GPU data captured — we've exhausted Tier 1/2 analysis.
                    # Offer deeper tiers (skip options already collected).
                    _print(
                        "  All Tier 1/2 data collected. To investigate further:",
                        style="yellow",
                    )
                    _print(
                        "  • TraceLens interval + kernel-category analysis: "
                        "already shown in the report above",
                        style="dim",
                    )
                    if not _att_already_run:
                        _print(
                            "  • ATT (Tier 3): per-instruction stall analysis",
                            style="dim",
                        )
                    _print(
                        "  • PC Sampling (Tier 3): instruction-level hotspots "
                        "within each kernel",
                        style="dim",
                    )
                    _print(
                        "  • rocprof-compute / Omniperf: roofline + detailed "
                        "micro-architecture metrics",
                        style="dim",
                    )
                    _print()
                    _menu_opt("d", "Go deeper: collect PC sampling data", "Tier 3 — stochastic")
                    if not _att_already_run:
                        _menu_opt("t", "Go deeper: collect ATT trace", "Tier 3 — instruction stall analysis")
                else:
                    # No GPU data at all — likely multiprocessing spawn issue.
                    _print(
                        "  Analysis result unchanged after re-profiling.",
                        style="yellow",
                    )
                    _print(
                        "  The profiler may not be capturing GPU kernels from this app.",
                        style="yellow",
                    )
                    _print(
                        "  See the ⚠ note above for multi-process profiling options.",
                        style="yellow",
                    )
                _print()
                if self._state.checkpoints:
                    _menu_opt("b", "Roll back to a checkpoint")
                _menu_opt("s", "Save session")
                _menu_opt("n", "Skip — stop re-profiling")
                _menu_opt("q", "Quit session")
            elif snap.plateau_detected and not already_reprofiled:
                # Optimization has plateaued — suggest deeper analysis tiers
                _print("  Optimization has plateaued. Consider deeper analysis:", style="yellow")
                if not _att_already_run:
                    _print(
                        "  \u2022 ATT (Tier 3): per-instruction stall analysis",
                        style="dim",
                    )
                _print(
                    "  \u2022 PC Sampling (Tier 3): instruction-level hotspots "
                    "within each kernel",
                    style="dim",
                )
                _print()
                _menu_opt("d", "Go deeper: collect PC sampling data", "Tier 3 \u2014 stochastic")
                if not _att_already_run:
                    _menu_opt("t", "Go deeper: collect ATT trace", "Tier 3 \u2014 instruction stall analysis")
                if not all_info:
                    _menu_opt("a", "Address all with AI optimization")
                if self._state.checkpoints:
                    _menu_opt("b", "Roll back to a checkpoint")
                _menu_opt("s", "Save session")
                _menu_opt("n", "Skip \u2014 proceed to re-profiling")
                _menu_opt("q", "Quit session")
            elif all_info:
                # Only profiling-guidance recommendations — no source code to optimize.
                # Direct the user to re-profile with the suggested commands.
                _menu_opt("r", "Re-profile with suggested commands")
                if self._state.checkpoints:
                    _menu_opt("b", "Roll back to a checkpoint")
                _menu_opt("s", "Save session")
                _menu_opt("n", "Skip")
                _menu_opt("q", "Quit session")
            else:
                _menu_opt("a", "Address all with AI optimization")
                if self._state.checkpoints:
                    _menu_opt("b", "Roll back to a checkpoint")
                _menu_opt("s", "Save session")
                _menu_opt("n", "Skip — proceed to re-profiling")
                _menu_opt("q", "Quit session")
            _print()
            try:
                choice = _input("  Enter choice: ").strip().lower()
            except EOFError:
                return None

            if choice == "q":
                return None
            if choice == "s":
                self._save_session()
                _print("  Session saved.", style="green")
                continue
            if choice in ("n", ""):
                return None
            if choice == "b" and self._state.checkpoints:
                self._show_checkpoint_picker()
                return None
            _show_deeper = (all_info and already_reprofiled and has_real_data) or (
                snap.plateau_detected and not already_reprofiled
            )
            if choice == "d" and _show_deeper:
                # Build PC sampling command and route it to Phase 7 as option [3].
                run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                new_dir = f"{self._trace_dir}/run_{run_id}"
                pc_cmd = (
                    f"ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1 rocprofv3 --pc-sampling"
                    f" -d {new_dir} -o results -- {self._state.app_command}"
                )
                snap.ai_recommended_command = pc_cmd
                _print()
                _print(
                    "  PC sampling command ready. Proceeding to re-profiling.",
                    style="dim",
                )
                _print("  Select [3] at the next prompt to run it.", style="dim")
                return None
            if (
                choice == "t"
                and _show_deeper
                and not _att_already_run
            ):
                # Build ATT command and route it to Phase 7 as option [3].
                # Phase 4 auto-detects stats_*.csv in the output dir and runs ATT analysis.
                run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                new_dir = f"{self._trace_dir}/run_{run_id}"
                att_cmd = (
                    f"rocprofv3 --att --att-library-path /opt/rocm/lib"
                    f" --att-target-cu 0"
                    f" -d {new_dir} -o results -- {self._state.app_command}"
                )
                snap.ai_recommended_command = att_cmd
                _print()
                _print(
                    "  ATT trace command ready. Proceeding to re-profiling.",
                    style="dim",
                )
                _print(
                    "  Select [3] at the next prompt to run it.",
                    style="dim",
                )
                _print(
                    "  Note: requires rocprof-trace-decoder installed (/opt/rocm/lib).",
                    style="dim",
                )
                return None
            if choice == "r" and all_info and not already_reprofiled:
                # Advance to re-profiling phase; AI-recommended command will be option [3].
                _print()
                _print(
                    "  Advancing to re-profiling. Select [3] to use the suggested command.",
                    style="dim",
                )
                return None
            if choice == "a" and not all_info:
                selected = recs
            elif choice.isdigit() and 1 <= int(choice) <= len(recs):
                selected = [recs[int(choice) - 1]]
                r = selected[0]
                # If the selected rec is INFO-level profiling guidance, direct to re-profiling.
                if r.get("priority", "INFO").upper() == "INFO":
                    _print()
                    _print(
                        "  This recommendation requires re-profiling with different flags,",
                        style="dim",
                    )
                    _print(
                        "  not source code changes. Proceeding to re-profiling step.",
                        style="dim",
                    )
                    return None
                _print()
                _print(
                    f"  ─── {r.get('issue', '')[:60]} [{r.get('priority', '')}] ───",
                    style=_AMD_RED,
                )
                if r.get("suggestion"):
                    _print(f"  Root Cause : {r['suggestion']}", style="dim")
                if r.get("estimated_impact"):
                    _print(f"  Impact     : {r['estimated_impact']}", style="green")
                for act in r.get("actions", [])[:5]:
                    _print(f"    • {act}", style="dim")
                _print()
            else:
                _print("  Invalid choice.", style="yellow")
                continue

            _print("  How would you like the optimization applied?", style=_AMD_RED)
            _menu_opt("1", "Edit files directly  (AI modifies source files in-place)")
            _menu_opt("2", "Provide a diff/patch file  (you review and apply manually)")
            _menu_opt("n", "Back to recommendations menu")
            _print()
            try:
                mode_choice = _input("  > ").strip().lower()
            except EOFError:
                return None
            if mode_choice == "1":
                return ("direct", selected)
            if mode_choice == "2":
                return ("diff", selected)
            # n → loop back to menu

    # ── Phase 6: Apply changes ─────────────────────────────────────────────────

    def _pick_file_from_source_paths(self) -> Optional[pathlib.Path]:
        """Present numbered list of source files; return chosen."""
        exts = {".hip", ".cpp", ".cu", ".cl", ".h", ".hpp", ".py"}
        files: List[pathlib.Path] = []
        for sp in self._state.source_paths:
            try:
                for p in sorted(pathlib.Path(sp).rglob("*")):
                    if p.suffix in exts and p.is_file():
                        files.append(p)
            except OSError:
                pass
        if not files:
            _print("  (No source files found in provided --source paths)", style="yellow")
            return None
        _print()
        _print("  Choose a file to edit:", style=_AMD_RED)
        for i, f in enumerate(files[:15], 1):
            try:
                label = f.relative_to(self._state.source_paths[0])
            except (ValueError, IndexError):
                label = f.name  # type: ignore[assignment]
            _menu_opt(str(i), str(label))
        try:
            choice = _input("  > ").strip()
            idx = int(choice) - 1
            if 0 <= idx < min(len(files), 15):
                return files[idx]
        except (ValueError, EOFError):
            pass
        return None

    def _llm_rewrite_file(
        self, file_path: pathlib.Path, suggestions: str
    ) -> Optional[str]:
        """Call LLM to rewrite file applying suggestions. Returns new content or None."""
        if not self._llm_provider:
            _print("  No LLM configured — cannot perform AI code edit.", style="yellow")
            return None
        try:
            original = file_path.read_text()
        except OSError as exc:
            _print(f"  (Cannot read {file_path.name}: {exc})", style="red")
            return None

        # Try conversation-based rewrite first (maintains cross-iteration context)
        conv = self._ensure_conv()
        if conv is not None:
            try:
                _rewrite_msg = (
                    "Apply these GPU optimizations to the source file below.\n\n"
                    "OUTPUT FORMAT — use SEARCH/REPLACE blocks:\n"
                    "<<<<<<< SEARCH\n<exact lines from original>\n=======\n"
                    "<replacement lines>\n>>>>>>> REPLACE\n\n"
                    "RULES:\n"
                    "1. SEARCH must be EXACT substring of the original file\n"
                    "2. Include 2-3 lines context around each change\n"
                    "3. Add // comment on changed lines\n"
                    "4. Output ONLY SEARCH/REPLACE blocks — no prose\n"
                    "5. Do NOT repeat changes that were already tried and showed no improvement\n\n"
                    f"=== OPTIMIZATIONS TO APPLY ===\n{suggestions}\n\n"
                    f"=== SOURCE FILE: {file_path.name} ===\n{original}"
                )
                _tokens: List[str] = []

                def _collect(tok: str) -> None:
                    _tokens.append(tok)

                with _Spinner(f"  {self._llm_provider} LLM rewriting {file_path.name}..."):
                    conv.send(_rewrite_msg, on_token=_collect, max_tokens=16384)

                result = "".join(_tokens)
                if result and result.strip():
                    # Try SEARCH/REPLACE parsing
                    if "<<<<<<< SEARCH" in result:
                        merged = _apply_search_replace_blocks(original, result)
                        if merged is not None:
                            return merged
                        _print(
                            "  ⚠  SEARCH/REPLACE blocks could not be applied (search text not found).",
                            style="yellow",
                        )
                        return None
                    # Full-file fallback with truncation detection
                    result = _strip_code_preamble(result)
                    if result and result.strip():
                        _ext = file_path.suffix.lower()
                        _is_c = _ext in (".cpp", ".c", ".cu", ".hip", ".hpp", ".h", ".cxx")
                        if _is_c:
                            if result.count("{") != result.count("}"):
                                _print("  ⚠  Truncated output (brace mismatch). Discarding.", style="yellow")
                                return None
                            # Strip trailing prose
                            _lines = result.rstrip().splitlines()
                            while _lines and not any(
                                c in _lines[-1].strip()
                                for c in ("{", "}", ";", "#", "//", "/*")
                            ) and not _lines[-1].strip().endswith(")"):
                                _lines.pop()
                            if _lines:
                                return "\n".join(_lines) + "\n"
                        return result
                return None
            except Exception as exc:
                _print(f"  (Conversation rewrite failed: {exc}; falling back to one-shot)", style="dim")
                # Fall through to existing LLMAnalyzer path

        try:
            from rocinsight.ai_analysis.llm_analyzer import LLMAnalyzer  # type: ignore[import]

            analyzer = LLMAnalyzer(
                provider=self._llm_provider,
                api_key=self._llm_api_key,
                model=self._llm_model,
            )
            system = (
                "You are a code transformation engine for AMD GPU source files.\n"
                "OUTPUT FORMAT — use SEARCH/REPLACE blocks (do NOT output the full file):\n\n"
                "For each change, output a block in this exact format:\n"
                "<<<<<<< SEARCH\n"
                "<exact lines from the original file to find>\n"
                "=======\n"
                "<replacement lines>\n"
                ">>>>>>> REPLACE\n\n"
                "RULES:\n"
                "1. The SEARCH section must be an EXACT substring of the original file —\n"
                "   copy it character-for-character including whitespace and comments.\n"
                "2. Include 2-3 lines of unchanged context around each change so the\n"
                "   match is unique.\n"
                "3. Add a short // inline comment on each changed line explaining why.\n"
                "4. Output ONLY the SEARCH/REPLACE blocks. No prose, no explanations,\n"
                "   no markdown. Start directly with <<<<<<< SEARCH.\n"
                "5. Each block is applied independently. Keep blocks small and focused —\n"
                "   one logical change per block.\n"
                "6. Do NOT output the full file. Only output the changed sections.\n"
            )
            user = (
                f"Apply these GPU optimizations to the source file below.\n\n"
                f"=== OPTIMIZATIONS TO APPLY ===\n{suggestions}\n\n"
                f"=== SOURCE FILE ===\n{original}"
            )
            # File rewrites can be large — use a generous timeout (5 min).
            _rewrite_timeout = 300
            with _Spinner(f"  {self._llm_provider} LLM rewriting {file_path.name}..."):
                if self._llm_provider == "openai":
                    try:
                        result = analyzer._call_openai(
                            system, user, max_tokens=16384, timeout=_rewrite_timeout
                        )
                    except Exception as exc:
                        if (
                            "too large" in str(exc).lower()
                            or "max_tokens" in str(exc).lower()
                        ):
                            result = analyzer._call_openai(
                                system, user, timeout=_rewrite_timeout
                            )
                        else:
                            raise
                elif self._llm_provider == "anthropic":
                    result = analyzer._call_anthropic(
                        system, user, max_tokens=16384, timeout=_rewrite_timeout
                    )
                elif self._llm_provider == "private":
                    result = analyzer._call_private(system, user)
                elif self._llm_provider == "claude-code":
                    result = analyzer._call_claude_code(system, user)
                else:
                    result = analyzer._call_local(system, user)
            if not result or not result.strip():
                return None

            # ── Try search/replace chunk format first ───────────────────
            if "<<<<<<< SEARCH" in result:
                merged = _apply_search_replace_blocks(original, result)
                if merged is not None:
                    return merged
                _print(
                    "  ⚠  SEARCH/REPLACE blocks found but could not be applied"
                    " (search text not found in file — file may have changed).",
                    style="yellow",
                )
                return None  # Don't fall through — raw SEARCH/REPLACE output is not a valid file

            # ── Full-file fallback ──────────────────────────────────────
            result = _strip_code_preamble(result)
            if not result or not result.strip():
                return None

            # Truncation detection for C/C++ full-file output
            _ext = file_path.suffix.lower()
            _is_c_family = _ext in (".cpp", ".c", ".cu", ".hip", ".hpp", ".h", ".cxx")
            if _is_c_family:
                _open = result.count("{")
                _close = result.count("}")
                if _open != _close:
                    _print(
                        f"  ⚠  LLM output appears truncated (braces: {_open} open,"
                        f" {_close} close). Discarding to prevent broken code.",
                        style="yellow",
                    )
                    return None
                # Braces balance — strip any trailing non-code prose the LLM
                # may have added after the last closing brace (e.g. "These
                # changes should improve...").  This avoids false truncation
                # rejections when the code itself is complete.
                _lines = result.rstrip().splitlines()
                while _lines:
                    _last = _lines[-1].strip()
                    if not _last:
                        _lines.pop()
                        continue
                    # Line looks like code if it has braces, semicolons,
                    # preprocessor directives, or C/C++ comment markers
                    if any(c in _last for c in ("{", "}", ";", "#", "//", "/*")):
                        break
                    # Also accept lines ending with ) — e.g. function closing
                    if _last.endswith(")"):
                        break
                    # Trailing prose — strip it
                    _lines.pop()
                if _lines:
                    result = "\n".join(_lines) + "\n"
            return result
        except Exception as exc:
            _print(f"  (LLM rewrite failed: {exc})", style="red")
            return None

    def _phase6_apply_direct(self, snap: _AnalysisSnapshot) -> Optional[str]:
        """Phase 6: AI edits source files in-place (.bak backup); waits for recompile.

        Returns None normally, "exit" if the user chose to end the session after a revert.
        The method loops internally when the user chooses "retry" after a revert.
        """
        suggestions = "\n\n".join(
            f"[{r.get('priority', '')}] {r.get('issue', '')}:\n"
            f"{r.get('suggestion', '')}\n"
            + "\n".join(f"  • {a}" for a in r.get("actions", []))
            for r in snap.recommendations
        )

        while True:  # retry loop — re-entered when user picks [f] Try a different fix
            chosen = self._pick_file_from_source_paths()
            if chosen is None:
                return None
            # Capture original content before the LLM call so the diff and backup
            # are consistent even if a build system touches the file mid-call.
            original = chosen.read_text()
            blacklist_block = self._build_blacklist_block()
            effective_suggestions = (
                (blacklist_block + "\n" + suggestions) if blacklist_block else suggestions
            )
            rewritten = self._llm_rewrite_file(chosen, effective_suggestions)
            while rewritten is None:
                try:
                    ans = _input("  Retry LLM rewrite? [y/N]  ").strip().lower()
                except EOFError:
                    return None
                if ans != "y":
                    return None
                rewritten = self._llm_rewrite_file(chosen, effective_suggestions)

            import difflib

            diff_lines = list(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    rewritten.splitlines(keepends=True),
                    fromfile=f"{chosen.name} (original)",
                    tofile=f"{chosen.name} (AI-edited)",
                    n=3,
                )
            )
            _print()
            _print(
                "  ── Proposed changes ─────────────────────────────────", style=_AMD_RED
            )
            for line in diff_lines[:120]:
                line = line.rstrip("\n")
                if line.startswith("+"):
                    _print(line, style="green")
                elif line.startswith("-"):
                    _print(line, style="red")
                else:
                    _print(line, style="dim")
            if len(diff_lines) > 120:
                _print(f"  ... ({len(diff_lines) - 120} more lines omitted)", style="dim")
            if not diff_lines:
                _print(
                    "  (No changes — rewritten file is identical to original)",
                    style="yellow",
                )
                return None
            _print()
            try:
                confirm = _input("  Apply these changes? [y/N]  ").strip().lower()
            except EOFError:
                return None
            if confirm != "y":
                _print("  Changes discarded.", style="dim")
                return None

            bak = chosen.with_suffix(chosen.suffix + ".bak")
            try:
                bak.write_text(original)
                chosen.write_text(rewritten)
                _print(f"  Backup : {bak}", style="dim")
                _print(f"  Updated: {chosen}", style="green")
                self._state.edit_history.append(
                    _EditRecord(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        file_path=str(chosen),
                        backup_path=str(bak),
                    )
                )
                self._save_session()
                # Create checkpoint after saving (captures file contents after the edit)
                _rel_path = (
                    os.path.relpath(str(chosen), self._state.repo_root)
                    if self._state.repo_root
                    else str(chosen)
                )
                _file_snapshots = {}
                try:
                    _file_snapshots[_rel_path] = chosen.read_text()
                except OSError:
                    pass
                self._create_checkpoint(
                    files_modified=[_rel_path],
                    edit_summary=_edit_summary_from_suggestions(suggestions),
                    file_snapshots=_file_snapshots,
                )
            except OSError as exc:
                _print(f"  (Write failed: {exc})", style="red")
                return None

            # Wait for recompile
            _print()
            _print("  Changes applied. Please recompile your application.", style=_AMD_RED)
            _print(
                "  Type 'done' when compiled, 'revert' to undo the AI edit,",
                style="dim",
            )
            _print("  'abort' to exit, or paste compilation errors.", style="dim")
            _compile_errors: List[str] = []
            _revert_action: str = "continue"
            while True:
                try:
                    resp = _input("  > ").strip()
                except EOFError:
                    break
                resp_lower = resp.lower()
                if resp_lower in ("done", "compiled", "ok", "yes", "y", ""):
                    _print("  Great — ready to re-profile.", style="green")
                    break
                if resp_lower in ("revert", "undo", "rollback", "v", "r"):
                    error_ctx = "\n".join(_compile_errors)
                    reverted, _revert_action = self._revert_last_edit(
                        failure_reason=error_ctx
                    )
                    break
                if resp_lower in ("abort", "cancel", "quit", "exit"):
                    _print("  Aborting. Backup preserved at: " + str(bak), style="dim")
                    _revert_action = "exit"
                    break
                # Treat as compilation error description — accumulate for context
                _compile_errors.append(resp)
                _print(
                    "  Error noted. Type 'done' when fixed or 'revert' to undo the edit.",
                    style="yellow",
                )

            if _revert_action == "retry":
                _print(
                    "  ── Trying a different fix ───────────────────────────",
                    style=_AMD_RED,
                )
                continue  # re-enter the while True loop above
            if _revert_action == "exit":
                return "exit"
            return None  # "continue" — proceed to Phase 7

    def _phase6_apply_diff(self, snap: _AnalysisSnapshot) -> None:
        """Phase 6 alt: Save suggestions to a patch file."""
        suggestions = "\n\n".join(
            f"[{r.get('priority', '')}] {r.get('issue', '')}:\n"
            f"  Suggestion: {r.get('suggestion', '')}\n"
            + "\n".join(f"  • {a}" for a in r.get("actions", []))
            for r in snap.recommendations
        )
        base = self._state.source_paths[0] if self._state.source_paths else "."
        diff_path = pathlib.Path(base) / "ai_optimizations.patch"
        try:
            diff_path.write_text(suggestions + "\n")
            _print(f"  Suggestions saved to: {diff_path}", style="green")
            _print("  Apply manually, recompile, then re-run profiling.", style="dim")
        except OSError as exc:
            _print(f"  (Could not save patch: {exc})", style="red")

    # ── Phase 7: Re-profiling loop ─────────────────────────────────────────────

    def _phase7_reprofiling_prompt(self) -> Optional[str]:
        """Ask which profiling command to use for re-profiling. Returns cmd or None."""
        while True:
            current = self._state.profiling_command
            ai_cmd: Optional[str] = None
            if self._state.analysis_history:
                ai_cmd = self._state.analysis_history[-1].ai_recommended_command

            _print()
            _print(
                "  Ready to re-profile. Which command would you like to run?", style=_AMD_RED
            )
            _menu_opt("1", "Same command as before:")
            _print(f"         {current}", style="dim")
            _menu_opt("2", "Let me edit the command first")
            if ai_cmd:
                _menu_opt("3", "Use AI-recommended command:")
                _print(f"         {ai_cmd}", style="dim")
            _menu_opt("s", "Save session")
            _menu_opt("n", "Stop — I'm done profiling")
            _print()
            try:
                choice = _input("  > ").strip().lower()
            except EOFError:
                return None
            if choice == "s":
                self._save_session()
                _print("  Session saved.", style="green")
                continue
            if choice in ("1", ""):
                return current
            elif choice == "2":
                try:
                    new_cmd = _input(
                        f"  Edit command (Enter to keep):\n  {current}\n  > "
                    ).strip()
                    if new_cmd and _has_shell_meta(new_cmd):
                        _print(
                            "  Command contains shell metacharacters — rejected for safety.",
                            style="red",
                        )
                        continue
                    return new_cmd or current
                except EOFError:
                    return current
            elif choice == "3" and ai_cmd:
                return ai_cmd
            return None

    # ── Session summary ────────────────────────────────────────────────────────

    def print_session_summary(self) -> None:
        """Print final session summary."""
        _print()
        _print("  ══════════════════════════════════════════", style=_AMD_RED_STYLE)
        _print("   Session Summary", style=_AMD_RED_STYLE)
        _print("  ══════════════════════════════════════════", style=_AMD_RED_STYLE)
        _print(f"  Iterations : {self._state.iteration_count}", style="white")

        if len(self._state.analysis_history) >= 2:
            first_bd = self._state.analysis_history[0].execution_breakdown or {}
            last_bd = self._state.analysis_history[-1].execution_breakdown or {}
            t0 = first_bd.get("total_runtime_ns", 0) / 1e9
            t1 = last_bd.get("total_runtime_ns", 0) / 1e9
            if t0 > 0:
                pct = (t1 - t0) / t0 * 100
                arrow = "▼" if pct < 0 else "▲"
                _print(
                    f"  GPU time   : {t0:.2f}s → {t1:.2f}s  "
                    f"({arrow} {abs(pct):.0f}%)",
                    style="white",
                )

        if self._state.edit_history:
            files = [pathlib.Path(e.file_path).name for e in self._state.edit_history]
            baks = [pathlib.Path(e.backup_path).name for e in self._state.edit_history]
            _print(f"  Modified   : {', '.join(files)}", style="white")
            _print(f"  Backups    : {', '.join(baks)}", style="dim")

        if self._state.trace_history:
            runs = [
                pathlib.Path(t.db_path).parent.name for t in self._state.trace_history
            ]
            _print(f"  Trace runs : {', '.join(runs)}", style="dim")

        if self._session_file.exists():
            _print(f"  Session    : {self._session_file}", style="dim")
            _print(
                f"  Resume     : rocinsight analyze --interactive "
                f'"{self._state.app_command}" '
                f"--resume-session {self._session_file}",
                style="dim",
            )

        _print("  ══════════════════════════════════════════", style=_AMD_RED_STYLE)
        _print()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the 7-phase workflow loop."""
        _print_startup_banner()

        try:
            self._init_checkpoints()
            self._prune_stale_worktrees()

            if self._resumed and self._state.trace_history:
                # ── Resumed session: skip Phase 1-2, re-analyze last DB ────────
                _print(
                    f"  Resuming session {self._session_id}",
                    style=f"bold {_AMD_RED}",
                )
                _print(
                    f"  Iteration {self._state.iteration_count}  ·  "
                    f"{len(self._state.trace_history)} trace run(s)  ·  "
                    f"{len(self._state.edit_history)} edit(s)",
                    style="dim",
                )
                _print()
                # Re-analyze the last collected DB so the user sees fresh recs
                last_run = self._state.trace_history[-1]
                snap = self._phase4_analyze(last_run.db_path)
                result = self._phase5_rec_menu(snap)
                if result is not None:
                    mode, selected_recs = result
                    scoped = _AnalysisSnapshot(
                        timestamp=snap.timestamp,
                        iteration=snap.iteration,
                        recommendations=selected_recs,
                        execution_breakdown=snap.execution_breakdown,
                        hotspots=snap.hotspots,
                        ai_recommended_command=snap.ai_recommended_command,
                    )
                    if mode == "direct":
                        if self._phase6_apply_direct(scoped) == "exit":
                            return
                    elif mode == "diff":
                        self._phase6_apply_diff(scoped)
                # Fall through to Phase 7
                next_cmd = self._phase7_reprofiling_prompt()
                if next_cmd is None:
                    return
                self._state.profiling_command = next_cmd
            else:
                # ── Fresh session: Phase 1 → 1b → 2 ──────────────────────────
                for sp in self._state.source_paths:
                    if not pathlib.Path(sp).exists():
                        _print(f"  Warning: --source path not found: {sp}", style="yellow")

                _app_info = self._classify_app_command(self._state.app_command)
                cmd = (
                    self._phase1b_quick_workload_analysis()
                    or self._build_profiling_command(_app_info)
                )
                self._state.profiling_command = cmd
                if not self._phase2_show_command(cmd):
                    return

            # Phases 3-7 loop
            while True:
                # Phase 3: run profiler
                if not self._phase3_run_profiler(self._state.profiling_command):
                    _print("  Trace collection failed or was aborted.", style="yellow")
                    break

                latest_run = self._state.trace_history[-1]

                # Phase 4: analysis
                snap = self._phase4_analyze(latest_run.db_path)

                # Phase 5: recommendations menu
                result = self._phase5_rec_menu(snap)
                if result is not None:
                    mode, selected_recs = result
                    scoped = _AnalysisSnapshot(
                        timestamp=snap.timestamp,
                        iteration=snap.iteration,
                        recommendations=selected_recs,
                        execution_breakdown=snap.execution_breakdown,
                        hotspots=snap.hotspots,
                        ai_recommended_command=snap.ai_recommended_command,
                    )
                    # Phase 6: apply
                    if mode == "direct":
                        if self._phase6_apply_direct(scoped) == "exit":
                            return
                    elif mode == "diff":
                        self._phase6_apply_diff(scoped)

                # Phase 7: re-profiling?
                next_cmd = self._phase7_reprofiling_prompt()
                if next_cmd is None:
                    break
                self._state.profiling_command = next_cmd

        except KeyboardInterrupt:
            _print()
            _print("  Interrupted.", style="yellow")
        finally:
            self._teardown_checkpoints()
            self._save_session()
            self.print_session_summary()
