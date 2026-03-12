"""Interactive session for rocpd analyze --interactive."""
from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from .llm_conversation import LLMConversation
from .llm_analyzer import load_reference_guide


# ── Session data ─────────────────────────────────────────────────────────────

@dataclass
class PersistentMenuItem:
    """A recommendation promoted to the main menu from a previous analysis."""
    id: str
    title: str
    priority: str               # "HIGH" | "MEDIUM" | "LOW"
    source: str                 # "profiling_analysis" | "code_change_analysis"
    added_at: str               # ISO-8601
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoryEntry:
    type: str                   # "profiling_run" | "code_change"
    timestamp: str
    db_path: str = ""
    files_modified: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class SessionData:
    session_id: str
    source_dir: str
    created_at: str
    last_updated: str
    history: List[HistoryEntry] = field(default_factory=list)
    persistent_menu_items: List[PersistentMenuItem] = field(default_factory=list)
    conversation: Optional[Dict[str, Any]] = None     # serialized LLMConversation
    sent_source_files: List[str] = field(default_factory=list)  # files already sent to LLM

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
            conversation=d.get("conversation"),      # None if key absent (backward compat)
            sent_source_files=d.get("sent_source_files", []),  # empty list for old sessions
        )


# ── SessionStore ──────────────────────────────────────────────────────────────

_DEFAULT_SESSIONS_DIR = pathlib.Path.home() / ".rocpd" / "sessions"


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
                return datetime.fromisoformat(s.created_at)
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
    from rich.status import Status as _RichStatus
    _RICH = True
except ImportError:
    _RICH = False

_console = Console() if _RICH else None

_PRI_STYLE = {"HIGH": "bold red", "MEDIUM": "yellow", "LOW": "green", "INFO": "blue"}


def _print(msg: str = "", style: str = "") -> None:
    if _RICH and _console:
        # markup=False so literal brackets like [p], [o], [n], [1] are not
        # consumed as Rich markup tags and are shown to the user as-is.
        _console.print(msg, style=style or None, markup=False)
    else:
        print(msg)


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

        RED   = "\033[31m"
        RESET = "\033[0m"
        lines: List[str] = []

        for y_char in range(height_px // 2):
            row = "  "   # leading indent
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
    else:
        # Fallback: plain text header with AMD red
        RED, BOLD, RESET = "\033[31m", "\033[1m", "\033[0m"
        print(f"\n  {BOLD}{RED}AMD ROCm{RESET}  AI Analysis — Interactive Session\n")


# ── InteractiveSession ────────────────────────────────────────────────────────

class InteractiveSession:
    """Top-level interactive menu for rocpd analyze --interactive."""

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
            warnings.warn(f"[LLMConversation] Could not load reference guide: {e}", stacklevel=3)
            fence = ""
        conv.initialize(
            "You are an expert AMD GPU performance engineer "
            "helping optimize a HIP/ROCm application.\n\n" + fence
        )
        return conv

    def _prompt_resume(self, existing: List[SessionData]) -> Optional[SessionData]:
        _print()
        _print(f"Found {len(existing)} previous session(s) for {self._source_dir}:",
               style="cyan")
        for i, s in enumerate(existing, 1):
            n_runs   = sum(1 for h in s.history if h.type == "profiling_run")
            n_change = sum(1 for h in s.history if h.type == "code_change")
            n_items  = len(s.persistent_menu_items)
            _print(f"  [{i}]  {s.session_id}  "
                   f"({n_runs} profiling run(s), {n_change} code change(s), "
                   f"{n_items} saved recommendation(s))")
        _print("  [n]  Start new session  (or press Enter)")
        _print()
        choice = _input("  > ").strip().lower()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(existing):
                return existing[idx]
            _print(f"  Invalid selection — starting new session.", style="dim")
        elif choice not in ("n", ""):
            _print(f"  Unrecognized input — starting new session.", style="dim")
        return None

    def _render_main_menu(self) -> None:
        src_label = pathlib.Path(self._source_dir).name if self._source_dir else "(no source)"
        n_runs = sum(1 for h in self._session.history if h.type == "profiling_run")
        db_label = f"  db: {pathlib.Path(self._db_path).name}" if self._db_path else ""
        runs_label = f"  runs: {n_runs}" if n_runs else ""
        status_line = (f"[dim]{db_label}{runs_label}  \\[s] save  \\[q] quit[/dim]"
                       if _RICH else f"{db_label}{runs_label}  [s] save  [q] quit")
        if _RICH and _console:
            _console.print(Panel(
                f"[bold]Source:[/bold] {src_label}   "
                f"[bold]Session:[/bold] {self._session.session_id}   "
                + status_line,
                title="[bold cyan]rocpd Interactive Analysis[/bold cyan]",
                border_style="blue",
            ))
        else:
            w = 70
            print("=" * w)
            print(f"  rocpd Interactive Analysis | {src_label}")
            print(f"  Session: {self._session.session_id}"
                  f"  {db_label}  [s] save  [q] quit")
            print("=" * w)

        _print()
        _print("  [p]  Profile app  — run rocprofv3, collect .db", style="white")
        _print("  [a]  Analyze .db  — load existing trace and find bottlenecks", style="white")
        _print("  [o]  Optimize     — AI code optimization suggestions", style="white")

        if self._session.persistent_menu_items:
            _print()
            _print("  ── Findings from this session " + "─" * 33, style="dim")
            for i, item in enumerate(self._session.persistent_menu_items, 1):
                pri = item.priority.upper()
                pri_style = _PRI_STYLE.get(pri, "white")
                src_tag = "  [code change]" if item.source == "code_change_analysis" else ""
                if _RICH and _console:
                    _console.print(
                        f"  [cyan bold]\\[{i}][/cyan bold]  "
                        f"[{pri_style}][{pri}][/{pri_style}]  "
                        f"{item.title}{src_tag}"
                    )
                else:
                    print(f"  [{i}]  [{pri}]  {item.title}{src_tag}")
        _print()

    def _path_analyze_db(self) -> None:
        """Prompt for a .db file, run Tier 1/2 analysis, show summary and add recommendations."""
        _print()
        if self._db_path:
            _print(f"  Current .db: {self._db_path}", style="dim")
            _print("  Enter a .db path to analyze, or press Enter to re-analyze current:", style="cyan")
        else:
            _print("  Enter path to a .db trace file:", style="cyan")
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
        self._session.history.append(HistoryEntry(
            type="profiling_run", timestamp=now, db_path=str(db_path)
        ))
        _print(f"  ✓ {added} new finding(s) added to menu.", style="green")

    def _show_analysis_summary(self, recs: List[Dict[str, Any]]) -> None:
        """Print a brief summary of findings after Tier 1/2 analysis."""
        if not recs:
            _print("  No significant bottlenecks found.", style="green")
            return
        high = [r for r in recs if r.get("priority", "").upper() == "HIGH"]
        med  = [r for r in recs if r.get("priority", "").upper() == "MEDIUM"]
        _print()
        _print("  ── Analysis Summary ────────────────────────────────────", style="cyan")
        for r in high:
            _print(f"  [HIGH]    {r.get('issue', r.get('title', ''))}", style="bold red")
        for r in med:
            _print(f"  [MEDIUM]  {r.get('issue', r.get('title', ''))}", style="yellow")
        _print()

    def run(self) -> None:
        """Main event loop."""
        _print_startup_banner()
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
                    self._pursue_recommendation(
                        self._session.persistent_menu_items[idx]
                    )
            else:
                _print("  Unknown choice. Enter p, a, o, s, q, or a number.", style="dim")

    def _save_and_quit(self) -> None:
        self._session.last_updated = datetime.now(timezone.utc).isoformat()
        if self._conv:
            self._session.conversation = self._conv.to_dict()
        self._session.sent_source_files = list(self._sent_source_files)
        self._store.save(self._session)
        _print("  Session saved. Goodbye.", style="cyan")

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
                self._session.persistent_menu_items.append(PersistentMenuItem(
                    id=rid,
                    title=rec.get("issue", rec.get("category", rid)),
                    priority=rec.get("priority", "INFO"),
                    source=source,
                    added_at=now,
                    detail=rec,
                ))
                existing_ids.add(rid)
                added += 1
        return added

    def _path_profiling(self, _source: str = "profiling_analysis") -> None:
        """Show profiling commands; let user pick one to run; auto-ingest output .db."""
        _print()
        _print("  ── Profiling Commands ──────────────────────────────────", style="cyan")
        _print()

        cmds = self._collect_profiling_commands()

        # Optional LLM annotation on tier0 metadata (no source text uploaded)
        if self._llm_provider and self._tier0:
            cmds = self._llm_annotate_profiling_plan(cmds)

        if not cmds:
            _print("  (no profiling commands available)", style="dim")
            return

        for i, (label, cmd) in enumerate(cmds, 1):
            _print(f"  [{i}]  {label}", style="white")
            _print(f"       $ {cmd}", style="dim")
            _print()

        _print("  Enter command number to run it, or Enter to skip:", style="cyan")
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
            _print(f"  Enter application to profile{hint}:", style="cyan")
            _print("  (e.g.  ./my_app --arg1 val1   or press Enter to use default)", style="dim")
            try:
                app_input = _input("  > ").strip()
            except EOFError:
                return
            if app_input:
                selected_cmd = selected_cmd.replace("-- ./app", f"-- {app_input}")
            elif auto_app and auto_app != "./app":
                selected_cmd = auto
            # else leave as-is (./app stays in command; user will see it)

        _print()
        _print(f"  Running: $ {selected_cmd}", style="cyan")
        _print()

        import subprocess
        proc = subprocess.run(selected_cmd, shell=True)
        _print()
        if proc.returncode != 0:
            _print(f"  Command exited with code {proc.returncode}.", style="yellow")

        # Try to find the output .db automatically from the command flags
        db_path = self._find_output_db(selected_cmd)
        if db_path:
            _print(f"  Found output: {db_path}", style="green")
        else:
            _print("  Enter path to the output .db file (or Enter to skip):", style="cyan")
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
        self._session.history.append(HistoryEntry(
            type="profiling_run",
            timestamp=now,
            db_path=str(db_path),
        ))
        self._db_path = str(db_path)
        _print(f"  ✓ {added} finding(s) added to menu.", style="green")

    def _resolve_app_placeholder(self, cmd: str) -> str:
        """Replace '-- ./app' placeholder with an actual binary found near source_dir."""
        if "-- ./app" not in cmd:
            return cmd
        # Look for any executable in source_dir (non-script, non-dot files)
        base = pathlib.Path(self._source_dir)
        for candidate in sorted(base.iterdir()):
            if (candidate.is_file()
                    and os.access(str(candidate), os.X_OK)
                    and not candidate.name.startswith(".")
                    and candidate.suffix not in {".sh", ".py", ".md", ".txt", ".cpp",
                                                  ".hip", ".cu", ".h", ".hpp"}):
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
        for rec in sorted(self._recs,
                          key=lambda r: priority_order.get(r.get("priority", "INFO"), 4)):
            for cmd in rec.get("commands", []):
                fc = cmd.get("full_command", "")
                label = (f"[{rec.get('priority','INFO')}] {rec.get('category','')} — "
                         f"{cmd.get('tool','')}: {cmd.get('description','')}")
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
                "programming_model":  getattr(plan, "programming_model", "HIP"),
                "kernel_count":       getattr(plan, "kernel_count", 0),
                "suggested_counters": getattr(plan, "suggested_counters", []),
                "risk_areas":         getattr(plan, "risk_areas", []),
                "detected_patterns":  [
                    {"id":          (p.get("pattern_id") if isinstance(p, dict) else getattr(p, "pattern_id", "")),
                     "severity":    (p.get("severity")   if isinstance(p, dict) else getattr(p, "severity",   "")),
                     "description": (p.get("description") if isinstance(p, dict) else getattr(p, "description", ""))}
                    for p in patterns
                ],
                "suggested_commands": [cmd for _, cmd in cmds],
            }
            user_msg = (
                f"Annotate this profiling plan (max 200 words, plain text only — no markdown): "
                f"{_json.dumps(metadata)}"
            )
            _print()
            _print("  ── LLM Profiling Advice ────────────────────────────", style="cyan")
            note = self._conv.send(user_msg, on_token=_print_token)
            _print()
        except Exception as exc:
            _print(f"  (LLM annotation skipped: {exc})", style="dim")
        return cmds

    def _run_tier1_analysis(self, db_path: str):
        """Run Tier 1/2 analysis on db_path; return (recs, breakdown) tuple.

        recs     — List[Dict] of recommendations
        breakdown — Dict with kernel_time_pct, memcpy_time_pct, api_overhead_pct,
                    idle_time_pct, total_runtime_ns; None if analysis fails
        """
        try:
            from rocpd.ai_analysis.api import analyze_database
            result = analyze_database(pathlib.Path(db_path))
            recs: List[Dict[str, Any]] = []
            for r in (
                result.recommendations.high_priority
                + result.recommendations.medium_priority
                + result.recommendations.low_priority
            ):
                recs.append({
                    "id": r.id,
                    "priority": r.priority,
                    "category": r.category,
                    "issue": r.title,
                    "suggestion": r.description,
                    "actions": r.next_steps,
                    "commands": [],
                })
            breakdown: Optional[Dict[str, Any]] = None
            eb = result.execution_breakdown
            if eb is not None:
                breakdown = {
                    "kernel_time_pct":  eb.kernel_time_pct,
                    "memcpy_time_pct":  eb.memcpy_time_pct,
                    "api_overhead_pct": eb.api_overhead_pct,
                    "idle_time_pct":    eb.idle_time_pct,
                    "total_runtime_ns": result.profiling_info.total_duration_ns,
                }
            return recs, breakdown
        except Exception as exc:
            _print(f"  (Tier 1 analysis failed: {exc})", style="red")
            return [], None

    def _extract_ai_commands(
        self, text: str, structured_cmds: List[str]
    ) -> List[str]:
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
        _print("  ── AI-suggested profiling commands ───────────────────────",
               style="cyan")
        for i, cmd in enumerate(commands, 1):
            _print(f"  [{i}]  $ {cmd}", style="dim")
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
            _print("  Enter application to profile:", style="cyan")
            try:
                app_input = _input("  > ").strip()
            except EOFError:
                return
            if app_input:
                cmd = cmd.replace("-- ./app", f"-- {app_input}")
            elif "-- ./app" not in auto:
                cmd = auto

        _print()
        _print(f"  Running: $ {cmd}", style="cyan")
        _print()
        proc = subprocess.run(cmd, shell=True)
        _print()
        if proc.returncode != 0:
            _print(f"  Command exited with code {proc.returncode}.", style="yellow")

        db_path = self._find_output_db(cmd)
        if not db_path:
            _print("  Enter path to the output .db file (or Enter to skip):",
                   style="cyan")
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
        self._session.history.append(HistoryEntry(
            type="profiling_run", timestamp=now, db_path=str(db_path)
        ))
        self._session.last_updated = now
        if self._conv:
            self._session.conversation = self._conv.to_dict()
        self._session.sent_source_files = list(self._sent_source_files)
        self._store.save(self._session)
        _print(f"  ✓ {added} finding(s) added to menu.", style="green")

    _TOKEN_BUDGET = 60_000  # characters (approximate token proxy)

    # Subdirectory names that look like backup/archive copies — skip them so
    # we don't send the same source file twice (e.g. original_code/).
    _SKIP_SUBDIRS = frozenset({
        "original_code", "original", "backup", "bak", "old", "archive",
        "reference", "orig", "before",
    })

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
            _print("  No LLM configured. Add --llm anthropic or --llm openai to get "
                   "AI-generated code suggestions. Showing rule-based hints instead:",
                   style="yellow")
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
            _print("  No kernel-containing files detected. "
                   "Run with --source-dir pointing at your source.", style="yellow")
            return

        _print()
        _print(f"  Analyzing {len(hot_files)} file(s):", style="cyan")
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
            _print("  ── Optimization Suggestions ─────────────────────────", style="cyan")
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
            self._session.history.append(HistoryEntry(
                type="code_change",
                timestamp=now,
                files_modified=modified,
                summary=f"Optimized {len(modified)} file(s) via LLM suggestions",
            ))
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
        _print("  Requesting optimization suggestions (based on detected patterns)...",
               style="dim")
        import json as _json

        # Build compact metadata from tier0 — same approach as annotate_profiling_plan
        plan = self._tier0
        patterns = getattr(plan, "detected_patterns", [])
        kernels  = getattr(plan, "detected_kernels", [])[:5]
        metadata = {
            "programming_model": getattr(plan, "programming_model", "HIP"),
            "kernel_count":      getattr(plan, "kernel_count", 0),
            "risk_areas":        getattr(plan, "risk_areas", []),
            "detected_patterns": [
                {
                    "id":          (p.get("pattern_id") if isinstance(p, dict)
                                    else getattr(p, "pattern_id", "")),
                    "severity":    (p.get("severity")   if isinstance(p, dict)
                                    else getattr(p, "severity",   "")),
                    "description": (p.get("description") if isinstance(p, dict)
                                    else getattr(p, "description", "")),
                    "count":       (p.get("count", 1)   if isinstance(p, dict)
                                    else getattr(p, "count", 1)),
                }
                for p in patterns
            ],
            "detected_kernels": [
                {
                    "name":        ("[KERNEL]" if isinstance(k, dict)
                                    else "[KERNEL]"),
                    "launch_type": (k.get("launch_type", "") if isinstance(k, dict)
                                    else getattr(k, "launch_type", "")),
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
        _print("  ── AI Optimization Suggestions ──────────────────────", style="cyan")
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

    def _offer_apply_suggestions(self, suggestions: str, llm_provider: Optional[str] = None) -> None:
        """Ask user whether to save the suggestions or let the LLM edit source code directly."""
        _print("  Apply these suggestions to your source files?", style="cyan")
        _print("    [s] Save suggestions to a file", style="dim")
        _print("    [e] Edit code with AI  (LLM rewrites a source file)", style="dim")
        _print("    [n] No, return to menu (default)", style="dim")
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
                p for p in sorted(pathlib.Path(self._source_dir).rglob("*"))
                if p.suffix in exts and p.is_file()
            ]
        except OSError:
            pass

        if not src_files:
            _print("  (No source files found in source directory)", style="yellow")
            return None

        _print()
        _print("  Choose a file to edit:", style="cyan")
        for i, p in enumerate(src_files[:15]):
            try:
                label = p.relative_to(pathlib.Path(self._source_dir))
            except ValueError:
                label = p.name
            _print(f"    [{i + 1}] {label}", style="dim")
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

    def _apply_suggestions_via_llm(self, suggestions: str, llm_provider: Optional[str]) -> None:
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
                fallback = self._llm_provider if self._llm_provider and self._llm_provider != "local" else None
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
                    _print("    • Or pass --llm anthropic / --llm openai when launching rocpd.", style="dim")
                    return

        if not llm_provider:
            _print("  No LLM configured. Pass --llm local/anthropic/openai to enable AI code edits.",
                   style="yellow")
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
        from rocpd.ai_analysis.llm_analyzer import LLMAnalyzer
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
        diff_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            rewritten.splitlines(keepends=True),
            fromfile=f"{chosen.name} (original)",
            tofile=f"{chosen.name} (AI-edited)",
            n=3,
        ))

        _print()
        _print("  ── Proposed changes ─────────────────────────────────", style="cyan")
        if diff_lines:
            for line in diff_lines[:120]:          # cap at 120 diff lines for readability
                line = line.rstrip("\n")
                if line.startswith("+"):
                    _print(line, style="green")
                elif line.startswith("-"):
                    _print(line, style="red")
                else:
                    _print(line, style="dim")
            if len(diff_lines) > 120:
                _print(f"  ... ({len(diff_lines) - 120} more diff lines omitted)", style="dim")
        else:
            _print("  (No changes — rewritten file is identical to original)", style="yellow")
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
            url = os.environ.get("ROCPD_LLM_LOCAL_URL", "http://localhost:11434")
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
            pri = rec.get("priority", "INFO") if isinstance(rec, dict) else getattr(rec, "priority", "INFO")
            if pri in ("HIGH", "MEDIUM"):
                issue    = rec.get("issue", rec.get("category", "")) if isinstance(rec, dict) else getattr(rec, "issue", "")
                suggest  = rec.get("suggestion", "") if isinstance(rec, dict) else getattr(rec, "suggestion", "")
                actions  = rec.get("actions", []) if isinstance(rec, dict) else getattr(rec, "actions", [])
                _print(f"  [{pri}] {issue}", style="yellow" if pri == "MEDIUM" else "red")
                if suggest:
                    _print(f"    → {suggest}", style="dim")
                for act in actions[:3]:
                    _print(f"      • {act}", style="dim")
                _print()
                shown += 1
        if shown == 0:
            _print("  No HIGH/MEDIUM priority suggestions found.", style="dim")
        _print("  To apply AI-generated code patches: re-run with --llm anthropic or --llm openai.", style="dim")

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
            _print("  ── AI Optimization Suggestions ──────────────────────", style="cyan")
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
        _print(f"  ── Suggestions for {name} ──────────────────────────────", style="cyan")
        _print(suggestion[:2000] + ("…" if len(suggestion) > 2000 else ""))
        _print()
        try:
            ans = _input(f"  Append LLM suggestions as comments to {name}? [y/N/diff]  ").strip().lower()
        except EOFError:
            return None
        if ans == "diff":
            _print("  (Diff view: LLM suggestions are advisory — showing suggestion text)",
                   style="dim")
            _print(suggestion, style="dim")
            try:
                ans = _input(f"  Append LLM suggestions as comments to {name}? [y/N]  ").strip().lower()
            except EOFError:
                return None
        if ans == "y":
            separator = "\n" + "=" * 72 + "\n"
            return (original + separator +
                    "// LLM OPTIMIZATION SUGGESTIONS:\n// " +
                    "\n// ".join(suggestion.splitlines()) + "\n")
        return None

    def _pursue_recommendation(self, item: PersistentMenuItem) -> None:
        """Show full recommendation and sub-menu: [r] run command, [m] back to main menu."""
        _print()
        _print(f"  ── {item.title} [{item.priority}] ──────────────────────────────",
               style="cyan")
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
            _print("  Next steps:", style="cyan")
            for act in actions:
                _print(f"    • {act}", style="dim")

        cmds = [c.get("full_command", "") for c in detail.get("commands", [])
                if c.get("full_command")]
        if cmds:
            _print()
            _print("  Suggested commands:", style="cyan")
            for i, cmd in enumerate(cmds, 1):
                _print(f"    [{i}]  $ {cmd}", style="dim")

        _print()
        if cmds:
            _print("  [r]  Run suggested command")
        else:
            _print("  [r]  Run a profiling command")
        _print("  [m]  Back to main menu")
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
                _print("  Enter application to profile:", style="cyan")
                try:
                    app_input = _input("  > ").strip()
                except EOFError:
                    return
                if app_input:
                    cmd = cmd.replace("-- ./app", f"-- {app_input}")

            _print()
            _print(f"  Running: $ {cmd}", style="cyan")
            _print()
            proc = subprocess.run(cmd, shell=True, check=False)
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
                self._session.history.append(HistoryEntry(
                    type="profiling_run", timestamp=now, db_path=str(db_path)
                ))
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


@dataclass
class _EditRecord:
    """Record of an AI-applied edit."""
    timestamp: str
    file_path: str
    backup_path: str


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


class WorkflowSession:
    """7-phase interactive profiling + optimization workflow.

    Triggered by: rocpd analyze --interactive "<app_command>"
    """

    _DEFAULT_TRACE_DIR = "/tmp/rocpd_trace"

    def __init__(
        self,
        app_command: str,
        source_paths: Optional[List[str]] = None,
        llm_provider: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        trace_dir: Optional[str] = None,
    ) -> None:
        self._state = WorkflowState(
            app_command=app_command,
            source_paths=list(source_paths or []),
        )
        self._llm_provider = llm_provider
        self._llm_api_key = llm_api_key
        self._llm_model = llm_model
        self._trace_dir = trace_dir or self._DEFAULT_TRACE_DIR

    # ── Phase 2: Profiling command generation ─────────────────────────────────

    def _build_profiling_command(self) -> str:
        """Build a rocprofv3 profiling command wrapping the user's app."""
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = f"{self._trace_dir}/run_{run_id}"
        return (
            f"rocprofv3 --sys-trace --kernel-trace --memory-copy-trace --stats "
            f"-d {out_dir} -o results "
            f"-- {self._state.app_command}"
        )

    def _phase2_show_command(self, cmd: str) -> bool:
        """Display boxed profiling command; return True if user approves."""
        _print()
        width = max(66, len(cmd) + 8)
        border = "─" * (width - 2)
        _print(f"╭{border}╮", style="cyan")
        _print(f"│  Profiling Command" + " " * (width - 21) + "│", style="cyan")
        _print(f"│" + " " * (width - 2) + "│", style="cyan")
        indent = "│  "
        tail   = "  │"
        avail  = width - len(indent) - len(tail)
        # Word-wrap command
        words = cmd.split()
        line  = ""
        for word in words:
            if line and len(line) + 1 + len(word) > avail:
                _print(f"{indent}{line:<{avail}}{tail}", style="cyan")
                line = word
            else:
                line = f"{line} {word}".lstrip()
        if line:
            _print(f"{indent}{line:<{avail}}{tail}", style="cyan")
        _print(f"│" + " " * (width - 2) + "│", style="cyan")
        _print(f"╰{border}╯", style="cyan")
        _print()
        try:
            ans = _input("  Would you like the interactive tool to run this command? [Y/n]  ").strip().lower()
        except EOFError:
            return False
        if ans in ("n", "no"):
            _print()
            _print("  Command not run. Copy it to run manually:", style="dim")
            _print(f"  $ {cmd}", style="dim")
            return False
        return True

    # ── Phase 3: Trace collection ──────────────────────────────────────────────

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
        On failure: ask retry / edit command / abort.
        """
        import shlex as _shlex

        while True:
            _print(f"  Running: $ {cmd}", style="cyan")
            _print()
            try:
                proc = subprocess.Popen(
                    _shlex.split(cmd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="", flush=True)
                proc.wait()
            except FileNotFoundError as exc:
                _print(f"  [error] Command not found: {exc}", style="red")

                class _FakeProc:
                    returncode = 127
                proc = _FakeProc()  # type: ignore[assignment]

            _print()
            if proc.returncode == 0:
                trace_files = self._find_trace_files(cmd)
                if trace_files:
                    _print(f"  ✓ Trace collected: {len(trace_files)} file(s)", style="green")
                    for tf in trace_files[:5]:
                        _print(f"    · {tf}", style="dim")
                    db_path = next(
                        (f for f in trace_files if f.endswith(".db")), trace_files[0]
                    )
                    self._state.trace_history.append(_TraceRun(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        command=cmd,
                        db_path=db_path,
                        trace_files=trace_files,
                    ))
                    return True
                # Ran OK but no files found — ask user for path
                _print("  Profiler completed but no trace files found.", style="yellow")
                try:
                    db_input = _input("  Enter path to .db file (or Enter to abort): ").strip()
                except EOFError:
                    return False
                if db_input and pathlib.Path(db_input).exists():
                    self._state.trace_history.append(_TraceRun(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        command=cmd,
                        db_path=db_input,
                        trace_files=[db_input],
                    ))
                    return True
                return False
            else:
                _print(f"  Profiling command failed (exit code {proc.returncode}).", style="red")
                _print("    [r]  Retry same command", style="dim")
                _print("    [e]  Edit the command and retry", style="dim")
                _print("    [a]  Abort", style="dim")
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
                            cmd = new_cmd
                    except EOFError:
                        return False
                    continue
                else:
                    return False

    # ── Phase 4: AI trace analysis ─────────────────────────────────────────────

    def _record_analysis(
        self,
        recs: List[Dict[str, Any]],
        execution_breakdown: Optional[Dict[str, Any]],
        hotspots: List[Dict[str, Any]],
        ai_recommended_command: Optional[str] = None,
    ) -> _AnalysisSnapshot:
        snap = _AnalysisSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            iteration=self._state.iteration_count,
            recommendations=recs,
            execution_breakdown=execution_breakdown,
            hotspots=hotspots,
            ai_recommended_command=ai_recommended_command,
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
        new_s  = nb.get("total_runtime_ns",  0) / 1e9
        if prev_s == 0:
            return
        pct   = (new_s - prev_s) / prev_s * 100
        arrow = "▼" if pct < 0 else "▲"
        _print()
        _print("  ── Performance Comparison ──────────────────────────────", style="cyan")
        _print(f"  {'Metric':<28}  {'Before':>8}  {'After':>8}  Change", style="bold")
        _print(f"  {'Total GPU time':<28}  {prev_s:>7.2f}s  {new_s:>7.2f}s  "
               f"{arrow} {abs(pct):.0f}%",
               style="green" if pct < 0 else "yellow")
        for key, label in [
            ("kernel_time_pct",  "Kernel %"),
            ("memcpy_time_pct",  "MemCopy %"),
            ("api_overhead_pct", "API overhead %"),
        ]:
            pv = pb.get(key, 0)
            nv = nb.get(key, 0)
            diff = nv - pv
            _print(f"  {label:<28}  {pv:>7.1f}%  {nv:>7.1f}%  "
                   f"{'▼' if diff < 0 else '▲'} {abs(diff):.1f}pp",
                   style="green" if diff < 0 else "yellow")
        _print()

    def _phase4_analyze(self, db_path: str) -> _AnalysisSnapshot:
        """Run Tier 1/2 analysis; print structured report; return snapshot."""
        iteration = len(self._state.analysis_history) + 1
        _print()
        if iteration == 1:
            header = "  ══ AI Trace Analysis Report " + "═" * 44
        else:
            header = f"  ══ AI Trace Analysis Report  (Run #{iteration}) " + "═" * 35
        _print(header, style="bold cyan")
        _print()

        recs: List[Dict[str, Any]] = []
        breakdown: Optional[Dict[str, Any]] = None
        hotspots: List[Dict[str, Any]] = []

        try:
            from rocpd.ai_analysis.api import analyze_database  # type: ignore[import]
            result = analyze_database(
                pathlib.Path(db_path),
                enable_llm=bool(self._llm_provider),
                llm_provider=self._llm_provider or None,
                llm_api_key=self._llm_api_key or None,
            )

            eb = result.execution_breakdown
            if eb:
                breakdown = {
                    "kernel_time_pct":  eb.kernel_time_pct,
                    "memcpy_time_pct":  eb.memcpy_time_pct,
                    "api_overhead_pct": eb.api_overhead_pct,
                    "idle_time_pct":    eb.idle_time_pct,
                    "total_runtime_ns": result.profiling_info.total_duration_ns,
                }
                total_s = result.profiling_info.total_duration_ns / 1e9
                _print("  Summary:", style="white")
                _print(f"    Total GPU active time : {total_s:.3f}s", style="dim")
                _print(f"    Kernel  {eb.kernel_time_pct:.1f}%  "
                       f"MemCopy {eb.memcpy_time_pct:.1f}%  "
                       f"Overhead {eb.api_overhead_pct:.1f}%", style="dim")
                _print()

            # Warn when GPU time is zero but profiling ran — likely multiprocessing
            total_ns = result.profiling_info.total_duration_ns if eb else 0
            if total_ns == 0 and self._state.trace_history:
                _print("  ⚠  No GPU kernel activity captured in the main process.",
                       style="yellow")
                _print("     If your app uses Python multiprocessing (e.g. vLLM, PyTorch",
                       style="yellow")
                _print("     DDP), GPU kernels run in spawned worker processes and are",
                       style="yellow")
                _print("     not captured in the main-process DB.", style="yellow")
                _print("     Try:  rocprof-sys --trace -- <app>  (multi-process aware)",
                       style="yellow")
                _print("     or profile a specific worker with  --pid <worker_pid>",
                       style="yellow")
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
                recs.append({
                    "id":               r.id,
                    "priority":         r.priority,
                    "category":         r.category,
                    "issue":            r.title,
                    "suggestion":       r.description,
                    "estimated_impact": r.estimated_impact,
                    "actions":          r.next_steps,
                    "commands":         raw_rec.get("commands", []),
                })

        except Exception as exc:
            _print(f"  (Analysis failed: {exc})", style="red")
            raw_recs = []

        # Source correlation note
        if self._state.source_paths:
            _print(f"  (Source paths provided: "
                   f"{', '.join(pathlib.Path(p).name for p in self._state.source_paths[:3])})",
                   style="dim")
            _print()

        # Print each finding; show recommended commands beneath each issue
        for i, rec in enumerate(recs, 1):
            pri   = rec.get("priority", "INFO")
            style = _PRI_STYLE.get(pri, "white")
            _print(f"  ─── Issue #{i}: {rec.get('issue', '')[:70]} ───", style="cyan")
            _print(f"  Severity   : {pri}", style=style)
            if rec.get("suggestion"):
                _print(f"  Root Cause : {rec['suggestion']}", style="dim")
            if rec.get("estimated_impact"):
                _print(f"  Impact     : {rec['estimated_impact']}", style="dim")
            for act in rec.get("actions", [])[:3]:
                _print(f"    • {act}", style="dim")
            cmds = rec.get("commands", [])
            if cmds:
                _print(f"  Suggested next commands:", style="dim")
                for cmd_obj in cmds[:3]:
                    fc = cmd_obj.get("full_command", "")
                    desc = cmd_obj.get("description", "")
                    if fc:
                        _print(f"    $ {fc}", style="cyan")
                    if desc:
                        _print(f"      ({desc})", style="dim")
            _print()

        if not recs:
            _print("  No significant bottlenecks detected.", style="green")
            _print()

        # Comparison with previous run
        if self._state.analysis_history:
            self._print_comparison(breakdown)

        # Derive AI-recommended re-profiling command from the first rocprofv3
        # command found in any recommendation, replacing the generic placeholder
        # with the actual application being profiled.
        ai_rec_cmd: Optional[str] = None
        app_cmd = self._state.app_command
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        new_out_dir = f"{self._trace_dir}/run_{run_id}"
        for rec in recs:
            for cmd_obj in rec.get("commands", []):
                if cmd_obj.get("tool") == "rocprofv3":
                    fc = cmd_obj.get("full_command", "")
                    if fc and "-- ./app" in fc:
                        # Replace placeholder app and generic output dir
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
                            "", fc,
                        )
                        fc = fc.strip()
                        fc = re.sub(r" {2,}", " ", fc)  # collapse extra spaces
                        ai_rec_cmd = fc
                        break
            if ai_rec_cmd:
                break

        # Don't re-suggest PMC counters that were already collected in the last run.
        # This prevents an infinite [r] → run → same INFO result → [r] loop.
        if ai_rec_cmd and self._state.trace_history:
            def _pmc_counters(cmd: str) -> set:
                return {c for m in re.finditer(
                    r'--pmc\s+((?:[A-Z_][A-Z0-9_]*(?:\s+|$))+)', cmd, re.IGNORECASE
                ) for c in m.group(1).split()}
            suggested = _pmc_counters(ai_rec_cmd)
            already   = _pmc_counters(self._state.trace_history[-1].command)
            if suggested and suggested.issubset(already):
                ai_rec_cmd = None  # all suggested counters already collected

        return self._record_analysis(recs, breakdown, hotspots,
                                     ai_recommended_command=ai_rec_cmd)

    # ── Phase 5: Recommendations menu ─────────────────────────────────────────

    def _phase5_rec_menu(
        self, snap: _AnalysisSnapshot
    ) -> Optional[tuple]:
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
            all_info
            and snap.iteration > 0
            and snap.ai_recommended_command is None
        )

        while True:
            _print()
            _print("  ── Recommendations ─────────────────────────────────────", style="bold cyan")
            for i, rec in enumerate(recs, 1):
                pri   = rec.get("priority", "INFO")
                style = _PRI_STYLE.get(pri, "white")
                issue = rec.get("issue", "")[:70]
                _print(f"  [{i}]  [{pri}]  {issue}", style=style)
            _print()
            if all_info and already_reprofiled:
                # Re-profiling already attempted with the suggested counters but
                # the analysis result is unchanged.  No new suggestions available.
                _print("  Analysis result unchanged after re-profiling.", style="yellow")
                _print("  The profiler may not be capturing GPU kernels from this app.", style="yellow")
                _print("  See the ⚠ note above for multi-process profiling options.", style="yellow")
                _print()
                _print("  [n]  Skip — stop re-profiling", style="dim")
                _print("  [q]  Quit session", style="dim")
            elif all_info:
                # Only profiling-guidance recommendations — no source code to optimize.
                # Direct the user to re-profile with the suggested commands.
                _print("  [r]  Re-profile with suggested commands", style="cyan")
                _print("  [n]  Skip", style="dim")
                _print("  [q]  Quit session", style="dim")
            else:
                _print("  [a]  Address all with AI optimization", style="dim")
                _print("  [n]  Skip — proceed to re-profiling", style="dim")
                _print("  [q]  Quit session", style="dim")
            _print()
            try:
                choice = _input("  Enter choice: ").strip().lower()
            except EOFError:
                return None

            if choice == "q":
                return None
            if choice in ("n", ""):
                return None
            if choice == "r" and all_info and not already_reprofiled:
                # Advance to re-profiling phase; AI-recommended command will be option [3].
                _print()
                _print("  Advancing to re-profiling. Select [3] to use the suggested command.",
                       style="dim")
                return None
            if choice == "a" and not all_info:
                selected = recs
            elif choice.isdigit() and 1 <= int(choice) <= len(recs):
                selected = [recs[int(choice) - 1]]
                r = selected[0]
                # If the selected rec is INFO-level profiling guidance, direct to re-profiling.
                if r.get("priority", "INFO").upper() == "INFO":
                    _print()
                    _print("  This recommendation requires re-profiling with different flags,",
                           style="dim")
                    _print("  not source code changes. Proceeding to re-profiling step.",
                           style="dim")
                    return None
                _print()
                _print(f"  ─── {r.get('issue', '')[:60]} [{r.get('priority', '')}] ───", style="cyan")
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

            _print("  How would you like the optimization applied?", style="cyan")
            _print("    [1]  Edit files directly (AI modifies source files in-place)", style="dim")
            _print("    [2]  Provide a diff/patch file (you review and apply manually)", style="dim")
            _print("    [n]  Back to recommendations menu", style="dim")
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
        exts   = {".hip", ".cpp", ".cu", ".cl", ".h", ".hpp", ".py"}
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
        _print("  Choose a file to edit:", style="cyan")
        for i, f in enumerate(files[:15], 1):
            try:
                label = f.relative_to(self._state.source_paths[0])
            except (ValueError, IndexError):
                label = f.name  # type: ignore[assignment]
            _print(f"    [{i}]  {label}", style="dim")
        try:
            choice = _input("  > ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(files):
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
        try:
            from rocpd.ai_analysis.llm_analyzer import LLMAnalyzer  # type: ignore[import]
            analyzer = LLMAnalyzer(
                provider=self._llm_provider,
                api_key=self._llm_api_key,
                model=self._llm_model,
            )
            system = (
                "You are an expert AMD GPU performance engineer. "
                "Rewrite the source file applying the optimization suggestions. "
                "Return ONLY the complete rewritten file — no explanation, no markdown. "
                "Add a short inline comment on each changed line explaining why."
            )
            user = f"=== SUGGESTIONS ===\n{suggestions}\n\n=== SOURCE FILE ===\n{original}"
            # File rewrites can be large — use a generous timeout (5 min).
            _rewrite_timeout = 300
            with _Spinner(f"  {self._llm_provider} LLM rewriting {file_path.name}..."):
                if self._llm_provider == "openai":
                    try:
                        result = analyzer._call_openai(
                            system, user, max_tokens=16384, timeout=_rewrite_timeout
                        )
                    except Exception as exc:
                        if "too large" in str(exc).lower() or "max_tokens" in str(exc).lower():
                            result = analyzer._call_openai(
                                system, user, timeout=_rewrite_timeout
                            )
                        else:
                            raise
                elif self._llm_provider == "anthropic":
                    result = analyzer._call_anthropic(
                        system, user, timeout=_rewrite_timeout
                    )
                elif self._llm_provider == "private":
                    result = analyzer._call_private(system, user)
                else:
                    result = analyzer._call_local(system, user)
            return result if result and result.strip() else None
        except Exception as exc:
            _print(f"  (LLM rewrite failed: {exc})", style="red")
            return None

    def _phase6_apply_direct(self, snap: _AnalysisSnapshot) -> None:
        """Phase 6: AI edits source files in-place (.bak backup); waits for recompile."""
        suggestions = "\n\n".join(
            f"[{r.get('priority','')}] {r.get('issue','')}:\n"
            f"{r.get('suggestion','')}\n"
            + "\n".join(f"  • {a}" for a in r.get("actions", []))
            for r in snap.recommendations
        )
        chosen = self._pick_file_from_source_paths()
        if chosen is None:
            return
        rewritten = self._llm_rewrite_file(chosen, suggestions)
        while rewritten is None:
            try:
                ans = _input("  Retry LLM rewrite? [y/N]  ").strip().lower()
            except EOFError:
                return
            if ans != "y":
                return
            rewritten = self._llm_rewrite_file(chosen, suggestions)

        import difflib
        original  = chosen.read_text()
        diff_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            rewritten.splitlines(keepends=True),
            fromfile=f"{chosen.name} (original)",
            tofile=f"{chosen.name} (AI-edited)",
            n=3,
        ))
        _print()
        _print("  ── Proposed changes ─────────────────────────────────", style="cyan")
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
            _print("  (No changes — rewritten file is identical to original)", style="yellow")
            return
        _print()
        try:
            confirm = _input("  Apply these changes? [y/N]  ").strip().lower()
        except EOFError:
            return
        if confirm != "y":
            _print("  Changes discarded.", style="dim")
            return

        bak = chosen.with_suffix(chosen.suffix + ".bak")
        try:
            bak.write_text(original)
            chosen.write_text(rewritten)
            _print(f"  Backup : {bak}", style="dim")
            _print(f"  Updated: {chosen}", style="green")
            self._state.edit_history.append(_EditRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                file_path=str(chosen),
                backup_path=str(bak),
            ))
        except OSError as exc:
            _print(f"  (Write failed: {exc})", style="red")
            return

        # Wait for recompile
        _print()
        _print("  Changes applied. Please recompile your application.", style="cyan")
        _print("  Type 'done' when compiled, 'abort' to exit, or describe errors.", style="dim")
        while True:
            try:
                resp = _input("  > ").strip().lower()
            except EOFError:
                break
            if resp in ("done", "compiled", "ok", "yes", "y", ""):
                _print("  Great — ready to re-profile.", style="green")
                break
            if resp in ("abort", "cancel", "quit", "exit"):
                _print("  Aborting. Backup preserved at: " + str(bak), style="dim")
                break
            # Treat as compilation error description
            _print(f"  Compilation error noted. Common causes: missing include, "
                   f"incorrect __launch_bounds__ syntax.", style="yellow")
            _print("  After fixing, type 'done'.", style="dim")

    def _phase6_apply_diff(self, snap: _AnalysisSnapshot) -> None:
        """Phase 6 alt: Save suggestions to a patch file."""
        suggestions = "\n\n".join(
            f"[{r.get('priority','')}] {r.get('issue','')}:\n"
            f"  Suggestion: {r.get('suggestion','')}\n"
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
        current = self._state.profiling_command
        ai_cmd: Optional[str] = None
        if self._state.analysis_history:
            ai_cmd = self._state.analysis_history[-1].ai_recommended_command

        _print()
        _print("  Ready to re-profile. Which command would you like to run?", style="cyan")
        _print(f"    [1]  Same command as before:", style="dim")
        _print(f"         {current}", style="dim")
        _print("    [2]  Let me edit the command first", style="dim")
        if ai_cmd:
            _print("    [3]  Use AI-recommended command:", style="dim")
            _print(f"         {ai_cmd}", style="dim")
        _print("    [n]  Stop — I'm done profiling", style="dim")
        _print()
        try:
            choice = _input("  > ").strip().lower()
        except EOFError:
            return None
        if choice in ("1", ""):
            return current
        elif choice == "2":
            try:
                new_cmd = _input(f"  Edit command (Enter to keep):\n  {current}\n  > ").strip()
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
        _print("  ══════════════════════════════════════════", style="bold cyan")
        _print("   Session Summary", style="bold cyan")
        _print("  ══════════════════════════════════════════", style="bold cyan")
        _print(f"  Iterations : {self._state.iteration_count}", style="white")

        if len(self._state.analysis_history) >= 2:
            first_bd = self._state.analysis_history[0].execution_breakdown or {}
            last_bd  = self._state.analysis_history[-1].execution_breakdown or {}
            t0 = first_bd.get("total_runtime_ns", 0) / 1e9
            t1 = last_bd.get("total_runtime_ns",  0) / 1e9
            if t0 > 0:
                pct   = (t1 - t0) / t0 * 100
                arrow = "▼" if pct < 0 else "▲"
                _print(f"  GPU time   : {t0:.2f}s → {t1:.2f}s  "
                       f"({arrow} {abs(pct):.0f}%)", style="white")

        if self._state.edit_history:
            files = [pathlib.Path(e.file_path).name for e in self._state.edit_history]
            baks  = [pathlib.Path(e.backup_path).name for e in self._state.edit_history]
            _print(f"  Modified   : {', '.join(files)}", style="white")
            _print(f"  Backups    : {', '.join(baks)}", style="dim")

        if self._state.trace_history:
            runs = [pathlib.Path(t.db_path).parent.name for t in self._state.trace_history]
            _print(f"  Trace runs : {', '.join(runs)}", style="dim")

        _print("  ══════════════════════════════════════════", style="bold cyan")
        _print()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the 7-phase workflow loop."""
        _print_startup_banner()

        # Phase 1: validate source paths
        for sp in self._state.source_paths:
            if not pathlib.Path(sp).exists():
                _print(f"  Warning: --source path not found: {sp}", style="yellow")

        # Phase 2: generate + confirm profiling command
        cmd = self._build_profiling_command()
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
                    self._phase6_apply_direct(scoped)
                elif mode == "diff":
                    self._phase6_apply_diff(scoped)

            # Phase 7: re-profiling?
            next_cmd = self._phase7_reprofiling_prompt()
            if next_cmd is None:
                break
            self._state.profiling_command = next_cmd

        self.print_session_summary()
