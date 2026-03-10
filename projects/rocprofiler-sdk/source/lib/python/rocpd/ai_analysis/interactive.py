"""Interactive session for rocpd analyze --interactive."""
from __future__ import annotations

import json
import os
import pathlib
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


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
    _RICH = True
except ImportError:
    _RICH = False

_console = Console() if _RICH else None

_PRI_STYLE = {"HIGH": "bold red", "MEDIUM": "yellow", "LOW": "green", "INFO": "blue"}


def _print(msg: str = "", style: str = "") -> None:
    if _RICH and _console:
        _console.print(msg, style=style or None)
    else:
        print(msg)


def _input(prompt: str) -> str:
    return input(prompt)


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
        self._session = self._init_session(resume_session_id)

    @property
    def session(self) -> SessionData:
        return self._session

    def _init_session(self, resume_id: Optional[str]) -> SessionData:
        # Explicit resume
        if resume_id:
            loaded = self._store.load(resume_id)
            if loaded:
                return loaded

        # Auto-detect
        existing = self._store.find_by_source_dir(self._source_dir)
        if existing:
            chosen = self._prompt_resume(existing)
            if chosen:
                return chosen

        # New session
        now = datetime.now(timezone.utc).isoformat()
        return SessionData(
            session_id=SessionStore.make_session_id(self._source_dir),
            source_dir=self._source_dir,
            created_at=now,
            last_updated=now,
        )

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
        _print("  [n]  Start new session")
        _print()
        choice = _input("  > ").strip().lower()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(existing):
                return existing[idx]
        return None

    def _render_main_menu(self) -> None:
        src_label = pathlib.Path(self._source_dir).name
        sid_label = self._session.session_id
        if _RICH and _console:
            from rich.panel import Panel
            _console.print(Panel(
                f"[bold]Source:[/bold] {src_label}   "
                f"[bold]Session:[/bold] {sid_label}   "
                f"[dim]\\[s] save  \\[q] quit[/dim]",
                title="[bold cyan]rocpd Interactive Analysis[/bold cyan]",
                border_style="blue",
            ))
        else:
            w = 70
            print("=" * w)
            print(f"  rocpd Interactive Analysis | {src_label}")
            print(f"  Session: {sid_label}  [s] save  [q] quit")
            print("=" * w)

        _print()
        _print("  [p]  Get profiling commands", style="white")
        _print("  [o]  Optimize source code  (AI — uploads source summary)",
               style="white")

        if self._session.persistent_menu_items:
            _print()
            _print("  ── From previous analysis " + "─" * 38, style="dim")
            for i, item in enumerate(self._session.persistent_menu_items, 1):
                pri_style = _PRI_STYLE.get(item.priority, "white")
                badge = f"[{item.priority}]"
                src_tag = " [code]" if item.source == "code_change_analysis" else ""
                if _RICH and _console:
                    _console.print(
                        f"  [cyan bold]\\[{i}][/cyan bold]  "
                        f"[cyan]✦[/cyan] {item.title}  "
                        f"[{pri_style}]{badge}[/{pri_style}]{src_tag}"
                    )
                else:
                    print(f"  [{i}]  ✦ {item.title}  {badge}{src_tag}")
        _print()

    def run(self) -> None:
        """Main event loop."""
        while True:
            self._render_main_menu()
            choice = _input("  > ").strip().lower()

            if choice == "q":
                self._save_and_quit()
                break
            elif choice == "s":
                self._store.save(self._session)
                _print("  Session saved.", style="green")
            elif choice == "p":
                self._path_profiling()
            elif choice == "o":
                self._path_optimize()
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(self._session.persistent_menu_items):
                    self._pursue_recommendation(
                        self._session.persistent_menu_items[idx]
                    )
            else:
                _print("  Unknown choice. Try p, o, 1–N, s, or q.", style="dim")

    def _save_and_quit(self) -> None:
        self._session.last_updated = datetime.now(timezone.utc).isoformat()
        self._store.save(self._session)
        _print("  Session saved. Goodbye.", style="cyan")

    # Stubs (implemented in later tasks)
    def _path_profiling(self) -> None:
        _print("  [path p — not yet implemented]", style="dim")

    def _path_optimize(self) -> None:
        _print("  [path o — not yet implemented]", style="dim")

    def _pursue_recommendation(self, item: PersistentMenuItem) -> None:
        _print(f"  [pursue {item.id} — not yet implemented]", style="dim")
