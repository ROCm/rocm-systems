"""Interactive session for rocpd analyze --interactive."""
from __future__ import annotations

import json
import os
import pathlib
import warnings
from dataclasses import dataclass, field
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
        from dataclasses import asdict
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
