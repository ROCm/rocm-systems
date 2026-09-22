"""tasks — beads_rust-compatible Python task store.

Implements the beads schema (SQLite + JSONL under .beads/) in pure Python
so no external binary is needed. Users who have `br` installed separately
can read/write the same .beads/ dir — format is compatible.

Operations: create, next (topological ready-queue), update, close,
query_by_kernel (for Revert-Advisor avoidance).

Tool class: side-effecting on the local ``.perfxpert/`` task store.
Does NOT modify user source or GPU traces. The MCP exposure registry omits
these module-level wrappers to maintain the §5.8 read-only invariant for
external MCP clients.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    type TEXT DEFAULT 'task',
    priority INTEGER DEFAULT 2,
    status TEXT DEFAULT 'ready',
    assignee TEXT,
    description TEXT,
    meta_json TEXT,
    created_at TEXT NOT NULL,
    closed_at TEXT,
    close_reason TEXT
);
CREATE TABLE IF NOT EXISTS dependencies (
    task_id TEXT NOT NULL REFERENCES tasks(id),
    depends_on_id TEXT NOT NULL REFERENCES tasks(id),
    PRIMARY KEY (task_id, depends_on_id)
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
"""


class TaskStore:
    """File-backed task store with beads-compatible semantics.

    Usage:
        from perfxpert.tools.tasks import TaskStore
        store = TaskStore(".beads")
        store.init()
        task_id = store.create("Profile matmul kernel", priority=1)
        task = store.next()
        store.update(task_id, status="in_progress")
        store.close(task_id, reason="done")
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.db_path = self.root / "tasks.db"
        self._conn: Optional[sqlite3.Connection] = None

    def _get(self) -> sqlite3.Connection:
        if self._conn is None:
            self.root.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA_SQL)
        return self._conn

    def init(self) -> None:
        """Explicitly initialize the store (equivalent to `br init`)."""
        self._get()  # triggers schema creation

    def close(self, task_id: str = None, reason: str = "done") -> None:
        """Close a task (status='closed'), or close the SQLite connection if no task_id.

        Overloaded: tasks.close(task_id, reason) OR store.close() (close connection).
        """
        if task_id is None:
            # Connection close
            if self._conn:
                self._conn.close()
                self._conn = None
            return

        conn = self._get()
        conn.execute(
            "UPDATE tasks SET status='closed', closed_at=?, close_reason=? WHERE id=?",
            (_now_iso(), reason, task_id),
        )
        conn.commit()

    def create(
        self,
        title: str,
        priority: int = 2,
        depends_on: List[str] = None,
        meta: Dict[str, Any] = None,
        description: str = None,
    ) -> str:
        task_id = f"pfx-{uuid.uuid4().hex[:8]}"
        conn = self._get()
        conn.execute(
            """INSERT INTO tasks (id, title, priority, status, description, meta_json, created_at)
               VALUES (?, ?, ?, 'ready', ?, ?, ?)""",
            (task_id, title, priority, description, json.dumps(meta or {}), _now_iso()),
        )
        if depends_on:
            for dep in depends_on:
                conn.execute(
                    "INSERT INTO dependencies (task_id, depends_on_id) VALUES (?, ?)",
                    (task_id, dep),
                )
        conn.commit()
        return task_id

    def next(self, exclude_ids: List[str] = None) -> Optional[Dict[str, Any]]:
        """Return the highest-priority ready task whose deps are all closed."""
        exclude_ids = set(exclude_ids or [])
        conn = self._get()
        rows = conn.execute("""SELECT t.id, t.title, t.priority, t.status, t.assignee, t.description,
                      t.meta_json, t.created_at
               FROM tasks t
               WHERE t.status = 'ready'
               ORDER BY t.priority ASC, t.created_at ASC""").fetchall()

        for row in rows:
            if row["id"] in exclude_ids:
                continue
            # Check all deps closed
            deps = conn.execute(
                """SELECT d.depends_on_id, t2.status
                   FROM dependencies d JOIN tasks t2 ON t2.id = d.depends_on_id
                   WHERE d.task_id = ?""",
                (row["id"],),
            ).fetchall()
            if all(d["status"] == "closed" for d in deps):
                return _row_to_dict(row)
        return None

    def update(
        self,
        task_id: str,
        status: str = None,
        assignee: str = None,
        meta: Dict[str, Any] = None,
    ) -> None:
        conn = self._get()
        updates = []
        values = []
        if status is not None:
            updates.append("status = ?")
            values.append(status)
        if assignee is not None:
            updates.append("assignee = ?")
            values.append(assignee)
        if meta is not None:
            # Merge, don't replace
            current_row = conn.execute("SELECT meta_json FROM tasks WHERE id = ?", (task_id,)).fetchone()
            current = json.loads(current_row["meta_json"] or "{}") if current_row else {}
            current.update(meta)
            updates.append("meta_json = ?")
            values.append(json.dumps(current))

        if updates:
            values.append(task_id)
            conn.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?", values)
            conn.commit()

    def query_by_kernel(self, kernel_name: str) -> List[Dict[str, Any]]:
        """Return all tasks whose meta.kernel matches (including closed).

        Used by Revert-Advisor to avoid proposing already-tried techniques.
        """
        conn = self._get()
        rows = conn.execute(
            "SELECT id, title, priority, status, assignee, description, meta_json, created_at FROM tasks"
        ).fetchall()
        result = []
        for row in rows:
            meta = json.loads(row["meta_json"] or "{}")
            if meta.get("kernel") == kernel_name:
                result.append(_row_to_dict(row))
        return result

    def show(self, task_id: str) -> Dict[str, Any]:
        """Fetch a single task by id."""
        conn = self._get()
        row = conn.execute(
            """SELECT id, title, priority, status, assignee, description,
                      meta_json, created_at FROM tasks WHERE id = ?""",
            (task_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Task not found: {task_id}")
        return _row_to_dict(row)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "priority": row["priority"],
        "status": row["status"],
        "assignee": row["assignee"],
        "description": row["description"],
        "meta": json.loads(row["meta_json"] or "{}"),
        "created_at": row["created_at"],
    }


# -- Module-level convenience wrappers ---------------------------------------
# Agents (see perfxpert/agents/correctness.py) call `tasks.query_by_kernel(name)`
# as if it were a free function. Provide wrappers that instantiate a default
# TaskStore so callers don't thread a store through the agent API.

_DEFAULT_STORE: Optional[TaskStore] = None


def _default_store() -> TaskStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        import os

        task_root_env = os.environ.get("PERFXPERT_TASK_ROOT")
        root = Path(task_root_env) if task_root_env else Path.home() / ".perfxpert"
        _DEFAULT_STORE = TaskStore(root)
        _DEFAULT_STORE.init()
        logger.info(
            "PerfXpert task store initialized at: %s " "(set PERFXPERT_TASK_ROOT to override)",
            root,
        )
    return _DEFAULT_STORE


def _store_for_root(root: str) -> TaskStore:
    store = TaskStore(Path(root))
    store.init()
    return store


def _store_root(root: Optional[str] = None) -> str:
    if root is not None:
        return str((Path(root) / ".beads").resolve())
    return str((Path.cwd() / ".beads").resolve())


def _with_store(method: str, *args, root: Optional[str] = None, **kwargs):
    store = _store_for_root(_store_root(root))
    try:
        return getattr(store, method)(*args, **kwargs)
    finally:
        store.close()


def create_at(root: str, *args, **kwargs) -> str:
    return _with_store("create", *args, root=root, **kwargs)


def next_at(root: str, exclude_ids: List[str] = None) -> Optional[Dict[str, Any]]:
    return _with_store("next", exclude_ids=exclude_ids, root=root)


def update_at(
    root: str,
    task_id: str,
    status: str = None,
    assignee: str = None,
    meta: Dict[str, Any] = None,
) -> None:
    _with_store("update", task_id, status=status, assignee=assignee, meta=meta, root=root)


def close_at(root: str, task_id: str, reason: str = "done") -> None:
    _with_store("close", task_id=task_id, reason=reason, root=root)


def query_by_kernel_at(root: str, kernel_name: str) -> List[Dict[str, Any]]:
    return _with_store("query_by_kernel", kernel_name, root=root)


def show_at(root: str, task_id: str) -> Dict[str, Any]:
    return _with_store("show", task_id, root=root)


def query_by_kernel(kernel_name: str) -> List[Dict[str, Any]]:
    """Return task rows whose metadata references `kernel_name`."""
    return _default_store().query_by_kernel(kernel_name)


def create(title: str, **kwargs: Any) -> str:
    return _default_store().create(title, **kwargs)


def next_task(exclude_ids: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    return _default_store().next(exclude_ids=exclude_ids)


next = next_task


def update(task_id: str, **kwargs: Any) -> None:
    _default_store().update(task_id, **kwargs)


def close(task_id: str = None, reason: str = "done") -> None:
    _default_store().close(task_id=task_id, reason=reason)


def show(task_id: str) -> Dict[str, Any]:
    return _default_store().show(task_id)
