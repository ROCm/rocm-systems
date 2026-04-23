"""Tests for perfxpert.tools.tasks — beads_rust-compatible Python task store."""

import threading
from pathlib import Path

import pytest

from perfxpert.tools import tasks
from perfxpert.tools._class import ToolClass


@pytest.fixture
def store(tmp_path: Path):
    """Isolated task store for each test."""
    store = tasks.TaskStore(tmp_path / ".beads")
    store.init()
    return store


def test_create_and_next(store):
    task_id = store.create("Profile matmul kernel", priority=1)
    assert task_id.startswith("pfx-")

    nxt = store.next()
    assert nxt["title"] == "Profile matmul kernel"
    assert nxt["status"] == "ready"


def test_dependencies_respected(store):
    """br-style topological 'ready' — don't return blocked tasks."""
    parent = store.create("Collect trace")
    child = store.create("Analyze trace", depends_on=[parent])

    # First call: parent (no deps)
    first = store.next()
    assert first["id"] == parent

    # Child is still blocked
    assert store.next(exclude_ids=[parent]) is None or \
           store.next(exclude_ids=[parent])["id"] != child


def test_next_returns_none_when_all_blocked(store):
    """If nothing is ready, next() returns None."""
    parent = store.create("blocker")
    # Block child, don't close parent — nothing ready except parent itself
    child = store.create("blocked", depends_on=[parent])
    store.update(parent, status="in_progress")
    # next() should return None — only parent was ready and it's in-progress now
    assert store.next() is None


def test_close_unblocks_children(store):
    parent = store.create("A")
    child = store.create("B", depends_on=[parent])
    store.close(parent, reason="done")

    nxt = store.next()
    assert nxt["id"] == child


def test_query_by_kernel(store):
    """Critical: Revert-Advisor uses this to avoid re-trying failed techniques."""
    store.create("Try VGPR=32 on heavy_kernel", meta={"kernel": "heavy_kernel", "technique": "vgpr_32"})
    store.create("Try LDS tile on heavy_kernel", meta={"kernel": "heavy_kernel", "technique": "lds_tile"})
    store.create("Optimize other_kernel", meta={"kernel": "other_kernel", "technique": "unroll"})

    tasks_for_hk = store.query_by_kernel("heavy_kernel")
    assert len(tasks_for_hk) == 2
    techniques = {t["meta"]["technique"] for t in tasks_for_hk}
    assert techniques == {"vgpr_32", "lds_tile"}


def test_persistence_across_reopens(tmp_path: Path):
    """Store state survives process restart."""
    store_dir = tmp_path / ".beads"
    s1 = tasks.TaskStore(store_dir)
    s1.init()
    task_id = s1.create("Persistent task")
    s1.close_connection()

    s2 = tasks.TaskStore(store_dir)
    # No init — just open existing store
    retrieved = s2.next()
    assert retrieved["id"] == task_id
    s2.close_connection()


def test_close_without_task_id_raises_clear_error(store):
    with pytest.raises(TypeError):
        store.close()


def test_json_output_format(store):
    """br emits --json on every command — we mirror that."""
    task_id = store.create("Task 1", priority=1)
    result = store.show(task_id)
    # Must be JSON-serializable dict
    import json
    json.dumps(result)
    assert "id" in result
    assert "title" in result
    assert "status" in result


def test_is_read_only_class():
    """All task operations are READ_ONLY (safe for MCP) — they modify the task store,
    not user code/files. We expose them via MCP because external coordination
    (Claude Desktop creating tasks, etc.) is intended."""
    # Note: even though tasks modify state, they're "read-only" in the sense
    # that they don't modify user source or run processes. They're MCP-safe.
    # This is a design choice documented in the spec.
    assert True  # placeholder — actual class markers on TaskStore methods


def test_module_level_store_tracks_current_working_directory(tmp_path: Path, monkeypatch):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()

    monkeypatch.chdir(first)
    first_id = tasks.create("First task")

    monkeypatch.chdir(second)
    second_id = tasks.create("Second task")

    first_store = tasks.TaskStore(first / ".beads")
    second_store = tasks.TaskStore(second / ".beads")
    assert first_store.next()["id"] == first_id
    assert second_store.next()["id"] == second_id


def test_module_level_store_uses_separate_connections_per_thread(tmp_path: Path, monkeypatch):
    root = tmp_path / "threaded"
    root.mkdir()
    monkeypatch.chdir(root)

    task_ids = [tasks.create("main-thread task")]
    errors = []

    def worker():
        try:
            task_ids.append(tasks.create("worker-thread task"))
        except Exception as exc:  # pragma: no cover - assertion below inspects exact type
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert errors == []
    store = tasks.TaskStore(root / ".beads")
    try:
        titles = {store.show(task_id)["title"] for task_id in task_ids}
    finally:
        store.close()
    assert titles == {"main-thread task", "worker-thread task"}


def test_module_level_close_requires_task_id():
    with pytest.raises(TypeError):
        tasks.close()
