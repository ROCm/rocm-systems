import json
import pathlib
import pytest
from unittest.mock import patch

from rocpd.ai_analysis.interactive import SessionStore, SessionData, PersistentMenuItem


class TestSessionStore:
    def test_save_and_load_roundtrip(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        data = SessionData(
            session_id="2026-03-10_14-23-01_myapp",
            source_dir="/tmp/myapp",
            created_at="2026-03-10T14:23:01Z",
            last_updated="2026-03-10T14:23:01Z",
        )
        store.save(data)
        loaded = store.load("2026-03-10_14-23-01_myapp")
        assert loaded.session_id == data.session_id
        assert loaded.source_dir == data.source_dir

    def test_load_nonexistent_returns_none(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        assert store.load("nonexistent") is None

    def test_find_by_source_dir(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        a = SessionData(session_id="2026-03-10_10-00-00_myapp",
                        source_dir="/tmp/myapp",
                        created_at="2026-03-10T10:00:00Z",
                        last_updated="2026-03-10T10:00:00Z")
        b = SessionData(session_id="2026-03-10_11-00-00_other",
                        source_dir="/tmp/other",
                        created_at="2026-03-10T11:00:00Z",
                        last_updated="2026-03-10T11:00:00Z")
        store.save(a)
        store.save(b)
        results = store.find_by_source_dir("/tmp/myapp")
        assert len(results) == 1
        assert results[0].session_id == a.session_id

    def test_save_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "deep" / "sessions"
        store = SessionStore(sessions_dir=nested)
        data = SessionData(session_id="s1", source_dir="/x",
                           created_at="t", last_updated="t")
        store.save(data)
        assert (nested / "s1.json").exists()

    def test_load_by_file_path(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        data = SessionData(session_id="s2", source_dir="/y",
                           created_at="t", last_updated="t")
        store.save(data)
        path = str(tmp_path / "s2.json")
        loaded = store.load(path)
        assert loaded.session_id == "s2"
