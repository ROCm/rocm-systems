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

    def test_find_by_source_dir_skips_malformed_json(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        # Write a valid session
        good = SessionData(session_id="good", source_dir="/tmp/myapp",
                           created_at="2026-03-10T10:00:00Z",
                           last_updated="2026-03-10T10:00:00Z")
        store.save(good)
        # Write a malformed JSON file
        (tmp_path / "bad.json").write_text("not valid json")
        # Should still return the valid session
        results = store.find_by_source_dir("/tmp/myapp")
        assert len(results) == 1
        assert results[0].session_id == "good"

    def test_make_session_id_contains_slug(self):
        sid = SessionStore.make_session_id("/home/user/my_project")
        assert "my_project" in sid

    def test_make_session_id_replaces_spaces(self):
        sid = SessionStore.make_session_id("/home/user/my project")
        assert " " not in sid

    def test_make_session_id_empty_name_uses_fallback(self):
        # A path whose last component is empty shouldn't crash
        sid = SessionStore.make_session_id("/")
        assert "session" in sid or len(sid) > 10  # just doesn't crash

    def test_find_by_source_dir_newest_first(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        older = SessionData(session_id="older", source_dir="/tmp/myapp",
                            created_at="2026-03-09T10:00:00Z",
                            last_updated="2026-03-09T10:00:00Z")
        newer = SessionData(session_id="newer", source_dir="/tmp/myapp",
                            created_at="2026-03-10T10:00:00Z",
                            last_updated="2026-03-10T10:00:00Z")
        store.save(older)
        store.save(newer)
        results = store.find_by_source_dir("/tmp/myapp")
        assert results[0].session_id == "newer"
