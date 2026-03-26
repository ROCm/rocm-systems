import pathlib
from unittest.mock import patch, MagicMock

from rocinsight.ai_analysis.interactive import (
    SessionStore,
    SessionData,
    PersistentMenuItem,
    InteractiveSession,
)


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
        a = SessionData(
            session_id="2026-03-10_10-00-00_myapp",
            source_dir="/tmp/myapp",
            created_at="2026-03-10T10:00:00Z",
            last_updated="2026-03-10T10:00:00Z",
        )
        b = SessionData(
            session_id="2026-03-10_11-00-00_other",
            source_dir="/tmp/other",
            created_at="2026-03-10T11:00:00Z",
            last_updated="2026-03-10T11:00:00Z",
        )
        store.save(a)
        store.save(b)
        results = store.find_by_source_dir("/tmp/myapp")
        assert len(results) == 1
        assert results[0].session_id == a.session_id

    def test_save_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "deep" / "sessions"
        store = SessionStore(sessions_dir=nested)
        data = SessionData(
            session_id="s1", source_dir="/x", created_at="t", last_updated="t"
        )
        store.save(data)
        assert (nested / "s1.json").exists()

    def test_load_by_file_path(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        data = SessionData(
            session_id="s2", source_dir="/y", created_at="t", last_updated="t"
        )
        store.save(data)
        path = str(tmp_path / "s2.json")
        loaded = store.load(path)
        assert loaded.session_id == "s2"

    def test_find_by_source_dir_skips_malformed_json(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        # Write a valid session
        good = SessionData(
            session_id="good",
            source_dir="/tmp/myapp",
            created_at="2026-03-10T10:00:00Z",
            last_updated="2026-03-10T10:00:00Z",
        )
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
        older = SessionData(
            session_id="older",
            source_dir="/tmp/myapp",
            created_at="2026-03-09T10:00:00Z",
            last_updated="2026-03-09T10:00:00Z",
        )
        newer = SessionData(
            session_id="newer",
            source_dir="/tmp/myapp",
            created_at="2026-03-10T10:00:00Z",
            last_updated="2026-03-10T10:00:00Z",
        )
        store.save(older)
        store.save(newer)
        results = store.find_by_source_dir("/tmp/myapp")
        assert results[0].session_id == "newer"


class TestInteractiveSessionMenu:
    def test_new_session_created_when_none_exist(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        with patch("rocinsight.ai_analysis.interactive._input", return_value=""):
            s = InteractiveSession(
                source_dir="/tmp/myapp",
                tier0_result=None,
                recommendations=[],
                database_path="",
                llm_provider=None,
                llm_api_key=None,
                llm_model=None,
                session_store=store,
                resume_session_id=None,
            )
        assert s.session.source_dir == "/tmp/myapp"
        assert s.session.session_id != ""

    def test_quit_saves_session(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        with patch("rocinsight.ai_analysis.interactive._input", side_effect=["q"]):
            s = InteractiveSession(
                source_dir="/tmp/myapp",
                tier0_result=None,
                recommendations=[],
                database_path="",
                llm_provider=None,
                llm_api_key=None,
                llm_model=None,
                session_store=store,
                resume_session_id=None,
            )
            s.run()
        assert len(store.find_by_source_dir("/tmp/myapp")) == 1

    def test_resume_loads_persistent_items(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        existing = SessionData(
            session_id="old-session",
            source_dir="/tmp/myapp",
            created_at="2026-03-09T10:00:00Z",
            last_updated="2026-03-09T10:00:00Z",
            persistent_menu_items=[
                PersistentMenuItem(
                    id="ROCPD-OCC-001",
                    title="Increase occupancy",
                    priority="HIGH",
                    source="profiling_analysis",
                    added_at="2026-03-09T10:30:00Z",
                )
            ],
        )
        store.save(existing)
        with patch("rocinsight.ai_analysis.interactive._input", return_value=""):
            s = InteractiveSession(
                source_dir="/tmp/myapp",
                tier0_result=None,
                recommendations=[],
                database_path="",
                llm_provider=None,
                llm_api_key=None,
                llm_model=None,
                session_store=store,
                resume_session_id="old-session",
            )
        assert len(s.session.persistent_menu_items) == 1
        assert s.session.persistent_menu_items[0].title == "Increase occupancy"

    def test_run_save_without_quit(self, tmp_path):
        """[s] saves session without exiting; [q] then exits."""
        store = SessionStore(sessions_dir=tmp_path)
        with patch("rocinsight.ai_analysis.interactive._input", side_effect=["s", "q"]):
            s = InteractiveSession(
                source_dir="/tmp/myapp",
                tier0_result=None,
                recommendations=[],
                database_path="",
                llm_provider=None,
                llm_api_key=None,
                llm_model=None,
                session_store=store,
                resume_session_id=None,
            )
            s.run()
        # Session should exist (saved by either [s] or [q])
        assert len(store.find_by_source_dir("/tmp/myapp")) == 1

    def test_run_eof_saves_and_exits(self, tmp_path):
        """EOFError on input triggers save-and-quit gracefully."""
        store = SessionStore(sessions_dir=tmp_path)
        with patch("rocinsight.ai_analysis.interactive._input", side_effect=EOFError()):
            s = InteractiveSession(
                source_dir="/tmp/myapp",
                tier0_result=None,
                recommendations=[],
                database_path="",
                llm_provider=None,
                llm_api_key=None,
                llm_model=None,
                session_store=store,
                resume_session_id=None,
            )
            s.run()  # must not raise
        assert len(store.find_by_source_dir("/tmp/myapp")) == 1

    def test_run_numeric_pursues_recommendation(self, tmp_path):
        """Entering a number calls _pursue_recommendation for that item."""
        store = SessionStore(sessions_dir=tmp_path)
        item = PersistentMenuItem(
            id="ROCPD-OCC-001",
            title="Increase occupancy",
            priority="HIGH",
            source="profiling_analysis",
            added_at="2026-03-10T10:00:00Z",
        )
        with patch("rocinsight.ai_analysis.interactive._input", side_effect=["1", "q"]):
            s = InteractiveSession(
                source_dir="/tmp/myapp",
                tier0_result=None,
                recommendations=[],
                database_path="",
                llm_provider=None,
                llm_api_key=None,
                llm_model=None,
                session_store=store,
                resume_session_id=None,
            )
            s.session.persistent_menu_items.append(item)
            pursued = []
            s._pursue_recommendation = lambda i: pursued.append(i.id)
            s.run()
        assert pursued == ["ROCPD-OCC-001"]

    def test_prompt_resume_invalid_choice_starts_new(self, tmp_path):
        """Out-of-range choice in resume prompt falls through to new session."""
        store = SessionStore(sessions_dir=tmp_path)
        existing = SessionData(
            session_id="old",
            source_dir="/tmp/myapp",
            created_at="2026-03-09T10:00:00Z",
            last_updated="2026-03-09T10:00:00Z",
        )
        store.save(existing)
        # "99" is out of range; should fall back to new session
        with patch("rocinsight.ai_analysis.interactive._input", return_value="99"):
            s = InteractiveSession(
                source_dir="/tmp/myapp",
                tier0_result=None,
                recommendations=[],
                database_path="",
                llm_provider=None,
                llm_api_key=None,
                llm_model=None,
                session_store=store,
                resume_session_id=None,
            )
        assert s.session.session_id != "old"


class TestPathProfiling:
    def _tier0(self):
        t = MagicMock()
        t.suggested_first_command = "rocprofv3 --sys-trace -- ./app"
        t.profiling_plan = MagicMock()
        t.profiling_plan.detected_kernels = []
        t.profiling_plan.detected_patterns = []
        t.profiling_plan.suggested_counters = ["SQ_WAVES", "GRBM_GUI_ACTIVE"]
        t.profiling_plan.risk_areas = ["sync_heavy"]
        t.profiling_plan.programming_model = "HIP"
        t.profiling_plan.kernel_count = 3
        return t

    def test_path_p_adds_history_entry_on_db_provided(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        fake_db = tmp_path / "trace.db"
        fake_db.touch()

        recs_from_analysis = [
            {
                "id": "ROCPD-OCC-001",
                "priority": "HIGH",
                "category": "OCCUPANCY",
                "issue": "Low waves",
                "suggestion": "Increase waves",
                "commands": [],
                "actions": [],
            }
        ]

        s = InteractiveSession(
            source_dir="/tmp/myapp",
            tier0_result=self._tier0(),
            recommendations=[],
            database_path="",
            llm_provider=None,
            llm_api_key=None,
            llm_model=None,
            session_store=store,
            resume_session_id=None,
        )

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        # _path_profiling now prompts: (1) command number, (2) app placeholder, (3) db path
        input_seq = ["1", "", str(fake_db)]
        with patch("rocinsight.ai_analysis.interactive._input", side_effect=input_seq):
            with patch("subprocess.run", return_value=mock_proc):
                with patch.object(
                    s,
                    "_resolve_app_placeholder",
                    return_value="rocprofv3 --sys-trace -- ./app",
                ):
                    with patch.object(
                        s,
                        "_run_tier1_analysis",
                        return_value=(recs_from_analysis, None),
                    ):
                        s._path_profiling()

        assert any(h.type == "profiling_run" for h in s.session.history)
        assert len(s.session.persistent_menu_items) == 1
        assert s.session.persistent_menu_items[0].source == "profiling_analysis"

    def test_path_p_skips_analysis_when_no_db(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        s = InteractiveSession(
            source_dir="/tmp/myapp",
            tier0_result=self._tier0(),
            recommendations=[],
            database_path="",
            llm_provider=None,
            llm_api_key=None,
            llm_model=None,
            session_store=store,
            resume_session_id=None,
        )
        with patch("rocinsight.ai_analysis.interactive._input", return_value=""):
            s._path_profiling()
        assert len(s.session.persistent_menu_items) == 0


class TestPathOptimize:
    def _session_with_tier0(self, tmp_path, files):
        """files: dict of {rel_path: content}"""
        for name, content in files.items():
            p = tmp_path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

        t = MagicMock()
        t.profiling_plan = MagicMock()
        kernels = []
        for name in files:
            k = MagicMock()
            k.file = name
            kernels.append(k)
        t.profiling_plan.detected_kernels = kernels
        t.suggested_first_command = ""
        return t

    def test_hot_files_selected_from_detected_kernels(self, tmp_path):
        files = {"kernel_a.hip": "// kernel A", "other.cpp": "// not a kernel"}
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        t = self._session_with_tier0(tmp_path, files)

        s = InteractiveSession(
            source_dir=str(tmp_path),
            tier0_result=t,
            recommendations=[],
            database_path="",
            llm_provider=None,
            llm_api_key=None,
            llm_model=None,
            session_store=store,
            resume_session_id=None,
        )
        hot = s._select_hot_files()
        names = [pathlib.Path(f).name for f, _ in hot]
        assert "kernel_a.hip" in names

    def test_token_budget_caps_files(self, tmp_path):
        files = {f"k{i}.hip": "x" * 25_000 for i in range(3)}
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        t = self._session_with_tier0(tmp_path, files)
        s = InteractiveSession(
            source_dir=str(tmp_path),
            tier0_result=t,
            recommendations=[],
            database_path="",
            llm_provider=None,
            llm_api_key=None,
            llm_model=None,
            session_store=store,
            resume_session_id=None,
        )
        hot = s._select_hot_files(budget=60_000)
        total = sum(len(c) for _, c in hot)
        assert total <= 60_000


class TestPursueRecommendation:
    def test_pursue_back_to_menu_keeps_item(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        item = PersistentMenuItem(
            id="ROCPD-OCC-001",
            title="Increase occupancy",
            priority="HIGH",
            source="profiling_analysis",
            added_at="2026-03-10T10:00:00Z",
            detail={
                "commands": [
                    {
                        "full_command": "rocprofv3 --pmc SQ_WAVES -- ./app",
                        "tool": "rocprofv3",
                        "description": "collect waves",
                    }
                ]
            },
        )
        s = InteractiveSession(
            source_dir="/tmp/myapp",
            tier0_result=None,
            recommendations=[],
            database_path="",
            llm_provider=None,
            llm_api_key=None,
            llm_model=None,
            session_store=store,
            resume_session_id=None,
        )
        s.session.persistent_menu_items.append(item)
        with patch("rocinsight.ai_analysis.interactive._input", return_value="m"):
            s._pursue_recommendation(item)
        # Item must still be in the list (not consumed)
        assert len(s.session.persistent_menu_items) == 1
