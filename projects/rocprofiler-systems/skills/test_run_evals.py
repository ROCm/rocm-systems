# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for run_evals.py.

They need only pytest. They never run the real `claude` CLI, never read the real home
folder, and build their own throwaway skill trees, so they do not depend on the
project's datasets or on anything else in the repository:

    python3 -m pytest skills/test_run_evals.py
"""

import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("run_evals.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("run_evals_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def evals():
    return _load_module()


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Point HOME at an empty folder so no test can see or touch the real one."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def no_real_claude(monkeypatch):
    """Fail loudly if a test would run the real claude CLI."""
    real_run = subprocess.run

    def guard(command, *args, **kwargs):
        if command and command[0] == "claude":
            raise AssertionError("a test tried to run the real claude CLI")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guard)


def case(case_id, trigger, **extra):
    data = {"id": case_id, "prompt": f"prompt {case_id}", "skill_should_trigger": trigger}
    data.update(extra)
    return data


def valid_cases(prefix=""):
    """The smallest valid dataset: 3 positive (one judged) and 2 negative cases."""
    return [
        case(f"{prefix}p1", True, expected_behavior=["says something"]),
        case(f"{prefix}p2", True),
        case(f"{prefix}p3", True),
        case(f"{prefix}n1", False),
        case(f"{prefix}n2", False),
    ]


def write_skill(root, name, cases=None):
    evals_dir = root / name / "evals"
    evals_dir.mkdir(parents=True)
    body = {"evaluations": valid_cases(name + "-") if cases is None else cases}
    (evals_dir / "evals.json").write_text(json.dumps(body))
    return root / name


class FakeRun:
    """Stand-in for subprocess.run that records calls and returns canned output."""

    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, "")


def tool_use(name, **payload):
    return {"type": "tool_use", "name": name, "input": payload}


def event(*blocks):
    return {"message": {"content": list(blocks)}}


# --- load_cases -------------------------------------------------------------------


class TestDatasetSchema:
    """The field names belong to the schema shared with the eval CI, not to this script."""

    def test_field_names_match_the_shared_schema(self, evals):
        assert evals.KNOWN_KEYS == {
            "id",
            "prompt",
            "skill_should_trigger",
            "note",
            "expected_behavior",
            "unexpected_behavior",
            "logs_contain",
            "files_exist",
        }
        assert evals.JUDGED_KEYS == ("expected_behavior", "unexpected_behavior")
        assert set(evals.BEHAVIOURAL_KEYS) == {
            "expected_behavior",
            "unexpected_behavior",
            "logs_contain",
            "files_exist",
        }


class TestLoadCases:
    def test_valid_dataset_is_returned_unchanged(self, evals, tmp_path):
        skill = write_skill(tmp_path, "demo")
        assert evals.load_cases(skill) == valid_cases("demo-")

    def test_negative_case_may_carry_a_note(self, evals, tmp_path):
        cases = valid_cases()
        cases[3]["note"] = "near miss"
        assert evals.load_cases(write_skill(tmp_path, "demo", cases)) == cases

    def test_unexpected_behavior_alone_counts_as_judged(self, evals, tmp_path):
        cases = valid_cases()
        cases[0].pop("expected_behavior")
        cases[0]["unexpected_behavior"] = ["does the wrong thing"]
        assert evals.load_cases(write_skill(tmp_path, "demo", cases))

    @pytest.mark.parametrize(
        "mutate, message",
        [
            pytest.param(lambda c: c[0].update(bogus=1), "unknown key", id="unknown-key"),
            pytest.param(
                lambda c: c[0].update(skill_should_trigger="yes"),
                "boolean skill_should_trigger",
                id="non-boolean-trigger",
            ),
            pytest.param(
                lambda c: c[1].update(id=c[0]["id"]),
                "duplicate case id",
                id="duplicate-id",
            ),
            pytest.param(
                lambda c: c[3].update(expected_behavior=["x"]),
                "is negative but sets",
                id="negative-with-behaviour",
            ),
            pytest.param(
                lambda c: c.pop(0),
                "at least 3 positive and 2 negative",
                id="too-few-positive",
            ),
            pytest.param(
                lambda c: c.pop(),
                "at least 3 positive and 2 negative",
                id="too-few-negative",
            ),
            pytest.param(
                lambda c: c[0].pop("expected_behavior"),
                "judged behaviour",
                id="no-judged-case",
            ),
            pytest.param(
                lambda c: (
                    c[0].pop("expected_behavior"),
                    c[0].update(logs_contain=["x"]),
                ),
                "judged behaviour",
                id="logs-contain-is-not-judged",
            ),
        ],
    )
    def test_rejects_malformed_dataset(self, evals, tmp_path, mutate, message):
        cases = valid_cases()
        mutate(cases)
        skill = write_skill(tmp_path, "demo", cases)
        with pytest.raises(SystemExit) as exc:
            evals.load_cases(skill)
        assert message in str(exc.value)
        assert "demo" in str(exc.value)


# --- check_unique_ids -------------------------------------------------------------


class TestCheckUniqueIds:
    def test_distinct_ids_pass(self, evals):
        evals.check_unique_ids({"a": valid_cases("a-"), "b": valid_cases("b-")})

    def test_id_shared_by_two_skills_is_rejected(self, evals):
        with pytest.raises(SystemExit) as exc:
            evals.check_unique_ids({"a": valid_cases(), "b": valid_cases()})
        assert "'p1' is used by a and b" in str(exc.value)


# --- discover_skills --------------------------------------------------------------


@pytest.mark.usefixtures("root")
class TestDiscoverSkills:
    @pytest.fixture
    def root(self, evals, tmp_path, monkeypatch):
        monkeypatch.setattr(evals, "SKILLS_ROOT", tmp_path)
        write_skill(tmp_path, "zeta")
        write_skill(tmp_path, "alpha")
        (tmp_path / "no-evals").mkdir()
        return tmp_path

    def test_finds_only_skills_with_a_dataset_sorted_by_name(self, evals):
        assert [p.name for p in evals.discover_skills(None)] == ["alpha", "zeta"]

    def test_filter_selects_one_skill(self, evals):
        assert [p.name for p in evals.discover_skills("zeta")] == ["zeta"]

    def test_unknown_skill_is_an_error(self, evals):
        with pytest.raises(SystemExit) as exc:
            evals.discover_skills("nope")
        assert "No skill named 'nope'" in str(exc.value)

    def test_skill_without_a_dataset_cannot_be_selected(self, evals):
        with pytest.raises(SystemExit):
            evals.discover_skills("no-evals")


# --- _skills_in -------------------------------------------------------------------


class TestSkillsIn:
    @pytest.mark.parametrize(
        "ev, expected",
        [
            pytest.param(
                event(tool_use("Skill", skill="demo")), ["demo"], id="skill-call"
            ),
            pytest.param(
                event(tool_use("Skill", skill="plugin:demo")),
                ["demo"],
                id="plugin-prefix",
            ),
            pytest.param(
                event(tool_use("Skill", skill="a"), tool_use("Skill", skill="b")),
                ["a", "b"],
                id="two-calls-keep-order",
            ),
            pytest.param(
                event(tool_use("Read", file_path="/ws/.claude/skills/demo/SKILL.md")),
                [],
                id="reading-a-skill-file-is-not-a-load",
            ),
            pytest.param(
                event(tool_use("Glob", pattern="**/*"), tool_use("Bash", command="ls")),
                [],
                id="other-tools",
            ),
            pytest.param(
                event({"type": "text", "text": "I will use Skill demo"}),
                [],
                id="text-block",
            ),
            pytest.param(
                {"message": {"content": "plain string"}}, [], id="string-content"
            ),
            pytest.param({"type": "result"}, [], id="no-message"),
            pytest.param(event("not a dict"), [], id="non-dict-block"),
            pytest.param(
                {"type": "system", "subtype": "permission_denied", "message": "denied"},
                [],
                id="string-message-as-in-a-permission-denial",
            ),
            pytest.param({"message": ["a", "list"]}, [], id="list-message"),
            pytest.param({"message": None}, [], id="null-message"),
            pytest.param({"message": {"content": None}}, [], id="null-content"),
            pytest.param(
                {
                    "message": {
                        "content": [{"type": "tool_use", "name": "Skill", "input": None}]
                    }
                },
                [],
                id="skill-call-with-null-input",
            ),
            pytest.param(event(tool_use("Skill")), [], id="skill-call-without-a-name"),
            pytest.param(
                event(tool_use("Skill", skill="")), [], id="skill-call-with-an-empty-name"
            ),
        ],
    )
    def test_names_loaded_by_skill_tool_calls_only(self, evals, ev, expected):
        assert evals._skills_in(ev) == expected


# --- claude_flags / personal_skills -----------------------------------------------


class TestClaudeFlags:
    def test_defaults_isolate_from_personal_settings(self, evals):
        flags = evals.claude_flags(evals.Options(None, True, False))
        assert flags == ["--no-session-persistence", "--setting-sources", "project"]

    def test_keep_user_settings_drops_the_isolation_flag(self, evals):
        flags = evals.claude_flags(evals.Options(None, False, False))
        assert flags == ["--no-session-persistence"]

    def test_model_is_passed_through(self, evals):
        flags = evals.claude_flags(evals.Options("some-model", False, False))
        assert flags[-2:] == ["--model", "some-model"]


class TestPersonalSkills:
    def test_lists_folders_that_hold_a_skill_file(self, evals, isolated_home):
        skills = isolated_home / ".claude" / "skills"
        for name in ("alpha", "beta"):
            (skills / name).mkdir(parents=True)
            (skills / name / "SKILL.md").write_text("x")
        (skills / "empty").mkdir()
        (skills / "stray.txt").write_text("x")
        assert evals.personal_skills() == {"alpha", "beta"}

    def test_missing_folder_gives_an_empty_set(self, evals):
        assert evals.personal_skills() == set()


# --- grade_judged -----------------------------------------------------------------


class TestGradeJudged:
    OPTIONS = None

    @pytest.fixture(autouse=True)
    def options(self, evals):
        self.OPTIONS = evals.Options("m", True, True)

    def judge(self, evals, monkeypatch, stdout, claim="c", hold=True, transcript="t"):
        fake = FakeRun(stdout)
        monkeypatch.setattr(subprocess, "run", fake)
        return evals.grade_judged(claim, transcript, hold, self.OPTIONS), fake

    def test_direct_json_verdict(self, evals, monkeypatch):
        result, _ = self.judge(evals, monkeypatch, '{"pass": true, "reason": "ok"}')
        assert result == (True, "ok")

    def test_verdict_wrapped_in_a_result_string(self, evals, monkeypatch):
        wrapped = json.dumps({"result": json.dumps({"pass": False, "reason": "no"})})
        result, _ = self.judge(evals, monkeypatch, wrapped)
        assert result == (False, "no")

    def test_claim_that_should_not_hold_inverts_the_verdict(self, evals, monkeypatch):
        verdict = '{"pass": false, "reason": "did not happen"}'
        result, _ = self.judge(evals, monkeypatch, verdict, hold=False)
        assert result == (True, "did not happen")

    @pytest.mark.parametrize(
        "stdout", ["not json", "[]", '{"result": "also not json"}', '{"result": 3}', ""]
    )
    def test_unparseable_output_fails_the_claim(self, evals, monkeypatch, stdout):
        result, _ = self.judge(evals, monkeypatch, stdout)
        assert result == (False, "judge returned unparseable output")

    def test_command_and_prompt(self, evals, monkeypatch):
        _, fake = self.judge(
            evals,
            monkeypatch,
            '{"pass": true, "reason": ""}',
            claim="my claim",
            hold=False,
        )
        command, kwargs = fake.calls[0]
        assert command[:2] == ["claude", "-p"]
        assert "my claim" in command[2] and "NOT met" in command[2]
        assert command[command.index("--output-format") + 1] == "json"
        schema = json.loads(command[command.index("--json-schema") + 1])
        assert schema["required"] == ["pass", "reason"]
        assert "--model" in command and "--setting-sources" in command
        assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["check"] is False

    def test_met_wording_when_the_claim_should_hold(self, evals, monkeypatch):
        _, fake = self.judge(evals, monkeypatch, '{"pass": true, "reason": ""}')
        assert "should be met" in fake.calls[0][0][2]

    def test_long_transcripts_are_truncated(self, evals, monkeypatch):
        _, fake = self.judge(
            evals, monkeypatch, '{"pass": true, "reason": ""}', transcript="Z" * 300000
        )
        assert fake.calls[0][0][2].count("Z") == 200000


# --- grade ------------------------------------------------------------------------


class TestGrade:
    @pytest.fixture
    def routing(self, evals):
        return evals.Options(None, True, False)

    @pytest.fixture
    def behavioural(self, evals):
        return evals.Options(None, True, True)

    def test_positive_case_passes_when_the_skill_loaded(self, evals, tmp_path, routing):
        failures = evals.grade(case("c", True), "demo", "", ["demo"], tmp_path, routing)
        assert failures == []

    def test_positive_case_fails_when_nothing_loaded(self, evals, tmp_path, routing):
        failures = evals.grade(case("c", True), "demo", "", [], tmp_path, routing)
        assert failures == ["expected demo to trigger, loaded=none"]

    def test_negative_case_fails_when_the_skill_loaded(self, evals, tmp_path, routing):
        failures = evals.grade(case("c", False), "demo", "", ["demo"], tmp_path, routing)
        assert failures == ["expected demo to not trigger, loaded=['demo']"]

    def test_another_skill_loading_does_not_count(self, evals, tmp_path, routing):
        failures = evals.grade(case("c", False), "demo", "", ["other"], tmp_path, routing)
        assert failures == []

    def test_routing_mode_ignores_behavioural_expectations(
        self, evals, tmp_path, routing
    ):
        extra = {"logs_contain": ["missing"], "files_exist": ["nope.txt"]}
        failures = evals.grade(
            case("c", True, **extra), "demo", "", ["demo"], tmp_path, routing
        )
        assert failures == []

    def test_logs_contain(self, evals, tmp_path, behavioural):
        positive = case("c", True, logs_contain=["found", "absent"])
        failures = evals.grade(
            positive, "demo", "a found b", ["demo"], tmp_path, behavioural
        )
        assert failures == ["logs missing 'absent'"]

    def test_files_exist_matches_anywhere_in_the_workspace(
        self, evals, tmp_path, behavioural
    ):
        (tmp_path / "sub" / "dir").mkdir(parents=True)
        (tmp_path / "sub" / "dir" / "out.json").write_text("x")
        (tmp_path / "top.txt").write_text("x")
        positive = case("c", True, files_exist=["out.json", "top.txt", "gone.txt"])
        failures = evals.grade(positive, "demo", "", ["demo"], tmp_path, behavioural)
        assert failures == ["missing artifact 'gone.txt'"]

    def test_judged_claims_are_graded_in_the_right_direction(
        self, evals, tmp_path, behavioural, monkeypatch
    ):
        asked = []

        def fake_judge(claim, _transcript, should_hold, _options):
            asked.append((claim, should_hold))
            return False, f"why {claim}"

        monkeypatch.setattr(evals, "grade_judged", fake_judge)
        positive = case("c", True, expected_behavior=["do"], unexpected_behavior=["dont"])
        failures = evals.grade(positive, "demo", "t", ["demo"], tmp_path, behavioural)
        assert asked == [("do", True), ("dont", False)]
        assert failures == [
            "expected behaviour not met: do (why do)",
            "unexpected behaviour occurred: dont (why dont)",
        ]

    def test_passing_judge_adds_no_failure(
        self, evals, tmp_path, behavioural, monkeypatch
    ):
        monkeypatch.setattr(evals, "grade_judged", lambda *a: (True, ""))
        positive = case("c", True, expected_behavior=["do"])
        assert evals.grade(positive, "demo", "", ["demo"], tmp_path, behavioural) == []

    def test_all_failures_are_reported_together(self, evals, tmp_path, behavioural):
        positive = case("c", True, logs_contain=["x"], files_exist=["y"])
        failures = evals.grade(positive, "demo", "", [], tmp_path, behavioural)
        assert len(failures) == 3


# --- run_agent --------------------------------------------------------------------


class TestRunAgent:
    STREAM = "\n".join(
        [
            json.dumps(
                {"type": "system", "subtype": "init", "skills": ["a", "b"], "model": "m1"}
            ),
            "this line is not json",
            json.dumps({"type": "assistant", **event(tool_use("Skill", skill="a"))}),
            json.dumps({"type": "result", "result": "done"}),
        ]
    )

    def test_parses_loaded_skills_visible_skills_and_model(
        self, evals, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(subprocess, "run", FakeRun(self.STREAM))
        transcript, loaded, visible, model = evals.run_agent(
            "hi", tmp_path, evals.Options(None, True, False)
        )
        assert loaded == ["a"]
        assert visible == {"a", "b"}
        assert model == "m1"
        assert "this line is not json" not in transcript
        assert len(transcript.splitlines()) == 3

    def test_events_that_are_not_objects_or_have_a_string_message_do_not_crash(
        self, evals, tmp_path, monkeypatch
    ):
        denied = {
            "type": "system",
            "subtype": "permission_denied",
            "tool_name": "Bash",
            "message": "This Bash command contains multiple operations.",
        }
        stream = "\n".join(
            [
                json.dumps({"type": "system", "subtype": "init", "skills": ["a"]}),
                json.dumps(denied),
                json.dumps("a bare json string"),
                json.dumps([1, 2]),
                "42",
                json.dumps({"type": "assistant", **event(tool_use("Skill", skill="a"))}),
            ]
        )
        monkeypatch.setattr(subprocess, "run", FakeRun(stream))
        transcript, loaded, visible, _ = evals.run_agent(
            "hi", tmp_path, evals.Options(None, True, True)
        )
        assert loaded == ["a"] and visible == {"a"}
        kept = [json.loads(line) for line in transcript.splitlines()]
        assert denied in kept
        assert all(isinstance(entry, dict) for entry in kept)

    def test_empty_output_is_handled(self, evals, tmp_path, monkeypatch):
        monkeypatch.setattr(subprocess, "run", FakeRun(""))
        result = evals.run_agent("hi", tmp_path, evals.Options(None, True, False))
        assert result == ("", [], set(), "")

    def test_routing_run_denies_tools_and_uses_the_short_timeout(
        self, evals, tmp_path, monkeypatch
    ):
        fake = FakeRun(self.STREAM)
        monkeypatch.setattr(subprocess, "run", fake)
        evals.run_agent("hi", tmp_path, evals.Options("mdl", True, False))
        command, kwargs = fake.calls[0]
        denied = command.index("--disallowedTools")
        assert command[denied + 1 : denied + 5] == [
            "Bash",
            "Edit",
            "Write",
            "NotebookEdit",
        ]
        assert "--permission-mode" not in command
        assert command[:3] == ["claude", "-p", "hi"]
        assert command[command.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in command
        assert command[command.index("--add-dir") + 1] == str(tmp_path)
        assert command[command.index("--model") + 1] == "mdl"
        assert command[command.index("--setting-sources") + 1] == "project"
        assert kwargs["cwd"] == tmp_path
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["timeout"] == 300 and kwargs["check"] is False

    def test_behavioural_run_allows_edits_and_uses_the_long_timeout(
        self, evals, tmp_path, monkeypatch
    ):
        fake = FakeRun(self.STREAM)
        monkeypatch.setattr(subprocess, "run", fake)
        evals.run_agent("hi", tmp_path, evals.Options(None, False, True))
        command, kwargs = fake.calls[0]
        assert command[command.index("--permission-mode") + 1] == "acceptEdits"
        assert "--disallowedTools" not in command
        assert "--setting-sources" not in command
        assert kwargs["timeout"] == 900

    def test_timeout_propagates(self, evals, tmp_path, monkeypatch):
        def slow(command, **_kwargs):
            raise subprocess.TimeoutExpired(command, 1)

        monkeypatch.setattr(subprocess, "run", slow)
        with pytest.raises(subprocess.TimeoutExpired):
            evals.run_agent("hi", tmp_path, evals.Options(None, True, False))


# --- build_workspace --------------------------------------------------------------


class TestBuildWorkspace:
    def fake_root(self, evals, tmp_path, monkeypatch, body):
        root = tmp_path / "skills"
        root.mkdir()
        script = root / "install-skills.sh"
        script.write_text("#!/usr/bin/env bash\nset -e\n" + body)
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setattr(evals, "SKILLS_ROOT", root)

    def test_installs_the_skills_under_the_workspace_home(
        self, evals, tmp_path, monkeypatch, isolated_home
    ):
        body = (
            '[[ "$1" == "--agent" && "$2" == "claude" ]] || exit 2\n'
            'mkdir -p "$HOME/.claude/skills/demo"\n'
            'echo ok > "$HOME/.claude/skills/demo/SKILL.md"\n'
        )
        self.fake_root(evals, tmp_path, monkeypatch, body)
        workspace = evals.build_workspace(tmp_path / "case")
        assert workspace == tmp_path / "case" / "workspace"
        assert (
            workspace / ".claude" / "skills" / "demo" / "SKILL.md"
        ).read_text() == "ok\n"
        assert not (isolated_home / ".claude").exists()

    def test_a_failing_install_script_is_an_error(self, evals, tmp_path, monkeypatch):
        self.fake_root(evals, tmp_path, monkeypatch, "exit 1\n")
        with pytest.raises(subprocess.CalledProcessError):
            evals.build_workspace(tmp_path / "case")


# --- run_case ---------------------------------------------------------------------


class TestRunCase:
    @pytest.fixture
    def patched(self, evals, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(evals, "build_workspace", lambda _root: workspace)
        return monkeypatch

    def test_passing_case_carries_visible_skills_and_model(
        self, evals, tmp_path, patched
    ):
        patched.setattr(
            evals, "run_agent", lambda *a: ("tr", ["demo"], {"demo", "other"}, "m")
        )
        outcome = evals.run_case(
            "demo", case("c", True), tmp_path, evals.Options(None, True, False)
        )
        assert outcome == evals.Outcome("demo", "c", [], {"demo", "other"}, "m")

    def test_workspace_folder_does_not_reveal_the_skill_or_case(
        self, evals, tmp_path, monkeypatch
    ):
        roots = []

        def record(root):
            roots.append(root)
            return tmp_path

        monkeypatch.setattr(evals, "build_workspace", record)
        monkeypatch.setattr(evals, "run_agent", lambda *a: ("t", ["demo"], set(), ""))
        options = evals.Options(None, True, False)
        evals.run_case("demo-skill", case("some-case", True), tmp_path, options)
        evals.run_case("demo-skill", case("other-case", True), tmp_path, options)
        names = [root.name for root in roots]
        assert all("demo" not in n and "case" not in n for n in names)
        assert len(set(names)) == 2
        assert all(root.parent == tmp_path for root in roots)

    def test_failing_case_reports_the_failure(self, evals, tmp_path, patched):
        patched.setattr(evals, "run_agent", lambda *a: ("tr", [], set(), "m"))
        outcome = evals.run_case(
            "demo", case("c", True), tmp_path, evals.Options(None, True, False)
        )
        assert outcome.failures == ["expected demo to trigger, loaded=none"]

    def test_timeout_becomes_a_failure(self, evals, tmp_path, patched):
        def slow(*_args):
            raise subprocess.TimeoutExpired("claude", 1)

        patched.setattr(evals, "run_agent", slow)
        outcome = evals.run_case(
            "demo", case("c", True), tmp_path, evals.Options(None, True, False)
        )
        assert outcome == evals.Outcome("demo", "c", ["timed out"], set())

    def test_transcript_is_saved_when_asked(self, evals, tmp_path, patched):
        patched.setattr(
            evals, "run_agent", lambda *a: ("the transcript", ["demo"], set(), "")
        )
        options = evals.Options(None, True, False, transcripts=tmp_path / "out" / "dir")
        evals.run_case("demo", case("c", True), tmp_path, options)
        saved = tmp_path / "out" / "dir" / "demo--c.jsonl"
        assert saved.read_text() == "the transcript"

    def test_nothing_is_saved_by_default(self, evals, tmp_path, patched):
        patched.setattr(evals, "run_agent", lambda *a: ("t", ["demo"], set(), ""))
        evals.run_case(
            "demo", case("c", True), tmp_path, evals.Options(None, True, False)
        )
        assert not list(tmp_path.rglob("*.jsonl"))


# --- main -------------------------------------------------------------------------


@pytest.mark.usefixtures("tree")
class TestMain:
    @pytest.fixture
    def tree(self, evals, tmp_path, monkeypatch):
        root = tmp_path / "skills"
        root.mkdir()
        write_skill(root, "alpha")
        write_skill(root, "beta")
        monkeypatch.setattr(evals, "SKILLS_ROOT", root)
        monkeypatch.setattr(evals.shutil, "which", lambda name: "/usr/bin/" + name)
        return root

    @pytest.fixture
    def run_main(self, evals, monkeypatch, capsys):
        """Run main() with a fake run_case and return (exit code, output, calls)."""

        def run(*argv, outcome=None):
            calls = []

            def fake_case(name, case_data, _tmp, options):
                calls.append((name, dict(case_data), options))
                if outcome:
                    return outcome(name, case_data)
                return evals.Outcome(name, case_data["id"], [], set(), "m")

            monkeypatch.setattr(evals, "run_case", fake_case)
            monkeypatch.setattr(sys, "argv", ["run_evals.py", *argv])
            code = evals.main()
            return code, capsys.readouterr().out, calls

        return run

    def test_list_validates_without_the_cli(self, evals, run_main, monkeypatch):
        def forbidden(_name):
            raise AssertionError("--list must not look for the CLI")

        monkeypatch.setattr(evals.shutil, "which", forbidden)
        code, out, calls = run_main("--list")
        assert code == 0 and calls == []
        assert (
            "alpha: 5 cases (3 positive)" in out and "beta: 5 cases (3 positive)" in out
        )

    def test_missing_cli_is_reported_as_skipped(self, evals, run_main, monkeypatch):
        monkeypatch.setattr(evals.shutil, "which", lambda name: None)
        code, out, calls = run_main()
        assert code == evals.EXIT_SKIPPED == 77
        assert "SKIP" in out and calls == []

    def test_all_cases_passing_exits_zero(self, run_main):
        code, out, calls = run_main()
        assert code == 0 and len(calls) == 10
        assert "10/10 passed" in out and "routing mode, model m" in out
        assert "PASS alpha-p1" in out and "FAIL" not in out

    def test_a_failing_case_exits_one_and_is_listed(self, evals, run_main):
        def outcome(name, case_data):
            failures = ["broken"] if case_data["id"] == "beta-n1" else []
            return evals.Outcome(name, case_data["id"], failures, set(), "m")

        code, out, _ = run_main(outcome=outcome)
        assert code == 1
        assert "FAIL beta-n1" in out and "     broken" in out and "9/10 passed" in out

    def test_skill_option_limits_the_run(self, run_main):
        _, out, calls = run_main("--skill", "beta")
        assert {name for name, _, _ in calls} == {"beta"}
        assert "5/5 passed" in out and "alpha" not in out

    def test_routing_mode_drops_behavioural_keys(self, run_main):
        _, _, calls = run_main()
        assert all("expected_behavior" not in case_data for _, case_data, _ in calls)

    def test_behavioural_mode_keeps_them(self, run_main):
        _, out, calls = run_main("--mode", "behavioural")
        assert any("expected_behavior" in case_data for _, case_data, _ in calls)
        assert calls[0][2].behavioural is True and "behavioural mode" in out

    def test_default_options(self, run_main):
        _, _, calls = run_main()
        options = calls[0][2]
        assert options.isolate is True and options.model is None
        assert options.behavioural is False and options.transcripts is None

    def test_option_flags_reach_the_cases(self, run_main, tmp_path):
        _, _, calls = run_main(
            "--keep-user-settings",
            "--model",
            "mm",
            "--save-transcripts",
            str(tmp_path / "t"),
        )
        options = calls[0][2]
        assert options.isolate is False and options.model == "mm"
        assert options.transcripts == tmp_path / "t"

    def test_zero_jobs_still_runs(self, run_main):
        code, _, calls = run_main("--jobs", "0")
        assert code == 0 and len(calls) == 10

    def test_case_id_shared_by_two_skills_is_rejected(
        self, evals, tmp_path, monkeypatch, run_main
    ):
        root = tmp_path / "dups"
        root.mkdir()
        write_skill(root, "alpha", valid_cases())
        write_skill(root, "beta", valid_cases())
        monkeypatch.setattr(evals, "SKILLS_ROOT", root)
        monkeypatch.setattr(evals.shutil, "which", lambda name: "/usr/bin/claude")
        with pytest.raises(SystemExit) as exc:
            run_main()
        assert "is used by alpha and beta" in str(exc.value)

    def test_warns_about_competing_personal_skills(self, run_main, evals, isolated_home):
        for name in ("my-own", "alpha"):
            (isolated_home / ".claude" / "skills" / name).mkdir(parents=True)
            (isolated_home / ".claude" / "skills" / name / "SKILL.md").write_text("x")

        def outcome(name, case_data):
            visible = {"alpha", "beta", "my-own", "deep-research"}
            return evals.Outcome(name, case_data["id"], [], visible, "m")

        _, out, _ = run_main(outcome=outcome)
        assert "WARNING: 1 personal skill(s)" in out and "my-own" in out
        assert "deep-research" not in out.split("WARNING")[1]

    def test_no_warning_when_only_the_installed_skills_are_visible(self, run_main, evals):
        def outcome(name, case_data):
            return evals.Outcome(name, case_data["id"], [], {"alpha", "beta"}, "m")

        _, out, _ = run_main(outcome=outcome)
        assert "WARNING" not in out

    def test_models_are_reported_sorted_and_deduplicated(self, run_main, evals):
        def outcome(name, case_data):
            model = "b-model" if name == "beta" else "a-model"
            return evals.Outcome(name, case_data["id"], [], set(), model)

        _, out, _ = run_main(outcome=outcome)
        assert "model a-model, b-model" in out

    def test_unknown_model_is_reported(self, run_main, evals):
        def outcome(name, case_data):
            return evals.Outcome(name, case_data["id"], [], set(), "")

        _, out, _ = run_main(outcome=outcome)
        assert "model unknown" in out
