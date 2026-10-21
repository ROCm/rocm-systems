#!/usr/bin/env python3
"""Unit tests for how the skill is packaged (stdlib unittest, no GPU required).

The AMD skills catalogue imports this skill from here unchanged, so its
authoring rules are checked here: a frontmatter that routes, a governance card,
lists rather than tables, and an eval dataset in the format skillscope, the
catalogue's runner, reads (evals/evals.json plus evals/hooks.py), whose
deterministic checks can fail.
"""

from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import types
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
SKILL = SKILL_DIR.name
EVALS_DIR = SKILL_DIR / "evals"
DATASET = EVALS_DIR / "evals.json"
HOOKS = EVALS_DIR / "hooks.py"

# The frontmatter fields the catalogue's runtimes read.
KNOWN_FIELDS = {"name", "description", "allowed-tools", "license", "compatibility", "metadata"}

# skillscope/schema/evals.schema.json: the keys each kind of evaluation may carry.
EXPECTATIONS = {"expected_behavior", "unexpected_behavior", "logs_contain", "files_exist"}
NON_TRIGGERING_KEYS = {"id", "prompt", "skill_should_trigger", "note"}
TRIGGERING_KEYS = NON_TRIGGERING_KEYS | EXPECTATIONS | {"workspace"}

# Everything skillscope reads from evals/. Anything else there is never run.
EVALS_FILES = {"evals.json", "extended_evals.json", "hooks.py", "machine.yml", "files"}


def _frontmatter() -> tuple[list[str], str]:
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.DOTALL)
    if match is None:
        raise AssertionError("SKILL.md has no frontmatter block")
    return match.group(1).splitlines(), match.group(2)


def _dataset() -> list[dict]:
    return json.loads(DATASET.read_text(encoding="utf-8"))["evaluations"]


def _load_hooks():
    """Import hooks.py the way skillscope does, with nothing stubbed."""
    spec = importlib.util.spec_from_file_location("decode_and_triage_evalhooks", HOOKS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log(*tool_uses: tuple[str, dict], result: str = f"No need for {SKILL} here.") -> str:
    """A run log as skillscope's Run.logs holds it: one stream-json event per line,
    with an init event that lists the skill, as a real one does."""
    events = [{"type": "system", "subtype": "init", "skills": [SKILL, "enable-recording"]}]
    for name, args in tool_uses:
        events.append(
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": name, "input": args}]},
            }
        )
    events.append({"type": "result", "result": result})
    return "\n".join(json.dumps(e, ensure_ascii=False) for e in events)


def _run(*tool_uses: tuple[str, dict], result: str) -> types.SimpleNamespace:
    """The two attributes of skillscope's Run that the hooks read."""
    return types.SimpleNamespace(logs=_log(*tool_uses, result=result), result_text=result)


LOADED = ("Skill", {"skill": SKILL})


class FrontmatterTests(unittest.TestCase):
    def test_only_fields_a_runtime_reads(self):
        lines, _ = _frontmatter()
        keys = {line.split(":", 1)[0] for line in lines if re.match(r"^[a-z-]+:", line)}
        self.assertLessEqual(keys, KNOWN_FIELDS)
        self.assertIn("allowed-tools", keys)

    def test_description_is_one_line_that_routes(self):
        lines, _ = _frontmatter()
        (line,) = [l for l in lines if l.startswith("description:")]
        value = line.split(":", 1)[1].strip()
        self.assertTrue(value.startswith('"') and value.endswith('"'), "not a one-line string")
        description = json.loads(value)
        self.assertLessEqual(len(description), 1024)
        for phrase in ("Use when", "Use PROACTIVELY when", "Do NOT activate when", ".hrr", "replay"):
            self.assertIn(phrase, description)
        # Mechanics are not what a user says.
        self.assertNotRegex(description, r"\.(sh|ps1|py)\b")

    def test_inputs_and_outputs_are_in_the_body(self):
        _, body = _frontmatter()
        self.assertIn("## Inputs and outputs", body)
        self.assertIn("## Prerequisites", body)

    def test_body_has_no_tables(self):
        _, body = _frontmatter()
        rows = [l for l in body.splitlines() if l.lstrip().startswith("|")]
        self.assertEqual(rows, [])


class SkillCardTests(unittest.TestCase):
    def test_required_sections_are_not_empty(self):
        text = (SKILL_DIR / "skill-card.md").read_text(encoding="utf-8")
        for section in ("Description", "Owner", "License"):
            match = re.search(rf"^## {section}\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
            self.assertIsNotNone(match, section)
            self.assertTrue(match.group(1).strip(), section)


class EvalDatasetTests(unittest.TestCase):
    """evals/evals.json against skillscope's schema and coverage bar."""

    def test_evals_dir_holds_only_what_the_runner_reads(self):
        present = {p.name for p in EVALS_DIR.iterdir() if p.name != "__pycache__"}
        self.assertLessEqual(present, EVALS_FILES)
        self.assertIn("evals.json", present)

    def test_dataset_matches_the_schema(self):
        top = json.loads(DATASET.read_text(encoding="utf-8"))
        self.assertLessEqual(set(top), {"evaluations", "comment"})
        cases = top["evaluations"]
        self.assertIsInstance(cases, list)
        self.assertTrue(cases)
        for case in cases:
            with self.subTest(case=case.get("id")):
                self.assertIsInstance(case.get("id"), str)
                self.assertTrue(case["id"].strip())
                self.assertIsInstance(case.get("prompt"), str)
                self.assertTrue(case["prompt"].strip())
                self.assertIn(case.get("skill_should_trigger"), (True, False))
                allowed = TRIGGERING_KEYS if case["skill_should_trigger"] else NON_TRIGGERING_KEYS
                self.assertLessEqual(set(case), allowed)
                if "note" in case:
                    self.assertIsInstance(case["note"], str)
                for key in EXPECTATIONS & set(case):
                    self.assertIsInstance(case[key], list, key)
                    for item in case[key]:
                        self.assertIsInstance(item, str, key)
                        self.assertTrue(item.strip(), key)
                if "workspace" in case:
                    self.assertTrue((SKILL_DIR / case["workspace"]).is_dir(), case["workspace"])

    def test_ids_are_unique(self):
        ids = [c["id"] for c in _dataset()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_coverage_bar(self):
        cases = _dataset()
        positive = [c for c in cases if c["skill_should_trigger"]]
        self.assertGreaterEqual(len(positive), 3)
        self.assertGreaterEqual(len(cases) - len(positive), 2)
        self.assertTrue(any(EXPECTATIONS & set(c) for c in positive))

    def test_no_case_asserts_the_skill_name_in_the_log(self):
        # The init event lists every installed skill, so this could not fail.
        for case in _dataset():
            for literal in case.get("logs_contain", []):
                self.assertNotIn(SKILL, literal, case["id"])


class EvalHooksTests(unittest.TestCase):
    """evals/hooks.py: imports bare, stages real fixtures, and its checks can fail."""

    @classmethod
    def setUpClass(cls):
        cls.hooks = _load_hooks()
        cls.cases = {c["id"]: c for c in _dataset()}

    def test_skill_name_in_the_log_is_not_activation(self):
        loaded = self.hooks.skill_loaded
        self.assertFalse(loaded(_log()))
        self.assertFalse(loaded(_log(("Skill", {"skill": "enable-recording"}))))
        self.assertFalse(loaded(_log(("Read", {"file_path": "notes.txt"}))))

    def test_activation_is_seen(self):
        for tool_use in (
            ("Skill", {"skill": SKILL}),
            ("Skill", {"skill": f"amd-skills:{SKILL}"}),
            ("Read", {"file_path": f"/tmp/w/.claude/skills/{SKILL}/SKILL.md"}),
            ("Bash", {"command": f".claude/skills/{SKILL}/scripts/triage_archive.sh --archive x"}),
        ):
            with self.subTest(tool_use=tool_use):
                self.assertTrue(self.hooks.skill_loaded(_log(tool_use)))

    def test_every_hooked_case_is_a_behavioral_case(self):
        # skillscope only calls the hooks for triggering cases with expectations.
        self.assertTrue(self.hooks.CASES)
        for case_id in self.hooks.CASES:
            with self.subTest(case=case_id):
                self.assertIn(case_id, self.cases)
                case = self.cases[case_id]
                self.assertIs(case["skill_should_trigger"], True)
                self.assertTrue(case.get("expected_behavior") or case.get("unexpected_behavior"))

    def test_suite_covers_fault_unreadable_clean_and_false_activation(self):
        staged = {f for files, _ in self.hooks.CASES.values() for f in files.values()}
        for fixture in ("replay_memory_fault.log", "version_mismatch.log", "replay_pass.log"):
            self.assertIn(fixture, staged)
        # False activation: graded by `skillscope routing` on the run log's tool calls.
        self.assertGreaterEqual(sum(1 for c in self.cases.values() if not c["skill_should_trigger"]), 2)

    def test_setup_stages_the_fixtures_under_the_prompted_names(self):
        for case_id, (files, _) in self.hooks.CASES.items():
            with self.subTest(case=case_id), tempfile.TemporaryDirectory() as workspace:
                self.hooks.setup(Path(workspace), types.SimpleNamespace(id=case_id), {})
                for name, fixture in files.items():
                    self.assertIn(name, self.cases[case_id]["prompt"])
                    staged = Path(workspace) / name
                    self.assertEqual(staged.read_bytes(), (self.hooks.FIXTURES / fixture).read_bytes())

    def _check(self, case_id: str, run) -> None:
        self.hooks.check(run, types.SimpleNamespace(id=case_id), {})

    def test_fault_check(self):
        good = f"MAF, illegal_memory_access at {self.hooks.FAULT_ADDRESS} in {self.hooks.FAULT_KERNEL}"
        self._check("triage-recorded-memory-fault", _run(LOADED, result=good))
        for run in (
            _run(result=good),
            _run(LOADED, result=f"MAF in {self.hooks.FAULT_KERNEL}"),
            _run(LOADED, result=f"MAF at {self.hooks.FAULT_ADDRESS}"),
        ):
            with self.assertRaises(AssertionError):
                self._check("triage-recorded-memory-fault", run)

    def test_unreadable_check(self):
        reply = "The archive format and the reader disagree; nothing was replayed."
        self._check("triage-unreadable-archive", _run(LOADED, result=reply))
        repaired = ("Bash", {"command": "hrr-playback --repair archive.hrr"})
        for run in (_run(result=reply), _run(LOADED, repaired, result=reply)):
            with self.assertRaises(AssertionError):
                self._check("triage-unreadable-archive", run)

    def test_clean_pass_check(self):
        self._check("triage-clean-pass-of-crashing-workload", _run(LOADED, result="Outcome PASS."))
        for run in (_run(result="Outcome PASS."), _run(LOADED, result="It did not reproduce.")):
            with self.assertRaises(AssertionError):
                self._check("triage-clean-pass-of-crashing-workload", run)

    def test_unhooked_cases_pass_through(self):
        self._check("no-archive-yet", _run(result=""))


if __name__ == "__main__":
    unittest.main()
