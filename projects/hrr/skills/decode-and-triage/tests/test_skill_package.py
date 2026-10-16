#!/usr/bin/env python3
"""Unit tests for how the skill is packaged (stdlib unittest, no GPU required).

The AMD skills catalogue imports this skill from here unchanged, so its
authoring rules are checked here: a frontmatter that routes, a governance card,
lists rather than tables, and an eval suite whose activation check can fail.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
SKILL = SKILL_DIR.name
EVALS = SKILL_DIR / "evals" / "evals.py"

# The frontmatter fields the catalogue's runtimes read.
KNOWN_FIELDS = {"name", "description", "allowed-tools", "license", "compatibility", "metadata"}


def _frontmatter() -> tuple[list[str], str]:
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.DOTALL)
    if match is None:
        raise AssertionError("SKILL.md has no frontmatter block")
    return match.group(1).splitlines(), match.group(2)


def _load_evals():
    """Import evals.py with a stand-in for the catalogue's harness."""
    sys.modules.setdefault("harness", types.SimpleNamespace(claude=None))
    spec = importlib.util.spec_from_file_location("decode_and_triage_evals", EVALS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log(*tool_uses: tuple[str, dict]) -> str:
    """A stream-json run log whose init event lists the skill, as a real one does."""
    events = [{"type": "system", "subtype": "init", "skills": [SKILL, "enable-recording"]}]
    for name, args in tool_uses:
        events.append(
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": name, "input": args}]},
            }
        )
    events.append({"type": "result", "result": f"No need for {SKILL} here."})
    return "\n".join(json.dumps(e) for e in events)


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


class EvalSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evals = _load_evals()

    def test_skill_name_in_the_log_is_not_activation(self):
        self.assertFalse(self.evals.skill_loaded(_log()))
        self.assertFalse(self.evals.skill_loaded(_log(("Skill", {"skill": "enable-recording"}))))
        self.assertFalse(self.evals.skill_loaded(_log(("Read", {"file_path": "notes.txt"}))))

    def test_activation_is_seen(self):
        for tool_use in (
            ("Skill", {"skill": SKILL}),
            ("Skill", {"skill": f"amd-skills:{SKILL}"}),
            ("Read", {"file_path": f"/tmp/w/.claude/skills/{SKILL}/SKILL.md"}),
            ("Bash", {"command": f".claude/skills/{SKILL}/scripts/triage_archive.sh --archive x"}),
        ):
            with self.subTest(tool_use=tool_use):
                self.assertTrue(self.evals.skill_loaded(_log(tool_use)))

    def test_suite_covers_fault_unreadable_clean_and_false_activation(self):
        names = {n for n in dir(self.evals) if n.startswith("test_")}
        self.assertLessEqual(
            {
                "test_fault_from_recorded_evidence",
                "test_unreadable_archive_is_not_a_workload_failure",
                "test_clean_pass_of_a_crashing_workload",
                "test_routes_out",
            },
            names,
        )
        self.assertTrue(self.evals.SHOULD_NOT_TRIGGER)

    def test_staged_fixtures_exist(self):
        staged = re.findall(r'_stage\(agent, "([^"]+)"', EVALS.read_text(encoding="utf-8"))
        self.assertTrue(staged)
        for name in staged:
            self.assertTrue((self.evals.FIXTURES / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
