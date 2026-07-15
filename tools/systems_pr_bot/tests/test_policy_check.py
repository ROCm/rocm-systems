"""Focused unit tests for the Systems PR Bot testing-declaration policy.

Covers the deterministic PR-body parser (`parse_testing_declarations`) and the
`ensure_unit_tests` policy behavior. These tests never touch the network — they
exercise pure functions only.
"""

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "policy_check.py"
_spec = importlib.util.spec_from_file_location("policy_check", _MODULE_PATH)
policy_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy_check)


MARKER = policy_check.TESTING_MARKER


def _load_policy():
    return policy_check.load_policy(_MODULE_PATH.parent / "policy.yml")


def _file(name, status="modified"):
    return {"filename": name, "status": status}


def test_parse_missing_marker_returns_empty():
    d = policy_check.parse_testing_declarations("no marker here")
    assert d.marker_present is False
    assert d.entries == {}


def test_parse_valid_multi_file():
    body = f"""Intro.

{MARKER}
## Testing

### File: `src/foo.cpp`
- **Description:** Fixed batching.
- **Tests:**
  - `ctest --test-dir build -R widget`
  - manual smoke test

### File: `src/bar.py`
- **Description:** New parser.
- **Tests:**
  - pytest tools/tests/test_bar.py
"""
    d = policy_check.parse_testing_declarations(body)
    assert d.marker_present is True
    assert set(d.entries) == {"src/foo.cpp", "src/bar.py"}
    assert d.entries["src/foo.cpp"].tests == [
        "ctest --test-dir build -R widget",
        "manual smoke test",
    ]
    assert d.entries["src/bar.py"].description == "New parser."
    assert d.invalid_paths == []
    assert d.duplicate_paths == []


def test_parse_plain_non_bold_variant():
    body = f"{MARKER}\n### File: src/x.go\n- Description: did a thing\n- Tests:\n    - go test ./...\n"
    d = policy_check.parse_testing_declarations(body)
    assert set(d.entries) == {"src/x.go"}
    assert d.entries["src/x.go"].tests == ["go test ./..."]


def test_parse_empty_tests_is_invalid():
    body = f"{MARKER}\n### File: `a.c`\n- **Description:** did stuff\n- **Tests:**\n"
    d = policy_check.parse_testing_declarations(body)
    assert d.entries == {}
    assert d.invalid_paths == ["a.c"]


def test_parse_empty_description_is_invalid():
    body = (
        f"{MARKER}\n### File: `a.c`\n- **Description:**\n- **Tests:**\n  - something\n"
    )
    d = policy_check.parse_testing_declarations(body)
    assert d.entries == {}
    assert d.invalid_paths == ["a.c"]


def test_parse_duplicate_paths_last_wins_and_flagged():
    body = (
        f"{MARKER}\n"
        "### File: `a.c`\n- **Description:** first\n- **Tests:**\n  - t1\n"
        "### File: `./a.c`\n- **Description:** second\n- **Tests:**\n  - t2\n"
    )
    d = policy_check.parse_testing_declarations(body)
    assert set(d.entries) == {"a.c"}
    assert d.entries["a.c"].description == "second"
    assert d.duplicate_paths == ["a.c"]


def test_ensure_docs_only_passes():
    policy = _load_policy()
    errors = []
    policy_check.ensure_unit_tests(
        policy, [_file("README.md"), _file("policy.yml")], "", errors
    )
    assert errors == []


def test_ensure_removed_code_file_ignored():
    policy = _load_policy()
    errors = []
    policy_check.ensure_unit_tests(
        policy, [_file("src/foo.cpp", status="removed")], "", errors
    )
    assert errors == []


def test_ensure_code_without_marker_fails():
    policy = _load_policy()
    errors = []
    policy_check.ensure_unit_tests(
        policy, [_file("src/foo.cpp")], "a description with no marker", errors
    )
    assert errors
    assert "No testing declaration" in errors[0]


def test_ensure_code_with_valid_declaration_passes():
    policy = _load_policy()
    body = (
        f"{MARKER}\n### File: `src/foo.cpp`\n"
        "- **Description:** did it\n- **Tests:**\n  - ctest -R foo\n"
    )
    errors = []
    policy_check.ensure_unit_tests(policy, [_file("src/foo.cpp")], body, errors)
    assert errors == []


def test_ensure_missing_entry_for_one_file_fails():
    policy = _load_policy()
    body = (
        f"{MARKER}\n### File: `src/foo.cpp`\n"
        "- **Description:** did it\n- **Tests:**\n  - ctest -R foo\n"
    )
    errors = []
    policy_check.ensure_unit_tests(
        policy, [_file("src/foo.cpp"), _file("src/bar.py")], body, errors
    )
    assert errors
    assert any("no testing declaration entry" in e.lower() for e in errors)


def test_ensure_incomplete_entry_reported_as_invalid():
    policy = _load_policy()
    body = (
        f"{MARKER}\n### File: `src/foo.cpp`\n- **Description:** did it\n- **Tests:**\n"
    )
    errors = []
    policy_check.ensure_unit_tests(policy, [_file("src/foo.cpp")], body, errors)
    assert errors
    assert any("incomplete" in e.lower() for e in errors)


def test_pr_has_code_files():
    policy = _load_policy()
    assert policy_check.pr_has_code_files(policy, [_file("src/foo.cpp")]) is True
    assert policy_check.pr_has_code_files(policy, [_file("README.md")]) is False
    assert (
        policy_check.pr_has_code_files(policy, [_file("src/foo.cpp", status="removed")])
        is False
    )
