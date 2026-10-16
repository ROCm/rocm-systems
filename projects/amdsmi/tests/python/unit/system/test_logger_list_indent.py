#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Human-readable output: a scalar list item sits one level under its key at any depth.

The renderer doubles the indent of every key line, but a list item such as ``- FWCTL`` has no
colon and skipped that pass, so a nested list (``static --nic`` ``CAPABILITY``) stayed at a fixed
shallow column. The real methods are compiled from source because ``amdsmi_logger.py`` imports
the CLI helpers.
"""

import ast
import re
import unittest
from pathlib import Path
from typing import Any, Dict, List

from common.common import cli_search_order, find_cli_dir

_CLI_DIR = find_cli_dir(*cli_search_order(Path(__file__).resolve().parent))
_LOGGER_PATH = Path(_CLI_DIR or ".") / "amdsmi_logger.py"
_METHODS = ["_capitalize_keys", "_convert_json_to_human_readable", "custom_dump"]
_INDENT_STEP = 4
_TOP_LEVEL = 0
_ONE_DEVICE_LEVEL = 1
_DEEP_NESTING = 3


def _load_renderer() -> Any:
    if not _CLI_DIR:
        raise unittest.SkipTest("amdsmi_cli not found")
    tree = ast.parse(_LOGGER_PATH.read_text(encoding="utf-8"))
    logger_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AMDSMILogger"
    )
    methods = [
        node
        for node in logger_class.body
        if isinstance(node, ast.FunctionDef) and node.name in _METHODS
    ]
    missing = set(_METHODS) - {node.name for node in methods}
    if missing:
        raise AssertionError(f"not found in amdsmi_logger.py: {sorted(missing)}")
    renderer = ast.ClassDef(
        name="Renderer", bases=[], keywords=[], body=methods, decorator_list=[], type_params=[]
    )
    module = ast.fix_missing_locations(ast.Module([renderer], []))
    # The production annotation ``Dict[str, any]`` names the builtin ``any`` and is evaluated
    # when the method is defined, so the namespace has to provide it.
    namespace: Dict[str, Any] = {"re": re, "Dict": Dict, "any": object}
    exec(compile(module, str(_LOGGER_PATH), "exec"), namespace)
    return namespace["Renderer"]()


def _nested(depth: int, leaf: Dict[str, Any], device: str = "ai_nic") -> Dict[str, Any]:
    """``leaf`` wrapped in ``depth`` dictionaries under a ``device`` entry."""
    node = leaf
    for level in range(depth):
        node = {f"level_{level}": node}
    return {device: {"0": node}}


def _render(data: Dict[str, Any]) -> List[str]:
    return _load_renderer()._convert_json_to_human_readable(data).splitlines()


def _indent_of(lines: List[str], text: str) -> int:
    line = next(line for line in lines if line.strip() == text)
    return len(line) - len(line.lstrip(" "))


class TestListItemIndent(unittest.TestCase):
    def _assert_items_one_level_under_key(self, lines: List[str]) -> None:
        key_indent = _indent_of(lines, "CAPABILITY:")
        self.assertEqual(_indent_of(lines, "- FWCTL"), key_indent + _INDENT_STEP)
        self.assertEqual(_indent_of(lines, "- NETDEV"), key_indent + _INDENT_STEP)

    def test_items_sit_one_level_under_their_key_at_every_depth(self) -> None:
        for depth in (_TOP_LEVEL, _ONE_DEVICE_LEVEL, _DEEP_NESTING):
            with self.subTest(depth=depth):
                leaf = {"capability": ["FWCTL", "NETDEV"]}
                self._assert_items_one_level_under_key(_render(_nested(depth, leaf)))

    def test_items_are_indented_under_a_gpu_and_under_a_top_level_key(self) -> None:
        leaf = {"capability": ["FWCTL", "NETDEV"]}
        with self.subTest(device="gpu"):
            self._assert_items_one_level_under_key(_render(_nested(_ONE_DEVICE_LEVEL, leaf, "gpu")))
        with self.subTest(device="top level key"):
            self._assert_items_one_level_under_key(_render(leaf))

    def test_the_key_after_a_list_returns_to_the_key_level(self) -> None:
        lines = _render(_nested(_DEEP_NESTING, {"capability": ["FWCTL"], "after": 1}))
        self.assertEqual(_indent_of(lines, "AFTER: 1"), _indent_of(lines, "CAPABILITY:"))


class TestMultilineValueIsNotAListItem(unittest.TestCase):
    """A continuation line of a multiline value keeps the alignment ``custom_dump`` gave it."""

    def _assert_continuation_aligned(self, value: str, continuation: str) -> None:
        for depth in (_TOP_LEVEL, _ONE_DEVICE_LEVEL):
            with self.subTest(depth=depth):
                lines = _render(_nested(depth, {"accelerators": value}))
                first = next(line for line in lines if "ACCELERATORS:" in line)
                column = first.index(value.split("\n")[0])
                self.assertEqual(_indent_of(lines, continuation), column)

    def test_a_plain_continuation_keeps_its_alignment(self) -> None:
        self._assert_continuation_aligned("0, 1\n2, 3", "2, 3")

    def test_a_continuation_that_starts_with_a_dash_and_space_keeps_its_alignment(self) -> None:
        self._assert_continuation_aligned("0, 1\n- 2, 3", "- 2, 3")

    def test_a_continuation_that_starts_with_a_bare_dash_keeps_its_alignment(self) -> None:
        self._assert_continuation_aligned("0\n-5", "-5")
