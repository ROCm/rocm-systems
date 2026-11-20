#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runner test selection with ``-k``/``--keyword`` and ``-x``/``--exclude``, and its help.

Each selection case lists a throwaway suite through ``common.run_test_dir``, so it sees the
selection a runner script makes for those arguments.
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from typing import List
from unittest import mock

from common import common

_PACKAGE = "runner_filter_fixture"
_ONE = f"{_PACKAGE}.test_alpha.TestAlpha.test_one"
_TWO = f"{_PACKAGE}.test_alpha.TestAlpha.test_two"
_SOURCES = {
    "__init__.py": "",
    "test_alpha.py": (
        "import unittest\n\n\n"
        "class TestAlpha(unittest.TestCase):\n"
        "    def test_one(self):\n"
        "        pass\n\n"
        "    def test_two(self):\n"
        "        pass\n"
    ),
    "test_broken.py": "import a_module_that_does_not_exist\n",
    "test_skipped.py": "import unittest\n\nraise unittest.SkipTest('not on this host')\n",
}


class TestRunnerFilters(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        package = os.path.join(cls._tmp.name, _PACKAGE)
        os.mkdir(package)
        for name, source in _SOURCES.items():
            with open(os.path.join(package, name), "w", encoding="utf-8") as handle:
                handle.write(source)
        # What discovery lists for the module that fails to import and the one that skips.
        cls.unloaded = [test_id for test_id in cls._list() if not test_id.startswith(_PACKAGE)]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    @classmethod
    def _list(cls, *args: str) -> List[str]:
        """Test ids ``run_test_dir`` lists for the runner arguments *args*."""
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", ["runner.py", "--list", *args]))
            # Discovery adds the fixture to sys.path and sys.modules; both are restored on exit.
            stack.enter_context(mock.patch.object(sys, "path", list(sys.path)))
            stack.enter_context(mock.patch.dict(sys.modules))
            stack.enter_context(contextlib.redirect_stdout(out))
            try:
                common.run_test_dir(_PACKAGE, "Runner filter fixture", cls._tmp.name)
            except SystemExit as stop:
                if stop.code != 0:
                    raise AssertionError(f"--list exited with {stop.code}")
        return [line.strip() for line in out.getvalue().splitlines()[1:]]

    def test_list_shows_every_test_and_each_module_that_did_not_load(self) -> None:
        self.assertEqual(len(self.unloaded), 2)
        self.assertEqual(self._list(), [_ONE, _TWO] + self.unloaded)

    def test_keyword_matches_anywhere_in_the_id(self) -> None:
        for pattern in ("test_alpha", "TestAlpha"):
            with self.subTest(pattern=pattern):
                self.assertEqual(self._list("-k", pattern), [_ONE, _TWO] + self.unloaded)

    def test_keyword_wildcards_match_as_a_glob(self) -> None:
        for pattern in ("Alpha.test_t*", "test_tw?", "test_[st]wo"):
            with self.subTest(pattern=pattern):
                self.assertEqual(self._list("-k", pattern), [_TWO] + self.unloaded)

    def test_keyword_never_hides_a_module_that_did_not_load(self) -> None:
        self.assertEqual(self._list("-k", "matches_no_test"), self.unloaded)

    def test_keyword_spellings_select_the_same_tests(self) -> None:
        for args in (
            ["-k", "test_one"],
            ["-ktest_one"],
            ["--keyword", "test_one"],
            ["--keyword=test_one"],
        ):
            with self.subTest(args=args):
                self.assertEqual(self._list(*args), [_ONE] + self.unloaded)

    def test_exclude_spellings_drop_the_same_tests(self) -> None:
        for args in (
            ["-x", "test_one"],
            ["-xtest_one"],
            ["--exclude", "test_one"],
            ["--exclude=test_one"],
        ):
            with self.subTest(args=args):
                self.assertEqual(self._list(*args), [_TWO] + self.unloaded)

    def test_exclude_applies_on_top_of_keyword(self) -> None:
        self.assertEqual(self._list("-k", "test_alpha", "-x", "test_two"), [_ONE] + self.unloaded)
        self.assertEqual(self._list("-k", "test_tw", "-x", "test_two"), self.unloaded)

    def test_exclude_takes_no_wildcards(self) -> None:
        self.assertEqual(self._list("-x", "test_t*"), [_ONE, _TWO] + self.unloaded)

    def test_help_leaves_keyword_to_the_runner(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            common.print_unittest_help()
        common.print_amdsmi_path_help(file=out)
        self.assertIn("--buffer", out.getvalue())
        self.assertNotIn("TESTNAMEPATTERNS", out.getvalue())
        self.assertIn("-k, --keyword PAT", out.getvalue())
