#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runner hygiene: leaf tests must run only through their suite runner.

Executing a test file standalone -- via an ``if __name__ == "__main__":``
``unittest.main()`` block -- bypasses the shared runner (``common.run_test_dir``):
the root-privilege check, the ``sys.path`` / CLI resolution, the GTest-style
summary, and (most importantly) the ``sys.modules`` isolation guard that stops
one test class from polluting another. Every leaf test must instead be run
through its suite runner with a ``-k`` / ``-x`` filter, e.g.::

    unit_tests.py       -k "TestClass"               -v    # unit/
    cli_unit_test.py    -k "test_some_cli_behavior"  -v    # cli/
    integration_test.py -k "TestDiscovery"           -v    # functional/

This meta-test fails if any ``test_*.py`` module ships a standalone ``__main__``
runner, and names the runner that owns it.

It also fails if a module builds a CLI path out of hardcoded ``amdsmi_cli`` /
``libexec`` components instead of calling ``common.find_cli_dir``. Hand-rolled
resolution silently skips the suite in whichever layout it was not written for,
and typically prefers the source tree over ``AMDSMI_PATH``, so an explicit
override is ignored.

Finally it fails if a module installs a ``sys.modules`` stub without going
through ``common.stub_modules``. The runner's isolation guard already catches
leaks at runtime, but only for stubs a running class installs after the baseline
is taken -- a class that skips, or a stub installed at import time, is invisible
to it. This check covers both.
"""

import os
import re
import unittest

# tests/python root: .../unit/system/test_runner_hygiene.py -> .../tests/python
# (or the installed .../python_unittest). os.walk from here covers every suite.
_TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Matches a module-level ``if __name__ == "__main__":`` guard (either quote style).
_MAIN_GUARD = re.compile(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", re.MULTILINE)

# A quoted CLI path component -- the building blocks of a hand-rolled resolver.
# The alternation is not itself quote-wrapped, so this file does not self-match.
_CLI_PATH_LITERAL = re.compile(r"""['"](?:amdsmi_cli|libexec)['"]""")

# A call to the shared resolver, which is what the literals above should be.
_FIND_CLI_DIR_CALL = re.compile(r"\bfind_cli_dir\s*\(")

# Installing a stub over a known module name -- the direction that leaks.
# Requires a literal key: ``sys.modules[spec.name] = mod`` is the importlib
# idiom for loading a real module from a path, not a stub, so it is not matched.
# Popping a name is cleanup, so it is not matched either.
_SYS_MODULES_ASSIGN = re.compile(r"""^\s*sys\.modules\[\s*['"][^'"]+['"]\s*\]\s*=""", re.MULTILINE)

# The shared save/restore helpers that every stub install should go through:
# ``stub_modules`` (class-scoped) or ``stub_modules_at_import`` (module-scoped,
# restored from tearDownModule).
_STUB_MODULES_CALL = re.compile(r"\bstub_modules(?:_at_import)?\s*\(")

# Top-level suite directory -> the runner script that discovers/runs it
# (see the common.run_test_dir callers).
_RUNNERS = {"unit": "unit_tests.py", "cli": "cli_unit_test.py", "functional": "integration_test.py"}

# TODO(amdsmi_team): most of these tests are packaging/ABI related - they should be under
# their own category/runner (will update later..)
_STANDALONE_ALLOWED = frozenset(
    {
        "test_abi_compat.py",
        "test_dual_copy_guard.py",
        "test_cpack_path_guard.py",
        "run_amdsmi_python_versions_test.py",
        "test_packaging_scriptlets.py",
        "test_upgrade_downgrade_guard.py",
    }
)


def _runner_for(rel_path):
    """Return the runner script that owns the suite containing *rel_path*
    (a path relative to the tests root)."""
    top = rel_path.split(os.sep)[0]
    return _RUNNERS.get(top, "the applicable runner")


# --- Failure-message pieces (edit these to change the wording/format) ---------
# Header: the "what/why" shown once. Keep it self-contained.
_MESSAGE_HEADER = (
    'These test modules define a standalone `if __name__ == "__main__"` runner.\n\n'
    "Running a test file directly bypasses the suite runner (root check,\n"
    "sys.path/CLI resolution, the GTest summary, and the sys.modules isolation\n"
    "guard), so it is not allowed. Remove the block and run the suite through its\n"
    'runner with a filter instead: `<runner>.py -k "<filter>" -v` (or `-x` to exclude).'
)


def _offender_block(rel_path, runner_width):
    """One indented block per offending file. ``runner_width`` pads the runner
    name so the trailing ``(see ...)`` hint lines up across offenders. Edit the
    template here to restyle every entry at once."""
    runner = _runner_for(rel_path)
    return (
        f"  [*] {rel_path}\n"
        f'      Run via: {runner:<{runner_width}} -k "<filter>" -v  '
        f"(see `{runner} -h` for more options)\n"
    )


def _format_message(offenders):
    """Assemble the full assertion message from the header and one block per
    offender."""
    # Pad every runner name to the widest one so the "(see ...)" hints align.
    runner_width = max((len(_runner_for(rel)) for rel in offenders), default=0)
    blocks = "\n".join(_offender_block(rel, runner_width) for rel in offenders)
    return f"{_MESSAGE_HEADER}\n\nOffenders:\n{blocks}"


_CLI_PATH_MESSAGE_HEADER = (
    "These test modules build a CLI path from hardcoded components instead of\n"
    "calling `common.find_cli_dir`.\n\n"
    "The CLI lives at `<rocm>/libexec/amdsmi_cli` when installed and at\n"
    "`projects/amdsmi/amdsmi_cli` in a source checkout, so a path written for one\n"
    "layout resolves to nothing in the other and the suite skips itself instead of\n"
    "failing. A hand-rolled resolver that probes the source tree first also ignores\n"
    "AMDSMI_PATH, which is the whole point of the override.\n\n"
    "Resolve the directory once, at module scope, and skip only when it is None:\n\n"
    "    from common.common import amdsmi_path, find_cli_dir\n\n"
    "    _CLI_DIR = find_cli_dir(amdsmi_path, os.path.dirname(os.path.abspath(__file__)))\n"
    '    SOME_PATH = os.path.join(_CLI_DIR, "subcommands", "static.py") if _CLI_DIR else None\n\n'
    "    if not SOME_PATH or not os.path.isfile(SOME_PATH):\n"
    "        raise unittest.SkipTest(\n"
    '            f"amd-smi CLI static.py not found (looked in {_CLI_DIR or amdsmi_path})"\n'
    "        )"
)


def _format_cli_path_message(offenders):
    blocks = "\n".join(f"  [*] {rel}" for rel in offenders)
    return f"{_CLI_PATH_MESSAGE_HEADER}\n\nOffenders:\n{blocks}\n"


_STUB_MESSAGE_HEADER = (
    "These test modules assign into `sys.modules` without going through\n"
    "`common.stub_modules`.\n\n"
    "A stub that outlives its class corrupts every later test module in the same\n"
    "interpreter -- an empty AMDSMIHelpers stub, for example, hides handle_gpus\n"
    "from test_cli_exit_codes. The runner's isolation guard catches most of these\n"
    "at runtime, but not all: a class that skips never installs its stub, and an\n"
    "import-time stub is already in the guard's baseline, so neither is reported.\n"
    "A static check is what closes those two gaps.\n\n"
    "stub_modules snapshots the names, installs the replacements, and restores\n"
    "them when the class finishes:\n\n"
    "    from common.common import stub_modules\n\n"
    "    @classmethod\n"
    "    def setUpClass(cls):\n"
    "        modules = _fake_modules()          # name -> module (None removes)\n"
    "        stub_modules(cls, modules)\n\n"
    "If the stub has to exist before this module's own imports run, use\n"
    "`stub_modules_at_import(modules)` and call the returned restore callable\n"
    "from `tearDownModule`."
)


def _format_stub_message(offenders):
    blocks = "\n".join(f"  [*] {rel}" for rel in offenders)
    return f"{_STUB_MESSAGE_HEADER}\n\nOffenders:\n{blocks}\n"


def _iter_test_sources():
    """Yield ``(rel_path, source)`` for every ``test_*.py`` under the tests root."""
    for root, _dirs, files in os.walk(_TESTS_ROOT):
        for fname in sorted(files):
            if not (fname.startswith("test_") and fname.endswith(".py")):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, encoding="utf-8") as handle:
                    source = handle.read()
            except OSError:
                continue
            yield os.path.relpath(path, _TESTS_ROOT), source


class TestRunnerHygiene(unittest.TestCase):
    def test_no_test_module_defines_standalone_main(self):
        offenders = []
        for rel_path, source in _iter_test_sources():
            if _MAIN_GUARD.search(source) and rel_path not in _STANDALONE_ALLOWED:
                offenders.append(rel_path)

        offenders.sort()
        self.assertEqual(offenders, [], _format_message(offenders))

    def test_cli_paths_resolve_through_find_cli_dir(self):
        this_file = os.path.basename(__file__)
        offenders = []
        for rel_path, source in _iter_test_sources():
            if os.path.basename(rel_path) == this_file:
                continue
            if _CLI_PATH_LITERAL.search(source) and not _FIND_CLI_DIR_CALL.search(source):
                offenders.append(rel_path)

        offenders.sort()
        self.assertEqual(offenders, [], _format_cli_path_message(offenders))

    def test_module_stubs_go_through_stub_modules(self):
        this_file = os.path.basename(__file__)
        offenders = []
        for rel_path, source in _iter_test_sources():
            if os.path.basename(rel_path) == this_file:
                continue
            if _SYS_MODULES_ASSIGN.search(source) and not _STUB_MODULES_CALL.search(source):
                offenders.append(rel_path)

        offenders.sort()
        self.assertEqual(offenders, [], _format_stub_message(offenders))
