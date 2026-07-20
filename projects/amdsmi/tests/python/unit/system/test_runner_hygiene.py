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
"""

import os
import re
import unittest

# tests/python root: .../unit/system/test_runner_hygiene.py -> .../tests/python
# (or the installed .../python_unittest). os.walk from here covers every suite.
_TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Matches a module-level ``if __name__ == "__main__":`` guard (either quote style).
_MAIN_GUARD = re.compile(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", re.MULTILINE)

# Top-level suite directory -> the runner script that discovers/runs it
# (see the common.run_test_dir callers).
_RUNNERS = {"unit": "unit_tests.py", "cli": "cli_unit_test.py", "functional": "integration_test.py"}


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


class TestRunnerHygiene(unittest.TestCase):
    def test_no_test_module_defines_standalone_main(self):
        offenders = []
        for root, _dirs, files in os.walk(_TESTS_ROOT):
            for fname in files:
                if not (fname.startswith("test_") and fname.endswith(".py")):
                    continue
                path = os.path.join(root, fname)
                try:
                    with open(path, encoding="utf-8") as handle:
                        source = handle.read()
                except OSError:
                    continue
                if _MAIN_GUARD.search(source):
                    offenders.append(os.path.relpath(path, _TESTS_ROOT))

        offenders.sort()
        self.assertEqual(offenders, [], _format_message(offenders))
