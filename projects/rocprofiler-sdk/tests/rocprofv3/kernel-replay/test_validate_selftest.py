#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Self-test for validate.py: does it actually reject a broken replay?

validate.py only ever sees output from a working implementation, so a check that
silently accepts everything looks identical to a check that works. This drives
validate.py's assertions over synthetic results -- one correct replay plus one
mutation per known failure mode -- and requires that the correct one is accepted
and every broken one is rejected.

Runs without a GPU or a ROCm install; the results JSON is constructed, not measured.
"""

import inspect

import pytest

import replay_fixtures as fixtures
import validate

# validate.py checks that take (json_data, ...) and are meaningful for synthetic data.
_ARGS = {
    "json_data": None,  # filled per call
    "expected_passes": fixtures.PASSES,
    "common_counters": fixtures.COMMON_COUNTERS,
    "pass_groups": fixtures.PASS_GROUPS,
}


def _checks():
    for name, fn in sorted(vars(validate).items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            yield name, fn


def _run_check(name, fn, doc):
    """Return None if the check accepted the document, else the assertion message."""
    kwargs = {}
    for param in inspect.signature(fn).parameters:
        kwargs[param] = doc if param == "json_data" else _ARGS[param]
    try:
        fn(**kwargs)
    except AssertionError as err:
        return f"{name}: {err}"
    return None


def _rejections(doc):
    return [msg for msg, _ in ((_run_check(n, f, doc), n) for n, f in _checks()) if msg]


def test_checks_were_discovered():
    names = [name for name, _ in _checks()]
    assert len(names) >= 7, f"expected validate.py to expose several checks, got {names}"
    assert (
        "test_pass_index_maps_to_requested_group" in names
    ), "the pass->group mapping check is missing from validate.py"


def test_correct_replay_is_accepted():
    rejections = _rejections(fixtures.golden())
    assert not rejections, (
        "validate.py rejected a correct replay, so the fixtures and the checks "
        f"disagree: {rejections}"
    )


@pytest.mark.parametrize("mode", sorted(fixtures.FAILURE_MODES))
def test_broken_replay_is_rejected(mode):
    description = fixtures.FAILURE_MODES[mode][0]
    rejections = _rejections(fixtures.broken(mode))
    assert rejections, (
        f"validate.py accepted a broken replay ({description}). Every check passed, "
        "so this failure mode would reach CI undetected."
    )


@pytest.mark.parametrize("mode", ["pass_groups_swapped", "pass_groups_skewed"])
def test_mapping_faults_need_the_mapping_check(mode):
    """These two are invisible to a union-of-counters check.

    They are the reason test_pass_index_maps_to_requested_group exists; if some other
    check starts catching them, this test documents that the coverage moved rather
    than silently overlapping.
    """
    doc = fixtures.broken(mode)
    rejections = _rejections(doc)
    assert any(
        "test_pass_index_maps_to_requested_group" in msg for msg in rejections
    ), f"{mode} was not caught by the mapping check; rejections={rejections}"
