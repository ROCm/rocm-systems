#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""Host tests for the binary-backed gtest filter preflight (v11 CR-1)."""

import os
import sys

import pytest

_RUNNER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER_ROOT not in sys.path:
    sys.path.insert(0, _RUNNER_ROOT)

from lib.gtest_preflight import (  # noqa: E402
    gtest_filter_matches,
    is_wildcard_filter,
    matching_tests,
    parse_gtest_list_tests,
    preflight_filter,
)

_LIST = """\
Running main() from gtest_main.cc
AllReduce.
  InPlace
  OutOfPlace
SendRecv.
  Basic  # GetParam() = 8-byte
ParamSuite/Foo.
  Case/0  # GetParam() = (1, 2)
  Case/1  # GetParam() = (3, 4)
"""


def test_parse_list_tests():
    fq = parse_gtest_list_tests(_LIST)
    assert "AllReduce.InPlace" in fq and "AllReduce.OutOfPlace" in fq
    assert "SendRecv.Basic" in fq                     # value-param comment stripped
    assert "ParamSuite/Foo.Case/0" in fq
    assert "Running" not in " ".join(fq)              # banner ignored


def test_is_wildcard_filter():
    for f in (None, "", "*", "ALL", "all", "*:*", " * "):
        assert is_wildcard_filter(f), f
    assert not is_wildcard_filter("AllReduce.OutOfPlace")
    assert not is_wildcard_filter("AllReduce.*")      # bounded to a suite -> not whole-binary


def test_gtest_filter_semantics():
    assert gtest_filter_matches("AllReduce.OutOfPlace", "AllReduce.OutOfPlace")
    assert gtest_filter_matches("AllReduce.OutOfPlace", "AllReduce.*")
    assert not gtest_filter_matches("SendRecv.Basic", "AllReduce.*")
    assert gtest_filter_matches("AllReduce.InPlace", "AllReduce.*:-*OutOfPlace")
    assert not gtest_filter_matches("AllReduce.OutOfPlace", "AllReduce.*:-*OutOfPlace")


def test_matching_tests():
    fq = parse_gtest_list_tests(_LIST)
    assert matching_tests(fq, "AllReduce.OutOfPlace") == ["AllReduce.OutOfPlace"]
    assert len(matching_tests(fq, "AllReduce.*")) == 2


def _fake_runner(output):
    def run(bp, env, timeout):
        # The preflight must strip pipeline env before invoking the binary.
        assert "RCCL_TEST_WARMUP_PROFILE" not in env
        assert "RCCL_TEST_READY_GO" not in env
        return output
    return run


def test_preflight_exactly_one_ok(tmp_path):
    b = tmp_path / "rccl-UnitTests"
    b.write_text("x")
    got = preflight_filter(str(b), "AllReduce.OutOfPlace",
                           env={"RCCL_TEST_WARMUP_PROFILE": "fork_coll", "RCCL_TEST_READY_GO": "1"},
                           runner=_fake_runner(_LIST))
    assert got == ["AllReduce.OutOfPlace"]


def test_preflight_zero_match_fatal(tmp_path):
    b = tmp_path / "rccl-UnitTests"; b.write_text("x")
    with pytest.raises(ValueError):
        preflight_filter(str(b), "Nope.Missing", runner=_fake_runner(_LIST))


def test_preflight_multi_match_fatal_for_gate(tmp_path):
    b = tmp_path / "rccl-UnitTests"; b.write_text("x")
    with pytest.raises(ValueError):
        preflight_filter(str(b), "AllReduce.*", curated=True, runner=_fake_runner(_LIST))
    # non-curated allows multi
    assert len(preflight_filter(str(b), "AllReduce.*", curated=False, runner=_fake_runner(_LIST))) == 2


def test_preflight_wildcard_fatal(tmp_path):
    b = tmp_path / "rccl-UnitTests"; b.write_text("x")
    with pytest.raises(ValueError):
        preflight_filter(str(b), "*", runner=_fake_runner(_LIST))


def test_list_tests_cached_by_identity(tmp_path):
    from lib.gtest_preflight import list_tests
    b = tmp_path / "rccl-UnitTests"; b.write_text("x")
    calls = {"n": 0}

    def counting(bp, env, timeout):
        calls["n"] += 1
        return _LIST

    cache = {}
    list_tests(str(b), cache=cache, runner=counting)
    list_tests(str(b), cache=cache, runner=counting)
    assert calls["n"] == 1  # second call served from cache


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
