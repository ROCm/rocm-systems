#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""Host tests for init-pipeline CLI validation (v11 CR-5: positive GO timeout)."""

import os
import sys

_RUNNER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER_ROOT not in sys.path:
    sys.path.insert(0, _RUNNER_ROOT)

from lib.test_parser import ArgumentParserInterface  # noqa: E402


def _args(argv):
    p = ArgumentParserInterface()
    p.add_arguments()
    return p, p.parser.parse_args(argv)


def test_go_timeout_zero_rejected_for_init_pipeline():
    p, a = _args(["-c", "x.json", "--exec-mode", "init-pipeline", "--go-timeout", "0"])
    errs = p.validate_init_pipeline(a)
    assert any("go-timeout" in e for e in errs)


def test_positive_go_timeout_ok():
    p, a = _args(["-c", "x.json", "--exec-mode", "init-pipeline", "--go-timeout", "120"])
    assert p.validate_init_pipeline(a) == []


def test_indefinite_override_allows_zero():
    p, a = _args(["-c", "x.json", "--exec-mode", "init-pipeline",
                  "--go-timeout", "0", "--allow-indefinite-go-wait"])
    assert p.validate_init_pipeline(a) == []


def test_serial_mode_no_go_timeout_requirement():
    p, a = _args(["-c", "x.json", "--exec-mode", "serial"])
    assert p.validate_init_pipeline(a) == []


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
