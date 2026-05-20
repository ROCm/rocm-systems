# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for environment variable redaction in logger wrappers."""

import logging
from typing import Callable

import pytest

from utils.logger import (
    REDACTED_VALUE,
    TRACE_LEVEL,
    _redact_env_vars,
    console_debug,
    console_error,
    console_log,
    console_warning,
    trace_logger,
)


def test_redact_env_vars_keeps_allowlisted():
    message = (
        "HSA_TOOLS_LIB='libfoo.so' "
        "HIP_VISIBLE_DEVICES=0 "
        "ROCPROFILER_METRICS_PATH='/tmp/x'"
    )

    assert _redact_env_vars(message) == message


def test_redact_env_vars_redacts_secrets():
    message = (
        "{'ANTHROPIC_API_KEY': 'sk-abc', "
        "'PATH': '/usr/bin', "
        "'HSA_TOOLS_LIB': 'libfoo.so'}"
    )

    redacted_message = _redact_env_vars(message)

    assert REDACTED_VALUE in redacted_message
    assert "sk-abc" not in redacted_message
    assert "/usr/bin" not in redacted_message
    assert "'ANTHROPIC_API_KEY': '<redacted>'" in redacted_message
    assert "'PATH': '<redacted>'" in redacted_message
    assert "'HSA_TOOLS_LIB': 'libfoo.so'" in redacted_message


def test_redact_handles_unquoted_values():
    message = "setting HOME=/root and HSA_TOOLS_LIB=libfoo.so"

    redacted_message = _redact_env_vars(message)

    assert "HOME=<redacted>" in redacted_message
    assert "/root" not in redacted_message
    assert "HSA_TOOLS_LIB=libfoo.so" in redacted_message


@pytest.mark.parametrize(
    "logger_wrapper,level",
    [
        (console_log, logging.INFO),
        (console_debug, logging.DEBUG),
        (console_warning, logging.WARNING),
        (lambda message: console_error(message, exit=False), logging.ERROR),
    ],
)
def test_each_wrapper_redacts_via_caplog(
    caplog,
    logger_wrapper: Callable[[str], None],
    level: int,
):
    caplog.set_level(level)
    message = (
        "{'SECRET_TOKEN': 'abc', 'HSA_TOOLS_LIB': 'libfoo.so', 'PATH': '/usr/bin'}"
    )

    logger_wrapper(message)

    assert REDACTED_VALUE in caplog.text
    assert "abc" not in caplog.text
    assert "/usr/bin" not in caplog.text
    assert "libfoo.so" in caplog.text


def test_trace_logger_redacts_and_interpolates(caplog):
    caplog.set_level(TRACE_LEVEL)

    trace_logger(
        "HSA_TOOLS_LIB=%s SECRET_TOKEN=%s",
        "libfoo.so",
        "abc",
    )

    assert REDACTED_VALUE in caplog.text
    assert "SECRET_TOKEN=<redacted>" in caplog.text
    assert "abc" not in caplog.text
    assert "HSA_TOOLS_LIB=libfoo.so" in caplog.text
