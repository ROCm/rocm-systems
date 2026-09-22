# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for hipfile.error.HipFileException."""

# pylint: disable=missing-function-docstring,redefined-outer-name
# pylint: disable=import-error  # hipfile not installed in lint env; extension faked

from unittest import mock

import hipfile.error
from hipfile.enums import OpError
from hipfile.error import HipFileException


def test_stores_error_codes():
    exc = HipFileException(OpError.INVALID_VALUE, 7)
    assert exc.hipfile_err == OpError.INVALID_VALUE
    assert exc.hip_err == 7


def test_str_includes_op_error_string():
    with mock.patch.object(
        hipfile.error, "hipFileGetOpErrorString", return_value="boom"
    ) as get_str:
        exc = HipFileException(OpError.INVALID_VALUE, 0)
        text = str(exc)
    get_str.assert_called_once_with(OpError.INVALID_VALUE)
    assert str(int(OpError.INVALID_VALUE)) in text
    assert "boom" in text


def test_is_an_exception():
    assert issubclass(HipFileException, Exception)
