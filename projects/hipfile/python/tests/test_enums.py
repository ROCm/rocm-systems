# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for hipfile.enums"""

# pylint: disable=missing-function-docstring
# pylint: disable=import-error  # hipfile not installed in lint env; extension faked

from hipfile.enums import OpError


def test_op_error_members_are_ints():
    # The high-level code compares error codes against raw ints returned by the
    # extension (e.g. ``err[0] != 0``), so members must behave as ints.
    assert isinstance(OpError.SUCCESS, int)
    assert OpError.SUCCESS == 0
