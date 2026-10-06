# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for hipfile.driver.Driver."""

# pylint: disable=missing-function-docstring,redefined-outer-name
# pylint: disable=import-error  # hipfile not installed in lint env; extension faked

from unittest import mock

import pytest

import hipfile.driver
from hipfile.driver import Driver
from hipfile.enums import OpError
from hipfile.error import HipFileException

_SUCCESS = (OpError.SUCCESS, 0)
_FAILURE = (OpError.DRIVER_NOT_INITIALIZED, 0)


def test_open_success():
    with mock.patch.object(
        hipfile.driver, "hipFileDriverOpen", autospec=True, return_value=_SUCCESS
    ) as drv_open:
        Driver().open()
    drv_open.assert_called_once_with()


def test_open_raises_on_error():
    with mock.patch.object(
        hipfile.driver, "hipFileDriverOpen", autospec=True, return_value=_FAILURE
    ):
        with pytest.raises(HipFileException) as excinfo:
            Driver().open()
    assert excinfo.value.hipfile_err == OpError.DRIVER_NOT_INITIALIZED


def test_close_success():
    with mock.patch.object(
        hipfile.driver, "hipFileDriverClose", autospec=True, return_value=_SUCCESS
    ) as drv_close:
        Driver().close()
    drv_close.assert_called_once_with()


def test_close_raises_on_error():
    with mock.patch.object(
        hipfile.driver, "hipFileDriverClose", autospec=True, return_value=_FAILURE
    ):
        with pytest.raises(HipFileException):
            Driver().close()


def test_use_count_delegates():
    with mock.patch.object(
        hipfile.driver, "hipFileUseCount", autospec=True, return_value=5
    ) as use_count:
        assert Driver.use_count() == 5
    use_count.assert_called_once_with()


def test_context_manager_opens_then_closes():
    with mock.patch.object(
        hipfile.driver, "hipFileDriverOpen", autospec=True, return_value=_SUCCESS
    ) as drv_open, mock.patch.object(
        hipfile.driver, "hipFileDriverClose", autospec=True, return_value=_SUCCESS
    ) as drv_close:
        with Driver() as drv:
            assert isinstance(drv, Driver)
            drv_open.assert_called_once_with()
            drv_close.assert_not_called()
        drv_close.assert_called_once_with()
