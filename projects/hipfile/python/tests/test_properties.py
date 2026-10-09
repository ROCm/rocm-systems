# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for hipfile.properties: get_version and driver_get_properties."""

# pylint: disable=missing-function-docstring,redefined-outer-name
# pylint: disable=import-error  # hipfile not installed in lint env; extension faked

from unittest import mock

import pytest

import hipfile.properties
from hipfile.enums import OpError
from hipfile.error import HipFileException
from hipfile.properties import driver_get_properties, get_version

_SUCCESS = (OpError.SUCCESS, 0)
_FAILURE = (OpError.DRIVER_NOT_INITIALIZED, 0)


def test_get_version_success():
    with mock.patch.object(
        hipfile.properties,
        "hipFileGetVersion",
        autospec=True,
        return_value=((4, 5, 6), _SUCCESS),
    ) as get_ver:
        assert get_version() == (4, 5, 6)
    get_ver.assert_called_once_with()


def test_get_version_raises_on_error():
    with mock.patch.object(
        hipfile.properties,
        "hipFileGetVersion",
        autospec=True,
        return_value=((0, 0, 0), _FAILURE),
    ):
        with pytest.raises(HipFileException):
            get_version()


# Note that test properties are mocked and may not reflect all properties
# that are tracked by the C library.
def test_driver_get_properties_success():
    props = {"feature_flags": 3, "max_batch_io_count": 7}
    with mock.patch.object(
        hipfile.properties,
        "hipFileDriverGetProperties",
        autospec=True,
        return_value=(props, _SUCCESS),
    ) as get_props:
        assert driver_get_properties() == props
    get_props.assert_called_once_with()


def test_driver_get_properties_raises_on_error():
    with mock.patch.object(
        hipfile.properties,
        "hipFileDriverGetProperties",
        autospec=True,
        return_value=({}, _FAILURE),
    ):
        with pytest.raises(HipFileException):
            driver_get_properties()
