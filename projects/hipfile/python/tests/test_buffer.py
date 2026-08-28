# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for hipfile.buffer.Buffer."""

# pylint: disable=missing-function-docstring,redefined-outer-name,protected-access
# pylint: disable=import-error  # hipfile not installed in lint env; extension faked

from unittest import mock

import pytest

import hipfile.buffer
from hipfile.buffer import Buffer
from hipfile.enums import OpError
from hipfile.error import HipFileException

_SUCCESS = (OpError.SUCCESS, 0)
_FAILURE = (OpError.MEMORY_ALREADY_REGISTERED, 0)


def test_from_ctypes_void_p_rejects_null(fake_void_p):
    with pytest.raises(ValueError):
        Buffer.from_ctypes_void_p(fake_void_p(None), 1024, 0)


def test_from_ctypes_void_p_forwards_value(fake_void_p):
    buf = Buffer.from_ctypes_void_p(fake_void_p(0xDEAD), 1024, 0)
    assert buf.ptr == 0xDEAD


def test_register_success():
    buf = Buffer(0x1000, 2048, 0)
    with mock.patch.object(
        hipfile.buffer, "hipFileBufRegister", autospec=True, return_value=_SUCCESS
    ) as reg:
        buf.register()
    reg.assert_called_once_with(0x1000, 2048, 0)
    assert buf._registered is True


def test_register_raises_on_error():
    buf = Buffer(0x1000, 2048, 0)
    with mock.patch.object(
        hipfile.buffer, "hipFileBufRegister", autospec=True, return_value=_FAILURE
    ):
        with pytest.raises(HipFileException):
            buf.register()
    assert buf._registered is False


def test_deregister_success():
    buf = Buffer(0x1000, 2048, 0)
    with mock.patch.object(
        hipfile.buffer, "hipFileBufRegister", autospec=True, return_value=_SUCCESS
    ):
        buf.register()
    with mock.patch.object(
        hipfile.buffer, "hipFileBufDeregister", autospec=True, return_value=_SUCCESS
    ) as dereg:
        buf.deregister()
    dereg.assert_called_once_with(0x1000)
    assert buf._registered is False


def test_deregister_is_noop_when_not_registered():
    buf = Buffer(0x1000, 2048, 0)
    with mock.patch.object(
        hipfile.buffer, "hipFileBufDeregister", autospec=True, return_value=_SUCCESS
    ) as dereg:
        buf.deregister()
    dereg.assert_not_called()


def test_deregister_is_idempotent_after_register():
    buf = Buffer(0x1000, 2048, 0)
    with mock.patch.object(
        hipfile.buffer, "hipFileBufRegister", autospec=True, return_value=_SUCCESS
    ):
        buf.register()
    with mock.patch.object(
        hipfile.buffer, "hipFileBufDeregister", autospec=True, return_value=_SUCCESS
    ) as dereg:
        buf.deregister()
        buf.deregister()
    # Second call is a no-op: the extension is only invoked once.
    dereg.assert_called_once_with(0x1000)


def test_deregister_raises_on_error():
    buf = Buffer(0x1000, 2048, 0)
    with mock.patch.object(
        hipfile.buffer, "hipFileBufRegister", autospec=True, return_value=_SUCCESS
    ):
        buf.register()
    with mock.patch.object(
        hipfile.buffer, "hipFileBufDeregister", autospec=True, return_value=_FAILURE
    ):
        with pytest.raises(HipFileException):
            buf.deregister()


def test_context_manager_registers_then_deregisters():
    buf = Buffer(0x1000, 2048, 0)
    with mock.patch.object(
        hipfile.buffer, "hipFileBufRegister", autospec=True, return_value=_SUCCESS
    ) as reg, mock.patch.object(
        hipfile.buffer, "hipFileBufDeregister", autospec=True, return_value=_SUCCESS
    ) as dereg:
        with buf as entered:
            assert entered is buf
            reg.assert_called_once_with(0x1000, 2048, 0)
            dereg.assert_not_called()
        dereg.assert_called_once_with(0x1000)
