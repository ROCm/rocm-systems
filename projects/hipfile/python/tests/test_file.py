# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for hipfile.file.FileHandle."""

# pylint: disable=missing-function-docstring,redefined-outer-name,protected-access
# pylint: disable=import-error  # hipfile not installed in lint env; extension faked

import errno
import os
from unittest import mock

import pytest

import hipfile.file
from hipfile.enums import FileHandleType, OpError
from hipfile.error import HipFileException
from hipfile.file import FileHandle

_SUCCESS = (OpError.SUCCESS, 0)
_FAILURE = (OpError.HANDLE_ALREADY_REGISTERED, 0)
_FAKE_FD = 11
_FAKE_HANDLE = 0xF11E


# --- handle_type setter guards --------------------------------------------


def test_handle_type_rejects_non_member():
    fh = FileHandle("/tmp/x", os.O_RDONLY)
    with pytest.raises(ValueError):
        fh.handle_type = 9999


def test_handle_type_rejects_win32():
    fh = FileHandle("/tmp/x", os.O_RDONLY)
    with pytest.raises(NotImplementedError):
        fh.handle_type = FileHandleType.OPAQUE_WIN32


def test_handle_type_rejects_change_while_open():
    fh = FileHandle("/tmp/x", os.O_RDONLY)
    fh._handle = _FAKE_HANDLE  # simulate an open handle
    with pytest.raises(RuntimeError):
        fh.handle_type = FileHandleType.USERSPACE_FS


# --- open -----------------------------------------------------------------


def test_open_registers_and_stores_handle():
    fh = FileHandle("/tmp/x", os.O_RDONLY)
    with mock.patch.object(
        hipfile.file.os, "open", return_value=_FAKE_FD
    ), mock.patch.object(
        hipfile.file,
        "hipFileHandleRegister",
        autospec=True,
        return_value=(_FAKE_HANDLE, _SUCCESS),
    ) as register:
        fh.open()
    assert fh.handle == _FAKE_HANDLE
    register.assert_called_once_with(_FAKE_FD, FileHandleType.OPAQUE_FD)


def test_open_closes_fd_when_registration_fails():
    fh = FileHandle("/tmp/x", os.O_RDONLY)
    with mock.patch.object(
        hipfile.file.os, "open", return_value=_FAKE_FD
    ), mock.patch.object(hipfile.file.os, "close") as os_close, mock.patch.object(
        hipfile.file,
        "hipFileHandleRegister",
        autospec=True,
        return_value=(0, _FAILURE),
    ):
        with pytest.raises(HipFileException):
            fh.open()
    os_close.assert_called_once_with(_FAKE_FD)
    assert fh.handle is None


def test_open_raises_when_already_open():
    fh = FileHandle("/tmp/x", os.O_RDONLY)
    fh._handle = _FAKE_HANDLE
    with pytest.raises(RuntimeError):
        fh.open()


# --- close ----------------------------------------------------------------


def test_close_deregisters_and_closes():
    fh = FileHandle("/tmp/x", os.O_RDONLY)
    fh._handle = _FAKE_HANDLE
    fh._fd = _FAKE_FD
    with mock.patch.object(
        hipfile.file, "hipFileHandleDeregister", autospec=True
    ) as dereg, mock.patch.object(hipfile.file.os, "close") as os_close:
        fh.close()
    dereg.assert_called_once_with(_FAKE_HANDLE)
    os_close.assert_called_once_with(_FAKE_FD)
    assert fh.handle is None
    assert fh._fd is None


def test_close_on_never_opened_is_noop():
    fh = FileHandle("/tmp/x", os.O_RDONLY)
    with mock.patch.object(
        hipfile.file, "hipFileHandleDeregister", autospec=True
    ) as dereg, mock.patch.object(hipfile.file.os, "close") as os_close:
        fh.close()
    dereg.assert_not_called()
    os_close.assert_not_called()


def test_close_is_idempotent_after_open():
    fh = FileHandle("/tmp/x", os.O_RDONLY)
    fh._handle = _FAKE_HANDLE
    fh._fd = _FAKE_FD
    with mock.patch.object(
        hipfile.file, "hipFileHandleDeregister", autospec=True
    ) as dereg, mock.patch.object(hipfile.file.os, "close") as os_close:
        fh.close()
        fh.close()
    # Cleanup runs exactly once despite the repeated call.
    dereg.assert_called_once_with(_FAKE_HANDLE)
    os_close.assert_called_once_with(_FAKE_FD)


# --- read / write ---------------------------------------------------------


def _open_handle():
    fh = FileHandle("/tmp/x", os.O_RDONLY)
    fh._handle = _FAKE_HANDLE
    return fh


@pytest.mark.parametrize("method", ["read", "write"])
def test_io_raises_when_not_open(method, fake_buffer):
    fh = FileHandle("/tmp/x", os.O_RDONLY)
    with pytest.raises(RuntimeError):
        getattr(fh, method)(fake_buffer, 64, 0, 0)


@pytest.mark.parametrize("method", ["read", "write"])
def test_io_success_returns_bytecount(method, fake_buffer):
    fh = _open_handle()
    wrapper = "hipFileRead" if method == "read" else "hipFileWrite"
    with mock.patch.object(
        hipfile.file, wrapper, autospec=True, return_value=(64, 0)
    ) as io_call:
        result = getattr(fh, method)(fake_buffer, 64, 8, 16)
    assert result == 64
    io_call.assert_called_once_with(_FAKE_HANDLE, fake_buffer.ptr, 64, 8, 16)


@pytest.mark.parametrize("method", ["read", "write"])
def test_io_system_error_raises_oserror(method, fake_buffer):
    fh = _open_handle()
    wrapper = "hipFileRead" if method == "read" else "hipFileWrite"
    with mock.patch.object(
        hipfile.file, wrapper, autospec=True, return_value=(-1, errno.EIO)
    ):
        with pytest.raises(OSError) as excinfo:
            getattr(fh, method)(fake_buffer, 64, 0, 0)
    assert excinfo.value.errno == errno.EIO


@pytest.mark.parametrize("method", ["read", "write"])
def test_io_hipfile_error_raises_exception(method, fake_buffer):
    fh = _open_handle()
    wrapper = "hipFileRead" if method == "read" else "hipFileWrite"
    # result < -1 encodes a negated hipFileOpError_t with extra_err == 0.
    op_err = OpError.INVALID_VALUE
    with mock.patch.object(
        hipfile.file, wrapper, autospec=True, return_value=(-int(op_err), 0)
    ):
        with pytest.raises(HipFileException) as excinfo:
            getattr(fh, method)(fake_buffer, 64, 0, 0)
    assert excinfo.value.hipfile_err == op_err
    assert excinfo.value.hip_err == 0


@pytest.mark.parametrize("method", ["read", "write"])
def test_io_hip_driver_error_carries_hip_err(method, fake_buffer):
    fh = _open_handle()
    wrapper = "hipFileRead" if method == "read" else "hipFileWrite"
    op_err = OpError.HIP_DRIVER_ERROR
    with mock.patch.object(
        hipfile.file, wrapper, autospec=True, return_value=(-int(op_err), 99)
    ):
        with pytest.raises(HipFileException) as excinfo:
            getattr(fh, method)(fake_buffer, 64, 0, 0)
    assert excinfo.value.hipfile_err == op_err
    assert excinfo.value.hip_err == 99
