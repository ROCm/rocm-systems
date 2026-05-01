# pylint: disable=C0114,C0115,C0116
import os
import stat
from sys import stderr

from hipfile._hipfile import (  # pylint: disable=E0401,E0611
    hipFileHandleRegister,
    hipFileHandleDeregister,
    hipFileRead,
    hipFileWrite,
)
from hipfile.enums import FileHandleType
from hipfile.error import HipFileException


class FileHandle:
    DEFAULT_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH

    def __init__(
        self, path, flags, mode=DEFAULT_MODE, handle_type=FileHandleType.OPAQUE_FD
    ):
        self._fd = None
        self._flags = flags
        self._handle = None
        self._handle_type = None
        self._mode = mode
        self._path = path

        self.handle_type = handle_type

    def __del__(self):
        try:
            self.close()
        except Exception:  # pylint: disable=W0718  # Suppress exceptions in a dtor
            print(
                "Failed to deregister hipFile.FileHandle at destruction time.",
                file=stderr,
            )

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def flags(self):
        return self._flags

    @property
    def handle(self):
        return self._handle

    @property
    def handle_type(self):
        return self._handle_type

    @handle_type.setter
    def handle_type(self, _handle_type):
        if self._handle is not None:
            raise RuntimeError("Cannot modify handle_type while FileHandle is open")
        if _handle_type not in FileHandleType:
            raise ValueError(f"'{_handle_type}' is not a member of enum FileHandleType")
        if _handle_type == FileHandleType.OPAQUE_WIN32:
            raise NotImplementedError(
                "FileHandle does not currently support Win32 Handles"
            )
        self._handle_type = _handle_type

    @property
    def mode(self):
        return self._mode

    @property
    def path(self):
        return self._path

    def open(self):
        if self._handle is not None:
            raise RuntimeError("The FileHandle is already open.")
        self._fd = os.open(self._path, self._flags, self._mode)
        handle, err = hipFileHandleRegister(self._fd, self._handle_type)
        if err[0] != 0:
            os.close(self._fd)
            self._fd = None
            raise HipFileException(err[0], err[1])
        self._handle = handle

    def close(self):
        if self._handle is not None:
            hipFileHandleDeregister(self._handle)
            self._handle = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def read(self, buffer, size, file_offset, buffer_offset):
        if self._handle is None:
            raise RuntimeError("The FileHandle is not open.")
        bytes_read, extra_err = hipFileRead(
            self._handle, buffer.ptr, size, file_offset, buffer_offset
        )
        if bytes_read == -1:
            # extra_err is errno
            raise OSError(extra_err, os.strerror(extra_err))
        if bytes_read < -1:
            # hipFile Error
            # If -bytes_read == OpError.HIP_DRIVER_ERROR, extra_err is hipError_t.
            # Otherwise, extra_err is 0.
            raise HipFileException(-bytes_read, extra_err)
        return bytes_read

    def write(self, buffer, size, file_offset, buffer_offset):
        if self._handle is None:
            raise RuntimeError("The FileHandle is not open.")
        bytes_written, extra_err = hipFileWrite(
            self._handle, buffer.ptr, size, file_offset, buffer_offset
        )
        if bytes_written == -1:
            # extra_err is errno
            raise OSError(extra_err, os.strerror(extra_err))
        if bytes_written < -1:
            # hipFile Error
            # If -bytes_written == OpError.HIP_DRIVER_ERROR, extra_err is hipError_t.
            # Otherwise, extra_err is 0.
            raise HipFileException(-bytes_written, extra_err)
        return bytes_written
