# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""GPU-accelerated file I/O through hipFile."""

from __future__ import annotations

import os
import stat
from sys import stderr
from typing import TYPE_CHECKING

from hipfile._hipfile import (  # pylint: disable=E0401,E0611
    hipFileHandleRegister,
    hipFileHandleDeregister,
    hipFileRead,
    hipFileWrite,
)
from hipfile.enums import FileHandleType
from hipfile.error import HipFileException

if TYPE_CHECKING:
    from types import TracebackType

    from hipfile.buffer import Buffer


class FileHandle:
    """Manage a file descriptor registered with the hipFile driver.

    Wraps ``os.open``/``os.close`` together with hipFile handle
    registration so that GPU-accelerated reads and writes can be
    performed through a single object.

    Supports the context-manager protocol for automatic
    ``open``/``close``::

        with FileHandle(path, os.O_RDONLY | os.O_DIRECT) as fh:
            fh.read(buf, size, 0, 0)
    """

    DEFAULT_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH  # 0o644

    def __init__(
        self,
        path: str | os.PathLike[str],
        flags: int,
        mode: int = DEFAULT_MODE,
        handle_type: FileHandleType = FileHandleType.OPAQUE_FD,
    ) -> None:
        """Initialize a ``FileHandle``.

        The file is **not** opened until ``open`` is called (or the
        context manager is entered).

        Parameters
        ----------
        path : str or os.PathLike[str]
            Filesystem path to open.
        flags : int
            Flags passed to ``os.open`` (e.g. ``os.O_RDONLY | os.O_DIRECT``).
        mode : int, optional
            Permission bits used when creating a file.  Defaults to
            ``0o644``.
        handle_type : FileHandleType, optional
            Type of handle to register.  Defaults to
            ``FileHandleType.OPAQUE_FD``.
        """
        self._fd: int | None = None
        self._flags = flags
        self._handle: int | None = None
        self._handle_type: FileHandleType | None = None
        self._mode = mode
        self._path = path

        self.handle_type = handle_type

    def __del__(self) -> None:
        """Close on garbage collection if still open."""
        try:
            self.close()
        except Exception:  # pylint: disable=W0718  # Suppress exceptions in a dtor
            print(
                "Failed to deregister hipFile.FileHandle at destruction time.",
                file=stderr,
            )

    def __enter__(self) -> FileHandle:
        """Open the file and return *self*."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the file."""
        self.close()

    @property
    def flags(self) -> int:
        """Flags passed to ``os.open``."""
        return self._flags

    @property
    def handle(self) -> int | None:
        """Opaque hipFile handle, or ``None`` if not open."""
        return self._handle

    @property
    def handle_type(self) -> FileHandleType | None:
        """Handle type used for registration."""
        return self._handle_type

    @handle_type.setter
    def handle_type(self, _handle_type: FileHandleType) -> None:
        """Set the handle type.

        Raises
        ------
        RuntimeError
            If the file handle is already open.
        ValueError
            If *_handle_type* is not a ``FileHandleType`` member.
        NotImplementedError
            If *_handle_type* is ``OPAQUE_WIN32``.
        """
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
    def mode(self) -> int:
        """File creation permission bits."""
        return self._mode

    @property
    def path(self) -> str | os.PathLike[str]:
        """Filesystem path."""
        return self._path

    def open(self) -> None:
        """Open the file and register it with the hipFile driver.

        Raises
        ------
        RuntimeError
            If the file handle is already open.
        HipFileException
            If hipFile handle registration fails.
        """
        if self._handle is not None:
            raise RuntimeError("The FileHandle is already open.")
        self._fd = os.open(self._path, self._flags, self._mode)
        handle, err = hipFileHandleRegister(self._fd, self._handle_type)
        if err[0] != 0:
            os.close(self._fd)
            self._fd = None
            raise HipFileException(err[0], err[1])
        self._handle = handle

    def close(self) -> None:
        """Deregister the hipFile handle and close the file descriptor.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._handle is not None:
            hipFileHandleDeregister(self._handle)
            self._handle = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def read(
        self, buffer: Buffer, size: int, file_offset: int, buffer_offset: int
    ) -> int:
        """Read from the file into a GPU buffer.

        Parameters
        ----------
        buffer : Buffer
            GPU buffer to read into.
        size : int
            Number of bytes to read.
        file_offset : int
            Byte offset within the file to start reading from.
        buffer_offset : int
            Byte offset within the GPU buffer to read into.

        Returns
        -------
        int
            Number of bytes actually read.

        Raises
        ------
        RuntimeError
            If the file handle is not open.
        OSError
            On a system-level I/O error (wraps ``errno``).
        HipFileException
            On a hipFile or HIP driver error.
        """
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

    def write(
        self, buffer: Buffer, size: int, file_offset: int, buffer_offset: int
    ) -> int:
        """Write from a GPU buffer into the file.

        Parameters
        ----------
        buffer : Buffer
            GPU buffer to write from.
        size : int
            Number of bytes to write.
        file_offset : int
            Byte offset within the file to start writing to.
        buffer_offset : int
            Byte offset within the GPU buffer to write from.

        Returns
        -------
        int
            Number of bytes actually written.

        Raises
        ------
        RuntimeError
            If the file handle is not open.
        OSError
            On a system-level I/O error (wraps ``errno``).
        HipFileException
            On a hipFile or HIP driver error.
        """
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
