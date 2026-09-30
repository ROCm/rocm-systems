# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""GPU memory buffer registration for hipFile I/O."""

from __future__ import annotations

from typing import TYPE_CHECKING
from sys import stderr

from hipfile._hipfile import (  # pylint: disable=E0401,E0611
    hipFileBufDeregister,
    hipFileBufRegister,
)
from hipfile.error import HipFileException

if TYPE_CHECKING:
    from ctypes import c_void_p
    from types import TracebackType


class Buffer:
    """Manage registration of a GPU memory buffer with hipFile.

    ``Buffer`` does **not** own the underlying GPU allocation; it only
    manages the hipFile registration lifetime.

    Supports the context-manager protocol for automatic
    ``register``/``deregister``::

        with Buffer.from_ctypes_void_p(ptr, length, 0) as buf:
            fh.read(buf, length, 0, 0)
    """

    @classmethod
    def from_ctypes_void_p(
        cls, ctypes_void_p: c_void_p, length: int, flags: int
    ) -> Buffer:
        """Create a ``Buffer`` from a ``ctypes.c_void_p``.

        Parameters
        ----------
        ctypes_void_p : ctypes.c_void_p
            Pointer to GPU memory.  Must not be null.
        length : int
            Size of the buffer in bytes.
        flags : int
            Registration flags (pass ``0`` for default behavior).

        Raises
        ------
        ValueError
            If *ctypes_void_p* is null.
        """
        if ctypes_void_p.value is None:
            raise ValueError("Cannot pass in a null pointer.")
        return cls(ctypes_void_p.value, length, flags)

    def __init__(self, buffer_ptr: int, length: int, flags: int) -> None:
        """Initialize a ``Buffer`` from a raw integer pointer.

        Parameters
        ----------
        buffer_ptr : int
            Integer address of the GPU memory.
        length : int
            Size of the buffer in bytes.
        flags : int
            Registration flags (pass ``0`` for default behavior).
        """
        self._buffer_ptr = buffer_ptr
        self._flags = flags
        self._length = length
        self._registered = False

    def __del__(self) -> None:
        """Deregister on garbage collection if still registered."""
        # We did not create the underlying buffer. Don't try to free it.
        try:
            self.deregister()
        except Exception:  # pylint: disable=W0718  # Suppress exceptions in a dtor
            print(
                "Failed to deregister hipFile.Buffer at destruction time.", file=stderr
            )

    def __enter__(self) -> Buffer:
        """Register the buffer and return *self*."""
        self.register()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Deregister the buffer."""
        self.deregister()

    @property
    def ptr(self) -> int:
        """Integer address of the GPU memory."""
        return self._buffer_ptr

    def deregister(self) -> None:
        """Deregister the buffer from the hipFile driver.

        This is a no-op if the buffer is not currently registered.

        Raises
        ------
        HipFileException
            If the deregistration call fails.
        """
        if self._registered:
            err = hipFileBufDeregister(self._buffer_ptr)
            if err[0] != 0:
                raise HipFileException(err[0], err[1])
            self._registered = False

    def register(self) -> None:
        """Register the buffer with the hipFile driver.

        Raises
        ------
        HipFileException
            If the registration call fails.
        """
        err = hipFileBufRegister(self._buffer_ptr, self._length, self._flags)
        if err[0] != 0:
            raise HipFileException(err[0], err[1])
        self._registered = True
