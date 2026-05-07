# pylint: disable=C0114,C0115,C0116
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

    @classmethod
    def from_ctypes_void_p(
        cls, ctypes_void_p: c_void_p, length: int, flags: int
    ) -> Buffer:
        if ctypes_void_p.value is None:
            raise ValueError("Cannot pass in a null pointer.")
        return cls(ctypes_void_p.value, length, flags)

    def __init__(self, buffer_ptr: int, length: int, flags: int) -> None:
        self._buffer_ptr = buffer_ptr
        self._flags = flags
        self._length = length
        self._registered = False

    def __del__(self) -> None:
        # We did not create the underlying buffer. Don't try to free it.
        try:
            self.deregister()
        except Exception:  # pylint: disable=W0718  # Suppress exceptions in a dtor
            print(
                "Failed to deregister hipFile.Buffer at destruction time.", file=stderr
            )

    def __enter__(self) -> Buffer:
        self.register()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.deregister()

    @property
    def ptr(self) -> int:
        return self._buffer_ptr

    def deregister(self) -> None:
        if self._registered:
            err = hipFileBufDeregister(self._buffer_ptr)
            if err[0] != 0:
                raise HipFileException(err[0], err[1])
            self._registered = False

    def register(self) -> None:
        err = hipFileBufRegister(self._buffer_ptr, self._length, self._flags)
        if err[0] != 0:
            raise HipFileException(err[0], err[1])
        self._registered = True
