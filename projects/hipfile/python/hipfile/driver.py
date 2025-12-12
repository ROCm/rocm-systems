# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""hipFile driver lifecycle management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hipfile._hipfile import (  # pylint: disable=E0401,E0611
    hipFileDriverOpen,
    hipFileDriverClose,
    hipFileUseCount,
)
from hipfile.error import HipFileException

if TYPE_CHECKING:
    from types import TracebackType


class Driver:
    """Manage the hipFile driver lifecycle.

    Wraps the driver open/close calls and supports the
    context-manager protocol::

        with Driver() as drv:
            ...  # driver is open
        # driver is closed

    The driver is reference-counted internally; multiple ``open``
    calls are balanced by an equal number of ``close`` calls.
    """

    @staticmethod
    def use_count() -> int:
        """Return the current driver reference count."""
        return hipFileUseCount()

    def __enter__(self) -> Driver:
        """Open the driver and return *self*."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the driver."""
        self.close()

    def close(self) -> None:
        """Close the hipFile driver.

        Raises
        ------
        HipFileException
            If the close call fails.
        """
        err = hipFileDriverClose()
        if err[0] != 0:
            raise HipFileException(err[0], err[1])

    def open(self) -> None:
        """Open the hipFile driver.

        Raises
        ------
        HipFileException
            If the open call fails.
        """
        err = hipFileDriverOpen()
        if err[0] != 0:
            raise HipFileException(err[0], err[1])
