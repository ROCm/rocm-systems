# pylint: disable=C0114,C0115,C0116
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

    @staticmethod
    def use_count() -> int:
        return hipFileUseCount()

    def __enter__(self) -> Driver:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        err = hipFileDriverClose()
        if err[0] != 0:
            raise HipFileException(err[0], err[1])

    def open(self) -> None:
        err = hipFileDriverOpen()
        if err[0] != 0:
            raise HipFileException(err[0], err[1])
