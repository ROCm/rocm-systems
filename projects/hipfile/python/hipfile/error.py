# pylint: disable=C0114,C0115,C0116
from __future__ import annotations

from hipfile._hipfile import hipFileGetOpErrorString  # pylint: disable=E0401,E0611
from hipfile.enums import OpError


class HipFileException(Exception):
    def __init__(self, hipfile_err: int, hip_err: int) -> None:
        self._hipfile_err = hipfile_err
        self._hip_err = hip_err

    @property
    def hipfile_err(self) -> int:
        return self._hipfile_err

    @property
    def hip_err(self) -> int:
        return self._hip_err

    def __str__(self) -> str:
        err_msg = f"{self._hipfile_err} - {hipFileGetOpErrorString(self._hipfile_err)}"
        if self._hipfile_err == OpError.HIP_DRIVER_ERROR:
            err_msg += f" {self._hip_err}"
        return err_msg
