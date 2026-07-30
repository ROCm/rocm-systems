# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Exception type for hipFile operations."""

from __future__ import annotations

from hipfile._hipfile import hipFileGetOpErrorString  # pylint: disable=E0401,E0611
from hipfile.enums import OpError


class HipFileException(Exception):
    """Exception raised when a hipFile operation fails.

    Wraps both the ``hipFileOpError_t`` code and, when the error is
    ``HIP_DRIVER_ERROR``, the underlying ``hipError_t`` from the HIP
    runtime.
    """

    def __init__(self, hipfile_err: int, hip_err: int) -> None:
        """Initialize a ``HipFileException``.

        Parameters
        ----------
        hipfile_err : int
            ``hipFileOpError_t`` value describing the failure.
        hip_err : int
            ``hipError_t`` value from the HIP runtime.  Only meaningful
            when *hipfile_err* equals ``OpError.HIP_DRIVER_ERROR``.
        """
        self._hipfile_err = hipfile_err
        self._hip_err = hip_err

    @property
    def hipfile_err(self) -> int:
        """``hipFileOpError_t`` code for this error."""
        return self._hipfile_err

    @property
    def hip_err(self) -> int:
        """``hipError_t`` code from the HIP runtime.

        Only meaningful when ``hipfile_err`` equals
        ``OpError.HIP_DRIVER_ERROR``.
        """
        return self._hip_err

    def __str__(self) -> str:
        """Return a human-readable description of the error."""
        err_msg = f"{self._hipfile_err} - {hipFileGetOpErrorString(self._hipfile_err)}"
        if self._hipfile_err == OpError.HIP_DRIVER_ERROR:
            err_msg += f" {self._hip_err}"
        return err_msg
