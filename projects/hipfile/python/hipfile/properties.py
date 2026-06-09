"""Query hipFile driver properties and version information."""

from __future__ import annotations

from hipfile._hipfile import (  # pylint: disable=E0401,E0611
    hipFileDriverGetProperties,
    hipFileGetVersion,
)
from hipfile.error import HipFileException


def driver_get_properties() -> dict[str, int]:
    """Return the current hipFile driver properties.

    Returns
    -------
    dict[str, int]
        Property names mapped to their integer values.

    Raises
    ------
    HipFileException
        If the properties query fails.
    """
    _props, err = hipFileDriverGetProperties()
    if err[0] != 0:
        raise HipFileException(err[0], err[1])
    return _props


def get_version() -> tuple[int, int, int]:
    """Return the hipFile driver version.

    Returns
    -------
    tuple[int, int, int]
        ``(major, minor, patch)`` version components.

    Raises
    ------
    HipFileException
        If the version query fails.
    """
    version_tuple, err = hipFileGetVersion()
    if err[0] != 0:
        raise HipFileException(err[0], err[1])
    return version_tuple
