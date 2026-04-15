# pylint: disable=C0114

from hipfile._hipfile import (  # pylint: disable=E0401,E0611
    # Constants
    VERSION_MAJOR as _VERSION_MAJOR,
    VERSION_MINOR as _VERSION_MINOR,
    VERSION_PATCH as _VERSION_PATCH,
)
from hipfile.buffer import Buffer
from hipfile.driver import Driver
from hipfile.enums import FileHandleType, OpError
from hipfile.error import HipFileException
from hipfile.file import FileHandle
from hipfile.properties import driver_get_properties, get_version

__all__ = [
    "__version__",
    "Driver",
    "FileHandle",
    "Buffer",
    "HipFileException",
    "FileHandleType",
    "OpError",
    "driver_get_properties",
    "get_version",
]
__version__ = f"{_VERSION_MAJOR}.{_VERSION_MINOR}.{_VERSION_PATCH}"
