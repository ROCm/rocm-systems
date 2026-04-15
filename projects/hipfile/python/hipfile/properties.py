# pylint: disable=C0114,C0116
from hipfile._hipfile import (  # pylint: disable=E0401,E0611
    hipFileDriverGetProperties,
    hipFileGetVersion,
)
from hipfile.error import HipFileException


def driver_get_properties():
    _props, err = hipFileDriverGetProperties()
    if err[0] != 0:
        raise HipFileException(err[0], err[1])
    return _props


def get_version():
    version_tuple, err = hipFileGetVersion()
    if err[0] != 0:
        raise HipFileException(err[0], err[1])
    return version_tuple
