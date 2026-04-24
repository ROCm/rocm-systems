# pylint: disable=C0114,C0115,C0116
from hipfile._hipfile import (  # pylint: disable=E0401,E0611
    hipFileDriverOpen,
    hipFileDriverClose,
    hipFileUseCount,
)
from hipfile.error import HipFileException


class Driver:

    @staticmethod
    def use_count():
        return hipFileUseCount()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        err = hipFileDriverClose()
        if err[0] != 0:
            raise HipFileException(err[0], err[1])

    def open(self):
        err = hipFileDriverOpen()
        if err[0] != 0:
            raise HipFileException(err[0], err[1])
