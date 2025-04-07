#!/usr/bin/env python@_VERSION@
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import

__author__ = "AMD ROCm"
__copyright__ = "Copyright 2024, Advanced Micro Devices, Inc."
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "AMD ROCm"
__status__ = "Development"

import os
import sys

from . import libpyrocprofsys
from .libpyrocprofsys.profiler import profiler_init as _profiler_init
from .libpyrocprofsys.profiler import profiler_finalize as _profiler_fini


__all__ = ["exec_", "_file", "_get_argv", "_initialize", "_finalize"]


PY3 = sys.version_info[0] == 3

# exec (from https://bitbucket.org/gutworth/six/):
if PY3:
    import builtins

    exec_ = getattr(builtins, "exec")
    del builtins
else:

    def exec_(_code_, _globs_=None, _locs_=None):
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec("""exec _code_ in _globs_, _locs_""")


def _file(back=2, only_basename=True, use_dirname=False, noquotes=True):
    """
    Returns the file name
    """

    from os.path import basename, dirname

    def get_fcode(back):
        fname = "<module>"
        try:
            fname = sys._getframe(back).f_code.co_filename
        except Exception as e:
            print(e)
            fname = "<module>"
        return fname

    result = None
    if only_basename is True:
        if use_dirname is True:
            result = "{}".format(
                join(
                    basename(dirname(get_fcode(back))),
                    basename(get_fcode(back)),
                )
            )
        else:
            result = "{}".format(basename(get_fcode(back)))
    else:
        result = "{}".format(get_fcode(back))

    if noquotes is False:
        result = "'{}'".format(result)

    return result


def _get_argv(init_file, argv=None):
    if argv is None:
        argv = sys.argv[:]

    if "--" in argv:
        _idx = argv.index("--")
        argv = sys.argv[(_idx + 1) :]

    if len(argv) > 1:
        if argv[0] == "-m":
            argv = argv[1:]
        elif argv[0] == "-c":
            argv[0] = os.path.basename(sys.executable)
        else:
            while len(argv) > 1 and argv[0].startswith("-"):
                argv = argv[1:]
                if os.path.exists(argv[0]):
                    break
    if len(argv) == 0:
        argv = [init_file]
    elif not os.path.exists(argv[0]):
        argv[0] = init_file

    return argv


def _initialize(_file):
    if not libpyrocprofsys.is_initialized():
        libpyrocprofsys.initialize(_get_argv(_file))


def _finalize():
    if libpyrocprofsys.is_initialized() and not libpyrocprofsys.is_finalized():
        _profiler_fini()
