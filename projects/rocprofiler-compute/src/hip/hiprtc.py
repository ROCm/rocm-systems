##############################################################################
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import ctypes
import os
from ctypes import (
    POINTER,
    byref,
    c_char,
    c_char_p,
    c_int,
    c_size_t,
    c_void_p,
)

_lib = ctypes.CDLL(f"{os.getenv('ROCM_PATH', '/opt/rocm')}/lib/libhiprtc.so")


_lib.hiprtcCreateProgram.restype = c_int
_lib.hiprtcCreateProgram.argtypes = [
    POINTER(c_void_p),
    c_char_p,
    c_char_p,
    c_int,
    POINTER(c_char_p),
    POINTER(c_char_p),
]

_lib.hiprtcDestroyProgram.restype = c_int
_lib.hiprtcDestroyProgram.argtypes = [
    POINTER(c_void_p),
]

_lib.hiprtcCompileProgram.restype = c_int
_lib.hiprtcCompileProgram.argtypes = [
    c_void_p,
    c_int,
    POINTER(c_char_p),
]

_lib.hiprtcGetProgramLogSize.restype = c_int
_lib.hiprtcGetProgramLogSize.argtypes = [
    c_void_p,
    POINTER(c_size_t),
]

_lib.hiprtcGetProgramLog.restype = c_int
_lib.hiprtcGetProgramLog.argtypes = [
    c_void_p,
    c_char_p,
]

_lib.hiprtcGetCodeSize.restype = c_int
_lib.hiprtcGetCodeSize.argtypes = [
    c_void_p,
    POINTER(c_size_t),
]

_lib.hiprtcGetCode.restype = c_int
_lib.hiprtcGetCode.argtypes = [
    c_void_p,
    c_char_p,
]

_lib.hiprtcAddNameExpression.restype = c_int
_lib.hiprtcAddNameExpression.argtypes = [
    c_void_p,
    c_char_p,
]

_lib.hiprtcGetLoweredName.restype = c_int
_lib.hiprtcGetLoweredName.argtypes = [c_void_p, c_char_p, POINTER(c_char_p)]


class HIPRTCError(Exception):
    def __init__(self, code: int) -> None:
        self.code = code
        self.message = f"HIP Error {self.code}"

    def __str__(self) -> str:
        return self.message


class HIPRTCProgram:
    def __init__(self, handle: POINTER) -> None:
        self.handle = handle

    def __del__(self) -> None:
        _lib.hiprtcDestroyProgram(self.handle)


# TODO: Handle headers
def hiprtcCreateProgram(src: str, name: str) -> HIPRTCProgram:
    src_bytes = src.encode("utf-8")
    name_bytes = name.encode("utf-8")

    prog = c_void_p()

    res = _lib.hiprtcCreateProgram(byref(prog), src_bytes, name_bytes, 0, None, None)

    if res != 0:
        raise HIPRTCError(res)

    return HIPRTCProgram(prog)


# TODO: Handle compile options
def hiprtcCompileProgram(prog: HIPRTCProgram) -> None:
    res = _lib.hiprtcCompileProgram(prog.handle, 0, None)

    if res != 0:
        raise HIPRTCError(res)


def hiprtcGetProgramLogSize(prog: HIPRTCProgram) -> int:
    size = c_size_t(0)

    res = _lib.hiprtcGetProgramLogSize(prog.handle, byref(size))

    if res != 0:
        raise HIPRTCError(res)

    return size.value


def hiprtcGetProgramLog(prog: HIPRTCProgram) -> str:
    size = hiprtcGetProgramLogSize(prog)
    buf = (ctypes.c_char * size)()

    res = _lib.hiprtcGetProgramLog(prog.handle, buf)

    if res != 0:
        raise HIPRTCError(res)

    return ctypes.string_at(buf, size).decode("utf-8", errors="ignore")


def hiprtcGetCodeSize(prog: HIPRTCProgram) -> int:
    size = c_size_t(0)
    res = _lib.hiprtcGetCodeSize(prog.handle, byref(size))

    if res != 0:
        raise HIPRTCError(res)

    return size.value


def hiprtcGetCode(prog: HIPRTCProgram) -> POINTER:
    size = hiprtcGetCodeSize(prog)
    buf = (c_char * size)()
    res = _lib.hiprtcGetCode(prog.handle, buf)

    if res != 0:
        raise HIPRTCError(res)

    return buf


def hiprtcGetLoweredName(prog: HIPRTCProgram, name_expression: str) -> str:
    expr_bytes = name_expression.encode("utf-8")
    name_bytes = c_char_p()

    res = _lib.hiprtcGetLoweredName(prog.handle, expr_bytes, name_bytes)

    if res != 0:
        raise HIPRTCError(res)

    return name_bytes.value.decode("utf-8")


def hiprtcAddNameExpression(prog: HIPRTCProgram, name_expression: str) -> None:
    expr_bytes = name_expression.encode("utf-8")

    res = _lib.hiprtcAddNameExpression(prog.handle, expr_bytes)

    if res != 0:
        raise HIPRTCError(res)
