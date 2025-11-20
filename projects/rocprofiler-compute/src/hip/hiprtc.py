import ctypes
from ctypes import (
    c_int,
    c_char,
    c_size_t,
    c_char_p,
    c_void_p,
    POINTER,
    byref,
)

_lib = ctypes.CDLL("libhiprtc.so")


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
    def __init__(self, code: int):
        self.code = code
        self.message = f"HIP Error {self.code}"

    def __str__(self):
        return self.message


class HIPRTCProgram:
    def __init__(self, handle):
        self.handle = handle

    def __del__(self):
        _lib.hiprtcDestroyProgram(self.handle)


# TODO: Handle headers
def hiprtcCreateProgram(src: str, name: str):

    src_bytes = src.encode("utf-8")
    name_bytes = name.encode("utf-8")

    prog = c_void_p()

    res = _lib.hiprtcCreateProgram(byref(prog), src_bytes, name_bytes, 0, None, None)

    if res != 0:
        raise HIPRTCError(res)

    return HIPRTCProgram(prog)


# def hiprtcDestroyProgram(prog):

#     res = _lib.hiprtcDestroyProgram(prog)

#     if res != 0:
#         raise HIPRTCError(res)


# TODO: Handle options
def hiprtcCompileProgram(prog):

    res = _lib.hiprtcCompileProgram(prog.handle, 0, None)

    if res != 0:
        raise HIPRTCError(res)


def hiprtcGetProgramLogSize(prog):

    size = c_size_t(0)

    res = _lib.hiprtcGetProgramLogSize(prog.handle, byref(size))

    if res != 0:
        raise HIPRTCError(res)

    return size.value


def hiprtcGetProgramLog(prog):

    size = hiprtcGetProgramLogSize(prog)
    buf = (ctypes.c_char * size)()

    res = _lib.hiprtcGetProgramLog(prog.handle, buf)

    if res != 0:
        raise HIPRTCError(res)

    return ctypes.string_at(buf, size).decode("utf-8", errors="ignore")


def hiprtcGetCodeSize(prog):
    size = c_size_t(0)
    res = _lib.hiprtcGetCodeSize(prog.handle, byref(size))

    if res != 0:
        raise HIPRTCError(res)

    return size.value


def hiprtcGetCode(prog):

    size = hiprtcGetCodeSize(prog)
    buf = (c_char * size)()
    res = _lib.hiprtcGetCode(prog.handle, buf)

    if res != 0:
        raise HIPRTCError(res)

    return buf


def hiprtcGetLoweredName(prog, name_expression):

    expr_bytes = name_expression.encode("utf-8")
    name_bytes = c_char_p()

    res = _lib.hiprtcGetLoweredName(prog.handle, expr_bytes, name_bytes)

    if res != 0:
        raise HIPRTCError(res)

    return name_bytes.value.decode("utf-8")


def hiprtcAddNameExpression(prog, name_expression):

    expr_bytes = name_expression.encode("utf-8")

    res = _lib.hiprtcAddNameExpression(prog.handle, expr_bytes)

    if res != 0:
        raise HIPRTCError(res)
