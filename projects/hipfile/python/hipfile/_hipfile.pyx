# cython: language_level=3
"""
Low-level Cython wrappers for the hipFile C API.

Every function mirrors the C API as closely as possible.
Functions that return ``hipFileError_t`` in C return a
``(hipFileOpError_t, hipError_t)`` 2-tuple here.
"""

from libc.errno cimport errno
from libc.string cimport memset
from libc.stdint cimport uintptr_t

cimport hipfile._chipfile as _c


# ---------------------------------------------------------------------------
#  Module-level constants
# ---------------------------------------------------------------------------

VERSION_MAJOR = _c.HIPFILE_VERSION_MAJOR
VERSION_MINOR = _c.HIPFILE_VERSION_MINOR
VERSION_PATCH = _c.HIPFILE_VERSION_PATCH
BASE_ERR      = _c.HIPFILE_BASE_ERR

# ---------------------------------------------------------------------------
#  Enum re-exports (C → Python)
#
#  ctypedef enum values from _chipfile.pxd are C-level only after cimport.
#  These assignments create Python-visible module attributes whose values
#  are resolved from the C enum at compile time.
# ---------------------------------------------------------------------------

# hipFileOpError_t
hipFileSuccess                 = <int>_c.hipFileSuccess
hipFileDriverNotInitialized    = <int>_c.hipFileDriverNotInitialized
hipFileDriverInvalidProps      = <int>_c.hipFileDriverInvalidProps
hipFileDriverUnsupportedLimit  = <int>_c.hipFileDriverUnsupportedLimit
hipFileDriverVersionMismatch   = <int>_c.hipFileDriverVersionMismatch
hipFileDriverVersionReadError  = <int>_c.hipFileDriverVersionReadError
hipFileDriverClosing           = <int>_c.hipFileDriverClosing
hipFilePlatformNotSupported    = <int>_c.hipFilePlatformNotSupported
hipFileIONotSupported          = <int>_c.hipFileIONotSupported
hipFileDeviceNotSupported      = <int>_c.hipFileDeviceNotSupported
hipFileDriverError             = <int>_c.hipFileDriverError
hipFileHipDriverError          = <int>_c.hipFileHipDriverError
hipFileHipPointerInvalid       = <int>_c.hipFileHipPointerInvalid
hipFileHipMemoryTypeInvalid    = <int>_c.hipFileHipMemoryTypeInvalid
hipFileHipPointerRangeError    = <int>_c.hipFileHipPointerRangeError
hipFileHipContextMismatch      = <int>_c.hipFileHipContextMismatch
hipFileInvalidMappingSize      = <int>_c.hipFileInvalidMappingSize
hipFileInvalidMappingRange     = <int>_c.hipFileInvalidMappingRange
hipFileInvalidFileType         = <int>_c.hipFileInvalidFileType
hipFileInvalidFileOpenFlag     = <int>_c.hipFileInvalidFileOpenFlag
hipFileDIONotSet               = <int>_c.hipFileDIONotSet
hipFileInvalidValue            = <int>_c.hipFileInvalidValue
hipFileMemoryAlreadyRegistered = <int>_c.hipFileMemoryAlreadyRegistered
hipFileMemoryNotRegistered     = <int>_c.hipFileMemoryNotRegistered
hipFilePermissionDenied        = <int>_c.hipFilePermissionDenied
hipFileDriverAlreadyOpen       = <int>_c.hipFileDriverAlreadyOpen
hipFileHandleNotRegistered     = <int>_c.hipFileHandleNotRegistered
hipFileHandleAlreadyRegistered = <int>_c.hipFileHandleAlreadyRegistered
hipFileDeviceNotFound          = <int>_c.hipFileDeviceNotFound
hipFileInternalError           = <int>_c.hipFileInternalError
hipFileGetNewFDFailed          = <int>_c.hipFileGetNewFDFailed
hipFileDriverSetupError        = <int>_c.hipFileDriverSetupError
hipFileIODisabled              = <int>_c.hipFileIODisabled
hipFileBatchSubmitFailed       = <int>_c.hipFileBatchSubmitFailed
hipFileGPUMemoryPinningFailed  = <int>_c.hipFileGPUMemoryPinningFailed
hipFileBatchFull               = <int>_c.hipFileBatchFull
hipFileAsyncNotSupported       = <int>_c.hipFileAsyncNotSupported
hipFileIOMaxError              = <int>_c.hipFileIOMaxError

# hipFileFileHandleType_t
hipFileHandleTypeOpaqueFD    = <int>_c.hipFileHandleTypeOpaqueFD
hipFileHandleTypeOpaqueWin32 = <int>_c.hipFileHandleTypeOpaqueWin32
hipFileHandleTypeUserspaceFS = <int>_c.hipFileHandleTypeUserspaceFS


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

cdef inline tuple _err(_c.hipFileError_t e):
    return (<int>e.err, <int>e.hip_drv_err)


# ---------------------------------------------------------------------------
#  Error-handling helpers (replacements for C macros)
# ---------------------------------------------------------------------------

def is_hipfile_err(int err_code):
    """Equivalent of the ``IS_HIPFILE_ERR`` C macro."""
    return abs(err_code) > _c.HIPFILE_BASE_ERR


def hipfile_errstr(int err_code):
    """Equivalent of the ``HIPFILE_ERRSTR`` C macro."""
    cdef const char *s = _c.hipFileGetOpErrorString(<_c.hipFileOpError_t>abs(err_code))
    if s == NULL:
        return ""
    return s.decode("utf-8")


def is_hip_drv_err(tuple err):
    """Equivalent of the ``IS_HIP_DRV_ERR`` C macro.

    Takes an error tuple as returned by the wrapper functions.
    """
    return err[0] == <int>_c.hipFileHipDriverError


def hip_drv_err(tuple err):
    """Equivalent of the ``HIP_DRV_ERR`` C macro.

    Takes an error tuple and returns the ``hipError_t`` component.
    """
    return err[1]


def hipFileGetOpErrorString(int status):
    """Wrapper for ``hipFileGetOpErrorString``."""
    cdef const char *s = _c.hipFileGetOpErrorString(<_c.hipFileOpError_t>status)
    if s == NULL:
        return ""
    return s.decode("utf-8")


# ---------------------------------------------------------------------------
#  Driver lifecycle
# ---------------------------------------------------------------------------

def hipFileDriverOpen():
    """Wrapper for ``hipFileDriverOpen``."""
    return _err(_c.hipFileDriverOpen())


def hipFileDriverClose():
    """Wrapper for ``hipFileDriverClose``."""
    return _err(_c.hipFileDriverClose())


def hipFileUseCount():
    """Wrapper for ``hipFileUseCount``."""
    return <int>_c.hipFileUseCount()


# ---------------------------------------------------------------------------
#  Version
# ---------------------------------------------------------------------------

def hipFileGetVersion():
    """Wrapper for ``hipFileGetVersion``.

    Returns ``((major, minor, patch), error_tuple)``.
    """
    cdef unsigned major = 0, minor = 0, patch = 0
    cdef _c.hipFileError_t e = _c.hipFileGetVersion(&major, &minor, &patch)
    return ((major, minor, patch), _err(e))


# ---------------------------------------------------------------------------
#  File handles
# ---------------------------------------------------------------------------

def hipFileHandleRegister(uintptr_t handle_value, int handle_type):
    """Wrapper for ``hipFileHandleRegister``.

    Parameters
    ----------
    handle_value : int
        POSIX file descriptor or Win32 HANDLE, depending on *handle_type*.
    handle_type : int
        Value from ``hipFileFileHandleType_t``.

    Returns ``(handle_int, error_tuple)``.  The handle is an opaque
    integer that must be passed back to other hipFile calls.
    """
    cdef _c.hipFileHandle_t fh = NULL
    cdef _c.hipFileDescr_t descr
    memset(&descr, 0, sizeof(descr))
    descr.type = <_c.hipFileFileHandleType_t>handle_type
    if handle_type == <int>_c.hipFileHandleTypeOpaqueWin32:
        descr.hFile = <void *>handle_value
    else:
        descr.fd = <int>handle_value
    cdef _c.hipFileError_t e = _c.hipFileHandleRegister(&fh, &descr)
    return (<uintptr_t>fh, _err(e))


def hipFileHandleDeregister(uintptr_t handle):
    """Wrapper for ``hipFileHandleDeregister``."""
    _c.hipFileHandleDeregister(<_c.hipFileHandle_t>handle)


# ---------------------------------------------------------------------------
#  Buffer registration
# ---------------------------------------------------------------------------

def hipFileBufRegister(uintptr_t buffer_base, size_t length, int flags=0):
    """Wrapper for ``hipFileBufRegister``."""
    return _err(_c.hipFileBufRegister(<const void *>buffer_base, length, flags))


def hipFileBufDeregister(uintptr_t buffer_base):
    """Wrapper for ``hipFileBufDeregister``."""
    return _err(_c.hipFileBufDeregister(<const void *>buffer_base))


# ---------------------------------------------------------------------------
#  Synchronous I/O
# ---------------------------------------------------------------------------

def hipFileRead(uintptr_t handle, uintptr_t buffer_base, size_t size,
                _c.hoff_t file_offset, _c.hoff_t buffer_offset):
    """Wrapper for ``hipFileRead``.

    Returns ``(result, extra)``:
      * ``result >= 0`` — number of bytes read, ``extra = 0``
      * ``result == -1`` — system error, ``extra = errno``
      * ``result < -1``  — negated ``hipFileOpError_t``; if
        ``-hipFileHipDriverError``, ``extra = hipError_t`` from
        ``hipPeekAtLastError()``, otherwise ``extra = 0``
    """
    cdef ssize_t ret = _c.hipFileRead(<_c.hipFileHandle_t>handle,
                                      <void *>buffer_base, size,
                                      file_offset, buffer_offset)
    cdef int extra = 0
    if ret == -1:
        extra = errno
    elif ret == -<int>_c.hipFileHipDriverError:
        extra = <int>_c.hipPeekAtLastError()
    return (ret, extra)


def hipFileWrite(uintptr_t handle, uintptr_t buffer_base, size_t size,
                 _c.hoff_t file_offset, _c.hoff_t buffer_offset):
    """Wrapper for ``hipFileWrite``.

    Returns ``(result, extra)``:
      * ``result >= 0`` — number of bytes written, ``extra = 0``
      * ``result == -1`` — system error, ``extra = errno``
      * ``result < -1``  — negated ``hipFileOpError_t``; if
        ``-hipFileHipDriverError``, ``extra = hipError_t`` from
        ``hipPeekAtLastError()``, otherwise ``extra = 0``
    """
    cdef ssize_t ret = _c.hipFileWrite(<_c.hipFileHandle_t>handle,
                                       <const void *>buffer_base, size,
                                       file_offset, buffer_offset)
    cdef int extra = 0
    if ret == -1:
        extra = errno
    elif ret == -<int>_c.hipFileHipDriverError:
        extra = <int>_c.hipPeekAtLastError()
    return (ret, extra)


# ---------------------------------------------------------------------------
#  Driver properties
# ---------------------------------------------------------------------------

def hipFileDriverGetProperties():
    """Wrapper for ``hipFileDriverGetProperties``.

    Returns ``(props_dict, error_tuple)``.
    """
    cdef _c.hipFileDriverProps_t props
    memset(&props, 0, sizeof(props))
    cdef _c.hipFileError_t e = _c.hipFileDriverGetProperties(&props)
    d = {
        "nvfs_major_version":        props.nvfs_major_version,
        "nvfs_minor_version":        props.nvfs_minor_version,
        "nvfs_poll_thresh_size":     props.nvfs_poll_thresh_size,
        "nvfs_max_direct_io_size":   props.nvfs_max_direct_io_size,
        "nvfs_driver_status_flags":  props.nvfs_driver_status_flags,
        "nvfs_driver_control_flags": props.nvfs_driver_control_flags,
        "feature_flags":             props.feature_flags,
        "max_device_cache_size":     props.max_device_cache_size,
        "per_buffer_cache_size":     props.per_buffer_cache_size,
        "max_device_pinned_mem_size": props.max_device_pinned_mem_size,
        "max_batch_io_count":        props.max_batch_io_count,
        "max_batch_io_timeout_msecs": props.max_batch_io_timeout_msecs,
    }
    return (d, _err(e))
