# pylint: disable=C0114,C0115,C0116
from enum import IntEnum

from hipfile._hipfile import (  # pylint: disable=E0401,E0611
    # hipFileOpError_t values (resolved from C at build time)
    hipFileSuccess,
    hipFileDriverNotInitialized,
    hipFileDriverInvalidProps,
    hipFileDriverUnsupportedLimit,
    hipFileDriverVersionMismatch,
    hipFileDriverVersionReadError,
    hipFileDriverClosing,
    hipFilePlatformNotSupported,
    hipFileIONotSupported,
    hipFileDeviceNotSupported,
    hipFileDriverError,
    hipFileHipDriverError,
    hipFileHipPointerInvalid,
    hipFileHipMemoryTypeInvalid,
    hipFileHipPointerRangeError,
    hipFileHipContextMismatch,
    hipFileInvalidMappingSize,
    hipFileInvalidMappingRange,
    hipFileInvalidFileType,
    hipFileInvalidFileOpenFlag,
    hipFileDIONotSet,
    hipFileInvalidValue,
    hipFileMemoryAlreadyRegistered,
    hipFileMemoryNotRegistered,
    hipFilePermissionDenied,
    hipFileDriverAlreadyOpen,
    hipFileHandleNotRegistered,
    hipFileHandleAlreadyRegistered,
    hipFileDeviceNotFound,
    hipFileInternalError,
    hipFileGetNewFDFailed,
    hipFileDriverSetupError,
    hipFileIODisabled,
    hipFileBatchSubmitFailed,
    hipFileGPUMemoryPinningFailed,
    hipFileBatchFull,
    hipFileAsyncNotSupported,
    hipFileIOMaxError,
    # hipFileFileHandleType_t values (resolved from C at build time)
    hipFileHandleTypeOpaqueFD,
    hipFileHandleTypeOpaqueWin32,
    hipFileHandleTypeUserspaceFS,
)


class OpError(IntEnum):
    """Python enum mirroring hipFileOpError_t.

    Values are sourced from the C enum via the Cython layer, not
    redefined.  Rebuilding the extension picks up any value changes
    in hipfile.h automatically.
    """

    SUCCESS = hipFileSuccess
    DRIVER_NOT_INITIALIZED = hipFileDriverNotInitialized
    DRIVER_INVALID_PROPS = hipFileDriverInvalidProps
    DRIVER_UNSUPPORTED_LIMIT = hipFileDriverUnsupportedLimit
    DRIVER_VERSION_MISMATCH = hipFileDriverVersionMismatch
    DRIVER_VERSION_READ_ERROR = hipFileDriverVersionReadError
    DRIVER_CLOSING = hipFileDriverClosing
    PLATFORM_NOT_SUPPORTED = hipFilePlatformNotSupported
    IO_NOT_SUPPORTED = hipFileIONotSupported
    DEVICE_NOT_SUPPORTED = hipFileDeviceNotSupported
    DRIVER_ERROR = hipFileDriverError
    HIP_DRIVER_ERROR = hipFileHipDriverError
    HIP_POINTER_INVALID = hipFileHipPointerInvalid
    HIP_MEMORY_TYPE_INVALID = hipFileHipMemoryTypeInvalid
    HIP_POINTER_RANGE_ERROR = hipFileHipPointerRangeError
    HIP_CONTEXT_MISMATCH = hipFileHipContextMismatch
    INVALID_MAPPING_SIZE = hipFileInvalidMappingSize
    INVALID_MAPPING_RANGE = hipFileInvalidMappingRange
    INVALID_FILE_TYPE = hipFileInvalidFileType
    INVALID_FILE_OPEN_FLAG = hipFileInvalidFileOpenFlag
    DIO_NOT_SET = hipFileDIONotSet
    INVALID_VALUE = hipFileInvalidValue
    MEMORY_ALREADY_REGISTERED = hipFileMemoryAlreadyRegistered
    MEMORY_NOT_REGISTERED = hipFileMemoryNotRegistered
    PERMISSION_DENIED = hipFilePermissionDenied
    DRIVER_ALREADY_OPEN = hipFileDriverAlreadyOpen
    HANDLE_NOT_REGISTERED = hipFileHandleNotRegistered
    HANDLE_ALREADY_REGISTERED = hipFileHandleAlreadyRegistered
    DEVICE_NOT_FOUND = hipFileDeviceNotFound
    INTERNAL_ERROR = hipFileInternalError
    GET_NEW_FD_FAILED = hipFileGetNewFDFailed
    DRIVER_SETUP_ERROR = hipFileDriverSetupError
    IO_DISABLED = hipFileIODisabled
    BATCH_SUBMIT_FAILED = hipFileBatchSubmitFailed
    GPU_MEMORY_PINNING_FAILED = hipFileGPUMemoryPinningFailed
    BATCH_FULL = hipFileBatchFull
    ASYNC_NOT_SUPPORTED = hipFileAsyncNotSupported
    IO_MAX_ERROR = hipFileIOMaxError


class FileHandleType(IntEnum):
    """Python enum mirroring hipFileFileHandleType_t.

    Values are sourced from the C enum via the Cython layer.
    """

    OPAQUE_FD = hipFileHandleTypeOpaqueFD
    OPAQUE_WIN32 = hipFileHandleTypeOpaqueWin32
    USERSPACE_FS = hipFileHandleTypeUserspaceFS
