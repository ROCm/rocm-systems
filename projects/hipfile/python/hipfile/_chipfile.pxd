# cython: language_level=3
"""
C declarations for hipFile API (extern from hipfile.h).

This .pxd file declares the subset of the hipFile C API that is
wrapped by the low-level Cython bindings.
"""

from libc.stdint cimport int64_t, uint64_t
from posix.types cimport off_t


# ---------------------------------------------------------------------------
#  HIP runtime stub — only hipError_t is needed
# ---------------------------------------------------------------------------

cdef extern from "hip/hip_runtime_api.h":
    ctypedef enum hipError_t:
        hipSuccess = 0

    hipError_t hipPeekAtLastError()


# ---------------------------------------------------------------------------
#  hipFile public API
# ---------------------------------------------------------------------------

cdef extern from "hipfile.h":

    # -- Version constants --------------------------------------------------

    int HIPFILE_VERSION_MAJOR
    int HIPFILE_VERSION_MINOR
    int HIPFILE_VERSION_PATCH
    int HIPFILE_BASE_ERR

    # -- Platform-independent types -----------------------------------------

    ctypedef off_t hoff_t

    # -- Error handling -----------------------------------------------------

    ctypedef enum hipFileOpError_t:
        hipFileSuccess
        hipFileDriverNotInitialized
        hipFileDriverInvalidProps
        hipFileDriverUnsupportedLimit
        hipFileDriverVersionMismatch
        hipFileDriverVersionReadError
        hipFileDriverClosing
        hipFilePlatformNotSupported
        hipFileIONotSupported
        hipFileDeviceNotSupported
        hipFileDriverError
        hipFileHipDriverError
        hipFileHipPointerInvalid
        hipFileHipMemoryTypeInvalid
        hipFileHipPointerRangeError
        hipFileHipContextMismatch
        hipFileInvalidMappingSize
        hipFileInvalidMappingRange
        hipFileInvalidFileType
        hipFileInvalidFileOpenFlag
        hipFileDIONotSet
        # 5021 intentionally unused
        hipFileInvalidValue
        hipFileMemoryAlreadyRegistered
        hipFileMemoryNotRegistered
        hipFilePermissionDenied
        hipFileDriverAlreadyOpen
        hipFileHandleNotRegistered
        hipFileHandleAlreadyRegistered
        hipFileDeviceNotFound
        hipFileInternalError
        hipFileGetNewFDFailed
        # 5032 intentionally unused
        hipFileDriverSetupError
        hipFileIODisabled
        hipFileBatchSubmitFailed
        hipFileGPUMemoryPinningFailed
        hipFileBatchFull
        hipFileAsyncNotSupported
        hipFileIOMaxError

    ctypedef struct hipFileError_t:
        hipFileOpError_t err
        hipError_t hip_drv_err

    # -- Opaque handles -----------------------------------------------------

    ctypedef void *hipFileHandle_t

    # -- File handle types --------------------------------------------------

    ctypedef enum hipFileFileHandleType_t:
        hipFileHandleTypeOpaqueFD
        hipFileHandleTypeOpaqueWin32
        hipFileHandleTypeUserspaceFS

    # -- Userspace FS ops (opaque — only needed as pointer type) ------------

    ctypedef struct hipFileFSOps_t:
        pass

    # -- File descriptor ----------------------------------------------------
    # The anonymous union is accessed via Cython C-name strings.

    ctypedef struct hipFileDescr_t:
        hipFileFileHandleType_t type
        int fd       "handle.fd"
        void *hFile  "handle.hFile"
        const hipFileFSOps_t *fs_ops

    # -- Driver status / control / feature flag enums -----------------------

    ctypedef enum hipFileDriverStatusFlags_t:
        hipFileLustreSupported
        hipFileWekaFSSupported
        hipFileNFSSupported
        hipFileGPFSSupported
        hipFileNVMeSupported
        hipFileNVMeoFSupported
        hipFileSCSISupported
        hipFileScaleFluxCSDSupported
        hipFileNVMeshSupported
        hipFileBeeGFSSupported
        # 10 reserved for YRCloudFile
        hipFileNVMeP2PSupported
        hipFileScatefsSupported

    ctypedef enum hipFileDriverControlFlags_t:
        hipFileUsePollMode
        hipFileAllowCompatMode

    ctypedef enum hipFileFeatureFlags_t:
        hipFileDynRoutingSupported
        hipFileBatchIOSupported
        hipFileStreamsSupported
        hipFileParallelIOSupported

    # -- Driver properties --------------------------------------------------
    # Nested anonymous struct ``nvfs`` is flattened with C-name strings.

    ctypedef struct hipFileDriverProps_t:
        unsigned int nvfs_major_version        "nvfs.major_version"
        unsigned int nvfs_minor_version        "nvfs.minor_version"
        uint64_t     nvfs_poll_thresh_size     "nvfs.poll_thresh_size"
        uint64_t     nvfs_max_direct_io_size   "nvfs.max_direct_io_size"
        unsigned int nvfs_driver_status_flags  "nvfs.driver_status_flags"
        unsigned int nvfs_driver_control_flags "nvfs.driver_control_flags"
        unsigned int feature_flags
        uint64_t     max_device_cache_size
        uint64_t     per_buffer_cache_size
        uint64_t     max_device_pinned_mem_size
        unsigned int max_batch_io_count
        unsigned int max_batch_io_timeout_msecs

    # -- Function declarations ----------------------------------------------

    # Error
    const char *hipFileGetOpErrorString(hipFileOpError_t status)

    # File handles
    hipFileError_t hipFileHandleRegister(hipFileHandle_t *fh,
                                         hipFileDescr_t *descr)
    void hipFileHandleDeregister(hipFileHandle_t fh)

    # Buffer registration
    hipFileError_t hipFileBufRegister(const void *buffer_base,
                                      size_t length, int flags)
    hipFileError_t hipFileBufDeregister(const void *buffer_base)

    # Synchronous I/O
    ssize_t hipFileRead(hipFileHandle_t fh, void *buffer_base, size_t size,
                        hoff_t file_offset, hoff_t buffer_offset)
    ssize_t hipFileWrite(hipFileHandle_t fh, const void *buffer_base,
                         size_t size, hoff_t file_offset,
                         hoff_t buffer_offset)

    # Driver lifecycle
    hipFileError_t hipFileDriverOpen()
    hipFileError_t hipFileDriverClose()
    int64_t hipFileUseCount()

    # Driver properties
    hipFileError_t hipFileDriverGetProperties(hipFileDriverProps_t *props)

    # Version
    hipFileError_t hipFileGetVersion(unsigned *major, unsigned *minor,
                                      unsigned *patch)
