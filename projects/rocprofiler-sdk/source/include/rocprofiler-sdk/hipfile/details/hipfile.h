/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <rocprofiler-sdk/hipfile/details/hipfile_headers.h>

#if ROCPROFILER_SDK_USE_SYSTEM_HIPFILE > 0
#    include <hipfile.h>
#else

#    include <hip/hip_runtime_api.h>

#    include <stdbool.h>
#    include <stdint.h>
#    include <stdlib.h>
#    include <sys/types.h>
#    include <time.h>

#    ifdef _WIN32
#        include <BaseTsd.h>
#        include <winsock2.h>
typedef __int64 hoff_t;
typedef SSIZE_T ssize_t;
#    else
#        include <sys/socket.h>
typedef off_t hoff_t;
#    endif

#    if defined(__GNUC__)
#        define HIPFILE_API __attribute__((visibility("default")))
#    else
#        define HIPFILE_API
#    endif

#    ifdef __cplusplus
extern "C" {
#    endif

#    define HIPFILE_VERSION_MAJOR 0
#    define HIPFILE_VERSION_MINOR 2
#    define HIPFILE_VERSION_PATCH 0

#    define HIPFILE_BASE_ERR 5000

typedef enum hipFileOpError
{
    hipFileSuccess                 = 0,
    hipFileDriverNotInitialized    = HIPFILE_BASE_ERR + 1,
    hipFileDriverInvalidProps      = HIPFILE_BASE_ERR + 2,
    hipFileDriverUnsupportedLimit  = HIPFILE_BASE_ERR + 3,
    hipFileDriverVersionMismatch   = HIPFILE_BASE_ERR + 4,
    hipFileDriverVersionReadError  = HIPFILE_BASE_ERR + 5,
    hipFileDriverClosing           = HIPFILE_BASE_ERR + 6,
    hipFilePlatformNotSupported    = HIPFILE_BASE_ERR + 7,
    hipFileIONotSupported          = HIPFILE_BASE_ERR + 8,
    hipFileDeviceNotSupported      = HIPFILE_BASE_ERR + 9,
    hipFileDriverError             = HIPFILE_BASE_ERR + 10,
    hipFileHipDriverError          = HIPFILE_BASE_ERR + 11,
    hipFileHipPointerInvalid       = HIPFILE_BASE_ERR + 12,
    hipFileHipMemoryTypeInvalid    = HIPFILE_BASE_ERR + 13,
    hipFileHipPointerRangeError    = HIPFILE_BASE_ERR + 14,
    hipFileHipContextMismatch      = HIPFILE_BASE_ERR + 15,
    hipFileInvalidMappingSize      = HIPFILE_BASE_ERR + 16,
    hipFileInvalidMappingRange     = HIPFILE_BASE_ERR + 17,
    hipFileInvalidFileType         = HIPFILE_BASE_ERR + 18,
    hipFileInvalidFileOpenFlag     = HIPFILE_BASE_ERR + 19,
    hipFileDIONotSet               = HIPFILE_BASE_ERR + 20,
    hipFileInvalidValue            = HIPFILE_BASE_ERR + 22,
    hipFileMemoryAlreadyRegistered = HIPFILE_BASE_ERR + 23,
    hipFileMemoryNotRegistered     = HIPFILE_BASE_ERR + 24,
    hipFilePermissionDenied        = HIPFILE_BASE_ERR + 25,
    hipFileDriverAlreadyOpen       = HIPFILE_BASE_ERR + 26,
    hipFileHandleNotRegistered     = HIPFILE_BASE_ERR + 27,
    hipFileHandleAlreadyRegistered = HIPFILE_BASE_ERR + 28,
    hipFileDeviceNotFound          = HIPFILE_BASE_ERR + 29,
    hipFileInternalError           = HIPFILE_BASE_ERR + 30,
    hipFileGetNewFDFailed          = HIPFILE_BASE_ERR + 31,
    hipFileDriverSetupError        = HIPFILE_BASE_ERR + 33,
    hipFileIODisabled              = HIPFILE_BASE_ERR + 34,
    hipFileBatchSubmitFailed       = HIPFILE_BASE_ERR + 35,
    hipFileGPUMemoryPinningFailed  = HIPFILE_BASE_ERR + 36,
    hipFileBatchFull               = HIPFILE_BASE_ERR + 37,
    hipFileAsyncNotSupported       = HIPFILE_BASE_ERR + 38,
    hipFileIOMaxError              = HIPFILE_BASE_ERR + 39,
} hipFileOpError_t;

typedef struct hipFileError
{
    hipFileOpError_t err;
    hipError_t       hip_drv_err;
} hipFileError_t;

typedef struct hipFileDriverProps
{
    struct
    {
        unsigned major_version;
        unsigned minor_version;
        uint64_t poll_thresh_size;
        uint64_t max_direct_io_size;
        unsigned driver_status_flags;
        unsigned driver_control_flags;
    } nvfs;
    unsigned feature_flags;
    uint64_t max_device_cache_size;
    uint64_t per_buffer_cache_size;
    uint64_t max_device_pinned_mem_size;
    unsigned max_batch_io_count;
    unsigned max_batch_io_timeout_msecs;
} hipFileDriverProps_t;

typedef struct hipFileRDMAInfo
{
    int         version;
    int         desc_len;
    const char* desc_str;
} hipFileRDMAInfo_t;

typedef struct hipFileFSOps
{
    const char* (*fs_type)(void* handle);
    int (*getRDMADeviceList)(void* handle, struct sockaddr** hostaddrs);
    int (*getRDMADevicePriority)(void* handle, char*, size_t, hoff_t, struct sockaddr* hostaddr);
    ssize_t (*read)(void* handle, char*, size_t, hoff_t, hipFileRDMAInfo_t*);
    ssize_t (*write)(void* handle, const char*, size_t, hoff_t, hipFileRDMAInfo_t*);
} hipFileFSOps_t;

typedef enum hipFileFileHandleType
{
    hipFileHandleTypeOpaqueFD    = 1,
    hipFileHandleTypeOpaqueWin32 = 2,
    hipFileHandleTypeUserspaceFS = 3,
} hipFileFileHandleType_t;

typedef struct hipFileDescr
{
    hipFileFileHandleType_t type;
    union
    {
        int   fd;
        void* hFile;
    } handle;
    const hipFileFSOps_t* fs_ops;
} hipFileDescr_t;

typedef void* hipFileHandle_t;

typedef enum hipFileOpcode
{
    hipFileBatchRead  = 0,
    hipFileBatchWrite = 1,
} hipFileOpcode_t;

typedef enum hipFileStatus
{
    hipFileWaiting  = 1 << 0,
    hipFilePending  = 1 << 1,
    hipFileInvalid  = 1 << 2,
    hipFileCanceled = 1 << 3,
    hipFileComplete = 1 << 4,
    hipFileTimeout  = 1 << 5,
    hipFileFailed   = 1 << 6,
} hipFileStatus_t;

typedef enum hipFileBatchMode
{
    hipFileBatch = 1,
} hipFileBatchMode_t;

typedef struct hipFileIOParams
{
    hipFileBatchMode_t mode;
    union
    {
        struct
        {
            void*   devPtr_base;
            int64_t file_offset;
            int64_t devPtr_offset;
            size_t  size;
        } batch;
    } u;
    hipFileHandle_t fh;
    hipFileOpcode_t opcode;
    void*           cookie;
} hipFileIOParams_t;

typedef struct hipFileIOEvents
{
    void*           cookie;
    hipFileStatus_t status;
    size_t          ret;
} hipFileIOEvents_t;

typedef void* hipFileBatchHandle_t;

typedef enum hipFileSizeTConfigParameter_t
{
    hipFileParamProfileStats,
    hipFileParamExecutionMaxIOQueueDepth,
    hipFileParamExecutionMaxIOThreads,
    hipFileParamExecutionMinIOThresholdSizeKB,
    hipFileParamExecutionMaxRequestParallelism,
    hipFileParamPropertiesMaxDirectIOSizeKB,
    hipFileParamPropertiesMaxDeviceCacheSizeKB,
    hipFileParamPropertiesPerBufferCacheSizeKB,
    hipFileParamPropertiesMaxDevicePinnedMemSizeKB,
    hipFileParamPropertiesIOBatchsize,
    hipFileParamPollthresholdSizeKB,
    hipFileParamPropertiesBatchIOTimeoutMs,
} hipFileSizeTConfigParameter_t;

typedef enum hipFileBoolConfigParameter_t
{
    hipFileParamPropertiesUsePollMode,
    hipFileParamPropertiesAllowCompatMode,
    hipFileParamForceCompatMode,
    hipFileParamFsMiscApiCheckAggressive,
    hipFileParamExecutionParallelIO,
    hipFileParamProfileNvtx,
    hipFileParamPropertiesAllowSystemMemory,
    hipFileParamUsePcip2pdma,
    hipFileParamPreferIOUring,
    hipFileParamForceOdirectMode,
    hipFileParamSkipTopologyDetection,
    hipFileParamStreamMemopsBypass,
} hipFileBoolConfigParameter_t;

typedef enum hipFileStringConfigParameter_t
{
    hipFileParamLoggingLevel,
    hipFileParamEnvLogfilePath,
    hipFileParamLogDir,
} hipFileStringConfigParameter_t;

HIPFILE_API const char*
hipFileGetOpErrorString(hipFileOpError_t status);
HIPFILE_API hipFileError_t
hipFileHandleRegister(hipFileHandle_t* fh, hipFileDescr_t* descr);
HIPFILE_API void
hipFileHandleDeregister(hipFileHandle_t fh);
HIPFILE_API hipFileError_t
hipFileBufRegister(const void* buffer_base, size_t length, int flags);
HIPFILE_API hipFileError_t
hipFileBufDeregister(const void* buffer_base);
HIPFILE_API ssize_t
hipFileRead(hipFileHandle_t fh,
            void*           buffer_base,
            size_t          size,
            hoff_t          file_offset,
            hoff_t          buffer_offset);
HIPFILE_API ssize_t
hipFileWrite(hipFileHandle_t fh,
             const void*     buffer_base,
             size_t          size,
             hoff_t          file_offset,
             hoff_t          buffer_offset);
HIPFILE_API hipFileError_t
hipFileDriverOpen(void);
HIPFILE_API hipFileError_t
hipFileDriverClose(void);
HIPFILE_API int64_t
hipFileUseCount(void);
HIPFILE_API hipFileError_t
hipFileDriverGetProperties(hipFileDriverProps_t* props);
HIPFILE_API hipFileError_t
hipFileDriverSetPollMode(bool poll, size_t poll_threshold_size);
HIPFILE_API hipFileError_t
hipFileDriverSetMaxDirectIOSize(size_t max_direct_io_size);
HIPFILE_API hipFileError_t
hipFileDriverSetMaxCacheSize(size_t max_cache_size);
HIPFILE_API hipFileError_t
hipFileDriverSetMaxPinnedMemSize(size_t max_pinned_size);
HIPFILE_API hipFileError_t
hipFileBatchIOSetUp(hipFileBatchHandle_t* batch_idp, unsigned max_nr);
HIPFILE_API hipFileError_t
hipFileBatchIOSubmit(hipFileBatchHandle_t batch_idp,
                     unsigned             nr,
                     hipFileIOParams_t*   iocbp,
                     unsigned             flags);
HIPFILE_API hipFileError_t
hipFileBatchIOGetStatus(hipFileBatchHandle_t batch_idp,
                        unsigned             min_nr,
                        unsigned*            nr,
                        hipFileIOEvents_t*   iocbp,
                        struct timespec*     timeout);
HIPFILE_API hipFileError_t
hipFileBatchIOCancel(hipFileBatchHandle_t batch_idp);
HIPFILE_API void
hipFileBatchIODestroy(hipFileBatchHandle_t batch_idp);
HIPFILE_API hipFileError_t
hipFileReadAsync(hipFileHandle_t fh,
                 void*           buffer_base,
                 size_t*         size_p,
                 hoff_t*         file_offset_p,
                 hoff_t*         buffer_offset_p,
                 ssize_t*        bytes_read_p,
                 hipStream_t     stream);
HIPFILE_API hipFileError_t
hipFileWriteAsync(hipFileHandle_t fh,
                  void*           buffer_base,
                  size_t*         size_p,
                  hoff_t*         file_offset_p,
                  hoff_t*         buffer_offset_p,
                  ssize_t*        bytes_written_p,
                  hipStream_t     stream);
HIPFILE_API hipFileError_t
hipFileStreamRegister(hipStream_t stream, unsigned flags);
HIPFILE_API hipFileError_t
hipFileStreamDeregister(hipStream_t stream);
HIPFILE_API hipFileError_t
hipFileGetVersion(unsigned* major, unsigned* minor, unsigned* patch);
HIPFILE_API hipFileError_t
hipFileGetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t* value);
HIPFILE_API hipFileError_t
hipFileGetParameterBool(hipFileBoolConfigParameter_t param, bool* value);
HIPFILE_API hipFileError_t
hipFileGetParameterString(hipFileStringConfigParameter_t param, char* desc_str, int len);
HIPFILE_API hipFileError_t
hipFileSetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t value);
HIPFILE_API hipFileError_t
hipFileSetParameterBool(hipFileBoolConfigParameter_t param, bool value);
HIPFILE_API hipFileError_t
hipFileSetParameterString(hipFileStringConfigParameter_t param, const char* desc_str);

#    ifdef __cplusplus
}
#    endif

#endif
