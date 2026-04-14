/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile.h"
#include "hipfile-cufile.h"
#include "hipfile-warnings.h"
#include "invalid-enum.h"
#include "test-common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <stdexcept>

#include <cuda.h>
#include <cufile.h>

/* These tests exist to ensure that the hipFile API stays compatible
 * with the cuFile API. They check that things like struct members,
 * enum values, etc. align with cufile.h.
 */

// Put tests inside the macros to suppress the global constructor
// warnings
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

TEST(ToHipFile, hipFileOpError_t)
{
    ASSERT_THROW(toHipFileOpError(invalidEnum<CUfileOpError>(CU_FILE_SUCCESS - 1)), std::invalid_argument);
    ASSERT_THROW(toHipFileOpError(invalidEnum<CUfileOpError>(CU_FILE_SUCCESS + 1)), std::invalid_argument);

    ASSERT_THROW(toHipFileOpError(invalidEnum<CUfileOpError>(CUFILEOP_BASE_ERR)), std::invalid_argument);
    ASSERT_THROW(toHipFileOpError(invalidEnum<CUfileOpError>(CUFILEOP_BASE_ERR + 21)), std::invalid_argument);
    ASSERT_THROW(toHipFileOpError(invalidEnum<CUfileOpError>(CUFILEOP_BASE_ERR + 32)), std::invalid_argument);

#define c2h(C, H) ASSERT_EQ(toHipFileOpError((C)), (H))
    c2h(CU_FILE_SUCCESS, hipFileSuccess);
    c2h(CU_FILE_DRIVER_NOT_INITIALIZED, hipFileDriverNotInitialized);
    c2h(CU_FILE_DRIVER_INVALID_PROPS, hipFileDriverInvalidProps);
    c2h(CU_FILE_DRIVER_UNSUPPORTED_LIMIT, hipFileDriverUnsupportedLimit);
    c2h(CU_FILE_DRIVER_VERSION_MISMATCH, hipFileDriverVersionMismatch);
    c2h(CU_FILE_DRIVER_VERSION_READ_ERROR, hipFileDriverVersionReadError);
    c2h(CU_FILE_DRIVER_CLOSING, hipFileDriverClosing);
    c2h(CU_FILE_PLATFORM_NOT_SUPPORTED, hipFilePlatformNotSupported);
    c2h(CU_FILE_IO_NOT_SUPPORTED, hipFileIONotSupported);
    c2h(CU_FILE_DEVICE_NOT_SUPPORTED, hipFileDeviceNotSupported);
    c2h(CU_FILE_NVFS_DRIVER_ERROR, hipFileDriverError);
    c2h(CU_FILE_CUDA_DRIVER_ERROR, hipFileHipDriverError);
    c2h(CU_FILE_CUDA_POINTER_INVALID, hipFileHipPointerInvalid);
    c2h(CU_FILE_CUDA_MEMORY_TYPE_INVALID, hipFileHipMemoryTypeInvalid);
    c2h(CU_FILE_CUDA_POINTER_RANGE_ERROR, hipFileHipPointerRangeError);
    c2h(CU_FILE_CUDA_CONTEXT_MISMATCH, hipFileHipContextMismatch);
    c2h(CU_FILE_INVALID_MAPPING_SIZE, hipFileInvalidMappingSize);
    c2h(CU_FILE_INVALID_MAPPING_RANGE, hipFileInvalidMappingRange);
    c2h(CU_FILE_INVALID_FILE_TYPE, hipFileInvalidFileType);
    c2h(CU_FILE_INVALID_FILE_OPEN_FLAG, hipFileInvalidFileOpenFlag);
    c2h(CU_FILE_DIO_NOT_SET, hipFileDIONotSet);
    c2h(CU_FILE_INVALID_VALUE, hipFileInvalidValue);
    c2h(CU_FILE_MEMORY_ALREADY_REGISTERED, hipFileMemoryAlreadyRegistered);
    c2h(CU_FILE_MEMORY_NOT_REGISTERED, hipFileMemoryNotRegistered);
    c2h(CU_FILE_PERMISSION_DENIED, hipFilePermissionDenied);
    c2h(CU_FILE_DRIVER_ALREADY_OPEN, hipFileDriverAlreadyOpen);
    c2h(CU_FILE_HANDLE_NOT_REGISTERED, hipFileHandleNotRegistered);
    c2h(CU_FILE_HANDLE_ALREADY_REGISTERED, hipFileHandleAlreadyRegistered);
    c2h(CU_FILE_DEVICE_NOT_FOUND, hipFileDeviceNotFound);
    c2h(CU_FILE_INTERNAL_ERROR, hipFileInternalError);
    c2h(CU_FILE_GETNEWFD_FAILED, hipFileGetNewFDFailed);
    c2h(CU_FILE_NVFS_SETUP_ERROR, hipFileDriverSetupError);
    c2h(CU_FILE_IO_DISABLED, hipFileIODisabled);
    c2h(CU_FILE_BATCH_SUBMIT_FAILED, hipFileBatchSubmitFailed);
    c2h(CU_FILE_GPU_MEMORY_PINNING_FAILED, hipFileGPUMemoryPinningFailed);
    c2h(CU_FILE_BATCH_FULL, hipFileBatchFull);
    c2h(CU_FILE_ASYNC_NOT_SUPPORTED, hipFileAsyncNotSupported);
    c2h(CU_FILE_IO_MAX_ERROR, hipFileIOMaxError);
#undef c2h

    ASSERT_THROW(toHipFileOpError(invalidEnum<CUfileOpError>(CU_FILE_IO_MAX_ERROR + 1)),
                 std::invalid_argument);
}

TEST(ToHipFile, hipFileError_t)
{
    ASSERT_THROW(
        (void)toHipFileError(CUfileError_t{invalidEnum<CUfileOpError>(CU_FILE_SUCCESS - 1), CUDA_SUCCESS}),
        std::invalid_argument);

    ASSERT_EQ(toHipFileError(CUfileError_t{CU_FILE_SUCCESS, CUDA_SUCCESS}), HIPFILE_SUCCESS);
    ASSERT_EQ(toHipFileError(CUfileError_t{CU_FILE_CUDA_DRIVER_ERROR, CUDA_ERROR_INVALID_VALUE}),
              HipFileDriverError(hipErrorInvalidValue));
    ASSERT_EQ(toHipFileError(CUfileError_t{CU_FILE_DRIVER_NOT_INITIALIZED, CUDA_SUCCESS}),
              HipFileOpError(hipFileDriverNotInitialized));
}

TEST(ToHipFile, hipFileDriverProps_t)
{
    CUfileDrvProps_t cdp;

    rfill(static_cast<void *>(&cdp), sizeof(cdp));

    hipFileDriverProps_t hdp = toHipFileDriverProps(cdp);

    ASSERT_EQ(hdp.nvfs.major_version, cdp.nvfs.major_version);
    ASSERT_EQ(hdp.nvfs.minor_version, cdp.nvfs.minor_version);
    ASSERT_EQ(hdp.nvfs.poll_thresh_size, cdp.nvfs.poll_thresh_size);
    ASSERT_EQ(hdp.nvfs.max_direct_io_size, cdp.nvfs.max_direct_io_size);
    ASSERT_EQ(hdp.nvfs.driver_status_flags, cdp.nvfs.dstatusflags);
    ASSERT_EQ(hdp.nvfs.driver_control_flags, cdp.nvfs.dcontrolflags);
    ASSERT_EQ(hdp.feature_flags, cdp.fflags);
    ASSERT_EQ(hdp.max_device_cache_size, cdp.max_device_cache_size);
    ASSERT_EQ(hdp.per_buffer_cache_size, cdp.per_buffer_cache_size);
    ASSERT_EQ(hdp.max_device_pinned_mem_size, cdp.max_device_pinned_mem_size);
    ASSERT_EQ(hdp.max_batch_io_count, cdp.max_batch_io_size);
    ASSERT_EQ(hdp.max_batch_io_timeout_msecs, cdp.max_batch_io_timeout_msecs);
}

TEST(toHipFile, hipFileStatus_t)
{
    ASSERT_THROW(toHipFileStatus(invalidEnum<CUfileStatus_t>(CUFILE_WAITING - 1)), std::invalid_argument);

#define c2h(C, H) ASSERT_EQ(toHipFileStatus((C)), (H))
    c2h(CUFILE_WAITING, hipFileWaiting);
    c2h(CUFILE_PENDING, hipFilePending);
    c2h(CUFILE_INVALID, hipFileInvalid);
    c2h(CUFILE_CANCELED, hipFileCanceled);
    c2h(CUFILE_COMPLETE, hipFileComplete);
    c2h(CUFILE_TIMEOUT, hipFileTimeout);
    c2h(CUFILE_FAILED, hipFileFailed);
#undef c2h

    ASSERT_THROW(toHipFileStatus(invalidEnum<CUfileStatus_t>(CUFILE_FAILED + 1)), std::invalid_argument);
}

TEST(ToHipFile, hipFileIOEvents_t)
{
    {
        CUfileIOEvents_t cioe{};
        cioe.status = invalidEnum<CUfileStatus_t>(CUFILE_WAITING - 1);

        ASSERT_THROW(toHipFileIOEvents(cioe), std::invalid_argument);
    }

    {
        CUfileIOEvents_t cioe{};
        cioe.status = invalidEnum<CUfileStatus_t>(CUFILE_FAILED + 1);

        ASSERT_THROW(toHipFileIOEvents(cioe), std::invalid_argument);
    }

    {
        CUfileIOEvents_t cioe;

        rfill(static_cast<void *>(&cioe), sizeof(cioe));
        cioe.status = CUFILE_COMPLETE;

        auto hioe = toHipFileIOEvents(cioe);
        ASSERT_EQ(hioe.status, hipFileComplete);
        ASSERT_EQ(hioe.cookie, cioe.cookie);
        ASSERT_EQ(hioe.ret, cioe.ret);
    }
}

TEST(ToCufile, CUFileSizeTConfigParameter_t)
{
    ASSERT_THROW(toCUFileSizeTConfigParameter(
                     invalidEnum<hipFileSizeTConfigParameter_t>(hipFileParamProfileStats - 1)),
                 std::invalid_argument);

#define h2c(H, C) ASSERT_EQ(toCUFileSizeTConfigParameter((H)), (C))
    h2c(hipFileParamProfileStats, CUFILE_PARAM_PROFILE_STATS);
    h2c(hipFileParamExecutionMaxIOQueueDepth, CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH);
    h2c(hipFileParamExecutionMaxIOThreads, CUFILE_PARAM_EXECUTION_MAX_IO_THREADS);
    h2c(hipFileParamExecutionMinIOThresholdSizeKB, CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB);
    h2c(hipFileParamExecutionMaxRequestParallelism, CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM);
    h2c(hipFileParamPropertiesMaxDirectIOSizeKB, CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB);
    h2c(hipFileParamPropertiesMaxDeviceCacheSizeKB, CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB);
    h2c(hipFileParamPropertiesPerBufferCacheSizeKB, CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB);
    h2c(hipFileParamPropertiesMaxDevicePinnedMemSizeKB,
        CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB);
    h2c(hipFileParamPropertiesIOBatchsize, CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE);
    h2c(hipFileParamPollthresholdSizeKB, CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB);
    h2c(hipFileParamPropertiesBatchIOTimeoutMs, CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS);
#undef h2c

    ASSERT_THROW(toCUFileSizeTConfigParameter(
                     invalidEnum<hipFileSizeTConfigParameter_t>(hipFileParamPropertiesBatchIOTimeoutMs + 1)),
                 std::invalid_argument);
}

TEST(ToCufile, CUFileBoolConfigParameter_t)
{
    ASSERT_THROW(toCUFileBoolConfigParameter(
                     invalidEnum<hipFileBoolConfigParameter_t>(hipFileParamPropertiesUsePollMode - 1)),
                 std::invalid_argument);

#define h2c(H, C) ASSERT_EQ(toCUFileBoolConfigParameter((H)), (C))
    h2c(hipFileParamPropertiesUsePollMode, CUFILE_PARAM_PROPERTIES_USE_POLL_MODE);
    h2c(hipFileParamPropertiesAllowCompatMode, CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE);
    h2c(hipFileParamForceCompatMode, CUFILE_PARAM_FORCE_COMPAT_MODE);
    h2c(hipFileParamFsMiscApiCheckAggressive, CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE);
    h2c(hipFileParamExecutionParallelIO, CUFILE_PARAM_EXECUTION_PARALLEL_IO);
    h2c(hipFileParamProfileNvtx, CUFILE_PARAM_PROFILE_NVTX);
    h2c(hipFileParamPropertiesAllowSystemMemory, CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY);
    h2c(hipFileParamUsePcip2pdma, CUFILE_PARAM_USE_PCIP2PDMA);
    h2c(hipFileParamPreferIOUring, CUFILE_PARAM_PREFER_IO_URING);
    h2c(hipFileParamForceOdirectMode, CUFILE_PARAM_FORCE_ODIRECT_MODE);
    h2c(hipFileParamSkipTopologyDetection, CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION);
    h2c(hipFileParamStreamMemopsBypass, CUFILE_PARAM_STREAM_MEMOPS_BYPASS);
#undef h2c

    ASSERT_THROW(toCUFileBoolConfigParameter(
                     invalidEnum<hipFileBoolConfigParameter_t>(hipFileParamStreamMemopsBypass + 1)),
                 std::invalid_argument);
}

TEST(ToCufile, CUFileStringConfigParameter_t)
{
    ASSERT_THROW(toCUFileStringConfigParameter(
                     invalidEnum<hipFileStringConfigParameter_t>(hipFileParamLoggingLevel - 1)),
                 std::invalid_argument);

#define h2c(H, C) ASSERT_EQ(toCUFileStringConfigParameter((H)), (C))
    h2c(hipFileParamLoggingLevel, CUFILE_PARAM_LOGGING_LEVEL);
    h2c(hipFileParamEnvLogfilePath, CUFILE_PARAM_ENV_LOGFILE_PATH);
    h2c(hipFileParamLogDir, CUFILE_PARAM_LOG_DIR);
#undef h2c

    ASSERT_THROW(
        toCUFileStringConfigParameter(invalidEnum<hipFileStringConfigParameter_t>(hipFileParamLogDir + 1)),
        std::invalid_argument);
}

TEST(ToCufile, CUfileDescr_t)
{
    {
        hipFileDescr_t hfd{};
        hfd.type = invalidEnum<hipFileFileHandleType>(hipFileHandleTypeOpaqueFD - 1);
        ASSERT_THROW(toCUfileDescr(hfd), std::invalid_argument);
    }

    {
        hipFileDescr_t hfd{};
        hfd.type      = hipFileHandleTypeOpaqueFD;
        hfd.handle.fd = 0x1234;
        hfd.fs_ops    = reinterpret_cast<const hipFileFSOps_t *>(0xCAFEBABE);

        auto cfd = toCUfileDescr(hfd);
        ASSERT_EQ(cfd.type, CU_FILE_HANDLE_TYPE_OPAQUE_FD);
        ASSERT_EQ(cfd.handle.fd, hfd.handle.fd);
        ASSERT_EQ(static_cast<const void *>(cfd.fs_ops), static_cast<const void *>(hfd.fs_ops));
    }

    {
        hipFileDescr_t hfd{};
        hfd.type         = hipFileHandleTypeOpaqueWin32;
        hfd.handle.hFile = reinterpret_cast<void *>(0x5678);
        hfd.fs_ops       = reinterpret_cast<const hipFileFSOps_t *>(0xBADF00D);

        auto cfd = toCUfileDescr(hfd);
        ASSERT_EQ(cfd.type, CU_FILE_HANDLE_TYPE_OPAQUE_WIN32);
        ASSERT_EQ(cfd.handle.handle, hfd.handle.hFile);
        ASSERT_EQ(static_cast<const void *>(cfd.fs_ops), static_cast<const void *>(hfd.fs_ops));
    }

    {
        hipFileDescr_t hfd{};
        hfd.type      = hipFileHandleTypeUserspaceFS;
        hfd.handle.fd = 0x90AB;
        hfd.fs_ops    = reinterpret_cast<const hipFileFSOps_t *>(0xFACEFEED);

        auto cfd = toCUfileDescr(hfd);
        ASSERT_EQ(cfd.type, CU_FILE_HANDLE_TYPE_USERSPACE_FS);
        ASSERT_EQ(cfd.handle.fd, hfd.handle.fd);
        ASSERT_EQ(static_cast<const void *>(cfd.fs_ops), static_cast<const void *>(hfd.fs_ops));
    }

    {
        hipFileDescr_t hfd{};
        hfd.type = invalidEnum<hipFileFileHandleType>(hipFileHandleTypeUserspaceFS + 1);
        ASSERT_THROW(toCUfileDescr(hfd), std::invalid_argument);
    }
}

TEST(ToCufile, CUfileFileHandleType)
{
    ASSERT_THROW(toCUfileFileHandleType(invalidEnum<hipFileFileHandleType>(hipFileHandleTypeOpaqueFD - 1)),
                 std::invalid_argument);

#define h2c(H, C) ASSERT_EQ(toCUfileFileHandleType((H)), (C))
    h2c(hipFileHandleTypeOpaqueFD, CU_FILE_HANDLE_TYPE_OPAQUE_FD);
    h2c(hipFileHandleTypeOpaqueWin32, CU_FILE_HANDLE_TYPE_OPAQUE_WIN32);
    h2c(hipFileHandleTypeUserspaceFS, CU_FILE_HANDLE_TYPE_USERSPACE_FS);
#undef h2c

    ASSERT_THROW(toCUfileFileHandleType(invalidEnum<hipFileFileHandleType>(hipFileHandleTypeUserspaceFS + 1)),
                 std::invalid_argument);
}

TEST(ToCufile, CUfileBatchMode_t)
{
    ASSERT_THROW(toCUfileBatchMode(invalidEnum<hipFileBatchMode_t>(hipFileBatch - 1)), std::invalid_argument);

    ASSERT_EQ(toCUfileBatchMode(hipFileBatch), CUFILE_BATCH);

    ASSERT_THROW(toCUfileBatchMode(invalidEnum<hipFileBatchMode_t>(hipFileBatch + 1)), std::invalid_argument);
}

TEST(ToCufile, CUfileOpcode_t)
{
    ASSERT_THROW(toCUfileOpcode(invalidEnum<hipFileOpcode_t>(hipFileBatchRead - 1)), std::invalid_argument);

#define h2c(H, C) ASSERT_EQ(toCUfileOpcode((H)), (C))
    h2c(hipFileBatchRead, CUFILE_READ);
    h2c(hipFileBatchWrite, CUFILE_WRITE);
#undef h2c

    ASSERT_THROW(toCUfileOpcode(invalidEnum<hipFileOpcode_t>(hipFileBatchWrite + 1)), std::invalid_argument);
}

TEST(ToCufile, CUfileIOParams_t)
{
    {
        hipFileIOParams_t hiop{};
        hiop.mode   = invalidEnum<hipFileBatchMode_t>(hipFileBatch + 1);
        hiop.opcode = hipFileBatchWrite;

        ASSERT_THROW(toCUfileIOParams(hiop), std::invalid_argument);
    }

    {
        hipFileIOParams_t hiop{};
        hiop.mode   = hipFileBatch;
        hiop.opcode = invalidEnum<hipFileOpcode_t>(hipFileBatchWrite + 1);

        ASSERT_THROW(toCUfileIOParams(hiop), std::invalid_argument);
    }

    {
        hipFileIOParams_t hiop;

        rfill(static_cast<void *>(&hiop), sizeof(hiop));

        hiop.mode   = hipFileBatch;
        hiop.opcode = hipFileBatchWrite;

        auto ciop = toCUfileIOParams(hiop);
        ASSERT_EQ(ciop.mode, CUFILE_BATCH);
        ASSERT_EQ(ciop.u.batch.devPtr_base, hiop.u.batch.devPtr_base);
        ASSERT_EQ(ciop.u.batch.file_offset, hiop.u.batch.file_offset);
        ASSERT_EQ(ciop.u.batch.devPtr_offset, hiop.u.batch.devPtr_offset);
        ASSERT_EQ(ciop.u.batch.size, hiop.u.batch.size);
        ASSERT_EQ(ciop.fh, ciop.fh);
        ASSERT_EQ(ciop.opcode, CUFILE_WRITE);
        ASSERT_EQ(ciop.cookie, hiop.cookie);
    }
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
