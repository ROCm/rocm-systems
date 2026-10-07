// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler-sdk/hipfile/api_args.h>

#include "fmt/core.h"
#include "fmt/ranges.h"

#include <type_traits>

#define ROCP_SDK_HIPFILE_FORMATTER(TYPE, ...)                                                      \
    template <>                                                                                    \
    struct formatter<TYPE> : rocprofiler::hipfile::details::base_formatter                         \
    {                                                                                              \
        template <typename Ctx>                                                                    \
        auto format(const TYPE& v, Ctx& ctx) const                                                 \
        {                                                                                          \
            return fmt::format_to(ctx.out(), __VA_ARGS__);                                         \
        }                                                                                          \
    };

#define ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(VALUE)                                                   \
    case VALUE:                                                                                    \
        return fmt::format_to(ctx.out(), #VALUE)

#define ROCP_SDK_HIPFILE_FORMAT_UNKNOWN(TYPE)                                                      \
    return fmt::format_to(                                                                         \
        ctx.out(), "{}_UNKNOWN={}", #TYPE, static_cast<std::underlying_type_t<TYPE>>(v))

namespace rocprofiler
{
namespace hipfile
{
namespace details
{
struct base_formatter
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }
};
}  // namespace details
}  // namespace hipfile
}  // namespace rocprofiler

namespace fmt
{
template <>
struct formatter<hipFileOpError_t> : rocprofiler::hipfile::details::base_formatter
{
    template <typename Ctx>
    auto format(hipFileOpError_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileSuccess);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileDriverNotInitialized);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileDriverInvalidProps);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileDriverUnsupportedLimit);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileDriverVersionMismatch);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileDriverVersionReadError);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileDriverClosing);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFilePlatformNotSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileIONotSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileDeviceNotSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileDriverError);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileHipDriverError);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileHipPointerInvalid);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileHipMemoryTypeInvalid);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileHipPointerRangeError);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileHipContextMismatch);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileInvalidMappingSize);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileInvalidMappingRange);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileInvalidFileType);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileInvalidFileOpenFlag);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileDIONotSet);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileInvalidValue);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileMemoryAlreadyRegistered);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileMemoryNotRegistered);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFilePermissionDenied);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileDriverAlreadyOpen);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileHandleNotRegistered);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileHandleAlreadyRegistered);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileDeviceNotFound);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileInternalError);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileGetNewFDFailed);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileDriverSetupError);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileIODisabled);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileBatchSubmitFailed);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileGPUMemoryPinningFailed);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileBatchFull);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileAsyncNotSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileIOMaxError);
        }
        ROCP_SDK_HIPFILE_FORMAT_UNKNOWN(hipFileOpError_t);
    }
};

template <>
struct formatter<hipFileDriverStatusFlags_t> : rocprofiler::hipfile::details::base_formatter
{
    template <typename Ctx>
    auto format(hipFileDriverStatusFlags_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileLustreSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileWekaFSSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileNFSSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileGPFSSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileNVMeSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileNVMeoFSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileSCSISupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileScaleFluxCSDSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileNVMeshSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileBeeGFSSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileNVMeP2PSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileScatefsSupported);
        }
        ROCP_SDK_HIPFILE_FORMAT_UNKNOWN(hipFileDriverStatusFlags_t);
    }
};

template <>
struct formatter<hipFileDriverControlFlags_t> : rocprofiler::hipfile::details::base_formatter
{
    template <typename Ctx>
    auto format(hipFileDriverControlFlags_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileUsePollMode);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileAllowCompatMode);
        }
        ROCP_SDK_HIPFILE_FORMAT_UNKNOWN(hipFileDriverControlFlags_t);
    }
};

template <>
struct formatter<hipFileFeatureFlags_t> : rocprofiler::hipfile::details::base_formatter
{
    template <typename Ctx>
    auto format(hipFileFeatureFlags_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileDynRoutingSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileBatchIOSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileStreamsSupported);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParallelIOSupported);
        }
        ROCP_SDK_HIPFILE_FORMAT_UNKNOWN(hipFileFeatureFlags_t);
    }
};

template <>
struct formatter<hipFileFileHandleType_t> : rocprofiler::hipfile::details::base_formatter
{
    template <typename Ctx>
    auto format(hipFileFileHandleType_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileHandleTypeOpaqueFD);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileHandleTypeOpaqueWin32);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileHandleTypeUserspaceFS);
        }
        ROCP_SDK_HIPFILE_FORMAT_UNKNOWN(hipFileFileHandleType_t);
    }
};

template <>
struct formatter<hipFileOpcode_t> : rocprofiler::hipfile::details::base_formatter
{
    template <typename Ctx>
    auto format(hipFileOpcode_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileBatchRead);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileBatchWrite);
        }
        ROCP_SDK_HIPFILE_FORMAT_UNKNOWN(hipFileOpcode_t);
    }
};

template <>
struct formatter<hipFileStatus_t> : rocprofiler::hipfile::details::base_formatter
{
    template <typename Ctx>
    auto format(hipFileStatus_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileWaiting);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFilePending);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileInvalid);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileCanceled);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileComplete);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileTimeout);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileFailed);
        }
        ROCP_SDK_HIPFILE_FORMAT_UNKNOWN(hipFileStatus_t);
    }
};

template <>
struct formatter<hipFileBatchMode_t> : rocprofiler::hipfile::details::base_formatter
{
    template <typename Ctx>
    auto format(hipFileBatchMode_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileBatch);
        }
        ROCP_SDK_HIPFILE_FORMAT_UNKNOWN(hipFileBatchMode_t);
    }
};

template <>
struct formatter<hipFileSizeTConfigParameter_t> : rocprofiler::hipfile::details::base_formatter
{
    template <typename Ctx>
    auto format(hipFileSizeTConfigParameter_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamProfileStats);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamExecutionMaxIOQueueDepth);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamExecutionMaxIOThreads);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamExecutionMinIOThresholdSizeKB);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamExecutionMaxRequestParallelism);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamPropertiesMaxDirectIOSizeKB);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamPropertiesMaxDeviceCacheSizeKB);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamPropertiesPerBufferCacheSizeKB);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamPropertiesMaxDevicePinnedMemSizeKB);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamPropertiesIOBatchsize);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamPollthresholdSizeKB);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamPropertiesBatchIOTimeoutMs);
        }
        ROCP_SDK_HIPFILE_FORMAT_UNKNOWN(hipFileSizeTConfigParameter_t);
    }
};

template <>
struct formatter<hipFileBoolConfigParameter_t> : rocprofiler::hipfile::details::base_formatter
{
    template <typename Ctx>
    auto format(hipFileBoolConfigParameter_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamPropertiesUsePollMode);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamPropertiesAllowCompatMode);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamForceCompatMode);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamFsMiscApiCheckAggressive);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamExecutionParallelIO);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamProfileNvtx);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamPropertiesAllowSystemMemory);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamUsePcip2pdma);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamPreferIOUring);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamForceOdirectMode);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamSkipTopologyDetection);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamStreamMemopsBypass);
        }
        ROCP_SDK_HIPFILE_FORMAT_UNKNOWN(hipFileBoolConfigParameter_t);
    }
};

template <>
struct formatter<hipFileStringConfigParameter_t> : rocprofiler::hipfile::details::base_formatter
{
    template <typename Ctx>
    auto format(hipFileStringConfigParameter_t v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamLoggingLevel);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamEnvLogfilePath);
            ROCP_SDK_HIPFILE_FORMAT_CASE_STMT(hipFileParamLogDir);
        }
        ROCP_SDK_HIPFILE_FORMAT_UNKNOWN(hipFileStringConfigParameter_t);
    }
};

ROCP_SDK_HIPFILE_FORMATTER(hipFileError_t,
                           "{}err={}, hip_drv_err={}{}",
                           '{',
                           v.err,
                           static_cast<int>(v.hip_drv_err),
                           '}')

ROCP_SDK_HIPFILE_FORMATTER(hipFileRDMAInfo_t,
                           "{}version={}, desc_len={}, desc_str={}{}",
                           '{',
                           v.version,
                           v.desc_len,
                           (v.desc_str) ? v.desc_str : "(null)",
                           '}')

template <>
struct formatter<hipFileDescr_t> : rocprofiler::hipfile::details::base_formatter
{
    template <typename Ctx>
    auto format(const hipFileDescr_t& v, Ctx& ctx) const
    {
        if(v.type == hipFileHandleTypeOpaqueWin32)
            return fmt::format_to(ctx.out(),
                                  "{}type={}, hFile={}, fs_ops={}{}",
                                  '{',
                                  v.type,
                                  v.handle.hFile,
                                  static_cast<const void*>(v.fs_ops),
                                  '}');
        return fmt::format_to(ctx.out(),
                              "{}type={}, fd={}, fs_ops={}{}",
                              '{',
                              v.type,
                              v.handle.fd,
                              static_cast<const void*>(v.fs_ops),
                              '}');
    }
};

ROCP_SDK_HIPFILE_FORMATTER(
    hipFileDriverProps_t,
    "{}nvfs={}, major_version={}, minor_version={}, poll_thresh_size={}, "
    "max_direct_io_size={}, driver_status_flags={}, driver_control_flags={}{}"
    ", feature_flags={}, max_device_cache_size={}, per_buffer_cache_size={}, "
    "max_device_pinned_mem_size={}, max_batch_io_count={}, "
    "max_batch_io_timeout_msecs={}{}",
    '{',
    '{',
    v.nvfs.major_version,
    v.nvfs.minor_version,
    v.nvfs.poll_thresh_size,
    v.nvfs.max_direct_io_size,
    v.nvfs.driver_status_flags,
    v.nvfs.driver_control_flags,
    '}',
    v.feature_flags,
    v.max_device_cache_size,
    v.per_buffer_cache_size,
    v.max_device_pinned_mem_size,
    v.max_batch_io_count,
    v.max_batch_io_timeout_msecs,
    '}')

ROCP_SDK_HIPFILE_FORMATTER(hipFileIOParams_t,
                           "{}mode={}, batch={}devPtr_base={}, file_offset={}, devPtr_offset={}, "
                           "size={}{}"
                           ", fh={}, opcode={}, cookie={}{}",
                           '{',
                           v.mode,
                           '{',
                           v.u.batch.devPtr_base,
                           v.u.batch.file_offset,
                           v.u.batch.devPtr_offset,
                           v.u.batch.size,
                           '}',
                           v.fh,
                           v.opcode,
                           v.cookie,
                           '}')

ROCP_SDK_HIPFILE_FORMATTER(hipFileIOEvents_t,
                           "{}cookie={}, status={}, ret={}{}",
                           '{',
                           v.cookie,
                           v.status,
                           v.ret,
                           '}')

ROCP_SDK_HIPFILE_FORMATTER(timespec, "{}tv_sec={}, tv_nsec={}{}", '{', v.tv_sec, v.tv_nsec, '}')

#if HIPFILE_RUNTIME_API_TABLE_STEP_VERSION >= 1
ROCP_SDK_HIPFILE_FORMATTER(hipFileOpCounter_t, "{}ok={}, err={}{}", '{', v.ok, v.err, '}')

ROCP_SDK_HIPFILE_FORMATTER(
    hipFileStatsLevel1_t,
    "{}read_ops={}, write_ops={}, hdl_register_ops={}, hdl_deregister_ops={}, "
    "buf_register_ops={}, buf_deregister_ops={}, read_bytes={}, write_bytes={}, "
    "read_bw_bytes_per_sec={}, write_bw_bytes_per_sec={}, read_lat_avg_us={}, "
    "write_lat_avg_us={}, read_ops_per_sec={}, write_ops_per_sec={}, read_lat_sum_us={}, "
    "write_lat_sum_us={}, batch_submit_ops={}, batch_complete_ops={}, batch_setup_ops={}, "
    "batch_cancel_ops={}, batch_destroy_ops={}, batch_enqueued_ops={}, "
    "batch_posix_enqueued_ops={}, batch_processed_ops={}, batch_posix_processed_ops={}, "
    "batch_nvfs_submit_ops={}, batch_p2p_submit_ops={}, batch_aio_submit_ops={}, "
    "batch_iouring_submit_ops={}, batch_mixed_io_submit_ops={}, batch_total_submit_ops={}, "
    "batch_read_bytes={}, batch_write_bytes={}, batch_read_bw_bytes={}, "
    "batch_write_bw_bytes={}, batch_submit_lat_avg_us={}, batch_completion_lat_avg_us={}, "
    "batch_submit_ops_per_sec={}, batch_complete_ops_per_sec={}, batch_submit_lat_sum_us={}, "
    "batch_completion_lat_sum_us={}, last_batch_read_bytes={}, last_batch_write_bytes={}{}",
    '{',
    v.read_ops,
    v.write_ops,
    v.hdl_register_ops,
    v.hdl_deregister_ops,
    v.buf_register_ops,
    v.buf_deregister_ops,
    v.read_bytes,
    v.write_bytes,
    v.read_bw_bytes_per_sec,
    v.write_bw_bytes_per_sec,
    v.read_lat_avg_us,
    v.write_lat_avg_us,
    v.read_ops_per_sec,
    v.write_ops_per_sec,
    v.read_lat_sum_us,
    v.write_lat_sum_us,
    v.batch_submit_ops,
    v.batch_complete_ops,
    v.batch_setup_ops,
    v.batch_cancel_ops,
    v.batch_destroy_ops,
    v.batch_enqueued_ops,
    v.batch_posix_enqueued_ops,
    v.batch_processed_ops,
    v.batch_posix_processed_ops,
    v.batch_nvfs_submit_ops,
    v.batch_p2p_submit_ops,
    v.batch_aio_submit_ops,
    v.batch_iouring_submit_ops,
    v.batch_mixed_io_submit_ops,
    v.batch_total_submit_ops,
    v.batch_read_bytes,
    v.batch_write_bytes,
    v.batch_read_bw_bytes,
    v.batch_write_bw_bytes,
    v.batch_submit_lat_avg_us,
    v.batch_completion_lat_avg_us,
    v.batch_submit_ops_per_sec,
    v.batch_complete_ops_per_sec,
    v.batch_submit_lat_sum_us,
    v.batch_completion_lat_sum_us,
    v.last_batch_read_bytes,
    v.last_batch_write_bytes,
    '}')

ROCP_SDK_HIPFILE_FORMATTER(hipFileStatsLevel2_t,
                           "{}basic={}, read_size_kb_hist=[{}], write_size_kb_hist=[{}]{}",
                           '{',
                           v.basic,
                           fmt::join(v.read_size_kb_hist, v.read_size_kb_hist + 32, ", "),
                           fmt::join(v.write_size_kb_hist, v.write_size_kb_hist + 32, ", "),
                           '}')

template <>
struct formatter<hipFilePerGpuStats_t> : rocprofiler::hipfile::details::base_formatter
{
    template <typename Ctx>
    auto format(const hipFilePerGpuStats_t& v, Ctx& ctx) const
    {
        auto out = fmt::format_to(ctx.out(), "{}uuid=[", '{');
        for(size_t i = 0; i < HIPFILE_GPU_UUID_LEN; ++i)
        {
            out = fmt::format_to(
                out, "{}{:02x}", (i > 0) ? ", " : "", static_cast<unsigned char>(v.uuid[i]));
        }
        return fmt::format_to(
            out,
            "], read_bytes={}, read_bw_bytes_per_sec={}, read_utilization={}, "
            "read_duration_us={}, n_total_reads={}, n_p2p_reads={}, n_nvfs_reads={}, "
            "n_posix_reads={}, n_unaligned_reads={}, n_dr_reads={}, n_sparse_regions={}, "
            "n_inline_regions={}, n_reads_err={}, write_bytes={}, write_bw_bytes_per_sec={}, "
            "write_utilization={}, write_duration_us={}, n_total_writes={}, n_p2p_writes={}, "
            "n_nvfs_writes={}, n_posix_writes={}, n_unaligned_writes={}, n_dr_writes={}, "
            "n_writes_err={}, n_mmap={}, n_mmap_ok={}, n_mmap_err={}, n_mmap_free={}, "
            "reg_bytes={}{}",
            v.read_bytes,
            v.read_bw_bytes_per_sec,
            v.read_utilization,
            v.read_duration_us,
            v.n_total_reads,
            v.n_p2p_reads,
            v.n_nvfs_reads,
            v.n_posix_reads,
            v.n_unaligned_reads,
            v.n_dr_reads,
            v.n_sparse_regions,
            v.n_inline_regions,
            v.n_reads_err,
            v.write_bytes,
            v.write_bw_bytes_per_sec,
            v.write_utilization,
            v.write_duration_us,
            v.n_total_writes,
            v.n_p2p_writes,
            v.n_nvfs_writes,
            v.n_posix_writes,
            v.n_unaligned_writes,
            v.n_dr_writes,
            v.n_writes_err,
            v.n_mmap,
            v.n_mmap_ok,
            v.n_mmap_err,
            v.n_mmap_free,
            v.reg_bytes,
            '}');
    }
};

ROCP_SDK_HIPFILE_FORMATTER(hipFileStatsLevel3_t,
                           "{}detailed={}, num_gpus={}, per_gpu_stats=[{}]{}",
                           '{',
                           v.detailed,
                           v.num_gpus,
                           fmt::join(v.per_gpu_stats, v.per_gpu_stats + HIPFILE_MAX_GPUS, ", "),
                           '}')
#endif
}  // namespace fmt

#undef ROCP_SDK_HIPFILE_FORMAT_UNKNOWN
#undef ROCP_SDK_HIPFILE_FORMAT_CASE_STMT
#undef ROCP_SDK_HIPFILE_FORMATTER
