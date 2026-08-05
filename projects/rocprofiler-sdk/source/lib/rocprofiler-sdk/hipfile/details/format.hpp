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
}  // namespace fmt

#undef ROCP_SDK_HIPFILE_FORMAT_UNKNOWN
#undef ROCP_SDK_HIPFILE_FORMAT_CASE_STMT
#undef ROCP_SDK_HIPFILE_FORMATTER
