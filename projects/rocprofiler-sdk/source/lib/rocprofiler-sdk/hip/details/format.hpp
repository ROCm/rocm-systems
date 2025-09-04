// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/rocprofiler-sdk/hip/details/ostream.hpp"

#include <rocprofiler-sdk/hip.h>

#include <hip/hip_runtime_api.h>
// must be included after runtime api
#include <hip/hip_deprecated.h>
#include <hip/hip_version.h>

#include "fmt/core.h"
#include "fmt/ranges.h"

#define ROCP_SDK_HIP_FORMATTER(TYPE, ...)                                                          \
    template <>                                                                                    \
    struct formatter<TYPE> : rocprofiler::hip::details::base_formatter                             \
    {                                                                                              \
        template <typename Ctx>                                                                    \
        auto format(const TYPE& v, Ctx& ctx) const                                                 \
        {                                                                                          \
            return fmt::format_to(ctx.out(), __VA_ARGS__);                                         \
        }                                                                                          \
    };

#define ROCP_SDK_HIP_OSTREAM_FORMATTER(TYPE)                                                       \
    template <>                                                                                    \
    struct formatter<TYPE> : rocprofiler::hip::details::base_formatter                             \
    {                                                                                              \
        template <typename Ctx>                                                                    \
        auto format(const TYPE& v, Ctx& ctx) const                                                 \
        {                                                                                          \
            auto _ss = std::stringstream{};                                                        \
            _ss << v;                                                                              \
            return fmt::format_to(ctx.out(), "{}", _ss.str());                                     \
        }                                                                                          \
    };

#define ROCP_SDK_HIP_FORMAT_CASE_STMT(PREFIX, SUFFIX)                                              \
    case PREFIX##SUFFIX: return fmt::format_to(ctx.out(), #SUFFIX)

// provide a default case when not building in CI
#if defined(ROCPROFILER_CI)
#    define ROCP_SDK_HIP_FORMAT_DFLT_CASE(PREFIX)
#else
#    define ROCP_SDK_HIP_FORMAT_DFLT_CASE(PREFIX)                                                  \
        default:                                                                                   \
            return fmt::format_to(ctx.out(), "{}_UNKNOWN={}", #PREFIX, static_cast<int>(v))
#endif

namespace rocprofiler
{
namespace hip
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
}  // namespace hip
}  // namespace rocprofiler

namespace fmt
{
template <>
struct formatter<rocprofiler_dim3_t> : rocprofiler::hip::details::base_formatter
{
    template <typename Ctx>
    auto format(const rocprofiler_dim3_t& v, Ctx& ctx) const
    {
        return fmt::format_to(ctx.out(), "{}z={}, y={}, x={}{}", '{', v.z, v.y, v.x, '}');
    }
};

ROCP_SDK_HIP_OSTREAM_FORMATTER(hipExtent)
ROCP_SDK_HIP_OSTREAM_FORMATTER(dim3)
ROCP_SDK_HIP_OSTREAM_FORMATTER(hipPitchedPtr)
ROCP_SDK_HIP_OSTREAM_FORMATTER(hipPos)
ROCP_SDK_HIP_OSTREAM_FORMATTER(hipMemcpy3DParms)
ROCP_SDK_HIP_OSTREAM_FORMATTER(hipMemAllocNodeParams)
ROCP_SDK_HIP_OSTREAM_FORMATTER(hipMemsetParams)
ROCP_SDK_HIP_OSTREAM_FORMATTER(hipKernelNodeParams)
ROCP_SDK_HIP_OSTREAM_FORMATTER(hipHostNodeParams)
ROCP_SDK_HIP_OSTREAM_FORMATTER(hipExternalSemaphoreSignalNodeParams)
ROCP_SDK_HIP_OSTREAM_FORMATTER(hipExternalSemaphoreWaitNodeParams)
ROCP_SDK_HIP_OSTREAM_FORMATTER(hipMemPoolProps)
ROCP_SDK_HIP_OSTREAM_FORMATTER(hipCtx_t)

ROCP_SDK_HIP_FORMATTER(hipMemcpyNodeParams,
                       "{}flags={}, copyParams={}{}",
                       '{',
                       v.flags,
                       v.copyParams,
                       '}')
ROCP_SDK_HIP_FORMATTER(hipChildGraphNodeParams,
                       "{}graph={}{}",
                       '{',
                       static_cast<void*>(v.graph),
                       '}')
ROCP_SDK_HIP_FORMATTER(hipEventWaitNodeParams,
                       "{}event={}{}",
                       '{',
                       static_cast<void*>(v.event),
                       '}')
ROCP_SDK_HIP_FORMATTER(hipEventRecordNodeParams,
                       "{}event={}{}",
                       '{',
                       static_cast<void*>(v.event),
                       '}')

ROCP_SDK_HIP_FORMATTER(hipMemFreeNodeParams, "{}dptr={}{}", '{', v.dptr, '}')
ROCP_SDK_HIP_FORMATTER(hipGraphInstantiateParams,
                       "{}errNode_out={}, flags={}, result_out={}, uploadStream={}{}",
                       '{',
                       static_cast<void*>(v.errNode_out),
                       v.flags,
                       v.result_out,
                       static_cast<void*>(v.uploadStream),
                       '}')
ROCP_SDK_HIP_FORMATTER(hipGraphEdgeData,
                       "{}from_port={}, to_port={}, type={}{}",
                       '{',
                       v.from_port,
                       v.to_port,
                       v.type,
                       '}')
#if HIP_RUNTIME_API_TABLE_STEP_VERSION < 13
ROCP_SDK_HIP_FORMATTER(HIP_MEMSET_NODE_PARAMS,
                       "{}dst={}, pitch={}, value={}, elementSize={}, width={}, height={}{}",
                       '{',
                       v.dst,
                       v.pitch,
                       v.value,
                       v.elementSize,
                       v.width,
                       v.height,
                       '}')
#endif
ROCP_SDK_HIP_FORMATTER(hipMemLocation, "{}type={}, id={}{}", '{', v.type, v.id, '}')
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 7
ROCP_SDK_HIP_FORMATTER(
    hipStreamBatchMemOpParams,
    "{}operation={}, waitValueOperation={}, waitValueAddress={}, waitValueValue={}, "
    "waitValueValue64={}, waitValueFlags={}, writeValueOperation={}, writeValueAddress={}, "
    "writeValueValue={}, writeValueValue64={}, writeValueFlags={}, flushRemoteWritesOperation={}, "
    "flushRemoteWritesFlags={}, memoryBarrierOperation={}, memoryBarrierFlags={}{}",
    '{',
    v.operation,
    v.waitValue.operation,
    v.waitValue.address,
    v.waitValue.value,
    v.waitValue.value64,
    v.waitValue.flags,
    v.writeValue.operation,
    v.writeValue.address,
    v.writeValue.value,
    v.writeValue.value64,
    v.writeValue.flags,
    v.flushRemoteWrites.operation,
    v.flushRemoteWrites.flags,
    v.memoryBarrier.operation,
    v.memoryBarrier.flags,
    '}')
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 8
ROCP_SDK_HIP_FORMATTER(hipBatchMemOpNodeParams,
                       "{}ctx={}, count={}, paramArray=[{}], flags={}{}",
                       '{',
                       static_cast<void*>(v.ctx),
                       v.count,
                       fmt::join(v.paramArray, v.paramArray + v.count, ", "),
                       v.flags,
                       '}')
#endif

template <>
struct formatter<hipGraphNodeType> : rocprofiler::hip::details::base_formatter
{
    template <typename Ctx>
    auto format(hipGraphNodeType v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, Kernel);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, Memcpy);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, Memset);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, Host);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, Graph);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, WaitEvent);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, EventRecord);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, ExtSemaphoreSignal);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, ExtSemaphoreWait);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, MemAlloc);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, MemFree);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, MemcpyFromSymbol);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, MemcpyToSymbol);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, Empty);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, Count);
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 7
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphNodeType, BatchMemOp);
#endif
            ROCP_SDK_HIP_FORMAT_DFLT_CASE(hipGraphNodeType);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<hipGraphInstantiateResult> : rocprofiler::hip::details::base_formatter
{
    template <typename Ctx>
    auto format(hipGraphInstantiateResult v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphInstantiate, Success);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphInstantiate, Error);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphInstantiate, InvalidStructure);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphInstantiate, NodeOperationNotSupported);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipGraphInstantiate, MultipleDevicesNotSupported);
            ROCP_SDK_HIP_FORMAT_DFLT_CASE(hipGraphInstantiate);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<hipMemAllocationType> : rocprofiler::hip::details::base_formatter
{
    template <typename Ctx>
    auto format(hipMemAllocationType v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemAllocationType, Invalid);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemAllocationType, Pinned);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemAllocationType, Max);
            ROCP_SDK_HIP_FORMAT_DFLT_CASE(hipMemAllocationType);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<hipMemLocationType> : rocprofiler::hip::details::base_formatter
{
    template <typename Ctx>
    auto format(hipMemLocationType v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemLocationType, Invalid);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemLocationType, Device);
            ROCP_SDK_HIP_FORMAT_DFLT_CASE(hipMemLocationType);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<hipMemAllocationHandleType> : rocprofiler::hip::details::base_formatter
{
    template <typename Ctx>
    auto format(hipMemAllocationHandleType v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemHandleType, None);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemHandleType, PosixFileDescriptor);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemHandleType, Win32);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemHandleType, Win32Kmt);
            ROCP_SDK_HIP_FORMAT_DFLT_CASE(hipMemHandleType);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<hipMemcpyKind> : rocprofiler::hip::details::base_formatter
{
    template <typename Ctx>
    auto format(hipMemcpyKind v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemcpy, HostToHost);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemcpy, HostToDevice);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemcpy, DeviceToHost);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemcpy, DeviceToDevice);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemcpy, Default);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipMemcpy, DeviceToDeviceNoCU);
            ROCP_SDK_HIP_FORMAT_DFLT_CASE(hipMemcpy);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<hipGraphNodeParams> : rocprofiler::hip::details::base_formatter
{
    template <typename Ctx>
    auto format(const hipGraphNodeParams& v, Ctx& ctx) const
    {
        switch(v.type)
        {
            case hipGraphNodeTypeKernel:
                return fmt::format_to(
                    ctx.out(), "{}type={}, kernel={}{}", '{', v.type, v.kernel, '}');
            case hipGraphNodeTypeMemcpy:
                return fmt::format_to(
                    ctx.out(), "{}type={}, memcpy={}{}", '{', v.type, v.memcpy, '}');
            case hipGraphNodeTypeMemset:
                return fmt::format_to(
                    ctx.out(), "{}type={}, memset={}{}", '{', v.type, v.memset, '}');
            case hipGraphNodeTypeHost:
                return fmt::format_to(ctx.out(), "{}type={}, host={}{}", '{', v.type, v.host, '}');
            case hipGraphNodeTypeGraph:
                return fmt::format_to(
                    ctx.out(), "{}type={}, graph={}{}", '{', v.type, v.graph, '}');
            case hipGraphNodeTypeWaitEvent:
                return fmt::format_to(
                    ctx.out(), "{}type={}, eventWait={}{}", '{', v.type, v.eventWait, '}');
            case hipGraphNodeTypeEventRecord:
                return fmt::format_to(
                    ctx.out(), "{}type={}, eventRecord={}{}", '{', v.type, v.eventRecord, '}');
            case hipGraphNodeTypeExtSemaphoreSignal:
                return fmt::format_to(
                    ctx.out(), "{}type={}, extSemSignal={}{}", '{', v.type, v.extSemSignal, '}');
            case hipGraphNodeTypeExtSemaphoreWait:
                return fmt::format_to(
                    ctx.out(), "{}type={}, extSemWait={}{}", '{', v.type, v.extSemWait, '}');
            case hipGraphNodeTypeMemAlloc:
                return fmt::format_to(
                    ctx.out(), "{}type={}, alloc={}{}", '{', v.type, v.alloc, '}');
            case hipGraphNodeTypeMemFree:
                return fmt::format_to(ctx.out(), "{}type={}, free={}{}", '{', v.type, v.free, '}');
            case hipGraphNodeTypeMemcpyFromSymbol:
            case hipGraphNodeTypeMemcpyToSymbol:
            case hipGraphNodeTypeEmpty:
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 8
            case hipGraphNodeTypeBatchMemOp:
#endif
            case hipGraphNodeTypeCount:
            {
                break;
            }
        }
        return fmt::format_to(ctx.out(), "{}type={}{}", '{', v.type, '}');
    }
};

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 7
template <>
struct formatter<hipStreamBatchMemOpType> : rocprofiler::hip::details::base_formatter
{
    template <typename Ctx>
    auto format(hipStreamBatchMemOpType v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipStreamMemOp, WaitValue32);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipStreamMemOp, WriteValue32);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipStreamMemOp, WaitValue64);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipStreamMemOp, WriteValue64);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipStreamMemOp, Barrier);
            ROCP_SDK_HIP_FORMAT_CASE_STMT(hipStreamMemOp, FlushRemoteWrites);
            ROCP_SDK_HIP_FORMAT_DFLT_CASE(hipStreamMemOp);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 11
ROCP_SDK_HIP_FORMATTER(
    hipLaunchConfig_st,
    "{}gridDim={}, blockDim={}, dynamicSmemBytes={}, stream={}, attrs={}, numAttrs={}",
    '{',
    v.gridDim,
    v.blockDim,
    v.dynamicSmemBytes,
    static_cast<void*>(v.stream),
    static_cast<void*>(v.attrs),
    v.numAttrs,
    '}')

ROCP_SDK_HIP_FORMATTER(HIP_LAUNCH_CONFIG_st,
                       "{}gridDimX={}, gridDimY={}, gridDimZ={}, blockDimX={}, blockDimY={}, "
                       "blockDimZ={}, sharedMemBytes={}, hStream={}, attrs={}, numAttrs={}",
                       '{',
                       v.gridDimX,
                       v.gridDimY,
                       v.gridDimZ,
                       v.blockDimX,
                       v.blockDimY,
                       v.blockDimZ,
                       v.sharedMemBytes,
                       static_cast<void*>(v.hStream),
                       static_cast<void*>(v.attrs),
                       v.numAttrs,
                       '}')
#endif
}  // namespace fmt

#undef ROCP_SDK_HIP_FORMATTER
#undef ROCP_SDK_HIP_OSTREAM_FORMATTER
#undef ROCP_SDK_HIP_FORMAT_CASE_STMT
#undef ROCP_SDK_HIP_FORMAT_DFLT_CASE
