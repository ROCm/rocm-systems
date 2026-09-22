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

#include <rocprofiler-sdk/hip.h>

#include <hip/hip_runtime_api.h>
// must be included after runtime api
#include <hip/hip_deprecated.h>

#include <cstdio>
#include <iomanip>
#include <ostream>
#include <string>
#include <string_view>

namespace rocprofiler
{
namespace hip
{
namespace detail
{
static int              HIP_depth_max     = 1;
static thread_local int HIP_depth_max_cnt = 0;
static std::string_view HIP_structs_regex = {};

inline static void
print_escaped_string(std::ostream& out, const char* v, size_t len)
{
    out << '"';
    for(size_t i = 0; i < len && v[i] != '\0'; ++i)
    {
        switch(v[i])
        {
            case '\"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\\b"; break;
            case '\f': out << "\\\f"; break;
            case '\n': out << "\\\n"; break;
            case '\r': out << "\\\r"; break;
            case '\t': out << "\\\t"; break;
            default:
                if(std::isprint((unsigned char) v[i]) != 0)
                    std::operator<<(out, v[i]);
                else
                {
                    std::ios_base::fmtflags flags(out.flags());
                    out << "\\x" << std::setfill('0') << std::setw(2) << std::hex
                        << (unsigned int) (unsigned char) v[i];
                    out.flags(flags);
                }
                break;
        }
    }
    out << '"';
}

template <typename T>
inline static std::ostream&
operator<<(std::ostream& out, const T& v)
{
    using std::              operator<<;
    static thread_local bool recursion = false;
    if(recursion == false)
    {
        recursion = true;
        out << v;
        recursion = false;
    }
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const unsigned char& v)
{
    out << (unsigned int) v;
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const char& v)
{
    out << (unsigned char) v;
    return out;
}

template <size_t N>
inline static std::ostream&
operator<<(std::ostream& out, const char (&v)[N])
{
    print_escaped_string(out, v, N);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const char* v)
{
    print_escaped_string(out, v, strlen(v));
    return out;
}
// End of basic ostream ops

inline static std::ostream&
operator<<(std::ostream& out, const __locale_struct& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("__locale_struct::__names").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "__names=");
            ::rocprofiler::hip::detail::operator<<(out, v.__names);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("__locale_struct::__ctype_toupper").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "__ctype_toupper=");
            ::rocprofiler::hip::detail::operator<<(out, v.__ctype_toupper);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("__locale_struct::__ctype_tolower").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "__ctype_tolower=");
            ::rocprofiler::hip::detail::operator<<(out, v.__ctype_tolower);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("__locale_struct::__ctype_b").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "__ctype_b=");
            ::rocprofiler::hip::detail::operator<<(out, v.__ctype_b);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("__locale_struct::__locales").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "__locales=");
            ::rocprofiler::hip::detail::operator<<(out, v.__locales);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipDeviceArch_t& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipDeviceArch_t::hasDynamicParallelism").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasDynamicParallelism=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasDynamicParallelism);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::has3dGrid").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "has3dGrid=");
            ::rocprofiler::hip::detail::operator<<(out, v.has3dGrid);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasSurfaceFuncs").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasSurfaceFuncs=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasSurfaceFuncs);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasSyncThreadsExt").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasSyncThreadsExt=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasSyncThreadsExt);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasThreadFenceSystem").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasThreadFenceSystem=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasThreadFenceSystem);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasFunnelShift").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasFunnelShift=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasFunnelShift);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasWarpShuffle").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasWarpShuffle=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasWarpShuffle);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasWarpBallot").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasWarpBallot=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasWarpBallot);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasWarpVote").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasWarpVote=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasWarpVote);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasDoubles").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasDoubles=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasDoubles);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasSharedInt64Atomics").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasSharedInt64Atomics=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasSharedInt64Atomics);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasGlobalInt64Atomics").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasGlobalInt64Atomics=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasGlobalInt64Atomics);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasFloatAtomicAdd").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasFloatAtomicAdd=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasFloatAtomicAdd);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasSharedFloatAtomicExch").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasSharedFloatAtomicExch=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasSharedFloatAtomicExch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasSharedInt32Atomics").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasSharedInt32Atomics=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasSharedInt32Atomics);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasGlobalFloatAtomicExch").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasGlobalFloatAtomicExch=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasGlobalFloatAtomicExch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceArch_t::hasGlobalInt32Atomics").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hasGlobalInt32Atomics=");
            ::rocprofiler::hip::detail::operator<<(out, v.hasGlobalInt32Atomics);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipUUID& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipUUID::bytes").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "bytes=");
            ::rocprofiler::hip::detail::operator<<(out, v.bytes);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipDeviceProp_tR0600& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipDeviceProp_t::asicRevision").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "asicRevision=");
            ::rocprofiler::hip::detail::operator<<(out, v.asicRevision);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::isLargeBar").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "isLargeBar=");
            ::rocprofiler::hip::detail::operator<<(out, v.isLargeBar);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::cooperativeMultiDeviceUnmatchedSharedMem")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::operator<<(out, "cooperativeMultiDeviceUnmatchedSharedMem=");
            ::rocprofiler::hip::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedSharedMem);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::cooperativeMultiDeviceUnmatchedBlockDim")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "cooperativeMultiDeviceUnmatchedBlockDim=");
            ::rocprofiler::hip::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedBlockDim);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::cooperativeMultiDeviceUnmatchedGridDim")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "cooperativeMultiDeviceUnmatchedGridDim=");
            ::rocprofiler::hip::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedGridDim);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::cooperativeMultiDeviceUnmatchedFunc")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "cooperativeMultiDeviceUnmatchedFunc=");
            ::rocprofiler::hip::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedFunc);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::hdpRegFlushCntl").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hdpRegFlushCntl=");
            ::rocprofiler::hip::detail::operator<<(out, v.hdpRegFlushCntl);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::hdpMemFlushCntl").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hdpMemFlushCntl=");
            ::rocprofiler::hip::detail::operator<<(out, v.hdpMemFlushCntl);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::arch").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "arch=");
            ::rocprofiler::hip::detail::operator<<(out, v.arch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::clockInstructionRate").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "clockInstructionRate=");
            ::rocprofiler::hip::detail::operator<<(out, v.clockInstructionRate);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxSharedMemoryPerMultiProcessor")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "maxSharedMemoryPerMultiProcessor=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxSharedMemoryPerMultiProcessor);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::gcnArchName").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "gcnArchName=");
            ::rocprofiler::hip::detail::operator<<(out, v.gcnArchName);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::hipReserved").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hipReserved=");
            ::rocprofiler::hip::detail::operator<<(out, v.hipReserved);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::reserved").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "reserved=");
            ::rocprofiler::hip::detail::operator<<(out, 0);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::unifiedFunctionPointers").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "unifiedFunctionPointers=");
            ::rocprofiler::hip::detail::operator<<(out, v.unifiedFunctionPointers);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::clusterLaunch").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "clusterLaunch=");
            ::rocprofiler::hip::detail::operator<<(out, v.clusterLaunch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::ipcEventSupported").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "ipcEventSupported=");
            ::rocprofiler::hip::detail::operator<<(out, v.ipcEventSupported);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::deferredMappingHipArraySupported")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "deferredMappingHipArraySupported=");
            ::rocprofiler::hip::detail::operator<<(out, v.deferredMappingHipArraySupported);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::memoryPoolSupportedHandleTypes")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "memoryPoolSupportedHandleTypes=");
            ::rocprofiler::hip::detail::operator<<(out, v.memoryPoolSupportedHandleTypes);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::gpuDirectRDMAWritesOrdering")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "gpuDirectRDMAWritesOrdering=");
            ::rocprofiler::hip::detail::operator<<(out, v.gpuDirectRDMAWritesOrdering);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::gpuDirectRDMAFlushWritesOptions")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "gpuDirectRDMAFlushWritesOptions=");
            ::rocprofiler::hip::detail::operator<<(out, v.gpuDirectRDMAFlushWritesOptions);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::gpuDirectRDMASupported").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "gpuDirectRDMASupported=");
            ::rocprofiler::hip::detail::operator<<(out, v.gpuDirectRDMASupported);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::memoryPoolsSupported").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "memoryPoolsSupported=");
            ::rocprofiler::hip::detail::operator<<(out, v.memoryPoolsSupported);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::timelineSemaphoreInteropSupported")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "timelineSemaphoreInteropSupported=");
            ::rocprofiler::hip::detail::operator<<(out, v.timelineSemaphoreInteropSupported);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::hostRegisterReadOnlySupported")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "hostRegisterReadOnlySupported=");
            ::rocprofiler::hip::detail::operator<<(out, v.hostRegisterReadOnlySupported);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::sparseHipArraySupported").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "sparseHipArraySupported=");
            ::rocprofiler::hip::detail::operator<<(out, v.sparseHipArraySupported);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::hostRegisterSupported").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hostRegisterSupported=");
            ::rocprofiler::hip::detail::operator<<(out, v.hostRegisterSupported);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::reservedSharedMemPerBlock").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "reservedSharedMemPerBlock=");
            ::rocprofiler::hip::detail::operator<<(out, v.reservedSharedMemPerBlock);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::accessPolicyMaxWindowSize").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "accessPolicyMaxWindowSize=");
            ::rocprofiler::hip::detail::operator<<(out, v.accessPolicyMaxWindowSize);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxBlocksPerMultiProcessor")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "maxBlocksPerMultiProcessor=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxBlocksPerMultiProcessor);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::directManagedMemAccessFromHost")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "directManagedMemAccessFromHost=");
            ::rocprofiler::hip::detail::operator<<(out, v.directManagedMemAccessFromHost);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::pageableMemoryAccessUsesHostPageTables")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "pageableMemoryAccessUsesHostPageTables=");
            ::rocprofiler::hip::detail::operator<<(out, v.pageableMemoryAccessUsesHostPageTables);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::sharedMemPerBlockOptin").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "sharedMemPerBlockOptin=");
            ::rocprofiler::hip::detail::operator<<(out, v.sharedMemPerBlockOptin);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::cooperativeMultiDeviceLaunch")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "cooperativeMultiDeviceLaunch=");
            ::rocprofiler::hip::detail::operator<<(out, v.cooperativeMultiDeviceLaunch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::cooperativeLaunch").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "cooperativeLaunch=");
            ::rocprofiler::hip::detail::operator<<(out, v.cooperativeLaunch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::canUseHostPointerForRegisteredMem")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "canUseHostPointerForRegisteredMem=");
            ::rocprofiler::hip::detail::operator<<(out, v.canUseHostPointerForRegisteredMem);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::computePreemptionSupported")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "computePreemptionSupported=");
            ::rocprofiler::hip::detail::operator<<(out, v.computePreemptionSupported);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::concurrentManagedAccess").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "concurrentManagedAccess=");
            ::rocprofiler::hip::detail::operator<<(out, v.concurrentManagedAccess);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::pageableMemoryAccess").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "pageableMemoryAccess=");
            ::rocprofiler::hip::detail::operator<<(out, v.pageableMemoryAccess);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::singleToDoublePrecisionPerfRatio")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "singleToDoublePrecisionPerfRatio=");
            ::rocprofiler::hip::detail::operator<<(out, v.singleToDoublePrecisionPerfRatio);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::hostNativeAtomicSupported").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hostNativeAtomicSupported=");
            ::rocprofiler::hip::detail::operator<<(out, v.hostNativeAtomicSupported);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::multiGpuBoardGroupID").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "multiGpuBoardGroupID=");
            ::rocprofiler::hip::detail::operator<<(out, v.multiGpuBoardGroupID);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::isMultiGpuBoard").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "isMultiGpuBoard=");
            ::rocprofiler::hip::detail::operator<<(out, v.isMultiGpuBoard);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::managedMemory").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "managedMemory=");
            ::rocprofiler::hip::detail::operator<<(out, v.managedMemory);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::regsPerMultiprocessor").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "regsPerMultiprocessor=");
            ::rocprofiler::hip::detail::operator<<(out, v.regsPerMultiprocessor);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::sharedMemPerMultiprocessor")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "sharedMemPerMultiprocessor=");
            ::rocprofiler::hip::detail::operator<<(out, v.sharedMemPerMultiprocessor);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::localL1CacheSupported").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "localL1CacheSupported=");
            ::rocprofiler::hip::detail::operator<<(out, v.localL1CacheSupported);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::globalL1CacheSupported").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "globalL1CacheSupported=");
            ::rocprofiler::hip::detail::operator<<(out, v.globalL1CacheSupported);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::streamPrioritiesSupported").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "streamPrioritiesSupported=");
            ::rocprofiler::hip::detail::operator<<(out, v.streamPrioritiesSupported);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxThreadsPerMultiProcessor")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "maxThreadsPerMultiProcessor=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxThreadsPerMultiProcessor);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::persistingL2CacheMaxSize").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "persistingL2CacheMaxSize=");
            ::rocprofiler::hip::detail::operator<<(out, v.persistingL2CacheMaxSize);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::l2CacheSize").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "l2CacheSize=");
            ::rocprofiler::hip::detail::operator<<(out, v.l2CacheSize);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::memoryBusWidth").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "memoryBusWidth=");
            ::rocprofiler::hip::detail::operator<<(out, v.memoryBusWidth);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::memoryClockRate").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "memoryClockRate=");
            ::rocprofiler::hip::detail::operator<<(out, v.memoryClockRate);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::unifiedAddressing").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "unifiedAddressing=");
            ::rocprofiler::hip::detail::operator<<(out, v.unifiedAddressing);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::asyncEngineCount").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "asyncEngineCount=");
            ::rocprofiler::hip::detail::operator<<(out, v.asyncEngineCount);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::tccDriver").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "tccDriver=");
            ::rocprofiler::hip::detail::operator<<(out, v.tccDriver);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::pciDomainID").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "pciDomainID=");
            ::rocprofiler::hip::detail::operator<<(out, v.pciDomainID);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::pciDeviceID").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "pciDeviceID=");
            ::rocprofiler::hip::detail::operator<<(out, v.pciDeviceID);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::pciBusID").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "pciBusID=");
            ::rocprofiler::hip::detail::operator<<(out, v.pciBusID);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::ECCEnabled").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "ECCEnabled=");
            ::rocprofiler::hip::detail::operator<<(out, v.ECCEnabled);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::concurrentKernels").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "concurrentKernels=");
            ::rocprofiler::hip::detail::operator<<(out, v.concurrentKernels);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::surfaceAlignment").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "surfaceAlignment=");
            ::rocprofiler::hip::detail::operator<<(out, v.surfaceAlignment);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxSurfaceCubemapLayered").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxSurfaceCubemapLayered=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxSurfaceCubemapLayered);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxSurfaceCubemap").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxSurfaceCubemap=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxSurfaceCubemap);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxSurface2DLayered").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxSurface2DLayered=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxSurface2DLayered);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxSurface1DLayered").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxSurface1DLayered=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxSurface1DLayered);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxSurface3D").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxSurface3D=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxSurface3D);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxSurface2D").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxSurface2D=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxSurface2D);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxSurface1D").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxSurface1D=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxSurface1D);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxTextureCubemapLayered").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTextureCubemapLayered=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTextureCubemapLayered);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxTexture2DLayered").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture2DLayered=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture2DLayered);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxTexture1DLayered").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture1DLayered=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture1DLayered);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxTextureCubemap").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTextureCubemap=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTextureCubemap);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxTexture3DAlt").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture3DAlt=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture3DAlt);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxTexture3D").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture3D=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture3D);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxTexture2DGather").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture2DGather=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture2DGather);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxTexture2DLinear").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture2DLinear=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture2DLinear);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxTexture2DMipmap").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture2DMipmap=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture2DMipmap);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxTexture2D").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture2D=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture2D);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxTexture1DLinear").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture1DLinear=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture1DLinear);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxTexture1DMipmap").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture1DMipmap=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture1DMipmap);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxTexture1D").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture1D=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture1D);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::computeMode").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "computeMode=");
            ::rocprofiler::hip::detail::operator<<(out, v.computeMode);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::canMapHostMemory").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "canMapHostMemory=");
            ::rocprofiler::hip::detail::operator<<(out, v.canMapHostMemory);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::integrated").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "integrated=");
            ::rocprofiler::hip::detail::operator<<(out, v.integrated);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::kernelExecTimeoutEnabled").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "kernelExecTimeoutEnabled=");
            ::rocprofiler::hip::detail::operator<<(out, v.kernelExecTimeoutEnabled);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::multiProcessorCount").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "multiProcessorCount=");
            ::rocprofiler::hip::detail::operator<<(out, v.multiProcessorCount);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::deviceOverlap").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "deviceOverlap=");
            ::rocprofiler::hip::detail::operator<<(out, v.deviceOverlap);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::texturePitchAlignment").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "texturePitchAlignment=");
            ::rocprofiler::hip::detail::operator<<(out, v.texturePitchAlignment);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::textureAlignment").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "textureAlignment=");
            ::rocprofiler::hip::detail::operator<<(out, v.textureAlignment);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::minor").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "minor=");
            ::rocprofiler::hip::detail::operator<<(out, v.minor);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::major").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "major=");
            ::rocprofiler::hip::detail::operator<<(out, v.major);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::totalConstMem").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "totalConstMem=");
            ::rocprofiler::hip::detail::operator<<(out, v.totalConstMem);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::clockRate").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "clockRate=");
            ::rocprofiler::hip::detail::operator<<(out, v.clockRate);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxGridSize").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxGridSize=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxGridSize);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxThreadsDim").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxThreadsDim=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxThreadsDim);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::maxThreadsPerBlock").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxThreadsPerBlock=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxThreadsPerBlock);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::memPitch").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "memPitch=");
            ::rocprofiler::hip::detail::operator<<(out, v.memPitch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::warpSize").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "warpSize=");
            ::rocprofiler::hip::detail::operator<<(out, v.warpSize);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::regsPerBlock").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "regsPerBlock=");
            ::rocprofiler::hip::detail::operator<<(out, v.regsPerBlock);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::sharedMemPerBlock").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "sharedMemPerBlock=");
            ::rocprofiler::hip::detail::operator<<(out, v.sharedMemPerBlock);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::totalGlobalMem").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "totalGlobalMem=");
            ::rocprofiler::hip::detail::operator<<(out, v.totalGlobalMem);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::luidDeviceNodeMask").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "luidDeviceNodeMask=");
            ::rocprofiler::hip::detail::operator<<(out, v.luidDeviceNodeMask);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::luid").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "luid=");
            ::rocprofiler::hip::detail::operator<<(out, v.luid);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::uuid").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "uuid=");
            ::rocprofiler::hip::detail::operator<<(out, v.uuid);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipDeviceProp_t::name").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "name=");
            ::rocprofiler::hip::detail::operator<<(out, v.name);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipDeviceProp_tR0000& v)
{
    using namespace ::rocprofiler::hip::detail;
    std::operator<<(out, '{');
    HIP_depth_max_cnt++;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view{"hipDeviceProp_t::pageableMemoryAccessUsesHostPageTables"}.find(
               HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "pageableMemoryAccessUsesHostPageTables=");
            ::rocprofiler::hip::detail::operator<<(out, v.pageableMemoryAccessUsesHostPageTables);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::pageableMemoryAccess"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "pageableMemoryAccess=");
            ::rocprofiler::hip::detail::operator<<(out, v.pageableMemoryAccess);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::concurrentManagedAccess"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "concurrentManagedAccess=");
            ::rocprofiler::hip::detail::operator<<(out, v.concurrentManagedAccess);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::directManagedMemAccessFromHost"}.find(
               HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "directManagedMemAccessFromHost=");
            ::rocprofiler::hip::detail::operator<<(out, v.directManagedMemAccessFromHost);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::managedMemory"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "managedMemory=");
            ::rocprofiler::hip::detail::operator<<(out, v.managedMemory);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::asicRevision"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "asicRevision=");
            ::rocprofiler::hip::detail::operator<<(out, v.asicRevision);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::isLargeBar"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "isLargeBar=");
            ::rocprofiler::hip::detail::operator<<(out, v.isLargeBar);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::cooperativeMultiDeviceUnmatchedSharedMem"}.find(
               HIP_structs_regex) != std::string_view::npos)
        {
            std::operator<<(out, "cooperativeMultiDeviceUnmatchedSharedMem=");
            ::rocprofiler::hip::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedSharedMem);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::cooperativeMultiDeviceUnmatchedBlockDim"}.find(
               HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "cooperativeMultiDeviceUnmatchedBlockDim=");
            ::rocprofiler::hip::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedBlockDim);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::cooperativeMultiDeviceUnmatchedGridDim"}.find(
               HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "cooperativeMultiDeviceUnmatchedGridDim=");
            ::rocprofiler::hip::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedGridDim);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::cooperativeMultiDeviceUnmatchedFunc"}.find(
               HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "cooperativeMultiDeviceUnmatchedFunc=");
            ::rocprofiler::hip::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedFunc);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::tccDriver"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "tccDriver=");
            ::rocprofiler::hip::detail::operator<<(out, v.tccDriver);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::ECCEnabled"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "ECCEnabled=");
            ::rocprofiler::hip::detail::operator<<(out, v.ECCEnabled);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::kernelExecTimeoutEnabled"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "kernelExecTimeoutEnabled=");
            ::rocprofiler::hip::detail::operator<<(out, v.kernelExecTimeoutEnabled);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::texturePitchAlignment"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "texturePitchAlignment=");
            ::rocprofiler::hip::detail::operator<<(out, v.texturePitchAlignment);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::textureAlignment"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "textureAlignment=");
            ::rocprofiler::hip::detail::operator<<(out, v.textureAlignment);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::memPitch"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "memPitch=");
            ::rocprofiler::hip::detail::operator<<(out, v.memPitch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::hdpRegFlushCntl"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hdpRegFlushCntl=");
            ::rocprofiler::hip::detail::operator<<(out, v.hdpRegFlushCntl);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::hdpMemFlushCntl"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hdpMemFlushCntl=");
            ::rocprofiler::hip::detail::operator<<(out, v.hdpMemFlushCntl);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::maxTexture3D"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture3D=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture3D);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::maxTexture2D"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture2D=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture2D);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::maxTexture1D"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture1D=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture1D);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::maxTexture1DLinear"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxTexture1DLinear=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxTexture1DLinear);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::cooperativeMultiDeviceLaunch"}.find(
               HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "cooperativeMultiDeviceLaunch=");
            ::rocprofiler::hip::detail::operator<<(out, v.cooperativeMultiDeviceLaunch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::cooperativeLaunch"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "cooperativeLaunch=");
            ::rocprofiler::hip::detail::operator<<(out, v.cooperativeLaunch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::integrated"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "integrated=");
            ::rocprofiler::hip::detail::operator<<(out, v.integrated);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::gcnArchName"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "gcnArchName=");
            ::rocprofiler::hip::detail::operator<<(out, v.gcnArchName);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::gcnArch"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "gcnArch=");
            ::rocprofiler::hip::detail::operator<<(out, v.gcnArch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::canMapHostMemory"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "canMapHostMemory=");
            ::rocprofiler::hip::detail::operator<<(out, v.canMapHostMemory);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::isMultiGpuBoard"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "isMultiGpuBoard=");
            ::rocprofiler::hip::detail::operator<<(out, v.isMultiGpuBoard);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::maxSharedMemoryPerMultiProcessor"}.find(
               HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "maxSharedMemoryPerMultiProcessor=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxSharedMemoryPerMultiProcessor);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::pciDeviceID"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "pciDeviceID=");
            ::rocprofiler::hip::detail::operator<<(out, v.pciDeviceID);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::pciBusID"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "pciBusID=");
            ::rocprofiler::hip::detail::operator<<(out, v.pciBusID);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::pciDomainID"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "pciDomainID=");
            ::rocprofiler::hip::detail::operator<<(out, v.pciDomainID);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::concurrentKernels"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "concurrentKernels=");
            ::rocprofiler::hip::detail::operator<<(out, v.concurrentKernels);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::arch"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "arch=");
            ::rocprofiler::hip::detail::operator<<(out, v.arch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::clockInstructionRate"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "clockInstructionRate=");
            ::rocprofiler::hip::detail::operator<<(out, v.clockInstructionRate);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::computeMode"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "computeMode=");
            ::rocprofiler::hip::detail::operator<<(out, v.computeMode);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::maxThreadsPerMultiProcessor"}.find(
               HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "maxThreadsPerMultiProcessor=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxThreadsPerMultiProcessor);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::l2CacheSize"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "l2CacheSize=");
            ::rocprofiler::hip::detail::operator<<(out, v.l2CacheSize);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::multiProcessorCount"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "multiProcessorCount=");
            ::rocprofiler::hip::detail::operator<<(out, v.multiProcessorCount);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::minor"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "minor=");
            ::rocprofiler::hip::detail::operator<<(out, v.minor);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::major"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "major=");
            ::rocprofiler::hip::detail::operator<<(out, v.major);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::totalConstMem"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "totalConstMem=");
            ::rocprofiler::hip::detail::operator<<(out, v.totalConstMem);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::memoryBusWidth"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "memoryBusWidth=");
            ::rocprofiler::hip::detail::operator<<(out, v.memoryBusWidth);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::memoryClockRate"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "memoryClockRate=");
            ::rocprofiler::hip::detail::operator<<(out, v.memoryClockRate);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::clockRate"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "clockRate=");
            ::rocprofiler::hip::detail::operator<<(out, v.clockRate);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::maxGridSize"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxGridSize=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxGridSize);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::maxThreadsDim"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxThreadsDim=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxThreadsDim);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::maxThreadsPerBlock"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxThreadsPerBlock=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxThreadsPerBlock);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::warpSize"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "warpSize=");
            ::rocprofiler::hip::detail::operator<<(out, v.warpSize);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::regsPerBlock"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "regsPerBlock=");
            ::rocprofiler::hip::detail::operator<<(out, v.regsPerBlock);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::sharedMemPerBlock"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "sharedMemPerBlock=");
            ::rocprofiler::hip::detail::operator<<(out, v.sharedMemPerBlock);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::totalGlobalMem"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "totalGlobalMem=");
            ::rocprofiler::hip::detail::operator<<(out, v.totalGlobalMem);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view{"hipDeviceProp_t::name"}.find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "name=");
            ::rocprofiler::hip::detail::operator<<(out, v.name);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipPointerAttribute_t& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipPointerAttribute_t::allocationFlags").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "allocationFlags=");
            ::rocprofiler::hip::detail::operator<<(out, v.allocationFlags);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipPointerAttribute_t::isManaged").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "isManaged=");
            ::rocprofiler::hip::detail::operator<<(out, v.isManaged);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipPointerAttribute_t::device").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "device=");
            ::rocprofiler::hip::detail::operator<<(out, v.device);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipPointerAttribute_t::type").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "type=");
            ::rocprofiler::hip::detail::operator<<(out, v.type);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipChannelFormatDesc& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipChannelFormatDesc::f").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "f=");
            ::rocprofiler::hip::detail::operator<<(out, v.f);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipChannelFormatDesc::w").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "w=");
            ::rocprofiler::hip::detail::operator<<(out, v.w);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipChannelFormatDesc::z").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipChannelFormatDesc::y").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipChannelFormatDesc::x").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const HIP_ARRAY_DESCRIPTOR& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("HIP_ARRAY_DESCRIPTOR::NumChannels").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "NumChannels=");
            ::rocprofiler::hip::detail::operator<<(out, v.NumChannels);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_ARRAY_DESCRIPTOR::Format").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "Format=");
            ::rocprofiler::hip::detail::operator<<(out, v.Format);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_ARRAY_DESCRIPTOR::Height").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "Height=");
            ::rocprofiler::hip::detail::operator<<(out, v.Height);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_ARRAY_DESCRIPTOR::Width").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "Width=");
            ::rocprofiler::hip::detail::operator<<(out, v.Width);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const HIP_ARRAY3D_DESCRIPTOR& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("HIP_ARRAY3D_DESCRIPTOR::Flags").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "Flags=");
            ::rocprofiler::hip::detail::operator<<(out, v.Flags);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_ARRAY3D_DESCRIPTOR::NumChannels").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "NumChannels=");
            ::rocprofiler::hip::detail::operator<<(out, v.NumChannels);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_ARRAY3D_DESCRIPTOR::Format").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "Format=");
            ::rocprofiler::hip::detail::operator<<(out, v.Format);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_ARRAY3D_DESCRIPTOR::Depth").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "Depth=");
            ::rocprofiler::hip::detail::operator<<(out, v.Depth);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_ARRAY3D_DESCRIPTOR::Height").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "Height=");
            ::rocprofiler::hip::detail::operator<<(out, v.Height);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_ARRAY3D_DESCRIPTOR::Width").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "Width=");
            ::rocprofiler::hip::detail::operator<<(out, v.Width);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hip_Memcpy2D& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hip_Memcpy2D::Height").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "Height=");
            ::rocprofiler::hip::detail::operator<<(out, v.Height);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hip_Memcpy2D::WidthInBytes").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "WidthInBytes=");
            ::rocprofiler::hip::detail::operator<<(out, v.WidthInBytes);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hip_Memcpy2D::dstPitch").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstPitch=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstPitch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hip_Memcpy2D::dstArray").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstArray=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstArray);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hip_Memcpy2D::dstDevice").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstDevice=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstDevice);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hip_Memcpy2D::dstMemoryType").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstMemoryType=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstMemoryType);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hip_Memcpy2D::dstY").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "dstY=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstY);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hip_Memcpy2D::dstXInBytes").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstXInBytes=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstXInBytes);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hip_Memcpy2D::srcPitch").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcPitch=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcPitch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hip_Memcpy2D::srcArray").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcArray=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcArray);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hip_Memcpy2D::srcDevice").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcDevice=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcDevice);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hip_Memcpy2D::srcMemoryType").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcMemoryType=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcMemoryType);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hip_Memcpy2D::srcY").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "srcY=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcY);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hip_Memcpy2D::srcXInBytes").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcXInBytes=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcXInBytes);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMipmappedArray& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipMipmappedArray::num_channels").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "num_channels=");
            ::rocprofiler::hip::detail::operator<<(out, v.num_channels);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMipmappedArray::format").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "format=");
            ::rocprofiler::hip::detail::operator<<(out, v.format);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMipmappedArray::flags").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "flags=");
            ::rocprofiler::hip::detail::operator<<(out, v.flags);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMipmappedArray::max_mipmap_level").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "max_mipmap_level=");
            ::rocprofiler::hip::detail::operator<<(out, v.max_mipmap_level);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMipmappedArray::min_mipmap_level").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "min_mipmap_level=");
            ::rocprofiler::hip::detail::operator<<(out, v.min_mipmap_level);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMipmappedArray::depth").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "depth=");
            ::rocprofiler::hip::detail::operator<<(out, v.depth);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMipmappedArray::height").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "height=");
            ::rocprofiler::hip::detail::operator<<(out, v.height);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMipmappedArray::width").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "width=");
            ::rocprofiler::hip::detail::operator<<(out, v.width);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMipmappedArray::type").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "type=");
            ::rocprofiler::hip::detail::operator<<(out, v.type);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMipmappedArray::desc").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "desc=");
            ::rocprofiler::hip::detail::operator<<(out, v.desc);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const HIP_TEXTURE_DESC& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("HIP_TEXTURE_DESC::reserved").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "reserved=");
            ::rocprofiler::hip::detail::operator<<(out, 0);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_TEXTURE_DESC::borderColor").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "borderColor=");
            ::rocprofiler::hip::detail::operator<<(out, v.borderColor);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_TEXTURE_DESC::maxMipmapLevelClamp").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxMipmapLevelClamp=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxMipmapLevelClamp);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_TEXTURE_DESC::minMipmapLevelClamp").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "minMipmapLevelClamp=");
            ::rocprofiler::hip::detail::operator<<(out, v.minMipmapLevelClamp);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_TEXTURE_DESC::mipmapLevelBias").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "mipmapLevelBias=");
            ::rocprofiler::hip::detail::operator<<(out, v.mipmapLevelBias);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_TEXTURE_DESC::mipmapFilterMode").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "mipmapFilterMode=");
            ::rocprofiler::hip::detail::operator<<(out, v.mipmapFilterMode);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_TEXTURE_DESC::maxAnisotropy").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxAnisotropy=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxAnisotropy);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_TEXTURE_DESC::flags").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "flags=");
            ::rocprofiler::hip::detail::operator<<(out, v.flags);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_TEXTURE_DESC::filterMode").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "filterMode=");
            ::rocprofiler::hip::detail::operator<<(out, v.filterMode);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_TEXTURE_DESC::addressMode").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "addressMode=");
            ::rocprofiler::hip::detail::operator<<(out, v.addressMode);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipResourceDesc& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipResourceDesc::resType").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "resType=");
            ::rocprofiler::hip::detail::operator<<(out, v.resType);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const HIP_RESOURCE_DESC& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("HIP_RESOURCE_DESC::flags").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "flags=");
            ::rocprofiler::hip::detail::operator<<(out, v.flags);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_RESOURCE_DESC::resType").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "resType=");
            ::rocprofiler::hip::detail::operator<<(out, v.resType);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipResourceViewDesc& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipResourceViewDesc::lastLayer").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "lastLayer=");
            ::rocprofiler::hip::detail::operator<<(out, v.lastLayer);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipResourceViewDesc::firstLayer").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "firstLayer=");
            ::rocprofiler::hip::detail::operator<<(out, v.firstLayer);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipResourceViewDesc::lastMipmapLevel").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "lastMipmapLevel=");
            ::rocprofiler::hip::detail::operator<<(out, v.lastMipmapLevel);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipResourceViewDesc::firstMipmapLevel").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "firstMipmapLevel=");
            ::rocprofiler::hip::detail::operator<<(out, v.firstMipmapLevel);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipResourceViewDesc::depth").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "depth=");
            ::rocprofiler::hip::detail::operator<<(out, v.depth);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipResourceViewDesc::height").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "height=");
            ::rocprofiler::hip::detail::operator<<(out, v.height);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipResourceViewDesc::width").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "width=");
            ::rocprofiler::hip::detail::operator<<(out, v.width);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipResourceViewDesc::format").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "format=");
            ::rocprofiler::hip::detail::operator<<(out, v.format);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const HIP_RESOURCE_VIEW_DESC& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("HIP_RESOURCE_VIEW_DESC::reserved").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "reserved=");
            ::rocprofiler::hip::detail::operator<<(out, 0);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_RESOURCE_VIEW_DESC::lastLayer").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "lastLayer=");
            ::rocprofiler::hip::detail::operator<<(out, v.lastLayer);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_RESOURCE_VIEW_DESC::firstLayer").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "firstLayer=");
            ::rocprofiler::hip::detail::operator<<(out, v.firstLayer);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_RESOURCE_VIEW_DESC::lastMipmapLevel").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "lastMipmapLevel=");
            ::rocprofiler::hip::detail::operator<<(out, v.lastMipmapLevel);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_RESOURCE_VIEW_DESC::firstMipmapLevel").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "firstMipmapLevel=");
            ::rocprofiler::hip::detail::operator<<(out, v.firstMipmapLevel);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_RESOURCE_VIEW_DESC::depth").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "depth=");
            ::rocprofiler::hip::detail::operator<<(out, v.depth);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_RESOURCE_VIEW_DESC::height").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "height=");
            ::rocprofiler::hip::detail::operator<<(out, v.height);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_RESOURCE_VIEW_DESC::width").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "width=");
            ::rocprofiler::hip::detail::operator<<(out, v.width);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_RESOURCE_VIEW_DESC::format").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "format=");
            ::rocprofiler::hip::detail::operator<<(out, v.format);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipPitchedPtr& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipPitchedPtr::ysize").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "ysize=");
            ::rocprofiler::hip::detail::operator<<(out, v.ysize);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipPitchedPtr::xsize").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "xsize=");
            ::rocprofiler::hip::detail::operator<<(out, v.xsize);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipPitchedPtr::pitch").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "pitch=");
            ::rocprofiler::hip::detail::operator<<(out, v.pitch);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExtent& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipExtent::depth").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "depth=");
            ::rocprofiler::hip::detail::operator<<(out, v.depth);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExtent::height").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "height=");
            ::rocprofiler::hip::detail::operator<<(out, v.height);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExtent::width").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "width=");
            ::rocprofiler::hip::detail::operator<<(out, v.width);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipPos& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipPos::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipPos::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipPos::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMemcpy3DParms& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipMemcpy3DParms::kind").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "kind=");
            ::rocprofiler::hip::detail::operator<<(out, v.kind);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemcpy3DParms::extent").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "extent=");
            ::rocprofiler::hip::detail::operator<<(out, v.extent);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemcpy3DParms::dstPtr").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstPtr=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstPtr);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemcpy3DParms::dstPos").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstPos=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstPos);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemcpy3DParms::dstArray").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstArray=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstArray);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemcpy3DParms::srcPtr").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcPtr=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcPtr);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemcpy3DParms::srcPos").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcPos=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcPos);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemcpy3DParms::srcArray").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcArray=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcArray);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const HIP_MEMCPY3D& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("HIP_MEMCPY3D::Depth").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "Depth=");
            ::rocprofiler::hip::detail::operator<<(out, v.Depth);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::Height").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "Height=");
            ::rocprofiler::hip::detail::operator<<(out, v.Height);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::WidthInBytes").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "WidthInBytes=");
            ::rocprofiler::hip::detail::operator<<(out, v.WidthInBytes);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::dstHeight").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstHeight=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstHeight);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::dstPitch").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstPitch=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstPitch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::dstArray").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstArray=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstArray);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::dstDevice").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstDevice=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstDevice);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::dstMemoryType").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstMemoryType=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstMemoryType);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::dstLOD").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstLOD=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstLOD);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::dstZ").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "dstZ=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstZ);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::dstY").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "dstY=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstY);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::dstXInBytes").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "dstXInBytes=");
            ::rocprofiler::hip::detail::operator<<(out, v.dstXInBytes);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::srcHeight").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcHeight=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcHeight);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::srcPitch").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcPitch=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcPitch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::srcArray").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcArray=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcArray);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::srcDevice").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcDevice=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcDevice);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::srcMemoryType").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcMemoryType=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcMemoryType);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::srcLOD").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcLOD=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcLOD);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::srcZ").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "srcZ=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcZ);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::srcY").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "srcY=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcY);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("HIP_MEMCPY3D::srcXInBytes").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "srcXInBytes=");
            ::rocprofiler::hip::detail::operator<<(out, v.srcXInBytes);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uchar1& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("uchar1::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uchar2& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("uchar2::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("uchar2::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uchar3& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("uchar3::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("uchar3::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("uchar3::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uchar4& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("uchar4::w").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "w=");
            ::rocprofiler::hip::detail::operator<<(out, v.w);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("uchar4::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("uchar4::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("uchar4::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const char1& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("char1::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const char2& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("char2::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("char2::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const char3& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("char3::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("char3::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("char3::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const char4& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("char4::w").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "w=");
            ::rocprofiler::hip::detail::operator<<(out, v.w);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("char4::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("char4::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("char4::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ushort1& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("ushort1::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ushort2& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("ushort2::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ushort2::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ushort3& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("ushort3::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ushort3::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ushort3::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ushort4& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("ushort4::w").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "w=");
            ::rocprofiler::hip::detail::operator<<(out, v.w);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ushort4::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ushort4::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ushort4::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const short1& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("short1::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const short2& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("short2::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("short2::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const short3& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("short3::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("short3::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("short3::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const short4& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("short4::w").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "w=");
            ::rocprofiler::hip::detail::operator<<(out, v.w);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("short4::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("short4::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("short4::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uint1& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("uint1::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uint2& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("uint2::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("uint2::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uint3& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("uint3::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("uint3::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("uint3::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uint4& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("uint4::w").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "w=");
            ::rocprofiler::hip::detail::operator<<(out, v.w);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("uint4::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("uint4::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("uint4::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const int1& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("int1::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const int2& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("int2::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("int2::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const int3& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("int3::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("int3::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("int3::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const int4& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("int4::w").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "w=");
            ::rocprofiler::hip::detail::operator<<(out, v.w);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("int4::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("int4::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("int4::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulong1& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("ulong1::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulong2& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("ulong2::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ulong2::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulong3& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("ulong3::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ulong3::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ulong3::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulong4& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("ulong4::w").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "w=");
            ::rocprofiler::hip::detail::operator<<(out, v.w);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ulong4::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ulong4::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ulong4::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const long1& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("long1::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const long2& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("long2::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("long2::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const long3& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("long3::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("long3::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("long3::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const long4& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("long4::w").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "w=");
            ::rocprofiler::hip::detail::operator<<(out, v.w);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("long4::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("long4::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("long4::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulonglong1& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("ulonglong1::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulonglong2& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("ulonglong2::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ulonglong2::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulonglong3& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("ulonglong3::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ulonglong3::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ulonglong3::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulonglong4& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("ulonglong4::w").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "w=");
            ::rocprofiler::hip::detail::operator<<(out, v.w);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ulonglong4::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ulonglong4::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("ulonglong4::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const longlong1& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("longlong1::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const longlong2& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("longlong2::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("longlong2::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const longlong3& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("longlong3::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("longlong3::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("longlong3::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const longlong4& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("longlong4::w").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "w=");
            ::rocprofiler::hip::detail::operator<<(out, v.w);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("longlong4::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("longlong4::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("longlong4::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const float1& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("float1::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const float2& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("float2::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("float2::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const float3& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("float3::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("float3::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("float3::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const float4& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("float4::w").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "w=");
            ::rocprofiler::hip::detail::operator<<(out, v.w);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("float4::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("float4::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("float4::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const double1& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("double1::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const double2& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("double2::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("double2::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const double3& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("double3::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("double3::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("double3::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const double4& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("double4::w").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "w=");
            ::rocprofiler::hip::detail::operator<<(out, v.w);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("double4::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("double4::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("double4::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const textureReference& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("textureReference::format").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "format=");
            ::rocprofiler::hip::detail::operator<<(out, v.format);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("textureReference::numChannels").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "numChannels=");
            ::rocprofiler::hip::detail::operator<<(out, v.numChannels);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("textureReference::textureObject").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "textureObject=");
            ::rocprofiler::hip::detail::operator<<(out, v.textureObject);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("textureReference::maxMipmapLevelClamp").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxMipmapLevelClamp=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxMipmapLevelClamp);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("textureReference::minMipmapLevelClamp").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "minMipmapLevelClamp=");
            ::rocprofiler::hip::detail::operator<<(out, v.minMipmapLevelClamp);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("textureReference::mipmapLevelBias").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "mipmapLevelBias=");
            ::rocprofiler::hip::detail::operator<<(out, v.mipmapLevelBias);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("textureReference::mipmapFilterMode").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "mipmapFilterMode=");
            ::rocprofiler::hip::detail::operator<<(out, v.mipmapFilterMode);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("textureReference::maxAnisotropy").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxAnisotropy=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxAnisotropy);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("textureReference::sRGB").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "sRGB=");
            ::rocprofiler::hip::detail::operator<<(out, v.sRGB);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("textureReference::channelDesc").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "channelDesc=");
            ::rocprofiler::hip::detail::operator<<(out, v.channelDesc);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("textureReference::filterMode").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "filterMode=");
            ::rocprofiler::hip::detail::operator<<(out, v.filterMode);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("textureReference::readMode").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "readMode=");
            ::rocprofiler::hip::detail::operator<<(out, v.readMode);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("textureReference::normalized").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "normalized=");
            ::rocprofiler::hip::detail::operator<<(out, v.normalized);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipTextureDesc& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipTextureDesc::maxMipmapLevelClamp").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxMipmapLevelClamp=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxMipmapLevelClamp);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipTextureDesc::minMipmapLevelClamp").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "minMipmapLevelClamp=");
            ::rocprofiler::hip::detail::operator<<(out, v.minMipmapLevelClamp);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipTextureDesc::mipmapLevelBias").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "mipmapLevelBias=");
            ::rocprofiler::hip::detail::operator<<(out, v.mipmapLevelBias);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipTextureDesc::mipmapFilterMode").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "mipmapFilterMode=");
            ::rocprofiler::hip::detail::operator<<(out, v.mipmapFilterMode);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipTextureDesc::maxAnisotropy").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxAnisotropy=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxAnisotropy);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipTextureDesc::normalizedCoords").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "normalizedCoords=");
            ::rocprofiler::hip::detail::operator<<(out, v.normalizedCoords);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipTextureDesc::borderColor").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "borderColor=");
            ::rocprofiler::hip::detail::operator<<(out, v.borderColor);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipTextureDesc::sRGB").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "sRGB=");
            ::rocprofiler::hip::detail::operator<<(out, v.sRGB);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipTextureDesc::readMode").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "readMode=");
            ::rocprofiler::hip::detail::operator<<(out, v.readMode);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipTextureDesc::filterMode").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "filterMode=");
            ::rocprofiler::hip::detail::operator<<(out, v.filterMode);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const surfaceReference& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("surfaceReference::surfaceObject").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "surfaceObject=");
            ::rocprofiler::hip::detail::operator<<(out, v.surfaceObject);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipIpcMemHandle_t&)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipIpcMemHandle_t::reserved").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "reserved=");
            ::rocprofiler::hip::detail::operator<<(out, 0);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipIpcEventHandle_t&)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipIpcEventHandle_t::reserved").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "reserved=");
            ::rocprofiler::hip::detail::operator<<(out, 0);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipFuncAttributes& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipFuncAttributes::sharedSizeBytes").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "sharedSizeBytes=");
            ::rocprofiler::hip::detail::operator<<(out, v.sharedSizeBytes);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFuncAttributes::ptxVersion").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "ptxVersion=");
            ::rocprofiler::hip::detail::operator<<(out, v.ptxVersion);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFuncAttributes::preferredShmemCarveout").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "preferredShmemCarveout=");
            ::rocprofiler::hip::detail::operator<<(out, v.preferredShmemCarveout);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFuncAttributes::numRegs").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "numRegs=");
            ::rocprofiler::hip::detail::operator<<(out, v.numRegs);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFuncAttributes::maxThreadsPerBlock").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "maxThreadsPerBlock=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxThreadsPerBlock);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFuncAttributes::maxDynamicSharedSizeBytes")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "maxDynamicSharedSizeBytes=");
            ::rocprofiler::hip::detail::operator<<(out, v.maxDynamicSharedSizeBytes);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFuncAttributes::localSizeBytes").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "localSizeBytes=");
            ::rocprofiler::hip::detail::operator<<(out, v.localSizeBytes);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFuncAttributes::constSizeBytes").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "constSizeBytes=");
            ::rocprofiler::hip::detail::operator<<(out, v.constSizeBytes);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFuncAttributes::cacheModeCA").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "cacheModeCA=");
            ::rocprofiler::hip::detail::operator<<(out, v.cacheModeCA);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFuncAttributes::binaryVersion").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "binaryVersion=");
            ::rocprofiler::hip::detail::operator<<(out, v.binaryVersion);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMemLocation& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipMemLocation::id").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "id=");
            ::rocprofiler::hip::detail::operator<<(out, v.id);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemLocation::type").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "type=");
            ::rocprofiler::hip::detail::operator<<(out, v.type);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMemAccessDesc& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipMemAccessDesc::flags").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "flags=");
            ::rocprofiler::hip::detail::operator<<(out, v.flags);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemAccessDesc::location").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "location=");
            ::rocprofiler::hip::detail::operator<<(out, v.location);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMemPoolProps& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipMemPoolProps::reserved").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "reserved=");
            ::rocprofiler::hip::detail::operator<<(out, 0);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemPoolProps::location").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "location=");
            ::rocprofiler::hip::detail::operator<<(out, v.location);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemPoolProps::handleTypes").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "handleTypes=");
            ::rocprofiler::hip::detail::operator<<(out, v.handleTypes);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemPoolProps::allocType").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "allocType=");
            ::rocprofiler::hip::detail::operator<<(out, v.allocType);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMemPoolPtrExportData&)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipMemPoolPtrExportData::reserved").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "reserved=");
            ::rocprofiler::hip::detail::operator<<(out, 0);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const dim3& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("dim3::z").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "z=");
            ::rocprofiler::hip::detail::operator<<(out, v.z);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("dim3::y").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "y=");
            ::rocprofiler::hip::detail::operator<<(out, v.y);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("dim3::x").find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "x=");
            ::rocprofiler::hip::detail::operator<<(out, v.x);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipLaunchParams& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipLaunchParams::stream").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "stream=");
            ::rocprofiler::hip::detail::operator<<(out, v.stream);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipLaunchParams::sharedMem").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "sharedMem=");
            ::rocprofiler::hip::detail::operator<<(out, v.sharedMem);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipLaunchParams::blockDim").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "blockDim=");
            ::rocprofiler::hip::detail::operator<<(out, v.blockDim);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipLaunchParams::gridDim").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "gridDim=");
            ::rocprofiler::hip::detail::operator<<(out, v.gridDim);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipFunctionLaunchParams& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipFunctionLaunchParams::hStream").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hStream=");
            ::rocprofiler::hip::detail::operator<<(out, v.hStream);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFunctionLaunchParams::sharedMemBytes").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "sharedMemBytes=");
            ::rocprofiler::hip::detail::operator<<(out, v.sharedMemBytes);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFunctionLaunchParams::blockDimZ").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "blockDimZ=");
            ::rocprofiler::hip::detail::operator<<(out, v.blockDimZ);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFunctionLaunchParams::blockDimY").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "blockDimY=");
            ::rocprofiler::hip::detail::operator<<(out, v.blockDimY);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFunctionLaunchParams::blockDimX").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "blockDimX=");
            ::rocprofiler::hip::detail::operator<<(out, v.blockDimX);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFunctionLaunchParams::gridDimZ").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "gridDimZ=");
            ::rocprofiler::hip::detail::operator<<(out, v.gridDimZ);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFunctionLaunchParams::gridDimY").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "gridDimY=");
            ::rocprofiler::hip::detail::operator<<(out, v.gridDimY);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFunctionLaunchParams::gridDimX").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "gridDimX=");
            ::rocprofiler::hip::detail::operator<<(out, v.gridDimX);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipFunctionLaunchParams::function").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "function=");
            ::rocprofiler::hip::detail::operator<<(out, v.function);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalMemoryHandleDesc& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipExternalMemoryHandleDesc::reserved").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "reserved=");
            ::rocprofiler::hip::detail::operator<<(out, 0);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalMemoryHandleDesc::flags").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "flags=");
            ::rocprofiler::hip::detail::operator<<(out, v.flags);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalMemoryHandleDesc::size").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "size=");
            ::rocprofiler::hip::detail::operator<<(out, v.size);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalMemoryHandleDesc_st::union ::handle.fd")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "handle.fd=");
            ::rocprofiler::hip::detail::operator<<(out, v.handle.fd);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalMemoryHandleDesc::type").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "type=");
            ::rocprofiler::hip::detail::operator<<(out, v.type);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalMemoryBufferDesc& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipExternalMemoryBufferDesc::reserved").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "reserved=");
            ::rocprofiler::hip::detail::operator<<(out, 0);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalMemoryBufferDesc::flags").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "flags=");
            ::rocprofiler::hip::detail::operator<<(out, v.flags);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalMemoryBufferDesc::size").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "size=");
            ::rocprofiler::hip::detail::operator<<(out, v.size);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalMemoryBufferDesc::offset").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "offset=");
            ::rocprofiler::hip::detail::operator<<(out, v.offset);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

#if HIP_VERSION_MAJOR >= 6
inline static std::ostream&
operator<<(std::ostream& out, const hipExternalMemoryMipmappedArrayDesc& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipExternalMemoryMipmappedArrayDesc::numLevels")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "numLevels=");
            ::rocprofiler::hip::detail::operator<<(out, v.numLevels);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalMemoryMipmappedArrayDesc::flags").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "flags=");
            ::rocprofiler::hip::detail::operator<<(out, v.flags);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalMemoryMipmappedArrayDesc::extent")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "extent=");
            ::rocprofiler::hip::detail::operator<<(out, v.extent);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalMemoryMipmappedArrayDesc::formatDesc")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "formatDesc=");
            ::rocprofiler::hip::detail::operator<<(out, v.formatDesc);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalMemoryMipmappedArrayDesc::offset")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "offset=");
            ::rocprofiler::hip::detail::operator<<(out, v.offset);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}
#endif

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalSemaphoreHandleDesc& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipExternalSemaphoreHandleDesc::reserved").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "reserved=");
            ::rocprofiler::hip::detail::operator<<(out, 0);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalSemaphoreHandleDesc::flags").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "flags=");
            ::rocprofiler::hip::detail::operator<<(out, v.flags);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalSemaphoreHandleDesc_st::union ::handle.fd")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "handle.fd=");
            ::rocprofiler::hip::detail::operator<<(out, v.handle.fd);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalSemaphoreHandleDesc::type").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "type=");
            ::rocprofiler::hip::detail::operator<<(out, v.type);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalSemaphoreSignalParams& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipExternalSemaphoreSignalParams::reserved").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "reserved=");
            ::rocprofiler::hip::detail::operator<<(out, 0);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalSemaphoreSignalParams::flags").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "flags=");
            ::rocprofiler::hip::detail::operator<<(out, v.flags);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalSemaphoreWaitParams& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipExternalSemaphoreWaitParams::reserved").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "reserved=");
            ::rocprofiler::hip::detail::operator<<(out, 0);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalSemaphoreWaitParams::flags").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "flags=");
            ::rocprofiler::hip::detail::operator<<(out, v.flags);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipHostNodeParams& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipHostNodeParams::fn").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "fn=");
            ::rocprofiler::hip::detail::operator<<(out, v.fn);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipKernelNodeParams& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipKernelNodeParams::sharedMemBytes").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "sharedMemBytes=");
            ::rocprofiler::hip::detail::operator<<(out, v.sharedMemBytes);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipKernelNodeParams::gridDim").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "gridDim=");
            ::rocprofiler::hip::detail::operator<<(out, v.gridDim);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipKernelNodeParams::blockDim").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "blockDim=");
            ::rocprofiler::hip::detail::operator<<(out, v.blockDim);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMemsetParams& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipMemsetParams::width").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "width=");
            ::rocprofiler::hip::detail::operator<<(out, v.width);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemsetParams::value").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "value=");
            ::rocprofiler::hip::detail::operator<<(out, v.value);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemsetParams::pitch").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "pitch=");
            ::rocprofiler::hip::detail::operator<<(out, v.pitch);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemsetParams::height").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "height=");
            ::rocprofiler::hip::detail::operator<<(out, v.height);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemsetParams::elementSize").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "elementSize=");
            ::rocprofiler::hip::detail::operator<<(out, v.elementSize);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMemAllocNodeParams& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipMemAllocNodeParams::bytesize").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "bytesize=");
            ::rocprofiler::hip::detail::operator<<(out, v.bytesize);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemAllocNodeParams::accessDescCount").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "accessDescCount=");
            ::rocprofiler::hip::detail::operator<<(out, v.accessDescCount);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemAllocNodeParams::accessDescs").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "accessDescs=");
            ::rocprofiler::hip::detail::operator<<(out, v.accessDescs);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemAllocNodeParams::poolProps").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "poolProps=");
            ::rocprofiler::hip::detail::operator<<(out, v.poolProps);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipAccessPolicyWindow& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipAccessPolicyWindow::num_bytes").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "num_bytes=");
            ::rocprofiler::hip::detail::operator<<(out, v.num_bytes);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipAccessPolicyWindow::missProp").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "missProp=");
            ::rocprofiler::hip::detail::operator<<(out, v.missProp);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipAccessPolicyWindow::hitRatio").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hitRatio=");
            ::rocprofiler::hip::detail::operator<<(out, v.hitRatio);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipAccessPolicyWindow::hitProp").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "hitProp=");
            ::rocprofiler::hip::detail::operator<<(out, v.hitProp);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipKernelNodeAttrValue& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipKernelNodeAttrValue::cooperative").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "cooperative=");
            ::rocprofiler::hip::detail::operator<<(out, v.cooperative);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipKernelNodeAttrValue::accessPolicyWindow").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "accessPolicyWindow=");
            ::rocprofiler::hip::detail::operator<<(out, v.accessPolicyWindow);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

// inline static std::ostream&
// operator<<(std::ostream& out, const HIP_MEMSET_NODE_PARAMS& v)
// {
//     std::operator<<(out, '{');
//     ++HIP_depth_max_cnt;
//     if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
//     {
//         if(std::string_view("HIP_MEMSET_NODE_PARAMS::height").find(HIP_structs_regex) !=
//            std::string_view::npos)
//         {
//             std::                       operator<<(out, "height=");
//             ::rocprofiler::hip::detail::operator<<(out, v.height);
//             std::                       operator<<(out, ", ");
//         }
//         if(std::string_view("HIP_MEMSET_NODE_PARAMS::width").find(HIP_structs_regex) !=
//            std::string_view::npos)
//         {
//             std::                       operator<<(out, "width=");
//             ::rocprofiler::hip::detail::operator<<(out, v.width);
//             std::                       operator<<(out, ", ");
//         }
//         if(std::string_view("HIP_MEMSET_NODE_PARAMS::elementSize").find(HIP_structs_regex) !=
//            std::string_view::npos)
//         {
//             std::                       operator<<(out, "elementSize=");
//             ::rocprofiler::hip::detail::operator<<(out, v.elementSize);
//             std::                       operator<<(out, ", ");
//         }
//         if(std::string_view("HIP_MEMSET_NODE_PARAMS::value").find(HIP_structs_regex) !=
//            std::string_view::npos)
//         {
//             std::                       operator<<(out, "value=");
//             ::rocprofiler::hip::detail::operator<<(out, v.value);
//             std::                       operator<<(out, ", ");
//         }
//         if(std::string_view("HIP_MEMSET_NODE_PARAMS::pitch").find(HIP_structs_regex) !=
//            std::string_view::npos)
//         {
//             std::                       operator<<(out, "pitch=");
//             ::rocprofiler::hip::detail::operator<<(out, v.pitch);
//             std::                       operator<<(out, ", ");
//         }
//         if(std::string_view("HIP_MEMSET_NODE_PARAMS::dst").find(HIP_structs_regex) !=
//            std::string_view::npos)
//         {
//             std::                       operator<<(out, "dst=");
//             ::rocprofiler::hip::detail::operator<<(out, v.dst);
//         }
//     };
//     HIP_depth_max_cnt--;
//     std::operator<<(out, '}');
//     return out;
// }

inline static std::ostream&
operator<<(std::ostream& out, const hipMemAllocationProp& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipMemAllocationProp::location").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "location=");
            ::rocprofiler::hip::detail::operator<<(out, v.location);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemAllocationProp::requestedHandleType").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "requestedHandleType=");
            ::rocprofiler::hip::detail::operator<<(out, v.requestedHandleType);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipMemAllocationProp::type").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "type=");
            ::rocprofiler::hip::detail::operator<<(out, v.type);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalSemaphoreSignalNodeParams& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipExternalSemaphoreSignalNodeParams::numExtSems")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "numExtSems=");
            ::rocprofiler::hip::detail::operator<<(out, v.numExtSems);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalSemaphoreSignalNodeParams::paramsArray")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "paramsArray=");
            ::rocprofiler::hip::detail::operator<<(out, v.paramsArray);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalSemaphoreSignalNodeParams::extSemArray")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "extSemArray=");
            ::rocprofiler::hip::detail::operator<<(out, v.extSemArray);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalSemaphoreWaitNodeParams& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipExternalSemaphoreWaitNodeParams::numExtSems")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "numExtSems=");
            ::rocprofiler::hip::detail::operator<<(out, v.numExtSems);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalSemaphoreWaitNodeParams::paramsArray")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "paramsArray=");
            ::rocprofiler::hip::detail::operator<<(out, v.paramsArray);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipExternalSemaphoreWaitNodeParams::extSemArray")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "extSemArray=");
            ::rocprofiler::hip::detail::operator<<(out, v.extSemArray);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipArrayMapInfo& v)
{
    std::operator<<(out, '{');
    ++HIP_depth_max_cnt;
    if(HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max)
    {
        if(std::string_view("hipArrayMapInfo::reserved").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "reserved=");
            ::rocprofiler::hip::detail::operator<<(out, 0);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipArrayMapInfo::flags").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "flags=");
            ::rocprofiler::hip::detail::operator<<(out, v.flags);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipArrayMapInfo::deviceBitMask").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "deviceBitMask=");
            ::rocprofiler::hip::detail::operator<<(out, v.deviceBitMask);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipArrayMapInfo::offset").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "offset=");
            ::rocprofiler::hip::detail::operator<<(out, v.offset);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipArrayMapInfo::union ::memHandle.memHandle")
               .find(HIP_structs_regex) != std::string_view::npos)
        {
            std::                       operator<<(out, "memHandle.memHandle=");
            ::rocprofiler::hip::detail::operator<<(out, v.memHandle.memHandle);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipArrayMapInfo::memHandleType").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "memHandleType=");
            ::rocprofiler::hip::detail::operator<<(out, v.memHandleType);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipArrayMapInfo::memOperationType").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "memOperationType=");
            ::rocprofiler::hip::detail::operator<<(out, v.memOperationType);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipArrayMapInfo::subresourceType").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "subresourceType=");
            ::rocprofiler::hip::detail::operator<<(out, v.subresourceType);
            std::                       operator<<(out, ", ");
        }
        if(std::string_view("hipArrayMapInfo::resourceType").find(HIP_structs_regex) !=
           std::string_view::npos)
        {
            std::                       operator<<(out, "resourceType=");
            ::rocprofiler::hip::detail::operator<<(out, v.resourceType);
        }
    };
    HIP_depth_max_cnt--;
    std::operator<<(out, '}');
    return out;
}
// end ostream ops for HIP
}  // namespace detail
}  // namespace hip
}  // namespace rocprofiler

inline static std::ostream&
operator<<(std::ostream& out, const __locale_struct& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipDeviceArch_t& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipUUID& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipDeviceProp_tR0000& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipDeviceProp_tR0600& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipPointerAttribute_t& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipChannelFormatDesc& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const HIP_ARRAY_DESCRIPTOR& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const HIP_ARRAY3D_DESCRIPTOR& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hip_Memcpy2D& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMipmappedArray& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const HIP_TEXTURE_DESC& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipResourceDesc& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const HIP_RESOURCE_DESC& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipResourceViewDesc& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const HIP_RESOURCE_VIEW_DESC& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipPitchedPtr& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExtent& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipPos& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMemcpy3DParms& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const HIP_MEMCPY3D& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uchar1& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uchar2& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uchar3& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uchar4& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const char1& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const char2& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const char3& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const char4& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ushort1& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ushort2& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ushort3& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ushort4& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const short1& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const short2& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const short3& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const short4& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uint1& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uint2& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uint3& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const uint4& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const int1& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const int2& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const int3& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const int4& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulong1& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulong2& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulong3& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulong4& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const long1& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const long2& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const long3& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const long4& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulonglong1& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulonglong2& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulonglong3& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const ulonglong4& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const longlong1& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const longlong2& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const longlong3& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const longlong4& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const float1& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const float2& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const float3& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const float4& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const double1& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const double2& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const double3& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const double4& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const textureReference& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipTextureDesc& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const surfaceReference& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipIpcMemHandle_t& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipIpcEventHandle_t& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipFuncAttributes& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMemLocation& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMemAccessDesc& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMemPoolProps& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMemPoolPtrExportData& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const dim3& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipLaunchParams& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipFunctionLaunchParams& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalMemoryHandleDesc& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalMemoryBufferDesc& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalMemoryMipmappedArrayDesc& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalSemaphoreHandleDesc& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalSemaphoreSignalParams& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalSemaphoreWaitParams& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipHostNodeParams& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipKernelNodeParams& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMemsetParams& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipMemAllocNodeParams& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipAccessPolicyWindow& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipKernelNodeAttrValue& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

// inline static std::ostream&
// operator<<(std::ostream& out, const HIP_MEMSET_NODE_PARAMS& v)
// {
//     ::rocprofiler::hip::detail::operator<<(out, v);
//     return out;
// }

inline static std::ostream&
operator<<(std::ostream& out, const hipMemAllocationProp& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalSemaphoreSignalNodeParams& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipExternalSemaphoreWaitNodeParams& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}

inline static std::ostream&
operator<<(std::ostream& out, const hipArrayMapInfo& v)
{
    ::rocprofiler::hip::detail::operator<<(out, v);
    return out;
}
