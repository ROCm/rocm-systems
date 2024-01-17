// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler-sdk-roctx/api_trace.h>
#include <rocprofiler-sdk-roctx/defines.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <rocprofiler-sdk-roctx/types.h>

#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"

#include <glog/logging.h>
#include <rocprofiler-register/rocprofiler-register.h>

#include <array>
#include <atomic>
#include <cassert>

#define ROCP_REG_VERSION                                                                           \
    ROCPROFILER_REGISTER_COMPUTE_VERSION_3(                                                        \
        ROCTX_VERSION_MAJOR, ROCTX_VERSION_MINOR, ROCTX_VERSION_PATCH)

ROCPROFILER_REGISTER_DEFINE_IMPORT(roctx, ROCP_REG_VERSION)

namespace roctx
{
namespace
{
constexpr size_t
compute_table_offset(size_t n)
{
    return (sizeof(uint64_t) + (n * sizeof(void*)));
}

constexpr size_t
compute_table_size(size_t nmembers)
{
    return (sizeof(uint64_t) + (nmembers * sizeof(void*)));
}

#define ROCTX_ASSERT_OFFSET(MEMBER, IDX)                                                           \
    static_assert(offsetof(roctxApiTable_t, MEMBER) == compute_table_offset(IDX),                  \
                  "Do not re-arrange the table members")

ROCTX_ASSERT_OFFSET(roctxMarkA_fn, 0);
ROCTX_ASSERT_OFFSET(roctxRangePushA_fn, 1);
ROCTX_ASSERT_OFFSET(roctxRangePop_fn, 2);
ROCTX_ASSERT_OFFSET(roctxRangeStartA_fn, 3);
ROCTX_ASSERT_OFFSET(roctxRangeStop_fn, 4);
ROCTX_ASSERT_OFFSET(roctxProfilerPause_fn, 5);
ROCTX_ASSERT_OFFSET(roctxProfilerResume_fn, 6);
ROCTX_ASSERT_OFFSET(roctxNameOsThread_fn, 7);
ROCTX_ASSERT_OFFSET(roctxNameHsaAgent_fn, 8);
ROCTX_ASSERT_OFFSET(roctxNameHipDevice_fn, 9);
ROCTX_ASSERT_OFFSET(roctxNameHipStream_fn, 10);
ROCTX_ASSERT_OFFSET(roctxGetThreadId_fn, 11);

#undef ROCTX_ASSERT_OFFSET

static_assert(
    sizeof(roctxApiTable_t) == compute_table_size(12),
    "Update table major/step version and add a new offset assertion if this fails to compile");

auto&
get_nested_range_level()
{
    static thread_local int value = 0;
    return value;
}

auto&
get_start_stop_range_id()
{
    static auto value = std::atomic<roctx_range_id_t>{};
    return value;
}

int
roctxGetThreadId(roctx_thread_id_t* tid)
{
    *tid = rocprofiler::common::get_tid();
    return 0;
}

void
roctxMarkA(const char*)
{}

int
roctxRangePushA(const char*)
{
    return get_nested_range_level()++;
}

int
roctxRangePop()
{
    if(get_nested_range_level() == 0) return -1;
    return --get_nested_range_level();
}

roctx_range_id_t
roctxRangeStartA(const char*)
{
    auto range_id = ++get_start_stop_range_id();
    return range_id;
}

void roctxRangeStop(roctx_range_id_t) {}

int roctxProfilerPause(roctx_thread_id_t /*tid*/) { return 0; }

int roctxProfilerResume(roctx_thread_id_t /*tid*/) { return 0; }

int
roctxNameOsThread(const char*)
{
    return 0;
}

int
roctxNameHsaAgent(const char*, const struct hsa_agent_s*)
{
    return 0;
}

int
roctxNameHipDevice(const char*, int)
{
    return 0;
}

int
roctxNameHipStream(const char*, const struct ihipStream_t*)
{
    return 0;
}

auto&
get_table_impl()
{
    rocprofiler::common::init_logging("ROCTX_LOG_LEVEL");

    static auto val = roctxApiTable_t{sizeof(roctxApiTable_t),
                                      &::roctx::roctxMarkA,
                                      &::roctx::roctxRangePushA,
                                      &::roctx::roctxRangePop,
                                      &::roctx::roctxRangeStartA,
                                      &::roctx::roctxRangeStop,
                                      &::roctx::roctxProfilerPause,
                                      &::roctx::roctxProfilerResume,
                                      &::roctx::roctxNameOsThread,
                                      &::roctx::roctxNameHsaAgent,
                                      &::roctx::roctxNameHipDevice,
                                      &::roctx::roctxNameHipStream,
                                      &::roctx::roctxGetThreadId};

    auto table_array = std::array<void*, 1>{&val};
    auto lib_id      = rocprofiler_register_library_indentifier_t{};
    auto rocp_reg_status =
        rocprofiler_register_library_api_table("roctx",
                                               &ROCPROFILER_REGISTER_IMPORT_FUNC(roctx),
                                               ROCP_REG_VERSION,
                                               table_array.data(),
                                               table_array.size(),
                                               &lib_id);

    LOG(INFO) << "[rocprofiler-sdk-roctx][" << getpid() << "] rocprofiler-register returned code "
              << rocp_reg_status << ": " << rocprofiler_register_error_string(rocp_reg_status);

    LOG_IF(WARNING, rocp_reg_status != ROCP_REG_SUCCESS && rocp_reg_status != ROCP_REG_NO_TOOLS)
        << "[rocprofiler-sdk-roctx][" << getpid()
        << "] rocprofiler-register failed with error code " << rocp_reg_status << ": "
        << rocprofiler_register_error_string(rocp_reg_status);

    return val;
}

const auto*
get_table()
{
    static const auto* tbl = &get_table_impl();
    return tbl;
}
}  // namespace
}  // namespace roctx

ROCTX_EXTERN_C_INIT

void
roctxMarkA(const char* message)
{
    ::roctx::get_table()->roctxMarkA_fn(message);
}

int
roctxRangePushA(const char* message)
{
    return ::roctx::get_table()->roctxRangePushA_fn(message);
}

int
roctxRangePop()
{
    return ::roctx::get_table()->roctxRangePop_fn();
}

roctx_range_id_t
roctxRangeStartA(const char* message)
{
    return ::roctx::get_table()->roctxRangeStartA_fn(message);
}

void
roctxRangeStop(roctx_range_id_t id)
{
    return ::roctx::get_table()->roctxRangeStop_fn(id);
}

int
roctxProfilerPause(roctx_thread_id_t tid)
{
    return ::roctx::get_table()->roctxProfilerPause_fn(tid);
}

int
roctxProfilerResume(roctx_thread_id_t tid)
{
    return ::roctx::get_table()->roctxProfilerResume_fn(tid);
}

int
roctxNameOsThread(const char* name)
{
    return ::roctx::get_table()->roctxNameOsThread_fn(name);
}

int
roctxNameHsaAgent(const char* name, const struct hsa_agent_s* agent)
{
    return ::roctx::get_table()->roctxNameHsaAgent_fn(name, agent);
}

int
roctxNameHipDevice(const char* name, int device_id)
{
    return ::roctx::get_table()->roctxNameHipDevice_fn(name, device_id);
}

int
roctxNameHipStream(const char* name, const struct ihipStream_t* stream)
{
    return ::roctx::get_table()->roctxNameHipStream_fn(name, stream);
}

int
roctxGetThreadId(roctx_thread_id_t* tid)
{
    return ::roctx::get_table()->roctxGetThreadId_fn(tid);
}

ROCTX_EXTERN_C_FINI
