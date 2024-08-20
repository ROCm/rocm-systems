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

#pragma once

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>

#define ROCPROFILER_VAR_NAME_COMBINE(X, Y) X##Y
#define ROCPROFILER_VARIABLE(X, Y)         ROCPROFILER_VAR_NAME_COMBINE(X, Y)

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = result;                 \
        if(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != ROCPROFILER_STATUS_SUCCESS)              \
        {                                                                                          \
            std::string status_msg =                                                               \
                rocprofiler_get_status_string(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));        \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) \
                      << ": " << status_msg << std::endl;                                          \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

#define C_API_BEGIN                                                                                \
    try                                                                                            \
    {
#define C_API_END                                                                                  \
    }                                                                                              \
    catch(std::exception & e)                                                                      \
    {                                                                                              \
        std::cerr << "Error in " << __FILE__ << ':' << __LINE__ << ' ' << e.what() << std::endl;   \
    }                                                                                              \
    catch(...) { std::cerr << "Error in " << __FILE__ << ':' << __LINE__ << std::endl; }

namespace ATTTest
{
struct TrackedIsa
{
    std::atomic<size_t> hitcount{0};
    std::atomic<size_t> latency{0};
    std::string         inst{};
};

struct pcInfo
{
    size_t addr;
    size_t marker_id;

    bool operator==(const pcInfo& other) const
    {
        return addr == other.addr && marker_id == other.marker_id;
    }
    bool operator<(const pcInfo& other) const
    {
        if(marker_id == other.marker_id) return addr < other.addr;
        return marker_id < other.marker_id;
    }
};

struct ToolData
{
    std::unordered_map<uint64_t, std::string>     kernel_id_to_kernel_name = {};
    std::map<pcInfo, std::unique_ptr<TrackedIsa>> isa_map;

    std::atomic<int> waves_started = 0;
    std::atomic<int> waves_ended   = 0;
    std::mutex       isa_map_mut;
    std::set<pcInfo> wave_start_locations{};
};

namespace Callbacks
{
void
tool_codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t*,
                              void* callback_data);

void
shader_data_callback(int64_t                 se_id,
                     void*                   se_data,
                     size_t                  data_size,
                     rocprofiler_user_data_t userdata);

void
callbacks_init();

void
callbacks_fini();

}  // namespace Callbacks
}  // namespace ATTTest
