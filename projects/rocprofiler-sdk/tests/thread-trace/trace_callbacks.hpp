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

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define ROCPROFILER_VAR_NAME_COMBINE(X, Y) X##Y
#define ROCPROFILER_VARIABLE(X, Y)         ROCPROFILER_VAR_NAME_COMBINE(X, Y)

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

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

namespace Callbacks
{
struct CodeobjInfo
{
    int64_t     addr = 0;
    size_t      size = 0;
    size_t      id   = 0;
    std::string filename{};
    std::string uri{};
};

struct ToolData
{
    ToolData(const char* out)
    : out_dir(out){};

    std::string              out_dir{};
    std::mutex               mut{};
    std::vector<CodeobjInfo> codeobjs{};
    std::vector<std::string> att_files{};

    std::unordered_map<uint64_t, std::string> kernel_id_to_kernel_name = {};
};

void
tool_codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t*,
                              void* callback_data);

void
shader_data_callback(rocprofiler_agent_id_t  agent,
                     int64_t                 se_id,
                     void*                   se_data,
                     size_t                  data_size,
                     rocprofiler_user_data_t userdata);

void
finalize_json(void* userdata);

}  // namespace Callbacks
