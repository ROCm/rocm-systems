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

#include "lib/common/defines.hpp"
#include "lib/common/filesystem.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <amd_comgr/amd_comgr.h>
#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <cxxabi.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <map>
#include <ostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

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

constexpr size_t BUFFER_SIZE_BYTES = 4096;
constexpr size_t WATERMARK         = (BUFFER_SIZE_BYTES / 2);

using rocprofiler_tool_buffer_kind_names_t =
    std::unordered_map<rocprofiler_buffer_tracing_kind_t, const char*>;
using rocprofiler_tool_buffer_kind_operation_names_t =
    std::unordered_map<rocprofiler_buffer_tracing_kind_t,
                       std::unordered_map<uint32_t, const char*>>;

struct rocprofiler_tool_buffer_name_info_t
{
    rocprofiler_tool_buffer_kind_names_t           kind_names      = {};
    rocprofiler_tool_buffer_kind_operation_names_t operation_names = {};
};

using rocprofiler_tool_callback_kind_names_t =
    std::unordered_map<rocprofiler_callback_tracing_kind_t, const char*>;
using rocprofiler_tool_callback_kind_operation_names_t =
    std::unordered_map<rocprofiler_callback_tracing_kind_t,
                       std::unordered_map<uint32_t, const char*>>;

struct rocprofiler_tool_callback_name_info_t
{
    rocprofiler_tool_callback_kind_names_t           kind_names      = {};
    rocprofiler_tool_callback_kind_operation_names_t operation_names = {};
};

rocprofiler_tool_buffer_name_info_t
get_buffer_id_names();

rocprofiler_tool_callback_name_info_t
get_callback_id_names();
