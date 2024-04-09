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

#include "lib/common/container/ring_buffer.hpp"
#include "lib/common/container/small_vector.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/filesystem.hpp"
#include "output_file.hpp"

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
                      << ": " << status_msg << "\n"                                                \
                      << std::flush;                                                               \
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

namespace common = ::rocprofiler::common;
namespace tool   = ::rocprofiler::tool;

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

struct rocprofiler_tool_marker_record_t
{
    rocprofiler_callback_tracing_kind_t kind;
    uint32_t                            op    = 0;
    uint32_t                            phase = 0;
    uint64_t                            pid   = 0;
    uint64_t                            tid   = 0;
    uint64_t                            cid   = 0;

    rocprofiler_timestamp_t start_timestamp;
    rocprofiler_timestamp_t end_timestamp;
};

struct rocprofiler_tool_counter_collection_record_t
{
    rocprofiler_profile_counting_dispatch_data_t                       dispatch_data;
    common::container::small_vector<rocprofiler_record_counter_t, 128> profiler_record;
    uint64_t                                                           pid                  = 0;
    uint64_t                                                           id                   = 0;
    uint64_t                                                           thread_id            = 0;
    uint64_t                                                           dispatch_index       = 0;
    uint64_t                                                           private_segment_size = 0;
    uint64_t                                                           arch_vgpr_count      = 0;
    uint64_t                                                           sgpr_count           = 0;
    uint64_t                                                           lds_block_size_v     = 0;
};

struct timestamps_t
{
    rocprofiler_timestamp_t app_start_time;
    rocprofiler_timestamp_t app_end_time;
};

using hip_ring_buffer_t =
    rocprofiler::common::container::ring_buffer<rocprofiler_buffer_tracing_hip_api_record_t>;
using hsa_ring_buffer_t =
    rocprofiler::common::container::ring_buffer<rocprofiler_buffer_tracing_hsa_api_record_t>;
using kernel_dispatch_ring_buffer_t = rocprofiler::common::container::ring_buffer<
    rocprofiler_buffer_tracing_kernel_dispatch_record_t>;
using memory_copy_ring_buffer_t =
    rocprofiler::common::container::ring_buffer<rocprofiler_buffer_tracing_memory_copy_record_t>;
using counter_collection_buffer_t =
    rocprofiler::common::container::ring_buffer<rocprofiler_record_counter_t>;
using marker_api_ring_buffer_t =
    rocprofiler::common::container::ring_buffer<rocprofiler_tool_marker_record_t>;
using counter_collection_ring_buffer_t =
    rocprofiler::common::container::ring_buffer<rocprofiler_tool_counter_collection_record_t>;
using scratch_memory_ring_buffer_t =
    rocprofiler::common::container::ring_buffer<rocprofiler_buffer_tracing_scratch_memory_record_t>;

using tool_get_agent_node_id_fn_t      = uint64_t (*)(rocprofiler_agent_id_t);
using tool_get_app_timestamps_fn_t     = timestamps_t* (*) ();
using tool_get_kernel_name_fn_t        = std::string_view (*)(uint64_t);
using tool_get_domain_name_fn_t        = std::string_view (*)(rocprofiler_buffer_tracing_kind_t);
using tool_get_operation_name_fn_t     = std::string_view (*)(rocprofiler_buffer_tracing_kind_t,
                                                          rocprofiler_tracing_operation_t);
using tool_get_callback_kind_name_fn_t = std::string_view (*)(rocprofiler_callback_tracing_kind_t);
using tool_get_callback_op_name_fn_t   = std::string_view (*)(rocprofiler_callback_tracing_kind_t,
                                                            uint32_t);
using tool_get_roctx_msg_fn_t          = std::string_view (*)(uint64_t);
using tool_get_counter_info_name_fn_t  = std::string (*)(uint64_t);
using tool_get_output_file_ref_fn_t    = rocprofiler::tool::output_file& (*) ();
using tool_get_output_file_ptr_fn_t    = rocprofiler::tool::output_file*& (*) ();

enum class buffer_type_t
{
    ROCPROFILER_TOOL_BUFFER_HSA = 0,
    ROCPROFILER_TOOL_BUFFER_HIP,
    ROCPROFILER_TOOL_BUFFER_MEMORY_COPY,
    ROCPROFILER_TOOL_BUFFER_COUNTER_COLLECTION,
    ROCPROFILER_TOOL_BUFFER_KERNEL_DISPATCH,
    ROCPROFILER_TOOL_BUFFER_MARKER_API,
    ROCPROFILER_TOOL_BUFFER_SCRATCH_MEMORY,
};

struct tool_table
{
    // node id
    tool_get_agent_node_id_fn_t tool_get_agent_node_id_fn = nullptr;
    // timestamps
    tool_get_app_timestamps_fn_t tool_get_app_timestamps_fn = nullptr;
    // names and messages
    tool_get_kernel_name_fn_t        tool_get_kernel_name_fn       = nullptr;
    tool_get_domain_name_fn_t        tool_get_domain_name_fn       = nullptr;
    tool_get_operation_name_fn_t     tool_get_operation_name_fn    = nullptr;
    tool_get_counter_info_name_fn_t  tool_get_counter_info_name_fn = nullptr;
    tool_get_callback_kind_name_fn_t tool_get_callback_kind_fn     = nullptr;
    tool_get_callback_op_name_fn_t   tool_get_callback_op_name_fn  = nullptr;
    tool_get_roctx_msg_fn_t          tool_get_roctx_msg_fn         = nullptr;
    // trace files
    tool_get_output_file_ref_fn_t tool_get_agent_info_file_fn         = nullptr;
    tool_get_output_file_ref_fn_t tool_get_kernel_trace_file_fn       = nullptr;
    tool_get_output_file_ref_fn_t tool_get_hsa_api_trace_file_fn      = nullptr;
    tool_get_output_file_ref_fn_t tool_get_hip_api_trace_file_fn      = nullptr;
    tool_get_output_file_ref_fn_t tool_get_marker_api_trace_file_fn   = nullptr;
    tool_get_output_file_ref_fn_t tool_get_counter_collection_file_fn = nullptr;
    tool_get_output_file_ref_fn_t tool_get_memory_copy_trace_file_fn  = nullptr;
    tool_get_output_file_ptr_fn_t tool_get_scratch_memory_file_fn     = nullptr;
    // stats files
    tool_get_output_file_ref_fn_t tool_get_kernel_stats_file_fn         = nullptr;
    tool_get_output_file_ref_fn_t tool_get_hip_stats_file_fn            = nullptr;
    tool_get_output_file_ref_fn_t tool_get_hsa_stats_file_fn            = nullptr;
    tool_get_output_file_ref_fn_t tool_get_memory_copy_stats_file_fn    = nullptr;
    tool_get_output_file_ptr_fn_t tool_get_scratch_memory_stats_file_fn = nullptr;
};
