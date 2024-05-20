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

#include "config.hpp"
#include "csv.hpp"
#include "generateCSV.hpp"
#include "helper.hpp"
#include "output_file.hpp"
#include "tmp_file.hpp"

#include "lib/common/demangle.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/core.h>
#include <unistd.h>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(CODECOV) && CODECOV > 0
extern "C" {
extern void
__gcov_dump(void);
}
#endif

namespace common = ::rocprofiler::common;
namespace tool   = ::rocprofiler::tool;

namespace
{
constexpr uint32_t lds_block_size = 128 * 4;

auto destructors = new std::vector<std::function<void()>>{};

template <typename Tp>
Tp&
get_dereference(Tp* ptr)
{
    return *CHECK_NOTNULL(ptr);
}

auto
get_destructors_lock()
{
    static auto _mutex = std::mutex{};
    return std::unique_lock<std::mutex>{_mutex};
}

template <typename Tp>
Tp*&
add_destructor(Tp*& ptr)
{
    auto _lk = get_destructors_lock();
    destructors->emplace_back([&ptr]() {
        delete ptr;
        ptr = nullptr;
    });
    return ptr;
}

#define ADD_DESTRUCTOR(PTR)                                                                        \
    {                                                                                              \
        static auto _once = std::once_flag{};                                                      \
        std::call_once(_once, []() { add_destructor(PTR); });                                      \
    }

tool::output_file*&
hsa_stats_file()
{
    static auto* _v = new tool::output_file{"hsa_stats",
                                            tool::csv::stats_csv_encoder{},
                                            {
                                                "Name",
                                                "Calls",
                                                "TotalDurationNs",
                                                "AverageNs",
                                                "Percentage",
                                                "MinNs",
                                                "MaxNs",
                                                "StdDev",
                                            }};
    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file&
get_hsa_stats_file()
{
    return get_dereference(hsa_stats_file());
}

tool::output_file*&
get_hsa_api_file()
{
    static auto* _v = new tool::output_file{"hsa_api_trace",
                                            tool::csv::api_csv_encoder{},
                                            {"Domain",
                                             "Function",
                                             "Process_Id",
                                             "Thread_Id",
                                             "Correlation_Id",
                                             "Start_Timestamp",
                                             "End_Timestamp"}};
    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file&
get_hsa_api_trace_file()
{
    return get_dereference(get_hsa_api_file());
}

tool::output_file*&
hip_stats_file()
{
    static auto* _v = new tool::output_file{"hip_stats",
                                            tool::csv::stats_csv_encoder{},
                                            {
                                                "Name",
                                                "Calls",
                                                "TotalDurationNs",
                                                "AverageNs",
                                                "Percentage",
                                                "MinNs",
                                                "MaxNs",
                                                "StdDev",
                                            }};
    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file&
get_hip_stats_file()
{
    return get_dereference(hip_stats_file());
}

tool::output_file*&
get_hip_api_file()
{
    static auto* _v = new tool::output_file{"hip_api_trace",
                                            tool::csv::api_csv_encoder{},
                                            {"Domain",
                                             "Function",
                                             "Process_Id",
                                             "Thread_Id",
                                             "Correlation_Id",
                                             "Start_Timestamp",
                                             "End_Timestamp"}};
    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file&
get_hip_api_trace_file()
{
    return get_dereference(get_hip_api_file());
}

tool::output_file*&
get_agent_info_file_impl()
{
    static auto* _v = new tool::output_file{"agent_info",
                                            tool::csv::agent_info_csv_encoder{},
                                            {"Node_Id",
                                             "Logical_Node_Id",
                                             "Agent_Type",
                                             "Cpu_Cores_Count",
                                             "Simd_Count",
                                             "Cpu_Core_Id_Base",
                                             "Simd_Id_Base",
                                             "Max_Waves_Per_Simd",
                                             "Lds_Size_In_Kb",
                                             "Gds_Size_In_Kb",
                                             "Num_Gws",
                                             "Wave_Front_Size",
                                             "Num_Xcc",
                                             "Cu_Count",
                                             "Array_Count",
                                             "Num_Shader_Banks",
                                             "Simd_Arrays_Per_Engine",
                                             "Cu_Per_Simd_Array",
                                             "Simd_Per_Cu",
                                             "Max_Slots_Scratch_Cu",
                                             "Gfx_Target_Version",
                                             "Vendor_Id",
                                             "Device_Id",
                                             "Location_Id",
                                             "Domain",
                                             "Drm_Render_Minor",
                                             "Num_Sdma_Engines",
                                             "Num_Sdma_Xgmi_Engines",
                                             "Num_Sdma_Queues_Per_Engine",
                                             "Num_Cp_Queues",
                                             "Max_Engine_Clk_Ccompute",
                                             "Max_Engine_Clk_Fcompute",
                                             "Sdma_Fw_Version",
                                             "Fw_Version",
                                             "Capability",
                                             "Cu_Per_Engine",
                                             "Max_Waves_Per_Cu",
                                             "Family_Id",
                                             "Workgroup_Max_Size",
                                             "Grid_Max_Size",
                                             "Local_Mem_Size",
                                             "Hive_Id",
                                             "Gpu_Id",
                                             "Workgroup_Max_Dim_X",
                                             "Workgroup_Max_Dim_Y",
                                             "Workgroup_Max_Dim_Z",
                                             "Grid_Max_Dim_X",
                                             "Grid_Max_Dim_Y",
                                             "Grid_Max_Dim_Z",
                                             "Name",
                                             "Vendor_Name",
                                             "Product_Name",
                                             "Model_Name"}};
    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file&
get_agent_info_file()
{
    return get_dereference(get_agent_info_file_impl());
}

tool::output_file*&
kernel_stats_file()
{
    static auto* _v = new tool::output_file{"kernel_stats",
                                            tool::csv::stats_csv_encoder{},
                                            {
                                                "Name",
                                                "Calls",
                                                "TotalDurationNs",
                                                "AverageNs",
                                                "Percentage",
                                                "MinNs",
                                                "MaxNs",
                                                "StdDev",
                                            }};
    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file&
get_kernel_stats_file()
{
    return get_dereference(kernel_stats_file());
}

tool::output_file*&
get_kernel_file()
{
    static auto* _v = new tool::output_file{"kernel_trace",
                                            tool::csv::kernel_trace_csv_encoder{},
                                            {"Kind",
                                             "Agent_Id",
                                             "Queue_Id",
                                             "Kernel_Id",
                                             "Kernel_Name",
                                             "Correlation_Id",
                                             "Start_Timestamp",
                                             "End_Timestamp",
                                             "Private_Segment_Size",
                                             "Group_Segment_Size",
                                             "Workgroup_Size_X",
                                             "Workgroup_Size_Y",
                                             "Workgroup_Size_Z",
                                             "Grid_Size_X",
                                             "Grid_Size_Y",
                                             "Grid_Size_Z"}};
    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file&
get_kernel_trace_file()
{
    return get_dereference(get_kernel_file());
}

tool::output_file*&
get_counter_collection_file()
{
    static auto* _v = new tool::output_file{"counter_collection",
                                            tool::csv::counter_collection_csv_encoder{},
                                            {"Correlation_Id",
                                             "Dispatch_Id",
                                             "Agent_Id",
                                             "Queue_Id",
                                             "Process_Id",
                                             "Thread_Id",
                                             "Grid_Size",
                                             "Kernel_Name",
                                             "Workgroup_Size",
                                             "LDS_Block_Size",
                                             "Scratch_Size",
                                             "VGPR_Count",
                                             "SGPR_Count",
                                             "Counter_Name",
                                             "Counter_Value"}};

    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file&
get_counter_file()
{
    return get_dereference(get_counter_collection_file());
}

tool::output_file*&
get_memory_copy_trace_file()
{
    static auto* _v = new tool::output_file{"memory_copy_trace",
                                            tool::csv::memory_copy_csv_encoder{},
                                            {"Kind",
                                             "Direction",
                                             "Source_Agent_Id",
                                             "Destination_Agent_Id",
                                             "Correlation_Id",
                                             "Start_Timestamp",
                                             "End_Timestamp"}};
    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file&
get_memory_copy_file()
{
    return get_dereference(get_memory_copy_trace_file());
}

tool::output_file*&
memory_copy_stats_file()
{
    static auto* _v = new tool::output_file{"memory_copy_stats",
                                            tool::csv::stats_csv_encoder{},
                                            {
                                                "Name",
                                                "Calls",
                                                "TotalDurationNs",
                                                "AverageNs",
                                                "Percentage",
                                                "MinNs",
                                                "MaxNs",
                                                "StdDev",
                                            }};
    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file&
get_memory_copy_stats_file()
{
    return get_dereference(memory_copy_stats_file());
}

tool::output_file*&
get_marker_api_file()
{
    static auto* _v = new tool::output_file{"marker_api_trace",
                                            tool::csv::marker_csv_encoder{},
                                            {"Domain",
                                             "Function",
                                             "Process_Id",
                                             "Thread_Id",
                                             "Correlation_Id",
                                             "Start_Timestamp",
                                             "End_Timestamp"}};
    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file&
get_marker_api_trace_file()
{
    return get_dereference(get_marker_api_file());
}

tool::output_file*&
get_scratch_memory_trace_file()
{
    static auto* _v = new tool::output_file{"scratch_memory_trace",
                                            tool::csv::scratch_memory_encoder{},
                                            {
                                                "Kind",
                                                "Operation",
                                                "Agent_Id",
                                                "Queue_Id",
                                                "Thread_Id",
                                                "Alloc_flags",
                                                "Start_Timestamp",
                                                "End_Timestamp",
                                            }};
    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file*&
get_scratch_memory_stats_file()
{
    static auto* _v = new tool::output_file{"scratch_memory_stats",
                                            tool::csv::stats_csv_encoder{},
                                            {
                                                "Name",
                                                "Calls",
                                                "TotalDurationNs",
                                                "AverageNs",
                                                "Percentage",
                                                "MinNs",
                                                "MaxNs",
                                                "StdDev",
                                            }};
    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file*&
get_list_basic_metrics_file()
{
    static auto* _v =
        new tool::output_file{"basic_metrics",
                              tool::csv::list_basic_metrics_csv_encoder{},
                              {"Agent-id", "Name", "Description", "Block", "Dimensions"}};
    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file*&
get_list_derived_metrics_file()
{
    static auto* _v =
        new tool::output_file{"derived_metrics",
                              tool::csv::list_derived_metrics_csv_encoder{},
                              {"Agent-id", "Name", "Description", "Expression", "Dimensions"}};
    ADD_DESTRUCTOR(_v);
    return _v;
}

#undef ADD_DESTRUCTOR

struct marker_entry
{
    uint64_t                cid     = 0;
    pid_t                   pid     = getpid();
    pid_t                   tid     = rocprofiler::common::get_tid();
    rocprofiler_user_data_t data    = {};
    std::string             message = {};
};

struct buffer_ids
{
    rocprofiler_buffer_id_t hsa_api_trace      = {};
    rocprofiler_buffer_id_t hip_api_trace      = {};
    rocprofiler_buffer_id_t kernel_trace       = {};
    rocprofiler_buffer_id_t memory_copy_trace  = {};
    rocprofiler_buffer_id_t counter_collection = {};
    rocprofiler_buffer_id_t scratch_memory     = {};

    auto as_array() const
    {
        return std::array<rocprofiler_buffer_id_t, 6>{hsa_api_trace,
                                                      hip_api_trace,
                                                      kernel_trace,
                                                      memory_copy_trace,
                                                      counter_collection,
                                                      scratch_memory};
    }
};

buffer_ids&
get_buffers()
{
    static auto _v = buffer_ids{};
    return _v;
}

using rocprofiler_code_object_data_t = rocprofiler_callback_tracing_code_object_load_data_t;
using rocprofiler_kernel_symbol_data_t =
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

struct kernel_symbol_data : rocprofiler_kernel_symbol_data_t
{
    using base_type = rocprofiler_kernel_symbol_data_t;

    kernel_symbol_data(const base_type& _base)
    : base_type{_base}
    , formatted_kernel_name{tool::format_name(CHECK_NOTNULL(_base.kernel_name))}
    , demangled_kernel_name{common::cxx_demangle(CHECK_NOTNULL(_base.kernel_name))}
    , truncated_kernel_name{common::truncate_name(demangled_kernel_name)}
    {}

    std::string formatted_kernel_name = {};
    std::string demangled_kernel_name = {};
    std::string truncated_kernel_name = {};
};

template <typename Tp>
Tp*
as_pointer(Tp&& _val)
{
    return new Tp{std::forward<Tp>(_val)};
}

using code_object_data_map_t   = std::unordered_map<uint64_t, rocprofiler_code_object_data_t>;
using kernel_symbol_data_map_t = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data>;
using targeted_kernels_set_t   = std::unordered_set<rocprofiler_kernel_id_t>;
using counter_dimension_info_map_t =
    std::unordered_map<uint64_t, std::vector<rocprofiler_record_dimension_info_t>>;
using agent_info_map_t = std::unordered_map<rocprofiler_agent_id_t, rocprofiler_agent_t>;
using marker_msg_t     = std::unordered_map<uint64_t, std::string>;

auto  code_obj_data          = common::Synchronized<code_object_data_map_t, true>{};
auto  kernel_data_mutex      = std::mutex{};
auto* kernel_data            = as_pointer(kernel_symbol_data_map_t{});
auto  marker_msg_mutex       = std::mutex{};
auto* marker_msg_cid         = as_pointer(marker_msg_t{});
auto  counter_dimension_data = common::Synchronized<counter_dimension_info_map_t, true>{};
auto  target_kernels         = common::Synchronized<targeted_kernels_set_t>{};
auto  dispatch_index         = std::atomic<uint64_t>{0};
auto* buffered_name_info     = as_pointer(get_buffer_id_names());
auto* callback_name_info     = as_pointer(get_callback_id_names());
auto* agent_info             = as_pointer(agent_info_map_t{});
auto* tool_functions         = as_pointer(tool_table{});
auto* stats_timestamp        = as_pointer(timestamps_t{});

bool
add_kernel_target(uint64_t _kern_id)
{
    return target_kernels
        .wlock([](targeted_kernels_set_t& _targets_v,
                  uint64_t                _kern_id_v) { return _targets_v.emplace(_kern_id_v); },
               _kern_id)
        .second;
}

bool
is_targeted_kernel(uint64_t _kern_id)
{
    return target_kernels.rlock(
        [](const targeted_kernels_set_t& _targets_v, uint64_t _kern_id_v) {
            return (_targets_v.count(_kern_id_v) > 0);
        },
        _kern_id);
}

auto&
get_client_ctx()
{
    static rocprofiler_context_id_t context_id;
    return context_id;
}

void
flush()
{
    ROCP_INFO << "flushing buffers...";
    for(auto itr : get_buffers().as_array())
    {
        if(itr.handle > 0)
        {
            ROCP_INFO << "flushing buffer " << itr.handle;
            ROCPROFILER_CALL(rocprofiler_flush_buffer(itr), "buffer flush");
        }
    }
    ROCP_INFO << "Buffers flushed";
}

std::string
get_file_name(buffer_type_t buffer_type)
{
    switch(buffer_type)
    {
        case buffer_type_t::ROCPROFILER_TOOL_BUFFER_HSA: return "hsa_trace"; break;
        case buffer_type_t::ROCPROFILER_TOOL_BUFFER_HIP: return "hip_trace"; break;
        case buffer_type_t::ROCPROFILER_TOOL_BUFFER_MARKER_API: return "marker_trace"; break;
        case buffer_type_t::ROCPROFILER_TOOL_BUFFER_MEMORY_COPY: return "memory_copy"; break;
        case buffer_type_t::ROCPROFILER_TOOL_BUFFER_COUNTER_COLLECTION:
            return "counter_collection";
            break;
        case buffer_type_t::ROCPROFILER_TOOL_BUFFER_KERNEL_DISPATCH:
            return "kernel_dispatch";
            break;
        case buffer_type_t::ROCPROFILER_TOOL_BUFFER_SCRATCH_MEMORY: return "scratch_memory"; break;
    }

    ROCP_FATAL << "buffer type " << static_cast<std::underlying_type_t<buffer_type_t>>(buffer_type)
               << " not supported";
    return std::string{};
}

std::string
compose_tmp_file_name(buffer_type_t buffer_type)
{
    return rocprofiler::tool::format(fmt::format("{}/.rocprofv3/{}-{}.dat",
                                                 rocprofiler::tool::get_config().tmp_directory,
                                                 "%ppid%-%pid%",
                                                 get_file_name(buffer_type)));
}

template <typename Tp>
std::tuple<Tp*, tmp_file*>
get_tmp_file_buffer(buffer_type_t type)
{
    static Tp*       _buffer   = new Tp(rocprofiler::common::units::get_page_size());
    static tmp_file* _tmp_file = new tmp_file(compose_tmp_file_name(type));
    return std::tuple(_buffer, _tmp_file);
}

template <typename Tp>
void
offload_buffer(buffer_type_t type)
{
    Tp*       _tmp_buf                    = nullptr;
    tmp_file* _tmp_file                   = nullptr;
    std::tie(_tmp_buf, _tmp_file)         = get_tmp_file_buffer<Tp>(type);
    auto                         _lk      = std::lock_guard<std::mutex>(_tmp_file->file_mutex);
    [[maybe_unused]] static auto _success = _tmp_file->open();
    auto&                        _fs      = _tmp_file->stream;
    _tmp_file->file_pos.emplace(_fs.tellg());
    _tmp_buf->save(_fs);
    _tmp_buf->clear();
    CHECK(_tmp_buf->is_empty() == true);
}

template <typename Tp, typename Tb>
void
write_ring_buffer(Tb* _v, buffer_type_t type)
{
    Tp*       _tmp_buf            = nullptr;
    tmp_file* _tmp_file           = nullptr;
    std::tie(_tmp_buf, _tmp_file) = get_tmp_file_buffer<Tp>(type);
    auto* ptr                     = _tmp_buf->request(false);
    if(ptr == nullptr)
    {
        offload_buffer<Tp>(type);
        ptr = _tmp_buf->request(false);
        CHECK(ptr != nullptr);
    }
    *ptr = std::move(*_v);
}

template <typename Tp>
void
flush_tmp_buffer(buffer_type_t type)
{
    Tp*       _tmp_buf            = nullptr;
    tmp_file* _tmp_file           = nullptr;
    std::tie(_tmp_buf, _tmp_file) = get_tmp_file_buffer<Tp>(type);
    if(!_tmp_buf->is_empty()) offload_buffer<Tp>(type);
}

template <typename Tp>
void
read_tmp_file(buffer_type_t type, std::vector<Tp>& _data)
{
    Tp*       _tmp_buf = nullptr;
    tmp_file* _tmp_file;
    std::tie(_tmp_buf, _tmp_file) = get_tmp_file_buffer<Tp>(type);
    auto  _lk                     = std::lock_guard<std::mutex>{_tmp_file->file_mutex};
    auto& _fs                     = _tmp_file->stream;
    if(_fs.is_open()) _fs.close();
    _tmp_file->open(std::ios::binary | std::ios::in);
    for(auto itr : _tmp_file->file_pos)
    {
        _fs.seekg(itr);  // set to the absolute position
        if(_fs.eof()) break;
        Tp _buffer;
        _buffer.load(_fs);
        _data.emplace_back(std::move(_buffer));
    }
}

std::string_view
get_callback_kind(rocprofiler_callback_tracing_kind_t kind)
{
    return CHECK_NOTNULL(callback_name_info)->kind_names.at(kind);
}

std::string_view
get_callback_op_name(rocprofiler_callback_tracing_kind_t kind, uint32_t op)
{
    return CHECK_NOTNULL(callback_name_info)->operation_names.at(kind).at(op);
}

std::string_view
get_roctx_msg(uint64_t cid)
{
    return CHECK_NOTNULL(marker_msg_cid)->at(cid);
}

void
cntrl_tracing_callback(rocprofiler_callback_tracing_record_t record,
                       rocprofiler_user_data_t*              user_data,
                       void*                                 cb_data)
{
    auto* ctx = static_cast<rocprofiler_context_id_t*>(cb_data);

    if(ctx && record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER &&
           record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause)
        {
            ROCPROFILER_CALL(rocprofiler_stop_context(*ctx), "pausing context");
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
                record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume)
        {
            ROCPROFILER_CALL(rocprofiler_start_context(*ctx), "resuming context");
        }

        auto ts = rocprofiler_timestamp_t{};
        rocprofiler_get_timestamp(&ts);

        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            user_data->value = ts;
        }
        else
        {
            rocprofiler_tool_marker_record_t marker_record;
            marker_record.kind            = record.kind;
            marker_record.phase           = record.phase;
            marker_record.op              = record.operation;
            marker_record.pid             = getpid();
            marker_record.tid             = rocprofiler::common::get_tid();
            marker_record.cid             = record.correlation_id.internal;
            marker_record.start_timestamp = user_data->value;
            marker_record.end_timestamp   = ts;
            buffer_type_t buffer_type     = buffer_type_t::ROCPROFILER_TOOL_BUFFER_MARKER_API;
            write_ring_buffer<marker_api_ring_buffer_t, rocprofiler_tool_marker_record_t>(
                &marker_record, buffer_type);
        }
    }
}

void
callback_tracing_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*              user_data,
                          void*                                 data)
{
    static thread_local auto stacked_range = std::vector<marker_entry>{};
    static auto              global_range =
        common::Synchronized<std::unordered_map<roctx_range_id_t, marker_entry>>{};

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API)
    {
        auto* marker_data =
            static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);

        auto ts = rocprofiler_timestamp_t{};
        rocprofiler_get_timestamp(&ts);

        if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA)
        {
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
            {
                {
                    std::lock_guard<std::mutex> lk(marker_msg_mutex);
                    std::string                 msg = marker_data->args.roctxMarkA.message;
                    CHECK_NOTNULL(marker_msg_cid)->emplace(record.correlation_id.internal, msg);
                }
                rocprofiler_tool_marker_record_t marker_record;
                marker_record.kind            = record.kind;
                marker_record.op              = record.operation;
                marker_record.phase           = record.phase;
                marker_record.pid             = getpid();
                marker_record.tid             = rocprofiler::common::get_tid();
                marker_record.cid             = record.correlation_id.internal;
                marker_record.start_timestamp = ts;
                marker_record.end_timestamp   = ts;
                buffer_type_t buffer_type     = buffer_type_t::ROCPROFILER_TOOL_BUFFER_MARKER_API;
                write_ring_buffer<marker_api_ring_buffer_t, rocprofiler_tool_marker_record_t>(
                    &marker_record, buffer_type);
            }
        }
        else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA)
        {
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
            {
                if(marker_data->args.roctxRangePushA.message)
                {
                    auto& val      = stacked_range.emplace_back();
                    val.message    = marker_data->args.roctxRangePushA.message;
                    val.data.value = ts;
                    val.cid        = record.correlation_id.internal;
                }
            }
        }
        else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop)
        {
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
            {
                ROCP_FATAL_IF(stacked_range.empty())
                    << "roctxRangePop invoked more times than roctxRangePush on thread "
                    << rocprofiler::common::get_tid();

                auto val = stacked_range.back();
                stacked_range.pop_back();
                {
                    std::lock_guard<std::mutex> lk(marker_msg_mutex);
                    std::string                 msg = val.message;
                    CHECK_NOTNULL(marker_msg_cid)->emplace(val.cid, msg);
                }
                rocprofiler_tool_marker_record_t marker_record;
                marker_record.kind            = record.kind;
                marker_record.op              = record.operation;
                marker_record.phase           = record.phase;
                marker_record.pid             = val.pid;
                marker_record.tid             = val.tid;
                marker_record.cid             = val.cid;
                marker_record.start_timestamp = val.data.value;
                marker_record.end_timestamp   = ts;
                buffer_type_t buffer_type     = buffer_type_t::ROCPROFILER_TOOL_BUFFER_MARKER_API;
                write_ring_buffer<marker_api_ring_buffer_t, rocprofiler_tool_marker_record_t>(
                    &marker_record, buffer_type);
            }
        }
        else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA)
        {
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
               marker_data->args.roctxRangeStartA.message)
            {
                auto _id          = marker_data->retval.roctx_range_id_t_retval;
                auto _entry       = marker_entry{};
                _entry.cid        = record.correlation_id.internal;
                _entry.data.value = ts;
                _entry.message    = marker_data->args.roctxRangeStartA.message;
                {
                    std::lock_guard<std::mutex> lk(marker_msg_mutex);
                    std::string                 msg = _entry.message;
                    CHECK_NOTNULL(marker_msg_cid)->emplace(_entry.cid, msg);
                }
                global_range.wlock(
                    [_id, &_entry](auto& map) { map.emplace(_id, std::move(_entry)); });
            }
        }
        else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop)
        {
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
            {
                auto   _id    = marker_data->args.roctxRangeStop.id;
                auto&& _entry = global_range.rlock(
                    [](const auto& map, auto _key) { return map.at(_key); }, _id);
                rocprofiler_tool_marker_record_t marker_record;
                marker_record.kind            = record.kind;
                marker_record.op              = record.operation;
                marker_record.phase           = record.phase;
                marker_record.pid             = _entry.pid;
                marker_record.tid             = 0;
                marker_record.cid             = _entry.cid;
                marker_record.start_timestamp = _entry.data.value;
                marker_record.end_timestamp   = ts;
                buffer_type_t buffer_type     = buffer_type_t::ROCPROFILER_TOOL_BUFFER_MARKER_API;
                write_ring_buffer<marker_api_ring_buffer_t, rocprofiler_tool_marker_record_t>(
                    &marker_record, buffer_type);
                global_range.wlock([](auto& map, auto _key) { return map.erase(_key); }, _id);
            }
        }
        else
        {
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
            {
                user_data->value = ts;
            }
            else
            {
                rocprofiler_tool_marker_record_t marker_record;
                marker_record.kind            = record.kind;
                marker_record.op              = record.operation;
                marker_record.phase           = record.phase;
                marker_record.pid             = getpid();
                marker_record.tid             = rocprofiler::common::get_tid();
                marker_record.cid             = record.correlation_id.internal;
                marker_record.start_timestamp = user_data->value;
                marker_record.end_timestamp   = ts;
                buffer_type_t buffer_type     = buffer_type_t::ROCPROFILER_TOOL_BUFFER_MARKER_API;
                write_ring_buffer<marker_api_ring_buffer_t, rocprofiler_tool_marker_record_t>(
                    &marker_record, buffer_type);
            }
        }
    }

    (void) record;
    (void) user_data;
    (void) data;
}

void
code_object_tracing_callback(rocprofiler_callback_tracing_record_t record,
                             rocprofiler_user_data_t*              user_data,
                             void*                                 data)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            auto* obj_data = static_cast<rocprofiler_code_object_data_t*>(record.payload);
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
            {
                code_obj_data.wlock(
                    [](code_object_data_map_t& cdata, rocprofiler_code_object_data_t* obj_data_v) {
                        cdata.emplace(obj_data_v->code_object_id, *obj_data_v);
                    },
                    CHECK_NOTNULL(obj_data));
            }
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            flush();
        }
    }

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* sym_data = static_cast<rocprofiler_kernel_symbol_data_t*>(record.payload);
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            std::pair<kernel_symbol_data_map_t::iterator, bool> itr;
            {
                std::lock_guard<std::mutex> lk(kernel_data_mutex);
                itr = CHECK_NOTNULL(kernel_data)
                          ->emplace(sym_data->kernel_id,
                                    kernel_symbol_data{get_dereference(sym_data)});
            }
            ROCP_WARNING_IF(!itr.second)
                << "duplicate kernel symbol data for kernel_id=" << sym_data->kernel_id;

            // add the kernel to the kernel_targets if
            if(itr.second)
            {
                // if kernel name is provided by user then by default all kernels in the application
                // are targeted
                if(tool::get_config().kernel_names.empty())
                {
                    add_kernel_target(sym_data->kernel_id);
                }
                else
                {
                    const auto& kernel_info = itr.first->second;
                    for(const auto& name : tool::get_config().kernel_names)
                    {
                        if(name == kernel_info.truncated_kernel_name)
                        {
                            add_kernel_target(itr.first->first);
                            break;
                        }
                        else
                        {
                            auto dkernel_name = std::string_view{kernel_info.demangled_kernel_name};
                            auto pos          = dkernel_name.find(name);
                            // if the demangled kernel name contains name and the next character is
                            // '(' then mark as found
                            if(pos != std::string::npos && (pos + 1) < dkernel_name.size() &&
                               dkernel_name.at(pos + 1) == '(')
                            {
                                add_kernel_target(itr.first->first);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    (void) user_data;
    (void) data;
}

std::string_view
get_kernel_name(uint64_t kernel_id)
{
    return CHECK_NOTNULL(kernel_data)->at(kernel_id).formatted_kernel_name;
}

std::string_view
get_domain_name(rocprofiler_buffer_tracing_kind_t record_kind)
{
    return CHECK_NOTNULL(buffered_name_info)->kind_names.at(record_kind);
}

uint64_t
get_agent_node_id(rocprofiler_agent_id_t agent_id)
{
    return agent_info->at(agent_id).logical_node_id;
}

std::string_view
get_operation_name(rocprofiler_buffer_tracing_kind_t kind, rocprofiler_tracing_operation_t op)
{
    return CHECK_NOTNULL(buffered_name_info)->operation_names.at(kind).at(op);
}

void
buffered_tracing_callback(rocprofiler_context_id_t /*context*/,
                          rocprofiler_buffer_id_t /*buffer_id*/,
                          rocprofiler_record_header_t** headers,
                          size_t                        num_headers,
                          void* /*user_data*/,
                          uint64_t /*drop_count*/)
{
    ROCP_INFO << "Executing buffered tracing callback for " << num_headers << " headers";

    ROCP_ERROR_IF(headers == nullptr)
        << "rocprofiler invoked a buffer callback with a null pointer to the array of headers. "
           "this should never happen";

    if(!headers) return;

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING)
        {
            if(header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
                    header->payload);

                buffer_type_t buffer_type = buffer_type_t::ROCPROFILER_TOOL_BUFFER_KERNEL_DISPATCH;

                write_ring_buffer<kernel_dispatch_ring_buffer_t,
                                  rocprofiler_buffer_tracing_kernel_dispatch_record_t>(record,
                                                                                       buffer_type);
            }

            else if(header->kind == ROCPROFILER_BUFFER_TRACING_HSA_CORE_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(header->payload);

                buffer_type_t buffer_type = buffer_type_t::ROCPROFILER_TOOL_BUFFER_HSA;
                write_ring_buffer<hsa_ring_buffer_t, rocprofiler_buffer_tracing_hsa_api_record_t>(
                    record, buffer_type);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload);

                buffer_type_t buffer_type = buffer_type_t::ROCPROFILER_TOOL_BUFFER_MEMORY_COPY;
                write_ring_buffer<memory_copy_ring_buffer_t,
                                  rocprofiler_buffer_tracing_memory_copy_record_t>(record,
                                                                                   buffer_type);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_scratch_memory_record_t*>(
                    header->payload);

                buffer_type_t buffer_type = buffer_type_t::ROCPROFILER_TOOL_BUFFER_SCRATCH_MEMORY;
                write_ring_buffer<scratch_memory_ring_buffer_t,
                                  rocprofiler_buffer_tracing_scratch_memory_record_t>(record,
                                                                                      buffer_type);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);
                buffer_type_t buffer_type = buffer_type_t::ROCPROFILER_TOOL_BUFFER_HIP;
                write_ring_buffer<hip_ring_buffer_t, rocprofiler_buffer_tracing_hip_api_record_t>(
                    record, buffer_type);
            }
            else
            {
                ROCP_FATAL << fmt::format(
                    "unsupported category + kind: {} + {}", header->category, header->kind);
            }
        }
    }
}

using counter_vec_t = std::vector<rocprofiler_counter_id_t>;
using agent_counter_map_t =
    std::unordered_map<rocprofiler_agent_id_t, std::optional<rocprofiler_profile_config_id_t>>;

rocprofiler_status_t
dimensions_info_callback(rocprofiler_counter_id_t                   id,
                         const rocprofiler_record_dimension_info_t* dim_info,
                         long unsigned int                          num_dims,
                         void*                                      user_data)
{
    if(user_data != nullptr)
    {
        auto* dimensions_info =
            static_cast<std::vector<rocprofiler_record_dimension_info_t>*>(user_data);
        for(size_t j = 0; j < num_dims; j++)
            dimensions_info->push_back(dim_info[j]);
    }
    else
    {
        counter_dimension_data.wlock(
            [&id, &dim_info, &num_dims](counter_dimension_info_map_t& counter_dimension_data_v) {
                std::vector<rocprofiler_record_dimension_info_t> dimensions;
                for(size_t dim = 0; dim < num_dims; dim++)
                    dimensions.emplace_back(dim_info[dim]);
                counter_dimension_data_v.emplace(std::make_pair(id.handle, dimensions));
            });
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

// this function creates a rocprofiler profile config on the first entry
auto
get_agent_profile(rocprofiler_agent_id_t agent_id)
{
    static auto data = common::Synchronized<agent_counter_map_t>{};

    auto profile = std::optional<rocprofiler_profile_config_id_t>{};
    data.ulock(
        [agent_id, &profile](const agent_counter_map_t& data_v) {
            auto itr = data_v.find(agent_id);
            if(itr != data_v.end())
            {
                profile = itr->second;
                return true;
            }
            return false;
        },
        [agent_id, &profile](agent_counter_map_t& data_v) {
            auto counters_v = counter_vec_t{};
            ROCPROFILER_CALL(
                rocprofiler_iterate_agent_supported_counters(
                    agent_id,
                    [](rocprofiler_agent_id_t,
                       rocprofiler_counter_id_t* counters,
                       size_t                    num_counters,
                       void*                     user_data) {
                        auto* vec = static_cast<counter_vec_t*>(user_data);
                        for(size_t i = 0; i < num_counters; i++)
                        {
                            ROCPROFILER_CALL(rocprofiler_iterate_counter_dimensions(
                                                 counters[i], dimensions_info_callback, nullptr),
                                             "iterate_dimension_info");

                            rocprofiler_counter_info_v0_t info;

                            ROCPROFILER_CALL(
                                rocprofiler_query_counter_info(counters[i],
                                                               ROCPROFILER_COUNTER_INFO_VERSION_0,
                                                               static_cast<void*>(&info)),
                                "Could not query counter_id");

                            if(tool::get_config().counters.count(info.name) > 0)
                                vec->emplace_back(counters[i]);
                        }
                        return ROCPROFILER_STATUS_SUCCESS;
                    },
                    static_cast<void*>(&counters_v)),
                "iterate agent supported counters");

            if(!counters_v.empty())
            {
                auto profile_v = rocprofiler_profile_config_id_t{};
                ROCPROFILER_CALL(rocprofiler_create_profile_config(
                                     agent_id, counters_v.data(), counters_v.size(), &profile_v),
                                 "Could not construct profile cfg");
                profile = profile_v;
            }

            data_v.emplace(agent_id, profile);
            return true;
        });

    return profile;
}

struct counter_dispatch_data
{
    uint64_t thread_id      = 0;
    uint64_t dispatch_index = 0;
};

void
dispatch_callback(rocprofiler_profile_counting_dispatch_data_t dispatch_data,
                  rocprofiler_profile_config_id_t*             config,
                  rocprofiler_user_data_t*                     user_data,
                  void* /*callback_data_args*/)
{
    auto kernel_id = dispatch_data.dispatch_info.kernel_id;
    auto agent_id  = dispatch_data.dispatch_info.agent_id;

    if(!is_targeted_kernel(kernel_id))
    {
        return;
    }
    else if(auto profile = get_agent_profile(agent_id))
    {
        *config        = *profile;
        user_data->ptr = new counter_dispatch_data{.thread_id      = common::get_tid(),
                                                   .dispatch_index = ++dispatch_index};
    }
}

std::string
get_counter_info_name(uint64_t record_id)
{
    auto info       = rocprofiler_counter_info_v0_t{};
    auto counter_id = rocprofiler_counter_id_t{};
    ROCPROFILER_CALL(rocprofiler_query_record_counter_id(record_id, &counter_id),
                     "query record counter id");
    ROCPROFILER_CALL(rocprofiler_query_counter_info(rocprofiler_counter_id_t{counter_id},
                                                    ROCPROFILER_COUNTER_INFO_VERSION_0,
                                                    static_cast<void*>(&info)),
                     "query counter info");
    std::string counter_name = info.name;
    return counter_name;
}

void
counter_record_callback(rocprofiler_profile_counting_dispatch_data_t dispatch_data,
                        rocprofiler_record_counter_t*                record_data,
                        size_t                                       record_count,
                        rocprofiler_user_data_t                      user_data,
                        void* /*callback_data_args*/)
{
    auto        kernel_id           = dispatch_data.dispatch_info.kernel_id;
    const auto* cnt_dispatch_data_v = static_cast<counter_dispatch_data*>(user_data.ptr);

    rocprofiler_tool_counter_collection_record_t counter_record;
    counter_record.dispatch_data          = dispatch_data;
    counter_record.dispatch_index         = cnt_dispatch_data_v->dispatch_index;
    counter_record.thread_id              = cnt_dispatch_data_v->thread_id;
    counter_record.pid                    = getpid();
    const kernel_symbol_data* kernel_info = nullptr;

    {
        std::lock_guard<std::mutex> lk(kernel_data_mutex);
        kernel_info = &(CHECK_NOTNULL(kernel_data)->at(kernel_id));
    }

    auto lds_block_size_v =
        (kernel_info->group_segment_size + (lds_block_size - 1)) & ~(lds_block_size - 1);

    counter_record.private_segment_size = kernel_info->private_segment_size;
    counter_record.arch_vgpr_count      = kernel_info->arch_vgpr_count;
    counter_record.sgpr_count           = kernel_info->sgpr_count;
    counter_record.lds_block_size_v     = lds_block_size_v;

    ROCP_FATAL_IF(!kernel_info) << "missing kernel information for kernel_id=" << kernel_id;

    ROCP_ERROR_IF(record_count == 0) << "zero record count for kernel_id=" << kernel_id
                                     << " (name=" << kernel_info->kernel_name << ")";

    for(size_t count = 0; count < record_count; count++)
        counter_record.profiler_record.push_back(
            static_cast<rocprofiler_record_counter_t>(record_data[count]));

    buffer_type_t buffer_type = buffer_type_t::ROCPROFILER_TOOL_BUFFER_COUNTER_COLLECTION;
    write_ring_buffer<counter_collection_ring_buffer_t,
                      rocprofiler_tool_counter_collection_record_t>(&counter_record, buffer_type);

    delete cnt_dispatch_data_v;
}

rocprofiler_status_t
list_metrics_iterate_agents(rocprofiler_agent_version_t,
                            const void** agents,
                            size_t       num_agents,
                            void*)
{
    for(size_t idx = 0; idx < num_agents; idx++)
    {
        const auto* agent      = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);
        auto        counters_v = counter_vec_t{};
        // TODO(aelwazir): To be changed back to use node id once ROCR fixes
        // the hsa_agents to use the real node id
        uint32_t node_id = agent->logical_node_id;
        ROCPROFILER_CALL(
            rocprofiler_iterate_agent_supported_counters(
                agent->id,
                [](rocprofiler_agent_id_t,
                   rocprofiler_counter_id_t* counters,
                   size_t                    num_counters,
                   void*                     user_data) {
                    auto* agent_node_id = static_cast<uint32_t*>(user_data);
                    for(size_t i = 0; i < num_counters; i++)
                    {
                        rocprofiler_counter_info_v0_t counter_info;
                        auto dimensions = std::vector<rocprofiler_record_dimension_info_t>{};
                        ROCPROFILER_CALL(
                            rocprofiler_iterate_counter_dimensions(counters[i],
                                                                   dimensions_info_callback,
                                                                   static_cast<void*>(&dimensions)),
                            "iterate_dimension_info");

                        ROCPROFILER_CALL(
                            rocprofiler_query_counter_info(counters[i],
                                                           ROCPROFILER_COUNTER_INFO_VERSION_0,
                                                           static_cast<void*>(&counter_info)),
                            "Could not query counter_id");

                        auto dimensions_info = std::stringstream{};
                        for(size_t j = 0; j != dimensions.size(); j++)
                        {
                            dimensions_info << dimensions[j].name
                                            << "[0:" << dimensions[j].instance_size - 1 << "]";
                            if(j != dimensions.size() - 1) dimensions_info << "\t";
                        }
                        if(!counter_info.is_derived && tool::get_config().list_metrics &&
                           !std::string(counter_info.block).empty())
                        {
                            auto counter_info_ss = std::stringstream{};
                            if(tool::get_config().list_metrics_output_file)
                            {
                                tool::csv::list_basic_metrics_csv_encoder::write_row(
                                    counter_info_ss,
                                    *agent_node_id,
                                    counter_info.name,
                                    counter_info.description,
                                    counter_info.block,
                                    dimensions_info.str());
                                get_dereference(get_list_basic_metrics_file())
                                    << counter_info_ss.str();
                            }
                            else
                            {
                                counter_info_ss << "gpu-agent" << *agent_node_id << ":"
                                                << "\t" << counter_info.name << "\n";
                                counter_info_ss << "Description:"
                                                << "\t" << counter_info.description << "\n";
                                counter_info_ss << "Block:"
                                                << "\t" << counter_info.block << "\n";
                                counter_info_ss << "Dimensions:"
                                                << "\t" << dimensions_info.str() << "\n";
                                counter_info_ss << "\n";
                                std::cout << counter_info_ss.str();
                            }
                        }
                        else if(counter_info.is_derived && tool::get_config().list_metrics)
                        {
                            auto counter_info_ss = std::stringstream{};
                            if(tool::get_config().list_metrics_output_file)
                            {
                                tool::csv::list_derived_metrics_csv_encoder::write_row(
                                    counter_info_ss,
                                    *agent_node_id,
                                    counter_info.name,
                                    counter_info.description,
                                    counter_info.expression,
                                    dimensions_info.str());
                                get_dereference(get_list_derived_metrics_file())
                                    << counter_info_ss.str();
                            }
                            else
                            {
                                counter_info_ss << "gpu-agent" << *agent_node_id << ":"
                                                << "\t" << counter_info.name << "\n"
                                                << "Description: " << counter_info.description
                                                << "\n";
                                counter_info_ss << "Expression: " << counter_info.expression
                                                << "\n";
                                counter_info_ss << "Dimensions: " << dimensions_info.str() << "\n";
                                counter_info_ss << "\n";
                                std::cout << counter_info_ss.str();
                            }
                        }
                    }
                    return ROCPROFILER_STATUS_SUCCESS;
                },
                reinterpret_cast<void*>(&node_id)),
            "Iterate rocprofiler counters");
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_client_finalize_t client_finalizer  = nullptr;
rocprofiler_client_id_t*      client_identifier = nullptr;

timestamps_t*
get_app_timestamps()
{
    return stats_timestamp;
}

void
init_tool_table()
{
    // agent and timestamp functions
    tool_functions->tool_get_agent_node_id_fn  = get_agent_node_id;
    tool_functions->tool_get_app_timestamps_fn = get_app_timestamps;

    // name functions
    tool_functions->tool_get_domain_name_fn       = get_domain_name;
    tool_functions->tool_get_kernel_name_fn       = get_kernel_name;
    tool_functions->tool_get_operation_name_fn    = get_operation_name;
    tool_functions->tool_get_counter_info_name_fn = get_counter_info_name;
    tool_functions->tool_get_callback_kind_fn     = get_callback_kind;
    tool_functions->tool_get_callback_op_name_fn  = get_callback_op_name;
    tool_functions->tool_get_roctx_msg_fn         = get_roctx_msg;

    // trace files
    tool_functions->tool_get_agent_info_file_fn         = get_agent_info_file;
    tool_functions->tool_get_kernel_trace_file_fn       = get_kernel_trace_file;
    tool_functions->tool_get_hsa_api_trace_file_fn      = get_hsa_api_trace_file;
    tool_functions->tool_get_hip_api_trace_file_fn      = get_hip_api_trace_file;
    tool_functions->tool_get_memory_copy_trace_file_fn  = get_memory_copy_file;
    tool_functions->tool_get_counter_collection_file_fn = get_counter_file;
    tool_functions->tool_get_marker_api_trace_file_fn   = get_marker_api_trace_file;
    tool_functions->tool_get_scratch_memory_file_fn     = get_scratch_memory_trace_file;

    // stats files
    tool_functions->tool_get_kernel_stats_file_fn         = get_kernel_stats_file;
    tool_functions->tool_get_hip_stats_file_fn            = get_hip_stats_file;
    tool_functions->tool_get_hsa_stats_file_fn            = get_hsa_stats_file;
    tool_functions->tool_get_memory_copy_stats_file_fn    = get_memory_copy_stats_file;
    tool_functions->tool_get_scratch_memory_stats_file_fn = get_scratch_memory_stats_file;
}

void
fini_tool_table()
{
    *tool_functions = tool_table{};
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    client_finalizer = fini_func;

    constexpr uint64_t buffer_size      = 4096;
    constexpr uint64_t buffer_watermark = 4096;

    rocprofiler_get_timestamp(&(stats_timestamp->app_start_time));

    init_tool_table();

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "create context failed");

    auto code_obj_ctx = rocprofiler_context_id_t{};
    ROCPROFILER_CALL(rocprofiler_create_context(&code_obj_ctx), "failed to create context");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(code_obj_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       code_object_tracing_callback,
                                                       nullptr),
        "code object tracing configure failed");
    ROCPROFILER_CALL(rocprofiler_start_context(code_obj_ctx), "start context failed");

    if(tool::get_config().marker_api_trace)
    {
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                             get_client_ctx(),
                             ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                             nullptr,
                             0,
                             callback_tracing_callback,
                             nullptr),
                         "callback tracing service failed to configure");

        auto pause_resume_ctx = rocprofiler_context_id_t{};
        ROCPROFILER_CALL(rocprofiler_create_context(&pause_resume_ctx), "failed to create context");

        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                             pause_resume_ctx,
                             ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                             nullptr,
                             0,
                             cntrl_tracing_callback,
                             static_cast<void*>(&get_client_ctx())),
                         "callback tracing service failed to configure");

        ROCPROFILER_CALL(rocprofiler_start_context(pause_resume_ctx), "start context failed");
    }

    if(tool::get_config().kernel_trace)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                                   buffer_size,
                                                   buffer_watermark,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   buffered_tracing_callback,
                                                   tool_data,
                                                   &get_buffers().kernel_trace),
                         "buffer creation");

        ROCPROFILER_CALL(
            rocprofiler_configure_buffer_tracing_service(get_client_ctx(),
                                                         ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                         nullptr,
                                                         0,
                                                         get_buffers().kernel_trace),
            "buffer tracing service for kernel dispatch configure");
    }

    if(tool::get_config().memory_copy_trace)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                                   buffer_size,
                                                   buffer_watermark,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   buffered_tracing_callback,
                                                   nullptr,
                                                   &get_buffers().memory_copy_trace),
                         "create memory copy buffer");

        ROCPROFILER_CALL(
            rocprofiler_configure_buffer_tracing_service(get_client_ctx(),
                                                         ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                                         nullptr,
                                                         0,
                                                         get_buffers().memory_copy_trace),
            "buffer tracing service for memory copy configure");
    }

    if(tool::get_config().scratch_memory)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                                   buffer_size,
                                                   buffer_watermark,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   buffered_tracing_callback,
                                                   tool_data,
                                                   &get_buffers().scratch_memory),
                         "buffer creation");

        ROCPROFILER_CALL(
            rocprofiler_configure_buffer_tracing_service(get_client_ctx(),
                                                         ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY,
                                                         nullptr,
                                                         0,
                                                         get_buffers().scratch_memory),
            "buffer tracing service for scratch memory configure");
    }

    if(tool::get_config().hsa_core_api_trace || tool::get_config().hsa_amd_ext_api_trace ||
       tool::get_config().hsa_image_ext_api_trace || tool::get_config().hsa_finalizer_ext_api_trace)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                                   buffer_size,
                                                   buffer_watermark,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   buffered_tracing_callback,
                                                   tool_data,
                                                   &get_buffers().hsa_api_trace),
                         "buffer creation");

        using optpair_t = std::pair<bool, rocprofiler_buffer_tracing_kind_t>;
        for(auto itr : {optpair_t{tool::get_config().hsa_core_api_trace,
                                  ROCPROFILER_BUFFER_TRACING_HSA_CORE_API},
                        optpair_t{tool::get_config().hsa_amd_ext_api_trace,
                                  ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API},
                        optpair_t{tool::get_config().hsa_image_ext_api_trace,
                                  ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API},
                        optpair_t{tool::get_config().hsa_finalizer_ext_api_trace,
                                  ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API}})
        {
            if(itr.first)
            {
                ROCPROFILER_CALL(
                    rocprofiler_configure_buffer_tracing_service(
                        get_client_ctx(), itr.second, nullptr, 0, get_buffers().hsa_api_trace),
                    "buffer tracing service for hsa api configure");
            }
        }
    }

    if(tool::get_config().hip_runtime_api_trace || tool::get_config().hip_compiler_api_trace)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                                   buffer_size,
                                                   buffer_watermark,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   buffered_tracing_callback,
                                                   tool_data,
                                                   &get_buffers().hip_api_trace),
                         "buffer creation");

        if(tool::get_config().hip_runtime_api_trace)
        {
            ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                                 get_client_ctx(),
                                 ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
                                 nullptr,
                                 0,
                                 get_buffers().hip_api_trace),
                             "buffer tracing service for hip api configure");
        }

        if(tool::get_config().hip_compiler_api_trace)
        {
            ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                                 get_client_ctx(),
                                 ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API,
                                 nullptr,
                                 0,
                                 get_buffers().hip_api_trace),
                             "buffer tracing service for hip compiler api configure");
        }
    }

    if(tool::get_config().counter_collection)
    {
        ROCPROFILER_CALL(
            rocprofiler_configure_callback_dispatch_profile_counting_service(
                get_client_ctx(), dispatch_callback, nullptr, counter_record_callback, nullptr),
            "Could not setup counting service");
    }

    for(auto itr : get_buffers().as_array())
    {
        if(itr.handle > 0)
        {
            auto cb_thread = rocprofiler_callback_thread_t{};

            ROCP_INFO << "creating dedicated callback thread for buffer " << itr.handle;
            ROCPROFILER_CALL(rocprofiler_create_callback_thread(&cb_thread),
                             "creating callback thread");

            ROCP_INFO << "assigning buffer " << itr.handle << " to callback thread "
                      << cb_thread.handle;
            ROCPROFILER_CALL(rocprofiler_assign_callback_thread(itr, cb_thread),
                             "assigning callback thread");
        }
    }

    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context failed");

    return 0;
}

void
api_registration_callback(rocprofiler_intercept_table_t,
                          uint64_t,
                          uint64_t,
                          void**,
                          uint64_t,
                          void*)
{
    ROCPROFILER_CALL(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                                        list_metrics_iterate_agents,
                                                        sizeof(rocprofiler_agent_t),
                                                        nullptr),
                     "Iterate rocporfiler agents")
}

namespace
{
template <typename Tp>
void
generate_output(buffer_type_t buffer_type)
{
    auto _data = std::vector<Tp>{};
    flush_tmp_buffer<Tp>(buffer_type);
    read_tmp_file<Tp>(buffer_type, _data);

    if(tool::get_config().output_format == "CSV")
        rocprofiler::tool::generate_csv(tool_functions, _data);

    auto [_tmp_buf, _tmp_file] = get_tmp_file_buffer<Tp>(buffer_type);
    _tmp_buf->destroy();
    delete _tmp_buf;
    delete _tmp_file;
}
}  // namespace

void
tool_fini(void* /*tool_data*/)
{
    client_identifier = nullptr;
    client_finalizer  = nullptr;

    rocprofiler_get_timestamp(&(stats_timestamp->app_end_time));

    rocprofiler_stop_context(get_client_ctx());
    flush();

    std::string_view output_format = tool::get_config().output_format;

    if(output_format == "CSV")
    {
        auto _agents = std::vector<rocprofiler_agent_v0_t>{};
        _agents.reserve(agent_info->size());
        for(auto& itr : *agent_info)
            _agents.emplace_back(itr.second);
        rocprofiler::tool::generate_csv(tool_functions, _agents);
    }

    if(tool::get_config().kernel_trace)
    {
        generate_output<kernel_dispatch_ring_buffer_t>(
            buffer_type_t::ROCPROFILER_TOOL_BUFFER_KERNEL_DISPATCH);
    }

    if(tool::get_config().hsa_core_api_trace || tool::get_config().hsa_amd_ext_api_trace ||
       tool::get_config().hsa_image_ext_api_trace || tool::get_config().hsa_finalizer_ext_api_trace)
    {
        generate_output<hsa_ring_buffer_t>(buffer_type_t::ROCPROFILER_TOOL_BUFFER_HSA);
    }

    if(tool::get_config().hip_runtime_api_trace || tool::get_config().hip_compiler_api_trace)
    {
        generate_output<hip_ring_buffer_t>(buffer_type_t::ROCPROFILER_TOOL_BUFFER_HIP);
    }

    if(tool::get_config().memory_copy_trace)
    {
        generate_output<memory_copy_ring_buffer_t>(
            buffer_type_t::ROCPROFILER_TOOL_BUFFER_MEMORY_COPY);
    }

    if(tool::get_config().marker_api_trace)
    {
        generate_output<marker_api_ring_buffer_t>(
            buffer_type_t::ROCPROFILER_TOOL_BUFFER_MARKER_API);
    }

    if(tool::get_config().counter_collection)
    {
        generate_output<counter_collection_ring_buffer_t>(
            buffer_type_t::ROCPROFILER_TOOL_BUFFER_COUNTER_COLLECTION);
    }

    if(tool::get_config().scratch_memory)
    {
        generate_output<scratch_memory_ring_buffer_t>(
            buffer_type_t::ROCPROFILER_TOOL_BUFFER_SCRATCH_MEMORY);
    }

    fini_tool_table();
    if(destructors)
    {
        for(const auto& itr : *destructors)
            itr();
        delete destructors;
        destructors = nullptr;
    }

#if defined(CODECOV) && CODECOV > 0
    __gcov_dump();
#endif
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    auto logging_cfg = rocprofiler::common::logging_config{.install_failure_handler = true};
    common::init_logging("ROCPROF", logging_cfg);
    FLAGS_colorlogtostderr = true;

    // set the client name
    id->name = "rocprofv3";

    // store client info
    client_identifier = id;

    // note that rocprofv3 is not the primary tool
    ROCP_WARNING_IF(priority > 0) << id->name << " has a priority of " << priority
                                  << " (not primary tool)";

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    ::atexit([]() {
        if(client_finalizer && client_identifier) client_finalizer(*client_identifier);
    });

    // ensure these pointers are not leaked
    add_destructor(buffered_name_info);
    add_destructor(callback_name_info);
    add_destructor(marker_msg_cid);
    add_destructor(kernel_data);
    add_destructor(tool_functions);
    add_destructor(agent_info);
    add_destructor(stats_timestamp);

    if(tool::get_config().list_metrics)
    {
        ROCPROFILER_CALL(rocprofiler_at_intercept_table_registration(
                             api_registration_callback, ROCPROFILER_HSA_TABLE, nullptr),
                         "api registration");
        return nullptr;
    }

    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(
            ROCPROFILER_AGENT_INFO_VERSION_0,
            [](rocprofiler_agent_version_t, const void** agents, size_t num_agents, void*) {
                for(size_t i = 0; i < num_agents; ++i)
                {
                    auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
                    agent_info->emplace(agent->id, *agent);
                }
                return ROCPROFILER_STATUS_SUCCESS;
            },
            sizeof(rocprofiler_agent_t),
            nullptr),
        "Iterate rocporfiler agents")

    ROCP_INFO << id->name << " is using rocprofiler-sdk v" << major << "." << minor << "." << patch
              << " (" << runtime_version << ")";

    // create configure data
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};

    // return pointer to configure data
    return &cfg;
    // data passed around all the callbacks
}
