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

#include "generateCSV.hpp"
#include "config.hpp"
#include "csv.hpp"
#include "generateStats.hpp"
#include "helper.hpp"
#include "statistics.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <unistd.h>

#include <cstdint>
#include <iomanip>
#include <string_view>
#include <utility>

namespace rocprofiler
{
namespace tool
{
namespace
{
tool::output_file
get_stats_output_file(std::string name)
{
    return tool::output_file{std::move(name),
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
}

void
write_stats(output_file&& ofs, const stats_entry_vec_t& data_v)
{
    auto data      = stats_entry_vec_t{};
    auto _duration = stats_data_t{};
    for(const auto& [id, value] : data_v)
    {
        data.emplace_back(id, value);
        _duration += value;
    }

    std::sort(data.begin(), data.end(), [](const auto& lhs, const auto& rhs) {
        return (lhs.second.get_sum() > rhs.second.get_sum());
    });

    constexpr float_type one_hundred = 100.0;

    const float_type _total_duration = _duration.get_sum();
    for(const auto& [name, value] : data)
    {
        auto       duration_ns = value.get_sum();
        auto       calls       = value.get_count();
        float_type avg_ns      = value.get_mean();
        float_type percent_v   = (duration_ns / _total_duration) * one_hundred;

        auto _row = std::stringstream{};
        rocprofiler::tool::csv::stats_csv_encoder::write_row<stats_formatter>(_row,
                                                                              name,
                                                                              calls,
                                                                              duration_ns,
                                                                              avg_ns,
                                                                              percentage{percent_v},
                                                                              value.get_min(),
                                                                              value.get_max(),
                                                                              value.get_stddev());
        ofs << _row.str() << std::flush;
    }
}
}  // namespace

void
generate_csv(tool_table* /*tool_functions*/, std::vector<rocprofiler_agent_v0_t>& data)
{
    if(data.empty()) return;

    std::sort(data.begin(), data.end(), [](rocprofiler_agent_v0_t lhs, rocprofiler_agent_v0_t rhs) {
        return lhs.node_id < rhs.node_id;
    });

    auto ofs = tool::output_file{"agent_info",
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

    for(auto& itr : data)
    {
        auto _type = std::string_view{};
        if(itr.type == ROCPROFILER_AGENT_TYPE_CPU)
            _type = "CPU";
        else if(itr.type == ROCPROFILER_AGENT_TYPE_GPU)
            _type = "GPU";
        else
            _type = "UNK";

        auto row_ss = std::stringstream{};
        rocprofiler::tool::csv::agent_info_csv_encoder::write_row(row_ss,
                                                                  itr.node_id,
                                                                  itr.logical_node_id,
                                                                  _type,
                                                                  itr.cpu_cores_count,
                                                                  itr.simd_count,
                                                                  itr.cpu_core_id_base,
                                                                  itr.simd_id_base,
                                                                  itr.max_waves_per_simd,
                                                                  itr.lds_size_in_kb,
                                                                  itr.gds_size_in_kb,
                                                                  itr.num_gws,
                                                                  itr.wave_front_size,
                                                                  itr.num_xcc,
                                                                  itr.cu_count,
                                                                  itr.array_count,
                                                                  itr.num_shader_banks,
                                                                  itr.simd_arrays_per_engine,
                                                                  itr.cu_per_simd_array,
                                                                  itr.simd_per_cu,
                                                                  itr.max_slots_scratch_cu,
                                                                  itr.gfx_target_version,
                                                                  itr.vendor_id,
                                                                  itr.device_id,
                                                                  itr.location_id,
                                                                  itr.domain,
                                                                  itr.drm_render_minor,
                                                                  itr.num_sdma_engines,
                                                                  itr.num_sdma_xgmi_engines,
                                                                  itr.num_sdma_queues_per_engine,
                                                                  itr.num_cp_queues,
                                                                  itr.max_engine_clk_ccompute,
                                                                  itr.max_engine_clk_fcompute,
                                                                  itr.sdma_fw_version.Value,
                                                                  itr.fw_version.Value,
                                                                  itr.capability.Value,
                                                                  itr.cu_per_engine,
                                                                  itr.max_waves_per_cu,
                                                                  itr.family_id,
                                                                  itr.workgroup_max_size,
                                                                  itr.grid_max_size,
                                                                  itr.local_mem_size,
                                                                  itr.hive_id,
                                                                  itr.gpu_id,
                                                                  itr.workgroup_max_dim.x,
                                                                  itr.workgroup_max_dim.y,
                                                                  itr.workgroup_max_dim.z,
                                                                  itr.grid_max_dim.x,
                                                                  itr.grid_max_dim.y,
                                                                  itr.grid_max_dim.z,
                                                                  itr.name,
                                                                  itr.vendor_name,
                                                                  itr.product_name,
                                                                  itr.model_name);
        ofs << row_ss.str();
    }
}

void
generate_csv(tool_table*                                                            tool_functions,
             const std::deque<rocprofiler_buffer_tracing_kernel_dispatch_record_t>& data,
             const stats_entry_t&                                                   stats)
{
    if(data.empty()) return;

    if(tool::get_config().stats && stats)
        write_stats(get_stats_output_file("kernel_stats"), stats.entries);

    auto ofs = tool::output_file{"kernel_trace",
                                 tool::csv::kernel_trace_csv_encoder{},
                                 {"Kind",
                                  "Agent_Id",
                                  "Queue_Id",
                                  "Thread_Id",
                                  "Dispatch_Id",
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

    for(const auto& record : data)
    {
        auto row_ss      = std::stringstream{};
        auto kernel_name = tool_functions->tool_get_kernel_name_fn(
            record.dispatch_info.kernel_id, record.correlation_id.external.value);
        rocprofiler::tool::csv::kernel_trace_csv_encoder::write_row(
            row_ss,
            tool_functions->tool_get_domain_name_fn(record.kind),
            tool_functions->tool_get_agent_node_id_fn(record.dispatch_info.agent_id),
            record.dispatch_info.queue_id.handle,
            record.thread_id,
            record.dispatch_info.dispatch_id,
            record.dispatch_info.kernel_id,
            kernel_name,
            record.correlation_id.internal,
            record.start_timestamp,
            record.end_timestamp,
            record.dispatch_info.private_segment_size,
            record.dispatch_info.group_segment_size,
            record.dispatch_info.workgroup_size.x,
            record.dispatch_info.workgroup_size.y,
            record.dispatch_info.workgroup_size.z,
            record.dispatch_info.grid_size.x,
            record.dispatch_info.grid_size.y,
            record.dispatch_info.grid_size.z);

        ofs << row_ss.str();
    }
}

void
generate_csv(tool_table*                                                    tool_functions,
             const std::deque<rocprofiler_buffer_tracing_hip_api_record_t>& data,
             const stats_entry_t&                                           stats)
{
    if(data.empty()) return;

    if(tool::get_config().stats && stats)
        write_stats(get_stats_output_file("hip_api_stats"), stats.entries);

    auto ofs = tool::output_file{"hip_api_trace",
                                 tool::csv::api_csv_encoder{},
                                 {"Domain",
                                  "Function",
                                  "Process_Id",
                                  "Thread_Id",
                                  "Correlation_Id",
                                  "Start_Timestamp",
                                  "End_Timestamp"}};
    for(const auto& record : data)
    {
        auto row_ss   = std::stringstream{};
        auto api_name = tool_functions->tool_get_operation_name_fn(record.kind, record.operation);
        rocprofiler::tool::csv::api_csv_encoder::write_row(
            row_ss,
            tool_functions->tool_get_domain_name_fn(record.kind),
            api_name,
            getpid(),
            record.thread_id,
            record.correlation_id.internal,
            record.start_timestamp,
            record.end_timestamp);

        ofs << row_ss.str();
    }
}

void
generate_csv(tool_table*                                                    tool_functions,
             const std::deque<rocprofiler_buffer_tracing_hsa_api_record_t>& data,
             const stats_entry_t&                                           stats)
{
    if(data.empty()) return;

    if(tool::get_config().stats && stats)
        write_stats(get_stats_output_file("hsa_api_stats"), stats.entries);

    auto ofs = tool::output_file{"hsa_api_trace",
                                 tool::csv::api_csv_encoder{},
                                 {"Domain",
                                  "Function",
                                  "Process_Id",
                                  "Thread_Id",
                                  "Correlation_Id",
                                  "Start_Timestamp",
                                  "End_Timestamp"}};

    for(const auto& record : data)
    {
        auto row_ss   = std::stringstream{};
        auto api_name = tool_functions->tool_get_operation_name_fn(record.kind, record.operation);
        rocprofiler::tool::csv::api_csv_encoder::write_row(
            row_ss,
            tool_functions->tool_get_domain_name_fn(record.kind),
            api_name,
            getpid(),
            record.thread_id,
            record.correlation_id.internal,
            record.start_timestamp,
            record.end_timestamp);

        ofs << row_ss.str();
    }
}

void
generate_csv(tool_table*                                                        tool_functions,
             const std::deque<rocprofiler_buffer_tracing_memory_copy_record_t>& data,
             const stats_entry_t&                                               stats)
{
    if(data.empty()) return;

    if(tool::get_config().stats && stats)
        write_stats(get_stats_output_file("memory_copy_stats"), stats.entries);

    auto ofs = tool::output_file{"memory_copy_trace",
                                 tool::csv::memory_copy_csv_encoder{},
                                 {"Kind",
                                  "Direction",
                                  "Source_Agent_Id",
                                  "Destination_Agent_Id",
                                  "Correlation_Id",
                                  "Start_Timestamp",
                                  "End_Timestamp"}};
    for(const auto& record : data)
    {
        auto row_ss   = std::stringstream{};
        auto api_name = tool_functions->tool_get_operation_name_fn(record.kind, record.operation);
        rocprofiler::tool::csv::memory_copy_csv_encoder::write_row(
            row_ss,
            tool_functions->tool_get_domain_name_fn(record.kind),
            api_name,
            tool_functions->tool_get_agent_node_id_fn(record.src_agent_id),
            tool_functions->tool_get_agent_node_id_fn(record.dst_agent_id),
            record.correlation_id.internal,
            record.start_timestamp,
            record.end_timestamp);

        ofs << row_ss.str();
    }
}

void
generate_csv(tool_table*                                                       tool_functions,
             const std::deque<rocprofiler_buffer_tracing_marker_api_record_t>& data,
             const stats_entry_t&                                              stats)
{
    if(data.empty()) return;

    if(tool::get_config().stats && stats)
        write_stats(get_stats_output_file("marker_api_stats"), stats.entries);

    auto ofs = tool::output_file{"marker_api_trace",
                                 tool::csv::marker_csv_encoder{},
                                 {"Domain",
                                  "Function",
                                  "Process_Id",
                                  "Thread_Id",
                                  "Correlation_Id",
                                  "Start_Timestamp",
                                  "End_Timestamp"}};
    for(const auto& record : data)
    {
        auto row_ss = std::stringstream{};
        auto _name  = std::string_view{};

        if(record.kind == ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API &&
           (record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA ||
            record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA ||
            record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA))
        {
            _name = tool_functions->tool_get_roctx_msg_fn(record.correlation_id.internal);
        }
        else
        {
            _name = tool_functions->tool_get_operation_name_fn(record.kind, record.operation);
        }

        tool::csv::marker_csv_encoder::write_row(
            row_ss,
            tool_functions->tool_get_domain_name_fn(record.kind),
            _name,
            getpid(),
            record.thread_id,
            record.correlation_id.internal,
            record.start_timestamp,
            record.end_timestamp);

        ofs << row_ss.str();
    }
}

void
generate_csv(tool_table*                                                     tool_functions,
             const std::deque<rocprofiler_tool_counter_collection_record_t>& data,
             const stats_entry_t&                                            stats)
{
    if(data.empty()) return;

    if(tool::get_config().stats && stats)
        write_stats(get_stats_output_file("counter_collection_stats"), stats.entries);

    auto ofs = tool::output_file{"counter_collection",
                                 tool::csv::counter_collection_csv_encoder{},
                                 {"Correlation_Id",
                                  "Dispatch_Id",
                                  "Agent_Id",
                                  "Queue_Id",
                                  "Process_Id",
                                  "Thread_Id",
                                  "Grid_Size",
                                  "Kernel_Id",
                                  "Kernel_Name",
                                  "Workgroup_Size",
                                  "LDS_Block_Size",
                                  "Scratch_Size",
                                  "VGPR_Count",
                                  "SGPR_Count",
                                  "Counter_Name",
                                  "Counter_Value",
                                  "Start_Timestamp",
                                  "End_Timestamp"}};
    for(const auto& record : data)
    {
        auto kernel_id          = record.dispatch_data.dispatch_info.kernel_id;
        auto counter_name_value = std::map<std::string, double>{};
        for(uint64_t i = 0; i < record.counter_count; i++)
        {
            const auto& count        = record.records.at(i);
            auto        rec          = count.record_counter;
            std::string counter_name = tool_functions->tool_get_counter_info_name_fn(rec.id);
            auto        search       = counter_name_value.find(counter_name);
            if(search == counter_name_value.end())
                counter_name_value.emplace(
                    std::pair<std::string, double>{counter_name, rec.counter_value});
            else
                search->second = search->second + rec.counter_value;
        }

        const auto& correlation_id = record.dispatch_data.correlation_id;

        auto magnitude = [](rocprofiler_dim3_t dims) { return (dims.x * dims.y * dims.z); };
        auto row_ss    = std::stringstream{};
        for(auto& itr : counter_name_value)
        {
            tool::csv::counter_collection_csv_encoder::write_row(
                row_ss,
                correlation_id.internal,
                record.dispatch_data.dispatch_info.dispatch_id,
                tool_functions->tool_get_agent_node_id_fn(
                    record.dispatch_data.dispatch_info.agent_id),
                record.dispatch_data.dispatch_info.queue_id.handle,
                getpid(),
                record.thread_id,
                magnitude(record.dispatch_data.dispatch_info.grid_size),
                record.dispatch_data.dispatch_info.kernel_id,
                tool_functions->tool_get_kernel_name_fn(kernel_id, correlation_id.external.value),
                magnitude(record.dispatch_data.dispatch_info.workgroup_size),
                record.lds_block_size_v,
                record.dispatch_data.dispatch_info.private_segment_size,
                record.arch_vgpr_count,
                record.sgpr_count,
                itr.first,
                itr.second,
                record.dispatch_data.start_timestamp,
                record.dispatch_data.end_timestamp);
        }
        ofs << row_ss.str();
    }
}

void
generate_csv(tool_table*                                                           tool_functions,
             const std::deque<rocprofiler_buffer_tracing_scratch_memory_record_t>& data,
             const stats_entry_t&                                                  stats)
{
    if(data.empty()) return;

    if(tool::get_config().stats && stats)
        write_stats(get_stats_output_file("scratch_memory_stats"), stats.entries);

    auto ofs = tool::output_file{"scratch_memory_trace",
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

    for(const auto& record : data)
    {
        auto row_ss    = std::stringstream{};
        auto kind_name = tool_functions->tool_get_domain_name_fn(record.kind);
        auto op_name   = tool_functions->tool_get_operation_name_fn(record.kind, record.operation);

        tool::csv::scratch_memory_encoder::write_row(
            row_ss,
            kind_name,
            op_name,
            tool_functions->tool_get_agent_node_id_fn(record.agent_id),
            record.queue_id.handle,
            record.thread_id,
            record.flags,
            record.start_timestamp,
            record.end_timestamp);

        ofs << row_ss.str();
    }
}

void
generate_csv(tool_table*                                                     tool_functions,
             const std::deque<rocprofiler_buffer_tracing_rccl_api_record_t>& data,
             const stats_entry_t&                                            stats)
{
    if(data.empty()) return;

    if(tool::get_config().stats && stats)
        write_stats(get_stats_output_file("rccl_api_stats"), stats.entries);

    auto ofs = tool::output_file{"rccl_api_trace",
                                 tool::csv::api_csv_encoder{},
                                 {"Domain",
                                  "Function",
                                  "Process_Id",
                                  "Thread_Id",
                                  "Correlation_Id",
                                  "Start_Timestamp",
                                  "End_Timestamp"}};
    for(const auto& record : data)
    {
        auto row_ss   = std::stringstream{};
        auto api_name = tool_functions->tool_get_operation_name_fn(record.kind, record.operation);
        rocprofiler::tool::csv::api_csv_encoder::write_row(
            row_ss,
            tool_functions->tool_get_domain_name_fn(record.kind),
            api_name,
            getpid(),
            record.thread_id,
            record.correlation_id.internal,
            record.start_timestamp,
            record.end_timestamp);

        ofs << row_ss.str();
    }
}

void
generate_csv(tool_table* /*tool_functions*/, const domain_stats_vec_t& data_v)
{
    using csv_encoder_t = rocprofiler::tool::csv::stats_csv_encoder;

    if(!tool::get_config().stats) return;

    auto _data        = data_v;
    auto _total_stats = stats_data_t{};
    for(const auto& itr : _data)
        _total_stats += itr.second.total;

    if(_total_stats.get_count() == 0) return;

    std::sort(_data.begin(), _data.end(), [](const auto& lhs, const auto& rhs) {
        return (lhs.second.total.get_sum() > rhs.second.total.get_sum());
    });

    auto ofs = get_stats_output_file("domain_stats");

    const float_type _total_duration = _total_stats.get_sum();
    for(const auto& [type, value] : _data)
    {
        auto name        = get_domain_column_name(type);
        auto duration_ns = value.total.get_sum();
        auto calls       = value.total.get_count();
        auto avg_ns      = value.total.get_mean();
        auto percent_v   = value.total.get_percent(_total_duration);

        auto _row = std::stringstream{};
        csv_encoder_t::write_row<stats_formatter>(_row,
                                                  name,
                                                  calls,
                                                  duration_ns,
                                                  avg_ns,
                                                  percentage{percent_v},
                                                  value.total.get_min(),
                                                  value.total.get_max(),
                                                  value.total.get_stddev());
        ofs << _row.str() << std::flush;
    }
}
}  // namespace tool
}  // namespace rocprofiler
