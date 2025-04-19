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

#include "generateCSV.hpp"
#include "csv.hpp"
#include "csv_output_file.hpp"
#include "domain_type.hpp"
#include "generateStats.hpp"
#include "output_config.hpp"
#include "output_stream.hpp"
#include "statistics.hpp"
#include "timestamps.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/cxx/utility.hpp>

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
tool::csv_output_file
get_stats_output_file(const output_config& cfg, std::string_view name)
{
    return tool::csv_output_file{cfg,
                                 name,
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

tool::csv_output_file
get_stats_output_file(const output_config& cfg, domain_type domain)
{
    return get_stats_output_file(cfg, get_domain_stats_file_name(domain));
}

void
write_stats(tool::csv_output_file&& ofs, const stats_entry_vec_t& data_v)
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
generate_csv(const output_config& cfg,
             const metadata& /*tool_metadata*/,
             std::vector<agent_info>& data)
{
    if(data.empty()) return;

    std::sort(data.begin(), data.end(), [](const agent_info& lhs, const agent_info& rhs) {
        return lhs.node_id < rhs.node_id;
    });

    auto ofs = tool::csv_output_file{cfg,
                                     "agent_info",
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
generate_csv(const output_config&                                               cfg,
             const metadata&                                                    tool_metadata,
             const generator<tool_buffer_tracing_kernel_dispatch_ext_record_t>& data,
             const stats_entry_t&                                               stats)
{
    if(data.empty()) return;

    if(cfg.stats && stats)
        write_stats(get_stats_output_file(cfg, domain_type::KERNEL_DISPATCH), stats.entries);
    auto ofs = tool::csv_output_file{cfg,
                                     domain_type::KERNEL_DISPATCH,
                                     tool::csv::kernel_trace_with_stream_csv_encoder{},
                                     {"Kind",
                                      "Agent_Id",
                                      "Queue_Id",
                                      "Stream_Id",
                                      "Thread_Id",
                                      "Dispatch_Id",
                                      "Kernel_Id",
                                      "Kernel_Name",
                                      "Correlation_Id",
                                      "Start_Timestamp",
                                      "End_Timestamp",
                                      "LDS_Block_Size",
                                      "Scratch_Size",
                                      "VGPR_Count",
                                      "Accum_VGPR_Count",
                                      "SGPR_Count",
                                      "Workgroup_Size_X",
                                      "Workgroup_Size_Y",
                                      "Workgroup_Size_Z",
                                      "Grid_Size_X",
                                      "Grid_Size_Y",
                                      "Grid_Size_Z"}};

    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto        row_ss      = std::stringstream{};
            auto        kernel_name = tool_metadata.get_kernel_name(record.dispatch_info.kernel_id,
                                                             record.correlation_id.external.value);
            const auto* kernel_info =
                tool_metadata.get_kernel_symbol(record.dispatch_info.kernel_id);
            auto lds_block_size_v =
                (kernel_info->group_segment_size + (lds_block_size - 1)) & ~(lds_block_size - 1);

            rocprofiler::tool::csv::kernel_trace_with_stream_csv_encoder::write_row(
                row_ss,
                tool_metadata.get_kind_name(record.kind),
                tool_metadata.get_agent_index(record.dispatch_info.agent_id, cfg.agent_index_value)
                    .as_string(),
                record.dispatch_info.queue_id.handle,
                record.stream_id.handle,
                record.thread_id,
                record.dispatch_info.dispatch_id,
                record.dispatch_info.kernel_id,
                kernel_name,
                record.correlation_id.internal,
                record.start_timestamp,
                record.end_timestamp,
                lds_block_size_v,
                record.dispatch_info.private_segment_size,
                kernel_info->arch_vgpr_count,
                kernel_info->accum_vgpr_count,
                kernel_info->sgpr_count,
                record.dispatch_info.workgroup_size.x,
                record.dispatch_info.workgroup_size.y,
                record.dispatch_info.workgroup_size.z,
                record.dispatch_info.grid_size.x,
                record.dispatch_info.grid_size.y,
                record.dispatch_info.grid_size.z);
            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const output_config&                                       cfg,
             const metadata&                                            tool_metadata,
             const generator<tool_buffer_tracing_hip_api_ext_record_t>& data,
             const stats_entry_t&                                       stats)
{
    if(data.empty()) return;

    if(cfg.stats && stats) write_stats(get_stats_output_file(cfg, domain_type::HIP), stats.entries);

    auto ofs = tool::csv_output_file{cfg,
                                     domain_type::HIP,
                                     tool::csv::api_csv_encoder{},
                                     {"Domain",
                                      "Function",
                                      "Process_Id",
                                      "Thread_Id",
                                      "Correlation_Id",
                                      "Start_Timestamp",
                                      "End_Timestamp"}};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto row_ss   = std::stringstream{};
            auto api_name = tool_metadata.get_operation_name(record.kind, record.operation);
            rocprofiler::tool::csv::api_csv_encoder::write_row(
                row_ss,
                tool_metadata.get_kind_name(record.kind),
                api_name,
                tool_metadata.process_id,
                record.thread_id,
                record.correlation_id.internal,
                record.start_timestamp,
                record.end_timestamp);

            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const output_config&                                          cfg,
             const metadata&                                               tool_metadata,
             const generator<rocprofiler_buffer_tracing_hsa_api_record_t>& data,
             const stats_entry_t&                                          stats)
{
    if(data.empty()) return;

    if(cfg.stats && stats) write_stats(get_stats_output_file(cfg, domain_type::HSA), stats.entries);

    auto ofs = tool::csv_output_file{cfg,
                                     domain_type::HSA,
                                     tool::csv::api_csv_encoder{},
                                     {"Domain",
                                      "Function",
                                      "Process_Id",
                                      "Thread_Id",
                                      "Correlation_Id",
                                      "Start_Timestamp",
                                      "End_Timestamp"}};

    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto row_ss   = std::stringstream{};
            auto api_name = tool_metadata.get_operation_name(record.kind, record.operation);
            rocprofiler::tool::csv::api_csv_encoder::write_row(
                row_ss,
                tool_metadata.get_kind_name(record.kind),
                api_name,
                tool_metadata.process_id,
                record.thread_id,
                record.correlation_id.internal,
                record.start_timestamp,
                record.end_timestamp);

            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const output_config&                                           cfg,
             const metadata&                                                tool_metadata,
             const generator<tool_buffer_tracing_memory_copy_ext_record_t>& data,
             const stats_entry_t&                                           stats)
{
    if(data.empty()) return;

    if(cfg.stats && stats)
        write_stats(get_stats_output_file(cfg, domain_type::MEMORY_COPY), stats.entries);

    auto ofs = tool::csv_output_file{cfg,
                                     domain_type::MEMORY_COPY,
                                     tool::csv::memory_copy_with_stream_csv_encoder{},
                                     {"Kind",
                                      "Direction",
                                      "Stream_Id",
                                      "Source_Agent_Id",
                                      "Destination_Agent_Id",
                                      "Correlation_Id",
                                      "Start_Timestamp",
                                      "End_Timestamp"}};

    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto row_ss   = std::stringstream{};
            auto api_name = tool_metadata.get_operation_name(record.kind, record.operation);
            rocprofiler::tool::csv::memory_copy_with_stream_csv_encoder::write_row(
                row_ss,
                tool_metadata.get_kind_name(record.kind),
                api_name,
                record.stream_id.handle,
                tool_metadata.get_agent_index(record.src_agent_id, cfg.agent_index_value)
                    .as_string(),
                tool_metadata.get_agent_index(record.dst_agent_id, cfg.agent_index_value)
                    .as_string(),
                record.correlation_id.internal,
                record.start_timestamp,
                record.end_timestamp);
            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const output_config&                                                 cfg,
             const metadata&                                                      tool_metadata,
             const generator<tool_buffer_tracing_memory_allocation_ext_record_t>& data,
             const stats_entry_t&                                                 stats)
{
    if(data.empty()) return;

    if(cfg.stats && stats)
        write_stats(get_stats_output_file(cfg, domain_type::MEMORY_ALLOCATION), stats.entries);

    auto ofs = tool::csv_output_file{cfg,
                                     domain_type::MEMORY_ALLOCATION,
                                     tool::csv::memory_allocation_csv_encoder{},
                                     {"Kind",
                                      "Operation",
                                      "Agent_Id",
                                      "Address",
                                      "Correlation_Id",
                                      "Start_Timestamp",
                                      "End_Timestamp",
                                      "Allocation_Size"}};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto agent_info = std::string{};
            // Free functions currently do not track agent information. Only set it on allocation
            // operations, otherwise set it to 0 currently
            if(record.operation == ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE ||
               record.operation == ROCPROFILER_MEMORY_ALLOCATION_VMEM_ALLOCATE)
            {
                agent_info = tool_metadata.get_agent_index(record.agent_id, cfg.agent_index_value)
                                 .as_string();
            }
            auto api_name = tool_metadata.get_operation_name(record.kind, record.operation);
            auto row_ss   = std::stringstream{};

            rocprofiler::tool::csv::memory_allocation_csv_encoder::write_row(
                row_ss,
                tool_metadata.get_kind_name(record.kind),
                api_name,
                agent_info,
                rocprofiler::sdk::utility::as_hex(record.address.handle, 16),
                record.correlation_id.internal,
                record.start_timestamp,
                record.end_timestamp,
                record.allocation_size);

            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const output_config&                                             cfg,
             const metadata&                                                  tool_metadata,
             const generator<rocprofiler_buffer_tracing_marker_api_record_t>& data,
             const stats_entry_t&                                             stats)
{
    if(data.empty()) return;

    if(cfg.stats && stats)
        write_stats(get_stats_output_file(cfg, domain_type::MARKER), stats.entries);

    auto ofs = tool::csv_output_file{cfg,
                                     domain_type::MARKER,
                                     tool::csv::marker_csv_encoder{},
                                     {"Domain",
                                      "Function",
                                      "Process_Id",
                                      "Thread_Id",
                                      "Correlation_Id",
                                      "Start_Timestamp",
                                      "End_Timestamp"}};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto row_ss = std::stringstream{};
            auto _name  = std::string_view{};

            if(record.kind == ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API &&
               (record.operation == ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxMarkA ||
                record.operation == ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxThreadRangeA ||
                record.operation == ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxProcessRangeA))
            {
                _name = tool_metadata.get_marker_message(record.correlation_id.internal);
            }
            else
            {
                _name = tool_metadata.get_operation_name(record.kind, record.operation);
            }

            tool::csv::marker_csv_encoder::write_row(row_ss,
                                                     tool_metadata.get_kind_name(record.kind),
                                                     _name,
                                                     tool_metadata.process_id,
                                                     record.thread_id,
                                                     record.correlation_id.internal,
                                                     record.start_timestamp,
                                                     record.end_timestamp);

            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const output_config&                    cfg,
             const metadata&                         tool_metadata,
             const generator<tool_counter_record_t>& data,
             const stats_entry_t&                    stats)
{
    if(data.empty()) return;

    if(cfg.stats && stats)
        write_stats(get_stats_output_file(cfg, domain_type::COUNTER_COLLECTION), stats.entries);

    auto ofs = tool::csv_output_file{cfg,
                                     domain_type::COUNTER_COLLECTION,
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
                                      "Accum_VGPR_Count",
                                      "SGPR_Count",
                                      "Counter_Name",
                                      "Counter_Value",
                                      "Start_Timestamp",
                                      "End_Timestamp"}};

    auto counter_id_to_name = std::unordered_map<rocprofiler_counter_id_t, std::string_view>{};
    for(const auto& itr : tool_metadata.get_counter_info())
        counter_id_to_name.emplace(itr.id, itr.name);

    for(auto ditr : data)
    {
        for(const auto& record : data.get(ditr))
        {
            auto kernel_id        = record.dispatch_data.dispatch_info.kernel_id;
            auto counter_id_value = std::map<rocprofiler_counter_id_t, double>{};
            auto record_vector    = record.read();

            // Accumulate counters based on ID
            for(auto& count : record_vector)
            {
                counter_id_value[count.id] += count.value;
            }

            const auto& correlation_id = record.dispatch_data.correlation_id;
            const auto* kernel_info    = tool_metadata.get_kernel_symbol(kernel_id);
            auto        lds_block_size_v =
                (kernel_info->group_segment_size + (lds_block_size - 1)) & ~(lds_block_size - 1);

            auto magnitude = [](rocprofiler_dim3_t dims) { return (dims.x * dims.y * dims.z); };
            auto row_ss    = std::stringstream{};
            for(auto& [counter_id, counter_value] : counter_id_value)
            {
                tool::csv::counter_collection_csv_encoder::write_row(
                    row_ss,
                    correlation_id.internal,
                    record.dispatch_data.dispatch_info.dispatch_id,
                    tool_metadata
                        .get_agent_index(record.dispatch_data.dispatch_info.agent_id,
                                         cfg.agent_index_value)
                        .as_string(),
                    record.dispatch_data.dispatch_info.queue_id.handle,
                    tool_metadata.process_id,
                    record.thread_id,
                    magnitude(record.dispatch_data.dispatch_info.grid_size),
                    record.dispatch_data.dispatch_info.kernel_id,
                    tool_metadata.get_kernel_name(kernel_id, correlation_id.external.value),
                    magnitude(record.dispatch_data.dispatch_info.workgroup_size),
                    lds_block_size_v,
                    record.dispatch_data.dispatch_info.private_segment_size,
                    kernel_info->arch_vgpr_count,
                    kernel_info->accum_vgpr_count,
                    kernel_info->sgpr_count,
                    counter_id_to_name.at(counter_id),
                    counter_value,
                    record.dispatch_data.start_timestamp,
                    record.dispatch_data.end_timestamp);
            }
            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const output_config&                                                 cfg,
             const metadata&                                                      tool_metadata,
             const generator<rocprofiler_buffer_tracing_scratch_memory_record_t>& data,
             const stats_entry_t&                                                 stats)
{
    if(data.empty()) return;

    if(cfg.stats && stats)
        write_stats(get_stats_output_file(cfg, domain_type::SCRATCH_MEMORY), stats.entries);

    auto ofs = tool::csv_output_file{cfg,
                                     domain_type::SCRATCH_MEMORY,
                                     tool::csv::scratch_memory_encoder{},
                                     {
                                         "Kind",
                                         "Operation",
                                         "Agent_Id",
                                         "Queue_Id",
                                         "Thread_Id",
                                         "Alloc_Flags",
                                         "Start_Timestamp",
                                         "End_Timestamp",
                                         "Allocation_Size",
                                     }};

    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto row_ss    = std::stringstream{};
            auto kind_name = tool_metadata.get_kind_name(record.kind);
            auto op_name   = tool_metadata.get_operation_name(record.kind, record.operation);

            tool::csv::scratch_memory_encoder::write_row(
                row_ss,
                kind_name,
                op_name,
                tool_metadata.get_agent_index(record.agent_id, cfg.agent_index_value).as_string(),
                record.queue_id.handle,
                record.thread_id,
                record.flags,
                record.start_timestamp,
                record.end_timestamp,
                record.allocation_size);

            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const output_config&                                               cfg,
             const metadata&                                                    tool_metadata,
             const generator<rocprofiler_buffer_tracing_rccl_api_ext_record_t>& data,
             const stats_entry_t&                                               stats)
{
    if(data.empty()) return;

    if(cfg.stats && stats)
        write_stats(get_stats_output_file(cfg, domain_type::RCCL), stats.entries);

    auto ofs = tool::csv_output_file{cfg,
                                     domain_type::RCCL,
                                     tool::csv::api_csv_encoder{},
                                     {"Domain",
                                      "Function",
                                      "Process_Id",
                                      "Thread_Id",
                                      "Correlation_Id",
                                      "Start_Timestamp",
                                      "End_Timestamp"}};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto row_ss   = std::stringstream{};
            auto api_name = tool_metadata.get_operation_name(record.kind, record.operation);
            rocprofiler::tool::csv::api_csv_encoder::write_row(
                row_ss,
                tool_metadata.get_kind_name(record.kind),
                api_name,
                tool_metadata.process_id,
                record.thread_id,
                record.correlation_id.internal,
                record.start_timestamp,
                record.end_timestamp);

            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const output_config&                                                    cfg,
             const metadata&                                                         tool_metadata,
             const generator<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t>& data,
             const stats_entry_t&                                                    stats)
{
    if(data.empty()) return;

    if(cfg.stats && stats)
        write_stats(get_stats_output_file(cfg, domain_type::ROCDECODE), stats.entries);

    auto ofs = tool::csv_output_file{cfg,
                                     domain_type::ROCDECODE,
                                     tool::csv::api_csv_encoder{},
                                     {"Domain",
                                      "Function",
                                      "Process_Id",
                                      "Thread_Id",
                                      "Correlation_Id",
                                      "Start_Timestamp",
                                      "End_Timestamp"}};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto row_ss   = std::stringstream{};
            auto api_name = tool_metadata.get_operation_name(record.kind, record.operation);
            rocprofiler::tool::csv::api_csv_encoder::write_row(
                row_ss,
                tool_metadata.get_kind_name(record.kind),
                api_name,
                tool_metadata.process_id,
                record.thread_id,
                record.correlation_id.internal,
                record.start_timestamp,
                record.end_timestamp);

            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const output_config&                                              cfg,
             const metadata&                                                   tool_metadata,
             const generator<rocprofiler_buffer_tracing_rocjpeg_api_record_t>& data,
             const stats_entry_t&                                              stats)
{
    if(data.empty()) return;

    if(cfg.stats && stats)
        write_stats(get_stats_output_file(cfg, domain_type::ROCJPEG), stats.entries);

    auto ofs = tool::csv_output_file{cfg,
                                     domain_type::ROCJPEG,
                                     tool::csv::api_csv_encoder{},
                                     {"Domain",
                                      "Function",
                                      "Process_Id",
                                      "Thread_Id",
                                      "Correlation_Id",
                                      "Start_Timestamp",
                                      "End_Timestamp"}};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto row_ss   = std::stringstream{};
            auto api_name = tool_metadata.get_operation_name(record.kind, record.operation);
            rocprofiler::tool::csv::api_csv_encoder::write_row(
                row_ss,
                tool_metadata.get_kind_name(record.kind),
                api_name,
                tool_metadata.process_id,
                record.thread_id,
                record.correlation_id.internal,
                record.start_timestamp,
                record.end_timestamp);

            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const output_config&                                              cfg,
             const metadata&                                                   tool_metadata,
             const generator<rocprofiler_tool_pc_sampling_host_trap_record_t>& data,
             const stats_entry_t&                                              stats)
{
    if(data.empty()) return;

    if(cfg.stats && stats)
        write_stats(get_stats_output_file(cfg, domain_type::PC_SAMPLING_HOST_TRAP), stats.entries);

    auto ofs = tool::csv_output_file{cfg,
                                     domain_type::PC_SAMPLING_HOST_TRAP,
                                     tool::csv::pc_sampling_host_trap_csv_encoder{},
                                     {"Sample_Timestamp",
                                      "Exec_Mask",
                                      "Dispatch_Id",
                                      "Instruction",
                                      "Instruction_Comment",
                                      "Correlation_Id"}};
    for(auto ditr : data)
    {
        for(const auto& record : data.get(ditr))
        {
            if(record.inst_index == -1)
            {
                auto        row_ss = std::stringstream{};
                std::string inst_comment =
                    "Unrecognized code object id, physical virtual address of PC:" +
                    std::to_string(record.pc_sample_record.pc.code_object_offset);
                rocprofiler::tool::csv::pc_sampling_host_trap_csv_encoder::write_row(
                    row_ss,
                    record.pc_sample_record.timestamp,
                    record.pc_sample_record.exec_mask,
                    record.pc_sample_record.dispatch_id,
                    "",
                    inst_comment,
                    record.pc_sample_record.correlation_id.internal);

                ofs << row_ss.str();
            }
            else
            {
                auto row_ss = std::stringstream{};
                rocprofiler::tool::csv::pc_sampling_host_trap_csv_encoder::write_row(
                    row_ss,
                    record.pc_sample_record.timestamp,
                    record.pc_sample_record.exec_mask,
                    record.pc_sample_record.dispatch_id,
                    tool_metadata.get_instruction(record.inst_index),
                    tool_metadata.get_comment(record.inst_index),
                    record.pc_sample_record.correlation_id.internal);

                ofs << row_ss.str();
            }
        }
    }
}

void
generate_csv(const output_config&                                               cfg,
             const metadata&                                                    tool_metadata,
             const generator<rocprofiler_tool_pc_sampling_stochastic_record_t>& data,
             const stats_entry_t&                                               stats)
{
    if(data.empty()) return;

    if(cfg.stats && stats)
        write_stats(get_stats_output_file(cfg, domain_type::PC_SAMPLING_STOCHASTIC), stats.entries);

    auto ofs = tool::csv_output_file{cfg,
                                     domain_type::PC_SAMPLING_STOCHASTIC,
                                     tool::csv::pc_sampling_stochastic_csv_encoder{},
                                     {
                                         "Sample_Timestamp",
                                         "Exec_Mask",
                                         "Dispatch_Id",
                                         "Instruction",
                                         "Instruction_Comment",
                                         "Correlation_Id",
                                         "Wave_Issued_Instruction",
                                         "Instruction_Type",
                                         "Stall_Reason",
                                         "Wave_Count",
                                     }};
    for(auto ditr : data)
    {
        for(const auto& record : data.get(ditr))
        {
            std::string inst;
            std::string inst_comment;
            if(record.inst_index == -1)
            {
                // A sample originates from a blit kernel or self-modifying code,
                // so instruction cannot be decoded
                inst_comment = "Unrecognized code object id, physical virtual address of PC:" +
                               std::to_string(record.pc_sample_record.pc.code_object_offset);
            }
            else
            {
                // Provide decoded instruction and comment
                inst         = tool_metadata.get_instruction(record.inst_index);
                inst_comment = tool_metadata.get_comment(record.inst_index);
            }

            auto row_ss = std::stringstream{};
            rocprofiler::tool::csv::pc_sampling_stochastic_csv_encoder::write_row(
                row_ss,
                record.pc_sample_record.timestamp,
                record.pc_sample_record.exec_mask,
                record.pc_sample_record.dispatch_id,
                inst,
                inst_comment,
                record.pc_sample_record.correlation_id.internal,
                // As wave_issued is uint8_t of size 1, it can be dumped as char.
                // To prevent that, explicitly cast it to integer, so that CSV output
                // shows human-readable 0/1 values.
                static_cast<unsigned int>(record.pc_sample_record.wave_issued),
                std::string(rocprofiler_get_pc_sampling_instruction_type_name(
                    static_cast<rocprofiler_pc_sampling_instruction_type_t>(
                        record.pc_sample_record.inst_type))),
                std::string(rocprofiler_get_pc_sampling_instruction_not_issued_reason_name(
                    static_cast<rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
                        record.pc_sample_record.snapshot.reason_not_issued))),
                // Similar reasoning as for wave_issued.
                static_cast<unsigned int>(record.pc_sample_record.wave_count));

            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const output_config& cfg,
             const metadata& /*tool_metadata*/,
             const domain_stats_vec_t& data_v)
{
    using csv_encoder_t = rocprofiler::tool::csv::stats_csv_encoder;

    if(!cfg.stats) return;

    auto _data        = data_v;
    auto _total_stats = stats_data_t{};
    for(const auto& itr : _data)
        _total_stats += itr.second.total;

    if(_total_stats.get_count() == 0) return;

    std::sort(_data.begin(), _data.end(), [](const auto& lhs, const auto& rhs) {
        return (lhs.second.total.get_sum() > rhs.second.total.get_sum());
    });

    auto ofs = get_stats_output_file(cfg, "domain_stats");

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
