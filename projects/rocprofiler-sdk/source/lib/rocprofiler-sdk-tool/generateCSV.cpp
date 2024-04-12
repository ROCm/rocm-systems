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
#include "csv.hpp"
#include "helper.hpp"
#include "statistics.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>

#include <utility>

namespace rocprofiler
{
namespace tool
{
namespace
{
using float_type   = double;
using stats_data_t = statistics<uint64_t, float_type>;
using stats_map_t  = std::map<std::string_view, stats_data_t>;

void
write_stats(output_file& ofs, timestamps_t* app_timestamps, const stats_map_t& data)
{
    auto       _ss          = std::stringstream{};
    float_type app_duration = (app_timestamps->app_end_time - app_timestamps->app_start_time);
    for(const auto& [id, value] : data)
    {
        auto        duration_ns = value.get_sum();
        auto        calls       = value.get_count();
        const auto& name        = id;
        float_type  avg_ns      = value.get_mean();
        float_type  percentage  = (duration_ns / app_duration) * static_cast<float_type>(100);

        rocprofiler::tool::csv::stats_csv_encoder::write_row(_ss,
                                                             name,
                                                             calls,
                                                             duration_ns,
                                                             avg_ns,
                                                             percentage,
                                                             value.get_min(),
                                                             value.get_max(),
                                                             value.get_stddev());
    }
    ofs << _ss.str();
}
}  // namespace

void
generate_csv(tool_table* tool_functions, std::vector<rocprofiler_agent_v0_t>& data)
{
    std::sort(data.begin(), data.end(), [](rocprofiler_agent_v0_t lhs, rocprofiler_agent_v0_t rhs) {
        return lhs.node_id < rhs.node_id;
    });

    for(auto& itr : data)
    {
        auto _type = std::string_view{};
        if(itr.type == ROCPROFILER_AGENT_TYPE_CPU)
            _type = "CPU";
        else if(itr.type == ROCPROFILER_AGENT_TYPE_GPU)
            _type = "GPU";
        else
            _type = "UNK";

        auto agent_info_ss = std::stringstream{};
        rocprofiler::tool::csv::agent_info_csv_encoder::write_row(agent_info_ss,
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
        tool_functions->tool_get_agent_info_file_fn() << agent_info_ss.str();
    }
}

void
generate_csv(tool_table* tool_functions, std::vector<kernel_dispatch_ring_buffer_t>& data)
{
    auto kernel_stats = stats_map_t{};
    for(auto& buf : data)
    {
        while(true)
        {
            auto kernel_trace_ss                                        = std::stringstream{};
            rocprofiler_buffer_tracing_kernel_dispatch_record_t* record = buf.retrieve();
            if(record == nullptr) break;
            auto kernel_name =
                tool_functions->tool_get_kernel_name_fn(record->dispatch_info.kernel_id);
            rocprofiler::tool::csv::kernel_trace_csv_encoder::write_row(
                kernel_trace_ss,
                tool_functions->tool_get_domain_name_fn(record->kind),
                tool_functions->tool_get_agent_node_id_fn(record->dispatch_info.agent_id),
                record->dispatch_info.queue_id.handle,
                record->dispatch_info.kernel_id,
                kernel_name,
                record->correlation_id.internal,
                record->start_timestamp,
                record->end_timestamp,
                record->dispatch_info.private_segment_size,
                record->dispatch_info.group_segment_size,
                record->dispatch_info.workgroup_size.x,
                record->dispatch_info.workgroup_size.y,
                record->dispatch_info.workgroup_size.z,
                record->dispatch_info.grid_size.x,
                record->dispatch_info.grid_size.y,
                record->dispatch_info.grid_size.z);

            if(tool::get_config().stats)
                kernel_stats[kernel_name] += (record->end_timestamp - record->start_timestamp);

            tool_functions->tool_get_kernel_trace_file_fn() << kernel_trace_ss.str();
        }
    }

    if(tool::get_config().stats)
        write_stats(tool_functions->tool_get_kernel_stats_file_fn(),
                    tool_functions->tool_get_app_timestamps_fn(),
                    kernel_stats);
}

void
generate_csv(tool_table* tool_functions, std::vector<hip_ring_buffer_t>& data)
{
    auto hip_stats = stats_map_t{};
    for(auto& buf : data)
    {
        while(true)
        {
            auto                                         hip_trace_ss = std::stringstream{};
            rocprofiler_buffer_tracing_hip_api_record_t* record       = buf.retrieve();
            if(record == nullptr) break;
            auto api_name =
                tool_functions->tool_get_operation_name_fn(record->kind, record->operation);
            rocprofiler::tool::csv::api_csv_encoder::write_row(
                hip_trace_ss,
                tool_functions->tool_get_domain_name_fn(record->kind),
                api_name,
                getpid(),
                record->thread_id,
                record->correlation_id.internal,
                record->start_timestamp,
                record->end_timestamp);

            if(tool::get_config().stats)
                hip_stats[api_name] += (record->end_timestamp - record->start_timestamp);

            tool_functions->tool_get_hip_api_trace_file_fn() << hip_trace_ss.str();
        }
    }

    if(tool::get_config().stats)
    {
        write_stats(tool_functions->tool_get_hip_stats_file_fn(),
                    tool_functions->tool_get_app_timestamps_fn(),
                    hip_stats);
    }
}

void
generate_csv(tool_table* tool_functions, std::vector<hsa_ring_buffer_t>& data)
{
    auto hsa_stats = stats_map_t{};
    for(auto& buf : data)
    {
        while(true)
        {
            auto                                         hsa_trace_ss = std::stringstream{};
            rocprofiler_buffer_tracing_hsa_api_record_t* record       = buf.retrieve();
            if(record == nullptr) break;
            auto api_name =
                tool_functions->tool_get_operation_name_fn(record->kind, record->operation);
            rocprofiler::tool::csv::api_csv_encoder::write_row(
                hsa_trace_ss,
                tool_functions->tool_get_domain_name_fn(record->kind),
                api_name,
                getpid(),
                record->thread_id,
                record->correlation_id.internal,
                record->start_timestamp,
                record->end_timestamp);

            if(tool::get_config().stats)
                hsa_stats[api_name] += (record->end_timestamp - record->start_timestamp);

            tool_functions->tool_get_hsa_api_trace_file_fn() << hsa_trace_ss.str();
        }
    }

    if(tool::get_config().stats)
    {
        write_stats(tool_functions->tool_get_hsa_stats_file_fn(),
                    tool_functions->tool_get_app_timestamps_fn(),
                    hsa_stats);
    }
}

void
generate_csv(tool_table* tool_functions, std::vector<memory_copy_ring_buffer_t>& data)
{
    auto memory_copy_stats = stats_map_t{};
    for(auto& buf : data)
    {
        while(true)
        {
            auto memory_copy_trace_ss                               = std::stringstream{};
            rocprofiler_buffer_tracing_memory_copy_record_t* record = buf.retrieve();
            if(record == nullptr) break;
            auto api_name =
                tool_functions->tool_get_operation_name_fn(record->kind, record->operation);
            rocprofiler::tool::csv::memory_copy_csv_encoder::write_row(
                memory_copy_trace_ss,
                tool_functions->tool_get_domain_name_fn(record->kind),
                api_name,
                tool_functions->tool_get_agent_node_id_fn(record->src_agent_id),
                tool_functions->tool_get_agent_node_id_fn(record->dst_agent_id),
                record->correlation_id.internal,
                record->start_timestamp,
                record->end_timestamp);

            if(tool::get_config().stats)
                memory_copy_stats[api_name] += (record->end_timestamp - record->start_timestamp);

            tool_functions->tool_get_memory_copy_trace_file_fn() << memory_copy_trace_ss.str();
        }
    }

    if(tool::get_config().stats)
    {
        write_stats(tool_functions->tool_get_memory_copy_stats_file_fn(),
                    tool_functions->tool_get_app_timestamps_fn(),
                    memory_copy_stats);
    }
}

void
generate_csv(tool_table* tool_functions, std::vector<marker_api_ring_buffer_t>& data)
{
    for(auto& buf : data)
    {
        while(true)
        {
            auto                              marker_api_trace_ss = std::stringstream{};
            rocprofiler_tool_marker_record_t* record              = buf.retrieve();
            if(record == nullptr) break;
            if(record->kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API &&
               ((record->op == ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA &&
                 record->phase == ROCPROFILER_CALLBACK_PHASE_EXIT) ||
                (record->op == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop &&
                 record->phase == ROCPROFILER_CALLBACK_PHASE_ENTER) ||
                (record->op == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop &&
                 record->phase == ROCPROFILER_CALLBACK_PHASE_ENTER)))
            {
                tool::csv::marker_csv_encoder::write_row(
                    marker_api_trace_ss,
                    tool_functions->tool_get_callback_kind_fn(record->kind),
                    tool_functions->tool_get_roctx_msg_fn(record->cid),
                    record->pid,
                    record->tid,
                    record->cid,
                    record->start_timestamp,
                    record->end_timestamp);
            }
            else
            {
                tool::csv::marker_csv_encoder::write_row(
                    marker_api_trace_ss,
                    tool_functions->tool_get_callback_kind_fn(record->kind),
                    tool_functions->tool_get_callback_op_name_fn(record->kind, record->op),
                    record->pid,
                    record->tid,
                    record->cid,
                    record->start_timestamp,
                    record->end_timestamp);
            }
            tool_functions->tool_get_marker_api_trace_file_fn() << marker_api_trace_ss.str();
        }
    }
}

void
generate_csv(tool_table* tool_functions, std::vector<counter_collection_ring_buffer_t>& data)
{
    for(auto& buf : data)
    {
        while(true)
        {
            rocprofiler_tool_counter_collection_record_t* record = buf.retrieve();
            if(record == nullptr) break;
            auto kernel_id          = record->dispatch_data.dispatch_info.kernel_id;
            auto counter_name_value = std::map<std::string, uint64_t>{};
            for(const auto& count : record->profiler_record)
            {
                auto        rec          = static_cast<rocprofiler_record_counter_t>(count);
                std::string counter_name = tool_functions->tool_get_counter_info_name_fn(rec.id);
                auto        search       = counter_name_value.find(counter_name);
                if(search == counter_name_value.end())
                    counter_name_value.emplace(
                        std::pair<std::string, uint64_t>{counter_name, rec.counter_value});
                else
                    search->second = search->second + rec.counter_value;
            }

            const auto& correlation_id = record->dispatch_data.correlation_id;

            auto magnitude = [](rocprofiler_dim3_t dims) { return (dims.x * dims.y * dims.z); };
            auto counter_collection_ss = std::stringstream{};
            for(auto& itr : counter_name_value)
            {
                tool::csv::counter_collection_csv_encoder::write_row(
                    counter_collection_ss,
                    correlation_id.internal,
                    record->dispatch_index,
                    tool_functions->tool_get_agent_node_id_fn(
                        record->dispatch_data.dispatch_info.agent_id),
                    record->dispatch_data.dispatch_info.queue_id.handle,
                    record->pid,
                    record->thread_id,
                    magnitude(record->dispatch_data.dispatch_info.grid_size),
                    tool_functions->tool_get_kernel_name_fn(kernel_id),
                    magnitude(record->dispatch_data.dispatch_info.workgroup_size),
                    record->lds_block_size_v,
                    record->private_segment_size,
                    record->arch_vgpr_count,
                    record->sgpr_count,
                    itr.first,
                    itr.second);
            }
            tool_functions->tool_get_counter_collection_file_fn() << counter_collection_ss.str();
        }
    }
}

void
generate_csv(tool_table* tool_functions, std::vector<scratch_memory_ring_buffer_t>& data)
{
    auto* ofs = tool_functions->tool_get_scratch_memory_file_fn();
    if(!ofs) throw std::runtime_error{"error creating scratch memory output file"};

    auto scratch_memory_stats = stats_map_t{};
    for(auto& buf : data)
    {
        while(true)
        {
            rocprofiler_buffer_tracing_scratch_memory_record_t* record = buf.retrieve();
            if(record == nullptr) break;

            auto scratch_memory_trace = std::stringstream{};
            auto kind_name            = tool_functions->tool_get_domain_name_fn(record->kind);
            auto op_name =
                tool_functions->tool_get_operation_name_fn(record->kind, record->operation);

            tool::csv::scratch_memory_encoder::write_row(
                scratch_memory_trace,
                kind_name,
                op_name,
                tool_functions->tool_get_agent_node_id_fn(record->agent_id),
                record->queue_id.handle,
                record->thread_id,
                record->flags,
                record->start_timestamp,
                record->end_timestamp);

            if(tool::get_config().stats)
                scratch_memory_stats[op_name] += (record->end_timestamp - record->start_timestamp);

            (*ofs) << scratch_memory_trace.str();
        }
    }

    if(tool::get_config().stats)
    {
        auto* stats_ofs = tool_functions->tool_get_scratch_memory_stats_file_fn();
        if(!stats_ofs) throw std::runtime_error{"error creating scratch memory stats output file"};
        write_stats(*stats_ofs, tool_functions->tool_get_app_timestamps_fn(), scratch_memory_stats);
    }
}
}  // namespace tool
}  // namespace rocprofiler
