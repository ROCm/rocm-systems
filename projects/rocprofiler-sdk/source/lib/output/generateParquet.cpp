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

#include "generateParquet.hpp"
#include "counter_info.hpp"
#include "domain_type.hpp"
#include "output_stream.hpp"

#include "lib/common/logging.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <map>
#include <string>
#include <unordered_map>

namespace rocprofiler
{
namespace tool
{
namespace
{
#define PARQUET_CHECK_OK(expr)                                                                     \
    do                                                                                             \
    {                                                                                              \
        auto _status = (expr);                                                                     \
        if(!_status.ok())                                                                          \
        {                                                                                          \
            ROCP_FATAL << "Arrow/Parquet error: " << _status.ToString();                           \
        }                                                                                          \
    } while(false)

std::string
get_parquet_filename(const output_config& cfg, std::string_view name)
{
    return get_output_filename(cfg, name, ".parquet");
}

void
write_table_as_parquet(const std::string&                   filename,
                       const std::shared_ptr<arrow::Table>& table)
{
    auto outfile_result = arrow::io::FileOutputStream::Open(filename);
    PARQUET_CHECK_OK(outfile_result.status());
    auto outfile = outfile_result.ValueUnsafe();

    auto writer_props = parquet::WriterProperties::Builder()
                            .compression(arrow::Compression::SNAPPY)
                            ->build();

    auto arrow_props = parquet::ArrowWriterProperties::Builder().store_schema()->build();

    PARQUET_CHECK_OK(
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile,
                                   table->num_rows(), writer_props, arrow_props));

    PARQUET_CHECK_OK(outfile->Close());

    ROCP_ERROR << "Wrote Parquet output: " << filename;
}

template <typename GeneratorT>
void
write_api_trace_parquet(const output_config& cfg,
                        const metadata&      tool_metadata,
                        std::string_view     domain_name,
                        const GeneratorT&    data)
{
    if(data.empty()) return;

    auto filename = get_parquet_filename(cfg, domain_name);

    auto schema = arrow::schema({
        arrow::field("Domain", arrow::utf8()),
        arrow::field("Function", arrow::utf8()),
        arrow::field("Process_Id", arrow::uint32()),
        arrow::field("Thread_Id", arrow::uint64()),
        arrow::field("Correlation_Id", arrow::uint64()),
        arrow::field("Start_Timestamp", arrow::uint64()),
        arrow::field("End_Timestamp", arrow::uint64()),
    });

    arrow::StringBuilder domain_builder;
    arrow::StringBuilder function_builder;
    arrow::UInt32Builder process_id_builder;
    arrow::UInt64Builder thread_id_builder;
    arrow::UInt64Builder correlation_id_builder;
    arrow::UInt64Builder start_ts_builder;
    arrow::UInt64Builder end_ts_builder;

    for(size_t idx = 0; idx < data.size(); ++idx)
    {
        auto records = data.get(idx);
        for(const auto& record : records)
        {
            PARQUET_CHECK_OK(
                domain_builder.Append(std::string(tool_metadata.get_kind_name(record.kind))));
            PARQUET_CHECK_OK(function_builder.Append(
                std::string(tool_metadata.get_operation_name(record.kind, record.operation))));
            PARQUET_CHECK_OK(process_id_builder.Append(tool_metadata.process_id));
            PARQUET_CHECK_OK(thread_id_builder.Append(record.thread_id));
            PARQUET_CHECK_OK(correlation_id_builder.Append(record.correlation_id.internal));
            PARQUET_CHECK_OK(start_ts_builder.Append(record.start_timestamp));
            PARQUET_CHECK_OK(end_ts_builder.Append(record.end_timestamp));
        }
    }

    auto arrays = arrow::ArrayVector{};
    auto arr    = std::shared_ptr<arrow::Array>{};

    auto finish = [&](auto& builder) {
        PARQUET_CHECK_OK(builder.Finish(&arr));
        arrays.push_back(arr);
    };

    finish(domain_builder);
    finish(function_builder);
    finish(process_id_builder);
    finish(thread_id_builder);
    finish(correlation_id_builder);
    finish(start_ts_builder);
    finish(end_ts_builder);

    auto table = arrow::Table::Make(schema, arrays);
    write_table_as_parquet(filename, table);
}

void
write_kernel_dispatch_parquet(
    const output_config&                                               cfg,
    const metadata&                                                    tool_metadata,
    const generator<tool_buffer_tracing_kernel_dispatch_ext_record_t>& data)
{
    if(data.empty()) return;

    auto filename =
        get_parquet_filename(cfg, get_domain_column_name(domain_type::KERNEL_DISPATCH));

    auto schema = arrow::schema({
        arrow::field("Kind", arrow::utf8()),
        arrow::field("Agent_Id", arrow::utf8()),
        arrow::field("Queue_Id", arrow::uint64()),
        arrow::field("Stream_Id", arrow::uint64()),
        arrow::field("Thread_Id", arrow::uint64()),
        arrow::field("Dispatch_Id", arrow::uint64()),
        arrow::field("Kernel_Id", arrow::uint64()),
        arrow::field("Kernel_Name", arrow::utf8()),
        arrow::field("Correlation_Id", arrow::uint64()),
        arrow::field("Start_Timestamp", arrow::uint64()),
        arrow::field("End_Timestamp", arrow::uint64()),
        arrow::field("LDS_Block_Size", arrow::uint64()),
        arrow::field("Scratch_Size", arrow::uint64()),
        arrow::field("VGPR_Count", arrow::uint32()),
        arrow::field("Accum_VGPR_Count", arrow::uint32()),
        arrow::field("SGPR_Count", arrow::uint32()),
        arrow::field("Workgroup_Size_X", arrow::uint32()),
        arrow::field("Workgroup_Size_Y", arrow::uint32()),
        arrow::field("Workgroup_Size_Z", arrow::uint32()),
        arrow::field("Grid_Size_X", arrow::uint32()),
        arrow::field("Grid_Size_Y", arrow::uint32()),
        arrow::field("Grid_Size_Z", arrow::uint32()),
    });

    arrow::StringBuilder kind_b, agent_id_b, kernel_name_b;
    arrow::UInt64Builder queue_id_b, stream_id_b, thread_id_b, dispatch_id_b, kernel_id_b,
        correlation_id_b, start_ts_b, end_ts_b, lds_b, scratch_b;
    arrow::UInt32Builder vgpr_b, accum_vgpr_b, sgpr_b, wg_x_b, wg_y_b, wg_z_b, gs_x_b, gs_y_b,
        gs_z_b;

    for(size_t idx = 0; idx < data.size(); ++idx)
    {
        auto records = data.get(idx);
        for(const auto& record : records)
        {
            const auto* kernel_info =
                tool_metadata.get_kernel_symbol(record.dispatch_info.kernel_id);
            auto kernel_name = tool_metadata.get_kernel_name(record.dispatch_info.kernel_id,
                                                             cfg.kernel_rename,
                                                             record.correlation_id.external.value);
            auto lds_block_size_v =
                (kernel_info->group_segment_size + (lds_block_size - 1)) & ~(lds_block_size - 1);

            PARQUET_CHECK_OK(
                kind_b.Append(std::string(tool_metadata.get_kind_name(record.kind))));
            PARQUET_CHECK_OK(agent_id_b.Append(
                tool_metadata.get_agent_index(record.dispatch_info.agent_id, cfg.agent_index_value)
                    .as_string()));
            PARQUET_CHECK_OK(queue_id_b.Append(record.dispatch_info.queue_id.handle));
            PARQUET_CHECK_OK(stream_id_b.Append(record.stream_id.handle));
            PARQUET_CHECK_OK(thread_id_b.Append(record.thread_id));
            PARQUET_CHECK_OK(dispatch_id_b.Append(record.dispatch_info.dispatch_id));
            PARQUET_CHECK_OK(kernel_id_b.Append(record.dispatch_info.kernel_id));
            PARQUET_CHECK_OK(kernel_name_b.Append(kernel_name));
            PARQUET_CHECK_OK(correlation_id_b.Append(record.correlation_id.internal));
            PARQUET_CHECK_OK(start_ts_b.Append(record.start_timestamp));
            PARQUET_CHECK_OK(end_ts_b.Append(record.end_timestamp));
            PARQUET_CHECK_OK(lds_b.Append(lds_block_size_v));
            PARQUET_CHECK_OK(scratch_b.Append(record.dispatch_info.private_segment_size));
            PARQUET_CHECK_OK(vgpr_b.Append(kernel_info->arch_vgpr_count));
            PARQUET_CHECK_OK(accum_vgpr_b.Append(kernel_info->accum_vgpr_count));
            PARQUET_CHECK_OK(sgpr_b.Append(kernel_info->sgpr_count));
            PARQUET_CHECK_OK(wg_x_b.Append(record.dispatch_info.workgroup_size.x));
            PARQUET_CHECK_OK(wg_y_b.Append(record.dispatch_info.workgroup_size.y));
            PARQUET_CHECK_OK(wg_z_b.Append(record.dispatch_info.workgroup_size.z));
            PARQUET_CHECK_OK(gs_x_b.Append(record.dispatch_info.grid_size.x));
            PARQUET_CHECK_OK(gs_y_b.Append(record.dispatch_info.grid_size.y));
            PARQUET_CHECK_OK(gs_z_b.Append(record.dispatch_info.grid_size.z));
        }
    }

    auto arrays = arrow::ArrayVector{};
    auto arr    = std::shared_ptr<arrow::Array>{};

    auto finish = [&](auto& builder) {
        PARQUET_CHECK_OK(builder.Finish(&arr));
        arrays.push_back(arr);
    };

    finish(kind_b);
    finish(agent_id_b);
    finish(queue_id_b);
    finish(stream_id_b);
    finish(thread_id_b);
    finish(dispatch_id_b);
    finish(kernel_id_b);
    finish(kernel_name_b);
    finish(correlation_id_b);
    finish(start_ts_b);
    finish(end_ts_b);
    finish(lds_b);
    finish(scratch_b);
    finish(vgpr_b);
    finish(accum_vgpr_b);
    finish(sgpr_b);
    finish(wg_x_b);
    finish(wg_y_b);
    finish(wg_z_b);
    finish(gs_x_b);
    finish(gs_y_b);
    finish(gs_z_b);

    auto table = arrow::Table::Make(schema, arrays);
    write_table_as_parquet(filename, table);
}

void
write_counter_collection_parquet(const output_config&                    cfg,
                                 const metadata&                         tool_metadata,
                                 const generator<tool_counter_record_t>& data)
{
    if(data.empty()) return;

    auto filename =
        get_parquet_filename(cfg, get_domain_column_name(domain_type::COUNTER_COLLECTION));

    auto schema = arrow::schema({
        arrow::field("Correlation_Id", arrow::uint64()),
        arrow::field("Dispatch_Id", arrow::uint64()),
        arrow::field("Agent_Id", arrow::utf8()),
        arrow::field("Queue_Id", arrow::uint64()),
        arrow::field("Process_Id", arrow::uint32()),
        arrow::field("Thread_Id", arrow::uint64()),
        arrow::field("Grid_Size", arrow::uint64()),
        arrow::field("Kernel_Id", arrow::uint64()),
        arrow::field("Kernel_Name", arrow::utf8()),
        arrow::field("Workgroup_Size", arrow::uint64()),
        arrow::field("LDS_Block_Size", arrow::uint64()),
        arrow::field("Scratch_Size", arrow::uint64()),
        arrow::field("VGPR_Count", arrow::uint32()),
        arrow::field("Accum_VGPR_Count", arrow::uint32()),
        arrow::field("SGPR_Count", arrow::uint32()),
        arrow::field("Counter_Name", arrow::utf8()),
        arrow::field("Counter_Value", arrow::float64()),
        arrow::field("Start_Timestamp", arrow::uint64()),
        arrow::field("End_Timestamp", arrow::uint64()),
    });

    arrow::UInt64Builder corr_b, dispatch_b, queue_b, thread_b, grid_b, kernel_id_b, wg_b, lds_b,
        scratch_b, start_b, end_b;
    arrow::UInt32Builder pid_b, vgpr_b, accum_vgpr_b, sgpr_b;
    arrow::StringBuilder agent_b, kname_b, cname_b;
    arrow::DoubleBuilder cval_b;

    auto counter_id_to_name = std::unordered_map<rocprofiler_counter_id_t, std::string_view>{};
    for(const auto& itr : tool_metadata.get_counter_info())
        counter_id_to_name.emplace(itr.id, itr.name);

    auto magnitude = [](rocprofiler_dim3_t dims) -> uint64_t {
        return static_cast<uint64_t>(dims.x) * dims.y * dims.z;
    };

    for(size_t idx = 0; idx < data.size(); ++idx)
    {
        auto records = data.get(idx);
        for(const auto& record : records)
        {
            auto kernel_id        = record.dispatch_data.dispatch_info.kernel_id;
            auto counter_id_value = std::map<rocprofiler_counter_id_t, double>{};
            auto record_vector    = record.read();

            for(auto& count : record_vector)
                counter_id_value[count.id] += count.value;

            const auto& correlation_id = record.dispatch_data.correlation_id;
            const auto* kernel_info    = tool_metadata.get_kernel_symbol(kernel_id);
            auto        lds_block_size_v =
                (kernel_info->group_segment_size + (lds_block_size - 1)) & ~(lds_block_size - 1);

            for(auto& [counter_id, counter_value] : counter_id_value)
            {
                PARQUET_CHECK_OK(corr_b.Append(correlation_id.internal));
                PARQUET_CHECK_OK(
                    dispatch_b.Append(record.dispatch_data.dispatch_info.dispatch_id));
                PARQUET_CHECK_OK(agent_b.Append(
                    tool_metadata
                        .get_agent_index(record.dispatch_data.dispatch_info.agent_id,
                                         cfg.agent_index_value)
                        .as_string()));
                PARQUET_CHECK_OK(
                    queue_b.Append(record.dispatch_data.dispatch_info.queue_id.handle));
                PARQUET_CHECK_OK(pid_b.Append(tool_metadata.process_id));
                PARQUET_CHECK_OK(thread_b.Append(record.thread_id));
                PARQUET_CHECK_OK(
                    grid_b.Append(magnitude(record.dispatch_data.dispatch_info.grid_size)));
                PARQUET_CHECK_OK(
                    kernel_id_b.Append(record.dispatch_data.dispatch_info.kernel_id));
                PARQUET_CHECK_OK(kname_b.Append(tool_metadata.get_kernel_name(
                    kernel_id, cfg.kernel_rename, correlation_id.external.value)));
                PARQUET_CHECK_OK(
                    wg_b.Append(magnitude(record.dispatch_data.dispatch_info.workgroup_size)));
                PARQUET_CHECK_OK(lds_b.Append(lds_block_size_v));
                PARQUET_CHECK_OK(
                    scratch_b.Append(record.dispatch_data.dispatch_info.private_segment_size));
                PARQUET_CHECK_OK(vgpr_b.Append(kernel_info->arch_vgpr_count));
                PARQUET_CHECK_OK(accum_vgpr_b.Append(kernel_info->accum_vgpr_count));
                PARQUET_CHECK_OK(sgpr_b.Append(kernel_info->sgpr_count));
                PARQUET_CHECK_OK(
                    cname_b.Append(std::string(counter_id_to_name.at(counter_id))));
                PARQUET_CHECK_OK(cval_b.Append(counter_value));
                PARQUET_CHECK_OK(start_b.Append(record.dispatch_data.start_timestamp));
                PARQUET_CHECK_OK(end_b.Append(record.dispatch_data.end_timestamp));
            }
        }
    }

    auto arrays = arrow::ArrayVector{};
    auto arr    = std::shared_ptr<arrow::Array>{};

    auto finish = [&](auto& builder) {
        PARQUET_CHECK_OK(builder.Finish(&arr));
        arrays.push_back(arr);
    };

    finish(corr_b);
    finish(dispatch_b);
    finish(agent_b);
    finish(queue_b);
    finish(pid_b);
    finish(thread_b);
    finish(grid_b);
    finish(kernel_id_b);
    finish(kname_b);
    finish(wg_b);
    finish(lds_b);
    finish(scratch_b);
    finish(vgpr_b);
    finish(accum_vgpr_b);
    finish(sgpr_b);
    finish(cname_b);
    finish(cval_b);
    finish(start_b);
    finish(end_b);

    auto table = arrow::Table::Make(schema, arrays);
    write_table_as_parquet(filename, table);
}

void
write_memory_copy_parquet(
    const output_config&                                           cfg,
    const metadata&                                                tool_metadata,
    const generator<tool_buffer_tracing_memory_copy_ext_record_t>& data)
{
    if(data.empty()) return;

    auto filename = get_parquet_filename(cfg, get_domain_column_name(domain_type::MEMORY_COPY));

    auto schema = arrow::schema({
        arrow::field("Kind", arrow::utf8()),
        arrow::field("Direction", arrow::utf8()),
        arrow::field("Stream_Id", arrow::uint64()),
        arrow::field("Source_Agent_Id", arrow::utf8()),
        arrow::field("Destination_Agent_Id", arrow::utf8()),
        arrow::field("Correlation_Id", arrow::uint64()),
        arrow::field("Start_Timestamp", arrow::uint64()),
        arrow::field("End_Timestamp", arrow::uint64()),
    });

    arrow::StringBuilder kind_b, dir_b, src_b, dst_b;
    arrow::UInt64Builder stream_b, corr_b, start_b, end_b;

    for(size_t idx = 0; idx < data.size(); ++idx)
    {
        auto records = data.get(idx);
        for(const auto& record : records)
        {
            PARQUET_CHECK_OK(
                kind_b.Append(std::string(tool_metadata.get_kind_name(record.kind))));
            PARQUET_CHECK_OK(dir_b.Append(
                std::string(tool_metadata.get_operation_name(record.kind, record.operation))));
            PARQUET_CHECK_OK(stream_b.Append(record.stream_id.handle));
            PARQUET_CHECK_OK(src_b.Append(
                tool_metadata.get_agent_index(record.src_agent_id, cfg.agent_index_value)
                    .as_string()));
            PARQUET_CHECK_OK(dst_b.Append(
                tool_metadata.get_agent_index(record.dst_agent_id, cfg.agent_index_value)
                    .as_string()));
            PARQUET_CHECK_OK(corr_b.Append(record.correlation_id.internal));
            PARQUET_CHECK_OK(start_b.Append(record.start_timestamp));
            PARQUET_CHECK_OK(end_b.Append(record.end_timestamp));
        }
    }

    auto arrays = arrow::ArrayVector{};
    auto arr    = std::shared_ptr<arrow::Array>{};
    auto finish = [&](auto& builder) {
        PARQUET_CHECK_OK(builder.Finish(&arr));
        arrays.push_back(arr);
    };

    finish(kind_b);
    finish(dir_b);
    finish(stream_b);
    finish(src_b);
    finish(dst_b);
    finish(corr_b);
    finish(start_b);
    finish(end_b);

    auto table = arrow::Table::Make(schema, arrays);
    write_table_as_parquet(filename, table);
}

void
write_scratch_memory_parquet(
    const output_config&                                                 cfg,
    const metadata&                                                      tool_metadata,
    const generator<rocprofiler_buffer_tracing_scratch_memory_record_t>& data)
{
    if(data.empty()) return;

    auto filename = get_parquet_filename(cfg, get_domain_column_name(domain_type::SCRATCH_MEMORY));

    auto schema = arrow::schema({
        arrow::field("Kind", arrow::utf8()),
        arrow::field("Operation", arrow::utf8()),
        arrow::field("Agent_Id", arrow::utf8()),
        arrow::field("Queue_Id", arrow::uint64()),
        arrow::field("Thread_Id", arrow::uint64()),
        arrow::field("Alloc_Flags", arrow::uint64()),
        arrow::field("Start_Timestamp", arrow::uint64()),
        arrow::field("End_Timestamp", arrow::uint64()),
        arrow::field("Allocation_Size", arrow::uint64()),
    });

    arrow::StringBuilder kind_b, op_b, agent_b;
    arrow::UInt64Builder queue_b, thread_b, flags_b, start_b, end_b, alloc_b;

    for(size_t idx = 0; idx < data.size(); ++idx)
    {
        auto records = data.get(idx);
        for(const auto& record : records)
        {
            PARQUET_CHECK_OK(
                kind_b.Append(std::string(tool_metadata.get_kind_name(record.kind))));
            PARQUET_CHECK_OK(op_b.Append(
                std::string(tool_metadata.get_operation_name(record.kind, record.operation))));
            PARQUET_CHECK_OK(agent_b.Append(
                tool_metadata.get_agent_index(record.agent_id, cfg.agent_index_value).as_string()));
            PARQUET_CHECK_OK(queue_b.Append(record.queue_id.handle));
            PARQUET_CHECK_OK(thread_b.Append(record.thread_id));
            PARQUET_CHECK_OK(flags_b.Append(record.flags));
            PARQUET_CHECK_OK(start_b.Append(record.start_timestamp));
            PARQUET_CHECK_OK(end_b.Append(record.end_timestamp));
            PARQUET_CHECK_OK(alloc_b.Append(record.allocation_size));
        }
    }

    auto arrays = arrow::ArrayVector{};
    auto arr    = std::shared_ptr<arrow::Array>{};
    auto finish = [&](auto& builder) {
        PARQUET_CHECK_OK(builder.Finish(&arr));
        arrays.push_back(arr);
    };

    finish(kind_b);
    finish(op_b);
    finish(agent_b);
    finish(queue_b);
    finish(thread_b);
    finish(flags_b);
    finish(start_b);
    finish(end_b);
    finish(alloc_b);

    auto table = arrow::Table::Make(schema, arrays);
    write_table_as_parquet(filename, table);
}

void
write_marker_parquet(const output_config&                                             cfg,
                     const metadata&                                                  tool_metadata,
                     const generator<rocprofiler_buffer_tracing_marker_api_record_t>& data)
{
    if(data.empty()) return;

    auto filename = get_parquet_filename(cfg, get_domain_column_name(domain_type::MARKER));

    auto schema = arrow::schema({
        arrow::field("Domain", arrow::utf8()),
        arrow::field("Function", arrow::utf8()),
        arrow::field("Process_Id", arrow::uint32()),
        arrow::field("Thread_Id", arrow::uint64()),
        arrow::field("Correlation_Id", arrow::uint64()),
        arrow::field("Start_Timestamp", arrow::uint64()),
        arrow::field("End_Timestamp", arrow::uint64()),
    });

    arrow::StringBuilder domain_b, func_b;
    arrow::UInt32Builder pid_b;
    arrow::UInt64Builder tid_b, corr_b, start_b, end_b;

    for(size_t idx = 0; idx < data.size(); ++idx)
    {
        auto records = data.get(idx);
        for(const auto& record : records)
        {
            auto _name = std::string_view{};
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

            PARQUET_CHECK_OK(
                domain_b.Append(std::string(tool_metadata.get_kind_name(record.kind))));
            PARQUET_CHECK_OK(func_b.Append(std::string(_name)));
            PARQUET_CHECK_OK(pid_b.Append(tool_metadata.process_id));
            PARQUET_CHECK_OK(tid_b.Append(record.thread_id));
            PARQUET_CHECK_OK(corr_b.Append(record.correlation_id.internal));
            PARQUET_CHECK_OK(start_b.Append(record.start_timestamp));
            PARQUET_CHECK_OK(end_b.Append(record.end_timestamp));
        }
    }

    auto arrays = arrow::ArrayVector{};
    auto arr    = std::shared_ptr<arrow::Array>{};
    auto finish = [&](auto& builder) {
        PARQUET_CHECK_OK(builder.Finish(&arr));
        arrays.push_back(arr);
    };

    finish(domain_b);
    finish(func_b);
    finish(pid_b);
    finish(tid_b);
    finish(corr_b);
    finish(start_b);
    finish(end_b);

    auto table = arrow::Table::Make(schema, arrays);
    write_table_as_parquet(filename, table);
}

#undef PARQUET_CHECK_OK
}  // namespace

void
write_parquet(
    const output_config&                                                    cfg,
    const metadata&                                                         tool_metadata,
    const std::vector<agent_info>&                                          /*agent_data*/,
    const generator<tool_buffer_tracing_hip_api_ext_record_t>&              hip_api_gen,
    const generator<rocprofiler_buffer_tracing_hsa_api_record_t>&           hsa_api_gen,
    const generator<tool_buffer_tracing_kernel_dispatch_ext_record_t>&      kernel_dispatch_gen,
    const generator<tool_buffer_tracing_memory_copy_ext_record_t>&          memory_copy_gen,
    const generator<rocprofiler_buffer_tracing_marker_api_record_t>&        marker_api_gen,
    const generator<tool_buffer_tracing_memory_allocation_ext_record_t>&    /*memory_alloc_gen*/,
    const generator<rocprofiler_buffer_tracing_scratch_memory_record_t>&    scratch_memory_gen,
    const generator<tool_buffer_tracing_kfd_record_t>&                      /*kfd_gen*/,
    const generator<rocprofiler_buffer_tracing_rccl_api_record_t>&          rccl_api_gen,
    const generator<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t>& rocdecode_api_gen,
    const generator<rocprofiler_buffer_tracing_rocjpeg_api_record_t>&       rocjpeg_api_gen,
    const generator<tool_counter_record_t>&                                 counter_collection_gen)
{
    write_kernel_dispatch_parquet(cfg, tool_metadata, kernel_dispatch_gen);
    write_counter_collection_parquet(cfg, tool_metadata, counter_collection_gen);
    write_memory_copy_parquet(cfg, tool_metadata, memory_copy_gen);
    write_scratch_memory_parquet(cfg, tool_metadata, scratch_memory_gen);
    write_marker_parquet(cfg, tool_metadata, marker_api_gen);

    write_api_trace_parquet(
        cfg, tool_metadata, get_domain_column_name(domain_type::HIP), hip_api_gen);
    write_api_trace_parquet(
        cfg, tool_metadata, get_domain_column_name(domain_type::HSA), hsa_api_gen);
    write_api_trace_parquet(
        cfg, tool_metadata, get_domain_column_name(domain_type::RCCL), rccl_api_gen);
    write_api_trace_parquet(
        cfg, tool_metadata, get_domain_column_name(domain_type::ROCDECODE), rocdecode_api_gen);
    write_api_trace_parquet(
        cfg, tool_metadata, get_domain_column_name(domain_type::ROCJPEG), rocjpeg_api_gen);
}
}  // namespace tool
}  // namespace rocprofiler
