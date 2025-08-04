// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/python/rocpd/source/csv.hpp"

#include "lib/common/defines.hpp"
#include "lib/common/hasher.hpp"
#include "lib/common/mpl.hpp"
#include "lib/output/csv.hpp"
#include "lib/output/csv_output_file.hpp"
#include "lib/output/generator.hpp"
#include "lib/output/metadata.hpp"
#include "lib/output/node_info.hpp"
#include "lib/output/output_config.hpp"
#include "lib/output/output_stream.hpp"
#include "lib/output/sql/common.hpp"
#include "lib/output/stream_info.hpp"
#include "lib/rocprofiler-sdk-tool/config.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <fmt/format.h>

#include <atomic>
#include <filesystem>
#include <future>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
const std::string STATS_HEADER = "\"Name\",\"Calls\",\"TotalDurationNs\","
                                 "\"AverageNs\",\"Percentage\",\"MinNs\",\"MaxNs\",\"StdDev\"";
const std::string API_TRACE_HEADER =
    "\"Guid\",\"Domain\",\"Function\",\"Process_Id\",\"Thread_Id\","
    "\"Correlation_Id\",\"Start_Timestamp\",\"End_Timestamp\"";
}  // namespace

namespace rocpd
{
namespace output
{
CsvManager::CsvManager(rocprofiler::tool::output_config output_cfg)
: config{std::move(output_cfg)}
{
    if(!ensure_output_directory())
    {
        ROCP_ERROR << "Failed to create csv output directory: " << config.output_path;
        return;
    }

    this->csv_configs = {
        {CsvType::KERNEL_DISPATCH,
         {"kernel_trace.csv",
          "\"Guid\",\"Kind\",\"Agent_Id\",\"Queue_Id\",\"Stream_Id\",\"Thread_Id\",\"Dispatch_Id\","
          "\"Kernel_Id\",\"Kernel_Name\",\"Correlation_Id\",\"Start_Timestamp\",\"End_Timestamp\","
          "\"LDS_Block_Size\",\"Scratch_Size\",\"VGPR_Count\",\"Accum_VGPR_Count\",\"SGPR_Count\","
          "\"Workgroup_Size_X\",\"Workgroup_Size_Y\",\"Workgroup_Size_Z\","
          "\"Grid_Size_X\",\"Grid_Size_Y\",\"Grid_Size_Z\""}},
        {CsvType::MEMORY_COPY,
         {"memory_copy_trace.csv",
          "\"Guid\",\"Kind\",\"Direction\",\"Stream_Id\",\"Source_Agent_Id\","
          "\"Destination_Agent_"
          "Id\","
          "\"Correlation_Id\",\"Start_Timestamp\",\"End_Timestamp\""}},
        {CsvType::MEMORY_ALLOCATION,
         {"memory_allocation_trace.csv",
          "\"Guid\",\"Kind\",\"Operation\",\"Agent_Id\",\"Allocation_Size\","
          "\"Address\","
          "\"Correlation_Id\",\"Start_Timestamp\",\"End_Timestamp\""}},
        {CsvType::SCRATCH_MEMORY,
         {"scratch_memory_trace.csv",
          "\"Kind\",\"Operation\",\"Agent_Id\",\"Queue_Id\",\"Thread_Id\","
          "\"Alloc_Flags\",\"Start_"
          "Timestamp\",\"End_Timestamp\""}},

        {CsvType::HIP_API, {"hip_api_trace.csv", API_TRACE_HEADER}},
        {CsvType::HSA_CSV_API, {"hsa_api_trace.csv", API_TRACE_HEADER}},
        {CsvType::MARKER, {"marker_api_trace.csv", API_TRACE_HEADER}},
        {CsvType::RCCL_API, {"rccl_api_trace.csv", API_TRACE_HEADER}},
        {CsvType::ROCDECODE_API, {"rocdecode_api_trace.csv", API_TRACE_HEADER}},
        {CsvType::ROCJPEG_API, {"rocjpeg_api_trace.csv", API_TRACE_HEADER}},

        {CsvType::COUNTER,
         {"counter_collection.csv",
          "\"Pid\",\"Correlation_Id\",\"Dispatch_Id\",\"Agent_Id\",\"Queue_Id\","
          "\"Process_Id\","
          "\"Thread_Id\","
          "\"Grid_Size\",\"Kernel_Id\",\"Kernel_Name\",\"Workgroup_Size\",\"LDS_"
          "Block_Size\","
          "\"Scratch_Size\",\"VGPR_Count\",\"Accum_VGPR_Count\",\"SGPR_Count\","
          "\"Counter_Name\",\"Counter_Value\",\"Start_Timestamp\",\"End_"
          "Timestamp\""}},
    };
}

bool
CsvManager::ensure_output_directory() const
{
    try
    {
        fs::create_directories(config.output_path);
        return true;
    } catch(const std::exception& e)
    {
        ROCP_ERROR << "Failed to create directory: " << e.what();
        return false;
    }
}

CsvManager::~CsvManager()
{
    for(auto& [type, stream] : streams)
    {
        if(stream.is_open())
        {
            stream.flush();
            stream.close();
        }
    }
}

std::ofstream&
CsvManager::get_stream(CsvType type)
{
    return streams[type];
}

bool
CsvManager::has_stream(CsvType type) const
{
    return streams.count(type) != 0u && streams.at(type).is_open();
}

bool
CsvManager::initialize_csv_file(CsvType type)
{
    if(has_stream(type)) return true;

    if(csv_configs.find(type) == csv_configs.end())
    {
        ROCP_ERROR << "No CSV configuration found for type: " << static_cast<int>(type);
        return false;
    }

    const auto& cfg = csv_configs[type];

    fs::path output_dir = config.output_path;
    fs::path filename =
        config.output_file.empty() ? cfg.filename : config.output_file + "_" + cfg.filename;

    file_paths[type] = (output_dir / filename).string();

    auto& path   = file_paths[type];
    auto& stream = streams[type];

    stream.open(path, std::ios::out);
    if(!stream.is_open())
    {
        ROCP_ERROR << "Failed to open CSV output file: " << path;
        return false;
    }

    stream << cfg.header << '\n';
    return true;
}

template <typename DataType>
bool
has_any_data(const rocprofiler::tool::generator<DataType>& data_gen)
{
    for(auto ditr : data_gen)
    {
        auto gen = data_gen.get(ditr);
        if(begin(gen) != end(gen))
        {
            return true;
        }
    }
    return false;
}

template <typename DataType, typename Processor>
void
process_data_to_csv(CsvManager&                                   csv_manager,
                    CsvType                                       csv_type,
                    const rocprofiler::tool::generator<DataType>& data_gen,
                    Processor                                     process_func)
{
    if(!has_any_data(data_gen)) return;

    if(!csv_manager.initialize_csv_file(csv_type)) return;

    for(auto ditr : data_gen)
    {
        auto gen = data_gen.get(ditr);
        for(auto it = begin(gen); it != end(gen); ++it)
        {
            process_func(csv_manager, csv_type, *it);
        }
    }
}

void
write_kernel_csv(
    CsvManager&                                                        csv_manager,
    const rocprofiler::tool::generator<rocpd::types::kernel_dispatch>& kernel_dispatch_gen)
{
    process_data_to_csv(
        csv_manager,
        CsvType::KERNEL_DISPATCH,
        kernel_dispatch_gen,
        [](CsvManager& cm, CsvType type, const rocpd::types::kernel_dispatch& kernel) {
            std::string kernel_identifier = cm.config.kernel_rename ? kernel.region : kernel.name;

            std::string agent_identifier = create_agent_index(cm.config.agent_index_value,
                                                              kernel.agent_abs_index,
                                                              kernel.agent_log_index,
                                                              kernel.agent_type_index,
                                                              std::string_view(kernel.agent_type))
                                               .as_string();

            cm.write_line(type,
                          fmt::format("\"{}\"", kernel.guid),
                          fmt::format("\"{}\"", "KERNEL_DISPATCH"),
                          fmt::format("\"{}\"", agent_identifier),
                          kernel.queue_id,
                          kernel.stream_id,
                          kernel.tid,
                          kernel.dispatch_id,
                          kernel.kernel_id,
                          fmt::format("\"{}\"", kernel_identifier),
                          kernel.stack_id,
                          kernel.start,
                          kernel.end,
                          kernel.lds_size,
                          kernel.scratch_size,
                          kernel.arch_vgpr_count,
                          kernel.accum_vgpr_count,
                          kernel.sgpr_count,
                          kernel.workgroup_size.x,
                          kernel.workgroup_size.y,
                          kernel.workgroup_size.z,
                          kernel.grid_size.x,
                          kernel.grid_size.y,
                          kernel.grid_size.z);
        });
}

void
write_memory_copy_csv(
    CsvManager&                                                      csv_manager,
    const rocprofiler::tool::generator<rocpd::types::memory_copies>& memory_copies_gen)
{
    process_data_to_csv(csv_manager,
                        CsvType::MEMORY_COPY,
                        memory_copies_gen,
                        [](CsvManager& cm, CsvType type, const rocpd::types::memory_copies& mcopy) {
                            std::string src_agent_identifier =
                                create_agent_index(cm.config.agent_index_value,
                                                   mcopy.src_agent_abs_index,
                                                   mcopy.src_agent_log_index,
                                                   mcopy.src_agent_type_index,
                                                   std::string_view(mcopy.src_agent_type))
                                    .as_string();

                            std::string dst_agent_identifier =
                                create_agent_index(cm.config.agent_index_value,
                                                   mcopy.dst_agent_abs_index,
                                                   mcopy.dst_agent_log_index,
                                                   mcopy.dst_agent_type_index,
                                                   std::string_view(mcopy.dst_agent_type))
                                    .as_string();

                            cm.write_line(type,
                                          fmt::format("\"{}\"", mcopy.guid),
                                          fmt::format("\"{}\"", "MEMORY_COPY"),
                                          fmt::format("\"{}\"", mcopy.name),
                                          mcopy.stream_id,
                                          fmt::format("\"{}\"", src_agent_identifier),
                                          fmt::format("\"{}\"", dst_agent_identifier),
                                          mcopy.stack_id,
                                          mcopy.start,
                                          mcopy.end);
                        });
}

void
write_memory_allocation_csv(
    CsvManager&                                                          csv_manager,
    const rocprofiler::tool::generator<rocpd::types::memory_allocation>& memory_alloc_gen)
{
    process_data_to_csv(
        csv_manager,
        CsvType::MEMORY_ALLOCATION,
        memory_alloc_gen,
        [](CsvManager& cm, CsvType type, const rocpd::types::memory_allocation& malloc) {
            std::string operation = fmt::format("MEMORY_ALLOCATION_{}", malloc.type);

            std::string agent_identifier = create_agent_index(cm.config.agent_index_value,
                                                              malloc.agent_abs_index,
                                                              malloc.agent_log_index,
                                                              malloc.agent_type_index,
                                                              std::string_view(malloc.agent_type))
                                               .as_string();

            std::string agent_id =
                operation != "MEMORY_ALLOCATION_FREE" ? agent_identifier : "\"\"";
            std::string address = fmt::format("\"0x{:016x}\"", malloc.address);

            cm.write_line(type,
                          fmt::format("\"{}\"", malloc.guid),
                          fmt::format("\"{}\"", "MEMORY_ALLOCATION"),
                          fmt::format("\"{}\"", operation),
                          fmt::format("\"{}\"", agent_id),
                          malloc.size,
                          address,
                          malloc.stack_id,
                          malloc.start,
                          malloc.end);
        });
}

void
write_scratch_memory_csv(
    CsvManager&                                                       csv_manager,
    const rocprofiler::tool::generator<rocpd::types::scratch_memory>& scratch_memory_gen)
{
    process_data_to_csv(
        csv_manager,
        CsvType::SCRATCH_MEMORY,
        scratch_memory_gen,
        [](CsvManager& cm, CsvType type, const rocpd::types::scratch_memory& scratch_mem) {
            std::string agent_identifier =
                create_agent_index(cm.config.agent_index_value,
                                   scratch_mem.agent_abs_index,
                                   scratch_mem.agent_log_index,
                                   scratch_mem.agent_type_index,
                                   std::string_view(scratch_mem.agent_type))
                    .as_string();

            cm.write_line(type,
                          fmt::format("\"{}\"", "SCRATCH_MEMORY"),
                          fmt::format("\"SCRATCH_MEMORY_{}\"", scratch_mem.operation),
                          fmt::format("\"{}\"", agent_identifier),
                          scratch_mem.queue_id,
                          scratch_mem.tid,
                          scratch_mem.alloc_flags,
                          scratch_mem.start,
                          scratch_mem.end);
        });
}

void
write_hip_api_csv(CsvManager&                                               csv_manager,
                  const rocprofiler::tool::generator<rocpd::types::region>& hip_api_gen)
{
    process_data_to_csv(csv_manager,
                        CsvType::HIP_API,
                        hip_api_gen,
                        [](CsvManager& cm, CsvType type, const rocpd::types::region& api) {
                            // Skip entries that are not HIP API calls
                            if(api.category.find("HIP_") != 0) return;

                            cm.write_line(type,
                                          fmt::format("\"{}\"", api.guid),
                                          fmt::format("\"{}\"", api.category),
                                          fmt::format("\"{}\"", api.name),
                                          api.pid,
                                          api.tid,
                                          api.stack_id,
                                          api.start,
                                          api.end);
                        });
}

void
write_hsa_api_csv(CsvManager&                                               csv_manager,
                  const rocprofiler::tool::generator<rocpd::types::region>& hsa_api_gen)
{
    process_data_to_csv(csv_manager,
                        CsvType::HSA_CSV_API,
                        hsa_api_gen,
                        [](CsvManager& cm, CsvType type, const rocpd::types::region& api) {
                            // Skip entries that are not HSA API calls
                            if(api.category.find("HSA_") != 0) return;

                            cm.write_line(type,
                                          fmt::format("\"{}\"", api.guid),
                                          fmt::format("\"{}\"", api.category),
                                          fmt::format("\"{}\"", api.name),
                                          api.pid,
                                          api.tid,
                                          api.stack_id,
                                          api.start,
                                          api.end);
                        });
}

void
write_marker_api_csv(CsvManager&                                               csv_manager,
                     const rocprofiler::tool::generator<rocpd::types::region>& marker_api_gen)
{
    namespace tool = ::rocprofiler::tool;

    if(marker_api_gen.empty()) return;

    using marker_csv_encoder = tool::csv::csv_encoder<8>;

    auto ofs = tool::csv_output_file{csv_manager.config,
                                     domain_type::MARKER,
                                     marker_csv_encoder{},
                                     {"Guid",
                                      "Domain",
                                      "Function",
                                      "Process_Id",
                                      "Thread_Id",
                                      "Correlation_Id",
                                      "Start_Timestamp",
                                      "End_Timestamp"}};

    // write samples first and ignore the timestamp ordering w.r.t. regions for now
    for(auto ditr : marker_api_gen)
    {
        for(const auto& record : marker_api_gen.get(ditr))
        {
            auto row_ss = std::stringstream{};
            auto _name  = record.name;

            if(record.has_extdata())
            {
                if(auto _extdata = record.get_extdata(); !_extdata.message.empty())
                    _name = _extdata.message;
            }

            marker_csv_encoder::write_row(row_ss,
                                          record.guid,
                                          record.category,
                                          _name,
                                          record.pid,
                                          record.tid,
                                          record.stack_id,
                                          record.start,
                                          record.end);

            ofs << row_ss.str();
        }
    }
}

void
write_rccl_api_csv(CsvManager&                                               csv_manager,
                   const rocprofiler::tool::generator<rocpd::types::region>& rccl_api_gen)
{
    process_data_to_csv(csv_manager,
                        CsvType::RCCL_API,
                        rccl_api_gen,
                        [](CsvManager& cm, CsvType type, const rocpd::types::region& api) {
                            if(api.category.find("RCCL_") != 0) return;

                            cm.write_line(type,
                                          fmt::format("\"{}\"", api.guid),
                                          fmt::format("\"{}\"", api.category),
                                          fmt::format("\"{}\"", api.name),
                                          api.pid,
                                          api.tid,
                                          api.stack_id,
                                          api.start,
                                          api.end);
                        });
}

void
write_rocdecode_api_csv(CsvManager&                                               csv_manager,
                        const rocprofiler::tool::generator<rocpd::types::region>& rocdecode_api_gen)
{
    process_data_to_csv(csv_manager,
                        CsvType::ROCDECODE_API,
                        rocdecode_api_gen,
                        [](CsvManager& cm, CsvType type, const rocpd::types::region& api) {
                            if(api.category.find("ROCDECODE_") != 0) return;

                            cm.write_line(type,
                                          fmt::format("\"{}\"", api.guid),
                                          fmt::format("\"{}\"", api.category),
                                          fmt::format("\"{}\"", api.name),
                                          api.pid,
                                          api.tid,
                                          api.stack_id,
                                          api.start,
                                          api.end);
                        });
}

void
write_rocjpeg_api_csv(CsvManager&                                               csv_manager,
                      const rocprofiler::tool::generator<rocpd::types::region>& rocjpeg_api_gen)
{
    process_data_to_csv(csv_manager,
                        CsvType::ROCJPEG_API,
                        rocjpeg_api_gen,
                        [](CsvManager& cm, CsvType type, const rocpd::types::region& api) {
                            if(api.category.find("ROCJPEG_") != 0) return;

                            cm.write_line(type,
                                          fmt::format("\"{}\"", api.guid),
                                          fmt::format("\"{}\"", api.category),
                                          fmt::format("\"{}\"", api.name),
                                          api.pid,
                                          api.tid,
                                          api.stack_id,
                                          api.start,
                                          api.end);
                        });
}

void
write_agent_info_csv(CsvManager& csv_manager, const std::vector<rocpd::types::agent>& agents)
{
    if(agents.empty()) return;

    namespace tool               = ::rocprofiler::tool;
    using agent_info_csv_encoder = tool::csv::csv_encoder<54>;

    auto ofs = tool::csv_output_file{csv_manager.config,
                                     "agent_info",
                                     agent_info_csv_encoder{},
                                     {"Guid",
                                      "Node_Id",
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

    for(const auto& itr : agents)
    {
        auto row_ss = std::stringstream{};
        agent_info_csv_encoder::write_row(row_ss,
                                          itr.guid,
                                          itr.node_id,
                                          itr.logical_node_id,
                                          itr.type,
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
write_counters_csv(CsvManager&                                                csv_manager,
                   const rocprofiler::tool::generator<rocpd::types::counter>& counter_gen)
{
    process_data_to_csv(csv_manager,
                        CsvType::COUNTER,
                        counter_gen,
                        [](CsvManager& cm, CsvType type, const rocpd::types::counter& counter) {
                            std::string agent_identifier =
                                create_agent_index(cm.config.agent_index_value,
                                                   counter.agent_abs_index,
                                                   counter.agent_log_index,
                                                   counter.agent_type_index,
                                                   std::string_view(counter.agent_type))
                                    .as_string();

                            cm.write_line(type,
                                          counter.guid,
                                          counter.stack_id,
                                          counter.dispatch_id,
                                          fmt::format("\"{}\"", agent_identifier),
                                          counter.queue_id,
                                          counter.pid,
                                          counter.tid,
                                          counter.grid_size,
                                          counter.kernel_id,
                                          fmt::format("\"{}\"", counter.kernel_name),
                                          counter.workgroup_size,
                                          counter.lds_block_size,
                                          counter.scratch_size,
                                          counter.arch_vgpr_count,
                                          counter.accum_vgpr_count,
                                          counter.sgpr_count,
                                          fmt::format("\"{}\"", counter.counter_name),
                                          counter.value,
                                          counter.start,
                                          counter.end);
                        });
}

void
write_csvs(CsvManager&                                                          csv_manager,
           const rocprofiler::tool::generator<rocpd::types::kernel_dispatch>&   kernel_dispatch,
           const rocprofiler::tool::generator<rocpd::types::memory_copies>&     memory_copies,
           const rocprofiler::tool::generator<rocpd::types::memory_allocation>& memory_allocations,
           const rocprofiler::tool::generator<rocpd::types::region>&            hip_api_calls,
           const rocprofiler::tool::generator<rocpd::types::region>&            hsa_api_calls,
           const rocprofiler::tool::generator<rocpd::types::region>&            marker_api_calls,
           const rocprofiler::tool::generator<rocpd::types::counter>&           counters_calls,
           const rocprofiler::tool::generator<rocpd::types::scratch_memory>& scratch_memory_calls,
           const rocprofiler::tool::generator<rocpd::types::region>&         rccl_calls,
           const rocprofiler::tool::generator<rocpd::types::region>&         rocdecode_calls,
           const rocprofiler::tool::generator<rocpd::types::region>&         rocjpeg_calls)
{
    rocpd::output::write_kernel_csv(csv_manager, kernel_dispatch);
    rocpd::output::write_memory_copy_csv(csv_manager, memory_copies);
    rocpd::output::write_memory_allocation_csv(csv_manager, memory_allocations);
    rocpd::output::write_hip_api_csv(csv_manager, hip_api_calls);
    rocpd::output::write_hsa_api_csv(csv_manager, hsa_api_calls);
    rocpd::output::write_marker_api_csv(csv_manager, marker_api_calls);

    rocpd::output::write_counters_csv(csv_manager, counters_calls);
    rocpd::output::write_scratch_memory_csv(csv_manager, scratch_memory_calls);
    rocpd::output::write_rccl_api_csv(csv_manager, rccl_calls);

    rocpd::output::write_rocdecode_api_csv(csv_manager, rocdecode_calls);
    rocpd::output::write_rocjpeg_api_csv(csv_manager, rocjpeg_calls);
}
}  // namespace output
}  // namespace rocpd
