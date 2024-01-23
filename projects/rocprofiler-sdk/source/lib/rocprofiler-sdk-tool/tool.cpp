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
#include "helper.hpp"
#include "output_file.hpp"

#include "lib/common/demangle.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <glog/logging.h>

#include <fmt/core.h>
#include <unistd.h>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace common                     = ::rocprofiler::common;
namespace tool                       = ::rocprofiler::tool;
static const uint32_t lds_block_size = 128 * 4;

namespace
{}  // namespace

auto&
get_hsa_api_file()
{
    static auto _v = tool::output_file{"hsa_api_trace",
                                       tool::csv::hsa_csv_encoder{},
                                       {"Domain",
                                        "Function",
                                        "Process_Id",
                                        "Thread_Id",
                                        "Correlation_Id",
                                        "Start_Timestamp",
                                        "End_Timestamp"}};
    return _v;
}

auto&
get_kernel_trace_file()
{
    static auto _v = tool::output_file{"kernel_trace",
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
    return _v;
}

auto&
get_counter_collection_file()
{
    static auto _v = tool::output_file{"counter_collection",
                                       tool::csv::counter_collection_csv_encoder{},
                                       {"Counter_Id",
                                        "Agent_Id",
                                        "Queue_Id",
                                        "Process_Id",
                                        "Thread_Id",
                                        "Grid_Size",
                                        "Kernel-Name",
                                        "Workgroup_Size",
                                        "LDS_Block_Size",
                                        "Scratch_Size",
                                        "VGPR_Count",
                                        "SGPR_Count",
                                        "Counter_Name",
                                        "Counter_Value"}};
    return _v;
}

auto&
get_memory_copy_trace_file()
{
    static auto _v = tool::output_file{"memory_copy_trace",
                                       tool::csv::memory_copy_csv_encoder{},
                                       {"Kind",
                                        "Direction",
                                        "Source_Agent_Id",
                                        "Destination_Agent_Id",
                                        "Correlation_Id",
                                        "Start_Timestamp",
                                        "End_Timestamp"}};
    return _v;
}

rocprofiler_buffer_id_t&
get_hsa_api_trace_buffer()
{
    static rocprofiler_buffer_id_t hsa_api_buf = {};
    return hsa_api_buf;
}

rocprofiler_buffer_id_t&
get_kernel_trace_buffer()
{
    static rocprofiler_buffer_id_t kernel_trace_buf = {};
    return kernel_trace_buf;
}

rocprofiler_buffer_id_t&
get_counter_collection_buffer()
{
    static rocprofiler_buffer_id_t counter_collection_buf = {};
    return counter_collection_buf;
}

rocprofiler_buffer_id_t&
get_memory_copy_trace_buffer()
{
    static rocprofiler_buffer_id_t memory_copy_buf = {};
    return memory_copy_buf;
}

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

using kernel_symbol_data_map_t = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data>;
auto kernel_data               = common::Synchronized<kernel_symbol_data_map_t, true>{};
auto name_info                 = get_buffer_id_names();

auto&
get_client_ctx()
{
    static rocprofiler_context_id_t context_id;
    return context_id;
}

void
flush()
{
    for(auto itr : {get_memory_copy_trace_buffer(),
                    get_kernel_trace_buffer(),
                    get_counter_collection_buffer(),
                    get_hsa_api_trace_buffer()})
    {
        if(itr.handle > 0) ROCPROFILER_CALL(rocprofiler_flush_buffer(itr), "buffer flush");
    }
}

void
rocprofiler_tracing_callback(rocprofiler_callback_tracing_record_t record,
                             rocprofiler_user_data_t*              user_data,
                             void*                                 data)
{
    throw std::runtime_error{"not implemented"};

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
       record.operation == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            flush();
        }
    }

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* sym_data = static_cast<rocprofiler_kernel_symbol_data_t*>(record.payload);
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            kernel_data.wlock(
                [](kernel_symbol_data_map_t& kdata, rocprofiler_kernel_symbol_data_t* sym_data_v) {
                    kdata.emplace(sym_data_v->kernel_id, kernel_symbol_data{*sym_data_v});
                },
                sym_data);
        }
    }

    (void) user_data;
    (void) data;
}

void
buffered_callback(rocprofiler_context_id_t /*context*/,
                  rocprofiler_buffer_id_t /*buffer_id*/,
                  rocprofiler_record_header_t** headers,
                  size_t                        num_headers,
                  void* /*user_data*/,
                  uint64_t /*drop_count*/)
{
    static auto _sync = std::mutex{};
    auto        _lk   = std::lock_guard<std::mutex>{_sync};

    if(num_headers == 0)
        throw std::runtime_error{"rocprofiler invoked a buffer callback with no headers "
                                 "this should never happen"};

    else if(headers == nullptr)
        throw std::runtime_error{"rocprofiler invoked a buffer callback with a null pointer to the "
                                 "array of headers. this should never happen"};

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING)
        {
            if(header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
                    header->payload);
                std::string kernel_name = kernel_data.rlock(
                    [](const kernel_symbol_data_map_t&                      kdata,
                       rocprofiler_buffer_tracing_kernel_dispatch_record_t* record_v) {
                        return kdata.at(record_v->kernel_id).formatted_kernel_name;
                    },
                    record);

                auto kernel_trace_ss = std::stringstream{};
                tool::csv::kernel_trace_csv_encoder::write_row(
                    kernel_trace_ss,
                    name_info.kind_names.at(record->kind),
                    record->agent_id.handle,
                    record->queue_id.handle,
                    record->kernel_id,
                    std::move(kernel_name),
                    record->correlation_id.internal,
                    record->start_timestamp,
                    record->end_timestamp,
                    record->private_segment_size,
                    record->group_segment_size,
                    record->workgroup_size.x,
                    record->workgroup_size.y,
                    record->workgroup_size.z,
                    record->grid_size.x,
                    record->grid_size.y,
                    record->grid_size.z);

                get_kernel_trace_file() << kernel_trace_ss.str();
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_HSA_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(header->payload);

                auto hsa_trace_ss = std::stringstream{};
                tool::csv::hsa_csv_encoder::write_row(
                    hsa_trace_ss,
                    name_info.kind_names.at(record->kind),
                    name_info.operation_names.at(record->kind).at(record->operation),
                    getpid(),
                    record->thread_id,
                    record->correlation_id.internal,
                    record->start_timestamp,
                    record->end_timestamp);

                get_hsa_api_file() << hsa_trace_ss.str();
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload);

                auto memory_copy_trace_ss = std::stringstream{};
                tool::csv::memory_copy_csv_encoder::write_row(
                    memory_copy_trace_ss,
                    name_info.kind_names.at(record->kind),
                    name_info.operation_names.at(record->kind).at(record->operation),
                    record->src_agent_id.handle,
                    record->dst_agent_id.handle,
                    record->correlation_id.internal,
                    record->start_timestamp,
                    record->end_timestamp);

                get_memory_copy_trace_file() << memory_copy_trace_ss.str();
            }
            else
            {
                LOG(FATAL) << fmt::format(
                    "unsupported category + kind: {} + {}", header->category, header->kind);
            }
        }

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS && header->kind == 0)
        {
            auto* profiler_record = static_cast<rocprofiler_record_counter_t*>(header->payload);
            rocprofiler_tool_kernel_properties_t kernel_properties =
                GetKernelProperties(profiler_record->corr_id.internal);
            rocprofiler_counter_id_t counter_id;
            const char*              counter_name;
            size_t                   size, pos;
            rocprofiler_query_record_counter_id(profiler_record->id, &counter_id);
            rocprofiler_query_counter_name(counter_id, &counter_name, &size);
            rocprofiler_query_record_dimension_position(profiler_record->id, 0, &pos);

            auto counter_collection_ss = std::stringstream{};
            counter_collection_ss << counter_id.handle << ","
                                  << kernel_properties.gpu_agent.id.handle << ","
                                  << kernel_properties.queue_id.handle << "," << getpid() << ","
                                  << kernel_properties.thread_id << ",";
            counter_collection_ss << kernel_properties.grid_size << ","
                                  << kernel_properties.kernel_name << ","
                                  << kernel_properties.workgroup_size << ","
                                  << ((kernel_properties.lds_size + (lds_block_size - 1)) &
                                      ~(lds_block_size - 1))
                                  << "," << kernel_properties.scratch_size << ","
                                  << kernel_properties.arch_vgpr_count << ","
                                  << kernel_properties.sgpr_count << ",";
            /*
             Iterate through the N dimensional that is obtained for the counter.
              given instance id what is the counter id
              given counter id what is the counter name
              given instance how many dimension
              iterate through dimensions
                what is the dimension id
                what is the dimension name
                what pos in the dimension.
            */
            // ss << counter_name << "[" << info.name << "," << pos << "]" << ",";
            // ss << profiler_record->counter_value << "\n";
            counter_collection_ss << counter_name << "["
                                  << "," << pos << "]"
                                  << ",";
            counter_collection_ss << counter_name << ",";
            counter_collection_ss << profiler_record->counter_value << "\n";
            get_counter_collection_file() << counter_collection_ss.str() << "\n";
        }
    }
}

using counter_vec_t = std::vector<rocprofiler_counter_id_t>;
using agent_counter_map_t =
    std::unordered_map<const rocprofiler_agent_t*, std::optional<rocprofiler_profile_config_id_t>>;

// this function creates a rocprofiler profile config on the first entry
auto
get_agent_profile(const rocprofiler_agent_t* agent)
{
    static auto data = common::Synchronized<agent_counter_map_t>{};

    auto profile = std::optional<rocprofiler_profile_config_id_t>{};
    data.ulock(
        [agent, &profile](const agent_counter_map_t& data_v) {
            auto itr = data_v.find(agent);
            if(itr != data_v.end())
            {
                profile = itr->second;
                return true;
            }
            return false;
        },
        [agent, &profile](agent_counter_map_t& data_v) {
            auto counters_v = counter_vec_t{};
            ROCPROFILER_CALL(
                rocprofiler_iterate_agent_supported_counters(
                    *agent,
                    [](rocprofiler_counter_id_t* counters, size_t num_counters, void* user_data) {
                        auto* vec = static_cast<counter_vec_t*>(user_data);
                        for(size_t i = 0; i < num_counters; i++)
                        {
                            const char* name = nullptr;
                            size_t      len  = 0;

                            ROCPROFILER_CALL(
                                rocprofiler_query_counter_name(counters[i], &name, &len),
                                "Could not query name");

                            if(name && len > 0)
                            {
                                if(tool::get_config().counters.count(name) > 0)
                                    vec->emplace_back(counters[i]);
                            }
                        }
                        return ROCPROFILER_STATUS_SUCCESS;
                    },
                    static_cast<void*>(&counters_v)),
                "iterate agent supported counters");

            if(!counters_v.empty())
            {
                auto profile_v = rocprofiler_profile_config_id_t{};
                ROCPROFILER_CALL(rocprofiler_create_profile_config(
                                     *agent, counters_v.data(), counters_v.size(), &profile_v),
                                 "Could not construct profile cfg");
                profile = profile_v;
            }

            data_v.emplace(agent, profile);
            return true;
        });

    return profile;
}

void
dispatch_callback(rocprofiler_queue_id_t              queue_id,
                  const rocprofiler_agent_t*          agent,
                  rocprofiler_correlation_id_t        correlation_id,
                  const hsa_kernel_dispatch_packet_t* dispatch_packet,
                  uint64_t                            kernel_id,
                  void* /*callback_data_args*/,
                  rocprofiler_profile_config_id_t* config)
{
    rocprofiler_tool_kernel_properties_t kernel_properties;
    const auto&                          kernel_info =
        kernel_data.rlock([](const kernel_symbol_data_map_t& kdata,
                             uint64_t kernel_id_v) { return kdata.at(kernel_id_v); },
                          kernel_id);

    auto is_targeted_kernel = [&kernel_info]() {
        for(const auto& name : tool::get_config().kernel_names)
        {
            if(name == kernel_info.truncated_kernel_name)
                return true;
            else
            {
                auto dkernel_name = std::string_view{kernel_info.demangled_kernel_name};
                auto pos          = dkernel_name.find(name);
                // if the demangled kernel name contains name and the next character is '(' then
                // mark as found
                if(pos != std::string::npos && (pos + 1) < dkernel_name.size() &&
                   dkernel_name.at(pos + 1) == '(')
                    return true;
            }
        }
        return false;
    };

    if(!is_targeted_kernel()) return;

    auto profile = get_agent_profile(agent);

    if(profile)
    {
        kernel_properties.kernel_name = kernel_info.formatted_kernel_name;
        kernel_properties.queue_id    = queue_id;
        kernel_properties.gpu_agent   = *agent;
        kernel_properties.thread_id   = common::get_tid();
        populate_kernel_properties_data(&kernel_properties, dispatch_packet);
        SetKernelProperties(correlation_id.internal, kernel_properties);

        *config = *profile;
    }
}

rocprofiler_client_finalize_t client_finalizer  = nullptr;
rocprofiler_client_id_t*      client_identifier = nullptr;

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    client_finalizer = fini_func;

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "create context failed");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(get_client_ctx(),
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       code_object_tracing_callback,
                                                       nullptr),
        "code object tracing configure failed");

    if(tool::get_config().kernel_trace)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                                   4096,
                                                   2048,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   buffered_callback,
                                                   tool_data,
                                                   &get_kernel_trace_buffer()),
                         "buffer creation");

        ROCPROFILER_CALL(
            rocprofiler_configure_buffer_tracing_service(get_client_ctx(),
                                                         ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                         nullptr,
                                                         0,
                                                         get_kernel_trace_buffer()),
            "buffer tracing service for kernel dispatch configure");
    }

    if(tool::get_config().memory_copy_trace)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                                   4096,
                                                   2048,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   buffered_callback,
                                                   nullptr,
                                                   &get_memory_copy_trace_buffer()),
                         "create memory copy buffer");

        ROCPROFILER_CALL(
            rocprofiler_configure_buffer_tracing_service(get_client_ctx(),
                                                         ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                                         nullptr,
                                                         0,
                                                         get_memory_copy_trace_buffer()),
            "buffer tracing service for memory copy configure");
    }

    if(tool::get_config().hsa_api_trace)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                                   4096,
                                                   2048,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   buffered_callback,
                                                   tool_data,
                                                   &get_hsa_api_trace_buffer()),
                         "buffer creation");

        ROCPROFILER_CALL(
            rocprofiler_configure_buffer_tracing_service(get_client_ctx(),
                                                         ROCPROFILER_BUFFER_TRACING_HSA_API,
                                                         nullptr,
                                                         0,
                                                         get_hsa_api_trace_buffer()),
            "buffer tracing service for memory copy configure");
    }

    if(tool::get_config().counter_collection)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                                   4096,
                                                   2048,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   buffered_callback,
                                                   nullptr,
                                                   &get_counter_collection_buffer()),
                         "buffer creation failed");

        ROCPROFILER_CALL(
            rocprofiler_configure_buffered_dispatch_profile_counting_service(
                get_client_ctx(), get_counter_collection_buffer(), dispatch_callback, nullptr),
            "Could not setup buffered service");
    }

    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context failed");

    std::atexit([]() {
        if(client_finalizer && client_identifier) client_finalizer(*client_identifier);
    });

    return 0;
}

void
tool_fini(void* tool_data)
{
    client_identifier = nullptr;
    client_finalizer  = nullptr;

    flush();
    rocprofiler_stop_context(get_client_ctx());

    (void) (tool_data);
}

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /*version*/,
                      const char* /*runtime_version*/,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    common::init_logging("ROCPROF_LOG_LEVEL");
    FLAGS_colorlogtostderr = true;

    // only activate if main tool
    if(priority > 0) return nullptr;

    // set the client name
    id->name = "rocprofiler-tool";

    // store client info
    client_identifier = id;

    // create configure data
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};

    // return pointer to configure data
    return &cfg;
    // data passed around all the callbacks
}
