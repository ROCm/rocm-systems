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

#include "buffered_output.hpp"
#include "config.hpp"
#include "csv.hpp"
#include "domain_type.hpp"
#include "generateCSV.hpp"
#include "generateJSON.hpp"
#include "generateOTF2.hpp"
#include "generatePerfetto.hpp"
#include "generateStats.hpp"
#include "helper.hpp"
#include "output_file.hpp"
#include "statistics.hpp"
#include "tmp_file.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/units.hpp"
#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>
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
#include <algorithm>
#include <cassert>
#include <csignal>
#include <cstring>
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
get_list_basic_metrics_file()
{
    static auto* _v =
        new tool::output_file{"basic_metrics",
                              tool::csv::list_basic_metrics_csv_encoder{},
                              {"Agent_Id", "Name", "Description", "Block", "Dimensions"}};
    ADD_DESTRUCTOR(_v);
    return _v;
}

tool::output_file*&
get_list_derived_metrics_file()
{
    static auto* _v =
        new tool::output_file{"derived_metrics",
                              tool::csv::list_derived_metrics_csv_encoder{},
                              {"Agent_Id", "Name", "Description", "Expression", "Dimensions"}};
    ADD_DESTRUCTOR(_v);
    return _v;
}

#undef ADD_DESTRUCTOR

struct buffer_ids
{
    rocprofiler_buffer_id_t hsa_api_trace      = {};
    rocprofiler_buffer_id_t hip_api_trace      = {};
    rocprofiler_buffer_id_t kernel_trace       = {};
    rocprofiler_buffer_id_t memory_copy_trace  = {};
    rocprofiler_buffer_id_t counter_collection = {};
    rocprofiler_buffer_id_t scratch_memory     = {};
    rocprofiler_buffer_id_t rccl_api_trace     = {};

    auto as_array() const
    {
        return std::array<rocprofiler_buffer_id_t, 7>{hsa_api_trace,
                                                      hip_api_trace,
                                                      kernel_trace,
                                                      memory_copy_trace,
                                                      counter_collection,
                                                      scratch_memory,
                                                      rccl_api_trace};
    }
};

buffer_ids&
get_buffers()
{
    static auto _v = buffer_ids{};
    return _v;
}

using rocprofiler_code_object_data_t = rocprofiler_callback_tracing_code_object_load_data_t;

template <typename Tp>
Tp*
as_pointer(Tp&& _val)
{
    return new Tp{std::forward<Tp>(_val)};
}

template <typename Tp>
Tp*
as_pointer()
{
    return new Tp{};
}

using code_object_data_map_t = std::unordered_map<uint64_t, rocprofiler_code_object_data_t>;
using targeted_kernels_map_t =
    std::unordered_map<rocprofiler_kernel_id_t, std::unordered_set<uint32_t>>;
using counter_dimension_info_map_t =
    std::unordered_map<uint64_t, std::vector<rocprofiler_record_dimension_info_t>>;
using agent_info_map_t      = std::unordered_map<rocprofiler_agent_id_t, rocprofiler_agent_t>;
using kernel_iteration_t    = std::unordered_map<rocprofiler_kernel_id_t, uint32_t>;
using kernel_rename_map_t   = std::unordered_map<uint64_t, uint64_t>;
using kernel_rename_stack_t = std::stack<uint64_t>;

auto  code_obj_data          = as_pointer<common::Synchronized<code_object_data_map_t, true>>();
auto* kernel_data            = as_pointer<common::Synchronized<kernel_symbol_data_map_t, true>>();
auto* marker_msg_data        = as_pointer<common::Synchronized<marker_message_map_t, true>>();
auto  counter_dimension_data = common::Synchronized<counter_dimension_info_map_t, true>{};
auto  target_kernels         = common::Synchronized<targeted_kernels_map_t>{};
auto* buffered_name_info     = as_pointer(get_buffer_id_names());
auto* callback_name_info     = as_pointer(get_callback_id_names());
auto* agent_info             = as_pointer(agent_info_map_t{});
auto* tool_functions         = as_pointer(tool_table{});
auto* stats_timestamp        = as_pointer(timestamps_t{});
auto  kernel_iteration       = common::Synchronized<kernel_iteration_t, true>{};

thread_local auto thread_dispatch_rename      = as_pointer<kernel_rename_stack_t>();
thread_local auto thread_dispatch_rename_dtor = common::scope_destructor{[]() {
    delete thread_dispatch_rename;
    thread_dispatch_rename = nullptr;
}};

bool
add_kernel_target(uint64_t _kern_id, const std::unordered_set<uint32_t>& range)
{
    return target_kernels
        .wlock(
            [](targeted_kernels_map_t&             _targets_v,
               uint64_t                            _kern_id_v,
               const std::unordered_set<uint32_t>& _range) {
                return _targets_v.emplace(_kern_id_v, _range);
            },
            _kern_id,
            range)
        .second;
}

bool
is_targeted_kernel(uint64_t _kern_id)
{
    const std::unordered_set<uint32_t>* range = target_kernels.rlock(
        [](const auto& _targets_v, uint64_t _kern_id_v) -> const std::unordered_set<uint32_t>* {
            if(_targets_v.find(_kern_id_v) != _targets_v.end()) return &_targets_v.at(_kern_id_v);
            return nullptr;
        },
        _kern_id);

    if(range)
    {
        return kernel_iteration.rlock(
            [](const auto&                         _kernel_iter,
               uint64_t                            _kernel_id,
               const std::unordered_set<uint32_t>& _range) {
                auto itr = _kernel_iter.at(_kernel_id);

                // If the iteration range is not given then all iterations of the kernel is profiled
                if(_range.empty())
                    return true;
                else if(_range.find(itr) != _range.end())
                    return true;
                return false;
            },
            _kern_id,
            *range);
    }

    return false;
}

auto&
get_client_ctx()
{
    static rocprofiler_context_id_t context_id{0};
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

std::string_view
get_callback_kind(rocprofiler_callback_tracing_kind_t kind)
{
    return CHECK_NOTNULL(callback_name_info)->at(kind);
}

std::string_view
get_callback_op_name(rocprofiler_callback_tracing_kind_t kind, uint32_t op)
{
    return CHECK_NOTNULL(callback_name_info)->at(kind, op);
}

std::string_view
get_roctx_msg(uint64_t cid)
{
    return CHECK_NOTNULL(marker_msg_data)
        ->rlock(
            [](const auto& _data, uint64_t _cid_v) -> std::string_view { return _data.at(_cid_v); },
            cid);
}

int
set_kernel_rename_correlation_id(rocprofiler_thread_id_t                            thr_id,
                                 rocprofiler_context_id_t                           ctx_id,
                                 rocprofiler_external_correlation_id_request_kind_t kind,
                                 rocprofiler_tracing_operation_t                    op,
                                 uint64_t                 internal_corr_id,
                                 rocprofiler_user_data_t* external_corr_id,
                                 void*                    user_data)
{
    ROCP_FATAL_IF(kind != ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH)
        << "unexpected kind: " << kind;

    if(thread_dispatch_rename != nullptr && !thread_dispatch_rename->empty())
        external_corr_id->value = thread_dispatch_rename->top();

    common::consume_args(thr_id, ctx_id, kind, op, internal_corr_id, user_data);

    return 0;
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
            auto marker_record            = rocprofiler_buffer_tracing_marker_api_record_t{};
            marker_record.size            = sizeof(rocprofiler_buffer_tracing_marker_api_record_t);
            marker_record.kind            = convert_marker_tracing_kind(record.kind);
            marker_record.operation       = record.operation;
            marker_record.thread_id       = record.thread_id;
            marker_record.correlation_id  = record.correlation_id;
            marker_record.start_timestamp = user_data->value;
            marker_record.end_timestamp   = ts;
            write_ring_buffer(marker_record, domain_type::MARKER);
        }
    }
}

void
kernel_rename_callback(rocprofiler_callback_tracing_record_t record,
                       rocprofiler_user_data_t*              user_data,
                       void*                                 data)
{
    if(!rocprofiler::tool::get_config().kernel_rename || thread_dispatch_rename == nullptr) return;

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API)
    {
        auto* marker_data =
            static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);

        if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA &&
           record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT && marker_data->args.roctxMarkA.message)
        {
            thread_dispatch_rename->emplace(
                common::add_string_entry(marker_data->args.roctxMarkA.message));
        }
        else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA &&
                record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
                marker_data->args.roctxRangePushA.message)
        {
            thread_dispatch_rename->emplace(
                common::add_string_entry(marker_data->args.roctxRangePushA.message));
        }
        else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop &&
                record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            ROCP_FATAL_IF(thread_dispatch_rename->empty())
                << "roctxRangePop invoked more times than roctxRangePush on thread "
                << rocprofiler::common::get_tid();

            thread_dispatch_rename->pop();
        }
    }

    common::consume_args(user_data, data);
}

void
callback_tracing_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*              user_data,
                          void*                                 data)
{
    static thread_local auto stacked_range =
        std::vector<rocprofiler_buffer_tracing_marker_api_record_t>{};
    static auto global_range = common::Synchronized<
        std::unordered_map<roctx_range_id_t, rocprofiler_buffer_tracing_marker_api_record_t>>{};

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
                CHECK_NOTNULL(marker_msg_data)
                    ->wlock(
                        [](auto& _data, uint64_t _cid_v, std::string&& _msg) {
                            _data.emplace(_cid_v, std::move(_msg));
                        },
                        record.correlation_id.internal,
                        std::string{marker_data->args.roctxMarkA.message});

                auto marker_record      = rocprofiler_buffer_tracing_marker_api_record_t{};
                marker_record.size      = sizeof(rocprofiler_buffer_tracing_marker_api_record_t);
                marker_record.kind      = convert_marker_tracing_kind(record.kind);
                marker_record.operation = record.operation;
                marker_record.thread_id = record.thread_id;
                marker_record.correlation_id  = record.correlation_id;
                marker_record.start_timestamp = ts;
                marker_record.end_timestamp   = ts;
                write_ring_buffer(marker_record, domain_type::MARKER);
            }
        }
        else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA)
        {
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
            {
                if(marker_data->args.roctxRangePushA.message)
                {
                    CHECK_NOTNULL(marker_msg_data)
                        ->wlock(
                            [](auto& _data, uint64_t _cid_v, std::string&& _msg) {
                                _data.emplace(_cid_v, std::move(_msg));
                            },
                            record.correlation_id.internal,
                            std::string{marker_data->args.roctxRangePushA.message});

                    auto marker_record = rocprofiler_buffer_tracing_marker_api_record_t{};
                    marker_record.size = sizeof(rocprofiler_buffer_tracing_marker_api_record_t);
                    marker_record.kind = convert_marker_tracing_kind(record.kind);
                    marker_record.operation       = record.operation;
                    marker_record.thread_id       = record.thread_id;
                    marker_record.correlation_id  = record.correlation_id;
                    marker_record.start_timestamp = ts;
                    marker_record.end_timestamp   = 0;

                    stacked_range.emplace_back(marker_record);
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

                val.end_timestamp = ts;
                write_ring_buffer(val, domain_type::MARKER);
            }
        }
        else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA)
        {
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
               marker_data->args.roctxRangeStartA.message)
            {
                CHECK_NOTNULL(marker_msg_data)
                    ->wlock(
                        [](auto& _data, uint64_t _cid_v, std::string&& _msg) {
                            _data.emplace(_cid_v, std::move(_msg));
                        },
                        record.correlation_id.internal,
                        std::string{marker_data->args.roctxRangeStartA.message});

                auto marker_record      = rocprofiler_buffer_tracing_marker_api_record_t{};
                marker_record.size      = sizeof(rocprofiler_buffer_tracing_marker_api_record_t);
                marker_record.kind      = convert_marker_tracing_kind(record.kind);
                marker_record.operation = record.operation;
                marker_record.thread_id = record.thread_id;
                marker_record.correlation_id  = record.correlation_id;
                marker_record.start_timestamp = ts;
                marker_record.end_timestamp   = 0;

                auto _id = marker_data->retval.roctx_range_id_t_retval;
                global_range.wlock(
                    [](auto& map, roctx_range_id_t _range_id, auto&& _record) {
                        map.emplace(_range_id, std::move(_record));
                    },
                    _id,
                    marker_record);
            }
        }
        else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop)
        {
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
            {
                auto   _id    = marker_data->args.roctxRangeStop.id;
                auto&& _entry = global_range.rlock(
                    [](const auto& map, auto _key) { return map.at(_key); }, _id);

                _entry.end_timestamp = ts;
                write_ring_buffer(_entry, domain_type::MARKER);
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
                auto marker_record      = rocprofiler_buffer_tracing_marker_api_record_t{};
                marker_record.size      = sizeof(rocprofiler_buffer_tracing_marker_api_record_t);
                marker_record.kind      = convert_marker_tracing_kind(record.kind);
                marker_record.operation = record.operation;
                marker_record.thread_id = record.thread_id;
                marker_record.correlation_id  = record.correlation_id;
                marker_record.start_timestamp = user_data->value;
                marker_record.end_timestamp   = ts;
                write_ring_buffer(marker_record, domain_type::MARKER);
            }
        }
    }

    (void) data;
}

void
code_object_tracing_callback(rocprofiler_callback_tracing_record_t record,
                             rocprofiler_user_data_t*              user_data,
                             void*                                 data)
{
    auto ts = rocprofiler_timestamp_t{};
    ROCPROFILER_CALL(rocprofiler_get_timestamp(&ts), "get timestamp");
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            auto* obj_data = static_cast<rocprofiler_code_object_data_t*>(record.payload);

            code_obj_data->wlock(
                [](code_object_data_map_t& cdata, rocprofiler_code_object_data_t* obj_data_v) {
                    cdata.emplace(obj_data_v->code_object_id, *obj_data_v);
                },
                CHECK_NOTNULL(obj_data));
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
            auto itr = kernel_data->wlock([sym_data](auto& _data) {
                return _data.emplace(sym_data->kernel_id,
                                     kernel_symbol_data{get_dereference(sym_data)});
            });

            ROCP_WARNING_IF(!itr.second)
                << "duplicate kernel symbol data for kernel_id=" << sym_data->kernel_id;

            // add the kernel to the kernel_targets if
            if(itr.second)
            {
                // if kernel name is provided by user then by default all kernels in the application
                // are targeted
                const auto& kernel_info           = itr.first->second;
                auto        kernel_filter_include = tool::get_config().kernel_filter_include;
                auto        kernel_filter_exclude = tool::get_config().kernel_filter_exclude;
                auto        kernel_filter_range   = tool::get_config().kernel_filter_range;

                std::regex include_regex(kernel_filter_include);
                std::regex exclude_regex(kernel_filter_exclude);
                if(std::regex_search(kernel_info.formatted_kernel_name, include_regex))
                {
                    if(kernel_filter_exclude.empty() ||
                       !std::regex_search(kernel_info.formatted_kernel_name, exclude_regex))
                        add_kernel_target(sym_data->kernel_id, kernel_filter_range);
                }
            }
        }
    }

    (void) user_data;
    (void) data;
}

std::string_view
get_kernel_name(uint64_t kernel_id, uint64_t rename_id)
{
    if(rename_id > 0)
    {
        if(const auto* _name = common::get_string_entry(rename_id)) return std::string_view{*_name};
    }

    return CHECK_NOTNULL(kernel_data)->rlock([kernel_id](const auto& _data) -> std::string_view {
        return _data.at(kernel_id).formatted_kernel_name;
    });
}

std::string_view
get_domain_name(rocprofiler_buffer_tracing_kind_t record_kind)
{
    return CHECK_NOTNULL(buffered_name_info)->at(record_kind);
}

uint64_t
get_agent_node_id(rocprofiler_agent_id_t agent_id)
{
    return agent_info->at(agent_id).logical_node_id;
}

std::string_view
get_operation_name(rocprofiler_buffer_tracing_kind_t kind, rocprofiler_tracing_operation_t op)
{
    return CHECK_NOTNULL(buffered_name_info)->at(kind, op);
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

                write_ring_buffer(*record, domain_type::KERNEL_DISPATCH);
            }

            else if(header->kind == ROCPROFILER_BUFFER_TRACING_HSA_CORE_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(header->payload);

                write_ring_buffer(*record, domain_type::HSA);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload);

                write_ring_buffer(*record, domain_type::MEMORY_COPY);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_scratch_memory_record_t*>(
                    header->payload);

                write_ring_buffer(*record, domain_type::SCRATCH_MEMORY);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);

                write_ring_buffer(*record, domain_type::HIP);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_RCCL_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_rccl_api_record_t*>(header->payload);

                write_ring_buffer(*record, domain_type::RCCL);
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
        dimensions_info->reserve(num_dims);
        for(size_t j = 0; j < num_dims; j++)
            dimensions_info->emplace_back(dim_info[j]);
    }
    else
    {
        counter_dimension_data.wlock(
            [&id, &dim_info, &num_dims](counter_dimension_info_map_t& counter_dimension_data_v) {
                if(counter_dimension_data_v.find(id.handle) == counter_dimension_data_v.end())
                {
                    auto dimensions = std::vector<rocprofiler_record_dimension_info_t>{};
                    dimensions.reserve(num_dims);
                    for(size_t dim = 0; dim < num_dims; ++dim)
                        dimensions.emplace_back(dim_info[dim]);
                    counter_dimension_data_v.emplace(id.handle, std::move(dimensions));
                }
            });
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

struct tool_agent
{
    int64_t                       device_id = 0;
    const rocprofiler_agent_v0_t* agent     = nullptr;
};

using tool_agent_vec_t = std::vector<tool_agent>;

auto
get_gpu_agents()
{
    auto _gpu_agents = tool_agent_vec_t{};

    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(
            ROCPROFILER_AGENT_INFO_VERSION_0,
            [](rocprofiler_agent_version_t, const void** agents, size_t num_agents, void* _data) {
                auto* _gpu_agents_v = static_cast<tool_agent_vec_t*>(_data);
                for(size_t i = 0; i < num_agents; ++i)
                {
                    auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
                    if(agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                        _gpu_agents_v->emplace_back(tool_agent{0, agent});
                }
                return ROCPROFILER_STATUS_SUCCESS;
            },
            sizeof(rocprofiler_agent_t),
            &_gpu_agents),
        "Iterate rocporfiler agents")

    // make sure they are sorted by node id
    std::sort(_gpu_agents.begin(), _gpu_agents.end(), [](const auto& lhs, const auto& rhs) {
        return CHECK_NOTNULL(lhs.agent)->node_id < CHECK_NOTNULL(rhs.agent)->node_id;
    });

    int64_t _dev_id = 0;
    for(auto& itr : _gpu_agents)
        itr.device_id = _dev_id++;

    return _gpu_agents;
}

auto
get_agent_counter_info(const tool_agent_vec_t& _agents)
{
    using value_type =
        std::unordered_map<rocprofiler_agent_id_t, std::vector<rocprofiler_tool_counter_info_t>>;

    auto _data = value_type{};

    for(auto itr : _agents)
    {
        ROCPROFILER_CALL(
            rocprofiler_iterate_agent_supported_counters(
                itr.agent->id,
                [](rocprofiler_agent_id_t    id,
                   rocprofiler_counter_id_t* counters,
                   size_t                    num_counters,
                   void*                     user_data) {
                    auto* data_v = static_cast<value_type*>(user_data);
                    for(size_t i = 0; i < num_counters; ++i)
                    {
                        // populate global map
                        ROCPROFILER_CALL(rocprofiler_iterate_counter_dimensions(
                                             counters[i], dimensions_info_callback, nullptr),
                                         "iterate_dimension_info");

                        auto _info     = rocprofiler_counter_info_v0_t{};
                        auto _dim_ids  = std::vector<rocprofiler_counter_dimension_id_t>{};
                        auto _dim_info = std::vector<rocprofiler_record_dimension_info_t>{};

                        ROCPROFILER_CALL(
                            rocprofiler_query_counter_info(
                                counters[i], ROCPROFILER_COUNTER_INFO_VERSION_0, &_info),
                            "Could not query counter_id");

                        // populate local vector
                        ROCPROFILER_CALL(rocprofiler_iterate_counter_dimensions(
                                             counters[i], dimensions_info_callback, &_dim_info),
                                         "iterate_dimension_info");

                        _dim_ids.reserve(_dim_info.size());
                        for(auto ditr : _dim_info)
                            _dim_ids.emplace_back(ditr.id);

                        (*data_v)[id].emplace_back(
                            id, _info, std::move(_dim_ids), std::move(_dim_info));
                    }
                    return ROCPROFILER_STATUS_SUCCESS;
                },
                &_data),
            "iterate agent supported counters");

        // Skip unsupported agents
        if(_data.find(itr.agent->id) == _data.end()) continue;

        std::sort(_data.at(itr.agent->id).begin(),
                  _data.at(itr.agent->id).end(),
                  [](const auto& lhs, const auto& rhs) { return (lhs.id.handle < rhs.id.handle); });

        for(auto& citr : _data.at(itr.agent->id))
        {
            std::sort(citr.dimension_ids.begin(), citr.dimension_ids.end());
            std::sort(citr.dimension_info.begin(),
                      citr.dimension_info.end(),
                      [](const auto& lhs, const auto& rhs) { return (lhs.id < rhs.id); });
        }
    }

    return _data;
}

const tool_agent*
get_tool_agent(rocprofiler_agent_id_t id, const tool_agent_vec_t& data)
{
    for(const auto& itr : data)
    {
        if(id == itr.agent->id) return &itr;
    }

    return nullptr;
}

// this function creates a rocprofiler profile config on the first entry
auto
get_device_counting_service(rocprofiler_agent_id_t agent_id)
{
    static auto       data                    = common::Synchronized<agent_counter_map_t>{};
    static const auto gpu_agents              = get_gpu_agents();
    static const auto gpu_agents_counter_info = get_agent_counter_info(gpu_agents);

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
            auto        counters_v   = counter_vec_t{};
            auto        found_v      = std::vector<std::string_view>{};
            const auto* tool_agent_v = get_tool_agent(agent_id, gpu_agents);
            auto        expected_v   = tool::get_config().counters.size();

            constexpr auto device_qualifier = std::string_view{":device="};
            for(const auto& itr : tool::get_config().counters)
            {
                auto name_v = itr;
                if(auto pos = std::string::npos;
                   (pos = itr.find(device_qualifier)) != std::string::npos)
                {
                    name_v        = itr.substr(0, pos);
                    auto dev_id_s = itr.substr(pos + device_qualifier.length());

                    LOG_IF(FATAL,
                           dev_id_s.empty() ||
                               dev_id_s.find_first_not_of("0123456789") != std::string::npos)
                        << "invalid device qualifier format (':device=N) where N is the GPU id: "
                        << itr;

                    auto dev_id_v = std::stol(dev_id_s);
                    // skip this counter if the counter is for a specific device id (which doesn't
                    // this agent's device id)
                    if(dev_id_v != tool_agent_v->device_id)
                    {
                        --expected_v;  // is not expected
                        continue;
                    }
                }

                // search the gpu agent counter info for a counter with a matching name
                for(const auto& citr : gpu_agents_counter_info.at(agent_id))
                {
                    if(name_v == std::string_view{citr.name})
                    {
                        counters_v.emplace_back(citr.id);
                        found_v.emplace_back(itr);
                    }
                }
            }

            if(expected_v != counters_v.size())
            {
                auto requested_counters = fmt::format("{}",
                                                      fmt::join(tool::get_config().counters.begin(),
                                                                tool::get_config().counters.end(),
                                                                ", "));
                auto found_counters =
                    fmt::format("{}", fmt::join(found_v.begin(), found_v.end(), ", "));
                LOG(FATAL) << "Unable to find all counters for agent "
                           << tool_agent_v->agent->node_id << " (gpu-" << tool_agent_v->device_id
                           << ", " << tool_agent_v->agent->name << ") in [" << requested_counters
                           << "]. Found: [" << found_counters << "]";
            }

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

void
dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                  rocprofiler_profile_config_id_t*             config,
                  rocprofiler_user_data_t*                     user_data,
                  void* /*callback_data_args*/)
{
    auto kernel_id = dispatch_data.dispatch_info.kernel_id;
    auto agent_id  = dispatch_data.dispatch_info.agent_id;

    kernel_iteration.wlock(
        [](auto& _kernel_iter, rocprofiler_kernel_id_t _kernel_id) {
            auto itr = _kernel_iter.find(_kernel_id);
            if(itr == _kernel_iter.end())
                _kernel_iter.emplace(_kernel_id, 1);
            else
            {
                itr->second++;
            }
        },
        kernel_id);

    if(!is_targeted_kernel(kernel_id))
    {
        return;
    }
    else if(auto profile = get_device_counting_service(agent_id))
    {
        *config          = *profile;
        user_data->value = common::get_tid();
    }
}

std::string
get_counter_info_name(uint64_t record_id)
{
    auto info       = rocprofiler_counter_info_v0_t{};
    auto counter_id = rocprofiler_counter_id_t{};
    ROCPROFILER_CALL(rocprofiler_query_record_counter_id(record_id, &counter_id),
                     "query record counter id");
    if(rocprofiler_query_counter_info(rocprofiler_counter_id_t{counter_id},
                                      ROCPROFILER_COUNTER_INFO_VERSION_0,
                                      static_cast<void*>(&info)) != ROCPROFILER_STATUS_SUCCESS)
    {
        ROCP_FATAL << "Could not find name for record id: " << record_id;
    }
    return {info.name};
}

void
counter_record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                        rocprofiler_record_counter_t*                record_data,
                        size_t                                       record_count,
                        rocprofiler_user_data_t                      user_data,
                        void* /*callback_data_args*/)
{
    static const auto gpu_agents              = get_gpu_agents();
    static const auto gpu_agents_counter_info = get_agent_counter_info(gpu_agents);

    auto counter_record = rocprofiler_tool_counter_collection_record_t{};
    auto kernel_id      = dispatch_data.dispatch_info.kernel_id;

    counter_record.dispatch_data = dispatch_data;
    counter_record.thread_id     = user_data.value;

    const kernel_symbol_data* kernel_info =
        kernel_data->rlock([kernel_id](const auto& _data) { return &_data.at(kernel_id); });

    auto lds_block_size_v =
        (kernel_info->group_segment_size + (lds_block_size - 1)) & ~(lds_block_size - 1);

    counter_record.arch_vgpr_count  = kernel_info->arch_vgpr_count;
    counter_record.sgpr_count       = kernel_info->sgpr_count;
    counter_record.lds_block_size_v = lds_block_size_v;

    ROCP_FATAL_IF(!kernel_info) << "missing kernel information for kernel_id=" << kernel_id;

    ROCP_ERROR_IF(record_count == 0) << "zero record count for kernel_id=" << kernel_id
                                     << " (name=" << kernel_info->kernel_name << ")";

    for(size_t count = 0; count < record_count; count++)
    {
        // Unlikely to trigger, temporary until we move to buffered callbacks
        if(count >= counter_record.records.size())
        {
            ROCP_WARNING << "Exceeded maximum counter capacity, skipping remaining";
            break;
        }

        auto _counter_id = rocprofiler_counter_id_t{};
        ROCPROFILER_CALL(rocprofiler_query_record_counter_id(record_data[count].id, &_counter_id),
                         "query record counter id");
        counter_record.records[count] =
            rocprofiler_tool_record_counter_t{_counter_id, record_data[count]};
        counter_record.counter_count++;
    }

    write_ring_buffer(counter_record, domain_type::COUNTER_COLLECTION);
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
        if(agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;

        auto status = rocprofiler_iterate_agent_supported_counters(
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
                        rocprofiler_iterate_counter_dimensions(
                            counters[i], dimensions_info_callback, static_cast<void*>(&dimensions)),
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
                            get_dereference(get_list_basic_metrics_file()) << counter_info_ss.str();
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
                                            << "Description: " << counter_info.description << "\n";
                            counter_info_ss << "Expression: " << counter_info.expression << "\n";
                            counter_info_ss << "Dimensions: " << dimensions_info.str() << "\n";
                            counter_info_ss << "\n";
                            std::cout << counter_info_ss.str();
                        }
                    }
                }
                return ROCPROFILER_STATUS_SUCCESS;
            },
            reinterpret_cast<void*>(&node_id));

        ROCP_ERROR_IF(status != ROCPROFILER_STATUS_SUCCESS)
            << "Failed to iterate counters for agent " << node_id << " (" << agent->name << ")";
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_client_finalize_t client_finalizer  = nullptr;
rocprofiler_client_id_t*      client_identifier = nullptr;

void
initialize_logging()
{
    auto logging_cfg = rocprofiler::common::logging_config{.install_failure_handler = true};
    common::init_logging("ROCPROF", logging_cfg);
    FLAGS_colorlogtostderr = true;
}

void
initialize_rocprofv3()
{
    ROCP_INFO << "initializing rocprofv3...";
    if(int status = 0;
       rocprofiler_is_initialized(&status) == ROCPROFILER_STATUS_SUCCESS && status == 0)
    {
        ROCPROFILER_CALL(rocprofiler_force_configure(&rocprofiler_configure),
                         "force configuration");
    }

    LOG_IF(FATAL, !client_identifier) << "nullptr to client identifier!";
    LOG_IF(FATAL, !client_finalizer && !tool::get_config().list_metrics)
        << "nullptr to client finalizer!";  // exception for listing metrics
}

void
finalize_rocprofv3()
{
    ROCP_INFO << "finalizing rocprofv3...";
    if(client_finalizer && client_identifier)
    {
        client_finalizer(*client_identifier);
        client_finalizer  = nullptr;
        client_identifier = nullptr;
    }
}

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

    constexpr uint64_t buffer_size      = 32 * common::units::KiB;
    constexpr uint64_t buffer_watermark = 31 * common::units::KiB;

    rocprofiler_get_timestamp(&(stats_timestamp->app_start_time));

    init_tool_table();

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "create context failed");

    auto code_obj_ctx = rocprofiler_context_id_t{0};
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

        auto pause_resume_ctx = rocprofiler_context_id_t{0};
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

    if(tool::get_config().scratch_memory_trace)
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

    if(tool::get_config().rccl_api_trace)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                                   buffer_size,
                                                   buffer_watermark,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   buffered_tracing_callback,
                                                   tool_data,
                                                   &get_buffers().rccl_api_trace),
                         "buffer creation");

        ROCPROFILER_CALL(
            rocprofiler_configure_buffer_tracing_service(get_client_ctx(),
                                                         ROCPROFILER_BUFFER_TRACING_RCCL_API,
                                                         nullptr,
                                                         0,
                                                         get_buffers().rccl_api_trace),
            "buffer tracing service for rccl api configure");
    }

    if(tool::get_config().counter_collection)
    {
        ROCPROFILER_CALL(
            rocprofiler_configure_callback_dispatch_counting_service(
                get_client_ctx(), dispatch_callback, nullptr, counter_record_callback, nullptr),
            "Could not setup counting service");
    }

    if(tool::get_config().kernel_rename)
    {
        auto rename_ctx            = rocprofiler_context_id_t{0};
        auto marker_core_api_kinds = std::array<rocprofiler_tracing_operation_t, 3>{
            ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA,
            ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
            ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop};

        ROCPROFILER_CALL(rocprofiler_create_context(&rename_ctx), "failed to create context");

        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                             rename_ctx,
                             ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                             marker_core_api_kinds.data(),
                             marker_core_api_kinds.size(),
                             kernel_rename_callback,
                             nullptr),
                         "callback tracing service failed to configure");

        ROCPROFILER_CALL(rocprofiler_start_context(rename_ctx), "start context failed");

        auto external_corr_id_request_kinds =
            std::array<rocprofiler_external_correlation_id_request_kind_t, 1>{
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH};

        ROCPROFILER_CALL(rocprofiler_configure_external_correlation_id_request_service(
                             get_client_ctx(),
                             external_corr_id_request_kinds.data(),
                             external_corr_id_request_kinds.size(),
                             set_kernel_rename_correlation_id,
                             nullptr),
                         "Could not configure external correlation id request service");
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

using stats_data_t       = ::rocprofiler::tool::stats_data_t;
using stats_entry_t      = ::rocprofiler::tool::stats_entry_t;
using domain_stats_vec_t = ::rocprofiler::tool::domain_stats_vec_t;

template <typename Tp, domain_type DomainT>
void
generate_output(rocprofiler::tool::buffered_output<Tp, DomainT>& output_v,
                domain_stats_vec_t&                              contributions_v)
{
    if(!output_v) return;

    output_v.read();

    if(tool::get_config().stats || tool::get_config().summary_output)
    {
        output_v.stats = rocprofiler::tool::generate_stats(tool_functions, output_v.element_data);
    }

    if(output_v.stats)
    {
        contributions_v.emplace_back(output_v.buffer_type_v, output_v.stats);
    }

    if(tool::get_config().csv_output)
    {
        rocprofiler::tool::generate_csv(tool_functions, output_v.element_data, output_v.stats);
    }
}

void
tool_fini(void* /*tool_data*/)
{
    client_identifier = nullptr;
    client_finalizer  = nullptr;

    rocprofiler_get_timestamp(&(stats_timestamp->app_end_time));

    flush();
    rocprofiler_stop_context(get_client_ctx());
    flush();

    auto kernel_dispatch_output =
        kernel_dispatch_buffered_output_t{tool::get_config().kernel_trace};
    auto hsa_output         = hsa_buffered_output_t{tool::get_config().hsa_core_api_trace ||
                                            tool::get_config().hsa_amd_ext_api_trace ||
                                            tool::get_config().hsa_image_ext_api_trace ||
                                            tool::get_config().hsa_finalizer_ext_api_trace};
    auto hip_output         = hip_buffered_output_t{tool::get_config().hip_runtime_api_trace ||
                                            tool::get_config().hip_compiler_api_trace};
    auto memory_copy_output = memory_copy_buffered_output_t{tool::get_config().memory_copy_trace};
    auto marker_output      = marker_buffered_output_t{tool::get_config().marker_api_trace};
    auto counters_output =
        counter_collection_buffered_output_t{tool::get_config().counter_collection};
    auto scratch_memory_output =
        scratch_memory_buffered_output_t{tool::get_config().scratch_memory_trace};
    auto rccl_output = rccl_buffered_output_t{tool::get_config().rccl_api_trace};

    auto node_id_sort = [](const auto& lhs, const auto& rhs) { return lhs.node_id < rhs.node_id; };

    auto _agents = std::vector<rocprofiler_agent_v0_t>{};
    _agents.reserve(agent_info->size());
    for(auto& itr : *agent_info)
        _agents.emplace_back(itr.second);

    std::sort(_agents.begin(), _agents.end(), node_id_sort);

    if(tool::get_config().csv_output)
    {
        rocprofiler::tool::generate_csv(tool_functions, _agents);
    }

    auto contributions = domain_stats_vec_t{};

    generate_output(kernel_dispatch_output, contributions);
    generate_output(hsa_output, contributions);
    generate_output(hip_output, contributions);
    generate_output(memory_copy_output, contributions);
    generate_output(marker_output, contributions);
    generate_output(rccl_output, contributions);
    generate_output(counters_output, contributions);
    generate_output(scratch_memory_output, contributions);

    if(tool::get_config().stats && tool::get_config().csv_output)
    {
        rocprofiler::tool::generate_csv(tool_functions, contributions);
    }

    if(tool::get_config().json_output)
    {
        auto _counters = get_tool_counter_info();
        rocprofiler::tool::write_json(tool_functions,
                                      getpid(),
                                      contributions,
                                      _agents,
                                      _counters,
                                      &hip_output.element_data,
                                      &hsa_output.element_data,
                                      &kernel_dispatch_output.element_data,
                                      &memory_copy_output.element_data,
                                      &counters_output.element_data,
                                      &marker_output.element_data,
                                      &scratch_memory_output.element_data,
                                      &rccl_output.element_data);
    }

    if(tool::get_config().pftrace_output)
    {
        rocprofiler::tool::write_perfetto(tool_functions,
                                          getpid(),
                                          _agents,
                                          &hip_output.element_data,
                                          &hsa_output.element_data,
                                          &kernel_dispatch_output.element_data,
                                          &memory_copy_output.element_data,
                                          &marker_output.element_data,
                                          &scratch_memory_output.element_data,
                                          &rccl_output.element_data);
    }

    if(tool::get_config().otf2_output)
    {
        rocprofiler::tool::write_otf2(tool_functions,
                                      getpid(),
                                      _agents,
                                      &hip_output.element_data,
                                      &hsa_output.element_data,
                                      &kernel_dispatch_output.element_data,
                                      &memory_copy_output.element_data,
                                      &marker_output.element_data,
                                      &scratch_memory_output.element_data,
                                      &rccl_output.element_data);
    }

    if(tool::get_config().summary_output)
    {
        rocprofiler::tool::generate_stats(tool_functions, contributions);
    }

    auto destroy_output = [](auto& _buffered_output_v) { _buffered_output_v.destroy(); };

    destroy_output(kernel_dispatch_output);
    destroy_output(hsa_output);
    destroy_output(hip_output);
    destroy_output(memory_copy_output);
    destroy_output(marker_output);
    destroy_output(counters_output);
    destroy_output(scratch_memory_output);
    destroy_output(rccl_output);

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

std::map<uint64_t, std::string>
get_callback_roctx_msg()
{
    auto _data = marker_msg_data->rlock([](const auto& _data_v) { return _data_v; });
    auto _ret  = std::map<uint64_t, std::string>{};
    for(const auto& itr : _data)
        _ret.emplace(itr.first, itr.second);
    return _ret;
}

std::vector<kernel_symbol_data>
get_kernel_symbol_data()
{
    auto _data = kernel_data->rlock([](const auto& _data_v) {
        auto _info = std::vector<kernel_symbol_data>{};
        _info.reserve(_data_v.size());
        for(const auto& itr : _data_v)
            _info.emplace_back(itr.second);
        return _info;
    });

    uint64_t kernel_data_size = 0;
    for(const auto& itr : _data)
        kernel_data_size = std::max(kernel_data_size, itr.kernel_id);

    auto _symbol_data = std::vector<kernel_symbol_data>{};
    _symbol_data.resize(kernel_data_size + 1, kernel_symbol_data{});
    // index by the kernel id
    for(auto& itr : _data)
        _symbol_data.at(itr.kernel_id) = std::move(itr);

    return _symbol_data;
}

std::vector<rocprofiler_code_object_data_t>
get_code_object_data()
{
    auto _data = code_obj_data->rlock([](const auto& _data_v) {
        auto _info = std::vector<rocprofiler_code_object_data_t>{};
        _info.reserve(_data_v.size());
        for(const auto& itr : _data_v)
            _info.emplace_back(itr.second);
        return _info;
    });

    uint64_t _sz = 0;
    for(const auto& itr : _data)
        _sz = std::max(_sz, itr.code_object_id);

    auto _code_obj_data = std::vector<rocprofiler_code_object_data_t>{};
    _code_obj_data.resize(_sz + 1, rocprofiler_code_object_data_t{});
    // index by the code object id
    for(auto& itr : _data)
        _code_obj_data.at(itr.code_object_id) = itr;

    return _code_obj_data;
}

std::vector<rocprofiler_tool_counter_info_t>
get_tool_counter_info()
{
    auto _data = get_agent_counter_info(get_gpu_agents());
    auto _ret  = std::vector<rocprofiler_tool_counter_info_t>{};
    for(const auto& itr : _data)
    {
        for(const auto& iitr : itr.second)
            _ret.emplace_back(iitr);
    }
    return _ret;
}

std::vector<rocprofiler_record_dimension_info_t>
get_tool_counter_dimension_info()
{
    auto _data = get_agent_counter_info(get_gpu_agents());
    auto _ret  = std::vector<rocprofiler_record_dimension_info_t>{};
    for(const auto& itr : _data)
    {
        for(const auto& iitr : itr.second)
            for(const auto& ditr : iitr.dimension_info)
                _ret.emplace_back(ditr);
    }

    auto _sorter = [](const rocprofiler_record_dimension_info_t& lhs,
                      const rocprofiler_record_dimension_info_t& rhs) {
        return std::tie(lhs.id, lhs.instance_size) < std::tie(rhs.id, rhs.instance_size);
    };
    auto _equiv = [](const rocprofiler_record_dimension_info_t& lhs,
                     const rocprofiler_record_dimension_info_t& rhs) {
        return std::tie(lhs.id, lhs.instance_size) == std::tie(rhs.id, rhs.instance_size);
    };

    std::sort(_ret.begin(), _ret.end(), _sorter);
    _ret.erase(std::unique(_ret.begin(), _ret.end(), _equiv), _ret.end());

    return _ret;
}

namespace
{
using main_func_t = int (*)(int, char**, char**);

main_func_t&
get_main_function()
{
    static main_func_t user_main = nullptr;
    return user_main;
}

bool signal_handler_exit =
    rocprofiler::tool::get_env("ROCPROF_INTERNAL_TEST_SIGNAL_HANDLER_VIA_EXIT", false);
}  // namespace

#define ROCPROFV3_INTERNAL_API __attribute__((visibility("internal")));

extern "C" {
void
rocprofv3_set_main(main_func_t main_func) ROCPROFV3_INTERNAL_API;

void
rocprofv3_error_signal_handler(int signo)
{
    finalize_rocprofv3();
    // below is for testing purposes. re-raising the signal causes CTest to ignore WILL_FAIL ON
    if(signal_handler_exit) ::exit(signo);
    ::raise(signo);
}

int
rocprofv3_main(int argc, char** argv, char** envp) ROCPROFV3_INTERNAL_API;

rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    initialize_logging();

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

    // ensure these pointers are not leaked
    add_destructor(buffered_name_info);
    add_destructor(callback_name_info);
    add_destructor(marker_msg_data);
    add_destructor(code_obj_data);
    add_destructor(kernel_data);
    add_destructor(tool_functions);
    add_destructor(agent_info);
    add_destructor(stats_timestamp);

    // in case main wrapper is not used
    ::atexit(finalize_rocprofv3);

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

void
rocprofv3_set_main(main_func_t main_func)
{
    get_main_function() = main_func;
}

int
rocprofv3_main(int argc, char** argv, char** envp)
{
    initialize_logging();

    initialize_rocprofv3();

    struct sigaction sig_act = {};
    sigemptyset(&sig_act.sa_mask);
    sig_act.sa_flags   = SA_RESETHAND | SA_NODEFER;
    sig_act.sa_handler = &rocprofv3_error_signal_handler;
    for(auto signal_v : {SIGTERM, SIGSEGV, SIGINT, SIGILL, SIGABRT, SIGFPE})
    {
        if(sigaction(signal_v, &sig_act, nullptr) != 0)
        {
            auto _errno_v = errno;
            ROCP_ERROR << "error setting signal handler for " << signal_v
                       << " :: " << strerror(_errno_v);
        }
    }

    auto ret = CHECK_NOTNULL(get_main_function())(argc, argv, envp);

    finalize_rocprofv3();

    ROCP_INFO << "rocprofv3 finished. exit code: " << ret;
    return ret;
}
}
