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
#include "helper.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/units.hpp"
#include "lib/common/utility.hpp"
#include "lib/output/buffered_output.hpp"
#include "lib/output/counter_info.hpp"
#include "lib/output/csv.hpp"
#include "lib/output/csv_output_file.hpp"
#include "lib/output/domain_type.hpp"
#include "lib/output/generateCSV.hpp"
#include "lib/output/generateJSON.hpp"
#include "lib/output/generateOTF2.hpp"
#include "lib/output/generatePerfetto.hpp"
#include "lib/output/generateStats.hpp"
#include "lib/output/output_stream.hpp"
#include "lib/output/statistics.hpp"
#include "lib/output/tmp_file.hpp"
#include "lib/output/tmp_file_buffer.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/counters.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/core.h>

#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
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

#undef ADD_DESTRUCTOR

struct buffer_ids
{
    rocprofiler_buffer_id_t hsa_api_trace           = {};
    rocprofiler_buffer_id_t hip_api_trace           = {};
    rocprofiler_buffer_id_t kernel_trace            = {};
    rocprofiler_buffer_id_t memory_copy_trace       = {};
    rocprofiler_buffer_id_t memory_allocation_trace = {};
    rocprofiler_buffer_id_t counter_collection      = {};
    rocprofiler_buffer_id_t scratch_memory          = {};
    rocprofiler_buffer_id_t rccl_api_trace          = {};
    rocprofiler_buffer_id_t pc_sampling_host_trap   = {};

    auto as_array() const
    {
        return std::array<rocprofiler_buffer_id_t, 9>{hsa_api_trace,
                                                      hip_api_trace,
                                                      kernel_trace,
                                                      memory_copy_trace,
                                                      memory_allocation_trace,
                                                      counter_collection,
                                                      scratch_memory,
                                                      rccl_api_trace,
                                                      pc_sampling_host_trap};
    }
};

buffer_ids&
get_buffers()
{
    static auto _v = buffer_ids{};
    return _v;
}

template <typename Tp>
Tp*
as_pointer(Tp&& _val)
{
    return new Tp{std::forward<Tp>(_val)};
}

template <typename Tp, typename... Args>
Tp*
as_pointer(Args&&... _args)
{
    return new Tp{std::forward<Args>(_args)...};
}

template <typename Tp>
Tp*
as_pointer()
{
    return new Tp{};
}

using targeted_kernels_map_t =
    std::unordered_map<rocprofiler_kernel_id_t, std::unordered_set<uint32_t>>;
using counter_dimension_info_map_t =
    std::unordered_map<uint64_t, std::vector<rocprofiler_record_dimension_info_t>>;
using agent_info_map_t      = std::unordered_map<rocprofiler_agent_id_t, rocprofiler_agent_t>;
using kernel_iteration_t    = std::unordered_map<rocprofiler_kernel_id_t, uint32_t>;
using kernel_rename_map_t   = std::unordered_map<uint64_t, uint64_t>;
using kernel_rename_stack_t = std::stack<uint64_t>;

auto* tool_metadata    = as_pointer<tool::metadata>(tool::metadata::inprocess{});
auto  target_kernels   = common::Synchronized<targeted_kernels_map_t>{};
auto  kernel_iteration = common::Synchronized<kernel_iteration_t, true>{};

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

void
collection_period_cntrl(std::promise<void>&& _promise, rocprofiler_context_id_t _ctx)
{
    bool testing_cp          = tool::get_env("ROCPROF_COLLECTION_PERIOD_TESTING", false);
    auto log_fname           = get_output_filename(tool::get_config(), "collection_periods", "log");
    auto output_testing_file = std::ofstream{};

    if(testing_cp)
    {
        ROCP_INFO << "collection period test logging enabled: " << log_fname;
        output_testing_file.open(log_fname);
    }

    auto log_period = [testing_cp, &output_testing_file](
                          std::string_view label, auto _func, auto... _args) {
        ROCP_INFO << "collection period: " << label;

        auto beg = rocprofiler_timestamp_t{};
        if(testing_cp)
        {
            rocprofiler_get_timestamp(&beg);
        }

        _func(_args...);

        if(testing_cp)
        {
            auto end = rocprofiler_timestamp_t{};
            rocprofiler_get_timestamp(&end);
            output_testing_file << label << ":" << beg << ":" << end << '\n' << std::flush;
        }
    };

    auto sleep_for_nsec = [](auto _value) {
        if(_value > 0)
        {
            std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::nanoseconds{_value});
        }
    };

    auto periods = tool::get_config().collection_periods;
    _promise.set_value();  // allow the launching thread to proceed
    while(!periods.empty())
    {
        auto _period = periods.front();
        periods.pop();

        auto execute_period = [&]() {
            if(testing_cp) output_testing_file << "--" << '\n';

            log_period("delay", sleep_for_nsec, _period.delay);
            log_period("start", rocprofiler_start_context, _ctx);
            log_period("duration", sleep_for_nsec, _period.duration);
            log_period("stop", rocprofiler_stop_context, _ctx);
        };

        if(_period.repeat == 0)
        {
            execute_period();
        }
        else
        {
            for(size_t i = 0; i < _period.repeat; ++i)
            {
                execute_period();
            }
        }
    }
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
    {
        auto val = thread_dispatch_rename->top();
        if(tool_metadata) tool_metadata->add_external_correlation_id(val);
        external_corr_id->value = val;
    }

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
            tool::write_ring_buffer(marker_record, domain_type::MARKER);
        }
    }
}

void
kernel_rename_callback(rocprofiler_callback_tracing_record_t record,
                       rocprofiler_user_data_t*              user_data,
                       void*                                 data)
{
    if(!tool::get_config().kernel_rename || thread_dispatch_rename == nullptr) return;

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
                CHECK_NOTNULL(tool_metadata)
                    ->add_marker_message(record.correlation_id.internal,
                                         std::string{marker_data->args.roctxMarkA.message});

                auto marker_record      = rocprofiler_buffer_tracing_marker_api_record_t{};
                marker_record.size      = sizeof(rocprofiler_buffer_tracing_marker_api_record_t);
                marker_record.kind      = convert_marker_tracing_kind(record.kind);
                marker_record.operation = record.operation;
                marker_record.thread_id = record.thread_id;
                marker_record.correlation_id  = record.correlation_id;
                marker_record.start_timestamp = ts;
                marker_record.end_timestamp   = ts;
                tool::write_ring_buffer(marker_record, domain_type::MARKER);
            }
        }
        else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA)
        {
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
            {
                if(marker_data->args.roctxRangePushA.message)
                {
                    CHECK_NOTNULL(tool_metadata)
                        ->add_marker_message(
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
                tool::write_ring_buffer(val, domain_type::MARKER);
            }
        }
        else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA)
        {
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
               marker_data->args.roctxRangeStartA.message)
            {
                CHECK_NOTNULL(tool_metadata)
                    ->add_marker_message(record.correlation_id.internal,
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
                tool::write_ring_buffer(_entry, domain_type::MARKER);
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
                tool::write_ring_buffer(marker_record, domain_type::MARKER);
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
            auto* obj_data = static_cast<tool::rocprofiler_code_object_info_t*>(record.payload);

            CHECK_NOTNULL(tool_metadata)->add_code_object(*obj_data);
            if(tool::get_config().pc_sampling_host_trap)
            {
                CHECK_NOTNULL(tool_metadata)->add_decoder(obj_data);
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
        auto* sym_data = static_cast<tool::rocprofiler_kernel_symbol_info_t*>(record.payload);
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            auto success = CHECK_NOTNULL(tool_metadata)
                               ->add_kernel_symbol(kernel_symbol_info{
                                   get_dereference(sym_data),
                                   [](const char* val) { return tool::format_name(val); }});

            ROCP_WARNING_IF(!success)
                << "duplicate kernel symbol data for kernel_id=" << sym_data->kernel_id;

            // add the kernel to the kernel_targets if
            if(success)
            {
                // if kernel name is provided by user then by default all kernels in the
                // application are targeted
                const auto* kernel_info =
                    CHECK_NOTNULL(tool_metadata)->get_kernel_symbol(sym_data->kernel_id);
                auto kernel_filter_include = tool::get_config().kernel_filter_include;
                auto kernel_filter_exclude = tool::get_config().kernel_filter_exclude;
                auto kernel_filter_range   = tool::get_config().kernel_filter_range;

                std::regex include_regex(kernel_filter_include);
                std::regex exclude_regex(kernel_filter_exclude);
                if(std::regex_search(kernel_info->formatted_kernel_name, include_regex))
                {
                    if(kernel_filter_exclude.empty() ||
                       !std::regex_search(kernel_info->formatted_kernel_name, exclude_regex))
                        add_kernel_target(sym_data->kernel_id, kernel_filter_range);
                }
            }
        }
    }

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_HOST_KERNEL_SYMBOL_REGISTER)
    {
        auto* hst_data = static_cast<rocprofiler_host_kernel_symbol_data_t*>(record.payload);
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            auto success = CHECK_NOTNULL(tool_metadata)
                               ->add_host_function(host_function_info{
                                   get_dereference(hst_data),
                                   [](const char* val) { return tool::format_name(val); }});
            ROCP_WARNING_IF(!success)
                << "duplicate host function found for kernel_id=" << hst_data->kernel_id;

            // TODO : kernel filtering for host functions?!
        }
    }

    (void) user_data;
    (void) data;
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

                tool::write_ring_buffer(*record, domain_type::KERNEL_DISPATCH);
            }

            else if(header->kind == ROCPROFILER_BUFFER_TRACING_HSA_CORE_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(header->payload);

                tool::write_ring_buffer(*record, domain_type::HSA);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload);

                tool::write_ring_buffer(*record, domain_type::MEMORY_COPY);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_memory_allocation_record_t*>(
                    header->payload);

                tool::write_ring_buffer(*record, domain_type::MEMORY_ALLOCATION);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_scratch_memory_record_t*>(
                    header->payload);

                tool::write_ring_buffer(*record, domain_type::SCRATCH_MEMORY);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);

                tool::write_ring_buffer(*record, domain_type::HIP);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_RCCL_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_rccl_api_record_t*>(header->payload);

                tool::write_ring_buffer(*record, domain_type::RCCL);
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

auto
get_gpu_agents()
{
    return CHECK_NOTNULL(tool_metadata)->get_gpu_agents();
}

auto
get_agent_counter_info()
{
    return CHECK_NOTNULL(tool_metadata)->agent_counter_info;
}

// this function creates a rocprofiler profile config on the first entry
auto
get_device_counting_service(rocprofiler_agent_id_t agent_id)
{
    static auto       data                    = common::Synchronized<agent_counter_map_t>{};
    static const auto gpu_agents              = get_gpu_agents();
    static const auto gpu_agents_counter_info = get_agent_counter_info();

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
            auto        counters_v = counter_vec_t{};
            auto        found_v    = std::vector<std::string_view>{};
            const auto* agent_v    = tool_metadata->get_agent(agent_id);
            auto        expected_v = tool::get_config().counters.size();

            constexpr auto device_qualifier = std::string_view{":device="};
            for(const auto& itr : tool::get_config().counters)
            {
                auto name_v = itr;
                if(auto pos = std::string::npos;
                   (pos = itr.find(device_qualifier)) != std::string::npos)
                {
                    name_v        = itr.substr(0, pos);
                    auto dev_id_s = itr.substr(pos + device_qualifier.length());

                    ROCP_FATAL_IF(dev_id_s.empty() ||
                                  dev_id_s.find_first_not_of("0123456789") != std::string::npos)
                        << "invalid device qualifier format (':device=N) where N is the "
                           "GPU "
                           "id: "
                        << itr;

                    auto dev_id_v = std::stol(dev_id_s);
                    // skip this counter if the counter is for a specific device id (which
                    // doesn't this agent's device id)
                    if(dev_id_v != agent_v->gpu_index)
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
                ROCP_WARNING << "Unable to find all counters for agent " << agent_v->node_id
                             << " (gpu-" << agent_v->gpu_index << ", " << agent_v->name << ") in ["
                             << requested_counters << "]. Found: [" << found_counters << "]";
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

int64_t
get_instruction_index(rocprofiler_pc_t pc)
{
    if(pc.code_object_id == ROCPROFILER_CODE_OBJECT_ID_NONE)
        return -1;
    else
        return CHECK_NOTNULL(tool_metadata)->get_instruction_index(pc);
}

}  // namespace

void
rocprofiler_pc_sampling_callback(rocprofiler_context_id_t /* context_id*/,
                                 rocprofiler_buffer_id_t /* buffer_id*/,
                                 rocprofiler_record_header_t** headers,
                                 size_t                        num_headers,
                                 void* /*data*/,
                                 uint64_t /* drop_count*/)
{
    if(!headers) return;

    for(size_t i = 0; i < num_headers; i++)
    {
        auto* cur_header = headers[i];

        if(cur_header == nullptr)
        {
            throw std::runtime_error{
                "rocprofiler provided a null pointer to header. this should never happen"};
        }
        else if(cur_header->category == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
        {
            if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE)
            {
                auto* pc_sample = static_cast<rocprofiler_pc_sampling_record_host_trap_v0_t*>(
                    cur_header->payload);

                auto pc_sample_tool_record =
                    rocprofiler::tool::rocprofiler_tool_pc_sampling_host_trap_record_t(
                        *pc_sample, get_instruction_index(pc_sample->pc));

                rocprofiler::tool::write_ring_buffer(pc_sample_tool_record,
                                                     domain_type::PC_SAMPLING_HOST_TRAP);
            }
        }
        else
        {
            ROCP_FATAL << "unexpected rocprofiler_record_header_t category + kind";
        }
    }
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

void
counter_record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                        rocprofiler_record_counter_t*                record_data,
                        size_t                                       record_count,
                        rocprofiler_user_data_t                      user_data,
                        void* /*callback_data_args*/)
{
    static const auto gpu_agents              = get_gpu_agents();
    static const auto gpu_agents_counter_info = get_agent_counter_info();

    auto counter_record = tool::tool_counter_record_t{};

    counter_record.dispatch_data = dispatch_data;
    counter_record.thread_id     = user_data.value;

    auto serialized_records = std::vector<tool::tool_counter_value_t>{};
    serialized_records.reserve(record_count);

    for(size_t count = 0; count < record_count; ++count)
    {
        auto _counter_id = rocprofiler_counter_id_t{};
        ROCPROFILER_CALL(rocprofiler_query_record_counter_id(record_data[count].id, &_counter_id),
                         "query record counter id");
        serialized_records.emplace_back(
            tool::tool_counter_value_t{_counter_id, record_data[count].counter_value});
    }

    if(!serialized_records.empty())
    {
        counter_record.write(serialized_records);
        tool::write_ring_buffer(counter_record, domain_type::COUNTER_COLLECTION);
    }
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

    ROCP_FATAL_IF(!client_identifier) << "nullptr to client identifier!";
    ROCP_FATAL_IF(!client_finalizer && !tool::get_config().list_metrics)
        << "nullptr to client finalizer!";  // exception for listing metrics
}

void
finalize_rocprofv3(std::string_view context)
{
    ROCP_INFO << "invoked: finalize_rocprofv3";
    if(client_finalizer && client_identifier)
    {
        ROCP_INFO << "finalizing rocprofv3: caller='" << context << "'...";
        client_finalizer(*client_identifier);
        client_finalizer  = nullptr;
        client_identifier = nullptr;
    }
    else
    {
        ROCP_INFO << "finalize_rocprofv3('" << context << "') ignored: already finalized";
    }
}

bool
if_pc_sample_config_match(rocprofiler_agent_id_t           agent_id,
                          rocprofiler_pc_sampling_method_t pc_sampling_method,
                          rocprofiler_pc_sampling_unit_t   pc_sampling_unit,
                          uint64_t                         pc_sampling_interval)
{
    auto pc_sampling_config = CHECK_NOTNULL(tool_metadata)->get_pc_sample_config_info(agent_id);
    if(!pc_sampling_config.empty())
    {
        for(auto config : pc_sampling_config)
        {
            if(config.method == pc_sampling_method && config.unit == pc_sampling_unit &&
               config.min_interval <= pc_sampling_interval &&
               config.max_interval >= pc_sampling_interval)
                return true;
        }
    }
    return false;
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    client_finalizer = fini_func;

    constexpr uint64_t buffer_size      = 32 * common::units::KiB;
    constexpr uint64_t buffer_watermark = 31 * common::units::KiB;

    tool_metadata->init(tool::metadata::inprocess{});

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

    if(tool::get_config().memory_allocation_trace)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                                   buffer_size,
                                                   buffer_watermark,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   buffered_tracing_callback,
                                                   nullptr,
                                                   &get_buffers().memory_allocation_trace),
                         "create memory allocation buffer");

        ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                             get_client_ctx(),
                             ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION,
                             nullptr,
                             0,
                             get_buffers().memory_allocation_trace),
                         "buffer tracing service for memory allocation configure");
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

    if(tool::get_config().pc_sampling_host_trap)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                                   buffer_size,
                                                   buffer_watermark,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   rocprofiler_pc_sampling_callback,
                                                   tool_data,
                                                   &get_buffers().pc_sampling_host_trap),
                         "buffer creation");
        bool config_match_found = false;
        auto agent_ptr_vec      = get_gpu_agents();
        for(auto& itr : agent_ptr_vec)
        {
            auto method = tool::get_config().pc_sampling_method_value;
            auto unit   = tool::get_config().pc_sampling_unit_value;
            if(if_pc_sample_config_match(
                   itr->id, method, unit, tool::get_config().pc_sampling_interval))
            {
                config_match_found = true;
                int flags          = 0;
                ROCPROFILER_CALL(rocprofiler_configure_pc_sampling_service(
                                     get_client_ctx(),
                                     itr->id,
                                     method,
                                     unit,
                                     tool::get_config().pc_sampling_interval,
                                     get_buffers().pc_sampling_host_trap,
                                     flags),
                                 "configure PC sampling");
            }
        }
        if(!config_match_found)
            ROCP_FATAL << "Given PC sampling configuration is not supported on any of the agents";
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

    if(tool::get_config().collection_periods.empty())
    {
        ROCPROFILER_CHECK(rocprofiler_start_context(get_client_ctx()));
    }
    else
    {
        auto _prom = std::promise<void>{};
        auto _fut  = _prom.get_future();
        std::thread{collection_period_cntrl, std::move(_prom), get_client_ctx()}.detach();
        _fut.wait_for(std::chrono::seconds{1});  // wait for a max of 1 second
    }

    // Handle kernel id of zero
    bool include = std::regex_search("0", std::regex(tool::get_config().kernel_filter_include));
    bool exclude = std::regex_search("0", std::regex(tool::get_config().kernel_filter_exclude));
    if(include && (!exclude || tool::get_config().kernel_filter_exclude.empty()))
        add_kernel_target(0, tool::get_config().kernel_filter_range);

    tool_metadata->process_id = getpid();
    rocprofiler_get_timestamp(&(tool_metadata->process_start_ns));

    return 0;
}

using stats_data_t       = tool::stats_data_t;
using stats_entry_t      = tool::stats_entry_t;
using domain_stats_vec_t = tool::domain_stats_vec_t;

template <typename Tp, domain_type DomainT>
void
generate_output(tool::buffered_output<Tp, DomainT>& output_v, domain_stats_vec_t& contributions_v)
{
    if(!output_v) return;

    output_v.read();

    if(tool::get_config().stats || tool::get_config().summary_output)
    {
        output_v.stats =
            tool::generate_stats(tool::get_config(), *tool_metadata, output_v.get_generator());
    }

    if(output_v.stats)
    {
        contributions_v.emplace_back(output_v.buffer_type_v, output_v.stats);
    }

    if(tool::get_config().csv_output)
    {
        tool::generate_csv(
            tool::get_config(), *tool_metadata, output_v.get_generator(), output_v.stats);
    }
}

void
tool_fini(void* /*tool_data*/)
{
    client_identifier = nullptr;
    client_finalizer  = nullptr;

    tool_metadata->process_id = getpid();
    rocprofiler_get_timestamp(&(tool_metadata->process_end_ns));

    flush();
    rocprofiler_stop_context(get_client_ctx());
    flush();

    auto kernel_dispatch_output =
        tool::kernel_dispatch_buffered_output_t{tool::get_config().kernel_trace};
    auto hsa_output = tool::hsa_buffered_output_t{tool::get_config().hsa_core_api_trace ||
                                                  tool::get_config().hsa_amd_ext_api_trace ||
                                                  tool::get_config().hsa_image_ext_api_trace ||
                                                  tool::get_config().hsa_finalizer_ext_api_trace};
    auto hip_output = tool::hip_buffered_output_t{tool::get_config().hip_runtime_api_trace ||
                                                  tool::get_config().hip_compiler_api_trace};
    auto memory_copy_output =
        tool::memory_copy_buffered_output_t{tool::get_config().memory_copy_trace};
    auto marker_output = tool::marker_buffered_output_t{tool::get_config().marker_api_trace};
    auto counters_output =
        tool::counter_collection_buffered_output_t{tool::get_config().counter_collection};
    auto scratch_memory_output =
        tool::scratch_memory_buffered_output_t{tool::get_config().scratch_memory_trace};
    auto rccl_output = tool::rccl_buffered_output_t{tool::get_config().rccl_api_trace};
    auto memory_allocation_output =
        tool::memory_allocation_buffered_output_t{tool::get_config().memory_allocation_trace};
    auto counters_records_output =
        tool::counter_records_buffered_output_t{tool::get_config().counter_collection};
    auto pc_sampling_host_trap_output =
        tool::pc_sampling_host_trap_buffered_output_t{tool::get_config().pc_sampling_host_trap};

    auto node_id_sort = [](const auto& lhs, const auto& rhs) { return lhs.node_id < rhs.node_id; };

    auto _agents = CHECK_NOTNULL(tool_metadata)->agents;
    std::sort(_agents.begin(), _agents.end(), node_id_sort);

    if(tool::get_config().csv_output)
    {
        tool::generate_csv(tool::get_config(), *tool_metadata, _agents);
    }

    auto contributions = domain_stats_vec_t{};

    generate_output(kernel_dispatch_output, contributions);
    generate_output(hsa_output, contributions);
    generate_output(hip_output, contributions);
    generate_output(memory_copy_output, contributions);
    generate_output(memory_allocation_output, contributions);
    generate_output(marker_output, contributions);
    generate_output(rccl_output, contributions);
    generate_output(counters_output, contributions);
    generate_output(scratch_memory_output, contributions);
    generate_output(pc_sampling_host_trap_output, contributions);

    if(tool::get_config().stats && tool::get_config().csv_output)
    {
        tool::generate_csv(tool::get_config(), *tool_metadata, contributions);
    }

    if(tool::get_config().json_output)
    {
        auto json_ar = tool::open_json(tool::get_config());

        json_ar.start_process();
        tool::write_json(json_ar, tool::get_config(), *tool_metadata, getpid());
        tool::write_json(json_ar,
                         tool::get_config(),
                         *tool_metadata,
                         contributions,
                         hip_output.get_generator(),
                         hsa_output.get_generator(),
                         kernel_dispatch_output.get_generator(),
                         memory_copy_output.get_generator(),
                         counters_output.get_generator(),
                         marker_output.get_generator(),
                         scratch_memory_output.get_generator(),
                         rccl_output.get_generator(),
                         memory_allocation_output.get_generator(),
                         pc_sampling_host_trap_output.get_generator());
        json_ar.finish_process();

        tool::close_json(json_ar);
    }

    if(tool::get_config().pftrace_output)
    {
        tool::write_perfetto(tool::get_config(),
                             *tool_metadata,
                             _agents,
                             hip_output.get_generator(),
                             hsa_output.get_generator(),
                             kernel_dispatch_output.get_generator(),
                             memory_copy_output.get_generator(),
                             marker_output.get_generator(),
                             scratch_memory_output.get_generator(),
                             rccl_output.get_generator(),
                             memory_allocation_output.get_generator());
    }

    if(tool::get_config().otf2_output)
    {
        auto hip_elem_data               = hip_output.load_all();
        auto hsa_elem_data               = hsa_output.load_all();
        auto kernel_dispatch_elem_data   = kernel_dispatch_output.load_all();
        auto memory_copy_elem_data       = memory_copy_output.load_all();
        auto marker_elem_data            = marker_output.load_all();
        auto scratch_memory_elem_data    = scratch_memory_output.load_all();
        auto rccl_elem_data              = rccl_output.load_all();
        auto memory_allocation_elem_data = memory_allocation_output.load_all();

        tool::write_otf2(tool::get_config(),
                         *tool_metadata,
                         getpid(),
                         _agents,
                         &hip_elem_data,
                         &hsa_elem_data,
                         &kernel_dispatch_elem_data,
                         &memory_copy_elem_data,
                         &marker_elem_data,
                         &scratch_memory_elem_data,
                         &rccl_elem_data,
                         &memory_allocation_elem_data);
    }

    if(tool::get_config().summary_output)
    {
        tool::generate_stats(tool::get_config(), *tool_metadata, contributions);
    }

    auto destroy_output = [](auto& _buffered_output_v) { _buffered_output_v.destroy(); };

    destroy_output(kernel_dispatch_output);
    destroy_output(hsa_output);
    destroy_output(hip_output);
    destroy_output(memory_copy_output);
    destroy_output(memory_allocation_output);
    destroy_output(marker_output);
    destroy_output(counters_output);
    destroy_output(scratch_memory_output);
    destroy_output(rccl_output);
    destroy_output(counters_records_output);
    destroy_output(pc_sampling_host_trap_output);

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

std::vector<rocprofiler_record_dimension_info_t>
get_tool_counter_dimension_info()
{
    auto _data = get_agent_counter_info();
    auto _ret  = std::vector<rocprofiler_record_dimension_info_t>{};
    for(const auto& itr : _data)
    {
        for(const auto& iitr : itr.second)
            for(const auto& ditr : iitr.dimensions)
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

bool signal_handler_exit = tool::get_env("ROCPROF_INTERNAL_TEST_SIGNAL_HANDLER_VIA_EXIT", false);
}  // namespace

#define ROCPROFV3_INTERNAL_API __attribute__((visibility("internal")));

extern "C" {
void
rocprofv3_set_main(main_func_t main_func) ROCPROFV3_INTERNAL_API;

void
rocprofv3_error_signal_handler(int signo)
{
    ROCP_WARNING << __FUNCTION__ << " caught signal " << signo << "...";

    finalize_rocprofv3(__FUNCTION__);
    // below is for testing purposes. re-raising the signal causes CTest to ignore WILL_FAIL ON
    if(signal_handler_exit) ::quick_exit(signo);
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
    add_destructor(tool_metadata);

    // in case main wrapper is not used
    ::atexit([]() { finalize_rocprofv3("atexit"); });

    tool::get_tmp_file_name_callback() = [](domain_type type) -> std::string {
        return compose_tmp_file_name(tool::get_config(), type);
    };

    if(!tool::get_config().extra_counters_contents.empty())
    {
        std::string contents(tool::get_config().extra_counters_contents);
        size_t      length = contents.size();
        ROCPROFILER_CALL(rocprofiler_load_counter_definition(
                             contents.c_str(), length, ROCPROFILER_COUNTER_FLAG_APPEND_DEFINITION),
                         "Loading extra counters");
    }

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

    ROCP_INFO << "rocprofv3: main function wrapper will be invoked...";

    auto ret = CHECK_NOTNULL(get_main_function())(argc, argv, envp);

    ROCP_INFO << "rocprofv3: main function has returned with exit code: " << ret;

    finalize_rocprofv3(__FUNCTION__);

    ROCP_INFO << "rocprofv3 finished. exit code: " << ret;
    return ret;
}
}
