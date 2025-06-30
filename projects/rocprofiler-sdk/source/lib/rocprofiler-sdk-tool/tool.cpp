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

#define _GNU_SOURCE     1
#define _DEFAULT_SOURCE 1

#include "config.hpp"
#include "execution_profile.hpp"
#include "helper.hpp"
#include "stream_stack.hpp"

#include "lib/att-tool/att_lib_wrapper.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/simple_timer.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/static_tl_object.hpp"
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
#include "lib/output/generateRocpd.hpp"
#include "lib/output/generateStats.hpp"
#include "lib/output/metadata.hpp"
#include "lib/output/output_stream.hpp"
#include "lib/output/statistics.hpp"
#include "lib/output/stream_info.hpp"
#include "lib/output/timestamps.hpp"
#include "lib/output/tmp_file.hpp"
#include "lib/output/tmp_file_buffer.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/counters.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/version.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/core.h>

#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdlib>
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

#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>

#if defined(CODECOV) && CODECOV > 0
extern "C" {
extern void
__gcov_dump(void);
}
#endif

namespace common = ::rocprofiler::common;
namespace tool   = ::rocprofiler::tool;

extern "C" {
void
rocprofv3_error_signal_handler(int signo, siginfo_t*, void*);
}

namespace
{
using sigaction_t      = struct sigaction;
using signal_func_t    = sighandler_t (*)(int signum, sighandler_t handler);
using sigaction_func_t = int (*)(int signum,
                                 const struct sigaction* __restrict__ act,
                                 struct sigaction* __restrict__ oldact);

constexpr auto rocprofv3_num_signals     = NSIG;
constexpr auto rocprofv3_handled_signals = std::array<int, 4>{SIGINT, SIGQUIT, SIGABRT, SIGTERM};

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

struct chained_siginfo
{
    int                        signo   = 0;
    sighandler_t               handler = nullptr;
    std::optional<sigaction_t> action  = {};
};

auto&
get_chained_signals()
{
    using data_type  = std::array<std::optional<chained_siginfo>, rocprofv3_num_signals>;
    static auto*& _v = common::static_object<data_type>::construct();
    return *CHECK_NOTNULL(_v);
}

bool
is_handled_signal(int signum)
{
    for(auto itr : rocprofv3_handled_signals)
        if(itr == signum) return true;
    return false;
}

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
    rocprofiler_buffer_id_t rocdecode_api_trace     = {};
    rocprofiler_buffer_id_t rocjpeg_api_trace       = {};
    rocprofiler_buffer_id_t pc_sampling_stochastic  = {};

    auto as_array() const
    {
        return std::array<rocprofiler_buffer_id_t, 12>{hsa_api_trace,
                                                       hip_api_trace,
                                                       kernel_trace,
                                                       memory_copy_trace,
                                                       memory_allocation_trace,
                                                       counter_collection,
                                                       scratch_memory,
                                                       rccl_api_trace,
                                                       pc_sampling_host_trap,
                                                       rocdecode_api_trace,
                                                       rocjpeg_api_trace,
                                                       pc_sampling_stochastic};
    }
    auto pc_sampling_buffers_as_array() const
    {
        return std::array<rocprofiler_buffer_id_t, 2>{pc_sampling_host_trap,
                                                      pc_sampling_stochastic};
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
    std::unordered_map<rocprofiler_kernel_id_t, std::unordered_set<size_t>>;
using counter_dimension_info_map_t =
    std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_dimension_info_t>>;
using agent_info_map_t      = std::unordered_map<rocprofiler_agent_id_t, rocprofiler_agent_t>;
using kernel_iteration_t    = std::unordered_map<rocprofiler_kernel_id_t, size_t>;
using kernel_rename_map_t   = std::unordered_map<uint64_t, uint64_t>;
using kernel_rename_stack_t = std::stack<uint64_t>;

auto*      tool_metadata     = as_pointer<tool::metadata>(tool::metadata::inprocess{});
auto       target_kernels    = common::Synchronized<targeted_kernels_map_t>{};
auto*      execution_profile = as_pointer<common::Synchronized<tool::execution_profile_data>>();
auto       counter_collection_ctx = rocprofiler_context_id_t{0};
std::mutex att_shader_data;

thread_local auto thread_dispatch_rename      = as_pointer<kernel_rename_stack_t>();
thread_local auto thread_dispatch_rename_dtor = common::scope_destructor{[]() {
    delete thread_dispatch_rename;
    thread_dispatch_rename = nullptr;
}};

// Stores stream ids and kernel region ids for kernel-rename service and hip stream display service
struct kernel_rename_and_stream_data
{
    uint64_t                region_id = 0;  // roctx region correlation id
    rocprofiler_stream_id_t stream_id = {.handle = 0};
};

bool
add_kernel_target(uint64_t _kern_id, const std::unordered_set<size_t>& range)
{
    return target_kernels
        .wlock(
            [](targeted_kernels_map_t&           _targets_v,
               uint64_t                          _kern_id_v,
               const std::unordered_set<size_t>& _range) {
                return _targets_v.emplace(_kern_id_v, _range);
            },
            _kern_id,
            range)
        .second;
}

bool
is_targeted_kernel(uint64_t                                        _kern_id,
                   common::Synchronized<kernel_iteration_t, true>& _kernel_iteration)
{
    const std::unordered_set<size_t>* range = target_kernels.rlock(
        [](const auto& _targets_v, uint64_t _kern_id_v) -> const std::unordered_set<size_t>* {
            if(_targets_v.find(_kern_id_v) != _targets_v.end()) return &_targets_v.at(_kern_id_v);
            return nullptr;
        },
        _kern_id);

    if(range)
    {
        _kernel_iteration.wlock(
            [](auto& _kernel_iter, rocprofiler_kernel_id_t _kernel_id) {
                auto itr = _kernel_iter.find(_kernel_id);
                if(itr == _kernel_iter.end())
                    _kernel_iter.emplace(_kernel_id, 1);
                else
                    itr->second++;
            },
            _kern_id);

        return _kernel_iteration.rlock(
            [](const auto&                       _kernel_iter,
               uint64_t                          _kernel_id,
               const std::unordered_set<size_t>& _range) {
                auto itr = _kernel_iter.at(_kernel_id);
                // If the iteration range is not given then all iterations of the kernel is profiled
                if(_range.empty())
                {
                    if(!tool::get_config().advanced_thread_trace)
                        return true;
                    else if(itr == 1)
                        return true;
                }
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
    constexpr auto null_buffer_id = rocprofiler_buffer_id_t{.handle = 0};

    ROCP_INFO << "flushing buffers...";
    for(auto itr : get_buffers().as_array())
    {
        if(itr > null_buffer_id)
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
record_execution_profile(rocprofiler_thread_id_t                            thr_id,
                         rocprofiler_context_id_t                           ctx_id,
                         rocprofiler_external_correlation_id_request_kind_t kind,
                         rocprofiler_tracing_operation_t                    op,
                         uint64_t /*internal_corr_id*/,
                         rocprofiler_user_data_t* /*external_corr_id*/,
                         void* /*user_data*/)
{
    auto _record_data = [](tool::execution_profile_data&                      _data,
                           rocprofiler_thread_id_t                            _thr_id,
                           rocprofiler_context_id_t                           _ctx_id,
                           rocprofiler_external_correlation_id_request_kind_t _kind,
                           rocprofiler_tracing_operation_t                    _op) {
        _data.category_count[_kind] += 1;
        _data.category_op_count[_kind].emplace(_op);
        _data.threads.emplace(_thr_id);
        _data.contexts.emplace(_ctx_id);
    };

    if(execution_profile)
        execution_profile->wlock(std::move(_record_data), thr_id, ctx_id, kind, op);

    return 0;
}

template <typename Tp>
rocprofiler_stream_id_t
get_stream_id(Tp* _record)
{
    auto _stream_id = rocprofiler_stream_id_t{.handle = 0};
    if(_record->correlation_id.external.ptr != nullptr)
    {
        // Extract the stream id
        auto* _ecid_data =
            static_cast<kernel_rename_and_stream_data*>(_record->correlation_id.external.ptr);
        _stream_id                             = _ecid_data->stream_id;
        auto _region_id                        = _ecid_data->region_id;
        _record->correlation_id.external.value = _region_id;
        delete _ecid_data;
    }
    return _stream_id;
}

int
set_kernel_rename_and_stream_correlation_id(rocprofiler_thread_id_t  thr_id,
                                            rocprofiler_context_id_t ctx_id,
                                            rocprofiler_external_correlation_id_request_kind_t kind,
                                            rocprofiler_tracing_operation_t                    op,
                                            uint64_t                 internal_corr_id,
                                            rocprofiler_user_data_t* external_corr_id,
                                            void*                    user_data)
{
    // Check whether services are enabled
    const bool kernel_rename_service_enabled =
        kind == ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH &&
        tool::get_config().kernel_rename && thread_dispatch_rename != nullptr &&
        !thread_dispatch_rename->empty();

    const bool hip_stream_enabled =
        !tool::get_config().group_by_queue && rocprofiler::tool::stream::stream_stack_not_null();

    if(!kernel_rename_service_enabled && !hip_stream_enabled) return 1;

    auto* _info = new kernel_rename_and_stream_data{};

    // Get value for kernel rename service
    if(kernel_rename_service_enabled)
    {
        _info->region_id = thread_dispatch_rename->top();
        if(tool_metadata) tool_metadata->add_external_correlation_id(_info->region_id);
    }

    // Get stream ID from stream HIP display service
    if(hip_stream_enabled)
    {
        _info->stream_id = rocprofiler::tool::stream::get_stream_id();
    }

    // Set the external correlation id service to point to struct
    external_corr_id->ptr = _info;

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
        auto add_message = [](std::string_view val) {
            auto _hash_v = common::add_string_entry(val);
            return std::string_view{*common::get_string_entry(_hash_v)};
        };

        if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA &&
           record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT && marker_data->args.roctxMarkA.message)
        {
            thread_dispatch_rename->emplace(tool_metadata->add_kernel_rename_val(
                add_message(marker_data->args.roctxMarkA.message), record.correlation_id.internal));
        }
        else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA &&
                record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
                marker_data->args.roctxRangePushA.message)
        {
            thread_dispatch_rename->emplace(tool_metadata->add_kernel_rename_val(
                add_message(marker_data->args.roctxRangePushA.message),
                record.correlation_id.internal));
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

// Stores stream IDs onto stack when callback is triggered
void
hip_stream_display_callback(rocprofiler_callback_tracing_record_t record,
                            rocprofiler_user_data_t*              user_data,
                            void*                                 data)
{
    if(tool::get_config().group_by_queue || record.kind != ROCPROFILER_CALLBACK_TRACING_HIP_STREAM)
        return;
    // Extract stream ID from record
    auto* stream_handle_data =
        static_cast<rocprofiler_callback_tracing_hip_stream_data_t*>(record.payload);
    auto stream_id = stream_handle_data->stream_id;

    // STREAM_HANDLE_CREATE and DESTROY are no-ops
    if(record.operation == ROCPROFILER_HIP_STREAM_CREATE)
    {
        ROCP_TRACE
            << "Entered hip_stream_display_callback function for ROCPROFILER_HIP_STREAM_CREATE";
    }
    else if(record.operation == ROCPROFILER_HIP_STREAM_DESTROY)
    {
        ROCP_TRACE
            << "Entered hip_stream_display_callback function for ROCPROFILER_HIP_STREAM_DESTROY";
    }
    else if(record.operation == ROCPROFILER_HIP_STREAM_SET)
    {
        // Push the stream ID onto the stream stack when before underlying HIP function is called
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            ROCP_TRACE << "Entered hip_stream_display_callback function for "
                          "ROCPROFILER_HIP_STREAM_SET with ROCPROFILER_CALLBACK_PHASE_ENTER";
            rocprofiler::tool::stream::push_stream_id(stream_id);
        }
        // Pop stream ID off of stream stack after underlying HIP function is completed
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
        {
            ROCP_TRACE << "Entered hip_stream_display_callback function for "
                          "ROCPROFILER_HIP_STREAM_SET with ROCPROFILER_CALLBACK_PHASE_EXIT";
            rocprofiler::tool::stream::pop_stream_id();
        }
    }
    else
    {
        ROCP_FATAL << "Unsupported operation for ROCPROFILER_HIP_STREAM";
    }
    common::consume_args(user_data, data);
}

// Stores which runtimes have been initialized in metadata
void
runtime_initialization_callback(rocprofiler_callback_tracing_record_t record,
                                rocprofiler_user_data_t*              user_data,
                                void*                                 data)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION) return;
    ROCP_CI_LOG_IF(WARNING, tool_metadata == nullptr)
        << fmt::format("tool cannot record runtime initialization for {}",
                       tool_metadata->get_operation_name(record.kind, record.operation));
    if(tool_metadata)
    {
        tool_metadata->add_runtime_initialization(
            static_cast<rocprofiler_runtime_initialization_operation_t>(record.operation));
    }
    common::consume_args(user_data, data);
}

void
dummy_callback_tracing_callback(rocprofiler_callback_tracing_record_t /*record*/,
                                rocprofiler_user_data_t* /*user_data*/,
                                void* /*data*/)
{}

void
dummy_counter_dispatch_callback(rocprofiler_dispatch_counting_service_data_t,
                                rocprofiler_profile_config_id_t*,
                                rocprofiler_user_data_t*,
                                void*)
{}

void
dummy_counter_record_callback(rocprofiler_dispatch_counting_service_data_t,
                              rocprofiler_record_counter_t*,
                              size_t,
                              rocprofiler_user_data_t,
                              void*)
{}

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
            if(tool::get_config().pc_sampling_host_trap ||
               tool::get_config().pc_sampling_stochastic)
            {
                CHECK_NOTNULL(tool_metadata)->add_decoder(obj_data);
            }

            if(obj_data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY &&
               tool::get_config().advanced_thread_trace)
            {
                const char* gpu_name      = tool_metadata->agents_map.at(obj_data->rocp_agent).name;
                auto        filename      = fmt::format("{}_code_object_id_{}",
                                            std::string(gpu_name),
                                            std::to_string(obj_data->code_object_id));
                auto        output_stream = get_output_stream(tool::get_config(), filename, ".out");
                std::string output_filename =
                    get_output_filename(tool::get_config(), filename, ".out");

                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                output_stream.stream->write(reinterpret_cast<char*>(obj_data->memory_base),
                                            obj_data->memory_size);
                tool_metadata->code_object_load.wlock(
                    [](auto&                                 data_vec,
                       std::string                           file_name,
                       tool::rocprofiler_code_object_info_t* obj_data_v) {
                        data_vec.push_back({file_name,
                                            obj_data_v->code_object_id,
                                            obj_data_v->load_base,
                                            obj_data_v->load_size});
                    },
                    output_filename,
                    obj_data);
            }
            else if(obj_data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE &&
                    tool::get_config().advanced_thread_trace)
            {
                const char* gpu_name      = tool_metadata->agents_map.at(obj_data->rocp_agent).name;
                auto        filename      = fmt::format("{}_code_object_id_{}",
                                            std::string(gpu_name),
                                            std::to_string(obj_data->code_object_id));
                auto        output_stream = get_output_stream(tool::get_config(), filename, ".out");
                std::string output_filename =
                    get_output_filename(tool::get_config(), filename, ".out");

                uint8_t*      binary      = nullptr;
                size_t        buffer_size = 0;
                std::ifstream code_object_file(obj_data->uri, std::ios::binary | std::ios::ate);
                if(code_object_file.good())
                {
                    buffer_size = code_object_file.tellg();
                    code_object_file.seekg(0, std::ios::beg);
                    binary = new(std::nothrow) uint8_t[buffer_size];
                    if(binary &&
                       !code_object_file.read(reinterpret_cast<char*>(binary), buffer_size))
                    {
                        delete[] binary;
                        binary = nullptr;
                    }
                }
                // NOLINTBEGIN(performance-no-int-to-ptr)
                output_stream.stream->write(reinterpret_cast<char*>(obj_data->memory_base),
                                            obj_data->memory_size);
                // NOLINTEND(performance-no-int-to-ptr)
                tool_metadata->code_object_load.wlock(
                    [](auto&                                 data_vec,
                       std::string                           file_name,
                       tool::rocprofiler_code_object_info_t* obj_data_v) {
                        data_vec.push_back({file_name,
                                            obj_data_v->code_object_id,
                                            obj_data_v->load_base,
                                            obj_data_v->load_size});
                    },
                    output_filename,
                    obj_data);
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
            ROCP_TRACE << fmt::format("adding kernel symbol info for kernel_id={} :: {}",
                                      sym_data->kernel_id,
                                      sym_data->kernel_name);

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
dummy_buffered_tracing_callback(rocprofiler_context_id_t /*context*/,
                                rocprofiler_buffer_id_t /*buffer_id*/,
                                rocprofiler_record_header_t** /*headers*/,
                                size_t /*num_headers*/,
                                void* /*user_data*/,
                                uint64_t /*drop_count*/)
{}

void
buffered_tracing_callback(rocprofiler_context_id_t /*context*/,
                          rocprofiler_buffer_id_t /*buffer_id*/,
                          rocprofiler_record_header_t** headers,
                          size_t                        num_headers,
                          void* /* user_data*/,
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

                auto stream_id = get_stream_id(record);
                tool::write_ring_buffer(
                    tool::tool_buffer_tracing_kernel_dispatch_ext_record_t{*record, stream_id},
                    domain_type::KERNEL_DISPATCH);
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

                auto stream_id = get_stream_id(record);
                tool::write_ring_buffer(
                    tool::tool_buffer_tracing_memory_copy_ext_record_t{*record, stream_id},
                    domain_type::MEMORY_COPY);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_memory_allocation_record_t*>(
                    header->payload);

                auto stream_id = get_stream_id(record);
                tool::write_ring_buffer(
                    tool::tool_buffer_tracing_memory_allocation_ext_record_t{*record, stream_id},
                    domain_type::MEMORY_ALLOCATION);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_scratch_memory_record_t*>(
                    header->payload);

                tool::write_ring_buffer(*record, domain_type::SCRATCH_MEMORY);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_hip_api_ext_record_t*>(header->payload);

                auto stream_id = get_stream_id(record);
                tool::write_ring_buffer(
                    tool::tool_buffer_tracing_hip_api_ext_record_t{*record, stream_id},
                    domain_type::HIP);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_RCCL_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_rccl_api_record_t*>(header->payload);

                tool::write_ring_buffer(*record, domain_type::RCCL);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_ROCDECODE_API_EXT)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t*>(
                    header->payload);

                tool::write_ring_buffer(*record, domain_type::ROCDECODE);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_ROCJPEG_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_rocjpeg_api_record_t*>(header->payload);

                tool::write_ring_buffer(*record, domain_type::ROCJPEG);
            }
            else
            {
                ROCP_CI_LOG(WARNING) << fmt::format(
                    "unsupported ROCPROFILER_BUFFER_CATEGORY_TRACING kind: {} :: {}",
                    header->kind,
                    tool_metadata->get_kind_name(
                        static_cast<rocprofiler_buffer_tracing_kind_t>(header->kind)));
            }
        }
        else
        {
            ROCP_CI_LOG(WARNING) << fmt::format(
                "unsupported category + kind: {} + {}", header->category, header->kind);
        }
    }
}

using counter_vec_t = std::vector<rocprofiler_counter_id_t>;
using agent_counter_map_t =
    std::unordered_map<rocprofiler_agent_id_t, std::optional<rocprofiler_counter_config_id_t>>;

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

struct agent_profiles
{
    std::unordered_map<rocprofiler_agent_id_t, std::atomic<uint64_t>> current_iter;
    const uint64_t                                                    rotation;
    const std::unordered_map<rocprofiler_agent_id_t, std::vector<rocprofiler_counter_config_id_t>>
        profiles;
};

std::optional<rocprofiler_counter_config_id_t>
construct_counter_collection_profile(rocprofiler_agent_id_t       agent_id,
                                     const std::set<std::string>& counters)
{
    static const auto gpu_agents_counter_info = get_agent_counter_info();
    auto              profile                 = std::optional<rocprofiler_counter_config_id_t>{};
    auto              counters_v              = counter_vec_t{};
    auto              found_v                 = std::vector<std::string_view>{};
    const auto*       agent_v                 = tool_metadata->get_agent(agent_id);
    auto              expected_v              = counters.size();

    constexpr auto device_qualifier = std::string_view{":device="};
    for(const auto& itr : counters)
    {
        auto name_v = itr;
        if(auto pos = std::string::npos; (pos = itr.find(device_qualifier)) != std::string::npos)
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
        auto requested_counters =
            fmt::format("{}", fmt::join(counters.begin(), counters.end(), ", "));
        auto found_counters = fmt::format("{}", fmt::join(found_v.begin(), found_v.end(), ", "));
        ROCP_WARNING << "Unable to find all counters for agent " << agent_v->node_id << " (gpu-"
                     << agent_v->gpu_index << ", " << agent_v->name << ") in ["
                     << requested_counters << "]. Found: [" << found_counters << "]";
    }

    if(!counters_v.empty())
    {
        auto profile_v = rocprofiler_counter_config_id_t{};
        ROCPROFILER_CALL(rocprofiler_create_counter_config(
                             agent_id, counters_v.data(), counters_v.size(), &profile_v),
                         "Could not construct profile cfg");
        profile = profile_v;
    }
    return profile;
}

agent_profiles
generate_agent_profiles()
{
    std::unordered_map<rocprofiler_agent_id_t, std::vector<rocprofiler_counter_config_id_t>>
                                                                      profiles;
    std::unordered_map<rocprofiler_agent_id_t, std::atomic<uint64_t>> pos;
    for(const auto& agent : get_gpu_agents())
    {
        for(const auto& counter_set : tool::get_config().counters)
        {
            if(agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;
            auto profile = construct_counter_collection_profile(agent->id, counter_set);
            if(profile.has_value())
            {
                profiles[agent->id].push_back(profile.value());
            }
        }
        pos[agent->id] = 0;
    }
    return agent_profiles{std::move(pos), tool::get_config().counter_groups_interval, profiles};
}

// this function creates a rocprofiler profile config on the first entry
std::optional<rocprofiler_counter_config_id_t>
get_device_counting_service(rocprofiler_agent_id_t agent_id)
{
    static auto agent_profiles = generate_agent_profiles();

    auto agent_iter = agent_profiles.current_iter.find(agent_id);
    if(agent_iter == agent_profiles.current_iter.end())
    {
        return std::nullopt;
    }

    auto my_iter = agent_iter->second.fetch_add(1);

    const auto profiles = agent_profiles.profiles.find(agent_id);
    if(profiles == agent_profiles.profiles.end())
    {
        return std::nullopt;
    }

    if(profiles->second.empty()) return std::nullopt;

    uint64_t profile_pos = my_iter / agent_profiles.rotation;
    return profiles->second[profile_pos % profiles->second.size()];
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

std::vector<rocprofiler_thread_trace_parameter_t>
get_att_perfcounter_params(rocprofiler_agent_id_t                           agent,
                           std::vector<rocprofiler::tool::att_perfcounter>& att_perf_counters)
{
    std::vector<rocprofiler_thread_trace_parameter_t> _data{};
    if(att_perf_counters.empty()) return _data;

    static const auto agent_counter_info = get_agent_counter_info();

    for(const auto& att_perf_counter : att_perf_counters)
    {
        bool counter_found = false;

        for(const auto& counter_info_ : agent_counter_info.at(agent))
        {
            if(std::string_view(counter_info_.name) != att_perf_counter.counter_name) continue;

            auto param       = rocprofiler_thread_trace_parameter_t{};
            param.type       = ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTER;
            param.counter_id = counter_info_.id;
            param.simd_mask  = att_perf_counter.simd_mask;
            _data.emplace_back(param);
            counter_found = true;
            break;
        }

        ROCP_WARNING_IF(!counter_found)
            << "Agent " << agent.handle << " counter not found: " << att_perf_counter.counter_name;
    }

    return _data;
}

void
pc_sampling_callback(rocprofiler_context_id_t /* context_id*/,
                     rocprofiler_buffer_id_t /* buffer_id*/,
                     rocprofiler_record_header_t** headers,
                     size_t                        num_headers,
                     void* /*data*/,
                     uint64_t /* drop_count*/)
{
    if(!headers) return;

    // count number of valid VS invalid samples delivered by this callback
    uint64_t valid_samples_cnt   = 0;
    uint64_t invalid_samples_cnt = 0;

    for(size_t i = 0; i < num_headers; i++)
    {
        auto* cur_header = headers[i];

        if(cur_header == nullptr)
        {
            ROCP_CI_LOG(WARNING) << "rocprofiler provided a null pointer to buffer record header. "
                                    "this should never happen";
            continue;
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

                valid_samples_cnt++;
            }
            else if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE)
            {
                auto* pc_sample = static_cast<rocprofiler_pc_sampling_record_stochastic_v0_t*>(
                    cur_header->payload);

                auto pc_sample_tool_record =
                    rocprofiler::tool::rocprofiler_tool_pc_sampling_stochastic_record_t(
                        *pc_sample, get_instruction_index(pc_sample->pc));

                rocprofiler::tool::write_ring_buffer(pc_sample_tool_record,
                                                     domain_type::PC_SAMPLING_STOCHASTIC);
                valid_samples_cnt++;
            }
            else if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE)
            {
                invalid_samples_cnt++;
            }
        }
        else
        {
            ROCP_FATAL << "unexpected rocprofiler_record_header_t category + kind";
        }
    }

    // sum up number of valid/invalid samples for pc sampling stats
    tool_metadata->pc_sampling_stats.wlock(
        [valid_samples_cnt, invalid_samples_cnt](auto& pc_sampling_stats) {
            pc_sampling_stats.valid_samples += valid_samples_cnt;
            pc_sampling_stats.invalid_samples += invalid_samples_cnt;
        });
}

void
att_shader_data_callback(rocprofiler_agent_id_t  agent,
                         int64_t                 se_id,
                         void*                   se_data,
                         size_t                  data_size,
                         rocprofiler_user_data_t userdata)
{
    std::lock_guard<std::mutex> lock(att_shader_data);
    std::stringstream           filename;
    filename << fmt::format("{}_shader_engine_{}_{}", agent.handle, se_id, userdata.value);

    auto        dispatch_id     = static_cast<rocprofiler_dispatch_id_t>(userdata.value);
    auto        output_stream   = get_output_stream(tool::get_config(), filename.str(), ".att");
    std::string output_filename = get_output_filename(tool::get_config(), filename.str(), ".att");

    output_stream.stream->write(reinterpret_cast<char*>(se_data), data_size);
    tool_metadata->att_filenames[dispatch_id].first = agent;
    tool_metadata->att_filenames[dispatch_id].second.emplace_back(output_filename);
}

rocprofiler_thread_trace_control_flags_t
att_dispatch_callback(rocprofiler_agent_id_t /* agent_id  */,
                      rocprofiler_queue_id_t /* queue_id  */,
                      rocprofiler_async_correlation_id_t /* correlation_id */,
                      rocprofiler_kernel_id_t   kernel_id,
                      rocprofiler_dispatch_id_t dispatch_id,
                      void* /*userdata_config*/,
                      rocprofiler_user_data_t* userdata_shader)
{
    static auto kernel_iteration = common::Synchronized<kernel_iteration_t, true>{};

    userdata_shader->value = dispatch_id;

    if(is_targeted_kernel(kernel_id, kernel_iteration))
        return ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP;
    return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
}

void
counter_dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                          rocprofiler_counter_config_id_t*             config,
                          rocprofiler_user_data_t*                     user_data,
                          void* /*callback_data_args*/)
{
    static auto kernel_iteration = common::Synchronized<kernel_iteration_t, true>{};

    auto kernel_id = dispatch_data.dispatch_info.kernel_id;
    auto agent_id  = dispatch_data.dispatch_info.agent_id;

    if(!is_targeted_kernel(kernel_id, kernel_iteration))
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
                        rocprofiler_counter_record_t*                record_data,
                        size_t                                       record_count,
                        rocprofiler_user_data_t                      user_data,
                        void* /*callback_data_args*/)
{
    static const auto gpu_agents              = get_gpu_agents();
    static const auto gpu_agents_counter_info = get_agent_counter_info();

    auto counter_record = tool::tool_counter_record_t{};

    // must call get_stream_id on dispatch_data before copying to counter_record.dispatch_data
    // so that external correlation id is updated before copy is made
    counter_record.stream_id     = get_stream_id(&dispatch_data);
    counter_record.dispatch_data = dispatch_data;
    counter_record.thread_id     = user_data.value;
    auto serialized_records      = std::vector<tool::tool_counter_value_t>{};
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
    static auto _once = std::atomic<uint64_t>{0};
    if(_once++ == 0)
    {
        auto logging_cfg = rocprofiler::common::logging_config{.install_failure_handler = true};
        common::init_logging("ROCPROF", logging_cfg);
        FLAGS_colorlogtostderr = true;
    }
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
initialize_signal_handler(sigaction_func_t sigaction_func)
{
    if(sigaction_func == nullptr) sigaction_func = &sigaction;

    struct sigaction sig_act = {};
    sigemptyset(&sig_act.sa_mask);
    sig_act.sa_flags     = (SA_SIGINFO | SA_RESETHAND | SA_NOCLDSTOP);
    sig_act.sa_sigaction = &rocprofv3_error_signal_handler;
    for(auto signal_v : rocprofv3_handled_signals)
    {
        if(get_chained_signals().at(signal_v))
        {
            ROCP_INFO << "Skipping install of signal handler for signal " << signal_v
                      << " (already wrapped)";
            continue;
        }

        ROCP_INFO << "Installing signal handler for signal " << signal_v;
        if(sigaction_func(signal_v, &sig_act, nullptr) != 0)
        {
            auto _errno_v = errno;
            ROCP_ERROR << "error setting signal handler for " << signal_v
                       << " :: " << strerror(_errno_v);
        }
    }
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

void
configure_pc_sampling_on_all_agents(uint64_t                        buffer_size,
                                    uint64_t                        buffer_watermark,
                                    void*                           tool_data,
                                    rocprofiler_buffer_tracing_cb_t pc_sampling_cb)
{
    auto method = tool::get_config().pc_sampling_method_value;
    auto unit   = tool::get_config().pc_sampling_unit_value;

    // Find the proper buffer_id based on the method
    auto* buffer_id = (method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP)
                          ? &get_buffers().pc_sampling_host_trap
                          : &get_buffers().pc_sampling_stochastic;

    ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                               buffer_size,
                                               buffer_watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               pc_sampling_cb,
                                               tool_data,
                                               buffer_id),
                     "buffer creation");

    bool config_match_found = false;
    auto agent_ptr_vec      = get_gpu_agents();
    for(auto& itr : agent_ptr_vec)
    {
        if(if_pc_sample_config_match(
               itr->id, method, unit, tool::get_config().pc_sampling_interval))
        {
            config_match_found = true;
            int flags          = 0;
            ROCPROFILER_CALL(
                rocprofiler_configure_pc_sampling_service(get_client_ctx(),
                                                          itr->id,
                                                          method,
                                                          unit,
                                                          tool::get_config().pc_sampling_interval,
                                                          *buffer_id,
                                                          flags),
                "configure PC sampling");
        }
    }
    if(!config_match_found)
    {
        ROCP_ERROR << "Given PC sampling configuration is not supported on any of the agents";
        std::exit(EXIT_FAILURE);
    }
}

struct real_callbacks_t
{};

struct dummy_callbacks_t
{};

constexpr auto use_real_callbacks  = real_callbacks_t{};
constexpr auto use_dummy_callbacks = dummy_callbacks_t{};

struct tracing_callbacks_t
{
    tracing_callbacks_t() = delete;

    tracing_callbacks_t(real_callbacks_t)
    : code_object_tracing{code_object_tracing_callback}
    , cntrl_tracing{cntrl_tracing_callback}
    , kernel_rename{kernel_rename_callback}
    , hip_stream{hip_stream_display_callback}
    , callback_tracing{callback_tracing_callback}
    , buffered_tracing{buffered_tracing_callback}
    , pc_sampling{pc_sampling_callback}
    , att_dispatch{att_dispatch_callback}
    , att_shader_data{att_shader_data_callback}
    , counter_dispatch{counter_dispatch_callback}
    , counter_record{counter_record_callback}
    {}

    explicit tracing_callbacks_t(dummy_callbacks_t)
    : code_object_tracing{dummy_callback_tracing_callback}
    , cntrl_tracing{dummy_callback_tracing_callback}
    , kernel_rename{dummy_callback_tracing_callback}
    , hip_stream{dummy_callback_tracing_callback}
    , callback_tracing{dummy_callback_tracing_callback}
    , buffered_tracing{dummy_buffered_tracing_callback}
    , pc_sampling{dummy_buffered_tracing_callback}
    , counter_dispatch{dummy_counter_dispatch_callback}
    , counter_record{dummy_counter_record_callback}
    {}

    const rocprofiler_callback_tracing_cb_t               code_object_tracing = nullptr;
    const rocprofiler_callback_tracing_cb_t               cntrl_tracing       = nullptr;
    const rocprofiler_callback_tracing_cb_t               kernel_rename       = nullptr;
    const rocprofiler_callback_tracing_cb_t               hip_stream          = nullptr;
    const rocprofiler_callback_tracing_cb_t               callback_tracing    = nullptr;
    const rocprofiler_buffer_tracing_cb_t                 buffered_tracing    = nullptr;
    const rocprofiler_buffer_tracing_cb_t                 pc_sampling         = nullptr;
    const rocprofiler_thread_trace_dispatch_callback_t    att_dispatch        = nullptr;
    const rocprofiler_thread_trace_shader_data_callback_t att_shader_data     = nullptr;
    const rocprofiler_dispatch_counting_service_cb_t      counter_dispatch    = nullptr;
    const rocprofiler_dispatch_counting_record_cb_t       counter_record      = nullptr;
};

auto
get_tracing_callbacks()
{
    // for the benchmarking modes of sdk buffer/callback overhead, we are measuring the cost
    // of the SDK invoking the callbacks to the tool. We do not want to include the overhead
    // of the tool doing any work so we use "dummy" callbacks (i.e. functions which just
    // immediately return)
    if(tool::get_config().benchmark_mode == tool::config::benchmark::sdk_buffered_overhead ||
       tool::get_config().benchmark_mode == tool::config::benchmark::sdk_callback_overhead ||
       tool::get_config().benchmark_mode == tool::config::benchmark::execution_profile)
    {
        return tracing_callbacks_t{use_dummy_callbacks};
    }

    return tracing_callbacks_t{use_real_callbacks};
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    static constexpr auto null_context_id = rocprofiler_context_id_t{.handle = 0};
    static constexpr auto null_buffer_id  = rocprofiler_buffer_id_t{.handle = 0};

    auto _init_timer = common::simple_timer{"[rocprofv3] tool initialization"};

    client_finalizer = fini_func;

    const uint64_t buffer_size      = 16 * common::units::get_page_size();
    const uint64_t buffer_watermark = 15 * common::units::get_page_size();

    tool_metadata->init(tool::metadata::inprocess{});

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "create context failed");

    auto code_obj_ctx = null_context_id;
    ROCPROFILER_CALL(rocprofiler_create_context(&code_obj_ctx), "failed to create context");

    auto start_context = [](rocprofiler_context_id_t ctx_id, std::string_view msg) {
        using benchmark = tool::config::benchmark;
        // do not start context if we are benchmarking the overhead of a service
        // being available but unused by any contexts
        if(tool::get_config().benchmark_mode != benchmark::disabled_contexts_overhead &&
           ctx_id != null_context_id)
        {
            if(tool::get_config().benchmark_mode == benchmark::execution_profile)
            {
                ROCPROFILER_CHECK(rocprofiler_configure_external_correlation_id_request_service(
                    ctx_id, nullptr, 0, record_execution_profile, nullptr));
            }

            ROCP_INFO << fmt::format("starting {} context...", msg);
            ROCPROFILER_CHECK(rocprofiler_start_context(ctx_id));
        }
    };

    auto callbacks = get_tracing_callbacks();

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(code_obj_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       callbacks.code_object_tracing,
                                                       nullptr),
        "code object tracing configure failed");

    start_context(code_obj_ctx, "code object");

    if(tool::get_config().marker_api_trace)
    {
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                             get_client_ctx(),
                             ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                             nullptr,
                             0,
                             callbacks.callback_tracing,
                             nullptr),
                         "callback tracing service failed to configure");

        auto pause_resume_ctx = null_context_id;
        ROCPROFILER_CALL(rocprofiler_create_context(&pause_resume_ctx), "failed to create context");

        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                             pause_resume_ctx,
                             ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                             nullptr,
                             0,
                             callbacks.cntrl_tracing,
                             static_cast<void*>(&get_client_ctx())),
                         "callback tracing service failed to configure");

        start_context(pause_resume_ctx, "marker pause/resume");
    }

    struct buffer_service_config
    {
        bool                              option = false;
        rocprofiler_buffer_tracing_kind_t kind   = ROCPROFILER_BUFFER_TRACING_NONE;
        rocprofiler_buffer_id_t&          buffer_id;
    };

    for(auto&& itr : {buffer_service_config{tool::get_config().kernel_trace,
                                            ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                            get_buffers().kernel_trace},
                      buffer_service_config{tool::get_config().memory_copy_trace,
                                            ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                            get_buffers().memory_copy_trace},
                      buffer_service_config{tool::get_config().scratch_memory_trace,
                                            ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY,
                                            get_buffers().scratch_memory},
                      buffer_service_config{tool::get_config().hsa_core_api_trace,
                                            ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,
                                            get_buffers().hsa_api_trace},
                      buffer_service_config{tool::get_config().hsa_amd_ext_api_trace,
                                            ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,
                                            get_buffers().hsa_api_trace},
                      buffer_service_config{tool::get_config().hsa_image_ext_api_trace,
                                            ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API,
                                            get_buffers().hsa_api_trace},
                      buffer_service_config{tool::get_config().hsa_finalizer_ext_api_trace,
                                            ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API,
                                            get_buffers().hsa_api_trace},
                      buffer_service_config{tool::get_config().hip_runtime_api_trace,
                                            ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT,
                                            get_buffers().hip_api_trace},
                      buffer_service_config{tool::get_config().hip_compiler_api_trace,
                                            ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT,
                                            get_buffers().hip_api_trace},
                      buffer_service_config{tool::get_config().rccl_api_trace,
                                            ROCPROFILER_BUFFER_TRACING_RCCL_API,
                                            get_buffers().rccl_api_trace},
                      buffer_service_config{tool::get_config().memory_allocation_trace,
                                            ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION,
                                            get_buffers().memory_allocation_trace},
                      buffer_service_config{tool::get_config().rocdecode_api_trace,
                                            ROCPROFILER_BUFFER_TRACING_ROCDECODE_API_EXT,
                                            get_buffers().rocdecode_api_trace},
                      buffer_service_config{tool::get_config().rocjpeg_api_trace,
                                            ROCPROFILER_BUFFER_TRACING_ROCJPEG_API,
                                            get_buffers().rocjpeg_api_trace}})
    {
        if(itr.option)
        {
            // in sdk callback overhead benchmarking, we don't want to use the buffer services
            if(tool::get_config().benchmark_mode == tool::config::benchmark::sdk_callback_overhead)
                continue;

            if(itr.buffer_id == null_buffer_id)
            {
                ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                                           buffer_size,
                                                           buffer_watermark,
                                                           ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                           callbacks.buffered_tracing,
                                                           tool_data,
                                                           &itr.buffer_id),
                                 "buffer creation");

                ROCP_FATAL_IF(itr.buffer_id.handle == 0) << "failed to create buffer";

                auto cb_thread = rocprofiler_callback_thread_t{};

                ROCP_INFO << "creating dedicated callback thread for buffer "
                          << itr.buffer_id.handle;
                ROCPROFILER_CALL(rocprofiler_create_callback_thread(&cb_thread),
                                 "creating callback thread");

                ROCP_INFO << "assigning buffer " << itr.buffer_id.handle << " to callback thread "
                          << cb_thread.handle;
                ROCPROFILER_CALL(rocprofiler_assign_callback_thread(itr.buffer_id, cb_thread),
                                 "assigning callback thread");
            }

            ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                                 get_client_ctx(), itr.kind, nullptr, 0, itr.buffer_id),
                             "buffer tracing service configure");
        }
    }

    struct callback_service_config
    {
        bool                                option   = false;
        rocprofiler_callback_tracing_kind_t kind     = ROCPROFILER_CALLBACK_TRACING_NONE;
        rocprofiler_callback_tracing_cb_t   callback = nullptr;
    };

    for(auto&& itr : {callback_service_config{tool::get_config().kernel_trace,
                                              ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                              dummy_callback_tracing_callback},
                      callback_service_config{tool::get_config().memory_copy_trace,
                                              ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                                              dummy_callback_tracing_callback},
                      callback_service_config{tool::get_config().scratch_memory_trace,
                                              ROCPROFILER_CALLBACK_TRACING_SCRATCH_MEMORY,
                                              dummy_callback_tracing_callback},
                      callback_service_config{tool::get_config().hsa_core_api_trace,
                                              ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,
                                              dummy_callback_tracing_callback},
                      callback_service_config{tool::get_config().hsa_amd_ext_api_trace,
                                              ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
                                              dummy_callback_tracing_callback},
                      callback_service_config{tool::get_config().hsa_image_ext_api_trace,
                                              ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API,
                                              dummy_callback_tracing_callback},
                      callback_service_config{tool::get_config().hsa_finalizer_ext_api_trace,
                                              ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
                                              dummy_callback_tracing_callback},
                      callback_service_config{tool::get_config().hip_runtime_api_trace,
                                              ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                              dummy_callback_tracing_callback},
                      callback_service_config{tool::get_config().hip_compiler_api_trace,
                                              ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API,
                                              dummy_callback_tracing_callback},
                      callback_service_config{tool::get_config().rccl_api_trace,
                                              ROCPROFILER_CALLBACK_TRACING_RCCL_API,
                                              dummy_callback_tracing_callback},
                      callback_service_config{tool::get_config().memory_allocation_trace,
                                              ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION,
                                              dummy_callback_tracing_callback},
                      callback_service_config{tool::get_config().rocdecode_api_trace,
                                              ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API,
                                              dummy_callback_tracing_callback},
                      callback_service_config{tool::get_config().rocjpeg_api_trace,
                                              ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API,
                                              dummy_callback_tracing_callback}})
    {
        if(itr.option)
        {
            // in sdk callback overhead benchmarking, we don't want to use the buffer services
            if(tool::get_config().benchmark_mode != tool::config::benchmark::sdk_callback_overhead)
                continue;

            ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                                 get_client_ctx(), itr.kind, nullptr, 0, itr.callback, nullptr),
                             "callback tracing service failed to configure");
        }
    }

    if(tool::get_config().advanced_thread_trace)
    {
        auto     global_parameters = std::vector<rocprofiler_thread_trace_parameter_t>{};
        uint64_t target_cu         = tool::get_config().att_param_target_cu;
        uint64_t simd_select       = tool::get_config().att_param_simd_select;
        uint64_t buffer_sz         = tool::get_config().att_param_buffer_size;
        uint64_t shader_mask       = tool::get_config().att_param_shader_engine_mask;
        uint64_t perfcounter_ctrl  = tool::get_config().att_param_perf_ctrl;
        auto&    att_perf          = tool::get_config().att_param_perfcounters;
        bool     att_serialize_all = tool::get_config().att_serialize_all;

        global_parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, {target_cu}});
        global_parameters.push_back(
            {ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {simd_select}});
        global_parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {buffer_sz}});
        global_parameters.push_back(
            {ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {shader_mask}});
        global_parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SERIALIZE_ALL,
                                     {static_cast<uint64_t>(att_serialize_all)}});

        if(perfcounter_ctrl != 0 && !att_perf.empty())
        {
            global_parameters.push_back(
                {ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTERS_CTRL, {perfcounter_ctrl}});
        }
        else if(perfcounter_ctrl != 0 || !att_perf.empty())
        {
            ROCP_FATAL << "ATT Perf requires setting both perfcounter_ctrl and perfcounter list!";
        }

        for(auto& [id, agent] : tool_metadata->agents_map)
        {
            if(agent.type != ROCPROFILER_AGENT_TYPE_GPU) continue;

            auto agent_params = global_parameters;
            for(auto& counter : get_att_perfcounter_params(id, att_perf))
                agent_params.push_back(counter);

            ROCPROFILER_CALL(
                rocprofiler_configure_dispatch_thread_trace_service(get_client_ctx(),
                                                                    id,
                                                                    agent_params.data(),
                                                                    agent_params.size(),
                                                                    callbacks.att_dispatch,
                                                                    callbacks.att_shader_data,
                                                                    tool_data),
                "thread trace service configure");
        }
    }

    if(tool::get_config().counter_collection)
    {
        ROCPROFILER_CALL(rocprofiler_create_context(&counter_collection_ctx),
                         "failed to create counter collection context");
        ROCPROFILER_CALL(
            rocprofiler_configure_callback_dispatch_counting_service(counter_collection_ctx,
                                                                     callbacks.counter_dispatch,
                                                                     nullptr,
                                                                     callbacks.counter_record,
                                                                     nullptr),
            "Could not setup counting service");

        start_context(counter_collection_ctx, "counter collection");
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
                             callbacks.kernel_rename,
                             nullptr),
                         "callback tracing service failed to configure");

        start_context(rename_ctx, "kernel rename");
    }

    if(!tool::get_config().group_by_queue)
    {
        // Track stream ID information via callback service
        auto hip_stream_display_ctx = rocprofiler_context_id_t{0};

        ROCPROFILER_CALL(rocprofiler_create_context(&hip_stream_display_ctx),
                         "failed to create hip stream context");

        ROCPROFILER_CALL(
            rocprofiler_configure_callback_tracing_service(hip_stream_display_ctx,
                                                           ROCPROFILER_CALLBACK_TRACING_HIP_STREAM,
                                                           nullptr,
                                                           0,
                                                           callbacks.hip_stream,
                                                           nullptr),
            "hip stream tracing configure failed");

        start_context(hip_stream_display_ctx, "hip stream");

        // Track if HIP runtime has been initialized via runtime_intialization service
        auto runtime_initialization_ctx = rocprofiler_context_id_t{0};

        ROCPROFILER_CALL(rocprofiler_create_context(&runtime_initialization_ctx),
                         "failed to create runtime initialization context");

        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                             runtime_initialization_ctx,
                             ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION,
                             nullptr,
                             0,
                             runtime_initialization_callback,
                             nullptr),
                         "runtime initialization tracing configure failed");

        start_context(runtime_initialization_ctx, "runtime initialization");
    }

    if((tool::get_config().kernel_rename || !tool::get_config().group_by_queue) &&
       tool::get_config().benchmark_mode != tool::config::benchmark::execution_profile)
    {
        auto external_corr_id_request_kinds =
            std::array<rocprofiler_external_correlation_id_request_kind_t, 4>{
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION,
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_RUNTIME_API};

        ROCPROFILER_CALL(rocprofiler_configure_external_correlation_id_request_service(
                             get_client_ctx(),
                             external_corr_id_request_kinds.data(),
                             external_corr_id_request_kinds.size(),
                             set_kernel_rename_and_stream_correlation_id,
                             nullptr),
                         "Could not configure external correlation id request service");

        if(tool::get_config().counter_collection)
        {
            auto counter_external_corr_id_request_kinds =
                std::array<rocprofiler_external_correlation_id_request_kind_t, 1>{
                    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH};

            ROCPROFILER_CALL(rocprofiler_configure_external_correlation_id_request_service(
                                 counter_collection_ctx,
                                 counter_external_corr_id_request_kinds.data(),
                                 counter_external_corr_id_request_kinds.size(),
                                 set_kernel_rename_and_stream_correlation_id,
                                 nullptr),
                             "Could not configure external correlation id request service");
        }
    }

    if(tool::get_config().pc_sampling_host_trap)
    {
        configure_pc_sampling_on_all_agents(
            buffer_size, buffer_watermark, tool_data, callbacks.pc_sampling);
    }
    else if(tool::get_config().pc_sampling_stochastic)
    {
        configure_pc_sampling_on_all_agents(
            buffer_size, buffer_watermark, tool_data, callbacks.pc_sampling);
    }

    for(auto itr : get_buffers().pc_sampling_buffers_as_array())
    {
        if(itr > null_buffer_id)
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

    // Handle kernel id of zero
    bool include = std::regex_search("0", std::regex(tool::get_config().kernel_filter_include));
    bool exclude = std::regex_search("0", std::regex(tool::get_config().kernel_filter_exclude));
    if(include && (!exclude || tool::get_config().kernel_filter_exclude.empty()))
        add_kernel_target(0, tool::get_config().kernel_filter_range);

    if(tool::get_config().benchmark_mode == tool::config::benchmark::disabled_contexts_overhead)
    {
        ROCP_INFO << "rocprofv3 is not recording data because the overhead of inactive contexts is "
                     "being benchmarked";
    }
    else if(tool::get_config().selected_regions)
    {
        ROCP_WARNING << "rocprofv3 is only recording profiling data within regions of code "
                        "surrounded by roctxProfilerResume(0)/roctxProfilerPause";
    }
    else if(!tool::get_config().collection_periods.empty())
    {
        ROCP_INFO << "rocprofv3 will record data during the defined collection period(s)";

        auto _prom = std::promise<void>{};
        auto _fut  = _prom.get_future();
        std::thread{collection_period_cntrl, std::move(_prom), get_client_ctx()}.detach();
        _fut.wait_for(std::chrono::seconds{1});  // wait for a max of 1 second
    }
    else
    {
        ROCP_INFO << "rocprofv3 will record data starting now";

        start_context(get_client_ctx(), "primary rocprofv3");
    }

    tool_metadata->set_process_id(getpid(), getppid());
    // set_process_id should set process_start_ns unless it cannot read from /proc/<pid>/stat
    if(tool_metadata->process_start_ns == 0)
        rocprofiler_get_timestamp(&(tool_metadata->process_start_ns));

    return 0;
}

void
api_timestamps_callback(rocprofiler_intercept_table_t table_id,
                        uint64_t                      lib_version,
                        uint64_t                      lib_instance,
                        void** /*tables*/,
                        uint64_t /*num_tables*/,
                        void* /*user_data*/)
{
    static auto _once = std::once_flag{};

    // compute major/minor/patch version info
    uint32_t major = lib_version / 10000;
    uint32_t minor = (lib_version % 10000) / 100;
    uint32_t patch = lib_version % 100;

    const char* table_name = nullptr;
    ROCPROFILER_CHECK(rocprofiler_query_intercept_table_name(table_id, &table_name, nullptr));

    ROCP_WARNING_IF(table_id != ROCPROFILER_MARKER_CONTROL_TABLE &&
                    table_id != ROCPROFILER_MARKER_NAME_TABLE && table_name)
        << fmt::format("{} version {}.{}.{} initialized (instance={})",
                       table_name,
                       major,
                       minor,
                       patch,
                       lib_instance);

    std::call_once(_once, []() {
        if(CHECK_NOTNULL(tool_metadata)->process_start_ns == 0)
            rocprofiler_get_timestamp(&(tool_metadata->process_start_ns));
    });
}

using stats_data_t       = tool::stats_data_t;
using stats_entry_t      = tool::stats_entry_t;
using domain_stats_vec_t = tool::domain_stats_vec_t;
using cleanup_vec_t      = std::vector<std::function<void()>>;

struct output_data
{
    uint64_t num_output = 0;
    uint64_t num_bytes  = 0;
};

void
generate_config_output(const tool::config& cfg, const tool::metadata& tool_metadata_v)
{
    using JSONOutputArchive = ::cereal::PrettyJSONOutputArchive;

    constexpr auto json_prec   = 16;
    constexpr auto json_indent = JSONOutputArchive::Options::IndentChar::space;
    auto           json_opts   = JSONOutputArchive::Options{json_prec, json_indent, 2};
    auto           filename    = std::string_view{"config"};

    auto stream = get_output_stream(cfg, filename, ".json");
    {
        auto archive = JSONOutputArchive{*stream.stream, json_opts};

        archive.setNextName("rocprofiler-sdk-tool");
        archive.startNode();
        archive.makeArray();
        archive.startNode();  // first array entry

        auto timestamps =
            tool::timestamps_t{tool_metadata_v.process_start_ns, tool_metadata_v.process_end_ns};

        auto this_pid = tool_metadata_v.process_id;

        archive.setNextName("metadata");
        archive.startNode();
        archive(cereal::make_nvp("pid", this_pid));
        archive(cereal::make_nvp("init_time", timestamps.app_start_time));
        archive(cereal::make_nvp("fini_time", timestamps.app_end_time));
        archive(cereal::make_nvp("config", cfg));
        archive(cereal::make_nvp("command", common::read_command_line(this_pid)));

        {
            archive.setNextName("build_spec");
            archive.startNode();
            archive(cereal::make_nvp("version_major", ROCPROFILER_VERSION_MAJOR));
            archive(cereal::make_nvp("version_minor", ROCPROFILER_VERSION_MINOR));
            archive(cereal::make_nvp("version_patch", ROCPROFILER_VERSION_PATCH));
            archive(cereal::make_nvp("soversion", ROCPROFILER_SOVERSION));
            archive(cereal::make_nvp("compiler_id", std::string{ROCPROFILER_COMPILER_ID}));
            archive(
                cereal::make_nvp("compiler_version", std::string{ROCPROFILER_COMPILER_VERSION}));
            archive(cereal::make_nvp("git_describe", std::string{ROCPROFILER_GIT_DESCRIBE}));
            archive(cereal::make_nvp("git_revision", std::string{ROCPROFILER_GIT_REVISION}));
            archive(cereal::make_nvp("library_arch", std::string{ROCPROFILER_LIBRARY_ARCH}));
            archive(cereal::make_nvp("system_name", std::string{ROCPROFILER_SYSTEM_NAME}));
            archive(
                cereal::make_nvp("system_processor", std::string{ROCPROFILER_SYSTEM_PROCESSOR}));
            archive(cereal::make_nvp("system_version", std::string{ROCPROFILER_SYSTEM_VERSION}));
            archive.finishNode();  // build_spec
        }

        // save the execution profile
        if(execution_profile) archive(cereal::make_nvp("profile", execution_profile->get()));

        // save the environment variables
        {
            archive.setNextName("environment");
            archive.startNode();
            size_t idx = 0;
            while(true)
            {
                const auto* env_entry = environ[idx++];
                if(!env_entry)
                    break;
                else if(std::string_view{env_entry}.find('=') != std::string_view::npos)
                {
                    auto _entry = std::string{env_entry};
                    auto _pos   = _entry.find('=');
                    auto _name  = _entry.substr(0, _pos);
                    auto _value = _entry.substr(_pos + 1);
                    archive(cereal::make_nvp(_name.c_str(), _value));
                }
            }
            archive.finishNode();
        }

        archive.finishNode();  // metadata
        archive.finishNode();  // first array entry
        archive.finishNode();  // rocprofiler-sdk-tool
    }
    stream.close();
}

template <typename Tp, domain_type DomainT>
void
generate_output(tool::buffered_output<Tp, DomainT>& output_v,
                output_data&                        output_data_v,
                domain_stats_vec_t&                 contributions_v,
                cleanup_vec_t&                      cleanups_v)
{
    cleanups_v.emplace_back([&output_v]() { output_v.destroy(); });

    if(!output_v) return;

    // when benchmarking, we do not generate output
    if(tool::get_config().benchmark_mode != tool::config::benchmark::none) return;

    // opens temporary file and sets read position to beginning
    output_v.read();

    if(output_v.get_generator().empty()) return;

    // if it has reached this point, the generator is not empty
    auto _num_bytes = output_v.get_num_bytes();
    output_data_v.num_output += 1;
    output_data_v.num_bytes += _num_bytes;

    if(tool::get_config().stats || tool::get_config().summary_output)
    {
        output_v.stats =
            tool::generate_stats(tool::get_config(), *tool_metadata, output_v.get_generator());
    }

    if(output_v.stats)
    {
        contributions_v.emplace_back(output_v.buffer_type_v, output_v.stats);
    }

    if(tool::get_config().csv_output && _num_bytes >= tool::get_config().minimum_output_bytes)
    {
        tool::generate_csv(
            tool::get_config(), *tool_metadata, output_v.get_generator(), output_v.stats);
    }
}

void
tool_fini(void* /*tool_data*/)
{
    static bool _first = true;
    if(!_first) return;
    _first = false;

    client_identifier = nullptr;
    client_finalizer  = nullptr;

    auto _fini_timer = common::simple_timer{"[rocprofv3] tool finalization"};

    if(tool_metadata->process_end_ns == 0)
        rocprofiler_get_timestamp(&(tool_metadata->process_end_ns));

    flush();
    rocprofiler_stop_context(get_client_ctx());
    flush();

    auto kernel_dispatch_output =
        rocprofiler::tool::kernel_dispatch_buffered_output_ext_t{tool::get_config().kernel_trace};

    auto hsa_output = tool::hsa_buffered_output_t{tool::get_config().hsa_core_api_trace ||
                                                  tool::get_config().hsa_amd_ext_api_trace ||
                                                  tool::get_config().hsa_image_ext_api_trace ||
                                                  tool::get_config().hsa_finalizer_ext_api_trace};
    auto hip_output = tool::hip_buffered_output_t{tool::get_config().hip_runtime_api_trace ||
                                                  tool::get_config().hip_compiler_api_trace};
    auto memory_copy_output =
        tool::memory_copy_buffered_output_ext_t{tool::get_config().memory_copy_trace};
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
    auto rocdecode_output =
        tool::rocdecode_buffered_output_t{tool::get_config().rocdecode_api_trace};
    auto rocjpeg_output = tool::rocjpeg_buffered_output_t{tool::get_config().rocjpeg_api_trace};
    auto pc_sampling_stochastic_output =
        tool::pc_sampling_stochastic_buffered_output_t{tool::get_config().pc_sampling_stochastic};

    auto node_id_sort  = [](const auto& lhs, const auto& rhs) { return lhs.node_id < rhs.node_id; };
    auto agents_output = CHECK_NOTNULL(tool_metadata)->agents;
    std::sort(agents_output.begin(), agents_output.end(), node_id_sort);

    auto outdata       = output_data{};
    auto contributions = domain_stats_vec_t{};
    auto cleanups      = cleanup_vec_t{};

    auto run_cleanup = [&cleanups]() {
        for(const auto& itr : cleanups)
        {
            if(itr) itr();
        }
        cleanups.clear();
    };

    // generate the configuration output regardless of whether there is any data
    if(tool::get_config().output_config_file)
    {
        generate_config_output(tool::get_config(), *tool_metadata);
    }

    auto _dtor = common::scope_destructor{run_cleanup};

    generate_output(kernel_dispatch_output, outdata, contributions, cleanups);
    generate_output(hsa_output, outdata, contributions, cleanups);
    generate_output(hip_output, outdata, contributions, cleanups);
    generate_output(memory_copy_output, outdata, contributions, cleanups);
    generate_output(memory_allocation_output, outdata, contributions, cleanups);
    generate_output(marker_output, outdata, contributions, cleanups);
    generate_output(rccl_output, outdata, contributions, cleanups);
    generate_output(counters_output, outdata, contributions, cleanups);
    generate_output(scratch_memory_output, outdata, contributions, cleanups);
    generate_output(rocdecode_output, outdata, contributions, cleanups);
    generate_output(pc_sampling_host_trap_output, outdata, contributions, cleanups);
    generate_output(rocjpeg_output, outdata, contributions, cleanups);
    generate_output(pc_sampling_stochastic_output, outdata, contributions, cleanups);

    if(tool::get_config().advanced_thread_trace && !tool_metadata->att_filenames.empty())
    {
        outdata.num_output += 1;
    }

    ROCP_INFO << fmt::format("Number of services generating output: {} ({} kB)",
                             outdata.num_output,
                             (outdata.num_bytes / 1024));

    if(tool::get_config().csv_output && outdata.num_output > 0 &&
       outdata.num_bytes >= tool::get_config().minimum_output_bytes)
    {
        tool::generate_csv(tool::get_config(), *tool_metadata, agents_output);
    }

    if(tool::get_config().stats && tool::get_config().csv_output && outdata.num_output > 0 &&
       outdata.num_bytes >= tool::get_config().minimum_output_bytes)
    {
        tool::generate_csv(tool::get_config(), *tool_metadata, contributions);
    }

    if(tool::get_config().json_output && outdata.num_output > 0 &&
       outdata.num_bytes >= tool::get_config().minimum_output_bytes)
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
                         rocdecode_output.get_generator(),
                         rocjpeg_output.get_generator(),
                         pc_sampling_host_trap_output.get_generator(),
                         pc_sampling_stochastic_output.get_generator());
        json_ar.finish_process();

        tool::close_json(json_ar);
    }

    if(tool::get_config().pftrace_output && outdata.num_output > 0 &&
       outdata.num_bytes >= tool::get_config().minimum_output_bytes)
    {
        tool::write_perfetto(tool::get_config(),
                             *tool_metadata,
                             agents_output,
                             hip_output.get_generator(),
                             hsa_output.get_generator(),
                             kernel_dispatch_output.get_generator(),
                             memory_copy_output.get_generator(),
                             counters_output.get_generator(),
                             marker_output.get_generator(),
                             scratch_memory_output.get_generator(),
                             rccl_output.get_generator(),
                             memory_allocation_output.get_generator(),
                             rocdecode_output.get_generator(),
                             rocjpeg_output.get_generator());
    }

    if(tool::get_config().rocpd_output && outdata.num_output > 0 &&
       outdata.num_bytes >= tool::get_config().minimum_output_bytes)
    {
        tool::write_rocpd(tool::get_config(),
                          *tool_metadata,
                          agents_output,
                          hip_output.get_generator(),
                          hsa_output.get_generator(),
                          kernel_dispatch_output.get_generator(),
                          memory_copy_output.get_generator(),
                          marker_output.get_generator(),
                          memory_allocation_output.get_generator(),
                          scratch_memory_output.get_generator(),
                          rccl_output.get_generator(),
                          rocdecode_output.get_generator(),
                          counters_output.get_generator());
    }

    if(tool::get_config().otf2_output && outdata.num_output > 0 &&
       outdata.num_bytes >= tool::get_config().minimum_output_bytes)
    {
        auto hip_elem_data               = hip_output.load_all();
        auto hsa_elem_data               = hsa_output.load_all();
        auto kernel_dispatch_elem_data   = kernel_dispatch_output.load_all();
        auto memory_copy_elem_data       = memory_copy_output.load_all();
        auto marker_elem_data            = marker_output.load_all();
        auto scratch_memory_elem_data    = scratch_memory_output.load_all();
        auto rccl_elem_data              = rccl_output.load_all();
        auto memory_allocation_elem_data = memory_allocation_output.load_all();
        auto rocdecode_elem_data         = rocdecode_output.load_all();
        auto rocjpeg_elem_data           = rocjpeg_output.load_all();

        tool::write_otf2(tool::get_config(),
                         *tool_metadata,
                         getpid(),
                         agents_output,
                         &hip_elem_data,
                         &hsa_elem_data,
                         &kernel_dispatch_elem_data,
                         &memory_copy_elem_data,
                         &marker_elem_data,
                         &scratch_memory_elem_data,
                         &rccl_elem_data,
                         &memory_allocation_elem_data,
                         &rocdecode_elem_data,
                         &rocjpeg_elem_data);
    }

    if(tool::get_config().summary_output && outdata.num_output > 0 &&
       outdata.num_bytes >= tool::get_config().minimum_output_bytes)
    {
        tool::generate_stats(tool::get_config(), *tool_metadata, contributions);
    }

    if(tool::get_config().advanced_thread_trace)
    {
        auto decoder = rocprofiler::att_wrapper::ATTDecoder(tool::get_config().att_library_path);
        ROCP_FATAL_IF(!decoder.valid()) << "Decoder library not found!";

        auto codeobj     = tool_metadata->get_code_object_load_info();
        auto output_path = tool::format_path(tool::get_config().output_path);

        std::vector<std::string> perf{};
        for(auto& counter : tool::get_config().att_param_perfcounters)
        {
            std::stringstream ss;
            ss << counter.counter_name;

            if(counter.simd_mask != 0xF) ss << ':' << std::hex << counter.simd_mask;

            perf.emplace_back(ss.str());
        }

        for(auto& [dispatch_id, att_filename_data] : tool_metadata->att_filenames)
        {
            std::string formats = "json,csv";

            auto ui_name = std::stringstream{};
            ui_name << fmt::format("ui_output_agent_{}_dispatch_{}",
                                   std::to_string(att_filename_data.first.handle),
                                   dispatch_id);
            auto out_path = fmt::format("{}/{}", output_path, ui_name.str());
            auto in_path  = std::string(".");

            decoder.parse(in_path, out_path, att_filename_data.second, codeobj, perf, formats);
        }
    }

    run_cleanup();

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

std::vector<rocprofiler_counter_record_dimension_info_t>
get_tool_counter_dimension_info()
{
    auto _data = get_agent_counter_info();
    auto _ret  = std::vector<rocprofiler_counter_record_dimension_info_t>{};
    for(const auto& itr : _data)
    {
        for(const auto& iitr : itr.second)
            for(const auto& ditr : iitr.dimensions)
                _ret.emplace_back(ditr);
    }

    auto _sorter = [](const rocprofiler_counter_record_dimension_info_t& lhs,
                      const rocprofiler_counter_record_dimension_info_t& rhs) {
        return std::tie(lhs.id, lhs.instance_size) < std::tie(rhs.id, rhs.instance_size);
    };
    auto _equiv = [](const rocprofiler_counter_record_dimension_info_t& lhs,
                     const rocprofiler_counter_record_dimension_info_t& rhs) {
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

signal_func_t&
get_signal_function()
{
    static signal_func_t user_signal = nullptr;
    return user_signal;
}

sigaction_func_t&
get_sigaction_function()
{
    static sigaction_func_t user_sigaction = (sigaction_func_t) dlsym(RTLD_NEXT, "sigaction");
    return user_sigaction;
}

bool signal_handler_exit =
    rocprofiler::tool::get_env("ROCPROF_INTERNAL_TEST_SIGNAL_HANDLER_VIA_EXIT", false);
}  // namespace

#define ROCPROFV3_INTERNAL_API __attribute__((visibility("internal")));

std::optional<int>
wait_pid(pid_t _pid, int _opts = 0)
{
    auto this_pid  = getpid();
    auto this_ppid = getppid();
    auto this_tid  = common::get_tid();
    auto this_func = std::string_view{__FUNCTION__};

    ROCP_INFO << fmt::format("[PPID={}][PID={}][TID={}][{}] rocprofv3 waiting for child {}",
                             this_ppid,
                             this_pid,
                             this_tid,
                             this_func,
                             _pid);

    int   _status = 0;
    pid_t _pid_v  = -1;
    _opts |= WUNTRACED;
    do
    {
        if((_opts & WNOHANG) > 0)
        {
            std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
        }
        _pid_v = waitpid(_pid, &_status, _opts);
    } while(_pid_v == 0);

    if(_pid_v < 0) return std::nullopt;
    return _status;
}

extern "C" {
void
rocprofv3_set_main(main_func_t main_func) ROCPROFV3_INTERNAL_API;

int
diagnose_status(pid_t _pid, int _status)
{
    auto this_pid  = getpid();
    auto this_ppid = getppid();
    auto this_tid  = common::get_tid();
    auto this_func = std::string_view{__FUNCTION__};

    bool _normal_exit      = (WIFEXITED(_status) > 0);
    bool _unhandled_signal = (WIFSIGNALED(_status) > 0);
    bool _core_dump        = (WCOREDUMP(_status) > 0);
    bool _stopped          = (WIFSTOPPED(_status) > 0);
    int  _exit_status      = WEXITSTATUS(_status);
    int  _stop_signal      = (_stopped) ? WSTOPSIG(_status) : 0;
    int  _ec               = (_unhandled_signal) ? WTERMSIG(_status) : 0;

    ROCP_TRACE << fmt::format("[PPID={}][PID={}][TID={}][{}] diagnosing status for process {} :: "
                              "status: {}, normal exit: {}, unhandled signal: {}, core dump: {}, "
                              "stopped: {}, exit status: {}, stop signal: {}, exit code: {}",
                              this_ppid,
                              this_pid,
                              this_tid,
                              this_func,
                              _pid,
                              _status,
                              std::to_string(static_cast<int>(_normal_exit)),
                              std::to_string(static_cast<int>(_unhandled_signal)),
                              std::to_string(static_cast<int>(_core_dump)),
                              std::to_string(static_cast<int>(_stopped)),
                              _exit_status,
                              _stop_signal,
                              _ec);

    if(!_normal_exit)
    {
        if(_ec == 0) _ec = EXIT_FAILURE;
        ROCP_INFO << fmt::format(
            "[PPID={}][PID={}][TID={}][{}] process {} terminated abnormally. exit code: {}",
            this_ppid,
            this_pid,
            this_tid,
            this_func,
            _pid,
            _ec);
    }

    if(_stopped)
    {
        ROCP_INFO << fmt::format(
            "[PPID={}][PID={}][TID={}][{}] process {} stopped with signal {}. exit code: {}",
            this_ppid,
            this_pid,
            this_tid,
            this_func,
            _pid,
            _stop_signal,
            _ec);
    }

    if(_core_dump)
    {
        ROCP_INFO << fmt::format("[PPID={}][PID={}][TID={}][{}] process {} terminated and "
                                 "produced a core dump. exit code: {}",
                                 this_ppid,
                                 this_pid,
                                 this_tid,
                                 this_func,
                                 _pid,
                                 _ec);
    }

    if(_unhandled_signal)
    {
        ROCP_INFO << fmt::format(
            "[PPID={}][PID={}][TID={}][{}] process {} terminated because it received a signal "
            "({}) that was not handled. exit code: {}",
            this_ppid,
            this_pid,
            this_tid,
            this_func,
            _pid,
            _ec,
            _ec);
    }

    if(!_normal_exit && _exit_status > 0)
    {
        if(_exit_status == 127)
        {
            ROCP_INFO << fmt::format(
                "[PPID={}][PID={}][TID={}][{}] execv in process {} failed. exit code: {}",
                this_ppid,
                this_pid,
                this_tid,
                this_func,
                _pid,
                _ec);
        }
        else
        {
            ROCP_INFO << fmt::format("[PPID={}][PID={}][TID={}][{}] process {} terminated with "
                                     "a non-zero status. exit code: {}",
                                     this_ppid,
                                     this_pid,
                                     this_tid,
                                     this_func,
                                     _pid,
                                     _ec);
        }
    }

    return _ec;
}

void
rocprofv3_error_signal_handler(int signo, siginfo_t* info, void* ucontext)
{
    auto this_pid  = getpid();
    auto this_ppid = getppid();
    auto this_tid  = common::get_tid();
    auto this_func = std::string_view{__FUNCTION__};

    ROCP_WARNING << fmt::format("[PPID={}][PID={}][TID={}][{}] rocprofv3 caught signal {}...",
                                this_ppid,
                                this_pid,
                                this_tid,
                                this_func,
                                signo);

    static auto _once = std::once_flag{};
    std::call_once(_once, [&]() {
        auto get_children = [&this_pid]() {
            auto fname    = fmt::format("/proc/{}/task/{}/children", this_pid, this_pid);
            auto ifs      = std::ifstream{fname};
            auto children = std::vector<pid_t>{};
            while(ifs)
            {
                pid_t val = 0;
                ifs >> val;
                if(ifs && !ifs.eof() && val > 0) children.emplace_back(val);
            }
            return children;
        };

        auto _children = get_children();
        ROCP_WARNING << fmt::format(
            "[PPID={}][PID={}][TID={}][{}] rocprofv3 will wait for {} children to exit",
            this_ppid,
            this_pid,
            this_tid,
            this_func,
            _children.size());

        // wait for children
        for(auto itr : _children)
        {
            auto status = wait_pid(itr, WUNTRACED | WNOHANG);
            if(status) diagnose_status(itr, status.value());
        }

        ROCP_WARNING << fmt::format(
            "[PPID={}][PID={}][TID={}][{}] rocprofv3 finalizing after signal {}...",
            this_ppid,
            this_pid,
            this_tid,
            this_func,
            signo);

        finalize_rocprofv3(this_func);

        ROCP_INFO << fmt::format(
            "[PPID={}][PID={}][TID={}][{}] rocprofv3 finalizing after signal {}... complete",
            this_ppid,
            this_pid,
            this_tid,
            this_func,
            signo);

        if(get_chained_signals().at(signo))
        {
            ROCP_INFO << fmt::format(
                "[PPID={}][PID={}][TID={}][{}] rocprofv3 found chained signal handler for {}",
                this_ppid,
                this_pid,
                this_tid,
                this_func,
                signo);

            if(auto& _chained = *get_chained_signals().at(signo); _chained.action)
            {
                ROCP_TRACE << fmt::format("[PPID={}][PID={}][TID={}][{}] rocprofv3 found chained "
                                          "signal handler for {}... executing chained sigaction",
                                          this_ppid,
                                          this_pid,
                                          this_tid,
                                          this_func,
                                          signo);
                if((_chained.action->sa_flags & SA_SIGINFO) == SA_SIGINFO &&
                   _chained.action->sa_sigaction &&
                   _chained.action->sa_sigaction != &rocprofv3_error_signal_handler)
                {
                    ROCP_WARNING << fmt::format(
                        "[PPID={}][PID={}][TID={}][{}] rocprofv3 found chained signal handler for "
                        "{}... executing chained sigaction (SIGINFO)",
                        this_ppid,
                        this_pid,
                        this_tid,
                        this_func,
                        signo);
                    _chained.action->sa_sigaction(signo, info, ucontext);
                }
                else if((_chained.action->sa_flags & SA_SIGINFO) != SA_SIGINFO &&
                        _chained.action->sa_handler &&
                        _chained.action->sa_sigaction != &rocprofv3_error_signal_handler)
                {
                    ROCP_WARNING << fmt::format(
                        "[PPID={}][PID={}][TID={}][{}] rocprofv3 found chained signal handler for "
                        "{}... executing chained sigaction (HANDLER)",
                        this_ppid,
                        this_pid,
                        this_tid,
                        this_func,
                        signo);
                    _chained.action->sa_handler(signo);
                }
            }
            else
            {
                if(_chained.handler)
                {
                    ROCP_WARNING << fmt::format(
                        "[PPID={}][PID={}][TID={}][{}] rocprofv3 found chained signal handler for "
                        "{}... executing chained handler",
                        this_ppid,
                        this_pid,
                        this_tid,
                        this_func,
                        signo);
                    _chained.handler(signo);
                }
            }
        }
    });

    // below is for testing purposes. re-raising the signal causes CTest to ignore WILL_FAIL ON
    if(signal_handler_exit) ::quick_exit(signo);
    ::raise(signo);
}

int
rocprofv3_main(int argc, char** argv, char** envp) ROCPROFV3_INTERNAL_API;

sighandler_t
rocprofv3_signal(int signum, sighandler_t handler) ROCPROFV3_INTERNAL_API;

int
rocprofv3_sigaction(int signum,
                    const struct sigaction* __restrict__ act,
                    struct sigaction* __restrict__ oldact) ROCPROFV3_INTERNAL_API;

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
    add_destructor(execution_profile);

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

    int libs = ROCPROFILER_HSA_TABLE;
    if(tool::get_config().hip_compiler_api_trace) libs |= ROCPROFILER_HIP_COMPILER_TABLE;
    if(tool::get_config().hip_runtime_api_trace) libs |= ROCPROFILER_HIP_RUNTIME_TABLE;
    if(tool::get_config().rccl_api_trace) libs |= ROCPROFILER_RCCL_TABLE;
    if(tool::get_config().marker_api_trace) libs |= ROCPROFILER_MARKER_CORE_TABLE;
    if(tool::get_config().rocdecode_api_trace) libs |= ROCPROFILER_ROCDECODE_TABLE;
    if(tool::get_config().rocjpeg_api_trace) libs |= ROCPROFILER_ROCJPEG_TABLE;

    ROCPROFILER_CALL(
        rocprofiler_at_intercept_table_registration(api_timestamps_callback, libs, nullptr),
        "api registration");

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

#define LOG_FUNCTION_ENTRY(MSG, ...)                                                               \
    {                                                                                              \
        ROCP_INFO << fmt::format("[PPID={}][PID={}][TID={}][rocprofv3] {}" MSG,                    \
                                 getppid(),                                                        \
                                 getpid(),                                                         \
                                 rocprofiler::common::get_tid(),                                   \
                                 __FUNCTION__,                                                     \
                                 __VA_ARGS__);                                                     \
    }

sighandler_t
rocprofv3_signal(int signum, sighandler_t handler)
{
    static auto _once = std::once_flag{};
    std::call_once(_once,
                   []() { get_signal_function() = (signal_func_t) dlsym(RTLD_NEXT, "signal"); });

    if(!is_handled_signal(signum) || !tool::get_config().enable_signal_handlers)
        return CHECK_NOTNULL(get_signal_function())(signum, handler);

    get_chained_signals().at(signum) = chained_siginfo{signum, handler, std::nullopt};

    return get_signal_function()(
        signum, [](int signum_v) { rocprofv3_error_signal_handler(signum_v, nullptr, nullptr); });
}

int
rocprofv3_sigaction(int signum,
                    const struct sigaction* __restrict__ act,
                    struct sigaction* __restrict__ oldact)
{
    static auto _once = std::once_flag{};
    std::call_once(_once, []() {
        get_sigaction_function() = (sigaction_func_t) dlsym(RTLD_NEXT, "sigaction");
    });

    if(!is_handled_signal(signum) || !act || !tool::get_config().enable_signal_handlers)
        return CHECK_NOTNULL(get_sigaction_function())(signum, act, oldact);

    // make sure rocprofv3_error_signal_handler doesn't call itself
    if((act->sa_flags & SA_SIGINFO) == SA_SIGINFO &&
       act->sa_sigaction != &rocprofv3_error_signal_handler)
        get_chained_signals().at(signum) = chained_siginfo{signum, nullptr, *act};

    struct sigaction _upd_act = *act;
    _upd_act.sa_flags |= (SA_SIGINFO | SA_RESETHAND | SA_NOCLDSTOP);
    _upd_act.sa_sigaction = &rocprofv3_error_signal_handler;

    return get_sigaction_function()(signum, &_upd_act, oldact);
}

int
rocprofv3_main(int argc, char** argv, char** envp)
{
    auto convert_to_vec = [](char** inp) {
        auto        _data = std::vector<std::string_view>{};
        size_t      n     = 0;
        const char* p     = nullptr;
        if(!inp) return _data;
        do
        {
            p = inp[n++];
            if(p != nullptr) _data.emplace_back(p);
        } while(p != nullptr);
        return _data;
    };

    auto _argv = convert_to_vec(argv);
    // auto _envp = convect_to_vec(envp);

    LOG_FUNCTION_ENTRY("({}, '{}', ...)", argc, fmt::join(_argv.begin(), _argv.end(), " "));

    initialize_logging();

    initialize_rocprofv3();

    initialize_signal_handler(get_sigaction_function());

    ROCP_INFO << "rocprofv3: main function wrapper will be invoked...";

    auto _main_timer = std::optional<common::simple_timer>{};

    // should never happen but if it does, don't time
    if(!_argv.empty())
        _main_timer = common::simple_timer{
            fmt::format("[rocprofv3] '{}'", fmt::join(_argv.begin(), _argv.end(), " "))};

    if(tool_metadata && tool_metadata->process_start_ns == 0)
        rocprofiler_get_timestamp(&(tool_metadata->process_start_ns));

    auto ret = CHECK_NOTNULL(get_main_function())(argc, argv, envp);

    if(tool_metadata && tool_metadata->process_end_ns == 0)
        rocprofiler_get_timestamp(&(tool_metadata->process_end_ns));

    ROCP_INFO << "rocprofv3: main function has returned with exit code: " << ret;

    // reset so that it reports the timing
    if(_main_timer) _main_timer.reset();

    finalize_rocprofv3(__FUNCTION__);

    ROCP_INFO << "rocprofv3 finished. exit code: " << ret;
    return ret;
}
}
