// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/rocprofiler-sdk.hpp"
#include "api.hpp"
#include "common/synchronized.hpp"
#include "core/config.hpp"
#include "core/containers/stable_vector.hpp"
#include "core/debug.hpp"
#include "core/gpu.hpp"
#include "core/perfetto.hpp"
#include "core/rocprofiler-sdk.hpp"
#include "core/state.hpp"
#include "library/amd_smi.hpp"
#include "library/components/category_region.hpp"
#include "library/rocprofiler-sdk/counters.hpp"
#include "library/rocprofiler-sdk/fwd.hpp"
#include "library/rocprofiler-sdk/rccl.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"

#include <timemory/components/timing/wall_clock.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/unwind/processed_entry.hpp>
#include <timemory/variadic/lightweight_tuple.hpp>

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <timemory/defines.h>
#include <timemory/process/threading.hpp>
#include <timemory/utility/demangle.hpp>
#include <timemory/utility/types.hpp>

#include <atomic>
#include <cctype>
#include <cstdint>
#include <deque>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace
{
using tool_agent_vec_t = std::vector<tool_agent>;
client_data* tool_data = new client_data{};

void
thread_precreate(rocprofiler_runtime_library_t /*lib*/, void* /*tool_data*/)
{
    push_thread_state(ThreadState::Internal);
}

void
thread_postcreate(rocprofiler_runtime_library_t /*lib*/, void* /*tool_data*/)
{
    pop_thread_state();
}

#if(ROCPROFILER_VERSION < 700)
/**
 * @brief Stream ID.
 */
typedef struct rocprofiler_stream_id_t
{
    uint64_t handle;
} rocprofiler_stream_id_t;

#endif

auto&
get_stream_stack()
{
    static thread_local std::vector<rocprofiler_stream_id_t> _v{ rocprofiler_stream_id_t{
        0 } };
    return _v;
}

void
stream_id_push(rocprofiler_stream_id_t stream_id)
{
    get_stream_stack().emplace_back(stream_id);
}

rocprofiler_stream_id_t
stream_id_top()
{
    auto stream_id = get_stream_stack().back();
    return stream_id;
}

void
stream_id_pop()
{
    get_stream_stack().pop_back();
}

// Stores stream ids and kernel region ids for kernel-rename service and hip stream
// display service
struct kernel_rename_and_stream_data
{
    uint64_t                region_id = 0;  // roctx region correlation id
    rocprofiler_stream_id_t stream_id = { 0 };
};

template <typename Tp>
rocprofiler_stream_id_t
get_stream_id(Tp* _record)
{
    auto _stream_id = rocprofiler_stream_id_t{ 0 };
    if(_record->correlation_id.external.ptr != nullptr)
    {
        // Extract the stream id
        auto* _ecid_data = static_cast<kernel_rename_and_stream_data*>(
            _record->correlation_id.external.ptr);
        _stream_id                             = _ecid_data->stream_id;
        auto _region_id                        = _ecid_data->region_id;
        _record->correlation_id.external.value = _region_id;
        delete _ecid_data;
    }
    return _stream_id;
}

// this function creates a rocprofiler profile config on the first entry
std::vector<rocprofiler_counter_id_t>
create_agent_profile(rocprofiler_agent_id_t          agent_id,
                     const std::vector<std::string>& counters,
                     //  const tool_agent_vec_t&         gpu_agents,
                     //  const agent_counter_info_map_t& counters_info,
                     //  agent_counter_profile_map_t&    data)
                     client_data* data = tool_data)
{
    using counter_vec_t = std::vector<rocprofiler_counter_id_t>;

    // check if already created
    if(data->agent_counter_profiles.find(agent_id) != data->agent_counter_profiles.end())
        return counter_vec_t{};

    auto        profile      = std::optional<rocprofiler_profile_config_id_t>{};
    auto        expected_v   = counters.size();
    auto        found_v      = std::vector<std::string_view>{};
    auto        counters_v   = counter_vec_t{};
    const auto* tool_agent_v = data->get_gpu_tool_agent(agent_id);

    constexpr auto device_qualifier = std::string_view{ ":device=" };
    for(const auto& itr : counters)
    {
        auto name_v = itr;
        if(auto pos = std::string::npos;
           (pos = itr.find(device_qualifier)) != std::string::npos)
        {
            name_v        = itr.substr(0, pos);
            auto dev_id_s = itr.substr(pos + device_qualifier.length());

            ROCPROFSYS_CONDITIONAL_ABORT(dev_id_s.empty() ||
                                             dev_id_s.find_first_not_of("0123456789") !=
                                                 std::string::npos,
                                         "invalid device qualifier format (':device=N) "
                                         "where N is the GPU id: %s\n",
                                         itr.c_str());

            auto dev_id_v = std::stoul(dev_id_s);

            ROCPROFSYS_PRINT_F("tool agent device id=%lu, name=%s, device_id=%lu\n",
                               tool_agent_v->device_id, name_v.c_str(), dev_id_v);

            // skip this counter if the counter is for a specific device id (which
            // doesn't this agent's device id)
            if(dev_id_v != tool_agent_v->device_id)
            {
                --expected_v;  // is not expected
                continue;
            }
        }

        // Removes any numeric index enclosed in square brackets at the end of the string.
        // For example, "example[123]" will be converted to "example".
        auto _old_name_v = name_v;
        name_v =
            std::regex_replace(name_v, std::regex{ "^(.*)(\\[)([0-9]+)(\\])$" }, "$1");

        if(name_v != _old_name_v)
        {
            ROCPROFSYS_PRINT_F("tool agent device id=%lu, old_name=%s, name=%s\n",
                               tool_agent_v->device_id, _old_name_v.c_str(),
                               name_v.c_str());
        }
        else if(name_v == itr)
        {
            ROCPROFSYS_PRINT_F("tool agent device id=%lu, name=%s\n",
                               tool_agent_v->device_id, name_v.c_str());
        }

        // search the gpu agent counter info for a counter with a matching name
        for(const auto& citr : data->agent_counter_info.at(agent_id))
        {
            if(name_v == std::string_view{ citr.name })
            {
                counters_v.emplace_back(citr.id);
                found_v.emplace_back(itr);
            }
        }
    }

    if(counters_v.size() != expected_v)
    {
        auto requested_counters =
            timemory::join::join(timemory::join::array_config{ ", ", "", "" }, counters);
        auto found_counters =
            timemory::join::join(timemory::join::array_config{ ", ", "", "" }, found_v);

        ROCPROFSYS_ABORT_F(
            "Unable to find all counters for agent %i (gpu-%li, %s) in %s. Found: %s\n",
            tool_agent_v->agent->node_id, tool_agent_v->device_id,
            tool_agent_v->agent->name, requested_counters.c_str(),
            found_counters.c_str());
    }

    if(!counters_v.empty())
    {
        auto profile_v = rocprofiler_profile_config_id_t{};
        ROCPROFILER_CALL(rocprofiler_create_profile_config(
            agent_id, counters_v.data(), counters_v.size(), &profile_v));
        profile = profile_v;
    }

    data->agent_counter_profiles.emplace(agent_id, profile);

    return counters_v;
}

const kernel_symbol_data_t*
get_kernel_symbol_info(uint64_t _kernel_id)
{
    return tool_data->get_kernel_symbol_info(_kernel_id);
}

// Implementation of rocprofiler_callback_tracing_operation_args_cb_t
int
save_args(rocprofiler_callback_tracing_kind_t /*kind*/, int32_t /*operation*/,
          uint32_t /*arg_number*/, const void* const /*arg_value_addr*/,
          int32_t /*arg_indirection_count*/, const char* /*arg_type*/,
          const char* arg_name, const char*        arg_value_str,
          int32_t /*arg_dereference_count*/, void* data)
{
    auto* argvec = static_cast<callback_arg_array_t*>(data);
    argvec->emplace_back(arg_name, arg_value_str);
    return 0;
}

auto&
get_marker_pushed_ranges()
{
    static thread_local auto _v =
        std::vector<std::pair<tim::hash_value_t, rocprofiler_timestamp_t>>{};
    return _v;
}

auto&
get_marker_started_ranges()
{
    static thread_local auto _v =
        std::vector<std::pair<tim::hash_value_t, rocprofiler_timestamp_t>>{};
    return _v;
}

template <typename CategoryT>
void
tool_tracing_callback_start(CategoryT, rocprofiler_callback_tracing_record_t record,
                            rocprofiler_user_data_t* /*user_data*/,
                            rocprofiler_timestamp_t ts)
{
    // Required because of how some compilers handle templates. This may result in an
    // "unused variable" warning.
    (void) ts;

    auto _name = tool_data->callback_tracing_info.at(record.kind, record.operation);

    if constexpr(std::is_same<CategoryT, category::rocm_marker_api>::value)
    {
        if(record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API)
        {
            auto* _data = static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(
                record.payload);

            switch(record.operation)
            {
                case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA:
                {
                    _name      = _data->args.roctxRangePushA.message;
                    auto _hash = tim::add_hash_id(_name);
                    get_marker_pushed_ranges().emplace_back(_hash, ts);
                    break;
                }
                case ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA:
                {
                    _name      = _data->args.roctxRangeStartA.message;
                    auto _hash = tim::add_hash_id(_name);
                    get_marker_started_ranges().emplace_back(_hash, ts);
                    break;
                }
                case ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA:
                {
                    _name = _data->args.roctxMarkA.message;
                    tim::add_hash_id(_name);
                    break;
                }
                default:
                {
                    break;
                }
            }
        }
    }

    if(get_use_timemory())
    {
        component::category_region<category::rocm_marker_api>::start<quirk::timemory>(
            _name);
    }
}

template <typename CategoryT>
void
tool_tracing_callback_stop(
    CategoryT, rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t* user_data, rocprofiler_timestamp_t ts,
    std::optional<std::vector<tim::unwind::processed_entry>>& _bt_data)
{
    auto _name = tool_data->callback_tracing_info.at(record.kind, record.operation);

    uint64_t begin_ts = user_data->value;
    if constexpr(std::is_same<CategoryT, category::rocm_marker_api>::value)
    {
        if(record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API)
        {
            auto* _data = static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(
                record.payload);

            switch(record.operation)
            {
                case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop:
                {
                    ROCPROFSYS_CONDITIONAL_ABORT_F(
                        get_marker_pushed_ranges().empty(),
                        "roctxRangePop does not have corresponding roctxRangePush on "
                        "this thread");

                    auto _hash = get_marker_pushed_ranges().back().first;
                    _name      = tim::get_hash_identifier_fast(_hash);
                    begin_ts   = get_marker_pushed_ranges().back().second;
                    get_marker_pushed_ranges().pop_back();
                    break;
                }
                case ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop:
                {
                    ROCPROFSYS_CONDITIONAL_ABORT_F(
                        get_marker_started_ranges().empty(),
                        "roctxRangeStop does not have corresponding roctxRangeStart on "
                        "this thread");

                    auto _hash = get_marker_started_ranges().back().first;
                    _name      = tim::get_hash_identifier_fast(_hash);
                    begin_ts   = get_marker_started_ranges().back().second;
                    get_marker_started_ranges().pop_back();
                    break;
                }
                case ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA:
                {
                    _name = _data->args.roctxMarkA.message;
                    break;
                }
                case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA:
                case ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA:
                {
                    return;
                }
                default:
                {
                    break;
                }
            }
        }
    }

    if(get_use_timemory())
    {
        component::category_region<category::rocm_marker_api>::stop<quirk::timemory>(
            _name);
    }

    if(get_use_perfetto())
    {
        auto args = callback_arg_array_t{};
        if(config::get_perfetto_annotations())
        {
            rocprofiler_iterate_callback_tracing_kind_operation_args(record, save_args, 2,
                                                                     &args);
        }

        uint64_t _beg_ts   = begin_ts;
        uint64_t _end_ts   = ts;
        auto     stream_id = stream_id_top();

        tracing::push_perfetto_ts(
            CategoryT{}, _name.data(), _beg_ts,
            ::perfetto::Flow::ProcessScoped(record.correlation_id.internal),
            [&](::perfetto::EventContext ctx) {
                if(config::get_perfetto_annotations())
                {
                    tracing::add_perfetto_annotation(ctx, "begin_ns", _beg_ts);
                    tracing::add_perfetto_annotation(ctx, "corr_id",
                                                     record.correlation_id.internal);
                    if(stream_id.handle != 0)
                        tracing::add_perfetto_annotation(ctx, "stream_id",
                                                         stream_id.handle);
                    for(const auto& [key, val] : args)
                        tracing::add_perfetto_annotation(ctx, key, val);

                    if(_bt_data && !_bt_data->empty())
                    {
                        const std::string _unk    = "??";
                        size_t            _bt_cnt = 0;
                        for(const auto& itr : *_bt_data)
                        {
                            auto        _linfo = itr.lineinfo.get();
                            const auto* _func  = (itr.name.empty()) ? &_unk : &itr.name;
                            const auto* _loc =
                                (_linfo && !_linfo.location.empty())
                                    ? &_linfo.location
                                    : ((itr.location.empty()) ? &_unk : &itr.location);
                            auto _line = (_linfo && _linfo.line > 0)
                                             ? join("", _linfo.line)
                                             : ((itr.lineno == 0) ? std::string{ "?" }
                                                                  : join("", itr.lineno));
                            auto _entry =
                                join("", demangle(*_func), " @ ",
                                     join(':', ::basename(_loc->c_str()), _line));
                            if(_bt_cnt < 10)
                            {
                                // Prepend zero for better ordering in UI. Only one zero
                                // is ever necessary since stack depth is limited to 16.
                                tracing::add_perfetto_annotation(
                                    ctx, join("", "frame#0", _bt_cnt++), _entry);
                            }
                            else
                            {
                                tracing::add_perfetto_annotation(
                                    ctx, join("", "frame#", _bt_cnt++), _entry);
                            }
                        }
                    }
                }
            });
        tracing::pop_perfetto_ts(
            CategoryT{}, _name.data(), _end_ts, [&](::perfetto::EventContext ctx) {
                if(config::get_perfetto_annotations())
                    tracing::add_perfetto_annotation(ctx, "end_ns", _end_ts);
            });
    }
}

void
tool_control_callback(rocprofiler_callback_tracing_record_t record,
                      rocprofiler_user_data_t* /*user_data*/, void* /*callback_data*/)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API)
    {
        if(record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause &&
           record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            stop();
        }
        else if(record.operation ==
                    ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume &&
                record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
        {
            start();
        }
    }
}

void
tool_code_object_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t* /*user_data*/, void* /*callback_data*/)
{
    auto ts = rocprofiler_timestamp_t{};
    ROCPROFILER_CALL(rocprofiler_get_timestamp(&ts));

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            if(record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
            {
                auto data_v =
                    *static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(
                        record.payload);
                tool_data->code_object_records.wlock([ts, &record, &data_v](auto& _data) {
                    _data.emplace_back(
                        code_object_callback_record_t{ ts, record, data_v });
                });
            }
            else if(record.operation ==
                    ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
            {
                auto data_v = *static_cast<kernel_symbol_data_t*>(record.payload);
                tool_data->kernel_symbol_records.wlock(
                    [ts, &record, &data_v](auto& _data) {
                        _data.emplace_back(
                            new kernel_symbol_callback_record_t{ ts, record, data_v });
                    });
            }
        }
        return;
    }
}

auto&
get_kernel_dispatch_timestamps()
{
    static auto _v = std::unordered_map<rocprofiler_dispatch_id_t, timing_interval>{};
    return _v;
}

void
tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                      rocprofiler_user_data_t* user_data, void* /*callback_data*/)
{
    auto ts = rocprofiler_timestamp_t{};
    ROCPROFILER_CALL(rocprofiler_get_timestamp(&ts));

    const char* name = nullptr;
    rocprofiler_query_callback_tracing_kind_operation_name(record.kind, record.operation,
                                                           &name, nullptr);

    auto info = std::stringstream{};
    info << std::left << "tid=" << record.thread_id << ", cid=" << std::setw(3)
         << record.correlation_id.internal << ", kind=" << std::setw(2) << record.kind
         << ", operation=" << std::setw(3) << record.operation
         << ", phase=" << record.phase << ", dt_nsec=" << std::setw(8) << ts
         << ", name=" << name;

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        user_data->value = ts;
        switch(record.kind)
        {
            case ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API:
            case ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API:
            case ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API:
            case ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API:
            {
                tool_tracing_callback_start(category::rocm_hsa_api{}, record, user_data,
                                            ts);
                break;
            }
            case ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API:
            case ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API:
            {
                tool_tracing_callback_start(category::rocm_hip_api{}, record, user_data,
                                            ts);
                break;
            }
            case ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API:
            {
                tool_tracing_callback_start(category::rocm_marker_api{}, record,
                                            user_data, ts);
                break;
            }
#if(ROCPROFILER_VERSION >= 600)
            case ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API:
            {
                tool_tracing_callback_start(category::rocm_rocdecode_api{}, record,
                                            user_data, ts);
                break;
            }
#endif
#if(ROCPROFILER_VERSION >= 700)
            case ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API:
            {
                tool_tracing_callback_start(category::rocm_rocjpeg_api{}, record,
                                            user_data, ts);
                break;
            }
#endif
            case ROCPROFILER_CALLBACK_TRACING_RCCL_API:
            {
                tool_tracing_callback_start(category::rocm_rccl_api{}, record, user_data,
                                            ts);
                break;
            }
            case ROCPROFILER_CALLBACK_TRACING_NONE:
            case ROCPROFILER_CALLBACK_TRACING_LAST:
            case ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API:
            case ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API:
            case ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT:
            case ROCPROFILER_CALLBACK_TRACING_SCRATCH_MEMORY:
            case ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH:
            case ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY:
#if(ROCPROFILER_VERSION >= 600)
            case ROCPROFILER_CALLBACK_TRACING_OMPT:
            case ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION:
            case ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION:
#endif
#if(ROCPROFILER_VERSION >= 700)
            case ROCPROFILER_CALLBACK_TRACING_HIP_STREAM:
#endif
            {
                ROCPROFSYS_CI_ABORT(true, "unhandled callback record kind: %i\n",
                                    record.kind);
                break;
            }
        }
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        using backtrace_entry_vec_t = std::vector<tim::unwind::processed_entry>;

        constexpr size_t bt_stack_depth       = 16;
        constexpr size_t bt_ignore_depth      = 3;
        constexpr bool   bt_with_signal_frame = true;

        auto _bt_data = std::optional<backtrace_entry_vec_t>{};

        if(config::get_use_perfetto() && config::get_perfetto_annotations() &&
           tool_data->backtrace_operations.at(record.kind).count(record.operation) > 0)
        {
            auto _backtrace = tim::get_unw_stack<bt_stack_depth, bt_ignore_depth,
                                                 bt_with_signal_frame>();
            _bt_data        = backtrace_entry_vec_t{};
            _bt_data->reserve(_backtrace.size());
            for(auto itr : _backtrace)
            {
                if(itr)
                {
                    if(auto _val = binary::lookup_ipaddr_entry<false>(itr->address());
                       _val)
                    {
                        _bt_data->emplace_back(std::move(*_val));
                    }
                }
            }
        }

        switch(record.kind)
        {
            case ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API:
            case ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API:
            case ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API:
            case ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API:
            {
                tool_tracing_callback_stop(category::rocm_hsa_api{}, record, user_data,
                                           ts, _bt_data);
                break;
            }
            case ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API:
            case ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API:
            {
                tool_tracing_callback_stop(category::rocm_hip_api{}, record, user_data,
                                           ts, _bt_data);
                break;
            }
            case ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API:
            {
                tool_tracing_callback_stop(category::rocm_marker_api{}, record, user_data,
                                           ts, _bt_data);
                break;
            }
#if(ROCPROFILER_VERSION >= 600)
            case ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API:
            {
                tool_tracing_callback_stop(category::rocm_rocdecode_api{}, record,
                                           user_data, ts, _bt_data);
                break;
            }
#endif
#if(ROCPROFILER_VERSION >= 700)
            case ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API:
            {
                tool_tracing_callback_stop(category::rocm_rocjpeg_api{}, record,
                                           user_data, ts, _bt_data);
                break;
            }
#endif
            case ROCPROFILER_CALLBACK_TRACING_RCCL_API:
            {
                tool_tracing_callback_rccl(record, user_data->value, ts);
                tool_tracing_callback_stop(category::rocm_rccl_api{}, record, user_data,
                                           ts, _bt_data);
                break;
            }
            case ROCPROFILER_CALLBACK_TRACING_NONE:
            case ROCPROFILER_CALLBACK_TRACING_LAST:
            case ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API:
            case ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API:
            case ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT:
            case ROCPROFILER_CALLBACK_TRACING_SCRATCH_MEMORY:
            case ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH:
            case ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY:
#if(ROCPROFILER_VERSION >= 600)
            case ROCPROFILER_CALLBACK_TRACING_OMPT:
            case ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION:
            case ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION:
#endif
#if(ROCPROFILER_VERSION >= 700)
            case ROCPROFILER_CALLBACK_TRACING_HIP_STREAM:
#endif
            {
                ROCPROFSYS_CI_ABORT(true, "unhandled callback record kind: %i\n",
                                    record.kind);
                break;
            }
        }
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_NONE)
    {
        if(record.kind == ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH &&
           record.operation == ROCPROFILER_KERNEL_DISPATCH_COMPLETE)
        {
            auto* _data =
                static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(
                    record.payload);

            // save for post-processing
            get_kernel_dispatch_timestamps().emplace(
                _data->dispatch_info.dispatch_id,
                timing_interval{ _data->start_timestamp, _data->end_timestamp });
        }
        else
        {
            ROCPROFSYS_WARNING_F(
                1, "tool_tracing_callback: unhandled PHASE_NONE callback record\n\t%s\n",
                info.str().c_str());
        }
    }
    else
    {
        ROCPROFSYS_CI_ABORT(true, "unhandled callback record phase: %i\n", record.phase);
    }
}

using kernel_dispatch_bundle_t = tim::lightweight_tuple<tim::component::wall_clock>;

void
tool_tracing_buffered(rocprofiler_context_id_t /*context*/,
                      rocprofiler_buffer_id_t /*buffer_id*/,
                      rocprofiler_record_header_t** headers, size_t num_headers,
                      void* /*user_data*/, uint64_t /*drop_count*/)
{
    if(num_headers == 0 || headers == nullptr) return;

    auto _track_desc_stream = [](uint64_t _stream_id) {
        return JOIN("", "HIP Activity Stream ", _stream_id);
    };

    bool _group_by_queue = get_group_by_queue();

    static auto _mtx = std::mutex{};
    auto        _lk  = std::unique_lock<std::mutex>{ _mtx };

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(ROCPROFSYS_LIKELY(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING))
        {
            if(header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
                        header->payload);

                const auto* _kern_sym_data =
                    get_kernel_symbol_info(record->dispatch_info.kernel_id);

                auto        _name     = tim::demangle(_kern_sym_data->kernel_name);
                auto        _corr_id  = record->correlation_id.internal;
                auto        _beg_ns   = record->start_timestamp;
                auto        _end_ns   = record->end_timestamp;
                auto        _agent_id = record->dispatch_info.agent_id;
                auto        _queue_id = record->dispatch_info.queue_id;
                const auto* _agent    = tool_data->get_gpu_tool_agent(_agent_id);

                uint64_t _stream_id = get_stream_id(record).handle;
                ROCPROFSYS_CI_THROW(
                    _stream_id == 0,
                    "Unexpected zero stream_id in kernel dispatch record: %s.",
                    _name.c_str());

                if(get_use_timemory())
                {
                    const auto& _tinfo = thread_info::get(record->thread_id, SystemTID);
                    auto        _tid   = _tinfo->index_data->sequent_value;

                    auto _bundle = kernel_dispatch_bundle_t{ _name };

                    _bundle.push(_tid).start().stop();
                    _bundle.get([_beg_ns, _end_ns](tim::component::wall_clock* _wc) {
                        _wc->set_value(_end_ns - _beg_ns);
                        _wc->set_accum(_end_ns - _beg_ns);
                    });
                    _bundle.pop();
                }

                if(get_use_perfetto())
                {
                    // Lambda to add common perfetto annotations for kernel dispatch
                    auto add_perfetto_annotations = [&](::perfetto::EventContext ctx) {
                        if(config::get_perfetto_annotations())
                        {
                            tracing::add_perfetto_annotation(ctx, "begin_ns", _beg_ns);
                            tracing::add_perfetto_annotation(ctx, "end_ns", _end_ns);
                            tracing::add_perfetto_annotation(ctx, "corr_id", _corr_id);
                            tracing::add_perfetto_annotation(ctx, "stream_id",
                                                             _stream_id);

                            tracing::add_perfetto_annotation(ctx, "queue",
                                                             _queue_id.handle);
                            tracing::add_perfetto_annotation(
                                ctx, "dispatch_id", record->dispatch_info.dispatch_id);
                            tracing::add_perfetto_annotation(
                                ctx, "kernel_id", record->dispatch_info.kernel_id);
                            tracing::add_perfetto_annotation(
                                ctx, "private_segment_size",
                                record->dispatch_info.private_segment_size);
                            tracing::add_perfetto_annotation(
                                ctx, "group_segment_size",
                                record->dispatch_info.group_segment_size);
                            tracing::add_perfetto_annotation(
                                ctx, "workgroup_size",
                                JOIN("", "(",
                                     JOIN(',', record->dispatch_info.workgroup_size.x,
                                          record->dispatch_info.workgroup_size.y,
                                          record->dispatch_info.workgroup_size.z),
                                     ")"));
                            tracing::add_perfetto_annotation(
                                ctx, "grid_size",
                                JOIN("", "(",
                                     JOIN(',', record->dispatch_info.grid_size.x,
                                          record->dispatch_info.grid_size.y,
                                          record->dispatch_info.grid_size.z),
                                     ")"));
                        }
                    };

                    if(_group_by_queue)
                    {
                        auto _track_desc = [](int32_t _device_id_v, int64_t _queue_id_v) {
                            return JOIN("", "GPU Kernel Dispatch [", _device_id_v,
                                        "] Queue ", _queue_id_v);
                        };

                        const auto _track = tracing::get_perfetto_track(
                            category::rocm_kernel_dispatch{}, _track_desc,
                            _agent->device_id, _queue_id.handle);

                        tracing::push_perfetto(category::rocm_kernel_dispatch{},
                                               _name.c_str(), _track, _beg_ns,
                                               ::perfetto::Flow::ProcessScoped(_corr_id),
                                               add_perfetto_annotations);

                        tracing::pop_perfetto(category::rocm_kernel_dispatch{},
                                              _name.c_str(), _track, _end_ns);
                    }
                    else
                    {
                        const auto _track = tracing::get_perfetto_track(
                            category::rocm_hip_stream{}, _track_desc_stream, _stream_id);

                        tracing::push_perfetto(category::rocm_hip_stream{}, _name.c_str(),
                                               _track, _beg_ns,
                                               ::perfetto::Flow::ProcessScoped(_corr_id),
                                               add_perfetto_annotations);

                        tracing::pop_perfetto(category::rocm_hip_stream{}, _name.c_str(),
                                              _track, _end_ns);
                    }
                }
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(
                        header->payload);

                auto        _corr_id      = record->correlation_id.internal;
                auto        _beg_ns       = record->start_timestamp;
                auto        _end_ns       = record->end_timestamp;
                auto        _dst_agent_id = record->dst_agent_id;
                auto        _src_agent_id = record->src_agent_id;
                const auto* _dst_agent    = tool_data->get_agent(_dst_agent_id);
                const auto* _src_agent    = tool_data->get_agent(_src_agent_id);
                auto        _name =
                    tool_data->buffered_tracing_info.at(record->kind, record->operation);

                uint64_t _stream_id = get_stream_id(record).handle;
                ROCPROFSYS_CI_THROW(
                    _stream_id == 0,
                    "Unexpected zero stream_id in memory copy record: %s.", _name.data());

                if(get_use_timemory())
                {
                    const auto& _tinfo = thread_info::get(record->thread_id, SystemTID);
                    auto        _tid   = _tinfo->index_data->sequent_value;

                    auto _bundle = kernel_dispatch_bundle_t{ _name };

                    _bundle.push(_tid).start().stop();
                    _bundle.get([_beg_ns, _end_ns](tim::component::wall_clock* _wc) {
                        _wc->set_value(_end_ns - _beg_ns);
                        _wc->set_accum(_end_ns - _beg_ns);
                    });
                    _bundle.pop();
                }

                if(get_use_perfetto())
                {
                    auto add_perfetto_annotations = [&](::perfetto::EventContext ctx) {
                        if(config::get_perfetto_annotations())
                        {
                            tracing::add_perfetto_annotation(ctx, "begin_ns", _beg_ns);
                            tracing::add_perfetto_annotation(ctx, "end_ns", _end_ns);
                            tracing::add_perfetto_annotation(ctx, "corr_id", _corr_id);
                            tracing::add_perfetto_annotation(ctx, "stream_id",
                                                             _stream_id);
                            tracing::add_perfetto_annotation(ctx, "dst_agent",
                                                             _dst_agent->logical_node_id);
                            tracing::add_perfetto_annotation(ctx, "src_agent",
                                                             _src_agent->logical_node_id);
                        }
                    };

                    if(_group_by_queue)
                    {
                        auto _track_desc = [](int32_t                 _device_id_v,
                                              rocprofiler_thread_id_t _tid) {
                            const auto& _tid_v = thread_info::get(_tid, SystemTID);
                            return JOIN("", "GPU Memory Copy to Agent [", _device_id_v,
                                        "] Thread ", _tid_v->index_data->sequent_value);
                        };

                        const auto _track = tracing::get_perfetto_track(
                            category::rocm_memory_copy{}, _track_desc,
                            _dst_agent->logical_node_id, record->thread_id);

                        tracing::push_perfetto(category::rocm_memory_copy{}, _name.data(),
                                               _track, _beg_ns,
                                               ::perfetto::Flow::ProcessScoped(_corr_id),
                                               add_perfetto_annotations);

                        tracing::pop_perfetto(category::rocm_memory_copy{}, "", _track,
                                              _end_ns);
                    }
                    else
                    {
                        const auto _track = tracing::get_perfetto_track(
                            category::rocm_hip_stream{}, _track_desc_stream, _stream_id);

                        tracing::push_perfetto(category::rocm_hip_stream{}, _name.data(),
                                               _track, _beg_ns,
                                               ::perfetto::Flow::ProcessScoped(_corr_id),
                                               add_perfetto_annotations);

                        tracing::pop_perfetto(category::rocm_hip_stream{}, "", _track,
                                              _end_ns);
                    }
                }
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(
                    header->payload);

                auto _corr_id = record->correlation_id.internal;
                auto _beg_ns  = record->start_timestamp;
                auto _end_ns  = record->end_timestamp;
                auto _name =
                    tool_data->buffered_tracing_info.at(record->kind, record->operation);

                uint64_t _stream_id = get_stream_id(record).handle;

                if(get_use_timemory())
                {
                    const auto& _tinfo = thread_info::get(record->thread_id, SystemTID);
                    auto        _tid   = _tinfo->index_data->sequent_value;

                    auto _bundle = kernel_dispatch_bundle_t{ _name };

                    _bundle.push(_tid).start().stop();
                    _bundle.get([_beg_ns, _end_ns](tim::component::wall_clock* _wc) {
                        _wc->set_value(_end_ns - _beg_ns);
                        _wc->set_accum(_end_ns - _beg_ns);
                    });
                    _bundle.pop();
                }

                if(get_use_perfetto() && _stream_id != 0)
                {
                    auto add_perfetto_annotations = [&](::perfetto::EventContext ctx) {
                        if(config::get_perfetto_annotations())
                        {
                            tracing::add_perfetto_annotation(ctx, "begin_ns", _beg_ns);
                            tracing::add_perfetto_annotation(ctx, "end_ns", _end_ns);
                            tracing::add_perfetto_annotation(ctx, "corr_id", _corr_id);
                            tracing::add_perfetto_annotation(ctx, "stream_id",
                                                             _stream_id);
                        }
                    };

                    if(_group_by_queue)
                    {
                        auto _track_desc = [](rocprofiler_thread_id_t _tid) {
                            const auto& _tid_v = thread_info::get(_tid, SystemTID);
                            return JOIN("", "GPU Memory Copy to Agent Thread ",
                                        _tid_v->index_data->sequent_value);
                        };

                        const auto _track = tracing::get_perfetto_track(
                            category::rocm_hip_api{}, _track_desc, record->thread_id);

                        tracing::push_perfetto(category::rocm_hip_api{}, _name.data(),
                                               _track, _beg_ns,
                                               ::perfetto::Flow::ProcessScoped(_corr_id),
                                               add_perfetto_annotations);

                        tracing::pop_perfetto(category::rocm_hip_api{}, "", _track,
                                              _end_ns);
                    }
                    else
                    {
                        const auto _track = tracing::get_perfetto_track(
                            category::rocm_hip_stream{}, _track_desc_stream, _stream_id);

                        tracing::push_perfetto(category::rocm_hip_stream{}, _name.data(),
                                               _track, _beg_ns,
                                               ::perfetto::Flow::ProcessScoped(_corr_id),
                                               add_perfetto_annotations);

                        tracing::pop_perfetto(category::rocm_hip_stream{}, "", _track,
                                              _end_ns);
                    }
                }
            }
            else
            {
                ROCPROFSYS_THROW(
                    "unexpected rocprofiler_record_header_t buffer tracing category "
                    "kind. category: %i, kind: %i\n",
                    header->category, header->kind);
            }
        }
        else
        {
            ROCPROFSYS_THROW("unexpected rocprofiler_record_header_t tracing category "
                             "kind. category: %i, kind: %i\n",
                             header->category, header->kind);
        }
    }
}

auto&
get_counter_dispatch_data()
{
    static auto _v =
        container::stable_vector<rocprofiler_dispatch_counting_service_data_t>{};
    return _v;
}

auto&
get_counter_dispatch_records()
{
    static auto _v = std::vector<counter_dispatch_record>{};
    return _v;
}

using counter_storage_map_t =
    std::unordered_map<rocprofiler_counter_id_t, counter_storage>;
using agent_counter_storage_map_t =
    std::unordered_map<rocprofiler_agent_id_t, counter_storage_map_t>;

auto*&
get_counter_storage()
{
    static auto* _v = new agent_counter_storage_map_t{};
    return _v;
}

void
counter_record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                        rocprofiler_record_counter_t* record_data, size_t record_count,
                        rocprofiler_user_data_t /*user_data*/,
                        void* /*callback_data_arg*/)
{
    auto* _agent_counter_storage = get_counter_storage();
    if(!_agent_counter_storage) return;

    static auto _mtx = std::mutex{};
    auto        _lk  = std::unique_lock<std::mutex>{ _mtx };

    auto _dispatch_id = dispatch_data.dispatch_info.dispatch_id;
    auto _agent_id    = dispatch_data.dispatch_info.agent_id;
    auto _scope       = scope::get_default();
    auto _interval    = timing_interval{};
    auto _aggregate =
        std::unordered_map<rocprofiler_counter_id_t, rocprofiler_record_counter_t>{};
    for(size_t i = 0; i < record_count; ++i)
    {
        auto _counter_id = rocprofiler_counter_id_t{};
        ROCPROFILER_CALL(
            rocprofiler_query_record_counter_id(record_data[i].id, &_counter_id));

        if(!_aggregate.emplace(_counter_id, record_data[i]).second)
        {
            _aggregate[_counter_id].counter_value += record_data[i].counter_value;
        }
    }

    if(_agent_counter_storage->count(_agent_id) == 0)
        _agent_counter_storage->emplace(_agent_id, counter_storage_map_t{});

    if(get_kernel_dispatch_timestamps().count(_dispatch_id) > 0)
    {
        _interval = get_kernel_dispatch_timestamps().at(_dispatch_id);
        get_kernel_dispatch_timestamps().erase(_dispatch_id);
    }

    for(const auto& itr : _aggregate)
    {
        if(_agent_counter_storage->at(_agent_id).count(itr.first) == 0)
        {
            const auto* _agent = tool_data->get_gpu_tool_agent(_agent_id);
            const auto* _info  = tool_data->get_tool_counter_info(_agent_id, itr.first);

            ROCPROFSYS_CONDITIONAL_ABORT_F(
                !_agent, "unable to find tool agent for agent (id=%zu)\n",
                _agent_id.handle);
            ROCPROFSYS_CONDITIONAL_ABORT_F(
                !_info,
                "unable to find counter info for counter (id=%zu) on agent (id=%zu)\n",
                itr.first.handle, _agent_id.handle);

            auto _dev_id = static_cast<uint32_t>(_agent->device_id);

            _agent_counter_storage->at(_agent_id).emplace(
                itr.first, counter_storage{ tool_data, _dev_id, 0, _info->name });
        }

        auto _event = counter_event{ counter_dispatch_record{
            &dispatch_data, _dispatch_id, itr.first, itr.second } };

        _agent_counter_storage->at(_agent_id).at(itr.first)(_event, _interval, _scope);
    }
}

void
dispatch_counting_service_callback(
    rocprofiler_dispatch_counting_service_data_t dispatch_data,
    rocprofiler_profile_config_id_t* config, rocprofiler_user_data_t* /*user_data*/,
    void*                            callback_data_arg)
{
    auto* _data = as_client_data(callback_data_arg);
    if(!_data || !config) return;

    if(auto itr =
           _data->agent_counter_profiles.find(dispatch_data.dispatch_info.agent_id);
       itr != _data->agent_counter_profiles.end() && itr->second)
    {
        *config = *itr->second;
    }
}

bool
is_initialized(rocprofiler_context_id_t ctx)
{
    return (ctx.handle > 0);
}

bool
is_active(rocprofiler_context_id_t ctx)
{
    int  status = 0;
    auto errc   = rocprofiler_context_is_active(ctx, &status);
    return (errc == ROCPROFILER_STATUS_SUCCESS && status > 0);
}

bool
is_valid(rocprofiler_context_id_t ctx)
{
    int  status = 0;
    auto errc   = rocprofiler_context_is_valid(ctx, &status);
    return (errc == ROCPROFILER_STATUS_SUCCESS && status > 0);
}

void
flush()
{
    if(!tool_data) return;

    for(auto itr : tool_data->get_buffers())
    {
        if(itr.handle > 0)
        {
            auto status = rocprofiler_flush_buffer(itr);
            if(status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
            {
                ROCPROFILER_CALL(status);
            }
        }
    }
}

int
set_kernel_rename_and_stream_correlation_id(
    rocprofiler_thread_id_t /* thr_id */, rocprofiler_context_id_t /* ctx_id */,
    rocprofiler_external_correlation_id_request_kind_t /* kind */,
    rocprofiler_tracing_operation_t /* op */, uint64_t /* internal_corr_id */,
    rocprofiler_user_data_t* external_corr_id, void* /* user_data */)
{
    auto* _info = new kernel_rename_and_stream_data{};

    _info->stream_id = stream_id_top();

    // Set the external correlation id service to point to struct
    external_corr_id->ptr = _info;

    return 0;
}

#if(ROCPROFILER_VERSION >= 700)
void
tool_hip_stream_callback(rocprofiler_callback_tracing_record_t record,
                         rocprofiler_user_data_t* /* user_data */, void* /* data */)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_HIP_STREAM) return;
    // Extract stream ID from record
    auto* stream_handle_data =
        static_cast<rocprofiler_callback_tracing_hip_stream_data_t*>(record.payload);
    auto stream_id = stream_handle_data->stream_id;

    // STREAM_HANDLE_CREATE and DESTROY are no-ops
    if(record.operation == ROCPROFILER_HIP_STREAM_CREATE)
    {
        ROCPROFSYS_VERBOSE_F(
            2, "Entered hip_streams_callback function for ROCPROFILER_HIP_STREAM_CREATE");
    }
    else if(record.operation == ROCPROFILER_HIP_STREAM_DESTROY)
    {
        ROCPROFSYS_VERBOSE_F(
            2,
            "Entered hip_streams_callback function for ROCPROFILER_HIP_STREAM_DESTROY");
    }
    else if(record.operation == ROCPROFILER_HIP_STREAM_SET)
    {
        // Push the stream ID onto the stream stack before underlying HIP function is
        // called
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            ROCPROFSYS_VERBOSE_F(
                2, "Entered hip_streams_callback function for ROCPROFILER_HIP_STREAM_SET "
                   "with ROCPROFILER_CALLBACK_PHASE_ENTER");
            stream_id_push(stream_id);
        }
        // Pop stream ID off of stream stack after underlying HIP function is completed
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
        {
            ROCPROFSYS_VERBOSE_F(
                2, "Entered hip_stream_callback function for ROCPROFILER_HIP_STREAM_SET "
                   "with ROCPROFILER_CALLBACK_PHASE_EXIT");
            stream_id_pop();
        }
    }
    else
    {
        ROCPROFSYS_FAIL_F("Unknown operation for hip_stream_callback!");
    }
}
#endif

int
tool_init(rocprofiler_client_finalize_t fini_func, void* user_data)
{
    auto domains = settings::instance()->at("ROCPROFSYS_ROCM_DOMAINS");

    ROCPROFSYS_VERBOSE_F(1, "Available ROCm Domains:\n");
    for(const auto& itr : domains->get_choices())
        ROCPROFSYS_VERBOSE_F(1, "- %s\n", itr.c_str());

    auto _callback_domains = rocprofiler_sdk::get_callback_domains();
    auto _buffered_domain  = rocprofiler_sdk::get_buffered_domains();
    auto _counter_events   = rocprofiler_sdk::get_rocm_events();
    auto _version          = rocprofiler_sdk::get_version();
    ROCPROFSYS_WARNING_IF(_version.formatted == 0,
                          "Warning! rocprofiler-sdk version not initialized\n");

    auto* _data        = as_client_data(user_data);
    _data->client_fini = fini_func;

    _data->initialize();
    if(!_counter_events.empty()) _data->initialize_event_info();

    ROCPROFILER_CALL(rocprofiler_create_context(&_data->primary_ctx));

    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
        _data->primary_ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
        tool_code_object_callback, _data));

    for(auto itr : {
            ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,
            ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
            ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API,
            ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
            ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
            ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API,
            ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
            ROCPROFILER_CALLBACK_TRACING_RCCL_API,
#if(ROCPROFILER_VERSION >= 600)
            ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API,
#endif
#if(ROCPROFILER_VERSION >= 700)
            ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API,
#endif
        })
    {
        if(_callback_domains.count(itr) > 0)
        {
            auto _ops = rocprofiler_sdk::get_operations(itr);
            _data->backtrace_operations.emplace(
                itr, rocprofiler_sdk::get_backtrace_operations(itr));
            ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                _data->primary_ctx, itr, _ops.data(), _ops.size(), tool_tracing_callback,
                _data));
        }
    }

    constexpr auto buffer_size = 8192;
    constexpr auto watermark   = 7936;

    // Configure external correlation id request service for kernel dispatch
    // and memory copy.

    auto external_corr_id_request_kinds =
        std::array<rocprofiler_external_correlation_id_request_kind_t, 3>{
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_RUNTIME_API
        };

    ROCPROFILER_CALL(rocprofiler_configure_external_correlation_id_request_service(
        _data->primary_ctx, external_corr_id_request_kinds.data(),
        external_corr_id_request_kinds.size(),
        set_kernel_rename_and_stream_correlation_id, _data));

#if(ROCPROFILER_VERSION >= 700)
    if((_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) > 0) ||
       (_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) > 0) ||
       (_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API) > 0))
    {
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
            _data->primary_ctx, ROCPROFILER_CALLBACK_TRACING_HIP_STREAM, nullptr, 0,
            tool_hip_stream_callback, nullptr));
    }
#endif

    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) > 0)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(
            _data->primary_ctx, buffer_size, watermark,
            ROCPROFILER_BUFFER_POLICY_LOSSLESS, tool_tracing_buffered, tool_data,
            &_data->kernel_dispatch_buffer));

        ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0,
            _data->kernel_dispatch_buffer));
    }

    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) > 0)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(
            _data->primary_ctx, buffer_size, watermark,
            ROCPROFILER_BUFFER_POLICY_LOSSLESS, tool_tracing_buffered, tool_data,
            &_data->memory_copy_buffer));

        ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0,
            _data->memory_copy_buffer));
    }

    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API) > 0)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(
            _data->primary_ctx, buffer_size, watermark,
            ROCPROFILER_BUFFER_POLICY_LOSSLESS, tool_tracing_buffered, tool_data,
            &_data->hip_runtime_api_buffer));

        ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, nullptr, 0,
            _data->hip_runtime_api_buffer));
    }

    if(!_counter_events.empty())
    {
        for(const auto& itr : _data->gpu_agents)
        {
            _data->agent_events.emplace(
                itr.agent->id,
                create_agent_profile(itr.agent->id, _counter_events, _data));
        }

        ROCPROFILER_CALL(rocprofiler_create_context(&_data->counter_ctx));

        auto _operations = std::array<rocprofiler_tracing_operation_t, 1>{
            ROCPROFILER_KERNEL_DISPATCH_COMPLETE
        };

        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
            _data->counter_ctx, ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
            _operations.data(), _operations.size(), tool_tracing_callback, _data));

        ROCPROFILER_CALL(rocprofiler_configure_callback_dispatch_counting_service(
            _data->counter_ctx, dispatch_counting_service_callback, _data,
            counter_record_callback, _data));

        // ROCPROFILER_CALL(rocprofiler_create_buffer(
        //     counter_ctx, buffer_size, watermark,
        //     ROCPROFILER_BUFFER_POLICY_LOSSLESS, tool_tracing_buffered, tool_data,
        //     &counter_collection_buffer));

        // for(const auto& itr : *agent_counter_profiles)
        // {
        //     ROCPROFILER_CALL(rocprofiler_configure_agent_profile_counting_service(
        //         counter_ctx, counter_collection_buffer, itr.first,
        //         agent_counter_profile_callback, nullptr));
        // }
    }

    for(const auto& itr : _data->get_buffers())
    {
        if(itr.handle > 0)
        {
            auto client_thread = rocprofiler_callback_thread_t{};
            ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread));
            ROCPROFILER_CALL(rocprofiler_assign_callback_thread(itr, client_thread));
        }
    }

    // throwaway context for handling the profiler control API. If primary_ctx were used,
    // we would get profiler pause callback but never get profiler resume callback
    {
        auto _local_ctx = rocprofiler_context_id_t{ 0 };
        ROCPROFILER_CALL(rocprofiler_create_context(&_local_ctx));
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
            _local_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API, nullptr, 0,
            tool_control_callback, _data));
    }

    if(!is_valid(_data->primary_ctx))
    {
        // notify rocprofiler that initialization failed and all the contexts, buffers,
        // etc. created should be ignored
        return -1;
    }

    gpu::add_device_metadata();

    if(config::get_use_process_sampling() && config::get_use_amd_smi())
    {
        ROCPROFSYS_VERBOSE_F(1, "Setting amd_smi state to active...\n");
        amd_smi::set_state(State::Active);
    }

    start();

    // no errors
    return 0;
}

void
tool_fini(void* callback_data)
{
    static std::atomic_flag _once = ATOMIC_FLAG_INIT;
    if(_once.test_and_set()) return;

    flush();
    stop();

    if(config::get_use_process_sampling() && config::get_use_amd_smi())
        amd_smi::shutdown();

    if(get_counter_storage())
    {
        get_counter_storage()->clear();
        delete get_counter_storage();
        get_counter_storage() = nullptr;
    }

    auto* _data        = as_client_data(callback_data);
    _data->client_id   = nullptr;
    _data->client_fini = nullptr;

    delete tool_data;
    tool_data = nullptr;
}
}  // namespace

void
setup()
{
    if(int status = 0;
       rocprofiler_is_initialized(&status) == ROCPROFILER_STATUS_SUCCESS && status == 0)
    {
        ROCPROFILER_CALL(rocprofiler_force_configure(&rocprofiler_configure));
    }
}

void
shutdown()
{
    // shutdown
    if(tool_data && tool_data->client_id && tool_data->client_fini)
        tool_data->client_fini(*tool_data->client_id);
}

void
config()
{}

void
post_process()
{}

void
sample()
{}

void
start()
{
    if(!tool_data) return;

    for(auto itr : tool_data->get_contexts())
    {
        if(is_initialized(itr) && !is_active(itr))
        {
            ROCPROFILER_CALL(rocprofiler_start_context(itr));
        }
    }
}

void
stop()
{
    if(!tool_data) return;

    for(auto itr : tool_data->get_contexts())
    {
        if(is_initialized(itr) && is_active(itr))
        {
            ROCPROFILER_CALL(rocprofiler_stop_context(itr));
        }
    }
}

std::vector<hardware_counter_info>
get_rocm_events_info()
{
    if(!tool_data)
    {
        auto _tool_data_v = client_data{};
        _tool_data_v.initialize_event_info();
        return _tool_data_v.events_info;
    }

    if(tool_data->events_info.empty()) tool_data->initialize_event_info();

    return tool_data->events_info;
}
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t version, const char* runtime_version, uint32_t priority,
                      rocprofiler_client_id_t* id)
{
    // only activate once
    {
        static bool _first = true;
        if(!_first) return nullptr;
        _first = false;
    }

    if(!tim::get_env("ROCPROFSYS_INIT_TOOLING", true)) return nullptr;
    if(!tim::settings::enabled()) return nullptr;

    if(!rocprofsys::config::settings_are_configured() &&
       rocprofsys::get_state() < rocprofsys::State::Active)
        rocprofsys_init_tooling_hidden();

    if(!rocprofsys::config::get_use_rocm())
    {
        return nullptr;
    }

    // set the client name
    id->name = "rocprofsys";

    // ensure tool data exists
    if(!rocprofsys::rocprofiler_sdk::tool_data)
        rocprofsys::rocprofiler_sdk::tool_data =
            new rocprofsys::rocprofiler_sdk::client_data{};

    // store client info
    rocprofsys::rocprofiler_sdk::tool_data->client_id = id;

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " is using rocprofiler-sdk v" << major << "." << minor << "."
         << patch << " (" << runtime_version << ")";

    ROCPROFSYS_VERBOSE_F(0, "%s\n", info.str().c_str());
    ROCPROFSYS_VERBOSE_F(2, "client_id=%u, priority=%u\n", id->handle, priority);

    ROCPROFILER_CALL(rocprofiler_at_internal_thread_create(
        rocprofsys::rocprofiler_sdk::thread_precreate,
        rocprofsys::rocprofiler_sdk::thread_postcreate,
        ROCPROFILER_LIBRARY | ROCPROFILER_HSA_LIBRARY | ROCPROFILER_HIP_LIBRARY |
            ROCPROFILER_MARKER_LIBRARY,
        nullptr));

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{ sizeof(rocprofiler_tool_configure_result_t),
                                             &::rocprofsys::rocprofiler_sdk::tool_init,
                                             &::rocprofsys::rocprofiler_sdk::tool_fini,
                                             rocprofsys::rocprofiler_sdk::tool_data };

    // return pointer to configure data
    return &cfg;
}
