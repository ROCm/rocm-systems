// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "core/rocprofiler-sdk.hpp"
#include "api.hpp"
#include "common/synchronized.hpp"
#include "core/common.hpp"
#include "core/common_types.hpp"
#include "core/config.hpp"
#include "core/containers/stable_vector.hpp"
#include "core/demangler.hpp"
#include "core/gpu.hpp"
#include "core/perfetto.hpp"
#include "core/state.hpp"
#include "core/trace_cache/buffer_storage.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/amd_smi.hpp"
#include "library/components/category_region.hpp"
#include "library/rocprofiler-sdk.hpp"
#include "library/rocprofiler-sdk/counters.hpp"
#include "library/rocprofiler-sdk/fwd.hpp"
#include "library/rocprofiler-sdk/rccl.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"

#include <algorithm>
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

#include <nlohmann/json.hpp>
#include <spdlog/fmt/ranges.h>

#include "logger/debug.hpp"

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

#if(ROCPROFILER_VERSION >= 600)

struct rocprofsys_ompt_data_storage_t
{
    rocprofiler_callback_tracing_record_t record;
    rocprofiler_timestamp_t               _beg_ts;
    function_args_t                       args;  // Required for orphan ENTER events
};

auto
ompt_get_unified_name(const rocprofiler_callback_tracing_record_t& record)
{
    std::string_view _name =
        tool_data->callback_tracing_info.at(record.kind, record.operation);

    // Forces omp_parallel begin and end to have same name, allowing track to connect
    if(record.operation == ROCPROFILER_OMPT_ID_parallel_begin ||
       record.operation == ROCPROFILER_OMPT_ID_parallel_end)
        _name = "omp_parallel";

    return _name;
}

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
        _record->correlation_id.external.ptr = nullptr;
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

    // Check if agent info is available (may not be for unsupported architectures)
    auto agent_info_it = data->agent_counter_info.find(agent_id);
    if(agent_info_it == data->agent_counter_info.end())
    {
        LOG_WARNING("Skipping GPU agent {} (device {}) due to unsupported "
                    "architecture or missing counter info",
                    agent_id.handle, tool_agent_v->device_id);

        data->agent_counter_profiles.emplace(agent_id, profile);
        return counter_vec_t{};
    }

    constexpr auto device_qualifier = std::string_view{ ":device=" };
    for(const auto& itr : counters)
    {
        auto name_v = itr;
        if(auto pos = std::string::npos;
           (pos = itr.find(device_qualifier)) != std::string::npos)
        {
            name_v        = itr.substr(0, pos);
            auto dev_id_s = itr.substr(pos + device_qualifier.length());

            if(dev_id_s.empty() ||
               dev_id_s.find_first_not_of("0123456789") != std::string::npos)
            {
                LOG_CRITICAL("invalid device qualifier format (':device=N) "
                             "where N is the GPU id: {}",
                             itr);
                ::rocprofsys::set_state(::rocprofsys ::State ::Finalized);
                std::abort();
            }

            auto dev_id_v = std::stoul(dev_id_s);

            LOG_DEBUG("tool agent device id={}, name={}, device_id={}",
                      tool_agent_v->device_id, name_v, dev_id_v);

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
            LOG_DEBUG("tool agent device id={}, old_name={}, name={}",
                      tool_agent_v->device_id, _old_name_v, name_v);
        }
        else if(name_v == itr)
        {
            LOG_DEBUG("tool agent device id={}, name={}", tool_agent_v->device_id,
                      name_v);
        }

        // search the gpu agent counter info for a counter with a matching name
        for(const auto& citr : agent_info_it->second)
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
        auto requested_counters = fmt::format("{}", fmt::join(counters, ", "));
        auto found_counters     = fmt::format("{}", fmt::join(found_v, ", "));

        // Determine which counters were not found
        auto missing_counters = std::vector<std::string>{};
        for(const auto& counter : counters)
        {
            if(std::find(found_v.begin(), found_v.end(), counter) == found_v.end())
                missing_counters.emplace_back(counter);
        }
        auto missing_counters_str = fmt::format("{}", fmt::join(missing_counters, ", "));

        // In production, warn and continue with available counters
        LOG_WARNING("Unable to find all counters for agent {} (gpu-{}, {}). "
                    "Requested: {}. Found: {}. Missing: {}. Continuing with "
                    "available counters.",
                    tool_agent_v->agent->node_id, tool_agent_v->device_id,
                    tool_agent_v->agent->name, requested_counters, found_counters,
                    missing_counters_str);

        if(get_is_continuous_integration())
        {
            LOG_CRITICAL(
                "Unable to find all counters for agent {} (gpu-{}, {}) in {}. Found: {}",
                tool_agent_v->agent->node_id, tool_agent_v->device_id,
                tool_agent_v->agent->name, requested_counters, found_counters);

            ::rocprofsys::set_state(::rocprofsys ::State ::Finalized);
            ::std ::abort();
        }
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

const rocprofiler_callback_tracing_code_object_load_data_t*
get_code_object_info(uint64_t _code_object_id)
{
    return tool_data->get_code_object_info(_code_object_id);
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

// Additional implementation of rocprofiler_callback_tracing_operation_args_cb_t
// for iterating through arguments in a callback for rocpd_arg table in database
int
iterate_args_callback(rocprofiler_callback_tracing_kind_t /*kind*/, int32_t /*operation*/,
                      uint32_t arg_number, const void* const /*arg_value_addr*/,
                      int32_t /*arg_indirection_count*/, const char* arg_type,
                      const char* arg_name, const char*        arg_value_str,
                      int32_t /*arg_dereference_count*/, void* data)
{
    auto* _data = static_cast<function_args_t*>(data);
    if(arg_type && arg_name && arg_value_str)
        _data->emplace_back(argument_info{ arg_number,
                                           rocprofsys::utility::demangle(arg_type),
                                           arg_name, arg_value_str });
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

template <typename Tp, typename... Args>
Tp*
as_pointer(Args&&... _args)
{
    return new Tp{ std::forward<Args>(_args)... };
}

template <typename... Tp>
void
consume_args(Tp&&...)
{}

auto
get_backtrace(std::optional<std::vector<tim::unwind::processed_entry>>& _bt_data)
{
    auto backtrace = nlohmann::json();

    if(_bt_data && !_bt_data->empty())
    {
        const std::string _unk    = "??";
        size_t            _bt_cnt = 0;
        for(const auto& itr : *_bt_data)
        {
            auto        _linfo = itr.lineinfo.get();
            const auto* _func  = (itr.name.empty()) ? &_unk : &itr.name;
            const auto* _loc   = (_linfo && !_linfo.location.empty())
                                     ? &_linfo.location
                                     : ((itr.location.empty()) ? &_unk : &itr.location);
            auto        _line  = (_linfo && _linfo.line > 0)
                                     ? fmt::format("{}", _linfo.line)
                                     : ((itr.lineno == 0) ? std::string{ "?" }
                                                          : fmt::format("{}", itr.lineno));
            auto _entry = fmt::format("{} @ {}:{}", rocprofsys::utility::demangle(*_func),
                                      ::basename(_loc->c_str()), _line);
            backtrace[fmt::format("frame#{}", _bt_cnt++)] = _entry;
        }
    }
    return backtrace;
}

template <typename CorrelationIdType>
uint64_t
get_parent_stack_id([[maybe_unused]] const CorrelationIdType& correlation_id)
{
#if(ROCPROFILER_VERSION >= 700)
    if constexpr(std::is_same_v<rocprofiler_correlation_id_t, CorrelationIdType>)
    {
        return correlation_id.ancestor;
    }
    else
    {
        return 0;
    }
#else
    return 0;
#endif
}

struct scope_destructor
{
    /// \fn scope_destructor(FuncT&& _fini, InitT&& _init)
    /// \tparam FuncT "std::function<void()> or void (*)()"
    /// \tparam InitT "std::function<void()> or void (*)()"
    /// \param _fini Function to execute when object is destroyed
    /// \param _init Function to execute when object is created (optional)
    ///
    /// \brief Provides a utility to perform an operation when exiting a scope.
    template <typename FuncT, typename InitT = void (*)()>
    scope_destructor(FuncT&& _fini, InitT&& _init = []() {});

    ~scope_destructor() { m_functor(); }

    // delete copy operations
    scope_destructor(const scope_destructor&)            = delete;
    scope_destructor& operator=(const scope_destructor&) = delete;

    // allow move operations
    scope_destructor(scope_destructor&& rhs) noexcept;
    scope_destructor& operator=(scope_destructor&& rhs) noexcept;

private:
    std::function<void()> m_functor = []() {};
};

template <typename FuncT, typename InitT>
scope_destructor::scope_destructor(FuncT&& _fini, InitT&& _init)
: m_functor{ std::forward<FuncT>(_fini) }
{
    _init();
}

inline scope_destructor::scope_destructor(scope_destructor&& rhs) noexcept
: m_functor{ std::move(rhs.m_functor) }
{
    rhs.m_functor = []() {};
}

inline scope_destructor&
scope_destructor::operator=(scope_destructor&& rhs) noexcept
{
    if(this != &rhs)
    {
        m_functor     = std::move(rhs.m_functor);
        rhs.m_functor = []() {};
    }
    return *this;
}

using kernel_rename_stack_t = std::stack<uint64_t>;

thread_local auto thread_dispatch_rename      = as_pointer<kernel_rename_stack_t>();
thread_local auto thread_dispatch_rename_dtor = scope_destructor{ []() {
    delete thread_dispatch_rename;
    thread_dispatch_rename                    = nullptr;
} };

template <typename Category>
void
cache_category()
{
    trace_cache::get_metadata_registry().add_string(trait::name<Category>::value);
}

void
cache_add_thread_info(uint64_t tid)
{
    trace_cache::get_metadata_registry().add_thread_info(
        { getppid(), getpid(), tid, 0, 0, "{}" });
}

void
cache_add_track(const char* track_name, uint64_t tid)
{
    trace_cache::get_metadata_registry().add_track({ track_name, tid, "{}" });
}

size_t
get_mem_copy_dst_address(
    [[maybe_unused]] const rocprofiler_buffer_tracing_memory_copy_record_t& record)
{
#if(ROCPROFILER_VERSION >= 700)
    return record.dst_address.value;
#else
    return 0;
#endif
}

size_t
get_mem_copy_src_address(
    [[maybe_unused]] const rocprofiler_buffer_tracing_memory_copy_record_t& record)
{
#if(ROCPROFILER_VERSION >= 700)
    return record.src_address.value;
#else
    return 0;
#endif
}

#if(ROCPROFILER_VERSION >= 600)
size_t
get_mem_alloc_address(
    [[maybe_unused]] const rocprofiler_buffer_tracing_memory_allocation_record_t& record)
{
#    if(ROCPROFILER_VERSION >= 700)
    return record.address.value;
#    else
    return static_cast<size_t>(record.address.handle);
#    endif
}
#endif

uint64_t
get_scratch_mem_alloc_size(
    [[maybe_unused]] const rocprofiler_buffer_tracing_scratch_memory_record_t& record)
{
// The version of rocprofiler_buffer_tracing_scratch_memory_record_t from ROCm < 7.1 does
// not have the allocation_size field. ROCPROFILER_VERSION for both ROCm 7.0 and 7.1
// is 1.0.0, so we need to check the ROCm version.
#if(ROCPROFSYS_USE_ROCM > 0 && ROCPROFSYS_ROCM_VERSION >= 70100)
    return record.allocation_size;
#else
    return 0;
#endif
}

void
cache_region(const rocprofiler_callback_tracing_record_t* record,
             const rocprofiler_timestamp_t                start_timestamp,
             const rocprofiler_timestamp_t end_timestamp, const std::string& call_stack,
             const std::string& args_str, const std::string& category,
             std::string_view name = {})

{
    // Use provided name if available, otherwise fall back to API operation name
    std::string _name;
    if(name.empty())
    {
        auto callback_tracing_info =
            trace_cache::get_metadata_registry().get_callback_tracing_info();
        _name = std::string{ callback_tracing_info.at(record->kind, record->operation) };
    }
    else
    {
        _name = std::string{ name };
    }

    trace_cache::get_buffer_storage().store(trace_cache::region_sample{
        record->thread_id, _name.c_str(), record->correlation_id.internal,
        get_parent_stack_id(record->correlation_id), start_timestamp, end_timestamp,
        call_stack.c_str(), args_str.c_str(), category.c_str() });
}

void
cache_kernel_dispatch(rocprofiler_buffer_tracing_kernel_dispatch_record_t* record,
                      uint64_t                                             stream_handle)
{
    auto queue_handle = record->dispatch_info.queue_id.handle;

    trace_cache::get_metadata_registry().add_queue(queue_handle);
    trace_cache::get_metadata_registry().add_stream(stream_handle);

    trace_cache::get_buffer_storage().store(trace_cache::kernel_dispatch_sample{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->dispatch_info.agent_id.handle, record->dispatch_info.kernel_id,
        record->dispatch_info.dispatch_id, record->dispatch_info.queue_id.handle,
        record->correlation_id.internal, get_parent_stack_id(record->correlation_id),
        record->dispatch_info.private_segment_size,
        record->dispatch_info.group_segment_size, record->dispatch_info.workgroup_size.x,
        record->dispatch_info.workgroup_size.y, record->dispatch_info.workgroup_size.z,
        record->dispatch_info.grid_size.x, record->dispatch_info.grid_size.y,
        record->dispatch_info.grid_size.z, stream_handle });
}

void
cache_scratch_memory(rocprofiler_buffer_tracing_scratch_memory_record_t* record,
                     uint64_t                                            stream_handle)
{
    trace_cache::get_metadata_registry().add_stream(stream_handle);
    trace_cache::get_buffer_storage().store(trace_cache::scratch_memory_sample{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->agent_id.handle, record->queue_id.handle,
        static_cast<int32_t>(record->kind), static_cast<int32_t>(record->operation),
        static_cast<int32_t>(record->flags), get_scratch_mem_alloc_size(*record),
        record->correlation_id.internal, get_parent_stack_id(record->correlation_id),
        stream_handle });
}

void
cache_memory_copy(rocprofiler_buffer_tracing_memory_copy_record_t* record,
                  uint64_t                                         stream_handle)
{
    trace_cache::get_metadata_registry().add_stream(stream_handle);
    trace_cache::get_buffer_storage().store(trace_cache::memory_copy_sample{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->dst_agent_id.handle, record->src_agent_id.handle,
        static_cast<int32_t>(record->kind), static_cast<int32_t>(record->operation),
        record->bytes, record->correlation_id.internal,
        get_parent_stack_id(record->correlation_id), get_mem_copy_dst_address(*record),
        get_mem_copy_src_address(*record), stream_handle });
}

#if(ROCPROFILER_VERSION >= 600)
void
cache_memory_allocation(rocprofiler_buffer_tracing_memory_allocation_record_t* record,
                        uint64_t stream_handle)
{
    trace_cache::get_metadata_registry().add_stream(stream_handle);
    trace_cache::get_buffer_storage().store(trace_cache::memory_allocate_sample{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->agent_id.handle, static_cast<int32_t>(record->kind),
        static_cast<int32_t>(record->operation), record->allocation_size,
        record->correlation_id.internal, get_parent_stack_id(record->correlation_id),
        get_mem_alloc_address(*record), stream_handle });
}
#endif

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
                    // A basic roctx marker region starts with roctxRangePushA ENTER and
                    // ends with roctxRangePop EXIT.
                    // Breaking instead of returning allows the roctxRangePop ENTER to be
                    // processed, which timemory will link to the roctxRangePop EXIT. As
                    // we do not push roctxRangePushA EXIT into timemory, it will think
                    // that the roctxRangePushA ENTER is still active when it is in fact
                    // not. This will cause the wall clock tree to be incorrect.
                    return;
                }
            }
        }
    }

    if(get_use_timemory())
    {
        tracing::push_timemory(CategoryT{}, _name);
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
                    if(get_marker_pushed_ranges().empty())
                    {
                        LOG_CRITICAL("roctxRangePop does not have corresponding "
                                     "roctxRangePush on this thread");
                        ::rocprofsys::set_state(::rocprofsys ::State ::Finalized);
                        ::std ::abort();
                    }

                    auto _hash = get_marker_pushed_ranges().back().first;
                    _name      = tim::get_hash_identifier_fast(_hash);
                    begin_ts   = get_marker_pushed_ranges().back().second;
                    get_marker_pushed_ranges().pop_back();
                    break;
                }
                case ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop:
                {
                    if(get_marker_started_ranges().empty())
                    {
                        LOG_CRITICAL("roctxRangeStop does not have corresponding "
                                     "roctxRangeStart on this thread");
                        ::rocprofsys::set_state(::rocprofsys ::State ::Finalized);
                        ::std ::abort();
                    }

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
        tracing::pop_timemory(CategoryT{}, _name);
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
                    tracing::add_perfetto_annotation(ctx, "stack_id",
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
                            auto _line =
                                (_linfo && _linfo.line > 0)
                                    ? fmt::format("{}", _linfo.line)
                                    : ((itr.lineno == 0) ? std::string{ "?" }
                                                         : fmt::format("{}", itr.lineno));
                            auto _entry = fmt::format(
                                "{} @ {}:{}", rocprofsys::utility::demangle(*_func),
                                ::basename(_loc->c_str()), _line);
                            if(_bt_cnt < 10)
                            {
                                // Prepend zero for better ordering in UI. Only one
                                // zero is ever necessary since stack depth is limited
                                // to 16.
                                tracing::add_perfetto_annotation(
                                    ctx, fmt::format("frame#0{}", _bt_cnt++), _entry);
                            }
                            else
                            {
                                tracing::add_perfetto_annotation(
                                    ctx, fmt::format("frame#{}", _bt_cnt++), _entry);
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

    // Insert callback trace into database
    auto args = function_args_t{};

    rocprofiler_iterate_callback_tracing_kind_operation_args(
        record, iterate_args_callback, 2, &args);

    auto     call_stack = get_backtrace(_bt_data);
    uint64_t _beg_ts    = begin_ts;
    uint64_t _end_ts    = ts;

    {
        cache_category<CategoryT>();
        cache_add_thread_info(record.thread_id);
        std::string args_str = get_args_string(args);
        cache_region(&record, _beg_ts, _end_ts, call_stack.dump(), args_str,
                     trait::name<CategoryT>::value, _name);
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
                trace_cache::get_metadata_registry().add_code_object(data_v);
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
                trace_cache::get_metadata_registry().add_kernel_symbol(data_v);
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

#if(ROCPROFILER_VERSION >= 600)

// An instant event is one that has its beg_ts = end_ts
void
ompt_cache_instant_event(
    rocprofiler_callback_tracing_record_t record, rocprofiler_timestamp_t _instant_ts,
    std::optional<std::vector<tim::unwind::processed_entry>>& _bt_data)
{
    auto args = function_args_t{};
    rocprofiler_iterate_callback_tracing_kind_operation_args(
        record, iterate_args_callback, 2, &args);
    auto call_stack = get_backtrace(_bt_data);

    cache_category<category::rocm_ompt_api>();
    cache_add_thread_info(record.thread_id);
    cache_region(&record, _instant_ts, _instant_ts, call_stack.dump(),
                 get_args_string(args), trait::name<category::rocm_ompt_api>::value);
}

// OMPT callbacks with no corresponding begin/end are treated as "instant"
void
ompt_cache_orphan_event(
    const rocprofsys_ompt_data_storage_t&                     stored_data,
    std::optional<std::vector<tim::unwind::processed_entry>>& _bt_data)
{
    auto call_stack = get_backtrace(_bt_data);
    cache_category<category::rocm_ompt_api>();
    cache_add_thread_info(stored_data.record.thread_id);
    cache_region(&stored_data.record, stored_data._beg_ts, stored_data._beg_ts,
                 call_stack.dump(), get_args_string(stored_data.args),
                 trait::name<category::rocm_ompt_api>::value);
}

// Any OMPT callback that can be of phase ENTER or EXIT is a standard callback.
//  I.e. it has an ompt_scope_endpoint_t in its definition (excluding
//  ROCPROFILER_OMPT_ID_nest_lock as it is a mutex)
auto&
get_ompt_standard_cb_storage()
{
    // uint64_t -> internal id from rocprofiler_correlation_id_t
    static thread_local auto _v =
        std::unordered_map<uint64_t, rocprofsys_ompt_data_storage_t>{};
    return _v;
}

// An OMPT parallel callback consists of ROCPROFILER_OMPT_ID_parallel_begin and
// ROCPROFILER_OMPT_ID_parallel_end
//  As the beginning and end can only occur on the same thread, they are connected into a
//  single track called "omp_parallel" for clarity. In this track, the information
//  contained within parallel_begin should be displayed as it contains all the information
//  that parallel_end has as well as the flags and number of threads/teams that were
//  requested.
auto&
get_ompt_parallel_cb_storage()
{
    // uintptr_t -> parallel_data (see callback definition)
    static thread_local auto _v =
        std::unordered_map<uintptr_t, rocprofsys_ompt_data_storage_t>{};
    return _v;
}

void
ompt_push_standard_callback(const rocprofiler_callback_tracing_record_t& record,
                            const rocprofiler_timestamp_t&               _beg_ts)
{
    auto args = function_args_t{};
    rocprofiler_iterate_callback_tracing_kind_operation_args(
        record, iterate_args_callback, 1, &args);
    get_ompt_standard_cb_storage().emplace(
        record.correlation_id.internal,
        rocprofsys_ompt_data_storage_t{ record, _beg_ts, args });
}

void
ompt_pop_standard_callback(
    const rocprofiler_callback_tracing_record_t&              record,
    const rocprofiler_timestamp_t&                            _end_ts,
    std::optional<std::vector<tim::unwind::processed_entry>>& _bt_data)
{
    auto it = get_ompt_standard_cb_storage().find(record.correlation_id.internal);

    if(it == get_ompt_standard_cb_storage().end())
    {
        auto args = function_args_t{};
        rocprofiler_iterate_callback_tracing_kind_operation_args(
            record, iterate_args_callback, 2, &args);
        ompt_cache_orphan_event(rocprofsys_ompt_data_storage_t{ record, _end_ts, args },
                                _bt_data);
        return;
    }

    auto stored_data = it->second;
    get_ompt_standard_cb_storage().erase(it);

    auto call_stack = get_backtrace(_bt_data);
    cache_category<category::rocm_ompt_api>();
    cache_add_thread_info(record.thread_id);
    cache_region(&record, stored_data._beg_ts, _end_ts, call_stack.dump(),
                 get_args_string(stored_data.args),
                 trait::name<category::rocm_ompt_api>::value);
}

void
ompt_push_parallel_callback(const rocprofiler_callback_tracing_record_t& record,
                            const rocprofiler_timestamp_t&               _beg_ts)
{
    auto* payload_data =
        static_cast<rocprofiler_callback_tracing_ompt_data_t*>(record.payload);
    const void* parallel_data_address = payload_data->args.parallel_begin.parallel_data;

    auto args = function_args_t{};
    rocprofiler_iterate_callback_tracing_kind_operation_args(
        record, iterate_args_callback, 1, &args);
    get_ompt_parallel_cb_storage().emplace(
        reinterpret_cast<uintptr_t>(parallel_data_address),
        rocprofsys_ompt_data_storage_t{ record, _beg_ts, args });
}

void
ompt_pop_parallel_callback(
    const rocprofiler_callback_tracing_record_t&              record,
    const rocprofiler_timestamp_t&                            _end_ts,
    std::optional<std::vector<tim::unwind::processed_entry>>& _bt_data)
{
    auto* payload_data =
        static_cast<rocprofiler_callback_tracing_ompt_data_t*>(record.payload);
    const void* parallel_data_address = payload_data->args.parallel_end.parallel_data;

    auto it = get_ompt_parallel_cb_storage().find(
        reinterpret_cast<uintptr_t>(parallel_data_address));

    if(it == get_ompt_parallel_cb_storage().end())
    {
        auto args = function_args_t{};
        rocprofiler_iterate_callback_tracing_kind_operation_args(
            record, iterate_args_callback, 2, &args);
        ompt_cache_orphan_event(rocprofsys_ompt_data_storage_t{ record, _end_ts, args },
                                _bt_data);
        return;
    }

    auto stored_data = it->second;
    get_ompt_parallel_cb_storage().erase(it);
    auto call_stack = get_backtrace(_bt_data);

    cache_category<category::rocm_ompt_api>();
    cache_add_thread_info(record.thread_id);
    cache_region(&record, stored_data._beg_ts, _end_ts, call_stack.dump(),
                 get_args_string(stored_data.args),
                 trait::name<category::rocm_ompt_api>::value);
}

void
ompt_finalize_orphan_events()
{
    auto empty_call_stack =
        std::optional<std::vector<tim::unwind::processed_entry>>{ std::nullopt };
    for(const auto& [parallel_data, stored_data] : get_ompt_parallel_cb_storage())
    {
        ompt_cache_orphan_event(stored_data, empty_call_stack);
    }

    for(const auto& [correlation_id, stored_data] : get_ompt_standard_cb_storage())
    {
        ompt_cache_orphan_event(stored_data, empty_call_stack);
    }

    get_ompt_parallel_cb_storage().clear();
    get_ompt_standard_cb_storage().clear();
}

// To handle events without finalization, perfetto push must occur in start
// Allows capture of worker thread implicit tasks and sync regions
void
ompt_tracing_callback_start(rocprofiler_callback_tracing_record_t record,
                            rocprofiler_user_data_t* /*user_data*/,
                            rocprofiler_timestamp_t ts)
{
    std::string_view _name = ompt_get_unified_name(record);

    if(get_use_timemory())
    {
        tracing::push_timemory(category::rocm_ompt_api{}, _name);
    }

    if(get_use_perfetto())
    {
        auto args = callback_arg_array_t{};
        if(config::get_perfetto_annotations())
        {
            rocprofiler_iterate_callback_tracing_kind_operation_args(record, save_args, 1,
                                                                     &args);
        }

        uint64_t _beg_ts   = ts;
        auto     stream_id = stream_id_top();

        tracing::push_perfetto_ts(
            category::rocm_ompt_api{}, _name.data(), _beg_ts,
            ::perfetto::Flow::ProcessScoped(record.correlation_id.internal),
            [&](::perfetto::EventContext ctx) {
                if(config::get_perfetto_annotations())
                {
                    tracing::add_perfetto_annotation(ctx, "begin_ns", _beg_ts);
                    tracing::add_perfetto_annotation(ctx, "stack_id",
                                                     record.correlation_id.internal);
                    if(stream_id.handle != 0)
                        tracing::add_perfetto_annotation(ctx, "stream_id",
                                                         stream_id.handle);
                    for(const auto& [key, val] : args)
                    {
                        tracing::add_perfetto_annotation(ctx, key, val);
                    }
                }
            });
    }
}

void
ompt_tracing_callback_stop(
    rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* /*user_data*/,
    rocprofiler_timestamp_t               ts,
    std::optional<std::vector<tim::unwind::processed_entry>>& _bt_data)
{
    std::string_view _name = ompt_get_unified_name(record);

    if(get_use_timemory())
    {
        tracing::pop_timemory(category::rocm_ompt_api{}, _name);
    }

    if(get_use_perfetto())
    {
        auto args = callback_arg_array_t{};
        if(config::get_perfetto_annotations())
        {
            rocprofiler_iterate_callback_tracing_kind_operation_args(record, save_args, 2,
                                                                     &args);
        }

        uint64_t _end_ts = ts;
        tracing::pop_perfetto_ts(
            category::rocm_ompt_api{}, _name.data(), _end_ts,
            [&](::perfetto::EventContext ctx) {
                if(config::get_perfetto_annotations())
                    tracing::add_perfetto_annotation(ctx, "end_ns", _end_ts);
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
                        auto _line =
                            (_linfo && _linfo.line > 0)
                                ? fmt::format("{}", _linfo.line)
                                : ((itr.lineno == 0) ? std::string{ "?" }
                                                     : fmt::format("{}", itr.lineno));
                        auto _entry = fmt::format("{} @ {}:{}",
                                                  rocprofsys::utility::demangle(*_func),
                                                  ::basename(_loc->c_str()), _line);
                        if(_bt_cnt < 10)
                        {
                            // Prepend zero for better ordering in UI. Only one zero
                            // is ever necessary since stack depth is limited to 16.
                            tracing::add_perfetto_annotation(
                                ctx, fmt::format("frame#0{}", _bt_cnt++), _entry);
                        }
                        else
                        {
                            tracing::add_perfetto_annotation(
                                ctx, fmt::format("frame#{}", _bt_cnt++), _entry);
                        }
                    }
                }
            });
    }
}

#endif

void
tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                      rocprofiler_user_data_t* user_data, void* /*callback_data*/)
{
    using backtrace_entry_vec_t  = std::vector<tim::unwind::processed_entry>;
    auto _bt_data                = std::optional<backtrace_entry_vec_t>{};
    auto populate_backtrace_data = [&]() {
        constexpr size_t backtrace_stack_depth       = 16;
        constexpr size_t backtrace_ignore_depth      = 3;
        constexpr bool   backtrace_with_signal_frame = true;
        auto             use_perfetto =
            (config::get_use_perfetto() && config::get_perfetto_annotations());
        auto use_rocpd = config::get_use_rocpd();

        if((use_perfetto || use_rocpd) &&
           tool_data->backtrace_operations.at(record.kind).count(record.operation) > 0)
        {
            auto _backtrace =
                tim::get_unw_stack<backtrace_stack_depth, backtrace_ignore_depth,
                                   backtrace_with_signal_frame>();
            _bt_data = backtrace_entry_vec_t{};
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
    };

#if(ROCPROFILER_VERSION >= 600)
    // Skip implicit_task associated with an "initial-task-begin" occurrence as
    // well as the thread_begin associated with an "initial-thread-begin" occurrence
    // as they are generated by our tool.
    // The two callbacks occur after our tool initializes OMPT but before the
    // first OpenMP region (user code) begins.
    // Note: Can occur multiple times (Ex: MPI+OpenMP hybrid)
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_OMPT)
    {
        auto* payload_data =
            static_cast<rocprofiler_callback_tracing_ompt_data_t*>(record.payload);
        switch(record.operation)
        {
            case ROCPROFILER_OMPT_ID_implicit_task:
            {
                int flag = payload_data->args.implicit_task.flags;
                if(flag & ompt_task_initial) return;  // Skips both the start and end
                break;
            }
            case ROCPROFILER_OMPT_ID_thread_begin:
            {
                ompt_thread_t thread_type = payload_data->args.thread_begin.thread_type;
                if(thread_type == ompt_thread_initial) return;
                break;
            }
            default: break;
        }
        // TODO: Once finalization issue is fixed, skip the corresponding end
        // of the thread_begin callback. Can be identified with:
        // - thread_end: The thread_data ptr from the thread_begin callback generated
        //    by the "initial-thread-begin" needs to match the thread_end's thread_data
        //    ptr
    }
#endif

    auto ts = rocprofiler_timestamp_t{};
    ROCPROFILER_CALL(rocprofiler_get_timestamp(&ts));
    const char* name = "";

    rocprofiler_query_callback_tracing_kind_operation_name(record.kind, record.operation,
                                                           &name, nullptr);

    auto info = std::stringstream{};
    info << std::left << "tid=" << record.thread_id << ", cid=" << std::setw(3)
         << record.correlation_id.internal << ", kind=" << std::setw(2) << record.kind
         << ", operation=" << std::setw(3) << record.operation
         << ", phase=" << record.phase << ", dt_nsec=" << std::setw(8) << ts
         << ", name=" << name;

    if(rocprofsys::get_state() != rocprofsys::State::Active)
    {
        LOG_WARNING("Callback called when tool is not active. {}", info.str().c_str());
        return;
    }

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
            case ROCPROFILER_CALLBACK_TRACING_OMPT:
            {
                ompt_tracing_callback_start(record, user_data, ts);
                ompt_push_standard_callback(record, ts);
                break;
            }
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
            case ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION:
            case ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION:
#endif
#if(ROCPROFILER_VERSION >= 700)
            case ROCPROFILER_CALLBACK_TRACING_HIP_STREAM:
#endif
            {
                if(get_is_continuous_integration())
                {
                    LOG_CRITICAL("Unhandled callback record kind: {}",
                                 static_cast<int>(record.kind));
                    ::rocprofsys::set_state(::rocprofsys::State::Finalized);
                    std::abort();
                }
                break;
            }
            default:
            {
                if(get_is_continuous_integration())
                {
                    LOG_CRITICAL("Unhandled callback record: {}", info.str());
                    ::rocprofsys::set_state(::rocprofsys::State::Finalized);
                    std::abort();
                }
                break;
            }
        }
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        populate_backtrace_data();

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
            case ROCPROFILER_CALLBACK_TRACING_OMPT:
            {
                ompt_tracing_callback_stop(record, user_data, ts, _bt_data);
                ompt_pop_standard_callback(record, ts, _bt_data);
                break;
            }
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
                auto* rccl_payload =
                    static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(
                        record.payload);
                tool_tracing_callback_rccl(record.operation, rccl_payload,
                                           user_data->value, ts);
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
            case ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION:
            case ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION:
#endif
#if(ROCPROFILER_VERSION >= 700)
            case ROCPROFILER_CALLBACK_TRACING_HIP_STREAM:
#endif
            {
                if(get_is_continuous_integration())
                {
                    LOG_CRITICAL("Unhandled callback record kind: {}",
                                 static_cast<int>(record.kind));
                    ::rocprofsys::set_state(::rocprofsys::State::Finalized);
                    std::abort();
                }
                break;
            }
            default:
            {
                if(get_is_continuous_integration())
                {
                    LOG_CRITICAL("Unhandled callback record: {}", info.str());
                    ::rocprofsys::set_state(::rocprofsys::State::Finalized);
                    std::abort();
                }
                break;
            }
        }
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_NONE)
    {
        switch(record.kind)
        {
            case ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH:
            {
                if(record.operation == ROCPROFILER_KERNEL_DISPATCH_COMPLETE)
                {
                    auto* _data =
                        static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(
                            record.payload);

                    // save for post-processing
                    get_kernel_dispatch_timestamps().emplace(
                        _data->dispatch_info.dispatch_id,
                        timing_interval{ _data->start_timestamp, _data->end_timestamp });
                }
            }
            break;
#if(ROCPROFILER_VERSION >= 600)
            case ROCPROFILER_CALLBACK_TRACING_OMPT:
            {
                // Callbacks that are received but that we do not process
                static const std::set<rocprofiler_ompt_operation_t> ompt_no_process = {
                    ROCPROFILER_OMPT_ID_callback_functions,  // "Fake" callback
                    // Not processed as these are received after our tool finalizes
                    ROCPROFILER_OMPT_ID_thread_end,
                };

                auto ompt_operation_type =
                    static_cast<rocprofiler_ompt_operation_t>(record.operation);
                if(ompt_no_process.find(ompt_operation_type) != ompt_no_process.end())
                    return;

                populate_backtrace_data();

                switch(ompt_operation_type)
                {
                    case ROCPROFILER_OMPT_ID_parallel_begin:
                        ompt_tracing_callback_start(record, user_data, ts);
                        ompt_push_parallel_callback(record, ts);
                        break;
                    case ROCPROFILER_OMPT_ID_parallel_end:
                        ompt_tracing_callback_stop(record, user_data, ts, _bt_data);
                        ompt_pop_parallel_callback(record, ts, _bt_data);
                        break;
                    // Unlike parallel callbacks, we cannot receive the corresponding end
                    // to thread_begin. Set thread_begin as "instant" so the user can
                    // see callback without it spanning the entire track
                    case ROCPROFILER_OMPT_ID_thread_begin:
                    case ROCPROFILER_OMPT_ID_lock_init:
                    case ROCPROFILER_OMPT_ID_lock_destroy:
                    // Although this has endpoint arg, treat it as instant event
                    case ROCPROFILER_OMPT_ID_nest_lock:
                    case ROCPROFILER_OMPT_ID_dispatch:
                    case ROCPROFILER_OMPT_ID_flush:
                    case ROCPROFILER_OMPT_ID_cancel:
                    case ROCPROFILER_OMPT_ID_device_initialize:
                    case ROCPROFILER_OMPT_ID_device_finalize:
                    case ROCPROFILER_OMPT_ID_device_load:
                    // case ROCPROFILER_OMPT_ID_device_unload: // Unsupported by runtime
                    case ROCPROFILER_OMPT_ID_task_create:
                    case ROCPROFILER_OMPT_ID_task_schedule:
                    case ROCPROFILER_OMPT_ID_mutex_released:
                    case ROCPROFILER_OMPT_ID_mutex_acquire:
                    case ROCPROFILER_OMPT_ID_mutex_acquired:
                    case ROCPROFILER_OMPT_ID_dependences:
                    case ROCPROFILER_OMPT_ID_task_dependence:
                    case ROCPROFILER_OMPT_ID_error:
                    {
                        // These callbacks are considered instant events and should start
                        // and immediately call stop as no corresponding "end" will be
                        // received
                        auto instant_ts = ts;
                        ompt_tracing_callback_start(record, user_data, instant_ts);
                        ompt_tracing_callback_stop(record, user_data, instant_ts,
                                                   _bt_data);
                        ompt_cache_instant_event(record, instant_ts, _bt_data);
                        break;
                    }
                    default:
                        LOG_WARNING("tool_tracing_callback: unhandled PHASE_NONE "
                                    "callback record: {}",
                                    info.str());
                }
            }
            break;
#endif
            default:
            {
                LOG_WARNING("tool_tracing_callback: unhandled PHASE_NONE "
                            "callback record: {}",
                            info.str());
            }
            break;
        }
    }
    else
    {
        if(get_is_continuous_integration())
        {
            LOG_CRITICAL("unhandled callback record phase: {}",
                         static_cast<int>(record.phase));
            ::rocprofsys::set_state(::rocprofsys ::State ::Finalized);
            ::std ::abort();
        }
        LOG_WARNING("tool_tracing_callback: unhandled callback record: {}", info.str());
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
        return fmt::format("HIP Activity Stream {}", _stream_id);
    };

    const bool _default_group_by_queue = get_group_by_queue();

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

                bool _group_by_queue = _default_group_by_queue;

                const auto* _kern_sym_data =
                    get_kernel_symbol_info(record->dispatch_info.kernel_id);

                auto _name = rocprofsys::utility::demangle(_kern_sym_data->kernel_name);
                auto _stack_id     = record->correlation_id.internal;
                auto _beg_ns       = record->start_timestamp;
                auto _end_ns       = record->end_timestamp;
                auto _agent_id     = record->dispatch_info.agent_id;
                auto _queue_id     = record->dispatch_info.queue_id;
                const auto* _agent = tool_data->get_gpu_tool_agent(_agent_id);

                uint64_t _stream_id = get_stream_id(record).handle;
                if(_stream_id == 0)
                {
                    // kernel_dispatch is not associated with a HIP stream
                    _group_by_queue = true;
                }

                {
                    cache_category<category::rocm_kernel_dispatch>();
                    cache_add_thread_info(record->thread_id);
                    cache_add_track(fmt::format("GPU Kernel Dispatch [{}] Queue {}",
                                                _agent->device_id, _queue_id.handle)
                                        .c_str(),
                                    record->thread_id);
                    cache_kernel_dispatch(record, _stream_id);
                }

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
                            tracing::add_perfetto_annotation(ctx, "stack_id", _stack_id);
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
                                fmt::format("({},{},{})",
                                            record->dispatch_info.workgroup_size.x,
                                            record->dispatch_info.workgroup_size.y,
                                            record->dispatch_info.workgroup_size.z));
                            tracing::add_perfetto_annotation(
                                ctx, "grid_size",
                                fmt::format("({},{},{})",
                                            record->dispatch_info.grid_size.x,
                                            record->dispatch_info.grid_size.y,
                                            record->dispatch_info.grid_size.z));
                        }
                    };

                    if(_group_by_queue)
                    {
                        auto _track_desc = [](int32_t _device_id_v, int64_t _queue_id_v) {
                            return fmt::format("GPU Kernel Dispatch [{}] Queue {}",
                                               _device_id_v, _queue_id_v);
                        };

                        const auto _track = tracing::get_perfetto_track(
                            category::rocm_kernel_dispatch{}, _track_desc,
                            _agent->device_id, _queue_id.handle);

                        tracing::push_perfetto(category::rocm_kernel_dispatch{},
                                               _name.c_str(), _track, _beg_ns,
                                               ::perfetto::Flow::ProcessScoped(_stack_id),
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
                                               ::perfetto::Flow::ProcessScoped(_stack_id),
                                               add_perfetto_annotations);

                        tracing::pop_perfetto(category::rocm_hip_stream{}, _name.c_str(),
                                              _track, _end_ns);
                    }
                }
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_scratch_memory_record_t*>(
                        header->payload);

                bool _group_by_queue = _default_group_by_queue;

                const auto* agent     = tool_data->get_gpu_tool_agent(record->agent_id);
                auto        device_id = static_cast<uint32_t>(agent->device_id);

                const auto& t_info = thread_info::get(record->thread_id, SystemTID);
                auto        thread_id_sequent = t_info->index_data->sequent_value;

                auto _corr_id = record->correlation_id.internal;
                auto _beg_ns  = record->start_timestamp;
                auto _end_ns  = record->end_timestamp;
                auto _name =
                    tool_data->buffered_tracing_info.at(record->kind, record->operation);

                auto _stream_id = get_stream_id(record).handle;
                if(_stream_id == 0)
                {
                    // Scratch memory event is not associated with a HIP stream
                    _group_by_queue = true;
                }

                {
                    auto track_name = fmt::format("GPU Scratch Memory [{}] Thread {}",
                                                  device_id, record->thread_id);
                    cache_category<category::rocm_scratch_memory>();
                    cache_add_thread_info(record->thread_id);
                    cache_add_track(track_name.c_str(), record->thread_id);
                    cache_scratch_memory(record, _stream_id);
                }

                if(get_use_timemory())
                {
                    auto _bundle = kernel_dispatch_bundle_t{ _name };

                    _bundle.push(thread_id_sequent).start().stop();
                    _bundle.get([_beg_ns, _end_ns](tim::component::wall_clock* _wc) {
                        _wc->set_value(_end_ns - _beg_ns);
                        _wc->set_accum(_end_ns - _beg_ns);
                    });
                    _bundle.pop();
                }

                if(get_use_perfetto())
                {
// The version of rocprofiler_buffer_tracing_scratch_memory_record_t from ROCm < 7.1 does
// not have the allocation_size field. ROCPROFILER_VERSION for both ROCm 7.0 and 7.1
// is 1.0.0, so we need to check the ROCm version.
#if(ROCPROFSYS_USE_ROCM > 0 && ROCPROFSYS_ROCM_VERSION >= 70100)
                    using counter_track = perfetto_counter_track<
                        rocprofiler_buffer_tracing_scratch_memory_record_t>;

                    if(!counter_track::exists(device_id))
                    {
                        auto track_name_alloc_size =
                            fmt::format("GPU Scratch Memory [{}] (S) Thread {}",
                                        device_id, thread_id_sequent);
                        counter_track::emplace(device_id, track_name_alloc_size, "bytes");
                    }

                    if(record->operation == ROCPROFILER_SCRATCH_MEMORY_ALLOC)
                    {
                        TRACE_COUNTER("rocm_scratch_memory",
                                      counter_track::at(device_id, 0), _beg_ns,
                                      record->allocation_size);
                    }
#endif
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
                        auto track_name_events = [&]() {
                            return fmt::format("GPU Scratch Memory (S) Events Thread {}",
                                               thread_id_sequent);
                        };
                        const auto _track = tracing::get_perfetto_track(
                            category::rocm_scratch_memory{}, track_name_events);

                        tracing::push_perfetto(category::rocm_scratch_memory{},
                                               _name.data(), _track, _beg_ns,
                                               ::perfetto::Flow::ProcessScoped(_corr_id),
                                               add_perfetto_annotations);

                        tracing::pop_perfetto(category::rocm_scratch_memory{}, "", _track,
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
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(
                        header->payload);

                bool _group_by_queue = _default_group_by_queue;

                auto        _stack_id     = record->correlation_id.internal;
                auto        _beg_ns       = record->start_timestamp;
                auto        _end_ns       = record->end_timestamp;
                auto        _dst_agent_id = record->dst_agent_id;
                auto        _src_agent_id = record->src_agent_id;
                const auto* _dst_agent    = tool_data->get_agent(_dst_agent_id);
                const auto* _src_agent    = tool_data->get_agent(_src_agent_id);
                auto        _name =
                    tool_data->buffered_tracing_info.at(record->kind, record->operation);

                uint64_t _stream_id = get_stream_id(record).handle;
                if(_stream_id == 0)
                {
                    // memory_copy is not associated with a HIP stream
                    _group_by_queue = true;
                }

                {
                    size_t      thread_idx = record->thread_id;
                    std::string track_name;

                    track_name = fmt::format("GPU Memory Copy to Agent [{}] Thread {}",
                                             _dst_agent->logical_node_id, thread_idx);

                    cache_category<category::rocm_memory_copy>();
                    cache_add_track(track_name.c_str(), record->thread_id);

                    cache_memory_copy(record, _stream_id);
                }

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
                            tracing::add_perfetto_annotation(ctx, "stack_id", _stack_id);
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
                            return fmt::format("GPU Memory Copy to Agent [{}] Thread {}",
                                               _device_id_v,
                                               _tid_v->index_data->sequent_value);
                        };

                        const auto _track = tracing::get_perfetto_track(
                            category::rocm_memory_copy{}, _track_desc,
                            _dst_agent->logical_node_id, record->thread_id);

                        tracing::push_perfetto(category::rocm_memory_copy{}, _name.data(),
                                               _track, _beg_ns,
                                               ::perfetto::Flow::ProcessScoped(_stack_id),
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
                                               ::perfetto::Flow::ProcessScoped(_stack_id),
                                               add_perfetto_annotations);

                        tracing::pop_perfetto(category::rocm_hip_stream{}, "", _track,
                                              _end_ns);
                    }
                }
            }
#if(ROCPROFILER_VERSION >= 600)
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_memory_allocation_record_t*>(
                        header->payload);

                uint64_t _stream_id = get_stream_id(record).handle;
                {
                    cache_category<category::rocm_memory_allocate>();
                    cache_add_thread_info(record->thread_id);
                    cache_memory_allocation(record, _stream_id);
                }
            }
#endif
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_HSA_CORE_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API)
            {
                // Not handling those buffered events
                continue;
            }
            else
            {
                throw std::runtime_error(fmt::format(
                    "unexpected rocprofiler_record_header_t buffer tracing category "
                    "kind. category: {}, kind: {}",
                    static_cast<int>(header->category), static_cast<int>(header->kind)));
            }
        }
        else
        {
            throw std::runtime_error(fmt::format(
                "unexpected rocprofiler_record_header_t buffer tracing category "
                "kind. category: {}, kind: {}",
                static_cast<int>(header->category), static_cast<int>(header->kind)));
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

            if(!_agent)
            {
                LOG_CRITICAL("unable to find tool agent for agent (id={})",
                             _agent_id.handle);
                ::rocprofsys::set_state(::rocprofsys ::State ::Finalized);
                ::std ::abort();
            }
            if(!_info)
            {
                LOG_CRITICAL(
                    "unable to find counter info for counter (id={}) on agent (id={})",
                    itr.first.handle, _agent_id.handle);
                ::rocprofsys::set_state(::rocprofsys ::State ::Finalized);
                ::std ::abort();
            }

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
        LOG_TRACE(" operation = ROCPROFILER_HIP_STREAM_CREATE");
    }
    else if(record.operation == ROCPROFILER_HIP_STREAM_DESTROY)
    {
        LOG_TRACE(" operation = ROCPROFILER_HIP_STREAM_DESTROY");
    }
    else if(record.operation == ROCPROFILER_HIP_STREAM_SET)
    {
        // Push the stream ID onto the stream stack before underlying HIP function is
        // called
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            LOG_TRACE(" operation = ROCPROFILER_HIP_STREAM_SET, phase = "
                      "ROCPROFILER_CALLBACK_PHASE_ENTER, stream_id={}",
                      (unsigned long) stream_id.handle);
            stream_id_push(stream_id);
        }
        // Pop stream ID off of stream stack after underlying HIP function is completed
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
        {
            LOG_TRACE("operation = ROCPROFILER_HIP_STREAM_SET, phase = "
                      "ROCPROFILER_CALLBACK_PHASE_EXIT, stream_id={}",
                      (unsigned long) stream_id.handle);
            stream_id_pop();
        }
    }
    else
    {
        LOG_CRITICAL("Unknown operation for hip_stream_callback!");
        ::rocprofsys::set_state(::rocprofsys ::State ::Finalized);
        ::std ::exit(1);
    }
}
#endif

int
tool_init(rocprofiler_client_finalize_t fini_func, void* user_data)
{
    auto domains = settings::instance()->at("ROCPROFSYS_ROCM_DOMAINS");

    std::stringstream _domains_ss;
    for(const auto& itr : domains->get_choices())
        _domains_ss << "- " << itr << "\n";
    LOG_DEBUG("Available ROCm Domains: \n {}", _domains_ss.str());

    auto _callback_domains = rocprofiler_sdk::get_callback_domains();
    auto _buffered_domain  = rocprofiler_sdk::get_buffered_domains();
    auto _counter_events   = rocprofiler_sdk::get_rocm_events();
    auto _version          = rocprofiler_sdk::get_version();
    if(_version.formatted == 0)
    {
        LOG_WARNING("rocprofiler-sdk version not initialized");
    }

    auto* _data        = as_client_data(user_data);
    _data->client_fini = fini_func;

    _data->initialize();
    if(!_counter_events.empty()) _data->initialize_event_info();

    ROCPROFILER_CALL(rocprofiler_create_context(&_data->primary_ctx));

    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
        _data->primary_ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
        tool_code_object_callback, _data));

    auto external_corr_id_request_kinds =
        std::array<rocprofiler_external_correlation_id_request_kind_t, 3>{
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,
#if(ROCPROFILER_VERSION >= 600)
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION
#endif
        };

    // Insert the default stream and queue info to ensure that the default entry is
    {
        trace_cache::get_metadata_registry().add_stream(0);
        trace_cache::get_metadata_registry().add_queue(0);
    }
    // ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
    //     _data->primary_ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
    //     tool_code_object_callback, _data));

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
            ROCPROFILER_CALLBACK_TRACING_OMPT,
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

    constexpr auto buffer_size = 16 * 4096;
    constexpr auto watermark   = 15 * 4096;

    ROCPROFILER_CALL(rocprofiler_configure_external_correlation_id_request_service(
        _data->primary_ctx, external_corr_id_request_kinds.data(),
        external_corr_id_request_kinds.size(),
        set_kernel_rename_and_stream_correlation_id, _data));

#if(ROCPROFILER_VERSION >= 700)
    if((_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) > 0) ||
       (_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) > 0))
    {
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
            _data->primary_ctx, ROCPROFILER_CALLBACK_TRACING_HIP_STREAM, nullptr, 0,
            tool_hip_stream_callback, nullptr));
    }
#endif

    if(_callback_domains.count(ROCPROFILER_CALLBACK_TRACING_RCCL_API) > 0)
    {
        rocprofiler_sdk::rccl_comm_data_initialize();
    }

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
    // ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,          ///< @see
    // ::rocprofiler_hsa_core_api_id_t ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,
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
    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY) > 0)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(
            _data->primary_ctx, buffer_size, watermark,
            ROCPROFILER_BUFFER_POLICY_LOSSLESS, tool_tracing_buffered, tool_data,
            &_data->scratch_memory_buffer));

        ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY, nullptr, 0,
            _data->scratch_memory_buffer));
    }

#if(ROCPROFILER_VERSION >= 600)
    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION) > 0)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(
            _data->primary_ctx, buffer_size, watermark,
            ROCPROFILER_BUFFER_POLICY_LOSSLESS, tool_tracing_buffered, tool_data,
            &_data->memory_alloc_buffer));
        if(_data->memory_alloc_buffer.handle == 0UL)
        {
            if(get_is_continuous_integration())
            {
                LOG_CRITICAL("Failed to create memory allocation buffer");
                ::rocprofsys::set_state(::rocprofsys ::State ::Finalized);
                ::std ::abort();
            }
        }
        auto _ops =
            rocprofiler_sdk::get_operations(ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION);

        ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION, nullptr, 0,
            _data->memory_alloc_buffer));
    }
#endif

    if(!_counter_events.empty())
    {
        for(const auto& itr : _data->gpu_agents)
        {
            const auto& _agent_id = rocprofiler_agent_id_t{ itr.agent->handle };
            _data->agent_events.emplace(
                _agent_id, create_agent_profile(_agent_id, _counter_events, _data));
        }

        ROCPROFILER_CALL(rocprofiler_create_context(&_data->counter_ctx));

        auto _operations = std::array<rocprofiler_tracing_operation_t, 1>{
            ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
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

    // throwaway context for handling the profiler control API. If primary_ctx were
    // used, we would get profiler pause callback but never get profiler resume
    // callback
    {
        auto _local_ctx = rocprofiler_context_id_t{ 0 };
        ROCPROFILER_CALL(rocprofiler_create_context(&_local_ctx));
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
            _local_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API, nullptr, 0,
            tool_control_callback, _data));
    }

    if(!is_valid(_data->primary_ctx))
    {
        // notify rocprofiler that initialization failed and all the contexts,
        // buffers, etc. created should be ignored
        return -1;
    }

    gpu::add_device_metadata();

    if(config::get_use_process_sampling() && config::get_use_amd_smi())
    {
        LOG_DEBUG("Setting amd_smi state to active...");
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

#if(ROCPROFILER_VERSION >= 600)
    ompt_finalize_orphan_events();
#endif

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
{}

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

    LOG_DEBUG("{}", info.str());
    LOG_DEBUG("client_id={}, priority={}", id->handle, priority);

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
