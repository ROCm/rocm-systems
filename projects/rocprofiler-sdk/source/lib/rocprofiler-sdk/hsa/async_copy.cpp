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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/hsa/async_copy.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/details/ostream.hpp"
#include "lib/rocprofiler-sdk/hsa/utils.hpp"
#include "rocprofiler-sdk/fwd.h"
#include "rocprofiler-sdk/hsa/api_id.h"

#include <glog/logging.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>

#include <cstdlib>

#define ROCP_HSA_TABLE_CALL(SEVERITY, EXPR)                                                        \
    auto ROCPROFILER_VARIABLE(rocp_hsa_table_call_, __LINE__) = (EXPR);                            \
    LOG_IF(SEVERITY, ROCPROFILER_VARIABLE(rocp_hsa_table_call_, __LINE__) != HSA_STATUS_SUCCESS)   \
        << #EXPR << " returned non-zero status code "                                              \
        << ROCPROFILER_VARIABLE(rocp_hsa_table_call_, __LINE__) << " "

#if defined(ROCPROFILER_CI)
#    define ROCP_CI_LOG_IF(NON_CI_LEVEL, ...) LOG_IF(FATAL, __VA_ARGS__)
#    define ROCP_CI_LOG(NON_CI_LEVEL, ...)    LOG(FATAL)
#else
#    define ROCP_CI_LOG_IF(NON_CI_LEVEL, ...) LOG_IF(NON_CI_LEVEL, __VA_ARGS__)
#    define ROCP_CI_LOG(NON_CI_LEVEL, ...)    LOG(NON_CI_LEVEL)
#endif

#define ROCPROFILER_LIB_ROCPROFILER_HSA_ASYNC_COPY_CPP_IMPL 1

// template specializations
#include "hsa.def.cpp"

namespace rocprofiler
{
namespace hsa
{
namespace async_copy
{
namespace
{
using context_t              = context::context;
using context_array_t        = common::container::small_vector<const context_t*>;
using external_corr_id_map_t = std::unordered_map<const context_t*, rocprofiler_user_data_t>;

template <size_t Idx>
struct async_copy_info;

#define SPECIALIZE_ASYNC_COPY_INFO(DIRECTION)                                                      \
    template <>                                                                                    \
    struct async_copy_info<ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_##DIRECTION>                     \
    {                                                                                              \
        static constexpr auto operation_idx = ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_##DIRECTION;  \
        static constexpr auto name          = #DIRECTION;                                          \
    };

SPECIALIZE_ASYNC_COPY_INFO(NONE)
SPECIALIZE_ASYNC_COPY_INFO(HOST_TO_HOST)
SPECIALIZE_ASYNC_COPY_INFO(HOST_TO_DEVICE)
SPECIALIZE_ASYNC_COPY_INFO(DEVICE_TO_HOST)
SPECIALIZE_ASYNC_COPY_INFO(DEVICE_TO_DEVICE)

#undef SPECIALIZE_ASYNC_COPY_INFO

template <size_t Idx, size_t... IdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id) return async_copy_info<Idx>::name;
    if constexpr(sizeof...(IdxTail) > 0)
        return name_by_id(id, std::index_sequence<IdxTail...>{});
    else
        return nullptr;
}

template <size_t Idx, size_t... IdxTail>
uint32_t
id_by_name(const char* name, std::index_sequence<Idx, IdxTail...>)
{
    if(std::string_view{async_copy_info<Idx>::name} == std::string_view{name})
        return async_copy_info<Idx>::operation_idx;
    if constexpr(sizeof...(IdxTail) > 0)
        return id_by_name(name, std::index_sequence<IdxTail...>{});
    else
        return ROCPROFILER_HSA_API_ID_NONE;
}

template <size_t... Idx>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < ROCPROFILER_HSA_API_ID_LAST) _vec.emplace_back(_v);
    };

    (_emplace(_id_list, async_copy_info<Idx>::operation_idx), ...);
}

template <size_t... Idx>
void
get_names(std::vector<const char*>& _name_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, const char* _v) {
        if(_v != nullptr && strnlen(_v, 1) > 0) _vec.emplace_back(_v);
    };

    (_emplace(_name_list, async_copy_info<Idx>::name), ...);
}

bool
context_filter(const context::context* ctx)
{
    return (ctx->buffered_tracer &&
            (ctx->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)));
}

constexpr auto null_rocp_agent_id =
    rocprofiler_agent_id_t{.handle = std::numeric_limits<uint64_t>::max()};

struct async_copy_data
{
    hsa_signal_t                        orig_signal = {};
    hsa_signal_t                        rocp_signal = {};
    rocprofiler_thread_id_t             tid         = common::get_tid();
    rocprofiler_agent_id_t              dst_agent   = null_rocp_agent_id;
    rocprofiler_agent_id_t              src_agent   = null_rocp_agent_id;
    rocprofiler_memory_copy_operation_t direction   = ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_NONE;
    context::correlation_id*            correlation_id  = nullptr;
    context::context_array_t            contexts        = {};
    external_corr_id_map_t              extern_corr_ids = {};
};

auto*
get_active_signals()
{
    static auto* _v = common::static_object<std::atomic<int64_t>>::construct();
    return _v;
}

template <typename Tp, typename Up>
constexpr Tp*
convert_hsa_handle(Up _hsa_object)
{
    static_assert(!std::is_pointer<Up>::value, "pass opaque struct");
    static_assert(!std::is_pointer<Tp>::value, "pass non-pointer type");
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<Tp*>(_hsa_object.handle);
}

bool
async_copy_handler(hsa_signal_value_t signal_value, void* arg)
{
    static auto sysclock_period = []() -> uint64_t {
        constexpr auto nanosec     = 1000000000UL;
        uint64_t       sysclock_hz = 0;
        ROCP_HSA_TABLE_CALL(ERROR,
                            get_table().core_->hsa_system_get_info_fn(
                                HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &sysclock_hz));
        return (nanosec / sysclock_hz);
    }();

    auto* _data            = static_cast<async_copy_data*>(arg);
    auto  copy_time        = hsa_amd_profiling_async_copy_time_t{};
    auto  copy_time_status = get_table().amd_ext_->hsa_amd_profiling_get_async_copy_time_fn(
        _data->rocp_signal, &copy_time);

    // normalize
    copy_time.start *= sysclock_period;
    copy_time.end *= sysclock_period;

    // if we encounter this in CI, it will cause test to fail
    ROCP_CI_LOG_IF(ERROR, copy_time_status == HSA_STATUS_SUCCESS && copy_time.end < copy_time.start)
        << "hsa_amd_profiling_get_async_copy_time for returned async times where the end time ("
        << copy_time.end << ") was less than the start time (" << copy_time.start << ")";

    // get the contexts that were active when the signal was created
    const auto& ctxs = _data->contexts;
    // we need to decrement this reference count at the end of the functions
    auto* _corr_id = _data->correlation_id;

    if(copy_time_status == HSA_STATUS_SUCCESS && !ctxs.empty())
    {
        const auto& _extern_corr_ids = _data->extern_corr_ids;

        for(const auto* itr : ctxs)
        {
            auto* _buffer = buffer::get_buffer(
                itr->buffered_tracer->buffer_data.at(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY));

            // go ahead and create the correlation id value since we expect at least one of these
            // domains will require it
            auto _corr_id_v =
                rocprofiler_correlation_id_t{.internal = 0, .external = context::null_user_data};
            if(_corr_id)
            {
                _corr_id_v.internal = _corr_id->internal;
                _corr_id_v.external = _extern_corr_ids.at(itr);
            }

            if(itr->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY))
            {
                if(copy_time_status == HSA_STATUS_SUCCESS)
                {
                    auto record = rocprofiler_buffer_tracing_memory_copy_record_t{
                        sizeof(rocprofiler_buffer_tracing_memory_copy_record_t),
                        ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                        _corr_id_v,
                        _data->direction,
                        copy_time.start * sysclock_period,
                        copy_time.end * sysclock_period,
                        _data->dst_agent,
                        _data->src_agent};

                    CHECK_NOTNULL(_buffer)->emplace(ROCPROFILER_BUFFER_CATEGORY_TRACING,
                                                    ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                                    record);
                }
            }
        }
    }

    if(_corr_id) _corr_id->ref_count.fetch_sub(1);

    auto* orig_amd_signal = convert_hsa_handle<amd_signal_t>(_data->orig_signal);

    // Original intercepted signal completion
    if(orig_amd_signal)
    {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto* rocp_amd_signal = convert_hsa_handle<amd_signal_t>(_data->rocp_signal);

        std::tie(orig_amd_signal->start_ts, orig_amd_signal->end_ts) =
            std::tie(rocp_amd_signal->start_ts, rocp_amd_signal->end_ts);

        const hsa_signal_value_t new_value =
            get_table().core_->hsa_signal_load_relaxed_fn(_data->orig_signal) - 1;

        LOG_IF(ERROR, signal_value != new_value) << "bad original signal value in " << __FUNCTION__;

        get_table().core_->hsa_signal_store_screlease_fn(_data->orig_signal, signal_value);
    }

    if(signal_value == 0)
    {
        ROCP_HSA_TABLE_CALL(ERROR, get_table().core_->hsa_signal_destroy_fn(_data->rocp_signal));
        delete _data;
        get_active_signals()->fetch_sub(1);
    }

    return (signal_value > 0);
}

enum async_copy_id
{
    async_copy_id           = ROCPROFILER_HSA_API_ID_hsa_amd_memory_async_copy,
    async_copy_on_engine_id = ROCPROFILER_HSA_API_ID_hsa_amd_memory_async_copy_on_engine,
    async_copy_rect_id      = ROCPROFILER_HSA_API_ID_hsa_amd_memory_async_copy_rect,
};

template <size_t Idx>
auto&
get_next_dispatch()
{
    using function_t     = typename hsa_api_meta<Idx>::function_type;
    static function_t _v = nullptr;
    return _v;
}

template <size_t Idx>
struct arg_indices;

#define HSA_ASYNC_COPY_DEFINE_ARG_INDICES(                                                         \
    ENUM_ID, DST_AGENT_IDX, SRC_AGENT_IDX, COMPLETION_SIGNAL_IDX)                                  \
    template <>                                                                                    \
    struct arg_indices<ENUM_ID>                                                                    \
    {                                                                                              \
        static constexpr auto dst_agent_idx         = DST_AGENT_IDX;                               \
        static constexpr auto src_agent_idx         = SRC_AGENT_IDX;                               \
        static constexpr auto completion_signal_idx = COMPLETION_SIGNAL_IDX;                       \
    };

HSA_ASYNC_COPY_DEFINE_ARG_INDICES(async_copy_id, 1, 3, 7)
HSA_ASYNC_COPY_DEFINE_ARG_INDICES(async_copy_on_engine_id, 1, 3, 7)
HSA_ASYNC_COPY_DEFINE_ARG_INDICES(async_copy_rect_id, 5, 5, 9)

template <typename FuncT, typename ArgsT, size_t... Idx>
decltype(auto)
invoke(FuncT&& _func, ArgsT&& _args, std::index_sequence<Idx...>)
{
    return std::forward<FuncT>(_func)(std::get<Idx>(_args)...);
}

template <size_t Idx, typename... Args>
hsa_status_t
async_copy_impl(Args... args)
{
    using meta_type = hsa_api_meta<Idx>;

    constexpr auto N = sizeof...(Args);

    auto&& _tied_args = std::tie(args...);
    auto   ctxs       = context::get_active_contexts(context_filter);

    // no active contexts so just execute original
    if(ctxs.empty())
    {
        return invoke(
            get_next_dispatch<Idx>(), std::move(_tied_args), std::make_index_sequence<N>{});
    }

    // determine the direction of the memory copy
    auto _direction    = ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_NONE;
    auto _src_agent_id = rocprofiler_agent_id_t{};
    auto _dst_agent_id = rocprofiler_agent_id_t{};
    {
        // indices in the tuple with references to the arguments
        constexpr auto dst_agent_idx = arg_indices<Idx>::dst_agent_idx;
        constexpr auto src_agent_idx = arg_indices<Idx>::src_agent_idx;

        // extract the completion signal argument and the destination hsa_agent_t
        auto _hsa_dst_agent = std::get<dst_agent_idx>(_tied_args);
        auto _hsa_src_agent = std::get<src_agent_idx>(_tied_args);

        // map the hsa agents to rocprofiler agents
        auto _rocp_dst_agent = agent::get_rocprofiler_agent(_hsa_dst_agent);
        auto _rocp_src_agent = agent::get_rocprofiler_agent(_hsa_src_agent);

        if(_rocp_dst_agent && _rocp_src_agent)
        {
            _src_agent_id = _rocp_src_agent->id;
            _dst_agent_id = _rocp_dst_agent->id;
            if(_rocp_src_agent->type == ROCPROFILER_AGENT_TYPE_CPU)
            {
                if(_rocp_dst_agent->type == ROCPROFILER_AGENT_TYPE_CPU)
                    _direction = ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_HOST_TO_HOST;
                else if(_rocp_dst_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                    _direction = ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_HOST_TO_DEVICE;
                else
                {
                    ROCP_CI_LOG(WARNING)
                        << meta_type::name
                        << " had an unhandled destination type: " << _rocp_dst_agent->type;
                }
            }
            else if(_rocp_src_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            {
                if(_rocp_dst_agent->type == ROCPROFILER_AGENT_TYPE_CPU)
                    _direction = ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_DEVICE_TO_HOST;
                else if(_rocp_dst_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                    _direction = ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_DEVICE_TO_DEVICE;
                else
                {
                    ROCP_CI_LOG(WARNING)
                        << meta_type::name
                        << " had an unhandled destination type: " << _rocp_dst_agent->type;
                }
            }
            else
            {
                ROCP_CI_LOG(WARNING) << meta_type::name
                                     << " had an unhandled source type: " << _rocp_dst_agent->type;
            }
        }
        else
        {
            LOG_IF(ERROR, !_rocp_src_agent)
                << "failed to find source rocprofiler agent for hsa agent with handle="
                << _hsa_src_agent.handle;
            LOG_IF(ERROR, !_rocp_dst_agent)
                << "failed to find destination rocprofiler agent for hsa agent with handle="
                << _hsa_dst_agent.handle;
        }
    }

    // remove any contexts which do not wish to trace this memory copy direction
    ctxs.erase(std::remove_if(ctxs.begin(),
                              ctxs.end(),
                              [_direction](const context_t* ctx) {
                                  return !ctx->buffered_tracer->domains(
                                      ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, _direction);
                              }),
               ctxs.end());

    // if no contexts remain, execute as usual
    if(ctxs.empty())
    {
        return invoke(
            get_next_dispatch<Idx>(), std::move(_tied_args), std::make_index_sequence<N>{});
    }

    // at this point, we want to install our own signal handler
    auto* _data      = new async_copy_data{};
    _data->tid       = common::get_tid();
    _data->dst_agent = _dst_agent_id;
    _data->src_agent = _src_agent_id;
    _data->direction = _direction;
    _data->contexts  = ctxs;  // avoid using move in case code below accidentally uses ctxs

    constexpr auto           completion_signal_idx = arg_indices<Idx>::completion_signal_idx;
    auto&                    _completion_signal    = std::get<completion_signal_idx>(_tied_args);
    const hsa_signal_value_t _completion_signal_val =
        get_table().core_->hsa_signal_load_scacquire_fn(_completion_signal);

    {
        const uint32_t     num_consumers = 0;
        const hsa_agent_t* consumers     = nullptr;
        auto               _status       = get_table().core_->hsa_signal_create_fn(
            _completion_signal_val, num_consumers, consumers, &_data->rocp_signal);

        if(_status != HSA_STATUS_SUCCESS)
        {
            LOG(ERROR) << "hsa_signal_create returned non-zero error code " << _status;

            delete _data;
            return invoke(
                get_next_dispatch<Idx>(), std::move(_tied_args), std::make_index_sequence<N>{});
        }
        else
        {
            get_active_signals()->fetch_add(1);
        }
    }

    {
        auto _status =
            get_table().amd_ext_->hsa_amd_signal_async_handler_fn(_data->rocp_signal,
                                                                  HSA_SIGNAL_CONDITION_LT,
                                                                  _completion_signal_val,
                                                                  async_copy_handler,
                                                                  _data);

        if(_status != HSA_STATUS_SUCCESS)
        {
            LOG(ERROR) << "hsa_amd_signal_async_handler returned non-zero error code " << _status;

            ROCP_HSA_TABLE_CALL(ERROR, get_table().core_->hsa_signal_destroy_fn(_data->rocp_signal))
                << ":: failed to destroy signal after async handler failed";

            get_active_signals()->fetch_sub(1);

            delete _data;
            return invoke(
                get_next_dispatch<Idx>(), std::move(_tied_args), std::make_index_sequence<N>{});
        }
    }

    _data->correlation_id = context::get_latest_correlation_id();
    auto& extern_corr_ids = _data->extern_corr_ids;

    // increase the reference count to denote that this correlation id is being used in a kernel
    if(_data->correlation_id)
    {
        extern_corr_ids.reserve(_data->contexts.size());  // reserve for performance
        for(const auto* ctx : _data->contexts)
            extern_corr_ids.emplace(ctx,
                                    ctx->correlation_tracer.external_correlator.get(_data->tid));
        _data->correlation_id->ref_count.fetch_add(1);
    }

    _data->orig_signal = _completion_signal;
    _completion_signal = _data->rocp_signal;

    return invoke(get_next_dispatch<Idx>(), std::move(_tied_args), std::make_index_sequence<N>{});
}

template <size_t Idx, typename RetT, typename... Args>
auto get_async_copy_impl(RetT (*)(Args...))
{
    return &async_copy_impl<Idx, Args...>;
}

template <size_t Idx>
void
async_copy_save(hsa_api_table_t* _orig)
{
    auto  _meta              = hsa_api_meta<Idx>{};
    auto& _table             = _meta.get_table(_orig);
    auto& _func              = _meta.get_table_func(_table);
    get_next_dispatch<Idx>() = _func;
}

template <size_t... Idx>
void
async_copy_save(hsa_api_table_t* _orig, std::index_sequence<Idx...>)
{
    (async_copy_save<Idx>(_orig), ...);
}

template <size_t Idx>
void
async_copy_wrap(hsa_api_table_t* _orig)
{
    auto  _meta  = hsa_api_meta<Idx>{};
    auto& _table = _meta.get_table(_orig);
    auto& _func  = _meta.get_table_func(_table);

    CHECK_NOTNULL(get_next_dispatch<Idx>());
    _func = get_async_copy_impl<Idx>(_func);
}

template <size_t... Idx>
void
async_copy_wrap(hsa_api_table_t* _orig, std::index_sequence<Idx...>)
{
    (async_copy_wrap<Idx>(_orig), ...);
}

using async_copy_index_seq_t =
    std::index_sequence<async_copy_id, async_copy_on_engine_id, async_copy_rect_id>;
}  // namespace

// check out the assembly here... this compiles to a switch statement
const char*
name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_LAST>{});
}

uint32_t
id_by_name(const char* name)
{
    return id_by_name(name,
                      std::make_index_sequence<ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_LAST>{});
}

std::vector<uint32_t>
get_ids()
{
    auto _data = std::vector<uint32_t>{};
    _data.reserve(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_LAST);
    get_ids(_data, std::make_index_sequence<ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_LAST>{});
    return _data;
}

std::vector<const char*>
get_names()
{
    auto _data = std::vector<const char*>{};
    _data.reserve(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_LAST);
    get_names(_data, std::make_index_sequence<ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_LAST>{});
    return _data;
}
}  // namespace async_copy

void
async_copy_init(hsa_api_table_t* _orig)
{
    if(_orig)
    {
        async_copy::async_copy_save(_orig, async_copy::async_copy_index_seq_t{});

        auto ctxs = context::get_registered_contexts(async_copy::context_filter);
        if(!ctxs.empty())
        {
            _orig->amd_ext_->hsa_amd_profiling_async_copy_enable_fn(true);
            async_copy::async_copy_wrap(_orig, async_copy::async_copy_index_seq_t{});
        }
    }
}

void
async_copy_fini()
{
    if(!async_copy::get_active_signals()) return;
    while(async_copy::get_active_signals()->load() > 0)
    {
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
    }
}
}  // namespace hsa
}  // namespace rocprofiler
