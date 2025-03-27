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
#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"
#include "lib/rocprofiler-sdk/tracing/profiling_time.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa/api_id.h>
#include <rocprofiler-sdk/hsa/table_id.h>

#include <glog/logging.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>

#include <chrono>
#include <cstdlib>
#include <type_traits>

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

template <size_t OpIdx>
struct async_copy_info;

#define SPECIALIZE_ASYNC_COPY_INFO(DIRECTION)                                                      \
    template <>                                                                                    \
    struct async_copy_info<ROCPROFILER_MEMORY_COPY_##DIRECTION>                                    \
    {                                                                                              \
        static constexpr auto operation_idx = ROCPROFILER_MEMORY_COPY_##DIRECTION;                 \
        static constexpr auto name          = "MEMORY_COPY_" #DIRECTION;                           \
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
        return ROCPROFILER_MEMORY_COPY_LAST;
}

template <size_t... Idx>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < static_cast<uint32_t>(ROCPROFILER_MEMORY_COPY_LAST)) _vec.emplace_back(_v);
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
    auto has_buffered = (ctx->buffered_tracer &&
                         (ctx->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)));

    auto has_callback = (ctx->callback_tracer &&
                         (ctx->callback_tracer->domains(ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY)));

    return (has_buffered || has_callback);
}

constexpr auto null_rocp_agent_id =
    rocprofiler_agent_id_t{.handle = std::numeric_limits<uint64_t>::max()};

struct async_copy_data
{
    using timestamp_t     = rocprofiler_timestamp_t;
    using callback_data_t = rocprofiler_callback_tracing_memory_copy_data_t;
    using buffered_data_t = rocprofiler_buffer_tracing_memory_copy_record_t;

    hsa_signal_t                        orig_signal    = {};
    hsa_signal_t                        rocp_signal    = {};
    rocprofiler_thread_id_t             tid            = common::get_tid();
    rocprofiler_agent_id_t              dst_agent      = null_rocp_agent_id;
    rocprofiler_agent_id_t              src_agent      = null_rocp_agent_id;
    rocprofiler_address_t               dst_address    = {.value = 0};
    rocprofiler_address_t               src_address    = {.value = 0};
    rocprofiler_memory_copy_operation_t direction      = ROCPROFILER_MEMORY_COPY_NONE;
    uint64_t                            bytes_copied   = 0;
    uint64_t                            start_ts       = 0;
    context::correlation_id*            correlation_id = nullptr;
    tracing::tracing_data               tracing_data   = {};

    callback_data_t get_callback_data(timestamp_t _beg = 0, timestamp_t _end = 0) const;
    buffered_data_t get_buffered_record(const context_t* _ctx,
                                        timestamp_t      _beg = 0,
                                        timestamp_t      _end = 0) const;

    auto get_lock() { return std::make_unique<std::unique_lock<std::mutex>>(m_mtx); }

private:
    std::mutex m_mtx = {};
};

async_copy_data::callback_data_t
async_copy_data::get_callback_data(timestamp_t _beg, timestamp_t _end) const
{
    ROCP_FATAL_IF(direction == ROCPROFILER_MEMORY_COPY_NONE) << "direction has not been set";

    return common::init_public_api_struct(callback_data_t{},
                                          _beg,
                                          _end,
                                          dst_agent,
                                          src_agent,
                                          bytes_copied,
                                          dst_address,
                                          src_address);
}

async_copy_data::buffered_data_t
async_copy_data::get_buffered_record(const context_t* _ctx,
                                     timestamp_t      _beg,
                                     timestamp_t      _end) const
{
    ROCP_FATAL_IF(direction == ROCPROFILER_MEMORY_COPY_NONE) << "direction has not been set";

    auto _external_corr_id =
        (_ctx) ? tracing_data.external_correlation_ids.at(_ctx) : context::null_user_data;
    auto _corr_id = rocprofiler_async_correlation_id_t{correlation_id->internal, _external_corr_id};

    return common::init_public_api_struct(buffered_data_t{},
                                          ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                          direction,
                                          _corr_id,
                                          correlation_id->thread_idx,
                                          _beg,
                                          _end,
                                          dst_agent,
                                          src_agent,
                                          bytes_copied,
                                          dst_address,
                                          src_address);
}

struct active_signals
{
    active_signals();
    ~active_signals()                         = default;
    active_signals(const active_signals&)     = delete;
    active_signals(active_signals&&) noexcept = delete;
    active_signals& operator=(const active_signals&) = delete;
    active_signals& operator=(active_signals&&) noexcept = delete;

    void create();          // create hsa signal
    void destroy();         // destroy hsa signal
    void sync();            // wait for outstanding signal completion callbacks
    void fetch_add(int v);  // increment hsa signal value
    void fetch_sub(int v);  // decrement hsa signal value

private:
    hsa_signal_t         m_signal = {.handle = 0};
    std::atomic<int64_t> m_count  = 0;
};

active_signals::active_signals()
{
    // only create if not started finalization
    if(registration::get_fini_status() == 0) create();
}

void
active_signals::create()
{
    if(m_signal.handle != 0) return;

    // function pointer may be null during unit testing
    if(hsa::get_hsa_ref_count() > 0 && get_core_table()->hsa_signal_create_fn)
    {
        ROCP_HSA_TABLE_CALL(ERROR,
                            get_core_table()->hsa_signal_create_fn(0, 0, nullptr, &m_signal));
    }
}

void
active_signals::destroy()
{
    if(m_signal.handle == 0) return;

    // function pointer may be null during unit testing
    if(hsa::get_hsa_ref_count() > 0 && get_core_table()->hsa_signal_destroy_fn)
    {
        ROCP_HSA_TABLE_CALL(ERROR, get_core_table()->hsa_signal_destroy_fn(m_signal));
        m_signal.handle = 0;
    }
}

void
active_signals::sync()
{
    if(m_signal.handle == 0) return;

#if defined(ROCPROFILER_CI_STRICT_TIMESTAMPS) && ROCPROFILER_CI_STRICT_TIMESTAMPS > 0
    constexpr auto timeout_sec = std::chrono::seconds{5};
#else
    // wait a maximum of thirty seconds
    constexpr auto timeout_sec = std::chrono::seconds{30};
#endif

    constexpr auto timeout =
        std::chrono::duration_cast<std::chrono::nanoseconds>(timeout_sec).count();

    if(m_count.load() > 0)
    {
        auto _cnt_beg      = m_count.load();
        auto _signal_value = get_core_table()->hsa_signal_wait_scacquire_fn(
            m_signal, HSA_SIGNAL_CONDITION_LT, 1, timeout, HSA_WAIT_STATE_ACTIVE);
        auto _cnt_end = m_count.load();
        if(_signal_value != 0)
        {
            ROCP_CI_LOG_IF(WARNING, _cnt_end > 0)
                << "rocprofiler-sdk timed out after " << timeout_sec.count()
                << " seconds waiting for " << _cnt_beg
                << " completion callbacks from HSA for async memory copy tracing. " << _cnt_end
                << " completion callbacks were not delivered";
        }
    }
}

void
active_signals::fetch_add(int v)
{
    create();
    if(m_signal.handle == 0) return;

    m_count.fetch_add(1);
    get_core_table()->hsa_signal_add_screlease_fn(m_signal, v);
}

void
active_signals::fetch_sub(int v)
{
    if(m_signal.handle == 0) return;

    auto _cnt = m_count.load();
    ROCP_CI_LOG_IF(WARNING, _cnt == 0) << "active_signals count (currently = 0) was requested to "
                                          "decrement more times than it was incremented";

    if(_cnt > 0) m_count.fetch_sub(1);
    get_core_table()->hsa_signal_subtract_screlease_fn(m_signal, v);
}

active_signals*
get_active_signals()
{
    static auto*& _v = common::static_object<active_signals>::construct();
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
    // if we have fully finalized, delete the data and return
    if(registration::get_fini_status() > 0)
    {
        auto* _data = static_cast<async_copy_data*>(arg);
        delete _data;
        return false;
    }

    auto  ts               = common::timestamp_ns();
    auto* _data            = static_cast<async_copy_data*>(arg);
    auto  _lk              = _data->get_lock();
    auto  copy_time        = hsa_amd_profiling_async_copy_time_t{};
    auto  copy_time_status = get_amd_ext_table()->hsa_amd_profiling_get_async_copy_time_fn(
        _data->rocp_signal, &copy_time);

    auto _profile_time = tracing::profiling_time{copy_time_status, copy_time.start, copy_time.end};

    // we need to decrement this reference count at the end of the functions
    auto* _corr_id = _data->correlation_id;
    auto  _dtor    = common::scope_destructor{[&_lk, &_data, &_corr_id]() {
        _lk.reset();  // reset the unique_ptr so the lock is released
        delete _data;

        if(_corr_id) _corr_id->sub_ref_count();
    }};

    if(_profile_time.status == HSA_STATUS_SUCCESS)
    {
        _profile_time = tracing::adjust_profiling_time(
            "memcpy",
            "hsa_amd_profiling_get_async_copy_time",
            _profile_time,
            tracing::profiling_time{HSA_STATUS_SUCCESS, _data->start_ts, ts});
    }
    else
    {
        ROCP_CI_LOG(ERROR) << fmt::format(
            "hsa_amd_profiling_get_async_copy_time for the {} copy operation from agent-{} to "
            "agent-{} returned status={} :: {}",
            std::string_view{hsa::async_copy::name_by_id(_data->direction)},
            CHECK_NOTNULL(agent::get_agent(_data->src_agent))->node_id,
            CHECK_NOTNULL(agent::get_agent(_data->dst_agent))->node_id,
            static_cast<int>(copy_time_status),
            hsa::get_hsa_status_string(copy_time_status));
    }

    // get the contexts that were active when the signal was created
    const auto& tracing_data = _data->tracing_data;

    if(_profile_time.status == HSA_STATUS_SUCCESS && !tracing_data.empty())
    {
        if(!_data->tracing_data.callback_contexts.empty())
        {
            auto _tracer_data = _data->get_callback_data(_profile_time.start, _profile_time.end);

            tracing::execute_phase_exit_callbacks(_data->tracing_data.callback_contexts,
                                                  _data->tracing_data.external_correlation_ids,
                                                  ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                                                  _data->direction,
                                                  _tracer_data);
        }

        if(!_data->tracing_data.buffered_contexts.empty())
        {
            auto record =
                _data->get_buffered_record(nullptr, _profile_time.start, _profile_time.end);

            tracing::execute_buffer_record_emplace(_data->tracing_data.buffered_contexts,
                                                   _data->tid,
                                                   _data->correlation_id->internal,
                                                   _data->tracing_data.external_correlation_ids,
                                                   _data->correlation_id->ancestor,
                                                   ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                                   _data->direction,
                                                   record);
        }
    }

    // decrement the active signals
    if(get_active_signals()) get_active_signals()->fetch_sub(1);

    auto* orig_amd_signal = convert_hsa_handle<amd_signal_t>(_data->orig_signal);

    // Original intercepted signal completion
    if(orig_amd_signal)
    {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto* rocp_amd_signal = convert_hsa_handle<amd_signal_t>(_data->rocp_signal);

        std::tie(orig_amd_signal->start_ts, orig_amd_signal->end_ts) =
            std::tie(rocp_amd_signal->start_ts, rocp_amd_signal->end_ts);

        const hsa_signal_value_t new_value =
            get_core_table()->hsa_signal_load_relaxed_fn(_data->orig_signal) - 1;

        ROCP_ERROR_IF(signal_value != new_value) << "bad original signal value in " << __FUNCTION__;
        // Move to ROCP_TRACE when rebasing
        ROCP_INFO << "Decrementing Signal: " << std::hex << _data->orig_signal.handle << std::dec;
        get_core_table()->hsa_signal_store_screlease_fn(_data->orig_signal, signal_value);
    }

    ROCP_HSA_TABLE_CALL(ERROR, get_core_table()->hsa_signal_destroy_fn(_data->rocp_signal));

    return false;
}

enum async_copy_id
{
    async_copy_id           = ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_async_copy,
    async_copy_on_engine_id = ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_async_copy_on_engine,
    async_copy_rect_id      = ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_async_copy_rect,
};

template <size_t TableIdx, size_t OpIdx>
auto&
get_next_dispatch()
{
    using function_t     = typename hsa_api_meta<TableIdx, OpIdx>::function_type;
    static function_t _v = nullptr;
    return _v;
}

template <size_t Idx>
struct arg_indices;

#define HSA_ASYNC_COPY_DEFINE_ARG_INDICES(ENUM_ID,                                                 \
                                          DST_AGENT_IDX,                                           \
                                          SRC_AGENT_IDX,                                           \
                                          COMPLETION_SIGNAL_IDX,                                   \
                                          COPY_SIZE_IDX,                                           \
                                          DST_ADDR_IDX,                                            \
                                          SRC_ADDR_IDX)                                            \
    template <>                                                                                    \
    struct arg_indices<ENUM_ID>                                                                    \
    {                                                                                              \
        static constexpr auto dst_agent_idx         = DST_AGENT_IDX;                               \
        static constexpr auto src_agent_idx         = SRC_AGENT_IDX;                               \
        static constexpr auto completion_signal_idx = COMPLETION_SIGNAL_IDX;                       \
        static constexpr auto copy_size_idx         = COPY_SIZE_IDX;                               \
        static constexpr auto dst_address_idx       = DST_ADDR_IDX;                                \
        static constexpr auto src_address_idx       = SRC_ADDR_IDX;                                \
    };

HSA_ASYNC_COPY_DEFINE_ARG_INDICES(async_copy_id, 1, 3, 7, 4, 0, 2)
HSA_ASYNC_COPY_DEFINE_ARG_INDICES(async_copy_on_engine_id, 1, 3, 7, 4, 0, 2)
HSA_ASYNC_COPY_DEFINE_ARG_INDICES(async_copy_rect_id, 5, 5, 9, 4, 0, 2)

template <typename FuncT, typename ArgsT, size_t... Idx>
decltype(auto)
invoke(FuncT&& _func, ArgsT&& _args, std::index_sequence<Idx...>)
{
    return std::forward<FuncT>(_func)(std::get<Idx>(_args)...);
}

template <typename Tp>
uint64_t compute_copy_bytes(Tp);

template <typename Tp>
rocprofiler_address_t
compute_address(const Tp*);

template <>
uint64_t
compute_copy_bytes(size_t val)
{
    return val;
}

template <>
uint64_t
compute_copy_bytes(const hsa_dim3_t* val)
{
    return (val) ? (val->x * val->y * val->z) : 0;
}

template <>
rocprofiler_address_t
compute_address(const void* val)
{
    return rocprofiler_address_t{.ptr = val};
}

template <>
rocprofiler_address_t
compute_address(const hsa_pitched_ptr_t* val)
{
    return rocprofiler_address_t{.ptr = val->base};
}

template <size_t TableIdx, size_t OpIdx, typename... Args>
hsa_status_t
async_copy_impl(Args... args)
{
    using meta_type = hsa_api_meta<TableIdx, OpIdx>;

    constexpr auto N             = sizeof...(Args);
    constexpr auto copy_size_idx = arg_indices<OpIdx>::copy_size_idx;
    constexpr auto dst_addr_idx  = arg_indices<OpIdx>::dst_address_idx;
    constexpr auto src_addr_idx  = arg_indices<OpIdx>::src_address_idx;

    auto&& _tied_args = std::tie(args...);

    // determine the direction of the memory copy
    auto _direction    = ROCPROFILER_MEMORY_COPY_NONE;
    auto _src_agent_id = rocprofiler_agent_id_t{};
    auto _dst_agent_id = rocprofiler_agent_id_t{};
    {
        // indices in the tuple with references to the arguments
        constexpr auto dst_agent_idx = arg_indices<OpIdx>::dst_agent_idx;
        constexpr auto src_agent_idx = arg_indices<OpIdx>::src_agent_idx;

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
                    _direction = ROCPROFILER_MEMORY_COPY_HOST_TO_HOST;
                else if(_rocp_dst_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                    _direction = ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE;
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
                    _direction = ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST;
                else if(_rocp_dst_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                    _direction = ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE;
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
            ROCP_ERROR_IF(!_rocp_src_agent)
                << "failed to find source rocprofiler agent for hsa agent with handle="
                << _hsa_src_agent.handle;
            ROCP_ERROR_IF(!_rocp_dst_agent)
                << "failed to find destination rocprofiler agent for hsa agent with handle="
                << _hsa_dst_agent.handle;
        }
    }

    async_copy_data* _data = nullptr;

    {
        auto tracing_data = tracing::tracing_data{};

        tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                                   ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                   _direction,
                                   tracing_data);
        // if no contexts are tracing memory copies for this direction, execute as usual
        if(tracing_data.empty())
        {
            return invoke(get_next_dispatch<TableIdx, OpIdx>(),
                          std::move(_tied_args),
                          std::make_index_sequence<N>{});
        }

        _data               = new async_copy_data{};
        _data->tracing_data = std::move(tracing_data);
    }

    auto  _lk          = _data->get_lock();
    auto& tracing_data = _data->tracing_data;

    // at this point, we want to install our own signal handler
    _data->tid          = common::get_tid();
    _data->dst_agent    = _dst_agent_id;
    _data->src_agent    = _src_agent_id;
    _data->direction    = _direction;
    _data->bytes_copied = compute_copy_bytes(std::get<copy_size_idx>(_tied_args));
    _data->dst_address  = compute_address(std::get<dst_addr_idx>(_tied_args));
    _data->src_address  = compute_address(std::get<src_addr_idx>(_tied_args));

    constexpr auto           completion_signal_idx  = arg_indices<OpIdx>::completion_signal_idx;
    auto&                    _completion_signal     = std::get<completion_signal_idx>(_tied_args);
    const hsa_signal_value_t _completion_signal_val = 1;

    auto original_value = get_core_table()->hsa_signal_load_scacquire_fn(_completion_signal);

    {
        const uint32_t     num_consumers = 0;
        const hsa_agent_t* consumers     = nullptr;
        auto               _status       = get_core_table()->hsa_signal_create_fn(
            _completion_signal_val, num_consumers, consumers, &_data->rocp_signal);

        if(_status != HSA_STATUS_SUCCESS)
        {
            ROCP_ERROR << "hsa_signal_create returned non-zero error code " << _status;

            delete _data;
            return invoke(get_next_dispatch<TableIdx, OpIdx>(),
                          std::move(_tied_args),
                          std::make_index_sequence<N>{});
        }
    }

    {
        auto _status = get_amd_ext_table()->hsa_amd_signal_async_handler_fn(_data->rocp_signal,
                                                                            HSA_SIGNAL_CONDITION_LT,
                                                                            _completion_signal_val,
                                                                            async_copy_handler,
                                                                            _data);

        if(_status != HSA_STATUS_SUCCESS)
        {
            ROCP_ERROR << "hsa_amd_signal_async_handler returned non-zero error code " << _status;

            ROCP_HSA_TABLE_CALL(ERROR, get_core_table()->hsa_signal_destroy_fn(_data->rocp_signal))
                << ":: failed to destroy signal after async handler failed";

            delete _data;
            return invoke(get_next_dispatch<TableIdx, OpIdx>(),
                          std::move(_tied_args),
                          std::make_index_sequence<N>{});
        }
    }

    _data->correlation_id                 = context::get_latest_correlation_id();
    context::correlation_id* _corr_id_pop = nullptr;

    if(!_data->correlation_id)
    {
        constexpr auto ref_count = 1;
        _data->correlation_id    = context::correlation_tracing_service::construct(ref_count);
        _corr_id_pop             = _data->correlation_id;
    }

    // increase the reference count to denote that this correlation id is being used in a kernel
    _data->correlation_id->add_ref_count();

    // if we constructed a correlation id, this decrements the reference count after the underlying
    // function returns
    auto _corr_id_dtor = common::scope_destructor{[_corr_id_pop, _data]() {
        if(_corr_id_pop)
        {
            context::pop_latest_correlation_id(_corr_id_pop);
            _corr_id_pop->sub_ref_count();
        }
        _data->start_ts = common::timestamp_ns();
    }};

    auto thr_id = _data->correlation_id->thread_idx;
    tracing::populate_external_correlation_ids(tracing_data.external_correlation_ids,
                                               thr_id,
                                               ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,
                                               _direction,
                                               _data->correlation_id->internal);

    if(!tracing_data.callback_contexts.empty())
    {
        auto _tracer_data = _data->get_callback_data();

        tracing::execute_phase_enter_callbacks(tracing_data.callback_contexts,
                                               thr_id,
                                               _data->correlation_id->internal,
                                               tracing_data.external_correlation_ids,
                                               _data->correlation_id->ancestor,
                                               ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                                               _direction,
                                               _tracer_data);
    }

    _data->orig_signal = _completion_signal;
    _completion_signal = _data->rocp_signal;

    ROCP_INFO << "Memcpy Original Signal " << std::hex << _data->orig_signal.handle << std::dec
              << ": " << original_value << " | Replacement Signal: " << std::hex
              << _completion_signal.handle << std::dec << ": 1";

    CHECK_NOTNULL(get_active_signals())->fetch_add(1);

    return invoke(
        get_next_dispatch<TableIdx, OpIdx>(), std::move(_tied_args), std::make_index_sequence<N>{});
}

template <size_t TableIdx, size_t OpIdx, typename RetT, typename... Args>
auto get_async_copy_impl(RetT (*)(Args...))
{
    return &async_copy_impl<TableIdx, OpIdx, Args...>;
}

template <size_t TableIdx, size_t OpIdx>
void
async_copy_save(hsa_amd_ext_table_t* _orig, uint64_t _tbl_instance)
{
    static_assert(
        std::is_same<hsa_amd_ext_table_t, typename hsa_table_lookup<TableIdx>::type>::value,
        "unexpected type");

    auto _meta = hsa_api_meta<TableIdx, OpIdx>{};

    // original table and function
    auto& _orig_table = _meta.get_table(_orig);
    auto& _orig_func  = _meta.get_table_func(_orig_table);

    // table with copy function
    auto& _copy_func = get_next_dispatch<TableIdx, OpIdx>();

    ROCP_FATAL_IF(_copy_func && _tbl_instance == 0)
        << _meta.name << " has non-null function pointer " << _copy_func
        << " despite this being the first instance of the library being copies";

    if(!_copy_func)
    {
        ROCP_TRACE << "copying table entry for " << _meta.name;
        _copy_func = _orig_func;
    }
    else
    {
        ROCP_TRACE << "skipping copying table entry for " << _meta.name << " from table instance "
                   << _tbl_instance;
    }
}

template <size_t TableIdx, size_t... OpIdx>
void
async_copy_save(hsa_amd_ext_table_t* _orig, uint64_t _tbl_instance, std::index_sequence<OpIdx...>)
{
    static_assert(
        std::is_same<hsa_amd_ext_table_t, typename hsa_table_lookup<TableIdx>::type>::value,
        "unexpected type");

    (async_copy_save<TableIdx, OpIdx>(_orig, _tbl_instance), ...);
}

template <size_t TableIdx, size_t OpIdx>
void
async_copy_wrap(hsa_amd_ext_table_t* _orig)
{
    static_assert(
        std::is_same<hsa_amd_ext_table_t, typename hsa_table_lookup<TableIdx>::type>::value,
        "unexpected type");

    auto  _meta  = hsa_api_meta<TableIdx, OpIdx>{};
    auto& _table = _meta.get_table(_orig);
    auto& _func  = _meta.get_table_func(_table);

    auto& _dispatch = get_next_dispatch<TableIdx, OpIdx>();
    CHECK_NOTNULL(_dispatch);
    _func = get_async_copy_impl<TableIdx, OpIdx>(_func);
}

template <size_t TableIdx, size_t... OpIdx>
void
async_copy_wrap(hsa_amd_ext_table_t* _orig, std::index_sequence<OpIdx...>)
{
    static_assert(
        std::is_same<hsa_amd_ext_table_t, typename hsa_table_lookup<TableIdx>::type>::value,
        "unexpected type");

    (async_copy_wrap<TableIdx, OpIdx>(_orig), ...);
}

using async_copy_index_seq_t =
    std::index_sequence<async_copy_id, async_copy_on_engine_id, async_copy_rect_id>;
}  // namespace

// check out the assembly here... this compiles to a switch statement
const char*
name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ROCPROFILER_MEMORY_COPY_LAST>{});
}

uint32_t
id_by_name(const char* name)
{
    return id_by_name(name, std::make_index_sequence<ROCPROFILER_MEMORY_COPY_LAST>{});
}

std::vector<uint32_t>
get_ids()
{
    auto _data = std::vector<uint32_t>{};
    _data.reserve(ROCPROFILER_MEMORY_COPY_LAST);
    get_ids(_data, std::make_index_sequence<ROCPROFILER_MEMORY_COPY_LAST>{});
    return _data;
}

std::vector<const char*>
get_names()
{
    auto _data = std::vector<const char*>{};
    _data.reserve(ROCPROFILER_MEMORY_COPY_LAST);
    get_names(_data, std::make_index_sequence<ROCPROFILER_MEMORY_COPY_LAST>{});
    return _data;
}
}  // namespace async_copy

void
async_copy_init(hsa_api_table_t* _orig, uint64_t _tbl_instance)
{
    if(_orig && _orig->amd_ext_)
    {
        async_copy::async_copy_save<ROCPROFILER_HSA_TABLE_ID_AmdExt>(
            _orig->amd_ext_, _tbl_instance, async_copy::async_copy_index_seq_t{});

        auto ctxs = context::get_registered_contexts(async_copy::context_filter);
        if(!ctxs.empty())
        {
            _orig->amd_ext_->hsa_amd_profiling_async_copy_enable_fn(true);
            async_copy::async_copy_wrap<ROCPROFILER_HSA_TABLE_ID_AmdExt>(
                _orig->amd_ext_, async_copy::async_copy_index_seq_t{});
        }
    }
}

void
async_copy_sync()
{
    if(!async_copy::get_active_signals()) return;

    async_copy::get_active_signals()->sync();
}

void
async_copy_fini()
{
    if(!async_copy::get_active_signals()) return;

    async_copy_sync();
    async_copy::get_active_signals()->destroy();
}
}  // namespace hsa
}  // namespace rocprofiler
