// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprofiler-sdk/cxx/constants.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

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
using external_corr_id_map_t = tracing::external_correlation_id_map_t;

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

struct traced_copy_data
{
    using timestamp_t     = rocprofiler_timestamp_t;
    using callback_data_t = rocprofiler_callback_tracing_memory_copy_data_t;
    using buffered_data_t = rocprofiler_buffer_tracing_memory_copy_record_t;

    rocprofiler_agent_id_t              dst_agent    = sdk::null_agent_id;
    rocprofiler_agent_id_t              src_agent    = sdk::null_agent_id;
    rocprofiler_address_t               dst_address  = {.value = 0};
    rocprofiler_address_t               src_address  = {.value = 0};
    rocprofiler_memory_copy_operation_t direction    = ROCPROFILER_MEMORY_COPY_NONE;
    uint64_t                            bytes_copied = 0;
    tracing::tracing_data               tracing_data = {};

    callback_data_t get_callback_data(timestamp_t _beg = 0, timestamp_t _end = 0) const;
    buffered_data_t get_buffered_record(const context_t*               _ctx,
                                        const context::correlation_id* _corr_id,
                                        timestamp_t                    _beg = 0,
                                        timestamp_t                    _end = 0) const;
};

using traced_copy_data_vec_t = common::container::small_vector<traced_copy_data, 1>;

struct async_copy_data
{
    hsa_signal_t                  orig_signal    = {};
    hsa_signal_t                  rocp_signal    = {};
    rocprofiler_thread_id_t       tid            = common::get_tid();
    uint64_t                      start_ts       = 0;
    context::correlation_id*      correlation_id = nullptr;
    traced_copy_data_vec_t        traced_copies  = {};

    auto get_lock() { return std::make_unique<std::unique_lock<std::mutex>>(m_mtx); }

private:
    std::mutex m_mtx = {};
};

/**
 * @brief Builds the callback payload for one logical copy described by a traced signal.
 */
traced_copy_data::callback_data_t
traced_copy_data::get_callback_data(timestamp_t _beg, timestamp_t _end) const
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

/**
 * @brief Builds the buffered tracing record for one logical copy.
 */
traced_copy_data::buffered_data_t
traced_copy_data::get_buffered_record(const context_t*               _ctx,
                                      const context::correlation_id* _corr_id,
                                      timestamp_t                    _beg,
                                      timestamp_t                    _end) const
{
    ROCP_FATAL_IF(direction == ROCPROFILER_MEMORY_COPY_NONE) << "direction has not been set";
    ROCP_FATAL_IF(_corr_id == nullptr) << "correlation id has not been set";

    auto _external_corr_id =
        (_ctx) ? tracing_data.external_correlation_ids.at(_ctx) : context::null_user_data;
    auto _async_corr_id = rocprofiler_async_correlation_id_t{_corr_id->internal, _external_corr_id};

    return common::init_public_api_struct(buffered_data_t{},
                                          ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                          direction,
                                          _async_corr_id,
                                          _corr_id->thread_idx,
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

struct copy_metadata
{
    rocprofiler_agent_id_t              dst_agent = sdk::null_agent_id;
    rocprofiler_agent_id_t              src_agent = sdk::null_agent_id;
    rocprofiler_memory_copy_operation_t direction = ROCPROFILER_MEMORY_COPY_NONE;
};

bool
async_copy_handler(hsa_signal_value_t, void* arg);

/**
 * @brief Resolves rocprofiler agent ids and copy direction from HSA agents.
 *
 * When both HSA agents are GPU type, the agent handles alone are not sufficient to determine
 * the true transfer direction. OpenMP RTL (libomptarget) pins host memory and registers it
 * under a GPU agent, causing HSA to report src_agent == dst_agent (both GPU) even for
 * HOST<->DEVICE transfers. In that case, hsa_amd_pointer_info::agentOwner is consulted to
 * recover the true direction (fix for ROCM-9863).
 */
copy_metadata
get_copy_metadata(hsa_agent_t      _hsa_dst_agent,
                  hsa_agent_t      _hsa_src_agent,
                  const void*      _dst_address,
                  const void*      _src_address,
                  std::string_view _name)
{
    auto        _copy_meta      = copy_metadata{};
    const auto* _rocp_dst_agent = agent::get_rocprofiler_agent(_hsa_dst_agent);
    const auto* _rocp_src_agent = agent::get_rocprofiler_agent(_hsa_src_agent);

    if(_rocp_dst_agent && _rocp_src_agent)
    {
        _copy_meta.src_agent = _rocp_src_agent->id;
        _copy_meta.dst_agent = _rocp_dst_agent->id;

        if(_rocp_src_agent->type == ROCPROFILER_AGENT_TYPE_CPU)
        {
            if(_rocp_dst_agent->type == ROCPROFILER_AGENT_TYPE_CPU)
                _copy_meta.direction = ROCPROFILER_MEMORY_COPY_HOST_TO_HOST;
            else if(_rocp_dst_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                _copy_meta.direction = ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE;
            else
                ROCP_CI_LOG(WARNING)
                    << _name << " had an unhandled destination type: " << _rocp_dst_agent->type;
        }
        else if(_rocp_src_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            if(_rocp_dst_agent->type == ROCPROFILER_AGENT_TYPE_CPU)
                _copy_meta.direction = ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST;
            else if(_rocp_dst_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            {
                // ROCM-9863: when both HSA agents are GPU type, inspect the actual ownership
                // of each pointer with hsa_amd_pointer_info to disambiguate HOST<->DEVICE
                // copies that OpenMP RTL routes through a GPU-registered host staging buffer.
                auto _src_info = hsa_amd_pointer_info_t{};
                auto _dst_info = hsa_amd_pointer_info_t{};
                _src_info.size = sizeof(hsa_amd_pointer_info_t);
                _dst_info.size = sizeof(hsa_amd_pointer_info_t);

                auto* _ptr_info_fn = get_amd_ext_table()->hsa_amd_pointer_info_fn;

                bool _src_query_ok = false;
                bool _dst_query_ok = false;
                if(_ptr_info_fn)
                {
                    _src_query_ok =
                        (_ptr_info_fn(_src_address, &_src_info, nullptr, nullptr, nullptr) ==
                         HSA_STATUS_SUCCESS);
                    _dst_query_ok =
                        (_ptr_info_fn(_dst_address, &_dst_info, nullptr, nullptr, nullptr) ==
                         HSA_STATUS_SUCCESS);
                }

                // A pointer is considered GPU device memory if its agentOwner matches
                // the GPU agent handle passed to the copy operation.
                const bool _src_is_device =
                    _src_query_ok && (_src_info.agentOwner == _hsa_src_agent);
                const bool _dst_is_device =
                    _dst_query_ok && (_dst_info.agentOwner == _hsa_dst_agent);

                if(!_src_is_device && _dst_is_device)
                    _copy_meta.direction = ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE;
                else if(_src_is_device && !_dst_is_device)
                    _copy_meta.direction = ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST;
                else
                    _copy_meta.direction = ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE;
            }
            else
                ROCP_CI_LOG(WARNING)
                    << _name << " had an unhandled destination type: " << _rocp_dst_agent->type;
        }
        else
        {
            ROCP_CI_LOG(WARNING) << _name
                                 << " had an unhandled source type: " << _rocp_src_agent->type;
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

    return _copy_meta;
}

/**
 * @brief Fills a logical copy record and captures the tracing contexts interested in it.
 */
bool
populate_traced_copy_data(traced_copy_data&     _traced_copy,
                          const copy_metadata&  _copy_meta,
                          uint64_t              _bytes_copied,
                          rocprofiler_address_t _dst_address,
                          rocprofiler_address_t _src_address)
{
    if(_copy_meta.direction == ROCPROFILER_MEMORY_COPY_NONE) return false;

    _traced_copy.dst_agent    = _copy_meta.dst_agent;
    _traced_copy.src_agent    = _copy_meta.src_agent;
    _traced_copy.direction    = _copy_meta.direction;
    _traced_copy.bytes_copied = _bytes_copied;
    _traced_copy.dst_address  = _dst_address;
    _traced_copy.src_address  = _src_address;

    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                               ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                               _traced_copy.direction,
                               _traced_copy.tracing_data);

    return !_traced_copy.tracing_data.empty();
}

/**
 * @brief Creates the rocprofiler-owned completion signal and installs its async handler.
 */
bool
create_async_copy_signal(async_copy_data* _data)
{
    constexpr auto     _completion_signal_val = hsa_signal_value_t{1};
    const uint32_t     _num_consumers         = 0;
    const hsa_agent_t* _consumers             = nullptr;
    hsa_status_t       _status                = HSA_STATUS_SUCCESS;

    _status = get_core_table()->hsa_signal_create_fn(
        _completion_signal_val, _num_consumers, _consumers, &_data->rocp_signal);

    if(_status != HSA_STATUS_SUCCESS)
    {
        ROCP_ERROR << "hsa_signal_create returned non-zero error code " << _status;
        return false;
    }

    _status = get_amd_ext_table()->hsa_amd_signal_async_handler_fn(_data->rocp_signal,
                                                                   HSA_SIGNAL_CONDITION_LT,
                                                                   _completion_signal_val,
                                                                   async_copy_handler,
                                                                   _data);

    if(_status != HSA_STATUS_SUCCESS)
    {
        ROCP_ERROR << "hsa_amd_signal_async_handler returned non-zero error code " << _status;

        ROCP_HSA_TABLE_CALL(ERROR, get_core_table()->hsa_signal_destroy_fn(_data->rocp_signal))
            << ":: failed to destroy signal after async handler failed";

        _data->rocp_signal = {};
        return false;
    }

    return true;
}

/**
 * @brief Tears down rocprofiler-owned signal state for an intercepted copy submission.
 */
void
destroy_async_copy_data(async_copy_data* _data)
{
    if(!_data) return;

    if(_data->rocp_signal.handle != 0 && get_core_table()->hsa_signal_destroy_fn)
    {
        ROCP_HSA_TABLE_CALL(ERROR, get_core_table()->hsa_signal_destroy_fn(_data->rocp_signal));
        _data->rocp_signal = {};
    }

    delete _data;
}

/**
 * @brief Emits enter callbacks and external correlation ids for all logical copies on a signal.
 */
void
initialize_async_copy_tracing(async_copy_data* _data)
{
    ROCP_FATAL_IF(_data == nullptr || _data->correlation_id == nullptr)
        << "async copy tracing requires valid correlation data";

    auto _thread_id = _data->correlation_id->thread_idx;

    for(auto& _copy : _data->traced_copies)
    {
        tracing::populate_external_correlation_ids(
            _copy.tracing_data.external_correlation_ids,
            _thread_id,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,
            _copy.direction,
            _data->correlation_id->internal);

        if(!_copy.tracing_data.callback_contexts.empty())
        {
            auto _tracer_data = _copy.get_callback_data();

            tracing::execute_phase_enter_callbacks(_copy.tracing_data.callback_contexts,
                                                   _thread_id,
                                                   _data->correlation_id->internal,
                                                   _copy.tracing_data.external_correlation_ids,
                                                   _data->correlation_id->ancestor,
                                                   ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                                                   _copy.direction,
                                                   _tracer_data);
        }
    }
}

bool
async_copy_handler(hsa_signal_value_t, void* arg)
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
        if(!_data->traced_copies.empty())
        {
            const auto& _copy = _data->traced_copies.front();
            ROCP_CI_LOG(ERROR) << fmt::format(
                "hsa_amd_profiling_get_async_copy_time for the {} copy operation from agent-{} "
                "to agent-{} returned status={} :: {}",
                std::string_view{hsa::async_copy::name_by_id(_copy.direction)},
                CHECK_NOTNULL(agent::get_agent(_copy.src_agent))->node_id,
                CHECK_NOTNULL(agent::get_agent(_copy.dst_agent))->node_id,
                static_cast<int>(copy_time_status),
                hsa::get_hsa_status_string(copy_time_status));
        }
    }

    if(_profile_time.status == HSA_STATUS_SUCCESS)
    {
        for(auto& _copy : _data->traced_copies)
        {
            if(!_copy.tracing_data.empty())
            {
                if(!_copy.tracing_data.callback_contexts.empty())
                {
                    auto _tracer_data =
                        _copy.get_callback_data(_profile_time.start, _profile_time.end);

                    tracing::execute_phase_exit_callbacks(
                        _copy.tracing_data.callback_contexts,
                        _copy.tracing_data.external_correlation_ids,
                        ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                        _copy.direction,
                        _tracer_data);
                }

                if(!_copy.tracing_data.buffered_contexts.empty())
                {
                    auto _record = _copy.get_buffered_record(
                        nullptr, _data->correlation_id, _profile_time.start, _profile_time.end);

                    tracing::execute_buffer_record_emplace(
                        _copy.tracing_data.buffered_contexts,
                        _data->tid,
                        _data->correlation_id->internal,
                        _copy.tracing_data.external_correlation_ids,
                        _data->correlation_id->ancestor,
                        ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                        _copy.direction,
                        _record);
                }
            }
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

        ROCP_TRACE << "Decrementing Signal: " << std::hex << _data->orig_signal.handle << std::dec;
        get_core_table()->hsa_signal_subtract_screlease_fn(_data->orig_signal, 1);
    }

    ROCP_HSA_TABLE_CALL(ERROR, get_core_table()->hsa_signal_destroy_fn(_data->rocp_signal));

    return false;
}

enum async_copy_id
{
    async_copy_id           = ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_async_copy,
    async_copy_on_engine_id = ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_async_copy_on_engine,
    async_copy_rect_id      = ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_async_copy_rect,
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0A
    async_batch_copy_id = ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_async_batch_copy,
#endif
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

#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0A
/**
 * @brief Appends one logical copy from a batch descriptor when tracing is enabled for it.
 */
bool
append_batch_traced_copy_data(traced_copy_data_vec_t& _traced_copies,
                              hsa_agent_t             _dst_agent,
                              hsa_agent_t             _src_agent,
                              const void*             _dst_address,
                              const void*             _src_address,
                              uint64_t                _bytes_copied,
                              std::string_view        _name)
{
    if(_bytes_copied == 0) return false;

    auto _traced_copy = traced_copy_data{};
    auto _copy_meta = get_copy_metadata(_dst_agent, _src_agent, _dst_address, _src_address, _name);

    if(!populate_traced_copy_data(_traced_copy,
                                  _copy_meta,
                                  _bytes_copied,
                                  compute_address(_dst_address),
                                  compute_address(_src_address)))
        return false;

    _traced_copies.emplace_back(std::move(_traced_copy));
    return true;
}

/**
 * @brief Expands a batch descriptor into the logical copy records rocprofiler can represent.
 */
void
populate_batch_traced_copy_data(traced_copy_data_vec_t&         _traced_copies,
                                const hsa_amd_memory_copy_op_t& _copy_op,
                                std::string_view                _name)
{
    switch(_copy_op.type)
    {
        case HSA_AMD_MEMORY_COPY_OP_LINEAR:
            if(_copy_op.num_entries > 0)
            {
                for(uint32_t i = 0; i < _copy_op.num_entries; ++i)
                {
                    append_batch_traced_copy_data(_traced_copies,
                                                  _copy_op.dst_agent_list[i],
                                                  _copy_op.src_agent,
                                                  _copy_op.dst_list[i],
                                                  _copy_op.src_list[i],
                                                  _copy_op.size_list[i],
                                                  _name);
                }
            }
            else
            {
                append_batch_traced_copy_data(_traced_copies,
                                              _copy_op.dst_agent,
                                              _copy_op.src_agent,
                                              _copy_op.dst,
                                              _copy_op.src,
                                              _copy_op.size,
                                              _name);
            }
            break;
        case HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST:
            for(uint32_t i = 0; i < _copy_op.num_entries; ++i)
            {
                append_batch_traced_copy_data(_traced_copies,
                                              _copy_op.dst_agent_list[i],
                                              _copy_op.src_agent,
                                              _copy_op.dst_list[i],
                                              _copy_op.src,
                                              _copy_op.size,
                                              _name);
            }
            break;
        case HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP:
            if(_copy_op.num_entries > 0)
            {
                for(uint32_t i = 0; i < _copy_op.num_entries; ++i)
                {
                    append_batch_traced_copy_data(_traced_copies,
                                                  _copy_op.dst_agent_list[i],
                                                  _copy_op.src_agent,
                                                  _copy_op.dst_list[i],
                                                  _copy_op.src_list[i],
                                                  _copy_op.size_list[i],
                                                  _name);
                    append_batch_traced_copy_data(_traced_copies,
                                                  _copy_op.src_agent,
                                                  _copy_op.dst_agent_list[i],
                                                  _copy_op.src_list[i],
                                                  _copy_op.dst_list[i],
                                                  _copy_op.size_list[i],
                                                  _name);
                }
            }
            else
            {
                append_batch_traced_copy_data(_traced_copies,
                                              _copy_op.dst_agent,
                                              _copy_op.src_agent,
                                              _copy_op.dst,
                                              _copy_op.src,
                                              _copy_op.src_size,
                                              _name);
                append_batch_traced_copy_data(_traced_copies,
                                              _copy_op.src_agent,
                                              _copy_op.dst_agent,
                                              _copy_op.src,
                                              _copy_op.dst,
                                              _copy_op.dst_size,
                                              _name);
            }
            break;
        case HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRC:
        case HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_DST:
        case HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRCDST:
            append_batch_traced_copy_data(_traced_copies,
                                          _copy_op.dst_agent,
                                          _copy_op.src_agent,
                                          _copy_op.dst,
                                          _copy_op.src,
                                          _copy_op.size,
                                          _name);
            break;
        default: break;
    }
}

/**
 * @brief Intercepts batch copy submission and replaces traced op signals with rocprofiler signals.
 */
template <size_t TableIdx, size_t OpIdx>
hsa_status_t
async_batch_copy_impl(const hsa_amd_memory_copy_op_t* copy_ops,
                      uint32_t                        num_copy_ops,
                      uint32_t                        num_dep_signals,
                      const hsa_signal_t*             dep_signals)
{
    using meta_type        = hsa_api_meta<TableIdx, OpIdx>;
    using intercept_data_t = std::pair<uint32_t, async_copy_data*>;
    using intercept_lock_t = std::unique_ptr<std::unique_lock<std::mutex>>;

    auto& _dispatch = get_next_dispatch<TableIdx, OpIdx>();

    if(copy_ops == nullptr || num_copy_ops == 0)
        return _dispatch(copy_ops, num_copy_ops, num_dep_signals, dep_signals);

    auto _intercept_data  = std::vector<intercept_data_t>{};
    auto _intercept_locks = std::vector<intercept_lock_t>{};

    _intercept_data.reserve(num_copy_ops);
    _intercept_locks.reserve(num_copy_ops);

    for(uint32_t i = 0; i < num_copy_ops; ++i)
    {
        auto _traced_copies = traced_copy_data_vec_t{};
        populate_batch_traced_copy_data(_traced_copies, copy_ops[i], meta_type::name);

        if(_traced_copies.empty()) continue;

        auto* _data          = new async_copy_data{};
        _data->traced_copies = std::move(_traced_copies);

        if(!create_async_copy_signal(_data))
        {
            destroy_async_copy_data(_data);
            continue;
        }

        _intercept_data.emplace_back(i, _data);
    }

    if(_intercept_data.empty())
        return _dispatch(copy_ops, num_copy_ops, num_dep_signals, dep_signals);

    auto _wrapped_copy_ops =
        std::vector<hsa_amd_memory_copy_op_t>{copy_ops, copy_ops + num_copy_ops};

    auto _cleanup = [&_intercept_data](bool _decrement_active_signals,
                                       bool _decrement_corr_ref_count) {
        for(auto& [_idx, _data] : _intercept_data)
        {
            if(_decrement_active_signals && get_active_signals())
                get_active_signals()->fetch_sub(1);

            if(_decrement_corr_ref_count && _data && _data->correlation_id)
                _data->correlation_id->sub_ref_count();

            destroy_async_copy_data(_data);
            _data = nullptr;
        }
    };

    auto*                    _corr_id     = context::get_latest_correlation_id();
    context::correlation_id* _corr_id_pop = nullptr;

    if(!_corr_id)
    {
        constexpr auto ref_count = 1;
        _corr_id                 = context::correlation_tracing_service::construct(ref_count);
        _corr_id_pop             = _corr_id;
    }

    if(!_corr_id)
    {
        _cleanup(false, false);
        return _dispatch(copy_ops, num_copy_ops, num_dep_signals, dep_signals);
    }

    for(auto& [_idx, _data] : _intercept_data)
    {
        _intercept_locks.emplace_back(_data->get_lock());
        _data->correlation_id = _corr_id;
        _data->correlation_id->add_ref_count();

        auto& _completion_signal = _wrapped_copy_ops.at(_idx).completion_signal;
        auto  _original_value = get_core_table()->hsa_signal_load_scacquire_fn(_completion_signal);

        _data->orig_signal = _completion_signal;
        _completion_signal = _data->rocp_signal;

        ROCP_TRACE << "Memcpy Batch Original Signal " << std::hex << _data->orig_signal.handle
                   << std::dec << ": " << _original_value << " | Replacement Signal: " << std::hex
                   << _completion_signal.handle << std::dec << ": 1";

        CHECK_NOTNULL(get_active_signals())->fetch_add(1);
    }

    auto _start_ts = common::timestamp_ns();
    for(auto& [_idx, _data] : _intercept_data)
    {
        _data->start_ts = _start_ts;
        initialize_async_copy_tracing(_data);
    }

    auto _status = _dispatch(_wrapped_copy_ops.data(), num_copy_ops, num_dep_signals, dep_signals);

    if(_corr_id_pop)
    {
        context::pop_latest_correlation_id(_corr_id_pop);
        _corr_id_pop->sub_ref_count();
    }

    if(_status != HSA_STATUS_SUCCESS)
    {
        _intercept_locks.clear();
        _cleanup(true, true);
        return _status;
    }

    _intercept_locks.clear();

    return _status;
}
#endif

template <size_t TableIdx, size_t OpIdx, typename... Args>
hsa_status_t
async_copy_impl(Args... args)
{
    using meta_type = hsa_api_meta<TableIdx, OpIdx>;

    constexpr auto N             = sizeof...(Args);
    constexpr auto copy_size_idx = arg_indices<OpIdx>::copy_size_idx;
    constexpr auto dst_addr_idx  = arg_indices<OpIdx>::dst_address_idx;
    constexpr auto src_addr_idx  = arg_indices<OpIdx>::src_address_idx;
    constexpr auto dst_agent_idx = arg_indices<OpIdx>::dst_agent_idx;
    constexpr auto src_agent_idx = arg_indices<OpIdx>::src_agent_idx;

    auto&& _tied_args = std::tie(args...);

    async_copy_data* _data = nullptr;

    {
        auto _traced_copy = traced_copy_data{};
        auto _copy_meta   = get_copy_metadata(std::get<dst_agent_idx>(_tied_args),
                                            std::get<src_agent_idx>(_tied_args),
                                            compute_address(std::get<dst_addr_idx>(_tied_args)).ptr,
                                            compute_address(std::get<src_addr_idx>(_tied_args)).ptr,
                                            meta_type::name);

        if(!populate_traced_copy_data(_traced_copy,
                                      _copy_meta,
                                      compute_copy_bytes(std::get<copy_size_idx>(_tied_args)),
                                      compute_address(std::get<dst_addr_idx>(_tied_args)),
                                      compute_address(std::get<src_addr_idx>(_tied_args))))
        {
            return invoke(get_next_dispatch<TableIdx, OpIdx>(),
                          std::move(_tied_args),
                          std::make_index_sequence<N>{});
        }

        _data = new async_copy_data{};
        _data->traced_copies.emplace_back(std::move(_traced_copy));
    }

    auto _lk = _data->get_lock();

    constexpr auto completion_signal_idx = arg_indices<OpIdx>::completion_signal_idx;
    auto&          _completion_signal    = std::get<completion_signal_idx>(_tied_args);
    auto original_value = get_core_table()->hsa_signal_load_scacquire_fn(_completion_signal);

    if(!create_async_copy_signal(_data))
    {
        _lk.reset();
        destroy_async_copy_data(_data);
        return invoke(get_next_dispatch<TableIdx, OpIdx>(),
                      std::move(_tied_args),
                      std::make_index_sequence<N>{});
    }

    _data->correlation_id                 = context::get_latest_correlation_id();
    context::correlation_id* _corr_id_pop = nullptr;

    if(!_data->correlation_id)
    {
        constexpr auto ref_count = 1;
        _data->correlation_id    = context::correlation_tracing_service::construct(ref_count);
        _corr_id_pop             = _data->correlation_id;
    }

    if(!_data->correlation_id)
    {
        // During finalization - cleanup and execute without tracing
        _lk.reset();
        destroy_async_copy_data(_data);
        return invoke(get_next_dispatch<TableIdx, OpIdx>(),
                      std::move(_tied_args),
                      std::make_index_sequence<N>{});
    }

    // increase the reference count to denote that this correlation id is being used in a kernel
    _data->correlation_id->add_ref_count();

    _data->orig_signal = _completion_signal;
    _completion_signal = _data->rocp_signal;

    ROCP_TRACE << "Memcpy Original Signal " << std::hex << _data->orig_signal.handle << std::dec
               << ": " << original_value << " | Replacement Signal: " << std::hex
               << _completion_signal.handle << std::dec << ": 1";

    CHECK_NOTNULL(get_active_signals())->fetch_add(1);

    _data->start_ts = common::timestamp_ns();
    initialize_async_copy_tracing(_data);

    auto _status = invoke(
        get_next_dispatch<TableIdx, OpIdx>(), std::move(_tied_args), std::make_index_sequence<N>{});

    if(_corr_id_pop)
    {
        context::pop_latest_correlation_id(_corr_id_pop);
        _corr_id_pop->sub_ref_count();
    }

    if(_status != HSA_STATUS_SUCCESS)
    {
        if(get_active_signals()) get_active_signals()->fetch_sub(1);
        if(_data->correlation_id) _data->correlation_id->sub_ref_count();
        _lk.reset();
        destroy_async_copy_data(_data);
        return _status;
    }

    return _status;
}

template <size_t TableIdx, size_t OpIdx, typename RetT, typename... Args>
auto get_async_copy_impl(RetT (*)(Args...))
{
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0A
    if constexpr(OpIdx == async_batch_copy_id)
        return &async_batch_copy_impl<TableIdx, OpIdx>;
    else
#endif
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

    if(_tbl_instance == 0 && _orig_func != _copy_func)
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

using async_copy_index_seq_t = std::index_sequence<async_copy_id,
                                                   async_copy_on_engine_id,
                                                   async_copy_rect_id
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0A
                                                   ,
                                                   async_batch_copy_id
#endif
                                                   >;
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
