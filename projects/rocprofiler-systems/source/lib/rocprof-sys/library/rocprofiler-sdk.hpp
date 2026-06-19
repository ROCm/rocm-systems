// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/rocprofiler_sdk_backend.hpp"
#include "core/rocprofiler-sdk.hpp"
#include "core/timemory.hpp"
#include "library/rocprofiler-sdk/counters.hpp"
#include "library/rocprofiler-sdk/fwd.hpp"
#include "library/rocprofiler-sdk/kfd_events.hpp"
#include "library/rocprofiler-sdk/rccl.hpp"
#include "library/rocprofiler-sdk/trace_control.hpp"
#include "rocprofiler-sdk/roctx_client.hpp"

#include "api.hpp"
#include "binary/analysis.hpp"
#include "common/env_vars.hpp"
#include "common/synchronized.hpp"
#include "core/common.hpp"
#include "core/common_types.hpp"
#include "core/config.hpp"
#include "core/containers/stable_vector.hpp"
#include "core/demangler.hpp"
#include "core/gpu.hpp"
#include "core/output_file_registry.hpp"
#include "core/perfetto.hpp"
#include "core/perfetto_fwd.hpp"
#include "core/state.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/sampler.hpp"
#include "library/process_sampler.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"

#include "logger/debug.hpp"

#include <timemory/components/timing/wall_clock.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/unwind/processed_entry.hpp>
#include <timemory/variadic/lightweight_tuple.hpp>

// All rocprofiler-sdk headers come transitively through
// backends/rocprofiler_sdk/rocprofiler_sdk_backend.hpp (included via
// core/rocprofiler-sdk.hpp).

#include <nlohmann/json.hpp>
#include <spdlog/fmt/ranges.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

using hardware_counter_info = ::tim::hardware_counters::info;

template <typename Backend>
class library_sdk
{
public:
    // ─── Public API ──────────────────────────────────────────────────────────

    static void setup();
    static void shutdown();
    static void config();
    static void post_process();
    static void sample();
    static void start();
    static void stop();
    static void pause();
    static void resume();

    static std::shared_ptr<control::trace_control> get_trace_controller();
    static void                                    reset_sdk_session_guards();
    static std::vector<hardware_counter_info>      get_rocm_events_info();

    // ─── SDK registration entry points (called from extern "C") ──────────────

    static int  tool_init(typename Backend::client_finalize_t fini_func, void* user_data);
    static void tool_fini(void* callback_data);
    static bool sdk_tool_configure(std::uint32_t version, const char* runtime_version,
                                   typename Backend::client_id_t* id);

#if ROCPROFILER_VERSION >= 10200
    static int  tool_attach_init(typename Backend::client_detach_t detach_func,
                                 typename Backend::context_id*     context_ids,
                                 std::uint64_t context_ids_length, void* tool_attach_data);
    static void tool_attach_fini(void* tool_data_ptr);
#endif

    // ─── Static data members (formerly anon-namespace globals) ───────────────

    static client_data<Backend>*           tool_data;
    static std::shared_ptr<roctx_client<>> g_roctx_client;
    static std::atomic<bool>               tool_fini_done;
    static std::atomic<bool>               tool_init_done;
    static std::atomic<bool>               sdk_configured;

private:
    // ─── Nested helper types ──────────────────────────────────────────────────

    struct kernel_rename_and_stream_data
    {
        std::uint64_t               region_id = 0;
        typename Backend::stream_id stream_id = { 0 };
    };

    struct scope_destructor
    {
        template <typename FuncT, typename InitT = void (*)()>
        scope_destructor(FuncT&& _fini, InitT&& _init = []() {});

        ~scope_destructor() { m_functor(); }

        scope_destructor(const scope_destructor&)            = delete;
        scope_destructor& operator=(const scope_destructor&) = delete;
        scope_destructor(scope_destructor&& rhs) noexcept;
        scope_destructor& operator=(scope_destructor&& rhs) noexcept;

    private:
        std::function<void()> m_functor = []() {};
    };

    using kernel_rename_stack_t    = std::stack<std::uint64_t>;
    using kernel_dispatch_bundle_t = tim::lightweight_tuple<tim::component::wall_clock>;
    using tool_agent_vec_t         = std::vector<tool_agent>;
    using counter_storage_map_t =
        std::unordered_map<typename Backend::counter_id, counter_storage>;
    using agent_counter_storage_map_t =
        std::unordered_map<typename Backend::agent_id, counter_storage_map_t>;

#if ROCPROFILER_VERSION >= 600
    struct rocprofsys_ompt_data_storage_t
    {
        typename Backend::callback_tracing_record record;
        typename Backend::timestamp_t             _beg_ts;
        function_args_t                           args;
    };
#endif

    // ─── Thread-local state accessors ────────────────────────────────────────

    static kernel_rename_stack_t*& get_thread_dispatch_rename();
    static auto&                   get_stream_stack();

    // ─── OMPT storage (thread-local) ─────────────────────────────────────────

#if ROCPROFILER_VERSION >= 600
    static auto& get_ompt_standard_cb_storage();
    static auto& get_ompt_parallel_cb_storage();
#endif

    // ─── Counter storage ──────────────────────────────────────────────────────

    static auto&                         get_counter_dispatch_data();
    static auto&                         get_counter_dispatch_records();
    static agent_counter_storage_map_t*& get_counter_storage();
    static auto&                         get_kernel_dispatch_timestamps();

    // ─── Context helpers ─────────────────────────────────────────────────────

    static bool is_initialized(typename Backend::context_id ctx);
    static bool is_active(typename Backend::context_id ctx);
    static bool is_valid(typename Backend::context_id ctx);

    static void start_context(typename Backend::context_id ctx);
    static void stop_context(typename Backend::context_id ctx);
    static void start_context(const client_data<Backend>::context_id_vec_t& ctxs);
    static void stop_context(const client_data<Backend>::context_id_vec_t& ctxs);
    static void flush();

    // ─── roctx / counter / finalization helpers ───────────────────────────────

    static std::shared_ptr<roctx_client<>> get_roctx_client();
    static void                            flush_counter_storage_outputs();
    static void flush_counter_tracks_to_zero(typename Backend::timestamp_t ts);
    static void finalize_sdk_common();

    // ─── Thread callbacks ─────────────────────────────────────────────────────

    static void thread_precreate(typename Backend::runtime_library_t lib, void* data);
    static void thread_postcreate(typename Backend::runtime_library_t lib, void* data);

    // ─── Stream helpers ───────────────────────────────────────────────────────

    static void stream_id_push(typename Backend::stream_id stream_id);
    static typename Backend::stream_id stream_id_top();
    static void                        stream_id_pop();

    template <typename Tp>
    static typename Backend::stream_id get_stream_id(Tp* record);

    // ─── Utility helpers ──────────────────────────────────────────────────────

    template <typename Tp, typename... Args>
    static Tp* as_pointer(Args&&... args);

    template <typename... Tp>
    static void consume_args(Tp&&...);

    static auto get_backtrace(
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data);

    template <typename CorrelationIdType>
    static std::uint64_t get_parent_stack_id(const CorrelationIdType& correlation_id);

    static const kernel_symbol_data_t<Backend>* get_kernel_symbol_info(
        std::uint64_t kernel_id);
    static const typename Backend::code_object_load_data* get_code_object_info(
        std::uint64_t code_object_id);

    static size_t get_mem_copy_dst_address(
        const typename Backend::memory_copy_record& record);
    static size_t get_mem_copy_src_address(
        const typename Backend::memory_copy_record& record);
#if ROCPROFILER_VERSION >= 600
    static size_t get_mem_alloc_address(
        const typename Backend::memory_alloc_record& record);
#endif
    static std::uint64_t get_scratch_mem_alloc_size(
        const typename Backend::scratch_memory_record& record);

    // ─── Argument iteration callbacks ────────────────────────────────────────

    static int save_args(typename Backend::callback_tracing_kind kind,
                         std::int32_t operation, std::uint32_t arg_number,
                         const void* const arg_value_addr,
                         std::int32_t arg_indirection_count, const char* arg_type,
                         const char* arg_name, const char* arg_value_str,
                         std::int32_t arg_dereference_count, void* data);

    static int iterate_args_callback(typename Backend::callback_tracing_kind kind,
                                     std::int32_t operation, std::uint32_t arg_number,
                                     const void* const arg_value_addr,
                                     std::int32_t      arg_indirection_count,
                                     const char* arg_type, const char* arg_name,
                                     const char*  arg_value_str,
                                     std::int32_t arg_dereference_count, void* data);

    // ─── Cache helpers ────────────────────────────────────────────────────────

    template <typename Category>
    static void cache_category();

    static void cache_add_thread_info(std::uint64_t tid);
    static void cache_add_track(const char* track_name, std::uint64_t tid);

    static void cache_region(const typename Backend::callback_tracing_record* record,
                             typename Backend::timestamp_t start_timestamp,
                             typename Backend::timestamp_t end_timestamp,
                             const std::string& call_stack, const std::string& args_str,
                             const std::string& category, std::string_view name = {});

    static void cache_kernel_dispatch(typename Backend::kernel_dispatch_record* record,
                                      std::uint64_t stream_handle);
    static void cache_scratch_memory(typename Backend::scratch_memory_record* record,
                                     std::uint64_t stream_handle);
    static void cache_memory_copy(typename Backend::memory_copy_record* record,
                                  std::uint64_t                         stream_handle);
#if ROCPROFILER_VERSION >= 600
    static void cache_memory_allocation(typename Backend::memory_alloc_record* record,
                                        std::uint64_t stream_handle);
#endif

    // ─── Profiling configuration ─────────────────────────────────────────────

    static std::vector<typename Backend::counter_id> create_agent_profile(
        typename Backend::agent_id agent_id, const std::vector<std::string>& counters,
        client_data<Backend>* data = nullptr);

    static int set_kernel_rename_and_stream_correlation_id(
        typename Backend::thread_id thr_id, typename Backend::context_id ctx_id,
        typename Backend::external_correlation_request_kind kind,
        typename Backend::tracing_operation op, std::uint64_t internal_corr_id,
        typename Backend::user_data_t* external_corr_id, void* user_data);

    // ─── Tracing callbacks ────────────────────────────────────────────────────

    template <typename CategoryT>
    static void tool_tracing_callback_start(
        CategoryT, typename Backend::callback_tracing_record record,
        typename Backend::user_data_t* user_data, typename Backend::timestamp_t ts);

    template <typename CategoryT>
    static void tool_tracing_callback_stop(
        CategoryT, typename Backend::callback_tracing_record record,
        typename Backend::user_data_t* user_data, typename Backend::timestamp_t ts,
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data);

    static void tool_code_object_callback(
        typename Backend::callback_tracing_record record,
        typename Backend::user_data_t* user_data, void* callback_data);

    static void tool_tracing_callback(typename Backend::callback_tracing_record record,
                                      typename Backend::user_data_t*            user_data,
                                      void* callback_data);

    static void tool_tracing_buffered(typename Backend::context_id        context,
                                      typename Backend::buffer_id         buffer_id,
                                      typename Backend::record_header_t** headers,
                                      size_t num_headers, void* user_data,
                                      std::uint64_t drop_count);

    static void counter_record_callback(
        typename Backend::dispatch_counting_data dispatch_data,
        typename Backend::counter_record* record_data, size_t record_count,
        typename Backend::user_data_t user_data, void* callback_data_arg);

    static void dispatch_counting_service_callback(
        typename Backend::dispatch_counting_data dispatch_data,
        typename Backend::counter_config_id*     config,
        typename Backend::user_data_t* user_data, void* callback_data_arg);

#if ROCPROFILER_VERSION >= 600
    // ─── OMPT helpers ────────────────────────────────────────────────────────

    static std::string_view ompt_get_unified_name(
        const typename Backend::callback_tracing_record& record);

    template <typename ArgsT>
    static void ompt_iterate_operation_args(
        const typename Backend::callback_tracing_record& record, ArgsT& args);

    static void ompt_cache_instant_event(
        typename Backend::callback_tracing_record                 record,
        typename Backend::timestamp_t                             instant_ts,
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data);

    static void ompt_cache_orphan_event(
        const rocprofsys_ompt_data_storage_t&                     stored_data,
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data);

    static void ompt_push_standard_callback(
        const typename Backend::callback_tracing_record& record,
        const typename Backend::timestamp_t&             beg_ts);

    static void ompt_pop_standard_callback(
        const typename Backend::callback_tracing_record&          record,
        const typename Backend::timestamp_t&                      end_ts,
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data);

    static void ompt_push_parallel_callback(
        const typename Backend::callback_tracing_record& record,
        const typename Backend::timestamp_t&             beg_ts);

    static void ompt_pop_parallel_callback(
        const typename Backend::callback_tracing_record&          record,
        const typename Backend::timestamp_t&                      end_ts,
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data);

    static void ompt_tracing_callback_start(
        typename Backend::callback_tracing_record record,
        typename Backend::user_data_t* user_data, typename Backend::timestamp_t ts);

    static void ompt_tracing_callback_stop(
        typename Backend::callback_tracing_record record,
        typename Backend::user_data_t* user_data, typename Backend::timestamp_t ts,
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data);

    static void ompt_finalize_orphan_events();
#endif

#if ROCPROFILER_VERSION >= 700
    static void tool_hip_stream_callback(typename Backend::callback_tracing_record record,
                                         typename Backend::user_data_t* user_data,
                                         void*                          data);
#endif
};

// ─── Production type alias ───────────────────────────────────────────────────

using library_sdk_t = library_sdk<::rocprofsys::rocprofiler_sdk::backend>;

// ─── Backward-compat free functions ──────────────────────────────────────────

inline void
setup()
{
    library_sdk_t::setup();
}
inline void
shutdown()
{
    library_sdk_t::shutdown();
}
inline void
config()
{
    library_sdk_t::config();
}
inline void
post_process()
{
    library_sdk_t::post_process();
}
inline void
sample()
{
    library_sdk_t::sample();
}
inline void
start()
{
    library_sdk_t::start();
}
inline void
stop()
{
    library_sdk_t::stop();
}
inline void
pause()
{
    library_sdk_t::pause();
}
inline void
resume()
{
    library_sdk_t::resume();
}
inline std::shared_ptr<control::trace_control>
get_trace_controller()
{
    return library_sdk_t::get_trace_controller();
}
inline void
reset_sdk_session_guards()
{
    library_sdk_t::reset_sdk_session_guards();
}
inline std::vector<hardware_counter_info>
get_rocm_events_info()
{
    return library_sdk_t::get_rocm_events_info();
}

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys

// ─── Template implementations ────────────────────────────────────────────────

namespace rocprofsys::rocprofiler_sdk
{

// ─── Static member definitions ───────────────────────────────────────────────

template <typename Backend>
client_data<Backend>* library_sdk<Backend>::tool_data = new client_data<Backend>{};

template <typename Backend>
std::shared_ptr<roctx_client<>> library_sdk<Backend>::g_roctx_client = {};

template <typename Backend>
std::atomic<bool> library_sdk<Backend>::tool_fini_done{ false };

template <typename Backend>
std::atomic<bool> library_sdk<Backend>::tool_init_done{ false };

template <typename Backend>
std::atomic<bool> library_sdk<Backend>::sdk_configured{ false };

// ─── scope_destructor ────────────────────────────────────────────────────────

template <typename Backend>
template <typename FuncT, typename InitT>
library_sdk<Backend>::scope_destructor::scope_destructor(FuncT&& _fini, InitT&& _init)
: m_functor{ std::forward<FuncT>(_fini) }
{
    _init();
}

template <typename Backend>
library_sdk<Backend>::scope_destructor::scope_destructor(scope_destructor&& rhs) noexcept
: m_functor{ std::move(rhs.m_functor) }
{
    rhs.m_functor = []() {};
}

template <typename Backend>
typename library_sdk<Backend>::scope_destructor&
library_sdk<Backend>::scope_destructor::operator=(scope_destructor&& rhs) noexcept
{
    if(this != &rhs)
    {
        m_functor     = std::move(rhs.m_functor);
        rhs.m_functor = []() {};
    }
    return *this;
}

// ─── Utility helpers ─────────────────────────────────────────────────────────

template <typename Backend>
template <typename Tp, typename... Args>
Tp*
library_sdk<Backend>::as_pointer(Args&&... args)
{
    return new Tp{ std::forward<Args>(args)... };
}

template <typename Backend>
template <typename... Tp>
void
library_sdk<Backend>::consume_args(Tp&&...)
{}

// ─── Thread-local state ───────────────────────────────────────────────────────

template <typename Backend>
typename library_sdk<Backend>::kernel_rename_stack_t*&
library_sdk<Backend>::get_thread_dispatch_rename()
{
    static thread_local auto* _v    = as_pointer<kernel_rename_stack_t>();
    static thread_local auto  _dtor = scope_destructor{ []() {
        delete _v;
        _v                          = nullptr;
    } };
    return _v;
}

template <typename Backend>
auto&
library_sdk<Backend>::get_stream_stack()
{
    static thread_local std::vector<typename Backend::stream_id> _v{
        typename Backend::stream_id{ 0 }
    };
    return _v;
}

// ─── Stream helpers ───────────────────────────────────────────────────────────

template <typename Backend>
void
library_sdk<Backend>::stream_id_push(typename Backend::stream_id stream_id)
{
    get_stream_stack().emplace_back(stream_id);
}

template <typename Backend>
typename Backend::stream_id
library_sdk<Backend>::stream_id_top()
{
    return get_stream_stack().back();
}

template <typename Backend>
void
library_sdk<Backend>::stream_id_pop()
{
    get_stream_stack().pop_back();
}

template <typename Backend>
template <typename Tp>
typename Backend::stream_id
library_sdk<Backend>::get_stream_id(Tp* record)
{
    auto _stream_id = typename Backend::stream_id{ 0 };
    if(record->correlation_id.external.ptr != nullptr)
    {
        auto* _ecid_data = static_cast<kernel_rename_and_stream_data*>(
            record->correlation_id.external.ptr);
        _stream_id                            = _ecid_data->stream_id;
        auto _region_id                       = _ecid_data->region_id;
        record->correlation_id.external.value = _region_id;
        delete _ecid_data;
        record->correlation_id.external.ptr = nullptr;
    }
    return _stream_id;
}

// ─── Parent stack ID ─────────────────────────────────────────────────────────

template <typename Backend>
template <typename CorrelationIdType>
std::uint64_t
library_sdk<Backend>::get_parent_stack_id(
    [[maybe_unused]] const CorrelationIdType& correlation_id)
{
#if(ROCPROFILER_VERSION >= 700)
    if constexpr(std::is_same_v<typename Backend::correlation_id_t, CorrelationIdType>)
        return correlation_id.ancestor;
    else
        return 0;
#else
    return 0;
#endif
}

// ─── thread_precreate / thread_postcreate ────────────────────────────────────

template <typename Backend>
void
library_sdk<Backend>::thread_precreate(typename Backend::runtime_library_t /*lib*/,
                                       void* /*data*/)
{
    push_thread_state(ThreadState::Internal);
}

template <typename Backend>
void
library_sdk<Backend>::thread_postcreate(typename Backend::runtime_library_t /*lib*/,
                                        void* /*data*/)
{
    pop_thread_state();
}

// ─── Context helpers ─────────────────────────────────────────────────────────

template <typename Backend>
bool
library_sdk<Backend>::is_initialized(typename Backend::context_id ctx)
{
    return (ctx.handle > 0);
}

template <typename Backend>
bool
library_sdk<Backend>::is_active(typename Backend::context_id ctx)
{
    int  status = 0;
    auto errc   = Backend::context_is_active(ctx, &status);
    return (errc == ROCPROFILER_STATUS_SUCCESS && status > 0);
}

template <typename Backend>
bool
library_sdk<Backend>::is_valid(typename Backend::context_id ctx)
{
    int  status = 0;
    auto errc   = Backend::context_is_valid(ctx, &status);
    return (errc == ROCPROFILER_STATUS_SUCCESS && status > 0);
}

template <typename Backend>
void
library_sdk<Backend>::start_context(typename Backend::context_id ctx)
{
    if(is_initialized(ctx) && !is_active(ctx))
    {
        ROCPROFILER_CALL(Backend::start_context(ctx));
    }
}

template <typename Backend>
void
library_sdk<Backend>::stop_context(typename Backend::context_id ctx)
{
    if(is_initialized(ctx) && is_active(ctx))
    {
        ROCPROFILER_CALL(Backend::stop_context(ctx));
    }
}

template <typename Backend>
void
library_sdk<Backend>::start_context(const client_data<Backend>::context_id_vec_t& ctxs)
{
    std::for_each(std::begin(ctxs), std::end(ctxs),
                  [](const auto& ctx) { start_context(ctx); });
}

template <typename Backend>
void
library_sdk<Backend>::stop_context(const client_data<Backend>::context_id_vec_t& ctxs)
{
    std::for_each(std::begin(ctxs), std::end(ctxs),
                  [](const auto& ctx) { stop_context(ctx); });
}

template <typename Backend>
void
library_sdk<Backend>::flush()
{
    if(!tool_data) return;

    for(const auto& itr : tool_data->get_buffers())
    {
        if(itr.handle > 0)
        {
            auto status = Backend::flush_buffer(itr);
            if(status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
            {
                ROCPROFILER_CALL(status);
            }
        }
    }
}

// ─── roctx helper ────────────────────────────────────────────────────────────

template <typename Backend>
std::shared_ptr<roctx_client<>>
library_sdk<Backend>::get_roctx_client()
{
    if(!g_roctx_client)
    {
        const auto _domains = tim::delimit(
            config::get_setting_value<std::string>(std::string{ env_vars::ROCM_DOMAINS })
                .value_or(std::string{}),
            " ,;:\t\n");
        const auto has_marker_domain =
            (std::find(_domains.begin(), _domains.end(), "marker_api") !=
                 _domains.end() ||
             std::find(_domains.begin(), _domains.end(), "roctx") != _domains.end());
        const auto roctx_traced_regions = config::get_trace_region();
        const auto has_trace_regions    = !roctx_traced_regions.empty();

        if(!has_marker_domain && !has_trace_regions) return nullptr;

        const auto roctx_config = roctx_client_config{
            has_marker_domain,          config::get_use_perfetto(),
            config::get_use_timemory(), config::get_perfetto_annotations(),
            roctx_traced_regions,
        };
        g_roctx_client = std::make_shared<roctx_client<>>(roctx_config);
    }
    return g_roctx_client;
}

// ─── Public API ──────────────────────────────────────────────────────────────

template <typename Backend>
void
library_sdk<Backend>::setup()
{}

template <typename Backend>
void
library_sdk<Backend>::shutdown()
{
    auto roctx_client = get_roctx_client();
    if(roctx_client) roctx_client->get_controller()->shutdown();

    if(tool_data && tool_data->client_id && tool_data->client_fini)
        tool_data->client_fini(*tool_data->client_id);
}

template <typename Backend>
void
library_sdk<Backend>::config()
{}

template <typename Backend>
void
library_sdk<Backend>::post_process()
{}

template <typename Backend>
void
library_sdk<Backend>::sample()
{}

template <typename Backend>
void
library_sdk<Backend>::start()
{
    if(!tool_data) return;
    start_context(tool_data->get_all_contexts());
}

template <typename Backend>
void
library_sdk<Backend>::stop()
{
    if(!tool_data) return;
    stop_context(tool_data->get_all_contexts());
}

template <typename Backend>
void
library_sdk<Backend>::resume()
{
    flush_counter_tracks_to_zero(0);
    if(!tool_data) return;
    start_context(tool_data->get_main_contexts());
}

template <typename Backend>
void
library_sdk<Backend>::pause()
{
    if(!tool_data) return;
    stop_context(tool_data->get_main_contexts());
    flush_counter_tracks_to_zero(0);
}

template <typename Backend>
std::shared_ptr<control::trace_control>
library_sdk<Backend>::get_trace_controller()
{
    const auto roctx_client = get_roctx_client();
    if(!roctx_client) return nullptr;
    return roctx_client->get_controller();
}

template <typename Backend>
void
library_sdk<Backend>::reset_sdk_session_guards()
{
    tool_fini_done.store(false);
    tool_init_done.store(false);
}

template <typename Backend>
std::vector<hardware_counter_info>
library_sdk<Backend>::get_rocm_events_info()
{
    if(!tool_data)
    {
        auto _tool_data_v = client_data<Backend>{};
        _tool_data_v.initialize_event_info();
        return _tool_data_v.events_info;
    }
    if(tool_data->events_info.empty()) tool_data->initialize_event_info();
    return tool_data->events_info;
}

// ─── Argument iteration callbacks ────────────────────────────────────────────

template <typename Backend>
int
library_sdk<Backend>::save_args(typename Backend::callback_tracing_kind /*kind*/,
                                std::int32_t /*operation*/, std::uint32_t /*arg_number*/,
                                const void* const /*arg_value_addr*/,
                                std::int32_t /*arg_indirection_count*/,
                                const char* /*arg_type*/, const char* arg_name,
                                const char* arg_value_str,
                                std::int32_t /*arg_dereference_count*/, void* data)
{
    auto* argvec = static_cast<callback_arg_array_t*>(data);
    argvec->emplace_back(arg_name, arg_value_str);
    return 0;
}

template <typename Backend>
int
library_sdk<Backend>::iterate_args_callback(
    typename Backend::callback_tracing_kind /*kind*/, std::int32_t /*operation*/,
    std::uint32_t arg_number, const void* const /*arg_value_addr*/,
    std::int32_t /*arg_indirection_count*/, const char* arg_type, const char* arg_name,
    const char* arg_value_str, std::int32_t /*arg_dereference_count*/, void* data)
{
    auto* _data = static_cast<function_args_t*>(data);
    if(arg_type && arg_name && arg_value_str)
        _data->emplace_back(argument_info{ arg_number,
                                           rocprofsys::utility::demangle(arg_type),
                                           arg_name, arg_value_str });
    return 0;
}

// ─── Cache helpers ────────────────────────────────────────────────────────────

template <typename Backend>
template <typename Category>
void
library_sdk<Backend>::cache_category()
{
    trace_cache::get_metadata_registry().add_string(trait::name<Category>::value);
}

template <typename Backend>
void
library_sdk<Backend>::cache_add_thread_info(std::uint64_t tid)
{
    trace_cache::get_metadata_registry().add_thread_info(
        { getppid(), getpid(), tid, 0, 0, "{}" });
}

template <typename Backend>
void
library_sdk<Backend>::cache_add_track(const char* track_name, std::uint64_t tid)
{
    trace_cache::get_metadata_registry().add_track({ track_name, tid, "{}" });
}

template <typename Backend>
void
library_sdk<Backend>::cache_region(
    const typename Backend::callback_tracing_record* record,
    typename Backend::timestamp_t                    start_timestamp,
    typename Backend::timestamp_t end_timestamp, const std::string& call_stack,
    const std::string& args_str, const std::string& category, std::string_view name)
{
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

template <typename Backend>
void
library_sdk<Backend>::cache_kernel_dispatch(
    typename Backend::kernel_dispatch_record* record, std::uint64_t stream_handle)
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

template <typename Backend>
void
library_sdk<Backend>::cache_scratch_memory(
    typename Backend::scratch_memory_record* record, std::uint64_t stream_handle)
{
    trace_cache::get_metadata_registry().add_stream(stream_handle);
    trace_cache::get_buffer_storage().store(trace_cache::scratch_memory_sample{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->agent_id.handle, record->queue_id.handle,
        static_cast<std::int32_t>(record->kind),
        static_cast<std::int32_t>(record->operation),
        static_cast<std::int32_t>(record->flags), get_scratch_mem_alloc_size(*record),
        record->correlation_id.internal, get_parent_stack_id(record->correlation_id),
        stream_handle });
}

template <typename Backend>
void
library_sdk<Backend>::cache_memory_copy(typename Backend::memory_copy_record* record,
                                        std::uint64_t stream_handle)
{
    trace_cache::get_metadata_registry().add_stream(stream_handle);
    trace_cache::get_buffer_storage().store(trace_cache::memory_copy_sample{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->dst_agent_id.handle, record->src_agent_id.handle,
        static_cast<std::int32_t>(record->kind),
        static_cast<std::int32_t>(record->operation), record->bytes,
        record->correlation_id.internal, get_parent_stack_id(record->correlation_id),
        get_mem_copy_dst_address(*record), get_mem_copy_src_address(*record),
        stream_handle });
}

#if ROCPROFILER_VERSION >= 600
template <typename Backend>
void
library_sdk<Backend>::cache_memory_allocation(
    typename Backend::memory_alloc_record* record, std::uint64_t stream_handle)
{
    trace_cache::get_metadata_registry().add_stream(stream_handle);
    trace_cache::get_buffer_storage().store(trace_cache::memory_allocate_sample{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->agent_id.handle, static_cast<std::int32_t>(record->kind),
        static_cast<std::int32_t>(record->operation), record->allocation_size,
        record->correlation_id.internal, get_parent_stack_id(record->correlation_id),
        get_mem_alloc_address(*record), stream_handle });
}
#endif

// ─── Address/size extractors ──────────────────────────────────────────────────

template <typename Backend>
size_t
library_sdk<Backend>::get_mem_copy_dst_address(
    [[maybe_unused]] const typename Backend::memory_copy_record& record)
{
#if(ROCPROFILER_VERSION >= 700)
    return record.dst_address.value;
#else
    return 0;
#endif
}

template <typename Backend>
size_t
library_sdk<Backend>::get_mem_copy_src_address(
    [[maybe_unused]] const typename Backend::memory_copy_record& record)
{
#if(ROCPROFILER_VERSION >= 700)
    return record.src_address.value;
#else
    return 0;
#endif
}

#if ROCPROFILER_VERSION >= 600
template <typename Backend>
size_t
library_sdk<Backend>::get_mem_alloc_address(
    [[maybe_unused]] const typename Backend::memory_alloc_record& record)
{
#    if(ROCPROFILER_VERSION >= 700)
    return record.address.value;
#    else
    return static_cast<size_t>(record.address.handle);
#    endif
}
#endif

template <typename Backend>
std::uint64_t
library_sdk<Backend>::get_scratch_mem_alloc_size(
    [[maybe_unused]] const typename Backend::scratch_memory_record& record)
{
#if ROCPROFSYS_ROCM_VERSION >= 70100
    return record.allocation_size;
#else
    return 0;
#endif
}

// ─── Kernel / code object info lookups ───────────────────────────────────────

template <typename Backend>
const kernel_symbol_data_t<Backend>*
library_sdk<Backend>::get_kernel_symbol_info(std::uint64_t kernel_id)
{
    return tool_data->get_kernel_symbol_info(kernel_id);
}

template <typename Backend>
const typename Backend::code_object_load_data*
library_sdk<Backend>::get_code_object_info(std::uint64_t code_object_id)
{
    return tool_data->get_code_object_info(code_object_id);
}

// ─── Backtrace helper ─────────────────────────────────────────────────────────

template <typename Backend>
auto
library_sdk<Backend>::get_backtrace(
    std::optional<std::vector<tim::unwind::processed_entry>>& bt_data)
{
    auto backtrace = nlohmann::json::object();
    if(bt_data && !bt_data->empty())
    {
        const std::string _unk    = "??";
        size_t            _bt_cnt = 0;
        for(const auto& itr : *bt_data)
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

// ─── Counter storage helpers ──────────────────────────────────────────────────

template <typename Backend>
auto&
library_sdk<Backend>::get_counter_dispatch_data()
{
    static auto _v = container::stable_vector<typename Backend::dispatch_counting_data>{};
    return _v;
}

template <typename Backend>
auto&
library_sdk<Backend>::get_counter_dispatch_records()
{
    static auto _v = std::vector<counter_dispatch_record>{};
    return _v;
}

template <typename Backend>
typename library_sdk<Backend>::agent_counter_storage_map_t*&
library_sdk<Backend>::get_counter_storage()
{
    static auto* _v = new agent_counter_storage_map_t{};
    return _v;
}

template <typename Backend>
auto&
library_sdk<Backend>::get_kernel_dispatch_timestamps()
{
    static auto _v =
        std::unordered_map<typename Backend::dispatch_id_t, timing_interval<Backend>>{};
    return _v;
}

template <typename Backend>
void
library_sdk<Backend>::flush_counter_storage_outputs()
{
    auto* _agent_counter_storage = get_counter_storage();
    if(!_agent_counter_storage) return;

    auto _cleanup_keys = std::vector<std::pair<std::string, const counter_storage*>>{};
    for(const auto& [agent_id, counter_map] : *_agent_counter_storage)
    {
        static_cast<void>(agent_id);
        for(const auto& [counter_id, storage] : counter_map)
        {
            static_cast<void>(counter_id);
            _cleanup_keys.emplace_back(storage.storage_name + "cleanup", &storage);
        }
    }

    std::sort(_cleanup_keys.begin(), _cleanup_keys.end(),
              [](const auto& lhs, const auto& rhs) { return *lhs.second < *rhs.second; });

    for(const auto& [cleanup_key, storage] : _cleanup_keys)
    {
        if(!storage || !storage->storage) continue;
        if(storage->manager)
            storage->manager->cleanup(cleanup_key);
        else
            counter_storage::write(storage->storage.get(), storage->metric_name,
                                   storage->metric_description);
    }
}

template <typename Backend>
void
library_sdk<Backend>::flush_counter_tracks_to_zero(
    typename Backend::timestamp_t timestamp)
{
    if(timestamp == 0)
    {
        ROCPROFILER_CALL(Backend::get_timestamp(&timestamp));
    }
    auto* storage = get_counter_storage();
    if(!storage) return;
    for(auto& [agent_id, counter_map] : *storage)
        for(auto& [counter_id, cs] : counter_map)
            cs.write_zero(timestamp);
}

template <typename Backend>
void
library_sdk<Backend>::finalize_sdk_common()
{
#if(ROCPROFILER_VERSION >= 600)
    ompt_finalize_orphan_events();
#endif
    flush();
    stop();
    if(get_counter_storage())
    {
        flush_counter_storage_outputs();
        get_counter_storage()->clear();
        delete get_counter_storage();
        get_counter_storage() = nullptr;
    }
}

// ─── create_agent_profile ────────────────────────────────────────────────────

template <typename Backend>
std::vector<typename Backend::counter_id>
library_sdk<Backend>::create_agent_profile(typename Backend::agent_id      agent_id,
                                           const std::vector<std::string>& counters,
                                           client_data<Backend>*           data)
{
    using counter_vec_t = std::vector<typename Backend::counter_id>;
    if(!data) data = tool_data;

    if(data->agent_counter_profiles.find(agent_id) != data->agent_counter_profiles.end())
        return counter_vec_t{};

    auto        profile      = std::optional<typename Backend::counter_config_id>{};
    auto        expected_v   = counters.size();
    auto        found_v      = std::vector<std::string_view>{};
    auto        counters_v   = counter_vec_t{};
    const auto* tool_agent_v = data->get_gpu_tool_agent(agent_id);

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
                ::rocprofsys::set_state(::rocprofsys::State::Finalized);
                std::abort();
            }

            auto dev_id_v = std::stoul(dev_id_s);
            if(dev_id_v != tool_agent_v->device_id)
            {
                --expected_v;
                continue;
            }
        }

        auto _old_name_v = name_v;
        name_v =
            std::regex_replace(name_v, std::regex{ "^(.*)(\\[)([0-9]+)(\\])$" }, "$1");

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
        auto missing_counters = std::vector<std::string>{};
        for(const auto& counter : counters)
            if(std::find(found_v.begin(), found_v.end(), counter) == found_v.end())
                missing_counters.emplace_back(counter);

        LOG_WARNING("Unable to find all counters for agent {} (gpu-{}, {}). "
                    "Requested: {}. Found: {}. Missing: {}. Continuing with "
                    "available counters.",
                    tool_agent_v->agent->node_id, tool_agent_v->device_id,
                    tool_agent_v->agent->name,
                    fmt::format("{}", fmt::join(counters, ", ")),
                    fmt::format("{}", fmt::join(found_v, ", ")),
                    fmt::format("{}", fmt::join(missing_counters, ", ")));
    }

    if(!counters_v.empty())
    {
        auto profile_v = typename Backend::counter_config_id{};
        ROCPROFILER_CALL(Backend::create_counter_config(agent_id, counters_v.data(),
                                                        counters_v.size(), &profile_v));
        profile = profile_v;
    }

    data->agent_counter_profiles.emplace(agent_id, profile);
    return counters_v;
}

// ─── External correlation ID callback ────────────────────────────────────────

template <typename Backend>
int
library_sdk<Backend>::set_kernel_rename_and_stream_correlation_id(
    typename Backend::thread_id /*thr_id*/, typename Backend::context_id /*ctx_id*/,
    typename Backend::external_correlation_request_kind /*kind*/,
    typename Backend::tracing_operation /*op*/, std::uint64_t /*internal_corr_id*/,
    typename Backend::user_data_t* external_corr_id, void* /*user_data*/)
{
    auto* _info           = new kernel_rename_and_stream_data{};
    _info->stream_id      = stream_id_top();
    external_corr_id->ptr = _info;
    return 0;
}

// ─── tool_code_object_callback ───────────────────────────────────────────────

template <typename Backend>
void
library_sdk<Backend>::tool_code_object_callback(
    typename Backend::callback_tracing_record record,
    typename Backend::user_data_t* /*user_data*/, void* /*callback_data*/)
{
    auto ts = typename Backend::timestamp_t{};
    ROCPROFILER_CALL(Backend::get_timestamp(&ts));

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            if(record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
            {
                auto data_v = *static_cast<typename Backend::code_object_load_data*>(
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
                auto data_v =
                    *static_cast<kernel_symbol_data_t<Backend>*>(record.payload);
                tool_data->kernel_symbol_records.wlock(
                    [ts, &record, &data_v](auto& _data) {
                        _data.emplace_back(
                            kernel_symbol_callback_record_t{ ts, record, data_v });
                    });
                trace_cache::get_metadata_registry().add_kernel_symbol(data_v);
            }
        }
    }
}

// ─── tool_tracing_callback_start / stop ──────────────────────────────────────

template <typename Backend>
template <typename CategoryT>
void
library_sdk<Backend>::tool_tracing_callback_start(
    CategoryT, typename Backend::callback_tracing_record record,
    typename Backend::user_data_t* /*user_data*/, typename Backend::timestamp_t /*ts*/)
{
    auto _name = tool_data->callback_tracing_info.at(record.kind, record.operation);
    if(get_use_timemory()) tracing::push_timemory(CategoryT{}, _name);
}

template <typename Backend>
template <typename CategoryT>
void
library_sdk<Backend>::tool_tracing_callback_stop(
    CategoryT, typename Backend::callback_tracing_record record,
    typename Backend::user_data_t* user_data, typename Backend::timestamp_t ts,
    std::optional<std::vector<tim::unwind::processed_entry>>& _bt_data)
{
    auto _name    = tool_data->callback_tracing_info.at(record.kind, record.operation);
    auto begin_ts = user_data->value;

    if(get_use_timemory()) tracing::pop_timemory(CategoryT{}, _name);

    if(get_use_perfetto())
    {
        auto args = callback_arg_array_t{};
        if(config::get_perfetto_annotations())
        {
            Backend::iterate_callback_tracing_kind_operation_args(record, save_args, 2,
                                                                  &args);
        }

        std::uint64_t _beg_ts   = begin_ts;
        std::uint64_t _end_ts   = ts;
        auto          stream_id = stream_id_top();

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
                        size_t _bt_cnt = 0;
                        for(const auto& itr : *_bt_data)
                        {
                            const std::string _unk   = "??";
                            auto              _linfo = itr.lineinfo.get();
                            const auto* _func = (itr.name.empty()) ? &_unk : &itr.name;
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
                                tracing::add_perfetto_annotation(
                                    ctx, fmt::format("frame#0{}", _bt_cnt++), _entry);
                            else
                                tracing::add_perfetto_annotation(
                                    ctx, fmt::format("frame#{}", _bt_cnt++), _entry);
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

    auto args = function_args_t{};
    Backend::iterate_callback_tracing_kind_operation_args(record, iterate_args_callback,
                                                          2, &args);

    auto          call_stack = get_backtrace(_bt_data);
    std::uint64_t _beg_ts    = begin_ts;
    std::uint64_t _end_ts    = ts;

    cache_category<CategoryT>();
    cache_add_thread_info(record.thread_id);
    std::string args_str = get_args_string(args);
    cache_region(&record, _beg_ts, _end_ts, call_stack.dump(), args_str,
                 trait::name<CategoryT>::value, _name);
}

// ─── tool_tracing_callback (main dispatch) ───────────────────────────────────

template <typename Backend>
void
library_sdk<Backend>::tool_tracing_callback(
    typename Backend::callback_tracing_record record,
    typename Backend::user_data_t*            user_data, void* /*callback_data*/)
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
                        _bt_data->emplace_back(std::move(*_val));
                }
            }
        }
    };

#if(ROCPROFILER_VERSION >= 600)
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_OMPT)
    {
        auto* payload_data = static_cast<typename Backend::ompt_data_t*>(record.payload);
        if(!payload_data) return;
        switch(record.operation)
        {
            case ROCPROFILER_OMPT_ID_implicit_task:
            {
                int flag = payload_data->args.implicit_task.flags;
                if(flag & ompt_task_initial) return;
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
    }
#endif

    auto ts = typename Backend::timestamp_t{};
    ROCPROFILER_CALL(Backend::get_timestamp(&ts));
    const char* name = "";
    Backend::query_callback_op_name(record.kind, record.operation, &name, nullptr);

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
                tool_tracing_callback_start(category::rocm_hsa_api{}, record, user_data,
                                            ts);
                break;
            case ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API:
            case ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API:
                tool_tracing_callback_start(category::rocm_hip_api{}, record, user_data,
                                            ts);
                break;
#if(ROCPROFILER_VERSION >= 600)
            case ROCPROFILER_CALLBACK_TRACING_OMPT:
                ompt_tracing_callback_start(record, user_data, ts);
                ompt_push_standard_callback(record, ts);
                break;
            case ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API:
                tool_tracing_callback_start(category::rocm_rocdecode_api{}, record,
                                            user_data, ts);
                break;
#endif
#if(ROCPROFILER_VERSION >= 700)
            case ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API:
                tool_tracing_callback_start(category::rocm_rocjpeg_api{}, record,
                                            user_data, ts);
                break;
#endif
            case ROCPROFILER_CALLBACK_TRACING_RCCL_API:
                tool_tracing_callback_start(category::rocm_rccl_api{}, record, user_data,
                                            ts);
                break;
            case ROCPROFILER_CALLBACK_TRACING_NONE:
            case ROCPROFILER_CALLBACK_TRACING_LAST:
            case ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API:
            case ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API:
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
                LOG_CRITICAL("Unhandled callback record: {}",
                             static_cast<int>(record.kind));
                ::rocprofsys::set_state(::rocprofsys::State::Finalized);
                std::abort();
                break;
            }
            default:
                LOG_CRITICAL("Unhandled callback record: {}", info.str());
                ::rocprofsys::set_state(::rocprofsys::State::Finalized);
                std::abort();
                break;
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
                tool_tracing_callback_stop(category::rocm_hsa_api{}, record, user_data,
                                           ts, _bt_data);
                break;
            case ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API:
            case ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API:
                tool_tracing_callback_stop(category::rocm_hip_api{}, record, user_data,
                                           ts, _bt_data);
                break;
#if(ROCPROFILER_VERSION >= 600)
            case ROCPROFILER_CALLBACK_TRACING_OMPT:
                ompt_tracing_callback_stop(record, user_data, ts, _bt_data);
                ompt_pop_standard_callback(record, ts, _bt_data);
                break;
            case ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API:
                tool_tracing_callback_stop(category::rocm_rocdecode_api{}, record,
                                           user_data, ts, _bt_data);
                break;
#endif
#if(ROCPROFILER_VERSION >= 700)
            case ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API:
                tool_tracing_callback_stop(category::rocm_rocjpeg_api{}, record,
                                           user_data, ts, _bt_data);
                break;
#endif
            case ROCPROFILER_CALLBACK_TRACING_RCCL_API:
            {
                auto* rccl_payload =
                    static_cast<typename Backend::rccl_api_data*>(record.payload);
                tool_tracing_callback_rccl(record.operation, rccl_payload,
                                           user_data->value, ts);
                tool_tracing_callback_stop(category::rocm_rccl_api{}, record, user_data,
                                           ts, _bt_data);
                break;
            }
            case ROCPROFILER_CALLBACK_TRACING_NONE:
            case ROCPROFILER_CALLBACK_TRACING_LAST:
            case ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API:
            case ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API:
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
                LOG_CRITICAL("Unhandled callback record: {}",
                             static_cast<int>(record.kind));
                ::rocprofsys::set_state(::rocprofsys::State::Finalized);
                std::abort();
                break;
            }
            default:
                LOG_CRITICAL("Unhandled callback record: {}", info.str());
                ::rocprofsys::set_state(::rocprofsys::State::Finalized);
                std::abort();
                break;
        }
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_NONE)
    {
        switch(record.kind)
        {
            case ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH:
                if(record.operation == ROCPROFILER_KERNEL_DISPATCH_COMPLETE)
                {
                    auto* _data = static_cast<typename Backend::kernel_dispatch_data*>(
                        record.payload);
                    get_kernel_dispatch_timestamps().emplace(
                        _data->dispatch_info.dispatch_id,
                        timing_interval<Backend>{ _data->start_timestamp,
                                                  _data->end_timestamp });
                }
                break;
#if(ROCPROFILER_VERSION >= 600)
            case ROCPROFILER_CALLBACK_TRACING_OMPT:
            {
                static const std::set<typename Backend::ompt_operation_t>
                    ompt_no_process = {
                        ROCPROFILER_OMPT_ID_callback_functions,
                        ROCPROFILER_OMPT_ID_thread_end,
                    };
                auto ompt_operation_type =
                    static_cast<typename Backend::ompt_operation_t>(record.operation);
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
                    case ROCPROFILER_OMPT_ID_thread_begin:
                    case ROCPROFILER_OMPT_ID_lock_init:
                    case ROCPROFILER_OMPT_ID_lock_destroy:
                    case ROCPROFILER_OMPT_ID_nest_lock:
                    case ROCPROFILER_OMPT_ID_dispatch:
                    case ROCPROFILER_OMPT_ID_flush:
                    case ROCPROFILER_OMPT_ID_cancel:
                    case ROCPROFILER_OMPT_ID_device_initialize:
                    case ROCPROFILER_OMPT_ID_device_finalize:
                    case ROCPROFILER_OMPT_ID_device_load:
                    case ROCPROFILER_OMPT_ID_task_create:
                    case ROCPROFILER_OMPT_ID_task_schedule:
                    case ROCPROFILER_OMPT_ID_mutex_released:
                    case ROCPROFILER_OMPT_ID_mutex_acquire:
                    case ROCPROFILER_OMPT_ID_mutex_acquired:
                    case ROCPROFILER_OMPT_ID_dependences:
                    case ROCPROFILER_OMPT_ID_task_dependence:
                    case ROCPROFILER_OMPT_ID_error:
                    {
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
                break;
            }
#endif
            default:
                LOG_WARNING("tool_tracing_callback: unhandled PHASE_NONE "
                            "callback record: {}",
                            info.str());
                break;
        }
    }
    else
    {
        LOG_CRITICAL("unhandled callback record phase: {}",
                     static_cast<int>(record.phase));
        ::rocprofsys::set_state(::rocprofsys::State::Finalized);
        ::std::abort();
    }
}

#if ROCPROFILER_VERSION >= 600

// ─── OMPT helpers ────────────────────────────────────────────────────────────

template <typename Backend>
std::string_view
library_sdk<Backend>::ompt_get_unified_name(
    const typename Backend::callback_tracing_record& record)
{
    std::string_view _name =
        tool_data->callback_tracing_info.at(record.kind, record.operation);
    if(record.operation == ROCPROFILER_OMPT_ID_parallel_begin ||
       record.operation == ROCPROFILER_OMPT_ID_parallel_end)
        _name = "omp_parallel";
    return _name;
}

template <typename Backend>
template <typename ArgsT>
void
library_sdk<Backend>::ompt_iterate_operation_args(
    const typename Backend::callback_tracing_record& record, ArgsT& args)
{
    static_assert(std::is_same_v<ArgsT, callback_arg_array_t> ||
                      std::is_same_v<ArgsT, function_args_t>,
                  "ArgsT must be callback_arg_array_t or function_args_t");

    auto ompt_operation_type =
        static_cast<typename Backend::ompt_operation_t>(record.operation);
    auto max_deref = (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER ||
                      ompt_operation_type == ROCPROFILER_OMPT_ID_parallel_begin)
                         ? 1
                         : 2;

    if constexpr(std::is_same_v<ArgsT, callback_arg_array_t>)
        Backend::iterate_callback_tracing_kind_operation_args(record, save_args,
                                                              max_deref, &args);
    else
        Backend::iterate_callback_tracing_kind_operation_args(
            record, iterate_args_callback, max_deref, &args);

    static const auto ompt_has_flags = std::set<typename Backend::ompt_operation_t>{
        ROCPROFILER_OMPT_ID_parallel_begin, ROCPROFILER_OMPT_ID_parallel_end,
        ROCPROFILER_OMPT_ID_task_create,    ROCPROFILER_OMPT_ID_implicit_task,
        ROCPROFILER_OMPT_ID_cancel,
    };
    if(ompt_has_flags.find(ompt_operation_type) == ompt_has_flags.end()) return;

    auto append = [&args](const std::string& flag_type, const std::string& key,
                          const std::string& val) {
        if constexpr(std::is_same_v<ArgsT, callback_arg_array_t>)
            args.emplace_back(key, val);
        else
            args.emplace_back(argument_info{ static_cast<std::uint32_t>(args.size()),
                                             flag_type, key, val });
    };

    int   flags_val    = 0;
    auto* payload_data = static_cast<typename Backend::ompt_data_t*>(record.payload);
    if(!payload_data) return;

    switch(ompt_operation_type)
    {
        case ROCPROFILER_OMPT_ID_parallel_begin:
            flags_val = payload_data->args.parallel_begin.flags;
            break;
        case ROCPROFILER_OMPT_ID_parallel_end:
            flags_val = payload_data->args.parallel_end.flags;
            break;
        case ROCPROFILER_OMPT_ID_task_create:
            flags_val = payload_data->args.task_create.flags;
            break;
        case ROCPROFILER_OMPT_ID_implicit_task:
            flags_val = payload_data->args.implicit_task.flags;
            break;
        case ROCPROFILER_OMPT_ID_cancel:
            flags_val = payload_data->args.cancel.flags;
            break;
        default: break;
    }

    switch(ompt_operation_type)
    {
        case ROCPROFILER_OMPT_ID_parallel_begin:
        case ROCPROFILER_OMPT_ID_parallel_end:
        {
            const auto ft = std::string{ "ompt_parallel_flag_t" };
            if(flags_val & ompt_parallel_invoker_program)
                append(ft, "invoker", "program");
            else if(flags_val & ompt_parallel_invoker_runtime)
                append(ft, "invoker", "runtime");
            if(flags_val & ompt_parallel_league)
                append(ft, "invoker_cause", "teams_construct");
            else if(flags_val & ompt_parallel_team)
                append(ft, "invoker_cause", "parallel_construct");
            break;
        }
        case ROCPROFILER_OMPT_ID_task_create:
        {
            const auto ft = std::string{ "ompt_task_flag_t" };
            if(flags_val & ompt_task_initial)
                append(ft, "classification", "initial");
            else if(flags_val & ompt_task_implicit)
                append(ft, "classification", "implicit");
            else if(flags_val & ompt_task_explicit)
                append(ft, "classification", "explicit");
            else if(flags_val & ompt_task_target)
                append(ft, "classification", "target");
            std::string props;
            props.reserve(60);
            if(flags_val & ompt_task_undeferred) props += "undeferred, ";
            if(flags_val & ompt_task_untied) props += "untied, ";
            if(flags_val & ompt_task_final) props += "final, ";
            if(flags_val & ompt_task_mergeable) props += "mergeable, ";
            if(flags_val & ompt_task_merged) props += "merged, ";
            if(!props.empty())
                props.erase(props.size() - 2);
            else
                props = "none";
            append(ft, "properties", props);
            break;
        }
        case ROCPROFILER_OMPT_ID_implicit_task:
        {
            const auto ft = std::string{ "flags" };
            if(flags_val & ompt_task_initial)
                append(ft, "kind", "initial");
            else if(flags_val & ompt_task_implicit)
                append(ft, "kind", "implicit");
            break;
        }
        case ROCPROFILER_OMPT_ID_cancel:
        {
            const auto ft = std::string{ "ompt_cancel_flag_t" };
            if(flags_val & ompt_cancel_parallel)
                append(ft, "construct", "parallel");
            else if(flags_val & ompt_cancel_sections)
                append(ft, "construct", "sections");
            else if(flags_val & ompt_cancel_loop)
                append(ft, "construct", "loop");
            else if(flags_val & ompt_cancel_taskgroup)
                append(ft, "construct", "taskgroup");
            if(flags_val & ompt_cancel_activated)
                append(ft, "state", "activated");
            else if(flags_val & ompt_cancel_detected)
                append(ft, "state", "detected");
            else if(flags_val & ompt_cancel_discarded_task)
                append(ft, "state", "discarded_task");
            break;
        }
        default: break;
    }
}

template <typename Backend>
auto&
library_sdk<Backend>::get_ompt_standard_cb_storage()
{
    static thread_local auto _v =
        std::unordered_map<std::uint64_t, rocprofsys_ompt_data_storage_t>{};
    return _v;
}

template <typename Backend>
auto&
library_sdk<Backend>::get_ompt_parallel_cb_storage()
{
    static thread_local auto _v =
        std::unordered_map<uintptr_t, rocprofsys_ompt_data_storage_t>{};
    return _v;
}

template <typename Backend>
void
library_sdk<Backend>::ompt_cache_instant_event(
    typename Backend::callback_tracing_record                 record,
    typename Backend::timestamp_t                             instant_ts,
    std::optional<std::vector<tim::unwind::processed_entry>>& bt_data)
{
    auto args = function_args_t{};
    ompt_iterate_operation_args(record, args);
    auto call_stack = get_backtrace(bt_data);
    cache_category<category::rocm_ompt_api>();
    cache_add_thread_info(record.thread_id);
    cache_region(&record, instant_ts, instant_ts, call_stack.dump(),
                 get_args_string(args), trait::name<category::rocm_ompt_api>::value);
}

template <typename Backend>
void
library_sdk<Backend>::ompt_cache_orphan_event(
    const rocprofsys_ompt_data_storage_t&                     stored_data,
    std::optional<std::vector<tim::unwind::processed_entry>>& bt_data)
{
    auto call_stack = get_backtrace(bt_data);
    cache_category<category::rocm_ompt_api>();
    cache_add_thread_info(stored_data.record.thread_id);
    cache_region(&stored_data.record, stored_data._beg_ts, stored_data._beg_ts,
                 call_stack.dump(), get_args_string(stored_data.args),
                 trait::name<category::rocm_ompt_api>::value);
}

template <typename Backend>
void
library_sdk<Backend>::ompt_push_standard_callback(
    const typename Backend::callback_tracing_record& record,
    const typename Backend::timestamp_t&             beg_ts)
{
    auto args = function_args_t{};
    ompt_iterate_operation_args(record, args);
    get_ompt_standard_cb_storage().emplace(
        record.correlation_id.internal,
        rocprofsys_ompt_data_storage_t{ record, beg_ts, args });
}

template <typename Backend>
void
library_sdk<Backend>::ompt_pop_standard_callback(
    const typename Backend::callback_tracing_record&          record,
    const typename Backend::timestamp_t&                      end_ts,
    std::optional<std::vector<tim::unwind::processed_entry>>& bt_data)
{
    auto it = get_ompt_standard_cb_storage().find(record.correlation_id.internal);
    if(it == get_ompt_standard_cb_storage().end())
    {
        auto args = function_args_t{};
        ompt_iterate_operation_args(record, args);
        ompt_cache_orphan_event(rocprofsys_ompt_data_storage_t{ record, end_ts, args },
                                bt_data);
        return;
    }
    auto stored_data = it->second;
    get_ompt_standard_cb_storage().erase(it);
    auto call_stack = get_backtrace(bt_data);
    cache_category<category::rocm_ompt_api>();
    cache_add_thread_info(record.thread_id);
    cache_region(&record, stored_data._beg_ts, end_ts, call_stack.dump(),
                 get_args_string(stored_data.args),
                 trait::name<category::rocm_ompt_api>::value);
}

template <typename Backend>
void
library_sdk<Backend>::ompt_push_parallel_callback(
    const typename Backend::callback_tracing_record& record,
    const typename Backend::timestamp_t&             beg_ts)
{
    auto* payload_data = static_cast<typename Backend::ompt_data_t*>(record.payload);
    const void* parallel_data_address = payload_data->args.parallel_begin.parallel_data;
    auto        args                  = function_args_t{};
    ompt_iterate_operation_args(record, args);
    get_ompt_parallel_cb_storage().emplace(
        reinterpret_cast<uintptr_t>(parallel_data_address),
        rocprofsys_ompt_data_storage_t{ record, beg_ts, args });
}

template <typename Backend>
void
library_sdk<Backend>::ompt_pop_parallel_callback(
    const typename Backend::callback_tracing_record&          record,
    const typename Backend::timestamp_t&                      end_ts,
    std::optional<std::vector<tim::unwind::processed_entry>>& bt_data)
{
    auto* payload_data = static_cast<typename Backend::ompt_data_t*>(record.payload);
    const void* parallel_data_address = payload_data->args.parallel_end.parallel_data;

    auto it = get_ompt_parallel_cb_storage().find(
        reinterpret_cast<uintptr_t>(parallel_data_address));
    if(it == get_ompt_parallel_cb_storage().end())
    {
        auto args = function_args_t{};
        ompt_iterate_operation_args(record, args);
        ompt_cache_orphan_event(rocprofsys_ompt_data_storage_t{ record, end_ts, args },
                                bt_data);
        return;
    }
    auto stored_data = it->second;
    get_ompt_parallel_cb_storage().erase(it);
    auto call_stack = get_backtrace(bt_data);
    cache_category<category::rocm_ompt_api>();
    cache_add_thread_info(record.thread_id);
    cache_region(&record, stored_data._beg_ts, end_ts, call_stack.dump(),
                 get_args_string(stored_data.args),
                 trait::name<category::rocm_ompt_api>::value);
}

template <typename Backend>
void
library_sdk<Backend>::ompt_tracing_callback_start(
    typename Backend::callback_tracing_record record,
    typename Backend::user_data_t* /*user_data*/, typename Backend::timestamp_t ts)
{
    std::string_view _name = ompt_get_unified_name(record);
    if(get_use_timemory()) tracing::push_timemory(category::rocm_ompt_api{}, _name);
    if(get_use_perfetto())
    {
        auto args = callback_arg_array_t{};
        if(config::get_perfetto_annotations()) ompt_iterate_operation_args(record, args);
        std::uint64_t _beg_ts   = ts;
        auto          stream_id = stream_id_top();
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
                        tracing::add_perfetto_annotation(ctx, key, val);
                }
            });
    }
}

template <typename Backend>
void
library_sdk<Backend>::ompt_tracing_callback_stop(
    typename Backend::callback_tracing_record record,
    typename Backend::user_data_t* /*user_data*/, typename Backend::timestamp_t ts,
    std::optional<std::vector<tim::unwind::processed_entry>>& bt_data)
{
    std::string_view _name = ompt_get_unified_name(record);
    if(get_use_timemory()) tracing::pop_timemory(category::rocm_ompt_api{}, _name);
    if(get_use_perfetto())
    {
        auto args = callback_arg_array_t{};
        if(config::get_perfetto_annotations()) ompt_iterate_operation_args(record, args);
        std::uint64_t _end_ts = ts;
        tracing::pop_perfetto_ts(
            category::rocm_ompt_api{}, _name.data(), _end_ts,
            [&](::perfetto::EventContext ctx) {
                if(config::get_perfetto_annotations())
                    tracing::add_perfetto_annotation(ctx, "end_ns", _end_ts);
                if(bt_data && !bt_data->empty())
                {
                    size_t _bt_cnt = 0;
                    for(const auto& itr : *bt_data)
                    {
                        const std::string _unk   = "??";
                        auto              _linfo = itr.lineinfo.get();
                        const auto*       _func  = (itr.name.empty()) ? &_unk : &itr.name;
                        const auto*       _loc =
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
                            tracing::add_perfetto_annotation(
                                ctx, fmt::format("frame#0{}", _bt_cnt++), _entry);
                        else
                            tracing::add_perfetto_annotation(
                                ctx, fmt::format("frame#{}", _bt_cnt++), _entry);
                    }
                }
            });
    }
}

template <typename Backend>
void
library_sdk<Backend>::ompt_finalize_orphan_events()
{
    auto empty_call_stack =
        std::optional<std::vector<tim::unwind::processed_entry>>{ std::nullopt };
    for(const auto& [parallel_data, stored_data] : get_ompt_parallel_cb_storage())
        ompt_cache_orphan_event(stored_data, empty_call_stack);
    for(const auto& [correlation_id, stored_data] : get_ompt_standard_cb_storage())
        ompt_cache_orphan_event(stored_data, empty_call_stack);
    get_ompt_parallel_cb_storage().clear();
    get_ompt_standard_cb_storage().clear();
}

#endif  // ROCPROFILER_VERSION >= 600

// ─── tool_tracing_buffered ───────────────────────────────────────────────────
// (The full body is copied verbatim from the CPP; no SDK calls here to replace.)

template <typename Backend>
void
library_sdk<Backend>::tool_tracing_buffered(typename Backend::context_id /*context*/,
                                            typename Backend::buffer_id /*buffer_id*/,
                                            typename Backend::record_header_t** headers,
                                            size_t num_headers, void* /*user_data*/,
                                            std::uint64_t /*drop_count*/)
{
    if(num_headers == 0 || headers == nullptr) return;

    auto _track_desc_stream = [](std::uint64_t _stream_id) {
        return fmt::format("HIP Activity Stream {}", _stream_id);
    };

    const bool _default_group_by_queue = config::get_group_by_queue();

    static auto _mtx = std::mutex{};
    auto        _lk  = std::unique_lock<std::mutex>{ _mtx };

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(ROCPROFSYS_LIKELY(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING))
        {
            if(header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
            {
                auto* record = static_cast<typename Backend::kernel_dispatch_record*>(
                    header->payload);
                bool        _group_by_queue = _default_group_by_queue;
                const auto* _kern_sym_data =
                    get_kernel_symbol_info(record->dispatch_info.kernel_id);
                auto _name = rocprofsys::utility::demangle(_kern_sym_data->kernel_name);
                auto _stack_id           = record->correlation_id.internal;
                auto _beg_ns             = record->start_timestamp;
                auto _end_ns             = record->end_timestamp;
                auto _agent_id           = record->dispatch_info.agent_id;
                auto _queue_id           = record->dispatch_info.queue_id;
                const auto*   _agent     = tool_data->get_gpu_tool_agent(_agent_id);
                std::uint64_t _stream_id = get_stream_id(record).handle;
                if(_stream_id == 0) _group_by_queue = true;

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
                    const auto& _tinfo  = thread_info::get(record->thread_id, SystemTID);
                    auto        _tid    = _tinfo->index_data->sequent_value;
                    auto        _bundle = kernel_dispatch_bundle_t{ _name };
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
                        auto _track_desc = [](std::int32_t _device_id_v,
                                              std::int64_t _queue_id_v) {
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
                auto* record = static_cast<typename Backend::scratch_memory_record*>(
                    header->payload);
                bool        _group_by_queue = _default_group_by_queue;
                const auto* agent     = tool_data->get_gpu_tool_agent(record->agent_id);
                auto        device_id = static_cast<std::uint32_t>(agent->device_id);
                const auto& t_info    = thread_info::get(record->thread_id, SystemTID);
                auto        thread_id_sequent = t_info->index_data->sequent_value;
                auto        _corr_id          = record->correlation_id.internal;
                auto        _beg_ns           = record->start_timestamp;
                auto        _end_ns           = record->end_timestamp;
                auto        _name =
                    tool_data->buffered_tracing_info.at(record->kind, record->operation);
                auto _stream_id = get_stream_id(record).handle;
                if(_stream_id == 0) _group_by_queue = true;

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
                    static_cast<typename Backend::memory_copy_record*>(header->payload);
                bool        _group_by_queue = _default_group_by_queue;
                auto        _stack_id       = record->correlation_id.internal;
                auto        _beg_ns         = record->start_timestamp;
                auto        _end_ns         = record->end_timestamp;
                auto        _dst_agent_id   = record->dst_agent_id;
                auto        _src_agent_id   = record->src_agent_id;
                const auto* _dst_agent      = tool_data->get_agent(_dst_agent_id);
                const auto* _src_agent      = tool_data->get_agent(_src_agent_id);
                auto        _name =
                    tool_data->buffered_tracing_info.at(record->kind, record->operation);
                std::uint64_t _stream_id = get_stream_id(record).handle;
                if(_stream_id == 0) _group_by_queue = true;

                {
                    auto track_name =
                        fmt::format("GPU Memory Copy to Agent [{}] Thread {}",
                                    _dst_agent->logical_node_id, record->thread_id);
                    cache_category<category::rocm_memory_copy>();
                    cache_add_track(track_name.c_str(), record->thread_id);
                    cache_memory_copy(record, _stream_id);
                }

                if(get_use_timemory())
                {
                    const auto& _tinfo  = thread_info::get(record->thread_id, SystemTID);
                    auto        _tid    = _tinfo->index_data->sequent_value;
                    auto        _bundle = kernel_dispatch_bundle_t{ _name };
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
                        auto _track_desc = [](std::int32_t                _device_id_v,
                                              typename Backend::thread_id _tid) {
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
                    static_cast<typename Backend::memory_alloc_record*>(header->payload);
                std::uint64_t _stream_id = get_stream_id(record).handle;
                cache_category<category::rocm_memory_allocate>();
                cache_add_thread_info(record->thread_id);
                cache_memory_allocation(record, _stream_id);
            }
#endif
#if(ROCPROFILER_VERSION >= 10000)
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT)
            {
                auto* record = static_cast<typename Backend::kfd_page_fault_record*>(
                    header->payload);
                tool_kfd_page_fault_callback(tool_data, record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE)
            {
                auto* record = static_cast<typename Backend::kfd_page_migrate_record*>(
                    header->payload);
                tool_kfd_page_migrate_callback(tool_data, record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KFD_QUEUE)
            {
                auto* record =
                    static_cast<typename Backend::kfd_queue_record*>(header->payload);
                tool_kfd_queue_callback(tool_data, record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE)
            {
                auto* record = static_cast<typename Backend::kfd_event_queue_record*>(
                    header->payload);
                tool_kfd_event_queue_callback(tool_data, record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU)
            {
                auto* record = static_cast<typename Backend::kfd_event_unmap_record*>(
                    header->payload);
                tool_kfd_event_unmap_from_gpu_callback(tool_data, record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS)
            {
                auto* record = static_cast<typename Backend::kfd_event_dropped_record*>(
                    header->payload);
                tool_kfd_event_dropped_events_callback(tool_data, record);
            }
#endif
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_HSA_CORE_API ||
                    header->kind == ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API)
            {
                continue;
            }
            else
            {
                throw std::runtime_error(fmt::format(
                    "unexpected typename Backend::record_header_t buffer tracing "
                    "category "
                    "kind. category: {}, kind: {}",
                    static_cast<int>(header->category), static_cast<int>(header->kind)));
            }
        }
        else
        {
            throw std::runtime_error(fmt::format(
                "unexpected typename Backend::record_header_t buffer tracing category "
                "kind. category: {}, kind: {}",
                static_cast<int>(header->category), static_cast<int>(header->kind)));
        }
    }
}

// ─── Counter callbacks ────────────────────────────────────────────────────────

template <typename Backend>
void
library_sdk<Backend>::counter_record_callback(
    typename Backend::dispatch_counting_data dispatch_data,
    typename Backend::counter_record* record_data, size_t record_count,
    typename Backend::user_data_t /*user_data*/, void* /*callback_data_arg*/)
{
    auto* _agent_counter_storage = get_counter_storage();
    if(!_agent_counter_storage) return;

    static auto _mtx = std::mutex{};
    auto        _lk  = std::unique_lock<std::mutex>{ _mtx };

    auto _dispatch_id = dispatch_data.dispatch_info.dispatch_id;
    auto _agent_id    = dispatch_data.dispatch_info.agent_id;
    auto _scope       = scope::get_default();
    auto _interval    = timing_interval<Backend>{};
    auto _aggregate   = std::unordered_map<typename Backend::counter_id,
                                           typename Backend::counter_record>{};

    for(size_t i = 0; i < record_count; ++i)
    {
        auto _counter_id = typename Backend::counter_id{};
        ROCPROFILER_CALL(
            Backend::query_record_counter_id(record_data[i].id, &_counter_id));
        if(!_aggregate.emplace(_counter_id, record_data[i]).second)
            _aggregate[_counter_id].counter_value += record_data[i].counter_value;
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
                ::rocprofsys::set_state(::rocprofsys::State::Finalized);
                ::std::abort();
            }
            if(!_info)
            {
                LOG_CRITICAL("unable to find counter info for counter (id={}) on "
                             "agent (id={})",
                             itr.first.handle, _agent_id.handle);
                ::rocprofsys::set_state(::rocprofsys::State::Finalized);
                ::std::abort();
            }
            auto _dev_id = static_cast<std::uint32_t>(_agent->device_id);
            _agent_counter_storage->at(_agent_id).emplace(
                itr.first, counter_storage{ tool_data, _dev_id, 0, _info->name });
        }
        auto _event = counter_event{ counter_dispatch_record{
            &dispatch_data, _dispatch_id, itr.first, itr.second } };
        _agent_counter_storage->at(_agent_id).at(itr.first)(_event, _interval, _scope);
    }
}

template <typename Backend>
void
library_sdk<Backend>::dispatch_counting_service_callback(
    typename Backend::dispatch_counting_data dispatch_data,
    typename Backend::counter_config_id*     config,
    typename Backend::user_data_t* /*user_data*/, void* callback_data_arg)
{
    auto* _data = as_client_data<Backend>(callback_data_arg);
    if(!_data || !config) return;
    if(auto itr =
           _data->agent_counter_profiles.find(dispatch_data.dispatch_info.agent_id);
       itr != _data->agent_counter_profiles.end() && itr->second)
        *config = *itr->second;
}

#if ROCPROFILER_VERSION >= 700
// ─── tool_hip_stream_callback ────────────────────────────────────────────────

template <typename Backend>
void
library_sdk<Backend>::tool_hip_stream_callback(
    typename Backend::callback_tracing_record record,
    typename Backend::user_data_t* /*user_data*/, void* /*data*/)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_HIP_STREAM) return;
    auto* stream_handle_data =
        static_cast<typename Backend::hip_stream_data*>(record.payload);
    auto stream_id = stream_handle_data->stream_id;

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
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
            stream_id_push(stream_id);
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
            stream_id_pop();
    }
    else
    {
        LOG_CRITICAL("Unknown operation for hip_stream_callback!");
        ::rocprofsys::set_state(::rocprofsys::State::Finalized);
        ::std::exit(1);
    }
}
#endif  // ROCPROFILER_VERSION >= 700

// ─── tool_init ───────────────────────────────────────────────────────────────

template <typename Backend>
int
library_sdk<Backend>::tool_init(typename Backend::client_finalize_t fini_func,
                                void*                               user_data)
{
    if(tool_init_done.exchange(true)) return 0;

    auto domains = settings::instance()->at(std::string{ env_vars::ROCM_DOMAINS });
    std::stringstream _domains_ss;
    for(const auto& itr : domains->get_choices())
        _domains_ss << "- " << itr << "\n";
    LOG_DEBUG("Available ROCm Domains: \n {}", _domains_ss.str());

    auto _callback_domains = sdk_core<Backend>::get_callback_domains();
    auto _buffered_domain  = sdk_core<Backend>::get_buffered_domains();
    auto _counter_events   = sdk_core<Backend>::get_rocm_events();
    auto _version          = sdk_core<Backend>::get_version();
    if(_version.formatted == 0) LOG_WARNING("rocprofiler-sdk version not initialized");

    auto* _data        = as_client_data<Backend>(user_data);
    _data->client_fini = fini_func;

    _data->initialize();
    if(!_counter_events.empty()) _data->initialize_event_info();

    ROCPROFILER_CALL(Backend::create_context(&_data->primary_ctx));
    ROCPROFILER_CALL(Backend::create_context(&_data->code_object_ctx));
    ROCPROFILER_CALL(Backend::configure_callback_tracing_service(
        _data->code_object_ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
        tool_code_object_callback, _data));
    ROCPROFILER_CALL(Backend::create_context(&_data->control_ctx));

    auto external_corr_id_request_kinds =
        std::array<typename Backend::external_correlation_request_kind, 3>{
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,
#if(ROCPROFILER_VERSION >= 600)
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION
#endif
        };

    {
        trace_cache::get_metadata_registry().add_stream(0);
        trace_cache::get_metadata_registry().add_queue(0);
    }

    for(auto itr : {
            ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,
            ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
            ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API,
            ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
            ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
            ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API,
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
            auto _ops = sdk_core<Backend>::get_operations(itr);
            _data->backtrace_operations.emplace(
                itr, sdk_core<Backend>::get_backtrace_operations(itr));
            ROCPROFILER_CALL(Backend::configure_callback_tracing_service(
                _data->primary_ctx, itr, _ops.data(), _ops.size(), tool_tracing_callback,
                _data));
        }
    }

    constexpr auto buffer_size = 16 * 4096;
    constexpr auto watermark   = 15 * 4096;

    ROCPROFILER_CALL(Backend::configure_external_correlation_id_request_service(
        _data->primary_ctx, external_corr_id_request_kinds.data(),
        external_corr_id_request_kinds.size(),
        set_kernel_rename_and_stream_correlation_id, _data));

#if(ROCPROFILER_VERSION >= 700)
    if((_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) > 0) ||
       (_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) > 0))
    {
        ROCPROFILER_CALL(Backend::configure_callback_tracing_service(
            _data->primary_ctx, ROCPROFILER_CALLBACK_TRACING_HIP_STREAM, nullptr, 0,
            tool_hip_stream_callback, nullptr));
    }
#endif

    if(_callback_domains.count(ROCPROFILER_CALLBACK_TRACING_RCCL_API) > 0)
        rocprofiler_sdk::rccl_comm_data_initialize();

    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) > 0)
    {
        ROCPROFILER_CALL(Backend::create_buffer(
            _data->primary_ctx, buffer_size, watermark, Backend::BUFFER_POLICY_LOSSLESS,
            tool_tracing_buffered, tool_data, &_data->kernel_dispatch_buffer));
        ROCPROFILER_CALL(Backend::configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0,
            _data->kernel_dispatch_buffer));
    }

    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) > 0)
    {
        ROCPROFILER_CALL(Backend::create_buffer(
            _data->primary_ctx, buffer_size, watermark, Backend::BUFFER_POLICY_LOSSLESS,
            tool_tracing_buffered, tool_data, &_data->memory_copy_buffer));
        ROCPROFILER_CALL(Backend::configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0,
            _data->memory_copy_buffer));
    }

    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY) > 0)
    {
        ROCPROFILER_CALL(Backend::create_buffer(
            _data->primary_ctx, buffer_size, watermark, Backend::BUFFER_POLICY_LOSSLESS,
            tool_tracing_buffered, tool_data, &_data->scratch_memory_buffer));
        ROCPROFILER_CALL(Backend::configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY, nullptr, 0,
            _data->scratch_memory_buffer));
    }

#if(ROCPROFILER_VERSION >= 600)
    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION) > 0)
    {
        ROCPROFILER_CALL(Backend::create_buffer(
            _data->primary_ctx, buffer_size, watermark, Backend::BUFFER_POLICY_LOSSLESS,
            tool_tracing_buffered, tool_data, &_data->memory_alloc_buffer));
        if(_data->memory_alloc_buffer.handle == 0UL)
        {
            LOG_CRITICAL("Failed to create memory allocation buffer");
            ::rocprofsys::set_state(::rocprofsys::State::Finalized);
            ::std::abort();
        }
        ROCPROFILER_CALL(Backend::configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION, nullptr, 0,
            _data->memory_alloc_buffer));
    }
#endif

#if(ROCPROFILER_VERSION >= 10000)
    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT) > 0 ||
       _buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE) > 0 ||
       _buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KFD_QUEUE) > 0 ||
       _buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE) > 0 ||
       _buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU) > 0 ||
       _buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS) > 0)
    {
        rocprofiler_sdk::kfd_event_metadata_initialize(tool_data);
    }

    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT) > 0)
    {
        ROCPROFILER_CALL(Backend::create_buffer(
            _data->primary_ctx, buffer_size, watermark, Backend::BUFFER_POLICY_LOSSLESS,
            tool_tracing_buffered, tool_data, &_data->kfd_page_fault_buffer));
        ROCPROFILER_CALL(Backend::configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT, nullptr, 0,
            _data->kfd_page_fault_buffer));
    }
    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE) > 0)
    {
        ROCPROFILER_CALL(Backend::create_buffer(
            _data->primary_ctx, buffer_size, watermark, Backend::BUFFER_POLICY_LOSSLESS,
            tool_tracing_buffered, tool_data, &_data->kfd_page_migrate_buffer));
        ROCPROFILER_CALL(Backend::configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE, nullptr, 0,
            _data->kfd_page_migrate_buffer));
    }
    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KFD_QUEUE) > 0)
    {
        ROCPROFILER_CALL(Backend::create_buffer(
            _data->primary_ctx, buffer_size, watermark, Backend::BUFFER_POLICY_LOSSLESS,
            tool_tracing_buffered, tool_data, &_data->kfd_queue_buffer));
        ROCPROFILER_CALL(Backend::configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_KFD_QUEUE, nullptr, 0,
            _data->kfd_queue_buffer));
    }
    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE) > 0)
    {
        ROCPROFILER_CALL(Backend::create_buffer(
            _data->primary_ctx, buffer_size, watermark, Backend::BUFFER_POLICY_LOSSLESS,
            tool_tracing_buffered, tool_data, &_data->kfd_event_queue_buffer));
        auto kfd_event_queue_ops = std::array<typename Backend::tracing_operation, 1>{
            ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED
        };
        ROCPROFILER_CALL(Backend::configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE,
            kfd_event_queue_ops.data(), kfd_event_queue_ops.size(),
            _data->kfd_event_queue_buffer));
    }
    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU) > 0)
    {
        ROCPROFILER_CALL(Backend::create_buffer(
            _data->primary_ctx, buffer_size, watermark, Backend::BUFFER_POLICY_LOSSLESS,
            tool_tracing_buffered, tool_data, &_data->kfd_event_unmap_buffer));
        ROCPROFILER_CALL(Backend::configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
            nullptr, 0, _data->kfd_event_unmap_buffer));
    }
    if(_buffered_domain.count(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS) > 0)
    {
        ROCPROFILER_CALL(Backend::create_buffer(
            _data->primary_ctx, buffer_size, watermark, Backend::BUFFER_POLICY_LOSSLESS,
            tool_tracing_buffered, tool_data, &_data->kfd_event_dropped_buffer));
        ROCPROFILER_CALL(Backend::configure_buffer_tracing_service(
            _data->primary_ctx, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS,
            nullptr, 0, _data->kfd_event_dropped_buffer));
    }
#endif

    if(!_counter_events.empty())
    {
        for(const auto& itr : _data->gpu_agents)
        {
            const auto& _agent_id = typename Backend::agent_id{ itr.agent->handle };
            _data->agent_events.emplace(
                _agent_id, create_agent_profile(_agent_id, _counter_events, _data));
        }

        ROCPROFILER_CALL(Backend::create_context(&_data->counter_ctx));
        auto _operations = std::array<typename Backend::tracing_operation, 1>{
            ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
        };
        ROCPROFILER_CALL(Backend::configure_callback_tracing_service(
            _data->counter_ctx, ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
            _operations.data(), _operations.size(), tool_tracing_callback, _data));
        ROCPROFILER_CALL(Backend::configure_callback_dispatch_counting_service(
            _data->counter_ctx, dispatch_counting_service_callback, _data,
            counter_record_callback, _data));
    }

#if ROCPROFILER_VERSION >= 600
    const auto gpu_perf_counters_setting = get_gpu_perf_counters();
    if(!gpu_perf_counters_setting.empty() && !_data->gpu_agents.empty())
    {
        pmc::register_gpu_perf_counter_source(
            get_agent_manager_instance().get_agents_by_type(agent_type::GPU));
    }
#endif

    for(const auto& itr : _data->get_buffers())
    {
        if(itr.handle > 0)
        {
            auto client_thread = typename Backend::callback_thread_id{};
            ROCPROFILER_CALL(Backend::create_callback_thread(&client_thread));
            ROCPROFILER_CALL(Backend::assign_callback_thread(itr, client_thread));
        }
    }

    if(!is_valid(_data->primary_ctx)) return -1;

    gpu::add_device_metadata();

    if(config::get_use_process_sampling())
    {
        LOG_DEBUG("Setting PMC sampler state to active...");
        pmc::set_state(State::Active);
    }

    auto roctx_client = get_roctx_client();
    if(roctx_client)
    {
        roctx_client->configure_services(_data->get_control_context());
        const auto filtering_active =
            roctx_client->get_controller()->region_filter_active();
        if(!filtering_active)
            start();
        else
        {
            if(_data != nullptr)
            {
                start_context(_data->get_code_obj_context());
                start_context(_data->get_control_context());
            }
        }
    }
    else
    {
        start();
    }
    return 0;
}

// ─── tool_fini ───────────────────────────────────────────────────────────────

template <typename Backend>
void
library_sdk<Backend>::tool_fini(void* callback_data)
{
    if(tool_fini_done.exchange(true)) return;

    flush();
    stop();
    finalize_sdk_common();

    for(auto itr : tool_data->get_buffers())
    {
        while(itr.handle > 0 &&
              Backend::destroy_buffer(itr) == ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
        {
            std::this_thread::yield();
        }
    }

    auto* _data        = as_client_data<Backend>(callback_data);
    _data->client_id   = nullptr;
    _data->client_fini = nullptr;
    delete tool_data;
    tool_data = nullptr;
}

// ─── tool_attach_* (version >= 10200) ────────────────────────────────────────

#if ROCPROFILER_VERSION >= 10200

template <typename Backend>
void
library_sdk<Backend>::tool_attach_fini(void* /*tool_data_ptr*/)
{
    ::rocprofsys::rocprofiler_sdk::stop();
    ::rocprofsys::rocprofiler_sdk::flush();
    finalize_sdk_common();
    rocprofsys_flush_pending_region_cache_hidden();

    if(get_use_perfetto())
    {
        bool                             _perfetto_output_error = false;
        rocprofsys::output_file_registry _output_registry{};
        ::rocprofsys::perfetto::post_process(nullptr, _perfetto_output_error,
                                             _output_registry);
        if(_perfetto_output_error)
            LOG_ERROR("Perfetto output error occurred during attach finalization");
    }
    rocprofsys_finalize_hidden();
}

template <typename Backend>
int
library_sdk<Backend>::tool_attach_init([[maybe_unused]]
                                       typename Backend::client_detach_t detach_func,
                                       typename Backend::context_id*     context_ids,
                                       std::uint64_t          context_ids_length,
                                       [[maybe_unused]] void* tool_attach_data)
{
    static std::atomic<int> attach_count{ 0 };
    auto                    current_count = attach_count.fetch_add(1);

    if(current_count > 0)
    {
        LOG_INFO("Re-attaching to process {} (session {})", getpid(), current_count);
        rocprofsys_reset_for_reattach_hidden();
        reset_sdk_session_guards();

        if(get_use_perfetto()) ::rocprofsys::perfetto::start();
        trace_cache::get_buffer_storage().start(getpid());

        if(config::get_use_process_sampling())
        {
            ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
            ::rocprofsys::process_sampler::setup();
        }
        ::rocprofsys::set_state(::rocprofsys::State::Active);
    }

    for(std::uint64_t i = 0; i < context_ids_length; ++i)
    {
        ROCPROFILER_CALL(Backend::start_context(context_ids[i]));
    }
    ::rocprofsys::rocprofiler_sdk::start();
    return 0;
}

#endif  // ROCPROFILER_VERSION >= 10200

// ─── sdk_tool_configure ──────────────────────────────────────────────────────

template <typename Backend>
bool
library_sdk<Backend>::sdk_tool_configure(std::uint32_t                  version,
                                         const char*                    runtime_version,
                                         typename Backend::client_id_t* id)
{
    if(sdk_configured.exchange(true)) return true;

    if(!rocprofsys::config::settings_are_configured() ||
       rocprofsys::get_state() < rocprofsys::State::Active)
    {
        rocprofsys_init_tooling_hidden();
    }

    id->name = "rocprofsys";

    if(!tool_data) tool_data = new client_data<Backend>{};
    tool_data->client_id = id;

    std::uint32_t major = version / 10000;
    std::uint32_t minor = (version % 10000) / 100;
    std::uint32_t patch = version % 100;

    LOG_INFO("{} is using rocprofiler-sdk v{}.{}.{} ({})", id->name, major, minor, patch,
             runtime_version);

    ROCPROFILER_CALL(Backend::at_internal_thread_create(
        &library_sdk::thread_precreate, &library_sdk::thread_postcreate,
        ROCPROFILER_LIBRARY | ROCPROFILER_HSA_LIBRARY | ROCPROFILER_HIP_LIBRARY |
            ROCPROFILER_MARKER_LIBRARY,
        nullptr));

    return true;
}

}  // namespace rocprofsys::rocprofiler_sdk
