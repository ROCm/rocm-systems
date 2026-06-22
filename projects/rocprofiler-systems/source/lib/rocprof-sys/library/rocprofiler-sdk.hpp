// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/wrapper.hpp"
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
// backends/rocprofiler_sdk/rocprofiler_sdk_wrapper.hpp (included via
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

template <typename Wrapper>
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

    static int  tool_init(typename Wrapper::client_finalize_t fini_func, void* user_data);
    static void tool_fini(void* callback_data);
    static bool sdk_tool_configure(std::uint32_t version, const char* runtime_version,
                                   typename Wrapper::client_id_t* id);

#if ROCPROFILER_VERSION >= 10200
    static int  tool_attach_init(typename Wrapper::client_detach_t detach_func,
                                 typename Wrapper::context_id*     context_ids,
                                 std::uint64_t context_ids_length, void* tool_attach_data);
    static void tool_attach_fini(void* tool_data_ptr);
#endif

    // ─── Static data members (formerly anon-namespace globals) ───────────────

    static client_data<Wrapper>*                                         tool_data;
    static std::shared_ptr<roctx_client<Wrapper, default_marker_policy>> g_roctx_client;
    static std::atomic<bool>                                             tool_fini_done;
    static std::atomic<bool>                                             tool_init_done;
    static std::atomic<bool>                                             sdk_configured;

private:
    // ─── Nested helper types ──────────────────────────────────────────────────

    struct kernel_rename_and_stream_data
    {
        std::uint64_t               region_id = 0;
        typename Wrapper::stream_id stream_id = { 0 };
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
        std::unordered_map<typename Wrapper::counter_id, counter_storage<Wrapper>>;
    using agent_counter_storage_map_t =
        std::unordered_map<typename Wrapper::agent_id, counter_storage_map_t>;

#if ROCPROFILER_VERSION >= 600
    struct rocprofsys_ompt_data_storage_t
    {
        typename Wrapper::callback_tracing_record record;
        typename Wrapper::timestamp_t             _beg_ts;
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

    static bool is_initialized(typename Wrapper::context_id ctx);
    static bool is_active(typename Wrapper::context_id ctx);
    static bool is_valid(typename Wrapper::context_id ctx);

    static void start_context(typename Wrapper::context_id ctx);
    static void stop_context(typename Wrapper::context_id ctx);
    static void start_context(const client_data<Wrapper>::context_id_vec_t& ctxs);
    static void stop_context(const client_data<Wrapper>::context_id_vec_t& ctxs);
    static void flush();

    // ─── roctx / counter / finalization helpers ───────────────────────────────

    static std::shared_ptr<roctx_client<Wrapper, default_marker_policy>>
                get_roctx_client();
    static void flush_counter_storage_outputs();
    static void flush_counter_tracks_to_zero(typename Wrapper::timestamp_t ts);
    static void finalize_sdk_common();

    // ─── Thread callbacks ─────────────────────────────────────────────────────

    static void thread_precreate(typename Wrapper::runtime_library_t lib, void* data);
    static void thread_postcreate(typename Wrapper::runtime_library_t lib, void* data);

    // ─── Stream helpers ───────────────────────────────────────────────────────

    static void stream_id_push(typename Wrapper::stream_id stream_id);
    static typename Wrapper::stream_id stream_id_top();
    static void                        stream_id_pop();

    template <typename Tp>
    static typename Wrapper::stream_id get_stream_id(Tp* record);

    // ─── Utility helpers ──────────────────────────────────────────────────────

    template <typename Tp, typename... Args>
    static Tp* as_pointer(Args&&... args);

    template <typename... Tp>
    static void consume_args(Tp&&...);

    static auto get_backtrace(
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data);

    template <typename CorrelationIdType>
    static std::uint64_t get_parent_stack_id(const CorrelationIdType& correlation_id);

    static const kernel_symbol_data_t<Wrapper>* get_kernel_symbol_info(
        std::uint64_t kernel_id);
    static const typename Wrapper::code_object_load_data* get_code_object_info(
        std::uint64_t code_object_id);

    static size_t get_mem_copy_dst_address(
        const typename Wrapper::memory_copy_record& record);
    static size_t get_mem_copy_src_address(
        const typename Wrapper::memory_copy_record& record);
#if ROCPROFILER_VERSION >= 600
    static size_t get_mem_alloc_address(
        const typename Wrapper::memory_alloc_record& record);
#endif
    static std::uint64_t get_scratch_mem_alloc_size(
        const typename Wrapper::scratch_memory_record& record);

    // ─── Argument iteration callbacks ────────────────────────────────────────

    static int save_args(typename Wrapper::callback_tracing_kind kind,
                         std::int32_t operation, std::uint32_t arg_number,
                         const void* const arg_value_addr,
                         std::int32_t arg_indirection_count, const char* arg_type,
                         const char* arg_name, const char* arg_value_str,
                         std::int32_t arg_dereference_count, void* data);

    static int iterate_args_callback(typename Wrapper::callback_tracing_kind kind,
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

    static void cache_region(const typename Wrapper::callback_tracing_record* record,
                             typename Wrapper::timestamp_t start_timestamp,
                             typename Wrapper::timestamp_t end_timestamp,
                             const std::string& call_stack, const std::string& args_str,
                             const std::string& category, std::string_view name = {});

    static void cache_kernel_dispatch(typename Wrapper::kernel_dispatch_record* record,
                                      std::uint64_t stream_handle);
    static void cache_scratch_memory(typename Wrapper::scratch_memory_record* record,
                                     std::uint64_t stream_handle);
    static void cache_memory_copy(typename Wrapper::memory_copy_record* record,
                                  std::uint64_t                         stream_handle);
#if ROCPROFILER_VERSION >= 600
    static void cache_memory_allocation(typename Wrapper::memory_alloc_record* record,
                                        std::uint64_t stream_handle);
#endif

    // ─── Profiling configuration ─────────────────────────────────────────────

    static std::vector<typename Wrapper::counter_id> create_agent_profile(
        typename Wrapper::agent_id agent_id, const std::vector<std::string>& counters,
        client_data<Wrapper>* data = nullptr);

    static int set_kernel_rename_and_stream_correlation_id(
        typename Wrapper::thread_id thr_id, typename Wrapper::context_id ctx_id,
        typename Wrapper::external_correlation_request_kind kind,
        typename Wrapper::tracing_operation op, std::uint64_t internal_corr_id,
        typename Wrapper::user_data_t* external_corr_id, void* user_data);

    // ─── Tracing callbacks ────────────────────────────────────────────────────

    template <typename CategoryT>
    static void tool_tracing_callback_start(
        CategoryT, typename Wrapper::callback_tracing_record record,
        typename Wrapper::user_data_t* user_data, typename Wrapper::timestamp_t ts);

    template <typename CategoryT>
    static void tool_tracing_callback_stop(
        CategoryT, typename Wrapper::callback_tracing_record record,
        typename Wrapper::user_data_t* user_data, typename Wrapper::timestamp_t ts,
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data);

    static void tool_code_object_callback(
        typename Wrapper::callback_tracing_record record,
        typename Wrapper::user_data_t* user_data, void* callback_data);

    static void tool_tracing_callback(typename Wrapper::callback_tracing_record record,
                                      typename Wrapper::user_data_t*            user_data,
                                      void* callback_data);

    static void tool_tracing_buffered(typename Wrapper::context_id        context,
                                      typename Wrapper::buffer_id         buffer_id,
                                      typename Wrapper::record_header_t** headers,
                                      size_t num_headers, void* user_data,
                                      std::uint64_t drop_count);

    static void counter_record_callback(
        typename Wrapper::dispatch_counting_data dispatch_data,
        typename Wrapper::counter_record* record_data, size_t record_count,
        typename Wrapper::user_data_t user_data, void* callback_data_arg);

    static void dispatch_counting_service_callback(
        typename Wrapper::dispatch_counting_data dispatch_data,
        typename Wrapper::counter_config_id*     config,
        typename Wrapper::user_data_t* user_data, void* callback_data_arg);

#if ROCPROFILER_VERSION >= 600
    // ─── OMPT helpers ────────────────────────────────────────────────────────

    static std::string_view ompt_get_unified_name(
        const typename Wrapper::callback_tracing_record& record);

    template <typename ArgsT>
    static void ompt_iterate_operation_args(
        const typename Wrapper::callback_tracing_record& record, ArgsT& args);

    static void ompt_cache_instant_event(
        typename Wrapper::callback_tracing_record                 record,
        typename Wrapper::timestamp_t                             instant_ts,
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data);

    static void ompt_cache_orphan_event(
        const rocprofsys_ompt_data_storage_t&                     stored_data,
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data);

    static void ompt_push_standard_callback(
        const typename Wrapper::callback_tracing_record& record,
        const typename Wrapper::timestamp_t&             beg_ts);

    static void ompt_pop_standard_callback(
        const typename Wrapper::callback_tracing_record&          record,
        const typename Wrapper::timestamp_t&                      end_ts,
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data);

    static void ompt_push_parallel_callback(
        const typename Wrapper::callback_tracing_record& record,
        const typename Wrapper::timestamp_t&             beg_ts);

    static void ompt_pop_parallel_callback(
        const typename Wrapper::callback_tracing_record&          record,
        const typename Wrapper::timestamp_t&                      end_ts,
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data);

    static void ompt_tracing_callback_start(
        typename Wrapper::callback_tracing_record record,
        typename Wrapper::user_data_t* user_data, typename Wrapper::timestamp_t ts);

    static void ompt_tracing_callback_stop(
        typename Wrapper::callback_tracing_record record,
        typename Wrapper::user_data_t* user_data, typename Wrapper::timestamp_t ts,
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data);

    static void ompt_finalize_orphan_events();
#endif

#if ROCPROFILER_VERSION >= 700
    static void tool_hip_stream_callback(typename Wrapper::callback_tracing_record record,
                                         typename Wrapper::user_data_t* user_data,
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

template <typename Wrapper>
client_data<Wrapper>* library_sdk<Wrapper>::tool_data = new client_data<Wrapper>{};

template <typename Wrapper>
std::shared_ptr<roctx_client<Wrapper, default_marker_policy>>
    library_sdk<Wrapper>::g_roctx_client = {};

template <typename Wrapper>
std::atomic<bool> library_sdk<Wrapper>::tool_fini_done{ false };

template <typename Wrapper>
std::atomic<bool> library_sdk<Wrapper>::tool_init_done{ false };

template <typename Wrapper>
std::atomic<bool> library_sdk<Wrapper>::sdk_configured{ false };

// ─── scope_destructor ────────────────────────────────────────────────────────

template <typename Wrapper>
template <typename FuncT, typename InitT>
library_sdk<Wrapper>::scope_destructor::scope_destructor(FuncT&& _fini, InitT&& _init)
: m_functor{ std::forward<FuncT>(_fini) }
{
    _init();
}

template <typename Wrapper>
library_sdk<Wrapper>::scope_destructor::scope_destructor(scope_destructor&& rhs) noexcept
: m_functor{ std::move(rhs.m_functor) }
{
    rhs.m_functor = []() {};
}

template <typename Wrapper>
typename library_sdk<Wrapper>::scope_destructor&
library_sdk<Wrapper>::scope_destructor::operator=(scope_destructor&& rhs) noexcept
{
    if(this != &rhs)
    {
        m_functor     = std::move(rhs.m_functor);
        rhs.m_functor = []() {};
    }
    return *this;
}

// ─── Utility helpers ─────────────────────────────────────────────────────────

template <typename Wrapper>
template <typename Tp, typename... Args>
Tp*
library_sdk<Wrapper>::as_pointer(Args&&... args)
{
    return new Tp{ std::forward<Args>(args)... };
}

template <typename Wrapper>
template <typename... Tp>
void
library_sdk<Wrapper>::consume_args(Tp&&...)
{}

// ─── Thread-local state ───────────────────────────────────────────────────────

template <typename Wrapper>
typename library_sdk<Wrapper>::kernel_rename_stack_t*&
library_sdk<Wrapper>::get_thread_dispatch_rename()
{
    static thread_local auto* _v    = as_pointer<kernel_rename_stack_t>();
    static thread_local auto  _dtor = scope_destructor{ []() {
        delete _v;
        _v                          = nullptr;
    } };
    return _v;
}

template <typename Wrapper>
auto&
library_sdk<Wrapper>::get_stream_stack()
{
    static thread_local std::vector<typename Wrapper::stream_id> _v{
        typename Wrapper::stream_id{ 0 }
    };
    return _v;
}

// ─── Stream helpers ───────────────────────────────────────────────────────────

template <typename Wrapper>
void
library_sdk<Wrapper>::stream_id_push(typename Wrapper::stream_id stream_id)
{
    get_stream_stack().emplace_back(stream_id);
}

template <typename Wrapper>
typename Wrapper::stream_id
library_sdk<Wrapper>::stream_id_top()
{
    return get_stream_stack().back();
}

template <typename Wrapper>
void
library_sdk<Wrapper>::stream_id_pop()
{
    get_stream_stack().pop_back();
}

template <typename Wrapper>
template <typename Tp>
typename Wrapper::stream_id
library_sdk<Wrapper>::get_stream_id(Tp* record)
{
    auto _stream_id = typename Wrapper::stream_id{ 0 };
    if(record->correlation_id.external.ptr != nullptr)
    {
        // Extract the stream id
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

template <typename Wrapper>
template <typename CorrelationIdType>
std::uint64_t
library_sdk<Wrapper>::get_parent_stack_id(
    [[maybe_unused]] const CorrelationIdType& correlation_id)
{
#if(ROCPROFILER_VERSION >= 700)
    if constexpr(std::is_same_v<typename Wrapper::correlation_id_t, CorrelationIdType>)
        return correlation_id.ancestor;
    else
        return 0;
#else
    return 0;
#endif
}

// ─── thread_precreate / thread_postcreate ────────────────────────────────────

template <typename Wrapper>
void
library_sdk<Wrapper>::thread_precreate(typename Wrapper::runtime_library_t /*lib*/,
                                       void* /*data*/)
{
    push_thread_state(ThreadState::Internal);
}

template <typename Wrapper>
void
library_sdk<Wrapper>::thread_postcreate(typename Wrapper::runtime_library_t /*lib*/,
                                        void* /*data*/)
{
    pop_thread_state();
}

// ─── Context helpers ─────────────────────────────────────────────────────────

template <typename Wrapper>
bool
library_sdk<Wrapper>::is_initialized(typename Wrapper::context_id ctx)
{
    typename Wrapper::context_id ctx_default;
    return ctx != ctx_default;
}

template <typename Wrapper>
bool
library_sdk<Wrapper>::is_active(typename Wrapper::context_id ctx)
{
    int  status = 0;
    auto errc   = Wrapper::context_is_active(ctx, &status);
    return (errc == Wrapper::STATUS_SUCCESS && status > 0);
}

template <typename Wrapper>
bool
library_sdk<Wrapper>::is_valid(typename Wrapper::context_id ctx)
{
    int  status = 0;
    auto errc   = Wrapper::context_is_valid(ctx, &status);
    return (errc == Wrapper::STATUS_SUCCESS && status > 0);
}

template <typename Wrapper>
void
library_sdk<Wrapper>::start_context(typename Wrapper::context_id ctx)
{
    if(is_initialized(ctx) && !is_active(ctx))
    {
        rocprofiler_call<Wrapper>(Wrapper::start_context(ctx), "Wrapper::start_context");
    }
}

template <typename Wrapper>
void
library_sdk<Wrapper>::stop_context(typename Wrapper::context_id ctx)
{
    if(is_initialized(ctx) && is_active(ctx))
    {
        rocprofiler_call<Wrapper>(Wrapper::stop_context(ctx), "Wrapper::stop_context");
    }
}

template <typename Wrapper>
void
library_sdk<Wrapper>::start_context(const client_data<Wrapper>::context_id_vec_t& ctxs)
{
    std::for_each(std::begin(ctxs), std::end(ctxs),
                  [](const auto& ctx) { start_context(ctx); });
}

template <typename Wrapper>
void
library_sdk<Wrapper>::stop_context(const client_data<Wrapper>::context_id_vec_t& ctxs)
{
    std::for_each(std::begin(ctxs), std::end(ctxs),
                  [](const auto& ctx) { stop_context(ctx); });
}

template <typename Wrapper>
void
library_sdk<Wrapper>::flush()
{
    if(!tool_data) return;

    for(const auto& itr : tool_data->get_buffers())
    {
        if(itr.handle > 0)
        {
            auto status = Wrapper::flush_buffer(itr);
            if(status != Wrapper::STATUS_ERROR_BUFFER_BUSY)
            {
                rocprofiler_call<Wrapper>(status, "Wrapper::flush_buffer");
            }
        }
    }
}

// ─── roctx helper ────────────────────────────────────────────────────────────

template <typename Wrapper>
std::shared_ptr<roctx_client<Wrapper, default_marker_policy>>
library_sdk<Wrapper>::get_roctx_client()
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

        // Case 1: no marker domain and no trace regions — nothing to do
        if(!has_marker_domain && !has_trace_regions)
        {
            return nullptr;
        }

        const auto roctx_config = roctx_client_config{
            has_marker_domain,          config::get_use_perfetto(),
            config::get_use_timemory(), config::get_perfetto_annotations(),
            roctx_traced_regions,
        };
        g_roctx_client =
            std::make_shared<roctx_client<Wrapper, default_marker_policy>>(roctx_config);
    }
    return g_roctx_client;
}

// ─── Public API ──────────────────────────────────────────────────────────────

template <typename Wrapper>
void
library_sdk<Wrapper>::setup()
{}

template <typename Wrapper>
void
library_sdk<Wrapper>::shutdown()
{
    auto roctx_client = get_roctx_client();
    // Shutdown marker client (and trace_control) before rocprofiler-sdk finalization
    if(roctx_client)
    {
        roctx_client->get_controller()->shutdown();
    }

    // shutdown
    if(tool_data && tool_data->client_id && tool_data->client_fini)
        tool_data->client_fini(*tool_data->client_id);
}

template <typename Wrapper>
void
library_sdk<Wrapper>::config()
{}

template <typename Wrapper>
void
library_sdk<Wrapper>::post_process()
{}

template <typename Wrapper>
void
library_sdk<Wrapper>::sample()
{}

template <typename Wrapper>
void
library_sdk<Wrapper>::start()
{
    if(!tool_data) return;
    start_context(tool_data->get_all_contexts());
}

template <typename Wrapper>
void
library_sdk<Wrapper>::stop()
{
    if(!tool_data) return;
    stop_context(tool_data->get_all_contexts());
}

template <typename Wrapper>
void
library_sdk<Wrapper>::resume()
{
    flush_counter_tracks_to_zero(0);
    if(!tool_data) return;
    start_context(tool_data->get_main_contexts());
}

template <typename Wrapper>
void
library_sdk<Wrapper>::pause()
{
    if(!tool_data) return;
    stop_context(tool_data->get_main_contexts());
    flush_counter_tracks_to_zero(0);
}

template <typename Wrapper>
std::shared_ptr<control::trace_control>
library_sdk<Wrapper>::get_trace_controller()
{
    const auto roctx_client = get_roctx_client();
    if(!roctx_client) return nullptr;
    return roctx_client->get_controller();
}

template <typename Wrapper>
void
library_sdk<Wrapper>::reset_sdk_session_guards()
{
    tool_fini_done.store(false);
    tool_init_done.store(false);
}

template <typename Wrapper>
std::vector<hardware_counter_info>
library_sdk<Wrapper>::get_rocm_events_info()
{
    if(!tool_data)
    {
        auto _tool_data_v = client_data<Wrapper>{};
        _tool_data_v.initialize_event_info();
        return _tool_data_v.events_info;
    }
    if(tool_data->events_info.empty()) tool_data->initialize_event_info();
    return tool_data->events_info;
}

// ─── Argument iteration callbacks ────────────────────────────────────────────

template <typename Wrapper>
int
library_sdk<Wrapper>::save_args(typename Wrapper::callback_tracing_kind /*kind*/,
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

template <typename Wrapper>
int
library_sdk<Wrapper>::iterate_args_callback(
    typename Wrapper::callback_tracing_kind /*kind*/, std::int32_t /*operation*/,
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

template <typename Wrapper>
template <typename Category>
void
library_sdk<Wrapper>::cache_category()
{
    trace_cache::get_metadata_registry().add_string(trait::name<Category>::value);
}

template <typename Wrapper>
void
library_sdk<Wrapper>::cache_add_thread_info(std::uint64_t tid)
{
    trace_cache::get_metadata_registry().add_thread_info(
        { getppid(), getpid(), tid, 0, 0, "{}" });
}

template <typename Wrapper>
void
library_sdk<Wrapper>::cache_add_track(const char* track_name, std::uint64_t tid)
{
    trace_cache::get_metadata_registry().add_track({ track_name, tid, "{}" });
}

template <typename Wrapper>
void
library_sdk<Wrapper>::cache_region(
    const typename Wrapper::callback_tracing_record* record,
    typename Wrapper::timestamp_t                    start_timestamp,
    typename Wrapper::timestamp_t end_timestamp, const std::string& call_stack,
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

template <typename Wrapper>
void
library_sdk<Wrapper>::cache_kernel_dispatch(
    typename Wrapper::kernel_dispatch_record* record, std::uint64_t stream_handle)
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

template <typename Wrapper>
void
library_sdk<Wrapper>::cache_scratch_memory(
    typename Wrapper::scratch_memory_record* record, std::uint64_t stream_handle)
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

template <typename Wrapper>
void
library_sdk<Wrapper>::cache_memory_copy(typename Wrapper::memory_copy_record* record,
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
template <typename Wrapper>
void
library_sdk<Wrapper>::cache_memory_allocation(
    typename Wrapper::memory_alloc_record* record, std::uint64_t stream_handle)
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

template <typename Wrapper>
size_t
library_sdk<Wrapper>::get_mem_copy_dst_address(
    [[maybe_unused]] const typename Wrapper::memory_copy_record& record)
{
#if(ROCPROFILER_VERSION >= 700)
    return record.dst_address.value;
#else
    return 0;
#endif
}

template <typename Wrapper>
size_t
library_sdk<Wrapper>::get_mem_copy_src_address(
    [[maybe_unused]] const typename Wrapper::memory_copy_record& record)
{
#if(ROCPROFILER_VERSION >= 700)
    return record.src_address.value;
#else
    return 0;
#endif
}

#if ROCPROFILER_VERSION >= 600
template <typename Wrapper>
size_t
library_sdk<Wrapper>::get_mem_alloc_address(
    [[maybe_unused]] const typename Wrapper::memory_alloc_record& record)
{
#    if(ROCPROFILER_VERSION >= 700)
    return record.address.value;
#    else
    return static_cast<size_t>(record.address.handle);
#    endif
}
#endif

template <typename Wrapper>
std::uint64_t
library_sdk<Wrapper>::get_scratch_mem_alloc_size(
    [[maybe_unused]] const typename Wrapper::scratch_memory_record& record)
{
// The version of rocprofiler_buffer_tracing_scratch_memory_record_t from ROCm < 7.1 does
// not have the allocation_size field. ROCPROFILER_VERSION for both ROCm 7.0 and 7.1
// is 1.0.0, so we need to check the ROCm version.
#if ROCPROFSYS_ROCM_VERSION >= 70100
    return record.allocation_size;
#else
    return 0;
#endif
}

// ─── Kernel / code object info lookups ───────────────────────────────────────

template <typename Wrapper>
const kernel_symbol_data_t<Wrapper>*
library_sdk<Wrapper>::get_kernel_symbol_info(std::uint64_t kernel_id)
{
    return tool_data->get_kernel_symbol_info(kernel_id);
}

template <typename Wrapper>
const typename Wrapper::code_object_load_data*
library_sdk<Wrapper>::get_code_object_info(std::uint64_t code_object_id)
{
    return tool_data->get_code_object_info(code_object_id);
}

// ─── Backtrace helper ─────────────────────────────────────────────────────────

template <typename Wrapper>
auto
library_sdk<Wrapper>::get_backtrace(
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

template <typename Wrapper>
auto&
library_sdk<Wrapper>::get_counter_dispatch_data()
{
    static auto _v = container::stable_vector<typename Wrapper::dispatch_counting_data>{};
    return _v;
}

template <typename Wrapper>
auto&
library_sdk<Wrapper>::get_counter_dispatch_records()
{
    static auto _v = std::vector<counter_dispatch_record<Wrapper>>{};
    return _v;
}

template <typename Wrapper>
typename library_sdk<Wrapper>::agent_counter_storage_map_t*&
library_sdk<Wrapper>::get_counter_storage()
{
    static auto* _v = new agent_counter_storage_map_t{};
    return _v;
}

template <typename Wrapper>
auto&
library_sdk<Wrapper>::get_kernel_dispatch_timestamps()
{
    static auto _v =
        std::unordered_map<typename Wrapper::dispatch_id_t, timing_interval<Wrapper>>{};
    return _v;
}

template <typename Wrapper>
void
library_sdk<Wrapper>::flush_counter_storage_outputs()
{
    auto* _agent_counter_storage = get_counter_storage();
    if(!_agent_counter_storage) return;

    auto _cleanup_keys =
        std::vector<std::pair<std::string, const counter_storage<Wrapper>*>>{};
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
        {
            storage->manager->cleanup(cleanup_key);
        }
        else
        {
            // No tim::manager at construction (add_cleanup was skipped); still flush
            // timemory storage once at shutdown.
            counter_storage<Wrapper>::write(storage->storage.get(), storage->metric_name,
                                            storage->metric_description);
        }
    }
}

template <typename Wrapper>
void
library_sdk<Wrapper>::flush_counter_tracks_to_zero(
    typename Wrapper::timestamp_t timestamp)
{
    if(timestamp == 0)
    {
        rocprofiler_call<Wrapper>(Wrapper::get_timestamp(&timestamp),
                                  "Wrapper::get_timestamp");
    }
    auto* storage = get_counter_storage();
    if(!storage) return;
    for(auto& [agent_id, counter_map] : *storage)
        for(auto& [counter_id, cs] : counter_map)
            cs.write_zero(timestamp);
}

template <typename Wrapper>
void
library_sdk<Wrapper>::finalize_sdk_common()
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

template <typename Wrapper>
std::vector<typename Wrapper::counter_id>
library_sdk<Wrapper>::create_agent_profile(
    typename Wrapper::agent_id agent_id, const std::vector<std::string>& counters,
    //  const tool_agent_vec_t&         gpu_agents,
    //  const agent_counter_info_map_t& counters_info,
    //  agent_counter_profile_map_t&    data)
    client_data<Wrapper>* data)
{
    using counter_vec_t = std::vector<typename Wrapper::counter_id>;
    if(!data) data = tool_data;

    // check if already created
    if(data->agent_counter_profiles.find(agent_id) != data->agent_counter_profiles.end())
        return counter_vec_t{};

    auto        profile      = std::optional<typename Wrapper::counter_config_id>{};
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
                ::rocprofsys::set_state(::rocprofsys::State::Finalized);
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

        // Removes any numeric index enclosed in square brackets at the end of the
        // string. For example, "example[123]" will be converted to "example".
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

        LOG_WARNING("Unable to find all counters for agent {} (gpu-{}, {}). "
                    "Requested: {}. Found: {}. Missing: {}. Continuing with "
                    "available counters.",
                    tool_agent_v->agent->node_id, tool_agent_v->device_id,
                    tool_agent_v->agent->name, requested_counters, found_counters,
                    missing_counters_str);
    }

    if(!counters_v.empty())
    {
        auto profile_v = typename Wrapper::counter_config_id{};
        rocprofiler_call<Wrapper>(
            Wrapper::create_counter_config(agent_id, counters_v.data(), counters_v.size(),
                                           &profile_v),
            "Wrapper::create_counter_config");
        profile = profile_v;
    }

    data->agent_counter_profiles.emplace(agent_id, profile);
    return counters_v;
}

// ─── External correlation ID callback ────────────────────────────────────────

template <typename Wrapper>
int
library_sdk<Wrapper>::set_kernel_rename_and_stream_correlation_id(
    typename Wrapper::thread_id /*thr_id*/, typename Wrapper::context_id /*ctx_id*/,
    typename Wrapper::external_correlation_request_kind /*kind*/,
    typename Wrapper::tracing_operation /*op*/, std::uint64_t /*internal_corr_id*/,
    typename Wrapper::user_data_t* external_corr_id, void* /*user_data*/)
{
    auto* _info = new kernel_rename_and_stream_data{};

    _info->stream_id = stream_id_top();

    // Set the external correlation id service to point to struct
    external_corr_id->ptr = _info;
    return 0;
}

// ─── tool_code_object_callback ───────────────────────────────────────────────

template <typename Wrapper>
void
library_sdk<Wrapper>::tool_code_object_callback(
    typename Wrapper::callback_tracing_record record,
    typename Wrapper::user_data_t* /*user_data*/, void* /*callback_data*/)
{
    auto ts = typename Wrapper::timestamp_t{};
    rocprofiler_call<Wrapper>(Wrapper::get_timestamp(&ts), "Wrapper::get_timestamp");

    if(record.kind == Wrapper::CALLBACK_TRACING_CODE_OBJECT)
    {
        if(record.phase == Wrapper::CALLBACK_PHASE_ENTER)
        {
            if(record.operation == Wrapper::CODE_OBJECT_LOAD)
            {
                auto data_v = *static_cast<typename Wrapper::code_object_load_data*>(
                    record.payload);
                tool_data->code_object_records.wlock([ts, &record, &data_v](auto& _data) {
                    _data.emplace_back(
                        code_object_callback_record_t<Wrapper>{ ts, record, data_v });
                });
                // Insert callback trace into database
                trace_cache::get_metadata_registry().add_code_object(data_v);
            }
            else if(record.operation ==
                    Wrapper::CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
            {
                auto data_v =
                    *static_cast<kernel_symbol_data_t<Wrapper>*>(record.payload);
                tool_data->kernel_symbol_records.wlock([ts, &record,
                                                        &data_v](auto& _data) {
                    _data.emplace_back(
                        kernel_symbol_callback_record_t<Wrapper>{ ts, record, data_v });
                });
                trace_cache::get_metadata_registry().add_kernel_symbol(data_v);
            }
        }
    }
}

// ─── tool_tracing_callback_start / stop ──────────────────────────────────────

template <typename Wrapper>
template <typename CategoryT>
void
library_sdk<Wrapper>::tool_tracing_callback_start(
    CategoryT, typename Wrapper::callback_tracing_record record,
    typename Wrapper::user_data_t* /*user_data*/, typename Wrapper::timestamp_t /*ts*/)
{
    auto _name = tool_data->callback_tracing_info.at(record.kind, record.operation);
    if(get_use_timemory()) tracing::push_timemory(CategoryT{}, _name);
}

template <typename Wrapper>
template <typename CategoryT>
void
library_sdk<Wrapper>::tool_tracing_callback_stop(
    CategoryT, typename Wrapper::callback_tracing_record record,
    typename Wrapper::user_data_t* user_data, typename Wrapper::timestamp_t ts,
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
            Wrapper::iterate_callback_tracing_kind_operation_args(record, save_args, 2,
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
    Wrapper::iterate_callback_tracing_kind_operation_args(record, iterate_args_callback,
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

template <typename Wrapper>
void
library_sdk<Wrapper>::tool_tracing_callback(
    typename Wrapper::callback_tracing_record record,
    typename Wrapper::user_data_t*            user_data, void* /*callback_data*/)
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
    // Skip implicit_task associated with an "initial-task-begin" occurrence as
    // well as the thread_begin associated with an "initial-thread-begin" occurrence
    // as they are generated by our tool.
    // The two callbacks occur after our tool initializes OMPT but before the
    // first OpenMP region (user code) begins.
    // Note: Can occur multiple times (Ex: MPI+OpenMP hybrid)
    if(record.kind == Wrapper::CALLBACK_TRACING_OMPT)
    {
        auto* payload_data = static_cast<typename Wrapper::ompt_data_t*>(record.payload);
        if(!payload_data) return;
        switch(record.operation)
        {
            case Wrapper::OMPT_ID_implicit_task:
            {
                int flag = payload_data->args.implicit_task.flags;
                if(flag & ompt_task_initial) return;  // Skips both the start and end
                break;
            }
            case Wrapper::OMPT_ID_thread_begin:
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
        //    by the "initial-thread-begin" needs to match the thread_end's
        //    thread_data ptr
    }
#endif

    auto ts = typename Wrapper::timestamp_t{};
    rocprofiler_call<Wrapper>(Wrapper::get_timestamp(&ts), "Wrapper::get_timestamp");
    const char* name = "";
    Wrapper::query_callback_op_name(record.kind, record.operation, &name, nullptr);

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

    if(record.phase == Wrapper::CALLBACK_PHASE_ENTER)
    {
        user_data->value = ts;
        switch(record.kind)
        {
            case Wrapper::CALLBACK_TRACING_HSA_CORE_API:
            case Wrapper::CALLBACK_TRACING_HSA_AMD_EXT_API:
            case Wrapper::CALLBACK_TRACING_HSA_IMAGE_EXT_API:
            case Wrapper::CALLBACK_TRACING_HSA_FINALIZE_EXT_API:
                tool_tracing_callback_start(category::rocm_hsa_api{}, record, user_data,
                                            ts);
                break;
            case Wrapper::CALLBACK_TRACING_HIP_RUNTIME_API:
            case Wrapper::CALLBACK_TRACING_HIP_COMPILER_API:
                tool_tracing_callback_start(category::rocm_hip_api{}, record, user_data,
                                            ts);
                break;
#if(ROCPROFILER_VERSION >= 600)
            case Wrapper::CALLBACK_TRACING_OMPT:
                ompt_tracing_callback_start(record, user_data, ts);
                ompt_push_standard_callback(record, ts);
                break;
            case Wrapper::CALLBACK_TRACING_ROCDECODE_API:
                tool_tracing_callback_start(category::rocm_rocdecode_api{}, record,
                                            user_data, ts);
                break;
#endif
#if(ROCPROFILER_VERSION >= 700)
            case Wrapper::CALLBACK_TRACING_ROCJPEG_API:
                tool_tracing_callback_start(category::rocm_rocjpeg_api{}, record,
                                            user_data, ts);
                break;
#endif
            case Wrapper::CALLBACK_TRACING_RCCL_API:
                tool_tracing_callback_start(category::rocm_rccl_api{}, record, user_data,
                                            ts);
                break;
            // MARKER_CORE_API is handled by roctx_client on control_ctx
            case Wrapper::CALLBACK_TRACING_NONE:
            case Wrapper::CALLBACK_TRACING_LAST:
            case Wrapper::CALLBACK_TRACING_MARKER_CONTROL_API:
            case Wrapper::CALLBACK_TRACING_MARKER_CORE_API:
            case Wrapper::CALLBACK_TRACING_MARKER_NAME_API:
            case Wrapper::CALLBACK_TRACING_CODE_OBJECT:
            case Wrapper::CALLBACK_TRACING_SCRATCH_MEMORY:
            case Wrapper::CALLBACK_TRACING_KERNEL_DISPATCH:
            case Wrapper::CALLBACK_TRACING_MEMORY_COPY:
#if(ROCPROFILER_VERSION >= 600)
            case Wrapper::CALLBACK_TRACING_MEMORY_ALLOCATION:
            case Wrapper::CALLBACK_TRACING_RUNTIME_INITIALIZATION:
#endif
#if(ROCPROFILER_VERSION >= 700)
            case Wrapper::CALLBACK_TRACING_HIP_STREAM:
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
    else if(record.phase == Wrapper::CALLBACK_PHASE_EXIT)
    {
        populate_backtrace_data();
        switch(record.kind)
        {
            case Wrapper::CALLBACK_TRACING_HSA_CORE_API:
            case Wrapper::CALLBACK_TRACING_HSA_AMD_EXT_API:
            case Wrapper::CALLBACK_TRACING_HSA_IMAGE_EXT_API:
            case Wrapper::CALLBACK_TRACING_HSA_FINALIZE_EXT_API:
                tool_tracing_callback_stop(category::rocm_hsa_api{}, record, user_data,
                                           ts, _bt_data);
                break;
            case Wrapper::CALLBACK_TRACING_HIP_RUNTIME_API:
            case Wrapper::CALLBACK_TRACING_HIP_COMPILER_API:
                tool_tracing_callback_stop(category::rocm_hip_api{}, record, user_data,
                                           ts, _bt_data);
                break;
#if(ROCPROFILER_VERSION >= 600)
            case Wrapper::CALLBACK_TRACING_OMPT:
                ompt_tracing_callback_stop(record, user_data, ts, _bt_data);
                ompt_pop_standard_callback(record, ts, _bt_data);
                break;
            case Wrapper::CALLBACK_TRACING_ROCDECODE_API:
                tool_tracing_callback_stop(category::rocm_rocdecode_api{}, record,
                                           user_data, ts, _bt_data);
                break;
#endif
#if(ROCPROFILER_VERSION >= 700)
            case Wrapper::CALLBACK_TRACING_ROCJPEG_API:
                tool_tracing_callback_stop(category::rocm_rocjpeg_api{}, record,
                                           user_data, ts, _bt_data);
                break;
#endif
            case Wrapper::CALLBACK_TRACING_RCCL_API:
            {
                auto* rccl_payload =
                    static_cast<typename Wrapper::rccl_api_data*>(record.payload);
                tool_tracing_callback_rccl<Wrapper>(record.operation, rccl_payload,
                                                    user_data->value, ts);
                tool_tracing_callback_stop(category::rocm_rccl_api{}, record, user_data,
                                           ts, _bt_data);
                break;
            }
            case Wrapper::CALLBACK_TRACING_NONE:
            case Wrapper::CALLBACK_TRACING_LAST:
            case Wrapper::CALLBACK_TRACING_MARKER_CONTROL_API:
            case Wrapper::CALLBACK_TRACING_MARKER_CORE_API:
            case Wrapper::CALLBACK_TRACING_MARKER_NAME_API:
            case Wrapper::CALLBACK_TRACING_CODE_OBJECT:
            case Wrapper::CALLBACK_TRACING_SCRATCH_MEMORY:
            case Wrapper::CALLBACK_TRACING_KERNEL_DISPATCH:
            case Wrapper::CALLBACK_TRACING_MEMORY_COPY:
#if(ROCPROFILER_VERSION >= 600)
            case Wrapper::CALLBACK_TRACING_MEMORY_ALLOCATION:
            case Wrapper::CALLBACK_TRACING_RUNTIME_INITIALIZATION:
#endif
#if(ROCPROFILER_VERSION >= 700)
            case Wrapper::CALLBACK_TRACING_HIP_STREAM:
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
    else if(record.phase == Wrapper::CALLBACK_PHASE_NONE)
    {
        switch(record.kind)
        {
            case Wrapper::CALLBACK_TRACING_KERNEL_DISPATCH:
                if(record.operation == Wrapper::KERNEL_DISPATCH_COMPLETE)
                {
                    auto* _data = static_cast<typename Wrapper::kernel_dispatch_data*>(
                        record.payload);
                    // save for post-processing
                    get_kernel_dispatch_timestamps().emplace(
                        _data->dispatch_info.dispatch_id,
                        timing_interval<Wrapper>{ _data->start_timestamp,
                                                  _data->end_timestamp });
                }
                break;
#if(ROCPROFILER_VERSION >= 600)
            case Wrapper::CALLBACK_TRACING_OMPT:
            {
                // Callbacks that are received but that we do not process
                static const std::set<typename Wrapper::ompt_operation_t>
                    ompt_no_process = {
                        Wrapper::OMPT_ID_callback_functions,  // "Fake" callback
                        // Not processed as these are received after our tool
                        // finalizes
                        Wrapper::OMPT_ID_thread_end,
                    };
                auto ompt_operation_type =
                    static_cast<typename Wrapper::ompt_operation_t>(record.operation);
                if(ompt_no_process.find(ompt_operation_type) != ompt_no_process.end())
                    return;
                populate_backtrace_data();
                switch(ompt_operation_type)
                {
                    case Wrapper::OMPT_ID_parallel_begin:
                        ompt_tracing_callback_start(record, user_data, ts);
                        ompt_push_parallel_callback(record, ts);
                        break;
                    case Wrapper::OMPT_ID_parallel_end:
                        ompt_tracing_callback_stop(record, user_data, ts, _bt_data);
                        ompt_pop_parallel_callback(record, ts, _bt_data);
                        break;
                    // Unlike parallel callbacks, we cannot receive the corresponding
                    // end to thread_begin. Set thread_begin as "instant" so the user
                    // can see callback without it spanning the entire track
                    case Wrapper::OMPT_ID_thread_begin:
                    case Wrapper::OMPT_ID_lock_init:
                    case Wrapper::OMPT_ID_lock_destroy:
                    case Wrapper::OMPT_ID_nest_lock:
                    case Wrapper::OMPT_ID_dispatch:
                    case Wrapper::OMPT_ID_flush:
                    case Wrapper::OMPT_ID_cancel:
                    case Wrapper::OMPT_ID_device_initialize:
                    case Wrapper::OMPT_ID_device_finalize:
                    case Wrapper::OMPT_ID_device_load:
                    case Wrapper::OMPT_ID_task_create:
                    case Wrapper::OMPT_ID_task_schedule:
                    case Wrapper::OMPT_ID_mutex_released:
                    case Wrapper::OMPT_ID_mutex_acquire:
                    case Wrapper::OMPT_ID_mutex_acquired:
                    case Wrapper::OMPT_ID_dependences:
                    case Wrapper::OMPT_ID_task_dependence:
                    case Wrapper::OMPT_ID_error:
                    {
                        // These callbacks are considered instant events and should
                        // start and immediately call stop as no corresponding "end"
                        // will be received
                        auto instant_ts = ts;
                        ompt_tracing_callback_start(record, user_data, instant_ts);
                        ompt_tracing_callback_stop(record, user_data, instant_ts,
                                                   _bt_data);
                        // Although this has endpoint arg, treat it as instant event
                        ompt_cache_instant_event(record, instant_ts, _bt_data);
                        break;
                    }
                    // case ROCPROFILER_OMPT_ID_device_unload: // Unsupported by
                    // runtime
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

template <typename Wrapper>
std::string_view
library_sdk<Wrapper>::ompt_get_unified_name(
    const typename Wrapper::callback_tracing_record& record)
{
    std::string_view _name =
        tool_data->callback_tracing_info.at(record.kind, record.operation);

    // Forces omp_parallel begin and end to have same name, allowing track to connect
    if(record.operation == Wrapper::OMPT_ID_parallel_begin ||
       record.operation == Wrapper::OMPT_ID_parallel_end)
        _name = "omp_parallel";

    return _name;
}

template <typename Wrapper>
template <typename ArgsT>
void
library_sdk<Wrapper>::ompt_iterate_operation_args(
    const typename Wrapper::callback_tracing_record& record, ArgsT& args)
{
    static_assert(std::is_same_v<ArgsT, callback_arg_array_t> ||
                      std::is_same_v<ArgsT, function_args_t>,
                  "ArgsT must be callback_arg_array_t or function_args_t");

    auto ompt_operation_type =
        static_cast<typename Wrapper::ompt_operation_t>(record.operation);
    // ROCProfiler-SDK documentation recommends using 1 for the ENTER phase to avoid seg.
    // faults.
    auto max_deref = (record.phase == Wrapper::CALLBACK_PHASE_ENTER ||
                      ompt_operation_type == Wrapper::OMPT_ID_parallel_begin)
                         ? 1
                         : 2;

    // Perform standard iteration of arguments
    if constexpr(std::is_same_v<ArgsT, callback_arg_array_t>)
        Wrapper::iterate_callback_tracing_kind_operation_args(record, save_args,
                                                              max_deref, &args);
    else
        Wrapper::iterate_callback_tracing_kind_operation_args(
            record, iterate_args_callback, max_deref, &args);

    static const auto ompt_has_flags = std::set<typename Wrapper::ompt_operation_t>{
        Wrapper::OMPT_ID_parallel_begin, Wrapper::OMPT_ID_parallel_end,
        Wrapper::OMPT_ID_task_create,    Wrapper::OMPT_ID_implicit_task,
        Wrapper::OMPT_ID_cancel,
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
    auto* payload_data = static_cast<typename Wrapper::ompt_data_t*>(record.payload);
    if(!payload_data) return;

    // Extract flags value
    switch(ompt_operation_type)
    {
        case Wrapper::OMPT_ID_parallel_begin:
            flags_val = payload_data->args.parallel_begin.flags;
            break;
        case Wrapper::OMPT_ID_parallel_end:
            flags_val = payload_data->args.parallel_end.flags;
            break;
        case Wrapper::OMPT_ID_task_create:
            flags_val = payload_data->args.task_create.flags;
            break;
        case Wrapper::OMPT_ID_implicit_task:
            flags_val = payload_data->args.implicit_task.flags;
            break;
        case Wrapper::OMPT_ID_cancel: flags_val = payload_data->args.cancel.flags; break;
        default: break;
    }

    // Textual representation of flags adapted from OMPT 5.0 specification
    switch(ompt_operation_type)
    {
        case Wrapper::OMPT_ID_parallel_begin:  // ompt_parallel_flag_t
        case Wrapper::OMPT_ID_parallel_end:    // ompt_parallel_flag_t
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
        case Wrapper::OMPT_ID_task_create:  // ompt_task_flag_t
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

            // Multiple/none can be set
            std::string task_properties;
            task_properties.reserve(60);
            if(flags_val & ompt_task_undeferred) task_properties += "undeferred, ";
            if(flags_val & ompt_task_untied) task_properties += "untied, ";
            if(flags_val & ompt_task_final) task_properties += "final, ";
            if(flags_val & ompt_task_mergeable) task_properties += "mergeable, ";
            if(flags_val & ompt_task_merged) task_properties += "merged, ";

            if(!task_properties.empty())
                task_properties.erase(task_properties.size() - 2);
            else
                task_properties = "none";
            append(ft, "properties", task_properties);
            break;
        }
        case Wrapper::OMPT_ID_implicit_task:  // initial (1) or implicit (2)
        {
            const auto ft = std::string{ "flags" };
            // As of now, implicit_tasks with ompt_task_initial are filtered out
            if(flags_val & ompt_task_initial)
                append(ft, "kind", "initial");
            else if(flags_val & ompt_task_implicit)
                append(ft, "kind", "implicit");
            break;
        }
        case Wrapper::OMPT_ID_cancel:  // ompt_cancel_flag_t
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

template <typename Wrapper>
auto&
library_sdk<Wrapper>::get_ompt_standard_cb_storage()
{
    // std::uint64_t -> internal id from rocprofiler_correlation_id_t
    static thread_local auto _v =
        std::unordered_map<std::uint64_t, rocprofsys_ompt_data_storage_t>{};
    return _v;
}

template <typename Wrapper>
auto&
library_sdk<Wrapper>::get_ompt_parallel_cb_storage()
{
    // uintptr_t -> parallel_data (see callback definition)
    static thread_local auto _v =
        std::unordered_map<uintptr_t, rocprofsys_ompt_data_storage_t>{};
    return _v;
}

template <typename Wrapper>
void
library_sdk<Wrapper>::ompt_cache_instant_event(
    typename Wrapper::callback_tracing_record                 record,
    typename Wrapper::timestamp_t                             instant_ts,
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

template <typename Wrapper>
void
library_sdk<Wrapper>::ompt_cache_orphan_event(
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

template <typename Wrapper>
void
library_sdk<Wrapper>::ompt_push_standard_callback(
    const typename Wrapper::callback_tracing_record& record,
    const typename Wrapper::timestamp_t&             beg_ts)
{
    auto args = function_args_t{};
    ompt_iterate_operation_args(record, args);
    get_ompt_standard_cb_storage().emplace(
        record.correlation_id.internal,
        rocprofsys_ompt_data_storage_t{ record, beg_ts, args });
}

template <typename Wrapper>
void
library_sdk<Wrapper>::ompt_pop_standard_callback(
    const typename Wrapper::callback_tracing_record&          record,
    const typename Wrapper::timestamp_t&                      end_ts,
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

template <typename Wrapper>
void
library_sdk<Wrapper>::ompt_push_parallel_callback(
    const typename Wrapper::callback_tracing_record& record,
    const typename Wrapper::timestamp_t&             beg_ts)
{
    auto* payload_data = static_cast<typename Wrapper::ompt_data_t*>(record.payload);
    const void* parallel_data_address = payload_data->args.parallel_begin.parallel_data;
    auto        args                  = function_args_t{};
    ompt_iterate_operation_args(record, args);
    get_ompt_parallel_cb_storage().emplace(
        reinterpret_cast<uintptr_t>(parallel_data_address),
        rocprofsys_ompt_data_storage_t{ record, beg_ts, args });
}

template <typename Wrapper>
void
library_sdk<Wrapper>::ompt_pop_parallel_callback(
    const typename Wrapper::callback_tracing_record&          record,
    const typename Wrapper::timestamp_t&                      end_ts,
    std::optional<std::vector<tim::unwind::processed_entry>>& bt_data)
{
    auto* payload_data = static_cast<typename Wrapper::ompt_data_t*>(record.payload);
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

template <typename Wrapper>
void
library_sdk<Wrapper>::ompt_tracing_callback_start(
    typename Wrapper::callback_tracing_record record,
    typename Wrapper::user_data_t* /*user_data*/, typename Wrapper::timestamp_t ts)
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

template <typename Wrapper>
void
library_sdk<Wrapper>::ompt_tracing_callback_stop(
    typename Wrapper::callback_tracing_record record,
    typename Wrapper::user_data_t* /*user_data*/, typename Wrapper::timestamp_t ts,
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

template <typename Wrapper>
void
library_sdk<Wrapper>::ompt_finalize_orphan_events()
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

template <typename Wrapper>
void
library_sdk<Wrapper>::tool_tracing_buffered(typename Wrapper::context_id /*context*/,
                                            typename Wrapper::buffer_id /*buffer_id*/,
                                            typename Wrapper::record_header_t** headers,
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

        if(ROCPROFSYS_LIKELY(header->category == Wrapper::BUFFER_CATEGORY_TRACING))
        {
            if(header->kind == Wrapper::BUFFER_TRACING_KERNEL_DISPATCH)
            {
                auto* record = static_cast<typename Wrapper::kernel_dispatch_record*>(
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
            else if(header->kind == Wrapper::BUFFER_TRACING_SCRATCH_MEMORY)
            {
                // Scratch memory event is not associated with a HIP stream
                auto* record = static_cast<typename Wrapper::scratch_memory_record*>(
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
            else if(header->kind == Wrapper::BUFFER_TRACING_MEMORY_COPY)
            {
                // memory_copy is not associated with a HIP stream
                auto* record =
                    static_cast<typename Wrapper::memory_copy_record*>(header->payload);
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
                                              typename Wrapper::thread_id _tid) {
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
            else if(header->kind == Wrapper::BUFFER_TRACING_MEMORY_ALLOCATION)
            {
                auto* record =
                    static_cast<typename Wrapper::memory_alloc_record*>(header->payload);
                std::uint64_t _stream_id = get_stream_id(record).handle;
                cache_category<category::rocm_memory_allocate>();
                cache_add_thread_info(record->thread_id);
                cache_memory_allocation(record, _stream_id);
            }
#endif
#if(ROCPROFILER_VERSION >= 10000)
            else if(header->kind == Wrapper::BUFFER_TRACING_KFD_PAGE_FAULT)
            {
                auto* record = static_cast<typename Wrapper::kfd_page_fault_record*>(
                    header->payload);
                tool_kfd_page_fault_callback(tool_data, record);
            }
            else if(header->kind == Wrapper::BUFFER_TRACING_KFD_PAGE_MIGRATE)
            {
                auto* record = static_cast<typename Wrapper::kfd_page_migrate_record*>(
                    header->payload);
                tool_kfd_page_migrate_callback(tool_data, record);
            }
            else if(header->kind == Wrapper::BUFFER_TRACING_KFD_QUEUE)
            {
                auto* record =
                    static_cast<typename Wrapper::kfd_queue_record*>(header->payload);
                tool_kfd_queue_callback(tool_data, record);
            }
            else if(header->kind == Wrapper::BUFFER_TRACING_KFD_EVENT_QUEUE)
            {
                auto* record = static_cast<typename Wrapper::kfd_event_queue_record*>(
                    header->payload);
                tool_kfd_event_queue_callback(tool_data, record);
            }
            else if(header->kind == Wrapper::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU)
            {
                auto* record = static_cast<typename Wrapper::kfd_event_unmap_record*>(
                    header->payload);
                tool_kfd_event_unmap_from_gpu_callback(tool_data, record);
            }
            else if(header->kind == Wrapper::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS)
            {
                auto* record = static_cast<typename Wrapper::kfd_event_dropped_record*>(
                    header->payload);
                tool_kfd_event_dropped_events_callback(tool_data, record);
            }
#endif
            else if(header->kind == Wrapper::BUFFER_TRACING_HSA_CORE_API ||
                    header->kind == Wrapper::BUFFER_TRACING_HSA_AMD_EXT_API)
            {
                // Not handling those buffered events
                continue;
            }
            else
            {
                throw std::runtime_error(fmt::format(
                    "unexpected typename Wrapper::record_header_t buffer tracing "
                    "category "
                    "kind. category: {}, kind: {}",
                    static_cast<int>(header->category), static_cast<int>(header->kind)));
            }
        }
        else
        {
            throw std::runtime_error(fmt::format(
                "unexpected typename Wrapper::record_header_t buffer tracing category "
                "kind. category: {}, kind: {}",
                static_cast<int>(header->category), static_cast<int>(header->kind)));
        }
    }
}

// ─── Counter callbacks ────────────────────────────────────────────────────────

template <typename Wrapper>
void
library_sdk<Wrapper>::counter_record_callback(
    typename Wrapper::dispatch_counting_data dispatch_data,
    typename Wrapper::counter_record* record_data, size_t record_count,
    typename Wrapper::user_data_t /*user_data*/, void* /*callback_data_arg*/)
{
    auto* _agent_counter_storage = get_counter_storage();
    if(!_agent_counter_storage) return;

    static auto _mtx = std::mutex{};
    auto        _lk  = std::unique_lock<std::mutex>{ _mtx };

    auto _dispatch_id = dispatch_data.dispatch_info.dispatch_id;
    auto _agent_id    = dispatch_data.dispatch_info.agent_id;
    auto _scope       = scope::get_default();
    auto _interval    = timing_interval<Wrapper>{};
    auto _aggregate   = std::unordered_map<typename Wrapper::counter_id,
                                           typename Wrapper::counter_record>{};

    for(size_t i = 0; i < record_count; ++i)
    {
        auto _counter_id = typename Wrapper::counter_id{};
        rocprofiler_call<Wrapper>(
            Wrapper::query_record_counter_id(record_data[i].id, &_counter_id),
            "Wrapper::query_record_counter_id");
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
                itr.first,
                counter_storage<Wrapper>{ tool_data, _dev_id, 0, _info->name });
        }
        auto _event = counter_event<Wrapper>{ counter_dispatch_record<Wrapper>{
            &dispatch_data, _dispatch_id, itr.first, itr.second } };
        _agent_counter_storage->at(_agent_id).at(itr.first)(_event, _interval, _scope);
    }
}

template <typename Wrapper>
void
library_sdk<Wrapper>::dispatch_counting_service_callback(
    typename Wrapper::dispatch_counting_data dispatch_data,
    typename Wrapper::counter_config_id*     config,
    typename Wrapper::user_data_t* /*user_data*/, void* callback_data_arg)
{
    auto* _data = as_client_data<Wrapper>(callback_data_arg);
    if(!_data || !config) return;
    if(auto itr =
           _data->agent_counter_profiles.find(dispatch_data.dispatch_info.agent_id);
       itr != _data->agent_counter_profiles.end() && itr->second)
        *config = *itr->second;
}

#if ROCPROFILER_VERSION >= 700
// ─── tool_hip_stream_callback ────────────────────────────────────────────────

template <typename Wrapper>
void
library_sdk<Wrapper>::tool_hip_stream_callback(
    typename Wrapper::callback_tracing_record record,
    typename Wrapper::user_data_t* /*user_data*/, void* /*data*/)
{
    if(record.kind != Wrapper::CALLBACK_TRACING_HIP_STREAM) return;
    // Extract stream ID from record
    auto* stream_handle_data =
        static_cast<typename Wrapper::hip_stream_data*>(record.payload);
    auto stream_id = stream_handle_data->stream_id;

    // STREAM_HANDLE_CREATE and DESTROY are no-ops
    if(record.operation == Wrapper::HIP_STREAM_CREATE)
    {
        LOG_TRACE(" operation = ROCPROFILER_HIP_STREAM_CREATE");
    }
    else if(record.operation == Wrapper::HIP_STREAM_DESTROY)
    {
        LOG_TRACE(" operation = ROCPROFILER_HIP_STREAM_DESTROY");
    }
    else if(record.operation == Wrapper::HIP_STREAM_SET)
    {
        if(record.phase == Wrapper::CALLBACK_PHASE_ENTER)
        {
            // Push the stream ID onto the stream stack before underlying HIP function is
            // called
            LOG_TRACE(" operation = ROCPROFILER_HIP_STREAM_SET, phase = "
                      "ROCPROFILER_CALLBACK_PHASE_ENTER");
            stream_id_push(stream_id);
        }
        else if(record.phase == Wrapper::CALLBACK_PHASE_EXIT)
        {
            // Pop stream ID off of stream stack after underlying HIP function is
            // completed
            LOG_TRACE("operation = ROCPROFILER_HIP_STREAM_SET, phase = "
                      "ROCPROFILER_CALLBACK_PHASE_EXIT");
            stream_id_pop();
        }
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

template <typename Wrapper>
int
library_sdk<Wrapper>::tool_init(typename Wrapper::client_finalize_t fini_func,
                                void*                               user_data)
{
    // Only initialize once per session
    if(tool_init_done.exchange(true)) return 0;

    auto domains = settings::instance()->at(std::string{ env_vars::ROCM_DOMAINS });

    std::stringstream _domains_ss;
    for(const auto& itr : domains->get_choices())
        _domains_ss << "- " << itr << "\n";
    LOG_DEBUG("Available ROCm Domains: \n {}", _domains_ss.str());

    auto _callback_domains = sdk_core<Wrapper>::get_callback_domains();
    auto _buffered_domain  = sdk_core<Wrapper>::get_buffered_domains();
    auto _counter_events   = sdk_core<Wrapper>::get_rocm_events();
    auto _version          = sdk_core<Wrapper>::get_version();
    if(_version.formatted == 0)
    {
        LOG_WARNING("rocprofiler-sdk version not initialized");
    }

    auto* _data        = as_client_data<Wrapper>(user_data);
    _data->client_fini = fini_func;

    _data->initialize();
    if(!_counter_events.empty()) _data->initialize_event_info();

    rocprofiler_call<Wrapper>(Wrapper::create_context(&_data->primary_ctx),
                              "Wrapper::create_context");

    // Code object callback on its own always-on context so kernel names are captured
    // even when primary_ctx is stopped (region filter mode)
    rocprofiler_call<Wrapper>(Wrapper::create_context(&_data->code_object_ctx),
                              "Wrapper::create_context");
    rocprofiler_call<Wrapper>(Wrapper::configure_callback_tracing_service(
                                  _data->code_object_ctx,
                                  Wrapper::CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
                                  tool_code_object_callback, _data),
                              "Wrapper::configure_callback_tracing_service");

    // Control context for marker-based region filtering and pause/resume (always-on)
    rocprofiler_call<Wrapper>(Wrapper::create_context(&_data->control_ctx),
                              "Wrapper::create_context");

    auto external_corr_id_request_kinds =
        std::array<typename Wrapper::external_correlation_request_kind, 3>{
            Wrapper::EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
            Wrapper::EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,
#if(ROCPROFILER_VERSION >= 600)
            Wrapper::EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION
#endif
        };

    // Insert the default stream and queue info to ensure that the default entry exists
    {
        trace_cache::get_metadata_registry().add_stream(0);
        trace_cache::get_metadata_registry().add_queue(0);
    }

    // MARKER_CORE_API is handled by roctx_client on control_ctx
    // ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,          ///< @see
    // ::rocprofiler_hsa_core_api_id_t ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,
    for(auto itr : {
            Wrapper::CALLBACK_TRACING_HSA_CORE_API,
            Wrapper::CALLBACK_TRACING_HSA_AMD_EXT_API,
            Wrapper::CALLBACK_TRACING_HSA_IMAGE_EXT_API,
            Wrapper::CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
            Wrapper::CALLBACK_TRACING_HIP_RUNTIME_API,
            Wrapper::CALLBACK_TRACING_HIP_COMPILER_API,
            Wrapper::CALLBACK_TRACING_RCCL_API,
#if(ROCPROFILER_VERSION >= 600)
            Wrapper::CALLBACK_TRACING_OMPT,
            Wrapper::CALLBACK_TRACING_ROCDECODE_API,
#endif
#if(ROCPROFILER_VERSION >= 700)
            Wrapper::CALLBACK_TRACING_ROCJPEG_API,
#endif
        })
    {
        if(_callback_domains.count(itr) > 0)
        {
            auto _ops = sdk_core<Wrapper>::get_operations(itr);
            _data->backtrace_operations.emplace(
                itr, sdk_core<Wrapper>::get_backtrace_operations(itr));
            rocprofiler_call<Wrapper>(Wrapper::configure_callback_tracing_service(
                                          _data->primary_ctx, itr, _ops.data(),
                                          _ops.size(), tool_tracing_callback, _data),
                                      "Wrapper::configure_callback_tracing_service");
        }
    }

    constexpr auto buffer_size = 16 * 4096;
    constexpr auto watermark   = 15 * 4096;

    rocprofiler_call<Wrapper>(
        Wrapper::configure_external_correlation_id_request_service(
            _data->primary_ctx, external_corr_id_request_kinds.data(),
            external_corr_id_request_kinds.size(),
            set_kernel_rename_and_stream_correlation_id, _data),
        "Wrapper::configure_external_correlation_id_request_service");

#if(ROCPROFILER_VERSION >= 700)
    if((_buffered_domain.count(Wrapper::BUFFER_TRACING_KERNEL_DISPATCH) > 0) ||
       (_buffered_domain.count(Wrapper::BUFFER_TRACING_MEMORY_COPY) > 0))
    {
        rocprofiler_call<Wrapper>(Wrapper::configure_callback_tracing_service(
                                      _data->primary_ctx,
                                      Wrapper::CALLBACK_TRACING_HIP_STREAM, nullptr, 0,
                                      tool_hip_stream_callback, nullptr),
                                  "Wrapper::configure_callback_tracing_service");
    }
#endif

    if(_callback_domains.count(Wrapper::CALLBACK_TRACING_RCCL_API) > 0)
        rocprofiler_sdk::rccl_comm_data_initialize();

    if(_buffered_domain.count(Wrapper::BUFFER_TRACING_KERNEL_DISPATCH) > 0)
    {
        rocprofiler_call<Wrapper>(
            Wrapper::create_buffer(_data->primary_ctx, buffer_size, watermark,
                                   Wrapper::BUFFER_POLICY_LOSSLESS, tool_tracing_buffered,
                                   tool_data, &_data->kernel_dispatch_buffer),
            "Wrapper::create_buffer");
        rocprofiler_call<Wrapper>(Wrapper::configure_buffer_tracing_service(
                                      _data->primary_ctx,
                                      Wrapper::BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0,
                                      _data->kernel_dispatch_buffer),
                                  "Wrapper::configure_buffer_tracing_service");
    }

    if(_buffered_domain.count(Wrapper::BUFFER_TRACING_MEMORY_COPY) > 0)
    {
        rocprofiler_call<Wrapper>(
            Wrapper::create_buffer(_data->primary_ctx, buffer_size, watermark,
                                   Wrapper::BUFFER_POLICY_LOSSLESS, tool_tracing_buffered,
                                   tool_data, &_data->memory_copy_buffer),
            "Wrapper::create_buffer");
        rocprofiler_call<Wrapper>(Wrapper::configure_buffer_tracing_service(
                                      _data->primary_ctx,
                                      Wrapper::BUFFER_TRACING_MEMORY_COPY, nullptr, 0,
                                      _data->memory_copy_buffer),
                                  "Wrapper::configure_buffer_tracing_service");
    }

    if(_buffered_domain.count(Wrapper::BUFFER_TRACING_SCRATCH_MEMORY) > 0)
    {
        rocprofiler_call<Wrapper>(
            Wrapper::create_buffer(_data->primary_ctx, buffer_size, watermark,
                                   Wrapper::BUFFER_POLICY_LOSSLESS, tool_tracing_buffered,
                                   tool_data, &_data->scratch_memory_buffer),
            "Wrapper::create_buffer");
        rocprofiler_call<Wrapper>(Wrapper::configure_buffer_tracing_service(
                                      _data->primary_ctx,
                                      Wrapper::BUFFER_TRACING_SCRATCH_MEMORY, nullptr, 0,
                                      _data->scratch_memory_buffer),
                                  "Wrapper::configure_buffer_tracing_service");
    }

#if(ROCPROFILER_VERSION >= 600)
    if(_buffered_domain.count(Wrapper::BUFFER_TRACING_MEMORY_ALLOCATION) > 0)
    {
        rocprofiler_call<Wrapper>(
            Wrapper::create_buffer(_data->primary_ctx, buffer_size, watermark,
                                   Wrapper::BUFFER_POLICY_LOSSLESS, tool_tracing_buffered,
                                   tool_data, &_data->memory_alloc_buffer),
            "Wrapper::create_buffer");
        if(_data->memory_alloc_buffer.handle == 0UL)
        {
            LOG_CRITICAL("Failed to create memory allocation buffer");
            ::rocprofsys::set_state(::rocprofsys::State::Finalized);
            ::std::abort();
        }
        rocprofiler_call<Wrapper>(Wrapper::configure_buffer_tracing_service(
                                      _data->primary_ctx,
                                      Wrapper::BUFFER_TRACING_MEMORY_ALLOCATION, nullptr,
                                      0, _data->memory_alloc_buffer),
                                  "Wrapper::configure_buffer_tracing_service");
    }
#endif

#if(ROCPROFILER_VERSION >= 10000)
    if(_buffered_domain.count(Wrapper::BUFFER_TRACING_KFD_PAGE_FAULT) > 0 ||
       _buffered_domain.count(Wrapper::BUFFER_TRACING_KFD_PAGE_MIGRATE) > 0 ||
       _buffered_domain.count(Wrapper::BUFFER_TRACING_KFD_QUEUE) > 0 ||
       _buffered_domain.count(Wrapper::BUFFER_TRACING_KFD_EVENT_QUEUE) > 0 ||
       _buffered_domain.count(Wrapper::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU) > 0 ||
       _buffered_domain.count(Wrapper::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS) > 0)
    {
        // Initialize KFD event metadata
        rocprofiler_sdk::kfd_event_metadata_initialize(tool_data);
    }

    if(_buffered_domain.count(Wrapper::BUFFER_TRACING_KFD_PAGE_FAULT) > 0)
    {
        rocprofiler_call<Wrapper>(
            Wrapper::create_buffer(_data->primary_ctx, buffer_size, watermark,
                                   Wrapper::BUFFER_POLICY_LOSSLESS, tool_tracing_buffered,
                                   tool_data, &_data->kfd_page_fault_buffer),
            "Wrapper::create_buffer");
        rocprofiler_call<Wrapper>(Wrapper::configure_buffer_tracing_service(
                                      _data->primary_ctx,
                                      Wrapper::BUFFER_TRACING_KFD_PAGE_FAULT, nullptr, 0,
                                      _data->kfd_page_fault_buffer),
                                  "Wrapper::configure_buffer_tracing_service");
    }
    if(_buffered_domain.count(Wrapper::BUFFER_TRACING_KFD_PAGE_MIGRATE) > 0)
    {
        rocprofiler_call<Wrapper>(
            Wrapper::create_buffer(_data->primary_ctx, buffer_size, watermark,
                                   Wrapper::BUFFER_POLICY_LOSSLESS, tool_tracing_buffered,
                                   tool_data, &_data->kfd_page_migrate_buffer),
            "Wrapper::create_buffer");
        rocprofiler_call<Wrapper>(Wrapper::configure_buffer_tracing_service(
                                      _data->primary_ctx,
                                      Wrapper::BUFFER_TRACING_KFD_PAGE_MIGRATE, nullptr,
                                      0, _data->kfd_page_migrate_buffer),
                                  "Wrapper::configure_buffer_tracing_service");
    }
    if(_buffered_domain.count(Wrapper::BUFFER_TRACING_KFD_QUEUE) > 0)
    {
        rocprofiler_call<Wrapper>(
            Wrapper::create_buffer(_data->primary_ctx, buffer_size, watermark,
                                   Wrapper::BUFFER_POLICY_LOSSLESS, tool_tracing_buffered,
                                   tool_data, &_data->kfd_queue_buffer),
            "Wrapper::create_buffer");
        rocprofiler_call<Wrapper>(Wrapper::configure_buffer_tracing_service(
                                      _data->primary_ctx,
                                      Wrapper::BUFFER_TRACING_KFD_QUEUE, nullptr, 0,
                                      _data->kfd_queue_buffer),
                                  "Wrapper::configure_buffer_tracing_service");
    }
    if(_buffered_domain.count(Wrapper::BUFFER_TRACING_KFD_EVENT_QUEUE) > 0)
    {
        rocprofiler_call<Wrapper>(
            Wrapper::create_buffer(_data->primary_ctx, buffer_size, watermark,
                                   Wrapper::BUFFER_POLICY_LOSSLESS, tool_tracing_buffered,
                                   tool_data, &_data->kfd_event_queue_buffer),
            "Wrapper::create_buffer");
        // The only KFD_EVENT_QUEUE operation we want to process is RESTORE_RESCHEDULED.
        // All others are captured within paired KFD_QUEUE operations
        auto kfd_event_queue_ops = std::array<typename Wrapper::tracing_operation, 1>{
            Wrapper::KFD_EVENT_QUEUE_RESTORE_RESCHEDULED
        };
        rocprofiler_call<Wrapper>(
            Wrapper::configure_buffer_tracing_service(
                _data->primary_ctx, Wrapper::BUFFER_TRACING_KFD_EVENT_QUEUE,
                kfd_event_queue_ops.data(), kfd_event_queue_ops.size(),
                _data->kfd_event_queue_buffer),
            "Wrapper::configure_buffer_tracing_service");
    }
    if(_buffered_domain.count(Wrapper::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU) > 0)
    {
        rocprofiler_call<Wrapper>(
            Wrapper::create_buffer(_data->primary_ctx, buffer_size, watermark,
                                   Wrapper::BUFFER_POLICY_LOSSLESS, tool_tracing_buffered,
                                   tool_data, &_data->kfd_event_unmap_buffer),
            "Wrapper::create_buffer");
        rocprofiler_call<Wrapper>(Wrapper::configure_buffer_tracing_service(
                                      _data->primary_ctx,
                                      Wrapper::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                                      nullptr, 0, _data->kfd_event_unmap_buffer),
                                  "Wrapper::configure_buffer_tracing_service");
    }
    if(_buffered_domain.count(Wrapper::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS) > 0)
    {
        rocprofiler_call<Wrapper>(
            Wrapper::create_buffer(_data->primary_ctx, buffer_size, watermark,
                                   Wrapper::BUFFER_POLICY_LOSSLESS, tool_tracing_buffered,
                                   tool_data, &_data->kfd_event_dropped_buffer),
            "Wrapper::create_buffer");
        rocprofiler_call<Wrapper>(Wrapper::configure_buffer_tracing_service(
                                      _data->primary_ctx,
                                      Wrapper::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS,
                                      nullptr, 0, _data->kfd_event_dropped_buffer),
                                  "Wrapper::configure_buffer_tracing_service");
    }
#endif

    if(!_counter_events.empty())
    {
        // Resolve counter names to counter IDs per agent
        for(const auto& itr : _data->gpu_agents)
        {
            const auto& _agent_id = typename Wrapper::agent_id{ itr.agent->handle };
            _data->agent_events.emplace(
                _agent_id, create_agent_profile(_agent_id, _counter_events, _data));
        }

        // --- Dispatch-mode kernel counters ---
        rocprofiler_call<Wrapper>(Wrapper::create_context(&_data->counter_ctx),
                                  "Wrapper::create_context");
        auto _operations = std::array<typename Wrapper::tracing_operation, 1>{
            Wrapper::KERNEL_DISPATCH_COMPLETE,
        };
        rocprofiler_call<Wrapper>(
            Wrapper::configure_callback_tracing_service(
                _data->counter_ctx, Wrapper::CALLBACK_TRACING_KERNEL_DISPATCH,
                _operations.data(), _operations.size(), tool_tracing_callback, _data),
            "Wrapper::configure_callback_tracing_service");
        rocprofiler_call<Wrapper>(
            Wrapper::configure_callback_dispatch_counting_service(
                _data->counter_ctx, dispatch_counting_service_callback, _data,
                counter_record_callback, _data),
            "Wrapper::configure_callback_dispatch_counting_service");
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
            auto client_thread = typename Wrapper::callback_thread_id{};
            rocprofiler_call<Wrapper>(Wrapper::create_callback_thread(&client_thread),
                                      "Wrapper::create_callback_thread");
            rocprofiler_call<Wrapper>(Wrapper::assign_callback_thread(itr, client_thread),
                                      "Wrapper::assign_callback_thread");
        }
    }

    // notify rocprofiler that initialization failed and all the contexts,
    // buffers, etc. created should be ignored
    if(!is_valid(_data->primary_ctx)) return -1;

    gpu::add_device_metadata();

    if(config::get_use_process_sampling())
    {
        LOG_DEBUG("Setting PMC sampler state to active...");
        pmc::set_state(State::Active);
    }

    // Setup roctx client (must happen within tool_init for rocprofiler-sdk context
    // creation). Roctx client configures MARKER_CORE_API and MARKER_CONTROL_API
    // on control_ctx. trace_control's pause/resume callbacks are routed through
    // roctx_client (registered later in library.cpp).
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
    // no errors
    return 0;
}

// ─── tool_fini ───────────────────────────────────────────────────────────────

template <typename Wrapper>
void
library_sdk<Wrapper>::tool_fini(void* callback_data)
{
    if(tool_fini_done.exchange(true)) return;

    flush();
    stop();
    finalize_sdk_common();

    // Destroy buffers to drain in-flight async flush callbacks before
    // deleting tool_data, which those callbacks may still be accessing.
    // rocprofiler_destroy_buffer returns BUFFER_BUSY if a flush is still
    // in progress, so retry until it succeeds or buffer is not found.
    for(auto itr : tool_data->get_buffers())
    {
        while(itr.handle > 0 &&
              Wrapper::destroy_buffer(itr) == Wrapper::STATUS_ERROR_BUFFER_BUSY)
        {
            std::this_thread::yield();
        }
    }

    auto* _data        = as_client_data<Wrapper>(callback_data);
    _data->client_id   = nullptr;
    _data->client_fini = nullptr;
    delete tool_data;
    tool_data = nullptr;
}

// ─── tool_attach_* (version >= 10200) ────────────────────────────────────────

#if ROCPROFILER_VERSION >= 10200

template <typename Wrapper>
void
library_sdk<Wrapper>::tool_attach_fini(void* /*tool_data_ptr*/)
{
    // Stop and flush SDK contexts/buffers so that buffer callbacks
    // write their Perfetto events before Perfetto post-processing.
    ::rocprofsys::rocprofiler_sdk::stop();
    ::rocprofsys::rocprofiler_sdk::flush();
    finalize_sdk_common();

    // Flush any pending region cache entries
    rocprofsys_flush_pending_region_cache_hidden();

    // Write Perfetto trace output
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

template <typename Wrapper>
int
library_sdk<Wrapper>::tool_attach_init([[maybe_unused]]
                                       typename Wrapper::client_detach_t detach_func,
                                       typename Wrapper::context_id*     context_ids,
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

        // Restart Perfetto for a new tracing session
        if(get_use_perfetto()) ::rocprofsys::perfetto::start();

        trace_cache::get_buffer_storage().start(getpid());

        // Restart process sampler (AMD SMI, CPU freq polling thread)
        if(config::get_use_process_sampling())
        {
            ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
            ::rocprofsys::process_sampler::setup();
        }

        ::rocprofsys::set_state(::rocprofsys::State::Active);
    }

    // Start all contexts provided by the SDK
    for(std::uint64_t i = 0; i < context_ids_length; ++i)
    {
        rocprofiler_call<Wrapper>(Wrapper::start_context(context_ids[i]),
                                  "Wrapper::start_context");
    }

    ::rocprofsys::rocprofiler_sdk::start();

    return 0;
}

#endif  // ROCPROFILER_VERSION >= 10200

// ─── sdk_tool_configure ──────────────────────────────────────────────────────

template <typename Wrapper>
bool
library_sdk<Wrapper>::sdk_tool_configure(std::uint32_t                  version,
                                         const char*                    runtime_version,
                                         typename Wrapper::client_id_t* id)
{
    // Only configure once per attach session
    if(sdk_configured.exchange(true)) return true;

    // Ensure tooling is initialized and state is Active
    if(!rocprofsys::config::settings_are_configured() ||
       rocprofsys::get_state() < rocprofsys::State::Active)
    {
        rocprofsys_init_tooling_hidden();
    }

    // set the client name
    id->name = "rocprofsys";

    // store client info
    if(!tool_data) tool_data = new client_data<Wrapper>{};
    tool_data->client_id = id;

    // compute major/minor/patch version info
    std::uint32_t major = version / 10000;
    std::uint32_t minor = (version % 10000) / 100;
    std::uint32_t patch = version % 100;

    LOG_INFO("{} is using rocprofiler-sdk v{}.{}.{} ({})", id->name, major, minor, patch,
             runtime_version);

    rocprofiler_call<Wrapper>(Wrapper::at_internal_thread_create(
                                  &library_sdk::thread_precreate,
                                  &library_sdk::thread_postcreate,
                                  static_cast<typename Wrapper::runtime_library_t>(
                                      Wrapper::LIBRARY | Wrapper::HSA_LIBRARY |
                                      Wrapper::HIP_LIBRARY | Wrapper::MARKER_LIBRARY),
                                  nullptr),
                              "Wrapper::at_internal_thread_create");

    return true;
}

}  // namespace rocprofsys::rocprofiler_sdk
