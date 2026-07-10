// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/engine.hpp"
#include "core/perfetto/engine_impl.hpp"
#include <cstdint>

#include "core/config.hpp"
#include "core/perfetto/category_registry.hpp"
#include "core/perfetto/session_backend.hpp"
#include "logger/debug.hpp"

#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::core
{
namespace
{
// Thread-local pid tag consumed by the cached-mode interceptor TLS.
// Stored in TU scope so the static accessors can reach it without exposing
// the TLS in the header.
// -1 marks "no parser thread has claimed this thread yet"; pid 0 is the
// kernel/swapper on Linux but appears as init inside containers, so using
// 0 as the sentinel would be ambiguous.
thread_local int t_emitting_pid = -1;

// Process-global pointer to the currently-active cached-mode engine. Set
// on engine.start(cached_interceptor, ...), cleared on engine.stop(). The
// Interceptor TLS reads it at thread-local-state construction time (first
// emission on a worker thread) and caches the pointer; subsequent
// OnTracePacket calls dereference the per-thread cache. The stored collect
// callback is the type-erased bridge back into the concrete
// basic_cached_perfetto_engine<Backend> instantiation.
std::atomic<void*>                    g_active_cached_engine{ nullptr };
std::atomic<cached_engine_collect_fn> g_active_cached_collect{ nullptr };

// Surfaces violations of the "exactly one parser thread per pid" invariant
// that the lock-free hot path in collect_packet_bytes relies on. Concurrent
// writes to the same pid's byte vector would be UB (data race on
// std::vector internals). The check fires on set_emitting_pid; the actual
// emission path stays branch-free.
std::mutex                               g_pid_owner_mutex{};
std::unordered_map<int, std::thread::id> g_pid_owner_tids{};

// ----------------------------------------------------------------------------
// Cached-mode Interceptor
// ----------------------------------------------------------------------------

// Perfetto SDK keeps interceptors experimental: TracingMuxerImpl::
// RegisterInterceptor (perfetto.cc) silently rejects descriptors
// whose name is not one of {"test_interceptor", "console", "etwexport"}.
// The check predates upstreamable knobs and lives inside vendored
// submodule code we don't fork; using "test_interceptor" keeps the
// engine wired up without patching the SDK. The name is an internal
// binding key only -- registration side and TraceConfig.interceptor_config
// must agree, and that's its only effect.
constexpr const char* CACHED_INTERCEPTOR_NAME = "test_interceptor";

class cached_interceptor : public ::perfetto::Interceptor<cached_interceptor>
{
public:
    struct ThreadLocalState : ::perfetto::InterceptorBase::ThreadLocalState
    {
        ThreadLocalState(ThreadLocalStateArgs& /*args*/)
        : engine{ g_active_cached_engine.load(std::memory_order_acquire) }
        , collect{ g_active_cached_collect.load(std::memory_order_acquire) }
        , pid{ t_emitting_pid }
        {}

        // Cached at TLS-construction time (first emission on this thread).
        // Subsequent OnTracePacket calls use these without synchronisation.
        void*                    engine  = nullptr;
        cached_engine_collect_fn collect = nullptr;
        int                      pid     = -1;
    };

    static void OnTracePacket(InterceptorContext context)
    {
        auto& tls = context.GetThreadLocalState();
        if(tls.engine == nullptr || tls.collect == nullptr) return;
        if(tls.pid < 0) return;  // emitter never called set_emitting_pid

        // Primary safety: parser threads join via post_processor::run_multithreaded
        // before cached_perfetto_session destructs the engine. The reload below
        // is defense-in-depth: if the global has been cleared or rebound to a
        // different engine since this thread's TLS cached the pointer, refuse
        // the dereference so a hypothetical TLS-cache-outlives-engine race
        // degrades into a dropped packet rather than UAF.
        if(g_active_cached_engine.load(std::memory_order_acquire) != tls.engine) return;

        tls.collect(tls.engine, tls.pid, context.packet_data.data,
                    context.packet_data.size);
    }
};

// ----------------------------------------------------------------------------
// Perfetto SDK helpers
// ----------------------------------------------------------------------------

::perfetto::TraceConfig
make_trace_config(const engine_config& cfg)
{
    ::perfetto::TraceConfig trace_cfg{};

    const auto policy =
        (cfg.fill_policy == engine_config::fill_policy_t::discard)
            ? ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_DISCARD
            : ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_RING_BUFFER;
    auto* buffer_config = trace_cfg.add_buffers();
    buffer_config->set_size_kb(cfg.buffer_size_kb);
    buffer_config->set_fill_policy(policy);

    ::perfetto::protos::gen::TrackEventConfig track_event_cfg{};
    for(const auto& name : cfg.disabled_categories)
    {
        LOG_DEBUG("Disabling perfetto track event category: {}", name);
        track_event_cfg.add_disabled_categories(name);
    }

    trace_cfg.set_flush_period_ms(cfg.flush_period_ms);

    auto* ds_cfg = trace_cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

    ds_cfg->mutable_interceptor_config()->set_name(CACHED_INTERCEPTOR_NAME);
    return trace_cfg;
}

auto
make_tracing_error_callback()
{
    return [](::perfetto::TracingError err) {
        if(err.code == ::perfetto::TracingError::kTracingFailed)
            LOG_WARNING("Perfetto encountered a tracing error: {}", err.message);
    };
}

std::once_flag g_sdk_init_flag;
}  // namespace

engine_config
build_engine_config_from_settings()
{
    engine_config out{};

    out.buffer_size_kb = static_cast<std::uint32_t>(config::get_perfetto_buffer_size());
    out.shmem_size_hint_kb =
        static_cast<std::uint32_t>(config::get_perfetto_shmem_size_hint());
    out.flush_period_ms = config::get_perfetto_flush_period();

    out.fill_policy = (config::get_perfetto_fill_policy() == "discard")
                          ? engine_config::fill_policy_t::discard
                          : engine_config::fill_policy_t::ring_buffer;

    const auto& backend = config::get_perfetto_backend();
    if(backend == "system")
        out.backend = engine_config::backend_t::system;
    else if(backend == "all")
        out.backend = engine_config::backend_t::all;
    else
        out.backend = engine_config::backend_t::inprocess;

    const auto& disabled = config::get_disabled_categories();
    out.disabled_categories.assign(disabled.begin(), disabled.end());

    out.suppress_sdk_log_output =
        !config::output_filtering::is_log_output_enabled_for_current_mpi_rank();

    return out;
}

std::unique_ptr<::perfetto::TracingSession>
session_backend::new_trace() const
{
    return ::perfetto::Tracing::NewTrace();
}

void
session_backend::flush_track_events() const
{
    ::perfetto::TrackEvent::Flush();
}

void
tracing_session_deleter::operator()(::perfetto::TracingSession* session) const noexcept
{
    delete session;
}

void*
activate_cached_engine(void* engine, cached_engine_collect_fn collect) noexcept
{
    g_active_cached_collect.store(collect, std::memory_order_release);
    return g_active_cached_engine.exchange(engine, std::memory_order_acq_rel);
}

bool
clear_active_cached_engine(void* expected, void** observed) noexcept
{
    void*      local_expected = expected;
    const bool cleared        = g_active_cached_engine.compare_exchange_strong(
        local_expected, nullptr, std::memory_order_acq_rel, std::memory_order_acquire);
    if(cleared)
    {
        g_active_cached_collect.store(nullptr, std::memory_order_release);
    }
    if(observed != nullptr) *observed = local_expected;
    return cleared;
}

void
perfetto_sdk_backend::init_sdk(const engine_config& cfg) const
{
    // init_sdk only needs the SDK Initialize/Register calls; the TraceConfig
    // is built when the cached session starts.
    std::call_once(g_sdk_init_flag, [&cfg]() {
        ::perfetto::TracingInitArgs args{};
        args.shmem_size_hint_kb = cfg.shmem_size_hint_kb;

        if(cfg.backend != engine_config::backend_t::inprocess)
            args.backends |= ::perfetto::kSystemBackend;
        if(cfg.backend != engine_config::backend_t::system)
            args.backends |= ::perfetto::kInProcessBackend;

        if(cfg.suppress_sdk_log_output)
            args.log_message_callback = [](::perfetto::base::LogMessageCallbackArgs) {};

        ::perfetto::Tracing::Initialize(args);
        ::perfetto::TrackEvent::Register();
    });

    // Interceptor registration lives OUTSIDE the SDK-init call_once: the
    // SDK's TracingMuxerImpl::RegisterInterceptor is idempotent (it
    // returns early when the name already exists), but our call_once
    // would otherwise prevent a second engine instance (e.g. cached
    // mode after a live engine already initialized the SDK) from ever
    // requesting registration. Calling Register() per engine.init_sdk()
    // is cheap and guarantees the interceptor is available regardless
    // of which engine instance was first.
    ::perfetto::InterceptorDescriptor desc;
    desc.set_name(CACHED_INTERCEPTOR_NAME);
    cached_interceptor::Register(desc);
}

perfetto_sdk_backend::session_ptr
perfetto_sdk_backend::start_cached_session(const engine_config& cfg) const
{
    auto            trace_cfg = make_trace_config(cfg);
    session_backend backend{};
    auto            session =
        start_tracing_session(backend, trace_cfg, -1, make_tracing_error_callback());
    return session_ptr{ session.release() };
}

void
perfetto_sdk_backend::flush_and_stop(session_ptr& session) const
{
    if(!session) return;
    session_backend backend{};
    rocprofsys::core::flush_and_stop_session(backend, *session);
}

void
set_emitting_pid(int pid) noexcept
{
    try
    {
        const auto                  self = std::this_thread::get_id();
        std::lock_guard<std::mutex> lk{ g_pid_owner_mutex };
        if(t_emitting_pid >= 0)
        {
            auto it = g_pid_owner_tids.find(t_emitting_pid);
            if(it != g_pid_owner_tids.end() && it->second == self)
                g_pid_owner_tids.erase(it);
        }
        if(pid >= 0)
        {
            auto [it, inserted] = g_pid_owner_tids.try_emplace(pid, self);
            if(!inserted && it->second != self)
            {
                LOG_ERROR("perfetto cached emission: pid {} claimed by two parser "
                          "threads concurrently; the single-writer-per-pid "
                          "invariant of collect_packet_bytes is violated",
                          pid);
            }
        }
    } catch(...)
    {
        // Registry maintenance must not propagate; the hot path doesn't
        // depend on it for correctness, only diagnostics.
    }
    t_emitting_pid = pid;
}

int
get_emitting_pid() noexcept
{
    return t_emitting_pid;
}

template class basic_cached_perfetto_engine<perfetto_sdk_backend>;

}  // namespace rocprofsys::core
