// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/engine.hpp"
#include <cstdint>

#include "core/config.hpp"
#include "core/perfetto/category_registry.hpp"
#include "core/perfetto/packet_framing.hpp"
#include "core/perfetto/session_backend.hpp"
#include "core/perfetto/sinks.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <unistd.h>

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
// OnTracePacket calls dereference the per-thread cache. This is safe
// because cached-mode use is single-engine-at-a-time per process (cached
// and live are mutually exclusive via ROCPROFSYS_TRACE_LEGACY).
std::atomic<perfetto_engine*> g_active_cached_engine{ nullptr };

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
// RegisterInterceptor (perfetto.cc:~37265) silently rejects descriptors
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
        , pid{ t_emitting_pid }
        {}

        // Cached at TLS-construction time (first emission on this thread).
        // Subsequent OnTracePacket calls use these without synchronisation.
        perfetto_engine* engine = nullptr;
        int              pid    = -1;
    };

    static void OnTracePacket(InterceptorContext context)
    {
        auto& tls = context.GetThreadLocalState();
        if(tls.engine == nullptr) return;
        if(tls.pid < 0) return;  // emitter never called set_emitting_pid

        // Primary safety: parser threads join via post_processor::run_multithreaded
        // before cached_perfetto_session destructs the engine. The reload below
        // is defense-in-depth: if the global has been cleared or rebound to a
        // different engine since this thread's TLS cached the pointer, refuse
        // the dereference so a hypothetical TLS-cache-outlives-engine race
        // degrades into a dropped packet rather than UAF.
        if(g_active_cached_engine.load(std::memory_order_acquire) != tls.engine) return;

        tls.engine->collect_packet_bytes(tls.pid, context.packet_data.data,
                                         context.packet_data.size);
    }
};

// ----------------------------------------------------------------------------
// Perfetto SDK helpers
// ----------------------------------------------------------------------------

::perfetto::TraceConfig
make_trace_config(const engine_config& cfg, perfetto_engine::mode m)
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

    if(m == perfetto_engine::mode::cached_interceptor)
    {
        ds_cfg->mutable_interceptor_config()->set_name(CACHED_INTERCEPTOR_NAME);
    }
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

// 8 MiB up-front reservation per pid lets the hot path append packets with
// amortized O(1) push_back while typically avoiding any reallocation --
// each parser thread emits ~1-10 MiB of framed bytes for a non-trivial
// workload. Sized to dominate the typical case; larger traces still grow
// geometrically thereafter.
constexpr std::size_t COLLECTED_BYTES_SLAB_SIZE = std::size_t{ 8 } * 1024 * 1024;
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

std::once_flag perfetto_engine::s_sdk_init_flag;

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

bool
perfetto_engine::is_system_backend() const noexcept
{
    return m_cfg.backend != engine_config::backend_t::inprocess;
}

// Builds a fresh TracingSession in m_sessions[pid], wires the error
// callback, runs Setup(trace_cfg, fd), and StartBlocking. Takes
// m_sessions_mutex internally -- callers must NOT hold it. fd=-1 disables
// per-Perfetto-default fd output (cached-mode path).
void
perfetto_engine::start_session(pid_t pid, int fd, mode m)
{
    auto trace_cfg = make_trace_config(m_cfg, m);

    std::lock_guard<std::mutex> lk{ m_sessions_mutex };
    auto&                       slot = m_sessions[pid];
    session_backend             backend{};
    auto                        session =
        start_tracing_session(backend, trace_cfg, fd, make_tracing_error_callback());
    slot = tracing_session_ptr{ session.release() };
}

// ----------------------------------------------------------------------------
// perfetto_engine -- public API
// ----------------------------------------------------------------------------

perfetto_engine::perfetto_engine(engine_config cfg)
: m_cfg{ std::move(cfg) }
{}

perfetto_engine::~perfetto_engine() noexcept
{
    if(m_running)
    {
        // Last-ditch teardown so the engine never outlives its
        // TracingSession. Destructors must not throw.
        try
        {
            stop();
        } catch(const std::exception& e)
        {
            LOG_ERROR("perfetto_engine destructor ignored stop() exception: {}",
                      e.what());
        } catch(...)
        {
            LOG_ERROR("perfetto_engine destructor ignored unknown stop() exception");
        }
    }
}

void
perfetto_engine::init_sdk()
{
    // init_sdk only needs the SDK Initialize/Register calls; the TraceConfig
    // is built per-mode in start().
    std::call_once(s_sdk_init_flag, [this]() {
        ::perfetto::TracingInitArgs args{};
        args.shmem_size_hint_kb = m_cfg.shmem_size_hint_kb;

        if(m_cfg.backend != engine_config::backend_t::inprocess)
            args.backends |= ::perfetto::kSystemBackend;
        if(m_cfg.backend != engine_config::backend_t::system)
            args.backends |= ::perfetto::kInProcessBackend;

        if(m_cfg.suppress_sdk_log_output)
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

void
perfetto_engine::start(mode m, int fd)
{
    if(m != mode::live_fd)
    {
        LOG_WARNING("perfetto_engine::start(mode, fd) requires mode::live_fd; "
                    "use start(mode, trace_sink&) for cached_interceptor");
        return;
    }

    if(is_system_backend())
    {
        // System backend owns the session; engine has nothing to drive.
        // Match perfetto.cpp:start() pre-refactor behaviour (early return).
        return;
    }

    if(m_running)
    {
        LOG_WARNING("perfetto_engine::start() called while a session is "
                    "already active; replacing it");
    }

    const auto pid = static_cast<pid_t>(::getpid());
    start_session(pid, fd, mode::live_fd);

    m_active_pid   = pid;
    m_running      = true;
    m_current_mode = mode::live_fd;
    m_active_sink  = nullptr;
}

void
perfetto_engine::start(mode m, trace_sink& sink)
{
    if(m != mode::cached_interceptor)
    {
        LOG_WARNING("perfetto_engine::start(mode, sink) requires "
                    "mode::cached_interceptor; use start(mode, fd) for live_fd");
        return;
    }

    if(is_system_backend())
    {
        // System backend can't drive an interceptor-routed cached session;
        // mirror live-mode early return.
        return;
    }

    if(m_running)
    {
        LOG_WARNING("perfetto_engine::start() called while a session is "
                    "already active; replacing it");
    }

    const auto pid = static_cast<pid_t>(::getpid());
    start_session(pid, /*fd=*/-1, mode::cached_interceptor);

    {
        std::lock_guard<std::mutex> lk{ m_collector_mutex };
        m_collected_bytes.clear();
    }
    // Reset the freeze publication so a subsequent preregister_pids on this
    // engine instance is required before the hot path will accept emissions.
    m_collected_bytes_frozen.store(false, std::memory_order_release);

    m_active_pid   = pid;
    m_running      = true;
    m_current_mode = mode::cached_interceptor;
    m_active_sink  = &sink;

    // Only one cached engine may be the SDK interceptor's drain target at a
    // time. The interceptor's ThreadLocalState caches the engine pointer at
    // first emit per thread; silently overwriting g_active_cached_engine here
    // would leave already-running worker threads pointing at a stale engine.
    perfetto_engine* prev =
        g_active_cached_engine.exchange(this, std::memory_order_acq_rel);
    if(prev != nullptr && prev != this)
    {
        LOG_WARNING("perfetto_engine: cached engine already active when start() was "
                    "called; replacing it. Worker threads attached to the prior engine "
                    "may emit into stale state until they exit.");
    }
}

void
perfetto_engine::stop()
{
    if(!m_running) return;

    const auto current_mode = m_current_mode;
    const auto pid          = m_active_pid;
    auto*      sink         = m_active_sink;

    ::perfetto::TracingSession* session = nullptr;
    {
        std::lock_guard<std::mutex> lk{ m_sessions_mutex };
        auto                        it = m_sessions.find(pid);
        if(it != m_sessions.end()) session = it->second.get();
    }

    // Clear the active-engine pointer only if we are still the active engine;
    // a sibling engine that took over via start() owns its own slot. Surface
    // the lifecycle violation when a sibling clobbered the global between
    // our start() and stop() -- silent degradation here ends in UAF in the
    // interceptor TLS reader.
    if(current_mode == mode::cached_interceptor)
    {
        auto*      expected = this;
        const bool cleared  = g_active_cached_engine.compare_exchange_strong(
            expected, nullptr, std::memory_order_acq_rel, std::memory_order_acquire);
        if(!cleared && expected != nullptr)
        {
            LOG_WARNING(
                "perfetto_engine::stop() saw g_active_cached_engine pointing at a "
                "different instance ({} vs this={}) -- overlapping cached engines "
                "are not supported and worker threads attached to the prior engine "
                "may emit into stale state until they exit",
                static_cast<const void*>(expected), static_cast<const void*>(this));
        }
    }

    if(session == nullptr)
    {
        m_running     = false;
        m_active_pid  = 0;
        m_active_sink = nullptr;
        // Drive the sink's finalize() even when no session ever materialised,
        // so cached-mode owners observe the same teardown contract as a
        // session that ran to completion (e.g. the system backend short-circuit
        // or an SDK init failure that left start() with no live session).
        if(current_mode == mode::cached_interceptor && sink != nullptr)
            std::visit([](auto& s) { s.finalize(); }, *sink);
        return;
    }

    LOG_DEBUG("Flushing the perfetto trace data...");
    std::exception_ptr first_exc{};
    try
    {
        LOG_DEBUG("Stopping the perfetto trace session (blocking)...");
        session_backend backend{};
        flush_and_stop_session(backend, *session);
    } catch(...)
    {
        // Capture but proceed so cached-mode owners still get drain + finalize;
        // the first observed exception is rethrown below per the documented
        // teardown contract.
        first_exc = std::current_exception();
    }

    m_running     = false;
    m_active_pid  = 0;
    m_active_sink = nullptr;

    if(current_mode != mode::cached_interceptor)
    {
        if(first_exc) std::rethrow_exception(first_exc);
        return;
    }

    // Cached-mode drain: move collected bytes out under the lock, then
    // feed them through the sink without holding the lock (sink calls can
    // be slow / throw).
    std::unordered_map<int, std::vector<char>> drained;
    {
        std::lock_guard<std::mutex> lk{ m_collector_mutex };
        drained.swap(m_collected_bytes);
    }

    if(sink == nullptr)
    {
        if(first_exc) std::rethrow_exception(first_exc);
        return;
    }

    const auto dropped = m_dropped_packet_count.exchange(0, std::memory_order_relaxed);
    if(dropped > 0)
    {
        LOG_WARNING("perfetto cached collector dropped {} packet(s) during the "
                    "session (most likely allocation pressure)",
                    dropped);
    }

    for(auto& entry : drained)
    {
        auto&      bytes      = entry.second;
        const auto source_pid = entry.first;
        if(bytes.empty()) continue;
        try
        {
            std::visit(
                [source_pid, &bytes](auto& s) {
                    s.on_source_drained(source_pid, std::move(bytes));
                },
                *sink);
        } catch(...)
        {
            if(!first_exc) first_exc = std::current_exception();
        }
    }

    // finalize() must run regardless of per-source drain failures, and a
    // finalize-thrown exception is captured the same way as a per-source one
    // so the rethrow below honors the documented "first exception wins"
    // contract whether the throw came from a sink drain or its finalize.
    try
    {
        std::visit([](auto& s) { s.finalize(); }, *sink);
    } catch(...)
    {
        if(!first_exc) first_exc = std::current_exception();
    }

    if(first_exc) std::rethrow_exception(first_exc);
}

std::vector<char>
perfetto_engine::read_trace(pid_t pid)
{
    ::perfetto::TracingSession* session = nullptr;
    {
        std::lock_guard<std::mutex> lk{ m_sessions_mutex };
        auto                        it = m_sessions.find(pid);
        if(it != m_sessions.end()) session = it->second.get();
    }

    if(session == nullptr) return {};
    return std::vector<char>{ session->ReadTraceBlocking() };
}

void
perfetto_engine::destroy_session(pid_t pid)
{
    std::lock_guard<std::mutex> lk{ m_sessions_mutex };
    auto                        it = m_sessions.find(pid);
    if(it != m_sessions.end()) it->second.reset();
}

bool
perfetto_engine::is_running() const noexcept
{
    return m_running;
}

void
perfetto_engine::forget_session(pid_t pid)
{
    std::lock_guard<std::mutex> lk{ m_sessions_mutex };
    auto                        it = m_sessions.find(pid);
    if(it != m_sessions.end())
    {
        // Deliberate leak: the parent process owns the underlying
        // TracingSession after fork; calling unique_ptr::reset() in the
        // child would double-free when the parent eventually disposes its
        // own copy. release() detaches our pointer without destroying.
        (void) it->second.release();
    }
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

void
perfetto_engine::preregister_pids(const std::vector<int>& source_pids)
{
    if(m_collected_bytes_frozen.load(std::memory_order_acquire))
    {
        LOG_ERROR("preregister_pids called after the collector map was frozen; "
                  "new pids cannot be added without restarting the engine");
        return;
    }

    {
        std::lock_guard<std::mutex> lk{ m_collector_mutex };
        for(int pid : source_pids)
        {
            auto& bytes = m_collected_bytes[pid];
            bytes.reserve(COLLECTED_BYTES_SLAB_SIZE);
        }
    }

    // Release publishes every collector_mutex-guarded map write above to any
    // worker thread that observes the frozen flag via acquire in
    // collect_packet_bytes -- the hot path then runs lock-free safely.
    m_collected_bytes_frozen.store(true, std::memory_order_release);
}

void
perfetto_engine::collect_packet_bytes(int pid, const void* data,
                                      std::size_t size) noexcept
{
    if(data == nullptr || size == 0) return;

    // Acquire pairs with preregister_pids' release: refuses emissions until
    // the map population is published, eliminating the data race the previous
    // "frozen by convention" comment relied on.
    if(!m_collected_bytes_frozen.load(std::memory_order_acquire))
    {
        LOG_ERROR("perfetto cached collector dropped packet for pid {} -- "
                  "preregister_pids has not run yet",
                  pid);
        return;
    }

    try
    {
        auto it = m_collected_bytes.find(pid);
        if(it == m_collected_bytes.end())
        {
            LOG_ERROR("perfetto cached collector dropped packet for unregistered pid {}",
                      pid);
            return;
        }

        // Lock-free hot path: the map structure is frozen (preregister_pids
        // populated all entries before emission started), and this pid's
        // vector has exactly one writer thread (the parser owning this pid).
        // The framing wraps each raw TracePacket in a length-delimited
        // Trace.packets field so concatenation forms a valid Trace proto.
        // Per-call reserve() is deliberately omitted -- the 8 MiB slab from
        // preregister_pids absorbs typical traces and vector's geometric
        // growth handles overruns without the O(N^2) re-copy cost that a
        // per-call exact-fit reserve would incur.
        auto& bytes = it->second;
        bytes.push_back(static_cast<char>(TRACE_PACKETS_TAG));
        append_varint(bytes, static_cast<std::uint64_t>(size));
        bytes.insert(bytes.end(), static_cast<const char*>(data),
                     static_cast<const char*>(data) + size);
    } catch(...)
    {
        // The SDK interceptor callback frame is not exception-safe; bad_alloc
        // from vector growth (or any other throwing operation) is captured
        // here so it cannot unwind through Perfetto SDK code that does not
        // expect C++ exceptions. The per-packet LOG_ERROR can flood stderr
        // under sustained pressure, so the count is also tracked and
        // surfaced once at stop() time.
        m_dropped_packet_count.fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR("perfetto cached collector dropped packet for pid {} on internal "
                  "exception",
                  pid);
    }
}

}  // namespace rocprofsys::core
