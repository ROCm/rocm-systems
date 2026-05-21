// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/engine.hpp"
#include <cstdint>

#include "core/config.hpp"
#include "core/perfetto/driver.hpp"  // brings ROCPROFSYS_PERFETTO_CATEGORIES (TrackEvent type)
#include "core/perfetto/packet_framing.hpp"
#include "core/perfetto/sinks.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <unistd.h>

namespace rocprofsys
{
namespace core
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

    return out;
}

// ----------------------------------------------------------------------------
// Cached-mode Interceptor
// ----------------------------------------------------------------------------

namespace
{
// Perfetto SDK keeps interceptors experimental: TracingMuxerImpl::
// RegisterInterceptor (perfetto.cc:~37265) silently rejects descriptors
// whose name is not one of {"test_interceptor", "console", "etwexport"}.
// The check predates upstreamable knobs and lives inside vendored
// submodule code we don't fork; using "test_interceptor" keeps the
// engine wired up without patching the SDK. The name is an internal
// binding key only — registration side and TraceConfig.interceptor_config
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

        tls.engine->collect_packet_bytes(tls.pid, context.packet_data.data,
                                         context.packet_data.size);
    }
};

}  // namespace

// ----------------------------------------------------------------------------
// perfetto_engine::impl
// ----------------------------------------------------------------------------

struct perfetto_engine::impl
{
    explicit impl(engine_config c)
    : cfg{ std::move(c) }
    {}

    engine_config           cfg;
    ::perfetto::TraceConfig trace_cfg{};
    std::mutex              sessions_mutex{};
    std::unordered_map<pid_t, std::unique_ptr<::perfetto::TracingSession>> sessions{};
    pid_t active_pid{ 0 };
    bool  running{ false };

    perfetto_engine::mode current_mode{ perfetto_engine::mode::live_fd };
    trace_sink*           active_sink{ nullptr };

    // Cached-mode per-pid byte collector. Map structure is mutated under
    // collector_mutex during preregister_pids; once that call has completed,
    // the structure is frozen and `collected_bytes_frozen` is released so
    // worker threads can lock-free `find()` without racing the population.
    std::mutex                                 collector_mutex{};
    std::unordered_map<int, std::vector<char>> collected_bytes{};
    std::atomic<bool>                          collected_bytes_frozen{ false };

    static std::once_flag s_sdk_init_flag;
    static bool           s_sdk_init_succeeded;

    bool is_system_backend() const noexcept
    {
        return cfg.backend != engine_config::backend_t::inprocess;
    }

    void build_trace_config(perfetto_engine::mode m)
    {
        trace_cfg = ::perfetto::TraceConfig{};

        const auto policy =
            (cfg.fill_policy == engine_config::fill_policy_t::discard)
                ? ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_DISCARD
                : ::perfetto::protos::gen::
                      TraceConfig_BufferConfig_FillPolicy_RING_BUFFER;
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
    }
};

std::once_flag perfetto_engine::impl::s_sdk_init_flag;
bool           perfetto_engine::impl::s_sdk_init_succeeded = false;

// ----------------------------------------------------------------------------
// perfetto_engine — public API
// ----------------------------------------------------------------------------

perfetto_engine::perfetto_engine(engine_config cfg)
: m_impl(std::make_unique<impl>(std::move(cfg)))
{}

perfetto_engine::~perfetto_engine()
{
    if(m_impl && m_impl->running)
    {
        // Last-ditch teardown so the engine never outlives its
        // TracingSession. Destructors must not throw — swallow.
        try
        {
            stop();
        } catch(...)
        {}
    }
}

void
perfetto_engine::init_sdk()
{
    // init_sdk only needs the SDK Initialize/Register calls; the TraceConfig
    // is built per-mode in start().
    std::call_once(impl::s_sdk_init_flag, [this]() {
        ::perfetto::TracingInitArgs args{};
        args.shmem_size_hint_kb = m_impl->cfg.shmem_size_hint_kb;

        if(m_impl->cfg.backend != engine_config::backend_t::inprocess)
            args.backends |= ::perfetto::kSystemBackend;
        if(m_impl->cfg.backend != engine_config::backend_t::system)
            args.backends |= ::perfetto::kInProcessBackend;

        ::perfetto::Tracing::Initialize(args);
        ::perfetto::TrackEvent::Register();

        impl::s_sdk_init_succeeded = true;
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

    if(m_impl->is_system_backend())
    {
        // System backend owns the session; engine has nothing to drive.
        // Match perfetto.cpp:start() pre-refactor behaviour (early return).
        return;
    }

    if(m_impl->running)
    {
        LOG_WARNING("perfetto_engine::start() called while a session is "
                    "already active; replacing it");
    }

    m_impl->build_trace_config(mode::live_fd);

    const auto pid = static_cast<pid_t>(::getpid());

    std::lock_guard<std::mutex> lk{ m_impl->sessions_mutex };
    auto&                       slot = m_impl->sessions[pid];
    slot                             = ::perfetto::Tracing::NewTrace();
    slot->SetOnErrorCallback([](::perfetto::TracingError err) {
        if(err.code == ::perfetto::TracingError::kTracingFailed)
            LOG_WARNING("Perfetto encountered a tracing error: {}", err.message);
    });
    slot->Setup(m_impl->trace_cfg, fd);
    slot->StartBlocking();

    m_impl->active_pid   = pid;
    m_impl->running      = true;
    m_impl->current_mode = mode::live_fd;
    m_impl->active_sink  = nullptr;
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

    if(m_impl->is_system_backend())
    {
        // System backend can't drive an interceptor-routed cached session;
        // mirror live-mode early return.
        return;
    }

    if(m_impl->running)
    {
        LOG_WARNING("perfetto_engine::start() called while a session is "
                    "already active; replacing it");
    }

    m_impl->build_trace_config(mode::cached_interceptor);

    const auto pid = static_cast<pid_t>(::getpid());

    {
        std::lock_guard<std::mutex> lk{ m_impl->sessions_mutex };
        auto&                       slot = m_impl->sessions[pid];
        slot                             = ::perfetto::Tracing::NewTrace();
        slot->SetOnErrorCallback([](::perfetto::TracingError err) {
            if(err.code == ::perfetto::TracingError::kTracingFailed)
                LOG_WARNING("Perfetto encountered a tracing error: {}", err.message);
        });
        slot->Setup(m_impl->trace_cfg);
        slot->StartBlocking();
    }

    {
        std::lock_guard<std::mutex> lk{ m_impl->collector_mutex };
        m_impl->collected_bytes.clear();
    }
    // Reset the freeze publication so a subsequent preregister_pids on this
    // engine instance is required before the hot path will accept emissions.
    m_impl->collected_bytes_frozen.store(false, std::memory_order_release);

    m_impl->active_pid   = pid;
    m_impl->running      = true;
    m_impl->current_mode = mode::cached_interceptor;
    m_impl->active_sink  = &sink;

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
    if(!m_impl->running) return;

    const auto current_mode = m_impl->current_mode;
    const auto pid          = m_impl->active_pid;
    auto*      sink         = m_impl->active_sink;

    ::perfetto::TracingSession* session = nullptr;
    {
        std::lock_guard<std::mutex> lk{ m_impl->sessions_mutex };
        auto                        it = m_impl->sessions.find(pid);
        if(it != m_impl->sessions.end()) session = it->second.get();
    }

    // Clear the active-engine pointer only if we are still the active engine;
    // a sibling engine that took over via start() owns its own slot.
    if(current_mode == mode::cached_interceptor)
    {
        auto* expected = this;
        g_active_cached_engine.compare_exchange_strong(
            expected, nullptr, std::memory_order_acq_rel, std::memory_order_acquire);
    }

    if(session == nullptr)
    {
        m_impl->running     = false;
        m_impl->active_pid  = 0;
        m_impl->active_sink = nullptr;
        // Drive the sink's finalize() even when no session ever materialised,
        // so cached-mode owners observe the same teardown contract as a
        // session that ran to completion (e.g. the system backend short-circuit
        // or an SDK init failure that left start() with no live session).
        if(current_mode == mode::cached_interceptor && sink != nullptr)
            std::visit([](auto& s) { s.finalize(); }, *sink);
        return;
    }

    LOG_DEBUG("Flushing the perfetto trace data...");
    ::perfetto::TrackEvent::Flush();
    session->FlushBlocking();

    LOG_DEBUG("Stopping the perfetto trace session (blocking)...");
    session->StopBlocking();

    // Clear running state only after StopBlocking() returns successfully —
    // if FlushBlocking or StopBlocking threw, the destructor catch-all (which
    // checks running) gets a second chance to force-stop the session.
    m_impl->running     = false;
    m_impl->active_pid  = 0;
    m_impl->active_sink = nullptr;

    if(current_mode != mode::cached_interceptor) return;

    // Cached-mode drain: move collected bytes out under the lock, then
    // feed them through the sink without holding the lock (sink calls can
    // be slow / throw).
    std::unordered_map<int, std::vector<char>> drained;
    {
        std::lock_guard<std::mutex> lk{ m_impl->collector_mutex };
        drained.swap(m_impl->collected_bytes);
    }

    if(sink == nullptr) return;

    std::exception_ptr first_exc{};
    for(auto& [source_pid, bytes] : drained)
    {
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
        std::lock_guard<std::mutex> lk{ m_impl->sessions_mutex };
        auto                        it = m_impl->sessions.find(pid);
        if(it != m_impl->sessions.end()) session = it->second.get();
    }

    if(session == nullptr) return {};
    return std::vector<char>{ session->ReadTraceBlocking() };
}

void
perfetto_engine::destroy_session(pid_t pid)
{
    std::lock_guard<std::mutex> lk{ m_impl->sessions_mutex };
    auto                        it = m_impl->sessions.find(pid);
    if(it != m_impl->sessions.end()) it->second.reset();
}

bool
perfetto_engine::is_running() const noexcept
{
    return m_impl->running;
}

void
perfetto_engine::forget_session(pid_t pid)
{
    std::lock_guard<std::mutex> lk{ m_impl->sessions_mutex };
    auto                        it = m_impl->sessions.find(pid);
    if(it != m_impl->sessions.end()) (void) it->second.release();
}

void
set_emitting_pid(int pid) noexcept
{
    t_emitting_pid = pid;
}

int
get_emitting_pid() noexcept
{
    return t_emitting_pid;
}

// 8 MiB up-front reservation per pid lets the hot path append packets with
// amortized O(1) push_back while typically avoiding any reallocation —
// each parser thread emits ~1-10 MiB of framed bytes for a non-trivial
// workload. Sized to dominate the typical case; larger traces still grow
// geometrically thereafter.
namespace
{
constexpr std::size_t COLLECTED_BYTES_SLAB_SIZE = std::size_t{ 8 } * 1024 * 1024;
}

void
perfetto_engine::preregister_pids(const std::vector<int>& source_pids)
{
    if(m_impl->collected_bytes_frozen.load(std::memory_order_acquire))
    {
        LOG_ERROR("preregister_pids called after the map was frozen by the first "
                  "emission; new pids cannot be added without restarting the engine");
        return;
    }

    {
        std::lock_guard<std::mutex> lk{ m_impl->collector_mutex };
        for(int pid : source_pids)
        {
            auto& bytes = m_impl->collected_bytes[pid];
            bytes.reserve(COLLECTED_BYTES_SLAB_SIZE);
        }
    }

    // Release publishes every collector_mutex-guarded map write above to any
    // worker thread that observes the frozen flag via acquire in
    // collect_packet_bytes — the hot path then runs lock-free safely.
    m_impl->collected_bytes_frozen.store(true, std::memory_order_release);
}

void
perfetto_engine::collect_packet_bytes(int pid, const void* data,
                                      std::size_t size) noexcept
{
    if(data == nullptr || size == 0) return;

    // Acquire pairs with preregister_pids' release: refuses emissions until
    // the map population is published, eliminating the data race the previous
    // "frozen by convention" comment relied on.
    if(!m_impl->collected_bytes_frozen.load(std::memory_order_acquire))
    {
        LOG_ERROR("perfetto cached collector dropped packet for pid {} — "
                  "preregister_pids has not run yet",
                  pid);
        return;
    }

    try
    {
        auto it = m_impl->collected_bytes.find(pid);
        if(it == m_impl->collected_bytes.end())
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
        // Per-call reserve() is deliberately omitted — the 8 MiB slab from
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
        // expect C++ exceptions.
        LOG_ERROR("perfetto cached collector dropped packet for pid {} on internal "
                  "exception",
                  pid);
    }
}

}  // namespace core
}  // namespace rocprofsys
