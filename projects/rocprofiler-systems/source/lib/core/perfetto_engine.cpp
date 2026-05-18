// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto_engine.hpp"

#include "core/config.hpp"
#include "core/perfetto.hpp"  // brings ROCPROFSYS_PERFETTO_CATEGORIES (TrackEvent type)
#include "core/perfetto_sinks.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <exception>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

namespace rocprofsys
{
namespace core
{
namespace
{
// Thread-local pid tag consumed by the cached-mode interceptor TLS (D4).
// Stored in TU scope so the static accessors can reach it without exposing
// the TLS in the header.
thread_local int t_emitting_pid = 0;

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
constexpr const char* k_cached_interceptor_name = "test_interceptor";

class cached_interceptor
: public ::perfetto::Interceptor<cached_interceptor>
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
        int              pid    = 0;
    };

    static void OnTracePacket(InterceptorContext context)
    {
        auto& tls = context.GetThreadLocalState();
        if(tls.engine == nullptr) return;

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
    explicit impl(engine_config c) : cfg{ std::move(c) } {}

    engine_config           cfg;
    ::perfetto::TraceConfig trace_cfg{};
    std::mutex              sessions_mutex{};
    std::unordered_map<pid_t, std::unique_ptr<::perfetto::TracingSession>> sessions{};
    pid_t                                                                  active_pid{ 0 };
    bool                                                                   running{ false };

    perfetto_engine::mode current_mode{ perfetto_engine::mode::live_fd };
    trace_sink*           active_sink{ nullptr };

    // Cached-mode per-pid byte collector. Protected by collector_mutex.
    std::mutex                                       collector_mutex{};
    std::unordered_map<int, std::vector<char>>      collected_bytes{};

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
            ds_cfg->mutable_interceptor_config()->set_name(k_cached_interceptor_name);
        }
    }
};

std::once_flag perfetto_engine::impl::s_sdk_init_flag;
bool           perfetto_engine::impl::s_sdk_init_succeeded = false;

// ----------------------------------------------------------------------------
// perfetto_engine — public API
// ----------------------------------------------------------------------------

perfetto_engine::perfetto_engine(engine_config cfg)
: p_(std::make_unique<impl>(std::move(cfg)))
{}

perfetto_engine::~perfetto_engine()
{
    if(p_ && p_->running)
    {
        // Last-ditch teardown so the engine never outlives its
        // TracingSession. Destructors must not throw — swallow.
        try
        {
            stop();
        }
        catch(...)
        {}
    }
}

void
perfetto_engine::init_sdk()
{
    // build_trace_config defers mode selection to start(); init_sdk only
    // needs the SDK Initialize/Register calls. Use live_fd here just to
    // produce a valid base config; start() rebuilds with the correct mode.
    p_->build_trace_config(mode::live_fd);

    std::call_once(impl::s_sdk_init_flag, [this]() {
        ::perfetto::TracingInitArgs args{};
        args.shmem_size_hint_kb = p_->cfg.shmem_size_hint_kb;

        if(p_->cfg.backend != engine_config::backend_t::inprocess)
            args.backends |= ::perfetto::kSystemBackend;
        if(p_->cfg.backend != engine_config::backend_t::system)
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
    desc.set_name(k_cached_interceptor_name);
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

    if(p_->is_system_backend())
    {
        // System backend owns the session; engine has nothing to drive.
        // Match perfetto.cpp:start() pre-refactor behaviour (early return).
        return;
    }

    if(p_->running)
    {
        LOG_WARNING("perfetto_engine::start() called while a session is "
                    "already active; replacing it");
    }

    p_->build_trace_config(mode::live_fd);

    const auto pid = static_cast<pid_t>(::getpid());

    std::lock_guard<std::mutex> lk{ p_->sessions_mutex };
    auto&                       slot = p_->sessions[pid];
    slot                             = ::perfetto::Tracing::NewTrace();
    slot->SetOnErrorCallback([](::perfetto::TracingError err) {
        if(err.code == ::perfetto::TracingError::kTracingFailed)
            LOG_WARNING("Perfetto encountered a tracing error: {}", err.message);
    });
    slot->Setup(p_->trace_cfg, fd);
    slot->StartBlocking();

    p_->active_pid   = pid;
    p_->running      = true;
    p_->current_mode = mode::live_fd;
    p_->active_sink  = nullptr;
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

    if(p_->is_system_backend())
    {
        // System backend can't drive an interceptor-routed cached session;
        // mirror live-mode early return.
        return;
    }

    if(p_->running)
    {
        LOG_WARNING("perfetto_engine::start() called while a session is "
                    "already active; replacing it");
    }

    p_->build_trace_config(mode::cached_interceptor);

    const auto pid = static_cast<pid_t>(::getpid());

    {
        std::lock_guard<std::mutex> lk{ p_->sessions_mutex };
        auto&                       slot = p_->sessions[pid];
        slot                             = ::perfetto::Tracing::NewTrace();
        slot->SetOnErrorCallback([](::perfetto::TracingError err) {
            if(err.code == ::perfetto::TracingError::kTracingFailed)
                LOG_WARNING("Perfetto encountered a tracing error: {}", err.message);
        });
        slot->Setup(p_->trace_cfg);
        slot->StartBlocking();
    }

    {
        std::lock_guard<std::mutex> lk{ p_->collector_mutex };
        p_->collected_bytes.clear();
    }

    p_->active_pid   = pid;
    p_->running      = true;
    p_->current_mode = mode::cached_interceptor;
    p_->active_sink  = &sink;

    g_active_cached_engine.store(this, std::memory_order_release);
}

void
perfetto_engine::stop()
{
    if(!p_->running) return;  // RF6

    const auto current_mode = p_->current_mode;
    const auto pid          = p_->active_pid;
    auto*      sink         = p_->active_sink;

    ::perfetto::TracingSession* session = nullptr;
    {
        std::lock_guard<std::mutex> lk{ p_->sessions_mutex };
        auto                        it = p_->sessions.find(pid);
        if(it != p_->sessions.end()) session = it->second.get();
    }

    p_->running     = false;
    p_->active_pid  = 0;
    p_->active_sink = nullptr;

    if(current_mode == mode::cached_interceptor)
    {
        g_active_cached_engine.store(nullptr, std::memory_order_release);
    }

    if(session == nullptr) return;

    LOG_DEBUG("Flushing the perfetto trace data...");
    ::perfetto::TrackEvent::Flush();
    session->FlushBlocking();

    LOG_DEBUG("Stopping the perfetto trace session (blocking)...");
    session->StopBlocking();

    if(current_mode != mode::cached_interceptor) return;

    // Cached-mode drain: move collected bytes out under the lock, then
    // feed them through the sink without holding the lock (sink calls can
    // be slow / throw).
    std::unordered_map<int, std::vector<char>> drained;
    {
        std::lock_guard<std::mutex> lk{ p_->collector_mutex };
        drained.swap(p_->collected_bytes);
    }

    if(sink == nullptr) return;

    std::exception_ptr first_exc{};
    for(auto& [source_pid, bytes] : drained)
    {
        if(bytes.empty()) continue;
        try
        {
            sink->on_source_drained(source_pid, std::move(bytes));
        }
        catch(...)
        {
            if(!first_exc) first_exc = std::current_exception();
        }
    }

    sink->finalize();

    if(first_exc) std::rethrow_exception(first_exc);
}

std::vector<char>
perfetto_engine::read_trace(pid_t pid)
{
    ::perfetto::TracingSession* session = nullptr;
    {
        std::lock_guard<std::mutex> lk{ p_->sessions_mutex };
        auto                        it = p_->sessions.find(pid);
        if(it != p_->sessions.end()) session = it->second.get();
    }

    if(session == nullptr) return {};
    return std::vector<char>{ session->ReadTraceBlocking() };
}

void
perfetto_engine::release_session(pid_t pid)
{
    std::lock_guard<std::mutex> lk{ p_->sessions_mutex };
    auto                        it = p_->sessions.find(pid);
    if(it != p_->sessions.end()) it->second.reset();
}

bool
perfetto_engine::is_running() const noexcept
{
    return p_->running;
}

std::unique_ptr<::perfetto::TracingSession>&
perfetto_engine::session_ref(pid_t pid)
{
    std::lock_guard<std::mutex> lk{ p_->sessions_mutex };
    return p_->sessions[pid];
}

void
perfetto_engine::set_emitting_pid(int pid) noexcept
{
    t_emitting_pid = pid;
}

int
perfetto_engine::get_emitting_pid() noexcept
{
    return t_emitting_pid;
}

void
perfetto_engine::collect_packet_bytes(int pid, const void* data, std::size_t size)
{
    if(data == nullptr || size == 0) return;

    // Wrap each raw TracePacket in the length-delimited Trace.packets
    // field header so concatenation forms a valid Trace proto.
    // Trace.packets has field number 1, wire type 2 (LEN). Tag byte =
    // (1 << 3) | 2 = 0x0A. Length is varint-encoded.
    std::array<char, 11> header{};
    header[0]      = 0x0A;
    std::size_t hl = 1;
    {
        std::size_t v = size;
        while(v >= 0x80)
        {
            header[hl++] = static_cast<char>((v & 0x7F) | 0x80);
            v >>= 7;
        }
        header[hl++] = static_cast<char>(v);
    }

    std::lock_guard<std::mutex> lk{ p_->collector_mutex };
    auto&                       bytes = p_->collected_bytes[pid];
    bytes.reserve(bytes.size() + hl + size);
    bytes.insert(bytes.end(), header.data(), header.data() + hl);
    bytes.insert(bytes.end(), static_cast<const char*>(data),
                 static_cast<const char*>(data) + size);
}

}  // namespace core
}  // namespace rocprofsys
