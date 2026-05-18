// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto_engine.hpp"

#include "core/config.hpp"
#include "core/perfetto.hpp"  // brings ROCPROFSYS_PERFETTO_CATEGORIES (TrackEvent type)
#include "logger/debug.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <unistd.h>

namespace rocprofsys
{
namespace core
{
namespace
{
// Thread-local pid tag consumed by future cached-mode interceptor TLS (D4).
// Stored in TU scope so the static accessors can reach it without exposing
// the TLS in the header.
thread_local int t_emitting_pid = 0;
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

    static std::once_flag s_sdk_init_flag;
    static bool           s_sdk_init_succeeded;

    bool is_system_backend() const noexcept
    {
        return cfg.backend != engine_config::backend_t::inprocess;
    }

    void build_trace_config()
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
    p_->build_trace_config();

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
}

void
perfetto_engine::start(mode m, int fd)
{
    (void) m;  // only mode::live_fd defined in slice B

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

    p_->active_pid = pid;
    p_->running    = true;
}

void
perfetto_engine::stop()
{
    if(!p_->running) return;  // RF6

    auto pid = p_->active_pid;

    ::perfetto::TracingSession* session = nullptr;
    {
        std::lock_guard<std::mutex> lk{ p_->sessions_mutex };
        auto                        it = p_->sessions.find(pid);
        if(it != p_->sessions.end()) session = it->second.get();
    }

    p_->running    = false;
    p_->active_pid = 0;

    if(session == nullptr) return;

    LOG_DEBUG("Flushing the perfetto trace data...");
    ::perfetto::TrackEvent::Flush();
    session->FlushBlocking();

    LOG_DEBUG("Stopping the perfetto trace session (blocking)...");
    session->StopBlocking();
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
}  // namespace core
}  // namespace rocprofsys
