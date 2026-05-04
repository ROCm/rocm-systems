// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Meyers-singleton definitions for services::sampling() and services::causal_sampling().
// DEC-10: both accessors return the same default_sampling_service instance.
// DEC-11: causal_sampling() is a thin alias — same object, causal path throws at setup().
//
// Also owns the thread-local signal-handler state pointers and the
// rocprofsys_sampling_signal_handler free function (ODR: exactly one definition).
//
// Lives in library/ so it can include sampling/default_policies.hpp which
// depends on main-library symbols (perf.hpp, tracing.hpp, trace_cache).
// Must NOT be compiled into standalone test binaries.

#if defined(__linux__)
#    include "sampling/default_policies.hpp"
#endif

#if defined(ROCPROFSYS_USE_PAPI)
#    include "library/papi_bridge.hpp"
#endif

#include "core/perf.hpp"
#include "core/state.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "library/pmc/sampler.hpp"
#include "library/sampling_service_instantiation.hpp"
#include "library/thread_info.hpp"

#include "sampling/sampling_service.hpp"
#include <linux/perf_event.h>
#include <mutex>

namespace rocprofsys::services
{

#if defined(__linux__)

namespace
{
rocprofsys::sampling::sampling_config
make_production_config()
{
    rocprofsys::sampling::sampling_config cfg;
    cfg.realtime_signal      = rocprofsys::get_sampling_realtime_signal();
    cfg.cputime_signal       = rocprofsys::get_sampling_cputime_signal();
    cfg.overflow_signal      = rocprofsys::get_sampling_overflow_signal();
    cfg.realtime_freq        = rocprofsys::get_sampling_realtime_freq();
    cfg.cputime_freq         = rocprofsys::get_sampling_cputime_freq();
    cfg.realtime_delay       = rocprofsys::get_sampling_realtime_delay();
    cfg.cputime_delay        = rocprofsys::get_sampling_cputime_delay();
    cfg.overflow_freq        = rocprofsys::get_sampling_overflow_freq();
    cfg.duration             = rocprofsys::get_sampling_duration();
    cfg.use_causal           = rocprofsys::get_use_causal();
    cfg.trace_legacy         = rocprofsys::get_use_sampling_trace_legacy();
    cfg.use_perfetto         = rocprofsys::config::get_use_perfetto();
    cfg.perfetto_annotations = rocprofsys::config::get_perfetto_annotations();
    cfg.use_process_sampling = rocprofsys::config::get_use_process_sampling();
    cfg.use_amd_smi          = rocprofsys::config::get_use_amd_smi();
    auto event_opt =
        rocprofsys::get_setting_value<std::string>("ROCPROFSYS_SAMPLING_OVERFLOW_EVENT");
    if(event_opt) cfg.overflow_event = *event_opt;
#    if defined(ROCPROFSYS_USE_PAPI)
    rocprofsys::papi_bridge::detail::ensure_initialized();
    cfg.hw_counter_labels = rocprofsys::papi_bridge::get_labels();
#    endif
    return cfg;
}

rocprofsys::sampling::sampling_callbacks
make_production_callbacks()
{
    rocprofsys::sampling::sampling_callbacks cb;
    cb.resolve_signals = [](int64_t tid) {
        return rocprofsys::get_sampling_signals(tid);
    };
    cb.get_sys_tid = []() -> int64_t {
        return static_cast<int64_t>(rocprofsys::threading::get_sys_tid());
    };
    cb.resolve_thread_info =
        [](int64_t tid) -> std::optional<rocprofsys::sampling::thread_info_data> {
        auto const& info = rocprofsys::thread_info::get(tid, rocprofsys::SequentTID);
        if(!info)
        {
            auto const& alt = rocprofsys::thread_info::get(tid, rocprofsys::InternalTID);
            if(!alt) return std::nullopt;
            return rocprofsys::sampling::thread_info_data{
                static_cast<std::size_t>(alt->index_data->system_value),
                static_cast<std::size_t>(alt->index_data->sequent_value),
                alt->get_start(), alt->get_stop()
            };
        }
        return rocprofsys::sampling::thread_info_data{
            static_cast<std::size_t>(info->index_data->system_value),
            static_cast<std::size_t>(info->index_data->sequent_value), info->get_start(),
            info->get_stop()
        };
    };
    cb.is_thread_eligible = [](int64_t tid) {
        if(rocprofsys::get_thread_state() == rocprofsys::ThreadState::Disabled)
            return false;
        auto const& info = rocprofsys::thread_info::get(tid, rocprofsys::SequentTID);
        return !(info && info->is_offset);
    };
    cb.register_sampling_categories = []() {
        static std::once_flag flag;
        std::call_once(flag, [] {
            auto& reg = rocprofsys::trace_cache::get_metadata_registry();
            reg.add_string("timer_sampling");
            reg.add_string("overflow_sampling");
        });
    };
    cb.register_thread_info = [](int ppid, int pid, std::size_t sys_id) {
        rocprofsys::trace_cache::get_metadata_registry().add_thread_info(
            { static_cast<pid_t>(ppid), static_cast<pid_t>(pid), sys_id, 0, 0, "{}" });
    };
    cb.register_track = [](std::string name, std::size_t sys_id) {
        rocprofsys::trace_cache::get_metadata_registry().add_track(
            { std::move(name), sys_id, "{}" });
    };
    cb.configure_overflow_pe_attr = [](void* pe_attr_ptr, std::string const& event_name,
                                       double freq) {
        auto* pe_attr = static_cast<perf_event_attr*>(pe_attr_ptr);
        rocprofsys::perf::config_overflow_sampling(*pe_attr, event_name, freq);
    };
    cb.postfork_parent_reinit = []() { rocprofsys::pmc::postfork_parent_reinit(); };
    cb.postfork_child_cleanup = []() { rocprofsys::pmc::postfork_child_cleanup(); };
#    if defined(ROCPROFSYS_USE_PAPI)
    cb.setup_hw_counters    = [](int64_t tid) { rocprofsys::papi_bridge::setup(tid); };
    cb.teardown_hw_counters = [](int64_t tid) { rocprofsys::papi_bridge::teardown(tid); };
    cb.read_hw_counters     = &rocprofsys::papi_bridge::read;
#    endif
    return cb;
}
}  // namespace

rocprofsys::sampling::default_sampling_service&
sampling()
{
    static rocprofsys::sampling::default_sampling_service instance{
        make_production_config(), make_production_callbacks()
    };
    return instance;
}

rocprofsys::sampling::default_sampling_service&
causal_sampling()
{
    return sampling();
}

// ── Thin caller-facing wrappers ──────────────────────────────────────────────
// Defined here (the single TU with the full sampling_service template
// instantiation) so caller TUs can stay free of main-lib hook deps.

std::set<int>
sampling_setup(int64_t tid)
{
    return sampling().setup(tid);
}

std::set<int>
sampling_shutdown(int64_t tid)
{
    return sampling().shutdown(tid);
}

void
sampling_block_samples()
{
    sampling().block_samples();
}

void
sampling_unblock_samples()
{
    sampling().unblock_samples();
}

void
sampling_block_signals(std::set<int> sigs)
{
    sampling().block_signals(std::move(sigs));
}

void
sampling_unblock_signals(std::set<int> sigs)
{
    sampling().unblock_signals(std::move(sigs));
}

bool
sampling_is_setup_for_current_thread()
{
    using tls_t =
        rocprofsys::sampling::tl_state<rocprofsys::sampling::default_sampling_policies>;
    return tls_t::sampler != nullptr;
}

void
sampling_pause()
{
    sampling().pause();
}

void
sampling_resume()
{
    sampling().resume();
}

void
sampling_postfork_parent_reinit()
{
    sampling().postfork_parent_reinit();
}

void
sampling_postfork_child_cleanup()
{
    sampling().postfork_child_cleanup();
}

void
sampling_enter_child_process_mode()
{
    sampling().enter_child_process_mode();
}

std::set<int>
causal_sampling_setup(int64_t tid)
{
    return causal_sampling().setup(tid);
}

std::set<int>
causal_sampling_shutdown(int64_t tid)
{
    return causal_sampling().shutdown(tid);
}

void
causal_sampling_block_signals(std::set<int> sigs)
{
    causal_sampling().block_signals(std::move(sigs));
}

void
causal_sampling_unblock_signals(std::set<int> sigs)
{
    causal_sampling().unblock_signals(std::move(sigs));
}

void
causal_sampling_pause()
{
    causal_sampling().pause();
}

void
causal_sampling_resume()
{
    causal_sampling().resume();
}

void
sampling_shutdown_in_child_mode(int64_t tid)
{
    auto& svc = sampling();
    svc.enter_child_process_mode();
    svc.shutdown(tid);
}

#endif

}  // namespace rocprofsys::services

// Typed thread-local signal-handler state lives in sampling/policies/tl_state.hpp
// (definitions are inline thread_local template members).

#if defined(__linux__)

// ── Signal handler definition (single TU) ──────────────────────────────────
// Installed via sigaction in real_timer_trigger::start().
// The template body lives in sampling/src/sampling_signal_handler_impl.hpp;
// this extern "C" trampoline delegates to it with the production Policies.

#    include "sampling/src/sampling_signal_handler_impl.hpp"

#    include <csignal>

extern "C" void
rocprofsys_sampling_signal_handler(int sig, siginfo_t* /*info*/, void* ucontext)
{
    rocprofsys::sampling::sampling_signal_handler_body<
        rocprofsys::sampling::default_sampling_policies>(
        sig, ucontext, rocprofsys::services::sampling());
}

#endif  // __linux__
