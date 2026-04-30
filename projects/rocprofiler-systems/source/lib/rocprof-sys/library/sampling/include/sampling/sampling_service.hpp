// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/backtrace_record.hpp"
#include "sampling/platform_guard.hpp"
#include "sampling/platform_traits.hpp"
#include "sampling/sampling_callbacks.hpp"
#include "sampling/sampling_config.hpp"
#include "sampling/sampling_policies.hpp"
#include "sampling/src/pause_interval_registry.hpp"
#include "sampling/src/sampling_duration_controller.hpp"
#include "sampling/src/thread_sampler_state.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

namespace rocprofsys::sampling
{

// sampling_service<Policies> — the primary public surface.
// No virtual functions; all polymorphism via template policy parameters (D7).
// The 6 free functions in the old sampling.hpp are replaced by methods here (D4).
// Callers use the services::sampling() Meyers singleton accessor (DEC-10).
//
// Policies bundles 10 strategy types as nested using-aliases (P4 — single
// source of truth). Lifecycle orchestration (setup_wiring, shutdown_wiring,
// emit_resolved, postfork_*) is implemented directly on this class via
// private do_* methods; tests change behaviour by substituting the 10
// strategy policies, not by swapping orchestration.
template <class Policies>
class sampling_service
{
public:
    using policies             = Policies;
    using unwinder             = typename Policies::unwinder;
    using offload              = typename Policies::offload;
    using trace_sink           = typename Policies::trace_sink;
    using timer_trigger        = typename Policies::timer_trigger;
    using overflow_trigger     = typename Policies::overflow_trigger;
    using clock                = typename Policies::clock;
    using signal_dispatcher    = typename Policies::signal_dispatcher;
    using perfetto_sink        = typename Policies::perfetto_sink;
    using fatal_error          = typename Policies::fatal_error;
    using thread_info_resolver = typename Policies::thread_info_resolver;

    using thread_state_t        = thread_sampler_state<Policies>;
    using pause_registry_t      = pause_interval_registry<clock>;
    using duration_controller_t = sampling_duration_controller<clock>;

    sampling_service(sampling_config config, sampling_callbacks callbacks);
    ~sampling_service();

    sampling_service(sampling_service const&)            = delete;
    sampling_service& operator=(sampling_service const&) = delete;
    sampling_service(sampling_service&&)                 = delete;
    sampling_service& operator=(sampling_service&&)      = delete;

    // ----- driver API (replaces sampling.hpp free functions) -----
    std::set<int> setup(int64_t tid);
    std::set<int> shutdown(int64_t tid);

    void block_samples();
    void unblock_samples();
    void block_signals(std::set<int> sigs = {});
    void unblock_signals(std::set<int> sigs = {});

    void pause();
    void resume();

    void postfork_parent_reinit();
    void postfork_child_cleanup();

    std::set<int> get_signal_types(int64_t tid);

    // ----- introspection -----
    [[nodiscard]] size_t dropped_samples() const noexcept;
    [[nodiscard]] bool   is_paused() const noexcept;
    [[nodiscard]] bool   is_blocked() const noexcept;

    // ----- policy + state accessors (used by production hooks and tests) -----
    unwinder&              get_unwinder() noexcept { return unwinder_; }
    offload&               get_offload() noexcept { return offload_; }
    trace_sink&            get_trace_sink() noexcept { return trace_sink_; }
    signal_dispatcher&     signal_dispatcher_ref() noexcept { return signal_dispatcher_; }
    perfetto_sink&         get_perfetto_sink() noexcept { return perfetto_sink_; }
    clock&                 get_clock() noexcept { return clock_; }
    fatal_error&           get_fatal_error() noexcept { return fatal_; }
    pause_registry_t&      pause_registry() noexcept { return pause_registry_; }
    duration_controller_t& duration_controller() noexcept { return duration_controller_; }
    sampling_config const& config() const noexcept { return config_; }
    sampling_callbacks const& callbacks() const noexcept { return callbacks_; }

    // ----- production state setters -----
    // Called by postfork_child() to put the service into child-process mode, where
    // shutdown() skips per-tid processing (AC-20).
    void enter_child_process_mode() noexcept { child_process_mode_ = true; }

    // Access per-thread state registry.
    [[nodiscard]] thread_sampler_state_registry<Policies>& registry() noexcept
    {
        return registry_;
    }

private:
    sampling_config      config_;
    sampling_callbacks   callbacks_;
    thread_info_resolver thread_info_resolver_;
    unwinder             unwinder_;
    offload              offload_;
    trace_sink           trace_sink_;
    clock                clock_;
    signal_dispatcher    signal_dispatcher_;
    perfetto_sink        perfetto_sink_;
    fatal_error          fatal_;

    pause_registry_t                        pause_registry_;
    thread_sampler_state_registry<Policies> registry_;
    duration_controller_t                   duration_controller_;

    std::atomic<bool> blocked_{ false };
    bool              duration_disabled_  = false;
    bool              child_process_mode_ = false;

    // Per-thread signal set registry (tid → signal set).
    // In production populated lazily from rocprofsys::get_sampling_signals(tid).
    mutable std::mutex                         signal_types_mutex_;
    std::unordered_map<int64_t, std::set<int>> signal_types_;

    // Apply pthread_sigmask via signal_dispatcher_; route errors through fatal_.
    // verb is "Block" / "Unblock" — used in the LOG_DEBUG line.
    void apply_signal_mask(int how, std::set<int> sigs, char const* verb);

    // Lifecycle orchestration. Implemented in src/sampling_service_impl.hpp.
    // All main-library dependencies are injected via sampling_config callbacks
    // and the Policies::thread_info_resolver template parameter.
    bool check_thread_guards(int64_t tid);
    void do_setup_wiring(int64_t tid, thread_state_t* state, std::set<int> const& sigs);

    // Focused helpers decomposed from do_setup_wiring.
    void wire_tls(int64_t tid, thread_state_t* state);
    void arm_timer_triggers(int64_t tid, thread_state_t* state,
                            std::set<int> const& sigs);
    void arm_overflow_trigger(int64_t tid, thread_state_t* state,
                              std::set<int> const& sigs);
    void start_duration_controller(int64_t tid);
    void register_trace_cache_tracks(int64_t tid);
    void do_shutdown_wiring(int64_t tid) noexcept;
    void do_emit_resolved(int64_t tid);
    void do_postfork_parent_reinit();
    void do_postfork_child_cleanup();
};

#if defined(__linux__)
using default_sampling_service = sampling_service<default_sampling_policies>;
#endif

}  // namespace rocprofsys::sampling

// Production accessor — Meyers singleton (DEC-10).
// Returns the same instance for both sampling and causal_sampling.
namespace rocprofsys::services
{
#if defined(__linux__)
rocprofsys::sampling::default_sampling_service&
sampling();
rocprofsys::sampling::default_sampling_service&
causal_sampling();
#endif
}  // namespace rocprofsys::services

// Template method bodies live in src/sampling_service_impl.hpp. That header
// pulls platform types (sigset_t, linux/perf_event.h, libunwind.h) plus
// main-library headers (core/config.hpp, core/perf.hpp, etc.), so it is NOT
// included here to keep the public include/sampling/ surface platform-neutral
// (NFR-PORT-1). Instead, the bottom-include is in the two policy aggregators:
//   • library/sampling/default_policies.hpp   (production)
//   • library/sampling/tests/doubles/test_sampling_policies.hpp (test)
