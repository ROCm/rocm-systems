// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/backtrace_record.hpp"
#include "sampling/platform_guard.hpp"
#include "sampling/platform_traits.hpp"
#include "sampling/sampling_policies.hpp"
#include "sampling/src/pause_interval_registry.hpp"
#include "sampling/src/sampling_duration_controller.hpp"
#include "sampling/src/thread_sampler_state.hpp"

#include <atomic>
#include <cstdint>
#include <fstream>
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
template <class Policies>
class sampling_service
{
public:
    using unwinder          = typename Policies::unwinder;
    using offload           = typename Policies::offload;
    using trace_sink        = typename Policies::trace_sink;
    using timer_trigger     = typename Policies::timer_trigger;
    using overflow_trigger  = typename Policies::overflow_trigger;
    using clock             = typename Policies::clock;
    using signal_dispatcher = typename Policies::signal_dispatcher;
    using report_writer     = typename Policies::report_writer;
    using perfetto_sink     = typename Policies::perfetto_sink;
    using fatal_error       = typename Policies::fatal_error;

    sampling_service();
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

    void post_process();

    void postfork_parent_reinit();
    void postfork_child_cleanup();

    std::set<int> get_signal_types(int64_t tid);

    // ----- introspection (test/diagnostics) -----
    [[nodiscard]] size_t dropped_samples() const noexcept;
    [[nodiscard]] bool   is_paused() const noexcept;
    [[nodiscard]] bool   is_blocked() const noexcept;

    // ----- test injection seams (policy accessors) -----
    unwinder&          get_unwinder() noexcept { return unwinder_; }
    offload&           get_offload() noexcept { return offload_; }
    trace_sink&        get_trace_sink() noexcept { return trace_sink_; }
    signal_dispatcher& signal_dispatcher_ref() noexcept { return signal_dispatcher_; }
    report_writer&     report_writer_ref() noexcept { return report_writer_; }
    perfetto_sink&     get_perfetto_sink() noexcept { return perfetto_sink_; }
    clock&             get_clock() noexcept { return clock_; }
    fatal_error&       get_fatal_error() noexcept { return fatal_; }

    // ----- production state setters -----
    // Called by postfork_child() to put the service into child-process mode, where
    // shutdown() releases state without calling post_process (AC-20).
    void enter_child_process_mode() noexcept { child_process_test_ = true; }

    // ----- test seams for state injection -----
    void set_duration_disabled_for_test(bool v) noexcept { duration_disabled_ = v; }
    void set_causal_mode_for_test(bool v) noexcept { causal_mode_test_ = v; }
    void set_child_process_for_test(bool v) noexcept { child_process_test_ = v; }

    // Inject a backtrace_record into the ring buffer for tid (test-only seam, I-7 fix).
    // Creates the per-thread state entry if it does not exist yet.
    void inject_record_for_test(int64_t tid, backtrace_record const& rec)
    {
        registry_.emplace(tid);
        if(auto* state = registry_.at(tid)) state->ring_buffer().try_push(rec);
    }

    // Access per-thread state registry (test introspection).
    [[nodiscard]] thread_sampler_state_registry<Policies>& registry() noexcept
    {
        return registry_;
    }

private:
    unwinder          unwinder_;
    offload           offload_;
    trace_sink        trace_sink_;
    clock             clock_;
    signal_dispatcher signal_dispatcher_;
    report_writer     report_writer_;
    perfetto_sink     perfetto_sink_;
    fatal_error       fatal_;

    pause_interval_registry<clock>          pause_registry_;
    thread_sampler_state_registry<Policies> registry_;
    sampling_duration_controller<clock>     duration_controller_;

    // TSV output file handles — opened by open_report_writer_streams() in production.
    // Stored here (not in native_report_writer) so they outlive flush().
    std::ofstream m_tsv_wall;
    std::ofstream m_tsv_cpu;
    std::ofstream m_tsv_pct;
    std::ofstream m_tsv_trip;

    std::atomic<bool> blocked_{ false };
    bool              duration_disabled_  = false;
    bool              causal_mode_test_   = false;
    bool              child_process_test_ = false;

    // Per-thread signal set registry (tid → signal set).
    // In production populated lazily from rocprofsys::get_sampling_signals(tid).
    mutable std::mutex                         signal_types_mutex_;
    std::unordered_map<int64_t, std::set<int>> signal_types_;

    // Production wiring hooks — no-op in the generic template.
    // Explicit full specializations for default_sampling_policies are provided in
    // library/sampling_production_policies/sampling_service_production_hooks.hpp
    // and do the TLS wiring, timer arming, and thread-info early-return guards.
    bool setup_check_thread_guards(int64_t tid);
    void setup_production_wiring(int64_t tid, thread_sampler_state<Policies>* state,
                                 std::set<int> const& sigs);
    void shutdown_production_wiring(int64_t tid);

    // Called at the start of post_process() before flush().
    // Production specialization opens the TSV output files and calls
    // report_writer_.set_streams(...) so flush() has real file handles.
    void open_report_writer_streams();
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

// Template method bodies — include after class definition.
// Lives in src/ so it can use platform types (sigset_t, etc.)
// without polluting the public include/sampling/ headers (NFR-PORT-1).
#include "sampling/src/sampling_service_impl.hpp"
