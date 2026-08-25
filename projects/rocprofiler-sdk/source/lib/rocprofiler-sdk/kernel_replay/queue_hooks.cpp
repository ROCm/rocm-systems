// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/kernel_replay/queue_hooks.hpp"

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/local_context.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/replay_callbacks.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/replay_diagnostics.hpp"

#include <fmt/format.h>
#include <hsa/hsa.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string_view>
#include <thread>
#include <utility>

namespace rocprofiler
{
namespace kernel_replay
{
namespace
{
constexpr auto null_hsa_signal = hsa_signal_t{.handle = 0};

// Wait for `try_drain_once` to report success, in ~5s slices up to ~60s.
//
// A drain that does not converge is not necessarily an application bug. The HSA full profile
// requires an agent to make forward progress on multiple queues concurrently, which is exactly what
// licenses an application to keep two co-dependent kernels resident on the same agent -- and a
// persistent kernel, a spin-waiting cooperative kernel, or a collective whose peer has not reached
// the same point will all sit here indefinitely. So the timeout returns false and the caller
// declines the replay, rather than terminating the process: for a multi-hour job under a scheduler,
// killing the job is a far worse outcome than not profiling one dispatch.
//
// @return true if it drained, false on timeout.
template <typename TryDrainFn>
[[nodiscard]] bool
replay_wait_or_decline(TryDrainFn&& try_drain_once, std::string_view what)
{
    static constexpr int drain_slices = 5;
    static constexpr int max_slices   = 12;

    for(int i = 0; i < max_slices; ++i)
    {
        if(try_drain_once()) return true;
        ROCP_WARNING << fmt::format(
            "kernel replay: still waiting for {} (~{}s elapsed)", what, (i + 1) * drain_slices);
    }
    ROCP_ERROR << fmt::format("kernel replay: {} did not drain after ~{}s; declining replay for "
                              "this dispatch. A persistent, cooperative, or peer-dependent kernel "
                              "cannot be replayed -- exclude it with kernel filtering",
                              what,
                              max_slices * drain_slices);
    return false;
}

// Drain a queue's in-flight async completion handler(s) during replay. Unlike Queue::sync()'s
// teardown use (warn once and proceed), a replay pass must NOT proceed while a handler is still
// running: PASS-EXIT, the tool's continue-decision, restore(), and the next submit would race the
// handler that is still emitting records, releasing signals, and dropping correlation-id refs.
// Each sync() call blocks up to one ~5s slice and reports whether the queue drained.
[[nodiscard]] bool
replay_drain_or_decline(const hsa::Queue& queue)
{
    return replay_wait_or_decline([&]() { return queue.sync(); },
                                  "this queue's async completion handler(s)");
}

// Drain every queue on `agent` before snapshotting, WITHOUT holding the queue-map lock across the
// wait. iterate_queues holds the queue-map read lock for the duration of its callback, so a
// per-sibling blocking drain there would block stream creation/destruction for the whole (up to
// ~60s) drain. Instead poll each queue's in-flight async count under a brief read lock and sleep
// between polls, so the map lock is held only for the microsecond poll -- never across the wait.
// Safe against concurrent queue destruction: a Queue is only dereferenced while the read lock is
// held (destroy_queue erases under the write lock), and the live set is re-read every poll. The
// per-agent writer lock held by the replay window blocks new dispatches on the agent, so in-flight
// work only decreases and the poll converges; on a genuinely stuck queue it declines rather than
// aborts, matching replay_drain_or_decline.
//
// @return true if every queue on the agent went idle, false on timeout.
[[nodiscard]] bool
replay_drain_agent_or_decline(hsa_agent_t agent)
{
    auto* queue_controller = hsa::get_queue_controller();
    if(queue_controller == nullptr) return true;

    constexpr auto poll_interval = std::chrono::milliseconds{2};
    constexpr auto max_wait      = std::chrono::seconds{60};
    const auto     deadline      = std::chrono::steady_clock::now() + max_wait;

    for(;;)
    {
        int64_t in_flight = 0;
        queue_controller->iterate_queues([&](const hsa::Queue* sibling) {
            if(sibling != nullptr && sibling->get_agent().get_hsa_agent().handle == agent.handle)
                in_flight += sibling->active_async_packets();
        });

        if(in_flight == 0) return true;

        if(std::chrono::steady_clock::now() >= deadline)
        {
            ROCP_ERROR << fmt::format(
                "kernel replay: agent-wide drain stuck ({} async handler(s) still active after "
                "~60s); declining replay for this dispatch",
                in_flight);
            return false;
        }

        std::this_thread::sleep_for(poll_interval);
    }
}

// RAII marker for "this thread is inside a replay window on this agent", read by both dispatch
// paths to avoid a non-recursive deadlock. State and predicates live in replay_diagnostics.hpp.
struct replay_window_scope
{
    explicit replay_window_scope(rocprofiler_agent_id_t agent)
    : m_agent{agent}
    {
        enter_replay_window(m_agent);
    }
    ~replay_window_scope() { exit_replay_window(m_agent); }

    replay_window_scope(const replay_window_scope&)     = delete;
    replay_window_scope(replay_window_scope&&) noexcept = delete;
    replay_window_scope& operator=(const replay_window_scope&) = delete;
    replay_window_scope& operator=(replay_window_scope&&) noexcept = delete;

private:
    rocprofiler_agent_id_t m_agent;
};
}  // namespace

bool
run_replay_window(const replay_dispatch_t& dispatch)
{
    const auto& queue = *dispatch.queue;
    // Copied, not referenced: the caller's pointer aims into the application's AQL packet buffer,
    // and the window outlives the point at which the interceptor is done reading it.
    const auto packet           = *dispatch.packet;
    const auto thr_id           = dispatch.thread_id;
    const auto internal_corr_id = dispatch.internal_correlation_id;
    const auto ancestor_corr_id = dispatch.ancestor_correlation_id;

    const auto replay_plan = execute_config_phase_enter(
        queue, packet, thr_id, internal_corr_id, ancestor_corr_id, dispatch.dispatch_id);

    // The tool declined replay for this dispatch. execute_config_phase_enter() has already closed
    // the CONFIG sequence, so there is nothing to unwind: the caller submits it normally.
    if(!replay_plan.replay_requested) return false;

    // Runs synchronously on the calling (interceptor) thread. Concurrent work is isolated by (a)
    // the per-agent WRITER lock taken below, which serializes this whole
    // drain->snap->passes->restore window against other replays *and* against non-replay dispatches
    // on this agent (they hold the reader lock across their submit, see dispatch_lock_for), and (b)
    // agent-scoped snapshots so a replay only saves/restores its own agent's device memory (other
    // GPUs untouched). Different agents hold different locks and run concurrently.
    const auto& core            = queue.core_api();
    hsa_agent_t replay_agent    = queue.get_agent().get_hsa_agent();
    const auto  replay_agent_id = queue.get_agent().get_rocp_agent()->id;

    const auto& policy       = resolve_policy();
    auto        outcome      = replay_outcome_t{};
    outcome.dispatch_id      = dispatch.dispatch_id;
    outcome.requested_passes = replay_plan.indefinite ? uint64_t{0} : replay_plan.total_passes;

    // The app's original completion signal (completion_signal is at the same offset for dispatch
    // and ext-dispatch packets, per the static_asserts in hsa/queue.cpp). It is suppressed on every
    // replay pass and fired once after the loop so the application observes a single completion
    // regardless of pass count / early exit / indefinite loop.
    hsa_signal_t app_completion_signal = packet.kernel_dispatch.completion_signal;

    // Our private drain barrier signal. Declared here so the decline path below can decide whether
    // it is safe to destroy: it is only safe once the barrier packet referencing it has actually
    // completed.
    hsa_signal_t drain_signal = null_hsa_signal;

    // Run the dispatch exactly once, with its original completion signal, and report why. Every
    // decline funnels through here so the CONFIG sequence is always closed and the application
    // always observes exactly one execution.
    //
    // `destroy_drain_signal` must be false whenever the drain barrier may still be pending: the
    // packet processor would decrement a destroyed signal. Leaking one signal on a path that is
    // already reporting a stuck queue is the lesser fault.
    auto decline_and_run_once = [&](decline_reason reason, bool destroy_drain_signal) {
        outcome.reason              = reason;
        outcome.reentrancy_observed = replay_reentrancy_observed(replay_agent_id);
        log_replay_outcome(outcome);
        execute_config_phase_exit(replay_plan, thr_id, internal_corr_id, ancestor_corr_id);
        if(destroy_drain_signal && drain_signal.handle != 0)
            hsa::get_core_table()->hsa_signal_destroy_fn(drain_signal);
        dispatch.submit_dispatch(/*is_replay_pass=*/false);
    };

    // A replay window cannot be opened inside a replay window on the same agent. Getting here means
    // a tool's KERNEL_REPLAY callback launched a kernel that itself matched the replay filter, on
    // the thread that already holds this agent's writer lock. The non-replay dispatch path can skip
    // its shared lock and let such a dispatch through; the writer side cannot -- std::unique_lock
    // on a shared_mutex this thread already owns exclusively never returns, and there is no timeout
    // to escape it. Decline and run once instead, and mark the enclosing window's outcome, whose
    // counters are no longer trustworthy.
    if(in_replay_window(replay_agent_id))
    {
        note_replay_reentrancy(replay_agent_id);
        ROCP_ERROR_IF(should_warn_replay_reentrancy())
            << "kernel replay: a dispatch that itself requested replay was launched from inside a "
               "replay window on the same agent, by a KERNEL_REPLAY callback (CONFIG, PASS, "
               "pass_count_cb, or replay_continue_cb). It is run once without replay to avoid "
               "deadlocking on the per-agent replay lock, and the enclosing dispatch's counters "
               "are "
               "not trustworthy because this kernel mutated device memory inside the snapshot "
               "window. Move GPU work out of the replay callbacks";
        decline_and_run_once(decline_reason::reentrant_dispatch, /*destroy_drain_signal=*/false);
        return true;
    }

    const auto replay_guard =
        std::unique_lock<std::shared_mutex>{agent_replay_mutex(replay_agent_id)};
    // Must be constructed after the writer lock: it marks the interval in which a nested dispatch
    // on this thread would deadlock on this agent's replay lock.
    const auto window_scope = replay_window_scope{replay_agent_id};

    // Admission control, before any device->host traffic. Two independent questions: whether the
    // snapshot would actually cover the application's data, and whether it would fit and finish in
    // a sane amount of time. Both are answerable from the tracker alone, so answering them here
    // costs nothing and turns a silently-wrong result (or an apparently hung job) into a diagnostic
    // naming the cause.
    outcome.untracked = memory_tracker::untracked_device_memory(replay_agent);
    if(const auto reason = check_untracked(outcome.untracked, policy);
       reason != decline_reason::none)
    {
        decline_and_run_once(reason, /*destroy_drain_signal=*/false);
        return true;
    }

    const auto footprint      = memory_snapshot::estimate_footprint(replay_agent);
    outcome.footprint_bytes   = footprint.bytes;
    outcome.footprint_regions = footprint.regions;
    if(const auto reason = check_admission(footprint, outcome.requested_passes, policy);
       reason != decline_reason::none)
    {
        decline_and_run_once(reason, /*destroy_drain_signal=*/false);
        return true;
    }

    // Drain barrier: fence the CPU against all prior in-flight GPU work on this queue so device
    // memory is stable before snapshotting.
    hsa::Queue::create_signal(0, &drain_signal, /*use_pool=*/false);
    {
        using namespace std::chrono_literals;

        dispatch.submit_barrier(drain_signal);

        if(!replay_wait_or_decline(
               [&]() {
                   return core.hsa_signal_wait_scacquire_fn(drain_signal,
                                                            HSA_SIGNAL_CONDITION_EQ,
                                                            0,
                                                            std::chrono::nanoseconds{5s}.count(),
                                                            HSA_WAIT_STATE_BLOCKED) == 0;
               },
               "this queue's prior GPU work"))
        {
            // The barrier packet is still queued and holds a reference to drain_signal, so it must
            // outlive us; deliberately leak it rather than let the packet processor decrement freed
            // memory.
            decline_and_run_once(decline_reason::queue_drain_stuck,
                                 /*destroy_drain_signal=*/false);
            return true;
        }
    }

    // Agent-wide drain. Sibling queues on this agent can have kernels in flight that mutate device
    // memory while we snapshot and restore. The per-agent replay lock blocks other threads'
    // replayed dispatches, since every kernel dispatch passes through that gate. It does not block
    // work already submitted to their queues. So wait for every queue on this agent to finish its
    // outstanding kernels before snapshotting. We wait outside the queue-map lock so it does not
    // block stream creation or destruction. See replay_drain_agent_or_decline. Async SDMA copies
    // bypass both the AQL queues and the replay gate and serializing those is a separate follow-up
    // (TODO: mkuriche, amd-vkale).
    if(!replay_drain_agent_or_decline(replay_agent))
    {
        // Our own barrier completed above, so the signal is safe to destroy here.
        decline_and_run_once(decline_reason::agent_drain_stuck, /*destroy_drain_signal=*/true);
        return true;
    }

    // Save this agent's tracked device allocations so every pass runs against identical inputs.
    // snap() returns ok=false if it could not capture the complete set (host memory pressure or a
    // failed copy)
    const auto snap_started = std::chrono::steady_clock::now();
    const auto snapshot     = memory_snapshot::snap(replay_agent);
    outcome.snap_seconds =
        std::chrono::duration<double>{std::chrono::steady_clock::now() - snap_started}.count();

    // Snapshot incomplete (memory pressure or a failed capture copy): restoring a partial snapshot
    // between passes would corrupt application data, so decline replay. Close the CONFIG sequence,
    // free our drain signal, and run this dispatch once still under the writer lock with its
    // original completion signal.
    if(!snapshot.ok)
    {
        ROCP_WARNING << "kernel replay: snapshot capture failed (memory pressure or copy error); "
                        "running this dispatch once without replay";
        decline_and_run_once(decline_reason::snapshot_failed, /*destroy_drain_signal=*/true);
        return true;
    }

    // Localized context control for this replay loop. This guard installs the thread-local routing
    // that connects the tool's PASS toggle callbacks (writers, via replay_local_start/stop_context)
    // to the services that read it at dispatch (via local_context_override). It lives for the whole
    // loop and is torn down when the guard exits; global context state is never touched. It
    // captures the contexts active now (loop start) as the toggle mask, so a tool may only
    // enable/disable one of those and a local start cannot promote a globally-stopped context
    // (local_context.hpp).
    auto local_ctx_tls_guard = scoped_local_context_control{context::get_active_contexts()};

    // Per-pass loop: PASS enter -> submit -> drain the async handler -> PASS exit -> ask the tool
    // whether to continue -> restore device memory before the next pass.
    for(uint64_t pass = 0;; ++pass)
    {
        const bool is_final = !replay_plan.indefinite && (pass == replay_plan.total_passes - 1);

        auto pass_state = pass_context_state_t{};
        execute_pass_phase_enter(
            replay_plan, pass, thr_id, internal_corr_id, ancestor_corr_id, pass_state);

        dispatch.submit_dispatch(/*is_replay_pass=*/true);

        // Drain this pass's async handler (separate HSA thread: reads counters, emits records,
        // releases signals/corr-id refs) before PASS EXIT / continue-decision / restore() / next
        // submit, else we race its record delivery and reuse buffers and signals it still holds.
        // This also implies GPU drain. Exactly one handler is in flight per pass (we drain before
        // each submit, under the agent writer lock).
        ROCP_ERROR_IF(queue.active_async_packets() > 1)
            << "kernel replay: more than one async handler in flight during a replay pass";

        if(!replay_drain_or_decline(queue))
        {
            // The pass we just submitted has not completed. We cannot restore over memory a running
            // kernel may still be writing, and we cannot trust this pass's record, so stop the loop
            // here. The trailing completion barrier below still queues behind the stuck work, which
            // is what lets the application make progress if it ever finishes.
            outcome.reason = decline_reason::pass_drain_stuck;
            ++outcome.executed_passes;
            break;
        }

        execute_pass_phase_exit(replay_plan, pass, pass_state);
        ++outcome.executed_passes;

        // Stop once the tool (or the fixed pass count) says we're done; the last executed pass
        // leaves device memory as the app expects, so no restore follows the break.
        if(!should_continue_replay(replay_plan, pass, is_final)) break;

        // Restore device memory between passes so the next pass sees identical inputs. A failed
        // host->device copy leaves the snapshot only partially applied; continuing would submit the
        // next pass over corrupted memory and (because the final pass skips restore) would also
        // leave that corruption visible to the application.
        //
        // This one stays fatal, unlike the drains above. There is no correct continuation from a
        // partial restore: some regions hold pre-kernel bytes and others hold post-kernel bytes,
        // the mix is unknown, and nothing can put it back. Handing that state to the application
        // would corrupt its results silently, which is strictly worse than terminating. Reaching it
        // requires a host->device DMA on a live allocation to fail, which is a device-level fault
        // rather than a policy decision.
        const auto restore_started = std::chrono::steady_clock::now();
        const auto restore_ok      = memory_snapshot::restore(snapshot);
        outcome.restore_total_seconds +=
            std::chrono::duration<double>{std::chrono::steady_clock::now() - restore_started}
                .count();

        if(!restore_ok)
        {
            outcome.reason = decline_reason::snapshot_failed;
            log_replay_outcome(outcome);
            ROCP_FATAL << "kernel replay: restore failed between passes (partial host->device "
                          "copy); aborting rather than continuing with corrupted device memory";
        }
    }

    outcome.reentrancy_observed = replay_reentrancy_observed(replay_agent_id);
    log_replay_outcome(outcome);

    execute_config_phase_exit(replay_plan, thr_id, internal_corr_id, ancestor_corr_id);

    // Fire the app's original completion signal once, now that the final executed pass has
    // completed (we already drained the final pass's handler above). A trailing barrier decrements
    // it exactly as the single-pass path would, so the application unblocks and the next kernel on
    // this GPU can dispatch. Deferred out of the per-pass path so early-exit and indefinite loops
    // signal on the actual last pass rather than at pass N-1.
    if(app_completion_signal.handle != 0) dispatch.submit_barrier(app_completion_signal);

    // Clean up our private signals (never the app's completion signal).
    if(drain_signal.handle != 0) hsa::get_core_table()->hsa_signal_destroy_fn(drain_signal);
    return true;
}

}  // namespace kernel_replay
}  // namespace rocprofiler
