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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/kfd/signal_less.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/kfd/env_parse.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_correlation.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_reader.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <fmt/core.h>

#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace kfd
{
// Process-lifetime singleton. The signal-less feature attaches once per process
// (init_kfd_profiler runs from the single interposition_init) and tears down at
// finalize; it is never re-attached. The hub's m_* fields are therefore not reset
// between attaches -- there is no second attach -- and after teardown the
// process-wide disable latch keeps the hub inert. If the SDK ever grows a
// re-attach path, this singleton and the OwnerRegistry/disable-latch it partners
// with would need an explicit reset; today that is deliberately unsupported.
signal_less_hub_t&
signal_less_hub()
{
    static auto*& _v = common::static_object<signal_less_hub_t>::construct();
    return *_v;
}

bool
signal_less_feature_enabled()
{
    // Read once: the answer must not change under a running process, and the
    // enqueue path cannot afford an env lookup per batch.
    static const bool _enabled = []() {
        // Strict opt-in, fail-closed: only an explicit true value activates a
        // feature that changes completion semantics process-wide.
        if(!env_bool_opt_in("ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS")) return false;
        // register the fork-child abandon handler HERE, where the feature
        // is decided -- every path that creates signal-less state passes this gate
        // first, so "state exists => a handler is registered" is a property of the
        // gate, not of the reader lifecycle (which may never start). If
        // pthread_atfork fails, the feature stays OFF: no state can then exist
        // without a handler to abandon it in a child.
        if(int _rc = pthread_atfork(nullptr, nullptr, signal_less_abandon_in_child); _rc != 0)
        {
            ROCP_WARNING << "KFD dispatch-log: pthread_atfork failed (" << _rc
                         << "), signal-less kernel-dispatch completion stays DISABLED";
            return false;
        }
        ROCP_WARNING << "KFD dispatch-log: signal-less kernel-dispatch completion is ACTIVE "
                        "(ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS). Eligible batches publish "
                        "their packets untouched and complete from firmware records instead of "
                        "completion signals; ineligible batches keep the signal path. Unset the "
                        "variable to return to signal-based completion.";
        return true;
    }();
    return _enabled;
}

namespace
{
// Free-standing counters (constant-initialized, so no lazy-init behind them).
std::atomic<uint64_t> g_counters[static_cast<size_t>(signal_less_counter::kCount)] = {};

// Constant-initialized (no guard variable, no dynamic initialization), so the
// atfork child handler can store to it with no lazy-init machinery behind it.
std::atomic<bool> g_child_stale{false};

// Completions the task group would not take. Only reachable once finalization
// has begun, so this is drained once, by the teardown thread, after the reader
// and processor are joined -- which is what keeps a client callback off them.
struct deferred_completions
{
    std::mutex                             mu   = {};
    std::vector<signal_less_hub_t::proven> held = {};
};

deferred_completions&
deferred()
{
    static auto*& _v = common::static_object<deferred_completions>::construct();
    return *_v;
}

// Guards the loss-ledger lookup so the correlation-id finalize path costs one
// atomic load, and never constructs the hub, until something is actually leaked.
std::atomic<bool>&
any_leaked()
{
    static auto _v = std::atomic<bool>{false};
    return _v;
}

OwnerRegistry&
registry_storage()
{
    static auto*& _v = common::static_object<OwnerRegistry>::construct();
    return *_v;
}

// submission gate. hand_off_proven and flush_deferred_completions take it
// SHARED over their whole body; the fence's abandon path takes it EXCLUSIVE to
// latch g_abandoned. The exclusive acquire waits out every in-flight submitter, so
// none can submit or append to deferred() after the abandoner flushed and joined
// -- the check-then-append window a bare atomic cannot close (TaskGroup::async
// increments its task count AFTER the check). Constant-initialised, so no lazy
// init hides behind them. Not on the per-record path.
std::shared_mutex g_submit_gate;
std::atomic<bool> g_abandoned{false};

}  // namespace

void
note_signal_less(signal_less_counter which, uint64_t n)
{
    // Nothing is counted unless the feature is actually active, so the default
    // path never touches these atomics.
    if(!signal_less_feature_enabled()) return;
    g_counters[static_cast<size_t>(which)].fetch_add(n, std::memory_order_relaxed);
}

signal_less_counter_array
signal_less_stats()
{
    auto _s = signal_less_counter_array{};
    for(size_t i = 0; i < _s.size(); ++i)
        _s[i] = g_counters[i].load(std::memory_order_relaxed);
    return _s;
}

const char*
signal_less_counter_name(signal_less_counter which)
{
    switch(which)
    {
        case signal_less_counter::entry_registered: return "registered";
        case signal_less_counter::eop_proven: return "eop-proven";
        case signal_less_counter::eop_unmatched: return "eop-unmatched";
        case signal_less_counter::finalizer_emitted: return "emitted";
        case signal_less_counter::finalizer_no_timing: return "no-timing";
        case signal_less_counter::register_refused: return "register-refused";
        case signal_less_counter::cap_evicted: return "cap-evicted";
        case signal_less_counter::kCount: break;
    }
    return "?";
}

bool
signal_less_child_stale()
{
    return g_child_stale.load(std::memory_order_acquire);
}

void
signal_less_abandon_in_child()
{
    // Async-signal-safe by construction: every statement is an atomic scalar store
    // or a load of an already-initialized static pointer. Nothing locks,
    // allocates, logs, joins or frees.
    //
    // static_object<T>::get() returns the pointer WITHOUT constructing it, so an
    // object this process never created stays uncreated. These accessors must
    // live in this TU: static_object's context type is per-TU, so get() from
    // elsewhere would observe a different (null) instantiation.
    g_child_stale.store(true, std::memory_order_release);

    if(auto* _hub = common::static_object<signal_less_hub_t>::get()) _hub->abandon_in_child();
    if(auto* _reg = common::static_object<OwnerRegistry>::get()) _reg->abandon_in_child();
}

void
hand_off_proven(signal_less_hub_t::proven&& p)
{
    if(g_child_stale.load(std::memory_order_acquire)) return;

    // the g_abandoned check, the submit, and the failed-submit deferred
    // append are all inside ONE shared acquisition, so the abandoner's exclusive
    // acquire cannot interleave. One read of g_abandoned: the latch is written only
    // under the exclusive gate, so it cannot transition while this shared region is
    // held.
    auto _gate = std::shared_lock<std::shared_mutex>{g_submit_gate};
    if(g_abandoned.load(std::memory_order_acquire))
    {
        // Abandoned mid-handoff: this record is dropped (no emit). record_kernel_end
        // already erased the entry, so drain_for_teardown cannot ledger it -- do it
        // here, and mark losses so correlation_id_finalize skips it instead of
        // force-retiring it as a dangling id. Lock order: g_submit_gate (shared) then
        // the hub's m_mu, the only place they nest and never the reverse.
        signal_less_hub().ledger_abandoned(p.correlation_id);
        note_signal_less_losses();
        return;
    }

    if(submit_complete_signal_less_dispatch(p)) return;

    // Deliberately NOT finalized here: this is the processor thread, which must
    // never run a client callback.
    auto& _d = deferred();
    auto  lk = std::lock_guard<std::mutex>{_d.mu};
    _d.held.emplace_back(std::move(p));
}

size_t
flush_deferred_completions()
{
    if(g_child_stale.load(std::memory_order_acquire)) return 0;

    // finalize_complete_signal_less_dispatch touches the metadata being
    // invalidated, so the whole body is inside the same shared gate the abandoner
    // waits out before latching.
    auto _gate = std::shared_lock<std::shared_mutex>{g_submit_gate};

    auto _taken = std::vector<signal_less_hub_t::proven>{};
    {
        auto& _d = deferred();
        auto  lk = std::lock_guard<std::mutex>{_d.mu};
        _taken.swap(_d.held);
    }
    // Outside the deferred() lock (still under the shared gate): the finalizer runs
    // client callbacks.
    for(auto& _p : _taken)
        finalize_complete_signal_less_dispatch(std::move(_p));
    return _taken.size();
}

OwnerRegistry&
owner_registry()
{
    return registry_storage();
}

void
poison_slot(uint32_t gpu_id, uint32_t doorbell_slot)
{
    if(g_child_stale.load(std::memory_order_acquire)) return;
    // Leak + permanently quarantine in one hub critical section. The returned
    // payloads are released here, off the hub lock; releasing one runs no client
    // code, so this is safe on the destroying / creating thread.
    auto _stranded = signal_less_hub().quarantine_slot(gpu_id, doorbell_slot);
    if(_stranded.empty()) return;
    note_signal_less_losses();
    ROCP_WARNING << fmt::format(
        "KFD dispatch-log: doorbell slot {} poisoned (structural ambiguity or truncated close); {} "
        "in-flight signal-less dispatch(es) emit no record and are signal-path-only for the rest "
        "of the process.",
        doorbell_slot,
        _stranded.size());
}

void
add_live_queue(uint64_t queue_token, uint32_t gpu_id, std::optional<uint32_t> doorbell_slot)
{
    if(g_child_stale.load(std::memory_order_acquire)) return;
    auto _result = owner_registry().add_queue(queue_token, gpu_id, doorbell_slot);
    if(_result != OwnerRegistry::add_result::collision) return;

    // A second live owner means a firmware record on this slot can no longer be
    // attributed to one queue: the registry's collision verdict converges on the
    // same poison as the window-overlap verdict.
    poison_slot(gpu_id, *doorbell_slot);
}

// Per-close ceiling on the hardware drain and, reused, the closed-window GC grace.
// The aggregate 2 s pool is deleted (TEARDOWN-LATENCY); this is the only
// close-path knob.
uint64_t
close_drain_budget_ns()
{
    static const uint64_t _v = []() {
        auto _ms = common::get_env("ROCPROFILER_KFD_DISPATCH_LOG_CLOSE_DRAIN_MS", 250);
        _ms      = std::max(_ms, 0);
        return static_cast<uint64_t>(_ms) * 1'000'000ull;
    }();
    return _v;
}

// N1: reader-quiesce budget per attempt. 0 == "one non-blocking check then treat
// as timed out". Read once (F3 pattern); reject-and-default via the shared parse
// contract, never clamp. The [0, 60000] upper bound keeps *1e6 far from overflow.
uint64_t
quiesce_budget_ns()
{
    static const uint64_t _v = []() {
        constexpr long _dflt_ms = 100;
        const long     _ms = env_long_in_range("ROCPROFILER_KFD_DISPATCH_LOG_QUIESCE_MS", 0, 60'000)
                             .value_or(_dflt_ms);
        ROCP_INFO << "KFD dispatch-log: reader-quiesce budget = " << _ms << " ms";
        return static_cast<uint64_t>(_ms) * 1'000'000ull;
    }();
    return _v;
}

void
signal_less_prime_env()
{
    // Every knob below caches its env read in a function-local static on FIRST USE,
    // and common::get_env* scans `environ`, which is not safe against a concurrent
    // setenv. Left lazy, that first use lands on whichever worker, producer or
    // teardown thread happens to reach it first. Calling them here -- from the one
    // serialized reader-start path, before any worker thread exists and before the
    // application is dispatching -- makes the cached value deterministic and takes
    // the environ scan off every concurrent thread.
    signal_less_feature_enabled();
    hub_max_pending_per_gpu();
    close_drain_budget_ns();
    quiesce_budget_ns();
}

void
signal_less_disable_permanently()
{
    if(g_child_stale.load(std::memory_order_acquire)) return;
    // Part 1: sticky process-wide latch, so resolve()/open_window() refuse
    // everywhere and register_batch() rejects under m_mu.
    signal_less_disable_latch().store(true, std::memory_order_release);
    // Part 2: drain the hub while it is still LIVE -- drain_for_teardown() sets
    // m_mode=stopping and leaks+ledgers every existing entry in one m_mu section.
    // Deliberately NOT m_abandoned (the fork-child latch, which would no-op the
    // drain and force-retire live ids). Part 3 (the register_batch latch check)
    // closes the register-vs-drain race by the mutex chain.
    auto _loss = signal_less_hub().drain_for_teardown();
    if(_loss.second.dispatches == 0) return;
    note_signal_less_losses();
    ROCP_WARNING << fmt::format(
        "KFD dispatch-log: signal-less disabled process-wide (a doorbell owner this process never "
        "windowed exists); {} in-flight dispatch(es) across {} correlation id(s) drained and "
        "ledgered.",
        _loss.second.dispatches,
        _loss.second.correlation_ids);
}

void
remove_live_queue(uint64_t queue_token)
{
    if(g_child_stale.load(std::memory_order_acquire)) return;
    owner_registry().remove_queue(queue_token);
}

void
note_signal_less_losses()
{
    any_leaked().store(true, std::memory_order_release);
}

bool
signal_less_id_is_leaked(uint64_t correlation_id)
{
    if(g_child_stale.load(std::memory_order_acquire)) return false;
    if(!any_leaked().load(std::memory_order_acquire)) return false;
    return signal_less_hub().is_ledgered(correlation_id);
}

namespace
{
// The one TERMINAL transition, fail-closed: latch no-new-eligibility, stop and
// JOIN the reader+processor so no further completion can ever be produced, then
// ledger everything still PENDING. Returns the loss stats.
//
// IDEMPOTENT BY CONSTRUCTION, deliberately with no run-once latch: every step
// already no-ops when it has nothing to do (stop_kfd_reader() returns on
// !running, drain_for_teardown() on an empty hub, the latch store is a repeat).
// A latch would be actively worse twice over -- a second caller would return
// BEFORE the first one's stop completed, which is exactly the guarantee a
// terminal fence owes; and a reader legitimately restarted after an earlier
// fence would then never be stopped at all.
//
// NOT DEADLOCK-PRONE BY CONSTRUCTION, two ways:
//  - it never joins the task group. A signal-less completion runs as a TaskGroup
//    task and may synchronously call a client callback that finalizes, reaching
//    here on a task-group worker; TaskGroup::join spins on a task count that very
//    task still holds, so joining there would wait for itself forever. Callers
//    that must also wait out already-submitted completions do that themselves,
//    after this returns.
//  - it holds NO lock across the join. The processor takes g_submit_gate SHARED
//    on its handoff path, so latching that gate exclusively around
//    stop_kfd_reader() would deadlock against the very thread being joined. The
//    threads joined here (reader, processor) never run client code and never
//    submit-and-wait, so joining them from any other thread terminates.
loss_stats
signal_less_terminal_stop()
{
    // Latched BEFORE the stop so a batch that already passed eligibility cannot
    // register into the hub after the drain below empties it: register_batch tests
    // this latch under m_mu, which orders it against drain_for_teardown().
    signal_less_disable_latch().store(true, std::memory_order_release);
    // The sole producer of PENDING->EOP_PROVEN. Once joined, no record_kernel_end
    // and therefore no hand_off_proven can run again.
    stop_kfd_reader();
    auto _loss = signal_less_hub().drain_for_teardown();
    if(_loss.second.dispatches > 0) note_signal_less_losses();
    return _loss.second;
}
}  // namespace

void
signal_less_teardown()
{
    // A forked child abandoned everything and owns none of it: running the
    // teardown there would join threads that do not exist.
    if(g_child_stale.load(std::memory_order_acquire)) return;

    // With the feature off there is no hub work, no retry-owner work and no
    // reader->task handoff, so the ordering constraint does not apply and the
    // existing finalize path is left byte-for-byte as it was.
    if(!signal_less_feature_enabled()) return;

    // Strict order. Each step is what makes the next final:
    //   1. stopping  -> eligibility fails, so no new PENDING is reserved
    //   2. quiesce   -> fences in-flight registration/publication
    //   3. join      -> only the reader creates PENDING->EOP_PROVEN, so after this
    //                   nothing can be added to the retry owner
    //   4. leak      -> whatever never got an EOP is ledgered, so finalize skips it
    //   5. flush     -> therefore final; leftovers finalize in place on THIS thread.
    //                   Order against step 4 is free: a deferred completion left the
    //                   hub at record_kernel_end, so the two touch disjoint sets
    //   6. join tasks-> safe only now: no producer can submit another task
    signal_less_hub().set_mode(session_mode::stopping);
    drain_signal_less_interceptor();
    // F1: HW-drain every live queue against one shared budget BEFORE the reader
    // quiesce, so a kernel in flight when main() returned finishes and its EOP is
    // still copyable when the quiesce below waits for it. Residual: a kernel longer
    // than the budget still strands, loudly, via the existing drain_for_teardown.
    drain_signal_less_queues_hw(kfd::steady_now_ns() + close_drain_budget_ns());
    // Safe only here: steps 1-2 closed the set of records the reader still has to
    // drain, so this two-stage wait (drain_epoch then pipe.empty) covers a finite
    // amount of work rather than a moving target. Bounded, and a no-op when the
    // reader never started. On timeout teardown still proceeds: stop_kfd_reader()
    // below is stronger than the fence's abandon, and drain_for_teardown() ledgers
    // whatever is left.
    wait_for_reader_quiesce();
    // Steps 3 and 5 in the primitive shared with the F23/F24 fence; step 4 (the
    // flush) stays here, outside it, so it runs on EVERY call -- the fence may
    // already have stopped an earlier reader, and a reader restarted since then can
    // still have deferred completions this thread must finalize.
    const auto   _loss    = signal_less_terminal_stop();
    const size_t _flushed = flush_deferred_completions();
    const size_t _leaked  = _loss.dispatches;
    if(_leaked > 0)
    {
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: {} signal-less dispatch(es) across {} correlation id(s) were still "
            "in flight at finalization; they emit no record and their correlation ids are not "
            "retired.",
            _loss.dispatches,
            _loss.correlation_ids);
    }

    join_signal_less_tasks();

    // The only signal-less summary: the reader does not print one too.
    const auto _c     = signal_less_stats();
    auto       _chain = std::string{};
    for(size_t i = 0; i < _c.size(); ++i)
        _chain += fmt::format("{}{}={}",
                              i == 0 ? "" : " ",
                              signal_less_counter_name(static_cast<signal_less_counter>(i)),
                              _c[i]);

    ROCP_WARNING << fmt::format(
        "KFD dispatch-log signal-less summary: {}; teardown finalized {} deferred and stranded {}",
        _chain,
        _flushed,
        _leaked);
}

void
signal_less_fence_completions()
{
    if(g_child_stale.load(std::memory_order_acquire)) return;
    if(!signal_less_feature_enabled()) return;
    // Finalization already ran: our atexit finalize()+destroy_static_objects() has
    // nulled the queue-registry static_object, so a second F24 call (from another
    // DSO's __cxa_finalize teardown) must not rlock it -> NULL deref. finalize() sets
    // fini_status at its very end, so this no-ops that re-entry (and the provably
    // redundant F23 call from invoke_client_finalizers, fini_status==-1) while the
    // genuinely-live detach paths (fini_status==0) still fence.
    if(registration::get_fini_status() != 0) return;
    // (a) no registration/publication is mid-flight.
    drain_signal_less_interceptor();
    // (b) the two-stage wait: every record present in the ring at this call has
    // been copied+published (Stage 1) and popped+processed (Stage 2), so every
    // completion whose firmware record had reached the ring is emitted or deferred.
    // N1: at most TWO attempts with the same budget, the second issued immediately
    // to absorb one scheduling miss under multi-tool contention -- not to extend
    // the deadline. reader_unavailable short-circuits to abandon without spending
    // attempt 2: a reader that is gone will not come back within the budget. The
    // counter is call-local, so a later fence starts fresh.
    bool _quiesced = wait_for_reader_quiesce(quiesce_budget_ns());
    if(!_quiesced && !reader_unavailable())
        _quiesced = wait_for_reader_quiesce(quiesce_budget_ns());
    if(!_quiesced)
    {
        // abandon-on-timeout: latch g_abandoned under the EXCLUSIVE gate, which
        // waits out every in-flight submitter, so none can submit or defer after
        // this. A proven already in flight is dropped AND ledgered by hand_off_proven.
        {
            auto _g = std::unique_lock<std::shared_mutex>{g_submit_gate};
            g_abandoned.store(true, std::memory_order_release);
        }
        // Set the process-wide disable latch BEFORE draining, so a batch that passed
        // eligibility cannot register into the emptied hub afterward: register_batch
        // checks this latch under m_mu, so a registration completed before the latch
        // is swept by the drain and one that acquires m_mu after it is refused. This
        // also stops resolve()/open_window(), matching "abandoned process-wide". Not
        // m_mode and not m_abandoned -- see signal_less_disable_permanently().
        signal_less_disable_latch().store(true, std::memory_order_release);
        auto _loss = signal_less_hub().drain_for_teardown();
        if(_loss.second.dispatches > 0) note_signal_less_losses();
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log fence: reader did not quiesce within the budget; signal-less "
            "abandoned process-wide, {} outstanding dispatch(es) across {} correlation id(s) "
            "ledgered.",
            _loss.second.dispatches,
            _loss.second.correlation_ids);
    }
    // (c) BOTH fence sites are TERMINAL, so quiescing is not enough: F23 frees the
    // client's callback/buffer machinery and F24 lets the HSA runtime unload, both
    // immediately after we return. A dispatch still PENDING here would otherwise
    // complete afterwards and deref a freed context* out of the copied
    // tracing_data, or convert ticks against a dead runtime. Make the transition
    // terminal instead: stop+join the reader/processor so no completion can be
    // produced at all, and ledger whatever never got an EOP. Fail-closed --
    // signal-less stays off for the rest of the process, the accepted cost of a
    // detach or an HSA shutdown. Idempotent, so a second fence call is harmless.
    const auto _loss = signal_less_terminal_stop();
    ROCP_WARNING_IF(_loss.dispatches > 0) << fmt::format(
        "KFD dispatch-log fence: signal-less stopped terminally (client detach or HSA shutdown); "
        "{} dispatch(es) across {} correlation id(s) were still in flight and are ledgered.",
        _loss.dispatches,
        _loss.correlation_ids);
    // (d) nothing is left deferred waiting to be finalized, and (e) every submitted
    // completion has finished executing -- both before returning. The flush is safe
    // here and not on a worker: the reader and processor are joined, so this thread
    // is the only one that can still touch a deferred completion.
    flush_deferred_completions();
    join_signal_less_tasks();
}
}  // namespace kfd
}  // namespace rocprofiler
