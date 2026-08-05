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
#include "lib/rocprofiler-sdk/kfd/kfd_correlation.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_reader.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <utility>

namespace rocprofiler
{
namespace kfd
{
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
        auto _v = common::get_env_optional("ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS");
        if(!_v || !parse_signal_less_env(*_v)) return false;
        ROCP_WARNING << "KFD dispatch-log: signal-less kernel-dispatch completion is ACTIVE "
                        "(ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS). Eligible batches publish "
                        "their packets untouched and complete from firmware records instead of "
                        "completion signals; ineligible batches keep the signal path. Unset the "
                        "variable to return to signal-based completion.";
        return true;
    }();
    return _enabled;
}

// Free-standing counters (constant-initialized, so no lazy-init behind them).
std::atomic<uint64_t> g_counters[static_cast<size_t>(signal_less_counter::kCount)] = {};

// Constant-initialized (no guard variable, no dynamic initialization), so the
// atfork child handler can store to it with no lazy-init machinery behind it.
std::atomic<bool> g_child_stale{false};

namespace
{
// Installed once at interposition init, before any queue exists, and published
// with a release store that the acquire load below pairs with -- so the reader
// thread either sees no ops at all or sees fully-constructed ones.
signal_less_ops&
ops_storage()
{
    static auto*& _v = common::static_object<signal_less_ops>::construct();
    return *_v;
}

std::atomic<bool>&
ops_ready()
{
    static auto _v = std::atomic<bool>{false};
    return _v;
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

ProfilingEnableTracker&
profiling_storage()
{
    static auto*& _v = common::static_object<ProfilingEnableTracker>::construct();
    return *_v;
}

retry_owner<signal_less_hub_t::proven>&
retry()
{
    static auto*& _v = common::static_object<retry_owner<signal_less_hub_t::proven>>::construct();
    return *_v;
}
}  // namespace

void
note_signal_less(signal_less_counter which, uint64_t n)
{
    // Nothing is counted unless the feature is actually active, so the default
    // path never touches these atomics.
    if(!signal_less_feature_enabled() || !signal_less_fully_wired()) return;
    g_counters[static_cast<size_t>(which)].fetch_add(n, std::memory_order_relaxed);
}

signal_less_counters
signal_less_stats()
{
    auto _at = [](signal_less_counter c) {
        return g_counters[static_cast<size_t>(c)].load(std::memory_order_relaxed);
    };
    auto _s                     = signal_less_counters{};
    _s.batch_eligible           = _at(signal_less_counter::batch_eligible);
    _s.entry_registered         = _at(signal_less_counter::entry_registered);
    _s.register_refused         = _at(signal_less_counter::register_refused);
    _s.eop_proven               = _at(signal_less_counter::eop_proven);
    _s.eop_unmatched            = _at(signal_less_counter::eop_unmatched);
    _s.handoff_submitted        = _at(signal_less_counter::handoff_submitted);
    _s.handoff_retried          = _at(signal_less_counter::handoff_retried);
    _s.finalizer_emitted        = _at(signal_less_counter::finalizer_emitted);
    _s.finalizer_no_timing      = _at(signal_less_counter::finalizer_no_timing);
    _s.no_timing_start_unknown  = _at(signal_less_counter::no_timing_start_unknown);
    _s.no_timing_convert_failed = _at(signal_less_counter::no_timing_convert_failed);
    _s.no_timing_bad_interval   = _at(signal_less_counter::no_timing_bad_interval);
    _s.no_timing_before_enqueue = _at(signal_less_counter::no_timing_before_enqueue);
    _s.no_timing_after_now      = _at(signal_less_counter::no_timing_after_now);
    return _s;
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
    if(auto* _prof = common::static_object<ProfilingEnableTracker>::get())
        _prof->abandon_in_child();
    if(auto* _retry = common::static_object<retry_owner<signal_less_hub_t::proven>>::get())
        _retry->abandon_in_child();
}

void
install_signal_less_ops(signal_less_ops ops)
{
    ops_storage() = std::move(ops);
    ops_ready().store(true, std::memory_order_release);
}

void
hand_off_proven(signal_less_hub_t::proven&& p)
{
    if(g_child_stale.load(std::memory_order_acquire)) return;
    if(!ops_ready().load(std::memory_order_acquire)) return;
    auto& _ops = ops_storage();

    auto _result = _ops.submit(p);
    if(_result == submit_result::accepted)
    {
        note_signal_less(signal_less_counter::handoff_submitted);
        return;
    }
    note_signal_less(signal_less_counter::handoff_retried);

    // The executor is closing: stop trying, the flush will finalize what is held.
    if(_result == submit_result::rejected_permanent) retry().close();

    // Deliberately NOT finalized here: this is the reader thread, which must
    // never run a client callback.
    if(!retry().hold(std::move(p), _ops.finalize_in_place))
        ROCP_WARNING << "KFD dispatch-log: signal-less retry owner is full; a completion was "
                        "finalized on the calling thread";
}

size_t
flush_retry_owner_now()
{
    if(g_child_stale.load(std::memory_order_acquire)) return 0;
    if(!ops_ready().load(std::memory_order_acquire)) return 0;
    auto& _ops = ops_storage();
    return retry().flush(_ops.submit, _ops.finalize_in_place);
}

size_t
retry_owner_size()
{
    return retry().size();
}

OwnerRegistry&
owner_registry()
{
    return registry_storage();
}

ProfilingEnableTracker&
profiling_tracker()
{
    return profiling_storage();
}

void
add_live_queue(uint64_t queue_token, uint32_t gpu_id, std::optional<uint32_t> doorbell_slot)
{
    if(g_child_stale.load(std::memory_order_acquire)) return;
    auto _result = owner_registry().add_queue(queue_token, gpu_id, doorbell_slot);
    if(_result != OwnerRegistry::add_result::collision) return;

    // A second live owner means a firmware record on this slot can no longer be
    // attributed to one queue.
    auto _stranded = signal_less_hub().quarantine_slot(gpu_id, *doorbell_slot);
    if(_stranded.empty()) return;

    note_signal_less_losses();
    ROCP_WARNING << fmt::format(
        "KFD dispatch-log: doorbell slot {} now has more than one live queue, so firmware records "
        "for it are ambiguous. The slot is quarantined for the rest of the process and {} "
        "in-flight signal-less dispatch(es) on it emit no record; those queues use signals.",
        *doorbell_slot,
        _stranded.size());
}

void
begin_close_signal_less_queue(uint64_t queue_token)
{
    if(g_child_stale.load(std::memory_order_acquire)) return;
    auto _slot = owner_registry().slot_of(queue_token);
    auto _gpu  = owner_registry().gpu_of(queue_token);
    if(!_slot || !_gpu) return;
    // Hub lock is taken and released inside; the caller fences gate_lock next.
    signal_less_hub().mark_slot_closing(*_gpu, *_slot);
}

// Per-queue ceiling on the close drain. On the reported repro a 300 ms delay
// took loss from ~46% to ~0, so the natural completion+copy+pair latency sits
// under that; 250 ms covers it while staying sub-second. Tunable because the
// right value is workload-dependent.
uint64_t
close_drain_budget_ns()
{
    static const uint64_t _v = []() {
        auto _ms = common::get_env("ROCPROFILER_KFD_DISPATCH_LOG_CLOSE_DRAIN_MS", 250);
        if(_ms < 0) _ms = 0;
        return static_cast<uint64_t>(_ms) * 1'000'000ull;
    }();
    return _v;
}

// Process-wide ceiling shared by every close, so N queues closing at teardown
// cannot multiply the per-queue budget into minutes. Comparable to the 5 s
// Queue::sync() timeout already on the destroy path, so it does not dominate.
std::atomic<uint64_t>&
close_drain_remaining_ns()
{
    static auto _v = std::atomic<uint64_t>{2'000'000'000ull};  // 2 s total
    return _v;
}

size_t
drain_close_signal_less_queue(uint64_t                             queue_token,
                              const std::function<bool(uint64_t)>& wait_hw_drained)
{
    if(g_child_stale.load(std::memory_order_acquire)) return 0;
    if(!signal_less_feature_enabled() || !signal_less_fully_wired()) return 0;

    auto _slot = owner_registry().slot_of(queue_token);
    auto _gpu  = owner_registry().gpu_of(queue_token);
    if(!_slot || !_gpu) return 0;

    auto& _hub = signal_less_hub();

    // A queue that never registered a signal-less dispatch has nothing to wait
    // for, so it pays nothing at destroy. Injectivity makes this slot's pending
    // count exactly this queue's outstanding count.
    if(_hub.pending_for_slot(*_gpu, *_slot) == 0) return 0;

    // ONE budget for both waits: the GPU finishing and the reader pairing are two
    // halves of the same close, and doubling the timeout would double teardown.
    const uint64_t _remaining = close_drain_remaining_ns().load(std::memory_order_relaxed);
    const uint64_t _budget    = std::min(close_drain_budget_ns(), _remaining);
    if(_budget == 0) return _hub.pending_for_slot(*_gpu, *_slot);

    const uint64_t _start    = steady_now_ns();
    const uint64_t _deadline = _start + _budget;

    // PHASE 1: until the hardware has finished, the EOP records for in-flight
    // dispatches do not exist yet, so polling the hub would see a low pending
    // count and strand kernels that were merely still running. Standard tracing
    // gets this from Queue::sync() waiting on _active_kernels; the inline path
    // never increments that, so we wait on the queue's own read index.
    const bool _hw_drained = wait_hw_drained ? wait_hw_drained(_deadline) : true;

    // PHASE 2: now the records exist, wait for the reader to pair them. No lock is
    // held across either wait: pending_for_slot() takes the hub lock and releases
    // it on every poll, and the caller already released gate_lock.
    auto _pending = _hub.pending_for_slot(*_gpu, *_slot);
    while(_pending > 0 && steady_now_ns() < _deadline)
    {
        nudge_reader();
        std::this_thread::sleep_for(std::chrono::microseconds{500});
        _pending = _hub.pending_for_slot(*_gpu, *_slot);
    }

    const uint64_t _spent = steady_now_ns() - _start;
    // Saturating: never wrap the shared budget below zero.
    auto& _pool = close_drain_remaining_ns();
    auto  _have = _pool.load(std::memory_order_relaxed);
    while(!_pool.compare_exchange_weak(
        _have, _spent >= _have ? 0 : _have - _spent, std::memory_order_relaxed))
    {}

    ROCP_INFO << fmt::format(
        "KFD dispatch-log: close drain for gpu {} slot {} waited {} us (hardware {}), {} "
        "dispatch(es) still outstanding",
        *_gpu,
        *_slot,
        _spent / 1000,
        _hw_drained ? "drained" : "still busy at the deadline",
        _pending);
    return _pending;
}

void
finish_close_signal_less_queue(uint64_t queue_token)
{
    if(g_child_stale.load(std::memory_order_acquire)) return;
    auto _slot = owner_registry().slot_of(queue_token);
    auto _gpu  = owner_registry().gpu_of(queue_token);
    if(!_slot || !_gpu) return;

    // Leak + permanently quarantine in one hub critical section. The returned
    // payloads are released here, off the hub lock; releasing one runs no client
    // code, so this is safe on the destroying thread.
    auto _stranded = signal_less_hub().quarantine_slot(*_gpu, *_slot);

    // The reader owns its retained starts, so ask it to purge rather than touching
    // them from this thread; results are behind their own lock and can be dropped
    // directly.
    request_reader_slot_purge(*_gpu, *_slot);

    if(_stranded.empty()) return;
    note_signal_less_losses();
    ROCP_WARNING << fmt::format(
        "KFD dispatch-log: queue destroyed with {} in-flight signal-less dispatch(es) on doorbell "
        "slot {}; they emit no record and their correlation ids are not retired. The slot is "
        "signal-path-only for the rest of the process.",
        _stranded.size(),
        *_slot);
}

void
remove_live_queue(uint64_t queue_token)
{
    if(g_child_stale.load(std::memory_order_acquire)) return;
    owner_registry().remove_queue(queue_token);
    profiling_tracker().forget(queue_token);
}

bool
signal_less_lazy_profiling()
{
    if(g_child_stale.load(std::memory_order_acquire)) return false;
    return signal_less_feature_enabled() && signal_less_fully_wired();
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

void
signal_less_teardown()
{
    // A forked child abandoned everything and owns none of it: running the
    // teardown there would join threads that do not exist.
    if(g_child_stale.load(std::memory_order_acquire)) return;

    // With the feature off there is no hub work, no retry-owner work and no
    // reader->task handoff, so the ordering constraint does not apply and the
    // existing finalize path is left byte-for-byte as it was.
    if(!signal_less_feature_enabled() || !signal_less_fully_wired()) return;

    // Strict order (design requirement 7). Each step is what makes the next final:
    //   1. stopping  -> eligibility fails, so no new PENDING is reserved
    //   2. quiesce   -> fences in-flight registration/publication
    //   3. join      -> only the reader creates PENDING->EOP_PROVEN, so after this
    //                   nothing can be added to the retry owner
    //   4. flush     -> therefore final; leftovers finalize in place on THIS thread
    //   5. leak      -> whatever never got an EOP is ledgered, so finalize skips it
    //   6. join tasks-> safe only now: no producer can submit another task
    signal_less_hub().set_mode(session_mode::stopping);
    if(ops_ready().load(std::memory_order_acquire) && ops_storage().quiesce_interceptor)
        ops_storage().quiesce_interceptor();
    stop_kfd_reader();
    const size_t _flushed = flush_retry_owner_now();

    auto         _loss   = signal_less_hub().drain_for_teardown();
    const size_t _leaked = _loss.second.dispatches;
    if(_leaked > 0)
    {
        note_signal_less_losses();
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: {} signal-less dispatch(es) across {} correlation id(s) were still "
            "in flight at finalization; they emit no record and their correlation ids are not "
            "retired.",
            _loss.second.dispatches,
            _loss.second.correlation_ids);
    }

    if(ops_ready().load(std::memory_order_acquire) && ops_storage().join_task_group)
        ops_storage().join_task_group();

    const auto _c = signal_less_stats();
    ROCP_WARNING << fmt::format(
        "KFD dispatch-log signal-less summary: {} eligible batch(es), {} entry(ies) registered ({} "
        "refused), {} EOP proven / {} unmatched, {} handed off ({} retried), {} record(s) emitted, "
        "{} completed without timing (start-unknown {} / convert-fail {} / bad-interval {} / "
        "before-enqueue {} / after-now {}); teardown finalized {} retry-owned and stranded {}",
        _c.batch_eligible,
        _c.entry_registered,
        _c.register_refused,
        _c.eop_proven,
        _c.eop_unmatched,
        _c.handoff_submitted,
        _c.handoff_retried,
        _c.finalizer_emitted,
        _c.finalizer_no_timing,
        _c.no_timing_start_unknown,
        _c.no_timing_convert_failed,
        _c.no_timing_bad_interval,
        _c.no_timing_before_enqueue,
        _c.no_timing_after_now,
        _flushed,
        _leaked);
}

void
signal_less_quiesce()
{
    if(g_child_stale.load(std::memory_order_acquire)) return;
    if(!signal_less_feature_enabled() || !signal_less_fully_wired()) return;
    if(!ops_ready().load(std::memory_order_acquire)) return;
    auto& _ops = ops_storage();

    // (a) no registration/publication is mid-flight...
    if(_ops.quiesce_interceptor) _ops.quiesce_interceptor();
    // (b) ...the reader has completed a full drain, so every record it already had
    // has been turned into a handoff...
    wait_for_reader_drain_barrier();
    // (c) ...nothing is parked waiting to be submitted...
    flush_retry_owner_now();
    // (d) ...and every submitted completion has finished executing.
    if(_ops.join_task_group) _ops.join_task_group();
}
}  // namespace kfd
}  // namespace rocprofiler
