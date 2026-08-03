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

#include <atomic>
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
        ROCP_WARNING << "KFD dispatch-log: signal-less kernel-dispatch completion requested via "
                        "ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS; the feature is still being "
                        "landed and remains inactive until every stage is present";
        return true;
    }();
    return _enabled;
}

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
    static auto*& _v =
        common::static_object<retry_owner<signal_less_hub_t::proven>>::construct();
    return *_v;
}
}  // namespace

void
install_signal_less_ops(signal_less_ops ops)
{
    ops_storage() = std::move(ops);
    ops_ready().store(true, std::memory_order_release);
}

void
hand_off_proven(signal_less_hub_t::proven&& p)
{
    if(!ops_ready().load(std::memory_order_acquire)) return;
    auto& _ops = ops_storage();

    auto _result = _ops.submit(p);
    if(_result == submit_result::accepted) return;

    // The executor is closing: stop trying, the flush will finalize what is held.
    if(_result == submit_result::rejected_permanent) retry().close();

    // Deliberately NOT finalized here. This runs on the reader thread, which must
    // never execute a client callback (invariant 11), and an EOP-proven entry can
    // never become LEAKED -- so it is parked until the teardown flush finalizes it
    // on a normal SDK thread.
    if(!retry().hold(std::move(p), _ops.finalize_in_place))
        ROCP_WARNING << "KFD dispatch-log: signal-less retry owner is full; a completion was "
                        "finalized on the calling thread";
}

size_t
flush_retry_owner()
{
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
    auto _result = owner_registry().add_queue(queue_token, gpu_id, doorbell_slot);
    if(_result != OwnerRegistry::add_result::collision) return;

    // A second live queue owns this slot, so a firmware record on it can no longer
    // be attributed to one dispatch. Quarantine it for the rest of the process:
    // that strands whatever was pending on the slot (P1, no record and no retire)
    // and refuses every later reservation for it, so both owners fall back to the
    // signal path. Done AFTER add_queue returned, so the registry lock is not held
    // while the hub lock is taken.
    auto _stranded = signal_less_hub().quarantine_slot(*doorbell_slot);
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
    auto _slot = owner_registry().slot_of(queue_token);
    if(!_slot) return;
    // Hub lock is taken and released inside; the caller fences gate_lock next.
    signal_less_hub().mark_slot_closing(*_slot);
}

void
finish_close_signal_less_queue(uint64_t queue_token)
{
    auto _slot = owner_registry().slot_of(queue_token);
    if(!_slot) return;

    // Leak + permanently quarantine in one hub critical section. The returned
    // payloads are released here, off the hub lock; releasing one runs no client
    // code, so this is safe on the destroying thread.
    auto _stranded = signal_less_hub().quarantine_slot(*_slot);

    // The reader owns its retained starts, so ask it to purge rather than touching
    // them from this thread; results are behind their own lock and can be dropped
    // directly.
    request_reader_slot_purge(*_slot);

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
    owner_registry().remove_queue(queue_token);
    profiling_tracker().forget(queue_token);
}

bool
signal_less_lazy_profiling()
{
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
    if(!any_leaked().load(std::memory_order_acquire)) return false;
    return signal_less_hub().is_ledgered(correlation_id);
}

bool
kfd_selection_enabled()
{
    // Emitting a KFD timestamp requires the whole signal-less path, not just the
    // env flag: until it is fully wired this stays false and every dispatch keeps
    // the Phase 1 behavior (HSA timestamps, signals retained).
    return signal_less_feature_enabled() && signal_less_fully_wired();
}
}  // namespace kfd
}  // namespace rocprofiler
