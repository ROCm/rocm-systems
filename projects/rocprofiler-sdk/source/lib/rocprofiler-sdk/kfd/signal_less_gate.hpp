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

#pragma once

#include <cstdint>
#include <string_view>

// Signal-less kernel-dispatch completion: the feature flag and the per-batch
// eligibility decision.
//
// Deliberately free of the SDK tracing/HSA headers so the decision table stays
// unit-testable on its own; the payload and the hub instance live in
// signal_less.hpp, which includes this.
//
// STATUS: the machinery is being landed in units. signal_less_fully_wired() is
// the master switch and stays false until the last one lands, so nothing here is
// observable yet regardless of the env flag. See the design plan's "Phasing".

namespace rocprofiler
{
namespace kfd
{
// ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS, read once and cached. Defaults OFF:
// only an explicit enable turns it on, so a typo or an unrelated value can never
// activate the feature.
bool
signal_less_feature_enabled();

// Whether this correlation id was deliberately leaked by the signal-less loss
// policy, in which case correlation_id_finalize() must NOT force-retire it: its
// kernel may still be running and its references were intentionally not dropped.
//
// Declared here, in the header with no tracing/HSA dependencies, so the
// correlation-id finalize path can call it without pulling the hub in. It is a
// single acquire load returning false until something is actually leaked, which
// with signal-less off is never -- so the finalize path is untouched.
bool
signal_less_id_is_leaked(uint64_t correlation_id);

// Record that the loss ledger is now non-empty. Called by the loss paths only.
void
note_signal_less_losses();

// Parse helper for the above; pure so the accepted spellings are unit-testable.
inline bool
parse_signal_less_env(std::string_view v)
{
    return v == "1" || v == "on" || v == "ON" || v == "true" || v == "TRUE" || v == "yes";
}

// MASTER SWITCH for observable signal-less behavior. The feature is being landed
// in units (hub, enqueue, reader handoff, finalizer, loss ledger, owner registry,
// generation closure, teardown order, fork); enabling it before all of them exist
// would hand dispatches to a completion path that cannot finish them. Flipping
// this to a runtime value is the LAST unit.
constexpr bool
signal_less_fully_wired()
{
    return false;
}

// Per-BATCH eligibility (design plan "Eligibility"). Every condition must hold;
// if ANY packet in the batch fails, the WHOLE batch keeps the signal path -- there
// are no mixed-mode batches. App-signal presence is deliberately NOT an input (P3).
struct eligibility_inputs
{
    bool feature_enabled       = false;  // env flag
    bool fully_wired           = false;  // signal_less_fully_wired()
    bool session_live_for_gpu  = false;  // a dlog session exists for THIS gpu
    bool reader_alive          = false;  // not poisoned / reader-dead
    bool doorbells_injective   = false;  // every packet's slot has one live owner
    bool hub_accepts_batch     = false;  // no live/tombstoned key, no quarantined slot
    bool payload_constructible = false;  // owned payload can be built for every packet
};

inline bool
batch_is_signal_less_eligible(const eligibility_inputs& in)
{
    return in.feature_enabled && in.fully_wired && in.session_live_for_gpu && in.reader_alive &&
           in.doorbells_injective && in.hub_accepts_batch && in.payload_constructible;
}

// Whether the doorbell slot has exactly ONE live owner across ALL live compute
// queues on the session GPU (correctness requirement 3).
//
// PLACEHOLDER: returns false until the all-live-queue reverse owner registry
// lands. Current-owner uniqueness alone is explicitly NOT sufficient (it cannot
// tell a current owner from a previous generation of the slot), so until the
// registry exists this reports "not trackable" and every batch stays on the
// signal path.
inline bool
doorbell_owner_is_injective(uint32_t /*doorbell_off*/)
{
    return false;
}
}  // namespace kfd
}  // namespace rocprofiler
