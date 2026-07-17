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

#include "lib/common/synchronized.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>

// DoorbellMap: the SDK-side bridge between a firmware record's doorbell_off (the
// only queue identity a record carries) and the SDK's rocprofiler_queue_id_t,
// plus a per-doorbell generation counter.
//
// The active KFD ABI has no ENUM_QUEUES ioctl, so this map is NOT populated from
// the kernel. It is built from the SDK's own view of queue lifecycle: bind() is
// called when a queue is observed at enqueue time, on_queue_destroyed() when a
// queue goes away. Destroying a queue bumps the doorbell's generation so a later
// record on a reused doorbell cannot be misattributed to the dead queue's
// dispatches.

namespace rocprofiler
{
namespace kfd
{
// Snapshot of a queue's doorbell identity, valid at the moment get_by_queue()
// was called. Captured into packet_data_t at enqueue time (later phase) so the
// completion path uses a stable key.
struct queue_doorbell_entry
{
    uint32_t doorbell_off = 0;
    uint32_t generation   = 0;
};

class DoorbellMap
{
public:
    DoorbellMap()  = default;
    ~DoorbellMap() = default;

    DoorbellMap(const DoorbellMap&)     = delete;
    DoorbellMap(DoorbellMap&&) noexcept = delete;
    DoorbellMap& operator=(const DoorbellMap&) = delete;
    DoorbellMap& operator=(DoorbellMap&&) noexcept = delete;

    // Record the doorbell_off for a live queue. Idempotent for an unchanged
    // (queue_id, doorbell_off) pair. Preserves any existing generation for the
    // doorbell and clears its "uncertain" mark (the queue is now confirmed).
    void bind(rocprofiler_queue_id_t queue_id, uint32_t doorbell_off);

    // Look up a queue's current doorbell + generation snapshot. nullopt if the
    // queue is unknown.
    std::optional<queue_doorbell_entry> get_by_queue(rocprofiler_queue_id_t queue_id) const;

    // Current generation for a doorbell_off (0 if never seen).
    uint32_t get_generation(uint32_t doorbell_off) const;

    // A queue was destroyed: drop its mappings, bump the doorbell generation so
    // future records are not misattributed, and mark the doorbell uncertain
    // until a new bind() confirms it.
    void on_queue_destroyed(rocprofiler_queue_id_t queue_id);

    // False while a doorbell's owning queue is in doubt (post-destroy, pre-rebind
    // or after an event-ring drop). Callers should fall back to HSA for uncertain
    // doorbells rather than risk misattribution.
    bool is_generation_certain(uint32_t doorbell_off) const;

    // --- Empirical doorbell binding (no clean queue->doorbell_off accessor) ---
    // There is no supported API to get a queue's record doorbell_off up front, so
    // we learn it from the firmware records themselves. The capture side calls
    // note_pending_dispatch() for a dispatch on a queue whose doorbell is not yet
    // known, recording (dispatch_idx_low32 -> queue_id). The reader side, on the
    // first record it sees for an unbound doorbell, calls bind_from_record() which
    // matches the record's dispatch_id against that hint and binds the doorbell to
    // the queue. Once bound, get_by_queue() succeeds and the hint is unused.

    // Returns true if the queue already has a known doorbell (caller can build the
    // key directly); false if unbound, in which case a hint was recorded.
    bool note_pending_dispatch(rocprofiler_queue_id_t queue_id, uint32_t dispatch_idx_low32);

    // Reader side: if doorbell_off is not yet bound, try to bind it using the
    // pending hint for dispatch_id. Returns true if a bind happened (now or
    // previously), false if the doorbell is still unknown (no matching hint yet).
    bool bind_from_record(uint32_t doorbell_off, uint32_t dispatch_id);

    // True once doorbell_off has been bound to a queue.
    bool is_bound(uint32_t doorbell_off) const;

private:
    struct map_data
    {
        std::unordered_map<uint64_t /*queue handle*/, queue_doorbell_entry>      by_queue;
        std::unordered_map<uint32_t /*doorbell_off*/, uint64_t /*queue handle*/> by_doorbell;
        std::unordered_map<uint32_t /*doorbell_off*/, uint32_t /*generation*/>   generations;
        std::unordered_set<uint32_t /*doorbell_off*/>                            uncertain;
        // Empirical-bind hint: dispatch_idx_low32 -> queue handle, recorded by
        // capture for dispatches on not-yet-bound queues; consumed by the reader.
        std::unordered_map<uint32_t /*dispatch_idx_low32*/, uint64_t /*queue handle*/>
            pending_index;
    };

    common::Synchronized<map_data> m_data = {};
};
}  // namespace kfd
}  // namespace rocprofiler
