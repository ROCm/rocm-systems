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
// the kernel. It is built from the SDK's own view of queue lifecycle:
// bind_and_resolve() is called when a queue is observed at enqueue time,
// on_queue_destroyed() when a queue goes away. Destroying a queue bumps the
// doorbell's generation so a later record on a reused doorbell cannot be
// misattributed to the dead queue's dispatches.

namespace rocprofiler
{
namespace kfd
{
// Both the capture side (from a queue's doorbell pointer) and the reader side
// (from a firmware record's doorbell_off) reduce the doorbell identity to a
// PAGE-RELATIVE dword index, so they agree on the correlation key without needing
// the process's absolute doorbell base (first_db_index), which neither the pointer
// nor the record encodes. The firmware record's doorbell_off is an absolute
// BAR dword index whose base is page-aligned, so masking to the page yields the
// same per-queue value both sides compute.
//
// 4 KiB page / 4-byte dword = 1024 dword slots per page. Collisions would only
// occur if a process's doorbells spanned more than one page (>512 queues at the
// 8-byte doorbell stride), far beyond any real workload; a collision would cause a
// key mismatch -> clean HSA fallback, never misattribution.
constexpr uint32_t kDoorbellSlotsPerPage = 1024;

// Reader side: absolute record doorbell_off -> page-relative slot index.
inline uint32_t
doorbell_off_to_page_slot(uint32_t record_doorbell_off)
{
    return record_doorbell_off & (kDoorbellSlotsPerPage - 1);
}

// Capture side: a queue's hardware doorbell pointer -> page-relative slot index.
// The pointer's offset within its page, in dwords (>>2): GFX12 dispatch-log
// records store a dword index, and adjacent 8-byte doorbells are 2 dwords apart.
inline uint32_t
doorbell_ptr_to_page_slot(uint64_t hardware_doorbell_ptr, uint64_t page_size)
{
    return static_cast<uint32_t>((hardware_doorbell_ptr & (page_size - 1)) >> 2);
}
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

    // Hot-path helper for the per-dispatch capture sites. Returns the queue's
    // doorbell identity (doorbell_off + generation), binding it the first time
    // this (queue_id, doorbell_off) pair is seen. Steady state is a single
    // read lock: once a queue is bound and certain for this doorbell_off, this
    // takes no write lock. The (comparatively rare) first dispatch per queue, or
    // a rebind after doorbell reuse, takes the write lock to bind. nullopt is
    // never returned for a valid doorbell_off; callers treat a returned entry as
    // a certain, usable correlation key. Records the (queue_id, doorbell_off)
    // binding and clears the doorbell's "uncertain" mark, preserving any existing
    // generation.
    queue_doorbell_entry bind_and_resolve(rocprofiler_queue_id_t queue_id, uint32_t doorbell_off);

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

private:
    struct map_data
    {
        std::unordered_map<uint64_t /*queue handle*/, queue_doorbell_entry>      by_queue;
        std::unordered_map<uint32_t /*doorbell_off*/, uint64_t /*queue handle*/> by_doorbell;
        std::unordered_map<uint32_t /*doorbell_off*/, uint32_t /*generation*/>   generations;
        std::unordered_set<uint32_t /*doorbell_off*/>                            uncertain;
    };

    // Shared bind logic; caller holds the write lock. Returns the bound entry.
    static queue_doorbell_entry bind_locked(map_data&              data,
                                            rocprofiler_queue_id_t queue_id,
                                            uint32_t               doorbell_off);

    common::Synchronized<map_data> m_data = {};
};
}  // namespace kfd
}  // namespace rocprofiler
