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

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

// DoorbellMap: time-as-generation identity.
//
// A firmware record carries no queue id, no VMID, no generation -- only a doorbell
// slot, a low-32 dispatch id, and a GPU timestamp. For each (gpu, doorbell_slot)
// the SDK keeps a time-ordered, non-overlapping list of OWNER WINDOWS (t_open,
// t_close). A dispatch is registered against the window open at its enqueue; a
// record is attributed by the hub to the registration whose window strictly
// contains the record's START tick. The clock is read exactly twice per queue
// lifetime (create/destroy), from that queue's own agent, never per dispatch.

namespace rocprofiler
{
namespace kfd
{
// Both the capture side (from a queue's doorbell pointer) and the reader side
// (from a firmware record's doorbell_off) reduce the doorbell identity to a
// PAGE-RELATIVE dword index. Correctness needs only that the two sides use the
// SAME modulus, so it is defined once here and shared -- they cannot drift apart.
// The value is the doorbell-page granularity (4 KiB / 1024 dwords), a fixed GPU
// aperture-layout constant, NOT the OS page size sysconf(_SC_PAGESIZE) returns:
// the mmap page size (used for the stream BO mapping in the reader) is a separate
// concern and legitimately differs on a 64 KiB-page host. The reduction drops the
// process's absolute doorbell base, which neither the pointer nor the record
// encodes.
constexpr uint32_t kDoorbellSlotsPerPage = 1024;
constexpr uint64_t kDoorbellPageBytes    = static_cast<uint64_t>(kDoorbellSlotsPerPage) * 4;

// Reader side: absolute record doorbell_off -> page-relative slot index.
inline uint32_t
doorbell_off_to_page_slot(uint32_t record_doorbell_off)
{
    return record_doorbell_off & (kDoorbellSlotsPerPage - 1);
}

// Capture side: a queue's hardware doorbell pointer -> page-relative slot index.
// The pointer's offset within its 4 KiB page, in dwords (>>2): GFX12 dispatch-log
// records store a dword index, and adjacent 8-byte doorbells are 2 dwords apart.
inline uint32_t
doorbell_ptr_to_page_slot(uint64_t hardware_doorbell_ptr)
{
    return static_cast<uint32_t>((hardware_doorbell_ptr & (kDoorbellPageBytes - 1)) >> 2);
}

// t_close sentinel: the window is still live.
constexpr uint64_t kWindowOpen = UINT64_MAX;

// One owner of a (gpu, slot) over a time interval. Immutable except for
// two write-once atomics plus one plain field, each written by exactly one thread
// at one point in the object's life: t_close and gc_deadline_ns by the destroying
// thread, superseded by the next creating thread under the map write lock.
struct owner_window
{
    uint32_t              slot        = 0;      // page-relative doorbell dword index
    bool                  first_owner = false;  // this slot had no prior owner, ever
    uint64_t              t_open      = 0;      // agent GPU tick, immutable after construction
    std::atomic<uint64_t> t_close     = {kWindowOpen};  // written once, by the destroy path
    std::atomic<bool>     superseded  = {false};        // a later window opened on this slot
    // steady ns; 0 while open. Deliberately PLAIN: the destroy thread writes it
    // BEFORE the release store of t_close, and the GC reads it only after an
    // acquire load of t_close that observed a real value, so that pair already
    // publishes it.
    uint64_t gc_deadline_ns = 0;
};
using window_ptr = std::shared_ptr<owner_window>;

struct open_result
{
    window_ptr w          = {};
    bool       overlapped = false;  // two live owners on one slot
};

// The process-wide signal-less disable latch. Header-only inline
// static so it is one instance across every TU with no link dependency. Set by
// signal_less_disable_permanently() on a doorbell-capture failure or an attach
// registration -- an owner this process never windowed exists, so first_owner can
// no longer be trusted. resolve()/open_window() then return nothing everywhere.
inline std::atomic<bool>&
signal_less_disable_latch()
{
    static std::atomic<bool> _v{false};
    return _v;
}

inline bool
signal_less_disabled()
{
    return signal_less_disable_latch().load(std::memory_order_acquire);
}

class DoorbellMap
{
public:
    DoorbellMap()  = default;
    ~DoorbellMap() = default;

    DoorbellMap(const DoorbellMap&)     = delete;
    DoorbellMap(DoorbellMap&&) noexcept = delete;
    DoorbellMap& operator=(const DoorbellMap&) = delete;
    DoorbellMap& operator=(DoorbellMap&&) noexcept = delete;

    // Queue create: open a window on (gpu, slot). `tick_sample` is read by the
    // caller BEFORE this call, from that queue's agent. Under the write lock:
    // supersede the previous window and floor t_open at its t_close (max),
    // or report `overlapped` if the previous owner is still live.
    open_result open_window(uint32_t               gpu_id,
                            rocprofiler_queue_id_t queue_id,
                            uint32_t               slot,
                            uint64_t               tick_sample);

    // Enqueue: a pure read-lock lookup of the window open for this queue, plus the
    // process-wide disable-latch check (one atomic load, no lock). No clock, no
    // bind -- the enqueue path never takes the write lock again.
    std::optional<window_ptr> resolve(uint32_t gpu_id, rocprofiler_queue_id_t queue_id);

    // Queue destroy: stamp t_close and the GC deadline (release), so any thread
    // that observes the closed window also observes its deadline. Returns the
    // window, or nullptr when the queue never opened one (SDMA, clock/capture
    // failure, attach, overlap-poisoned slot).
    window_ptr close_window(rocprofiler_queue_id_t queue_id,
                            uint64_t               tick_sample,
                            uint64_t               gc_deadline_ns);

private:
    struct map_data
    {
        std::unordered_map<uint64_t /*queue handle*/, window_ptr> by_queue;
        // Keyed by (gpu_id, doorbell_slot); time-ordered, pruned in open_window.
        std::map<std::pair<uint32_t, uint32_t>, std::vector<window_ptr>> by_slot;
    };

    common::Synchronized<map_data> m_data = {};
};
}  // namespace kfd
}  // namespace rocprofiler
