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

// Pure dispatch-log ring drain logic, factored out of kfd_reader.cpp so it can be
// unit-tested against an in-memory buffer without a GPU, an mmap, or the reader's
// singletons. The reader wraps this with the mmap pointers and the doorbell/results
// singletons; the test wraps it with a hand-built buffer and recording callbacks.
//
// Ring geometry: the buffer holds `num_regions` regions, each with its own
// wptr[i]/rptr[i] and `region_record_count` slots (per-region, a power of two).
// Region i's records occupy slots [i*region_record_count, (i+1)*region_record_count).
// Multiple queues can multiplex into one region; records carry their own
// (doorbell_off, dispatch_id), so interleaving within a region is expected.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>

namespace rocprofiler
{
namespace kfd
{
// Firmware wire record: 20 bytes, little-endian, fixed layout (dispatch_log_format).
constexpr uint32_t kFwRecBytes = 20;

// Maximum regions the drain supports (bounds drain_state::rptr[]). Geometry with
// more regions than this is rejected rather than partially drained.
constexpr uint32_t kMaxRegions = 8;

constexpr uint32_t kRecPadding = 0;
constexpr uint32_t kRecStart   = 1;  // dispatch_start
constexpr uint32_t kRecEop     = 2;  // end-of-pipe (completion)

struct fw_record
{
    uint32_t ts_lo;         // bytes 0-3:   low 32 bits of GPU timestamp
    uint32_t ts_hi;         // bytes 4-7:   high 32 bits
    uint32_t record_type;   // bytes 8-11:  0 padding, 1 dispatch_start, 2 eop
    uint32_t dispatch_id;   // bytes 12-15: low 32 bits of HSA queue write index
    uint32_t doorbell_off;  // bytes 16-19: queue identity (demux key)
};
static_assert(sizeof(fw_record) == kFwRecBytes,
              "fw_record must match the 20-byte firmware record layout");

// Per-pipe drain cursors + unmatched starts, carried across drain calls. One
// instance lives in dlog_session (reader) or is stack-local (test).
struct drain_state
{
    uint64_t rptr[kMaxRegions] = {};     // consumer read pos per region
    bool     rptr_init         = false;  // sync rptr to wptr on first drain

    struct pending_start
    {
        uint64_t start_ticks = 0;  // GPU ticks from the dispatch_start record
        uint64_t seen_at_ns  = 0;  // host clock when recorded, for aging
    };
    // dispatch_start records awaiting their matching eop, keyed by
    // (doorbell_off << 32 | dispatch_id).
    std::unordered_map<uint64_t, pending_start> pending_starts = {};

    // Age out unmatched starts (queue died mid-dispatch, ring overwrite) so the
    // map cannot grow unbounded. now_ns/max_age_ns passed in for testability.
    size_t evict_stale(uint64_t now_ns, uint64_t max_age_ns)
    {
        size_t removed = 0;
        for(auto it = pending_starts.begin(); it != pending_starts.end();)
        {
            if(now_ns > it->second.seen_at_ns && now_ns - it->second.seen_at_ns > max_age_ns)
            {
                it = pending_starts.erase(it);
                ++removed;
            }
            else
            {
                ++it;
            }
        }
        return removed;
    }
};

// Drain new firmware records from every pipe, pairing start/eop by
// (doorbell_off, dispatch_id). Pure: no singletons, no mmap ownership, no host
// clock. Callbacks preserve the reader's ordering (bind-then-deposit):
//   on_record(doorbell_off, dispatch_id)          -- every non-padding record, in order
//   on_pair(doorbell_off, dispatch_id, start, end)-- each completed start+eop pair
// Advances rptr_arr (the shared consumer cursor firmware/kernel read) and
// state.rptr. Returns completed-pair count. Returns 0 on the first call (cursor
// sync) and on invalid geometry.
template <typename OnRecord, typename OnPair>
uint64_t
drain_pipes(const uint8_t*           records_base,
            uint32_t                 num_regions,
            uint32_t                 region_record_count,
            const volatile uint64_t* wptr_arr,
            volatile uint64_t*       rptr_arr,
            drain_state&             state,
            uint64_t                 now_ns,
            OnRecord&&               on_record,
            OnPair&&                 on_pair)
{
    if(num_regions == 0) return 0;

    // Cap at the cursor storage (drain_state::rptr[]); reject anything larger rather
    // than silently draining a subset.
    const uint32_t npipes = std::min<uint32_t>(num_regions, kMaxRegions);
    if(num_regions > kMaxRegions) return 0;

    // Per-region slot count. Must be a power of two so idx masks to a physical slot.
    const uint32_t region_slots = region_record_count;
    if(region_slots == 0 || (region_slots & (region_slots - 1)) != 0) return 0;

    // First drain: sync each pipe's read cursor to the current wptr so we do not
    // replay pre-existing records (mirrors the reference dmabuf_drain_init).
    if(!state.rptr_init)
    {
        for(uint32_t p = 0; p < npipes; ++p)
        {
            uint64_t w    = __atomic_load_n(&wptr_arr[p], __ATOMIC_ACQUIRE);
            state.rptr[p] = w;
            __atomic_store_n(&rptr_arr[p], w, __ATOMIC_RELEASE);
        }
        state.rptr_init = true;
        return 0;
    }

    auto read_rec = [&](uint32_t region, uint64_t idx) {
        uint64_t slot = static_cast<uint64_t>(region) * region_slots + (idx & (region_slots - 1));
        auto     rec  = fw_record{};
        std::memcpy(&rec, records_base + slot * kFwRecBytes, sizeof(rec));
        return rec;
    };

    uint64_t seen = 0;
    for(uint32_t p = 0; p < npipes; ++p)
    {
        uint64_t w    = __atomic_load_n(&wptr_arr[p], __ATOMIC_ACQUIRE);
        uint64_t scan = state.rptr[p];

        if(w <= scan) continue;
        // Overrun recovery: if the producer lapped us, resume just behind it. In a
        // power-of-two ring, w - region_slots aliases the producer's current slot,
        // so +1 keeps recovery strictly behind it.
        if(w - scan > region_slots) scan = w - region_slots + 1;

        for(uint64_t idx = scan; idx != w; ++idx)
        {
            auto rec = read_rec(p, idx);
            if(rec.record_type == kRecPadding || rec.doorbell_off == 0) continue;

            const uint64_t ts =
                static_cast<uint64_t>(rec.ts_lo) | (static_cast<uint64_t>(rec.ts_hi) << 32);
            const uint64_t key = (static_cast<uint64_t>(rec.doorbell_off) << 32) |
                                 static_cast<uint64_t>(rec.dispatch_id);

            on_record(rec.doorbell_off, rec.dispatch_id);

            if(rec.record_type == kRecStart)
            {
                // Overwrite: dispatch_id is only low-32, so a key can recur; a
                // collision means the prior start is stale.
                state.pending_starts[key] = drain_state::pending_start{ts, now_ns};
            }
            else if(rec.record_type == kRecEop)
            {
                auto it = state.pending_starts.find(key);
                if(it != state.pending_starts.end())
                {
                    uint64_t start_ticks = it->second.start_ticks;
                    state.pending_starts.erase(it);
                    ++seen;
                    on_pair(rec.doorbell_off, rec.dispatch_id, start_ticks, ts);
                }
                // eop with no matching start: dropped; that dispatch uses HSA.
            }
        }

        state.rptr[p] = w;
        __atomic_store_n(&rptr_arr[p], w, __ATOMIC_RELEASE);
    }
    return seen;
}
}  // namespace kfd
}  // namespace rocprofiler
