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
// unit-tested against an in-memory buffer without a GPU or the reader's
// singletons.
//
// Ring geometry: `num_regions` regions, each with its own wptr[i]/rptr[i] and
// `region_record_count` slots (a power of two). Multiple queues can multiplex
// into one region; records carry their own (doorbell_off, dispatch_id).

#include <cstdint>
#include <cstring>
#include <map>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace kfd
{
// Firmware wire record: 20 bytes, little-endian, fixed layout (dispatch_log_format).
constexpr uint32_t kFwRecBytes = 20;

// Maximum regions the drain supports (bounds ring_cursors::rptr[]). Geometry with
// more regions than this is rejected rather than partially drained.
constexpr uint32_t kMaxRegions = 8;

// Dispatch-log ring size, overridable via ROCPROFILER_KFD_DISPATCH_LOG_SIZE_KB.
//
// REGISTER_BUFFER only accepts buffer_size == num_regions * 20 *
// region_record_count with region_record_count a power of two <= 2^24, and
// num_regions is ASIC-fixed but not reported until STREAM_OP_INFO, i.e. AFTER
// the size must be chosen. 80 * 2^k satisfies the rule at both 2 and 4 regions,
// so it needs no advance knowledge. k is capped at 23, making 640 MiB the
// largest accepted size. Requests are snapped DOWN onto this lattice so a
// reasonable-looking value can never become an EINVAL that disables the feature.
constexpr uint64_t kDlogMinRingBytes = 80ull << 10;  // 80 KiB; also the default
constexpr uint64_t kDlogMaxRingBytes = 80ull << 23;  // 640 MiB
constexpr uint64_t kDlogMaxRingKb    = 0xFFFFFFFFull / 1024;  // parse domain (uint32 field)

// Snap `want` down onto the 80 * 2^k lattice, clamped to
// [kDlogMinRingBytes, kDlogMaxRingBytes]. Every result is a buffer_size the
// kernel accepts whether the ASIC reports 2 or 4 regions.
inline uint64_t
dlog_snap_ring_bytes(uint64_t want)
{
    if(want >= kDlogMaxRingBytes) return kDlogMaxRingBytes;
    uint64_t sz = kDlogMinRingBytes;
    while((sz << 1) <= want)
        sz <<= 1;
    return sz;
}

// Returns the requested byte count, or 0 to mean "use the default".
inline uint64_t
dlog_ring_bytes_from_kb_str(std::string_view v)
{
    if(v.empty()) return 0;
    uint64_t kb = 0;
    for(char c : v)
    {
        if(c < '0' || c > '9') return 0;
        kb = kb * 10 + static_cast<uint64_t>(c - '0');
        if(kb > kDlogMaxRingKb) return 0;
    }
    return kb * 1024;
}

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

// `start_known` distinguishes the two EOP shapes: a matched START+EOP pair, and
// an EOP whose START was lost to a ring overwrite -- which still proves the
// kernel completed but carries no interval.
//
// `loss_free` false means the producer lapped the reader before or during the
// scan, so records around the collision may be torn and nothing may be
// published from them.
struct drained_record
{
    uint32_t doorbell_off = 0;
    uint32_t dispatch_id  = 0;
    uint64_t start_ticks  = 0;
    uint64_t end_ticks    = 0;
    bool     start_known  = false;
    bool     loss_free    = true;
};

    // Carries the copier's loss verdict, since pairing happens on another thread.
struct copied_record
{
    fw_record rec       = {};
    uint32_t  region    = 0;
    bool      loss_free = true;
};

// Reader-side state: ring cursors and loss counters. Touched ONLY by the
// ring-copier thread, so it needs no lock.
struct ring_cursors
{
    uint64_t rptr[kMaxRegions] = {};     // consumer read pos per region
    bool     rptr_init         = false;  // sync rptr to wptr on first drain

    // Overrun telemetry. The producer has lapped the reader once `w - rptr`
    // reaches region_slots -- its next write target is then the slot at rptr, so
    // that record is lost or torn.
    uint64_t overruns     = 0;  // laps observed
    uint64_t lost_records = 0;  // records the producer advanced past

    bool note_overrun(uint64_t dist, uint32_t region_slots)
    {
        if(dist < region_slots) return false;
        ++overruns;
        lost_records += dist - region_slots + 1;
        return true;
    }
};

// Processor-side state: start/eop pairing. Touched ONLY by the processor thread,
// so it needs no lock either. Deliberately separate from ring_cursors: the whole
// point of the split is that the copier never touches this.
struct pair_state
{
    struct pending_start
    {
        uint64_t start_ticks = 0;  // GPU ticks from the dispatch_start record
        uint64_t seen_at_ns  = 0;  // host clock when recorded, for aging
    };
    // dispatch_start records awaiting their matching eop, keyed by
    // (doorbell_off << 32 | dispatch_id).
    std::unordered_map<uint64_t, pending_start> pending_starts = {};

    uint64_t unmatched_eops = 0;  // EOPs whose START was lost (shape ii)

    // Pairing census: starts_seen far below eops_seen means the firmware is not
    // emitting STARTs, which is a different bug from a pairing mismatch.
    uint64_t starts_seen       = 0;
    uint64_t eops_seen         = 0;
    uint64_t starts_overwritten = 0;  // a START replaced a retained START on the same key

    // The pairing key is (doorbell_off, dispatch_id), so a START and EOP that
    // disagree on doorbell_off can never pair.
    struct doorbell_tally
    {
        uint64_t starts    = 0;
        uint64_t eops      = 0;
        uint64_t unmatched = 0;
    };
    std::map<uint32_t, doorbell_tally> per_doorbell = {};

    // Drop retained starts belonging to a page-relative doorbell slot whose queue
    // was destroyed, so a stale start cannot pair with a record from whatever
    // reuses the slot. Reader-thread only, like the rest of this state.
    size_t erase_slot(uint32_t page_slot, uint32_t slots_per_page)
    {
        size_t removed = 0;
        for(auto it = pending_starts.begin(); it != pending_starts.end();)
        {
            const auto _slot =
                static_cast<uint32_t>(it->first >> 32) & (slots_per_page - 1);
            if(_slot == page_slot)
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

    // Retained starts sharing a doorbell, with one example dispatch id.
    std::pair<size_t, uint32_t> starts_with_doorbell(uint32_t doorbell_off) const
    {
        size_t   count   = 0;
        uint32_t example = 0;
        for(const auto& itr : pending_starts)
        {
            if(static_cast<uint32_t>(itr.first >> 32) != doorbell_off) continue;
            if(count == 0) example = static_cast<uint32_t>(itr.first & 0xFFFFFFFFu);
            ++count;
        }
        return {count, example};
    }

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

// STAGE 1 (reader thread): copy raw records out of the shared ring, as fast as
// possible and touching nothing else. This is the only code reading the volatile
// mapping, and the time spent there is the window in which the producer can lap
// us -- so NO lock, no map, no pairing or timestamp math. Returns the number
// copied, or 0 on invalid geometry.
//
// ORDERING: a region's slots are copied BEFORE the release-store publishing the
// new rptr, so the kernel cannot see the slots as free while we are reading.
template <typename OutT>
uint64_t
copy_pipes(const uint8_t*           records_base,
           uint32_t                 num_regions,
           uint32_t                 region_record_count,
           const volatile uint64_t* wptr_arr,
           volatile uint64_t*       rptr_arr,
           ring_cursors&            cursors,
           OutT&                    out)
{
    if(num_regions == 0 || num_regions > kMaxRegions) return 0;

    const uint32_t region_slots = region_record_count;
    if(region_slots == 0 || (region_slots & (region_slots - 1)) != 0) return 0;

    if(!cursors.rptr_init)
    {
        for(uint32_t p = 0; p < num_regions; ++p)
        {
            cursors.rptr[p] = 0;
            __atomic_store_n(&rptr_arr[p], 0, __ATOMIC_RELEASE);
        }
        cursors.rptr_init = true;
    }

    uint64_t copied = 0;
    for(uint32_t p = 0; p < num_regions; ++p)
    {
        const uint64_t w    = __atomic_load_n(&wptr_arr[p], __ATOMIC_ACQUIRE);
        uint64_t       scan = cursors.rptr[p];
        if(w <= scan) continue;

    // The producer lapped us: slots it already overwrote are gone, and those it
    // is writing now may be torn.
        const bool lossy = cursors.note_overrun(w - scan, region_slots);
        if(w - scan > region_slots) scan = w - region_slots + 1;

        for(uint64_t idx = scan; idx != w; ++idx)
        {
            const uint64_t slot =
                static_cast<uint64_t>(p) * region_slots + (idx & (region_slots - 1));
            auto _out = copied_record{};
            std::memcpy(&_out.rec, records_base + slot * kFwRecBytes, sizeof(_out.rec));
            _out.region    = p;
            _out.loss_free = !lossy;
            out.emplace_back(_out);
            ++copied;
        }

        // Release: every memcpy above is ordered before the kernel can see these
        // slots as consumed.
        cursors.rptr[p] = w;
        __atomic_store_n(&rptr_arr[p], w, __ATOMIC_RELEASE);
    }
    return copied;
}

// STAGE 2 (processor thread): pair start/eop out of an already-copied batch.
// No ring access, so it may take locks and do timestamp work.
template <typename OnRecord>
uint64_t
pair_records(const copied_record* records,
             size_t               count,
             pair_state&          state,
             uint64_t             now_ns,
             OnRecord&&           on_record)
{
    uint64_t seen = 0;
    for(size_t i = 0; i < count; ++i)
    {
        const auto& rec = records[i].rec;
        if(rec.record_type == kRecPadding || rec.doorbell_off == 0) continue;

        const uint64_t ts =
            static_cast<uint64_t>(rec.ts_lo) | (static_cast<uint64_t>(rec.ts_hi) << 32);
        const uint64_t key = (static_cast<uint64_t>(rec.doorbell_off) << 32) |
                             static_cast<uint64_t>(rec.dispatch_id);

        if(rec.record_type == kRecStart)
        {
            ++state.starts_seen;
            ++state.per_doorbell[rec.doorbell_off].starts;
        // dispatch_id is only low-32, so a key can recur.
            if(state.pending_starts.count(key) != 0) ++state.starts_overwritten;
            state.pending_starts[key] = pair_state::pending_start{ts, now_ns};
            continue;
        }
        if(rec.record_type != kRecEop) continue;
        ++state.eops_seen;
        ++state.per_doorbell[rec.doorbell_off].eops;

        auto out         = drained_record{};
        out.doorbell_off = rec.doorbell_off;
        out.dispatch_id  = rec.dispatch_id;
        out.end_ticks    = ts;
        out.loss_free    = records[i].loss_free;

        auto it = state.pending_starts.find(key);
        if(it != state.pending_starts.end())
        {
            out.start_ticks = it->second.start_ticks;
            out.start_known = true;
            state.pending_starts.erase(it);
            ++seen;
        }
        else
        {
            // The START was lost (shape ii): the EOP still proves the kernel
            // finished, it just carries no interval.
            ++state.unmatched_eops;
            ++state.per_doorbell[rec.doorbell_off].unmatched;
        }
        on_record(out);
    }
    return seen;
}
}  // namespace kfd
}  // namespace rocprofiler
