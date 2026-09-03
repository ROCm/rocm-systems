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
// OPEN_STREAM only accepts buffer_size == num_regions * 20 * region_record_count
// with region_record_count a power of two <= 2^24, and num_regions is ASIC-fixed
// but not reported until STREAM_OP_INFO, i.e. AFTER the size must be chosen.
// 80 * 2^k satisfies the rule at both 2 and 4 regions, so it needs no advance
// knowledge. k is capped at 23, making 640 MiB the largest accepted size.
// Requests are snapped DOWN onto this lattice so a reasonable-looking value can
// never become an EINVAL that disables the feature.
constexpr uint64_t kDlogMinRingBytes     = 80ull << 10;           // 80 KiB floor
constexpr uint64_t kDlogDefaultRingBytes = 80ull << 17;           // 10 MiB (on the 80*2^k lattice)
constexpr uint64_t kDlogMaxRingBytes     = 80ull << 23;           // 640 MiB
constexpr uint64_t kDlogMaxRingKb        = 0xFFFFFFFFull / 1024;  // parse domain (uint32 field)

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
    // No loss_free field: a torn/lossy record is dropped at the top of pair_records
    // (the !copied_record.loss_free guard), so every record that reaches here is
    // trusted by construction. The loss verdict lives on the copy side
    // (copied_record.loss_free and the ring_cursors counts), not per drained record.
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

    // Overrun telemetry. Exclusive-end contract: the producer has LAPPED us only
    // once `w - rptr` EXCEEDS region_slots (== region_slots is merely exactly full).
    // overruns/lost_records are written ONLY by note_overrun.
    uint64_t overruns     = 0;  // laps observed
    uint64_t lost_records = 0;  // records the producer advanced past

    // Records copied while an aliasing producer write may have been in progress,
    // distinct from `lost` -- at exactly-full there is no overrun and no
    // loss, yet the oldest copied index may be torn. These are dropped downstream,
    // so they are real coverage loss even when nothing was lapped.
    uint64_t untrusted_records = 0;

    // wptr regressions observed (w < rptr): a stream reset that zeroed wptr[]
    // without zeroing our rptr[]. rptr is forced back to wptr so readiness
    // (wptr != rptr) converges instead of reporting ready forever with the drain
    // (w > rptr) refusing to advance.
    uint64_t wptr_regressions = 0;

    bool note_overrun(uint64_t dist, uint32_t region_slots)
    {
        if(dist <= region_slots) return false;  // == is exactly-full, not an overrun
        ++overruns;
        lost_records += dist - region_slots;
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
        // Number of outstanding STARTs sharing this raw key (>1 once a duplicate
        // arrives). `ambiguous` latches when a second outstanding START appears:
        // the raw key (doorbell_off, dispatch_id) carries no window, so a
        // recycled doorbell or a low-32 wrap makes it impossible to say which
        // dispatch an EOP belongs to. Once ambiguous, every EOP on the key is
        // dropped at this layer until `outstanding` drains to 0 or the key ages
        // out -- never forwarded as start-unknown (the hub cannot refuse it).
        uint32_t outstanding = 0;
        bool     ambiguous   = false;
    };
    // dispatch_start records awaiting their matching eop, keyed by
    // (doorbell_off << 32 | dispatch_id).
    std::unordered_map<uint64_t, pending_start> pending_starts = {};

    uint64_t unmatched_eops = 0;  // EOPs whose START was lost (shape ii)

    // Pairing census: starts_seen far below eops_seen means the firmware is not
    // emitting STARTs, which is a different bug from a pairing mismatch.
    uint64_t starts_seen        = 0;
    uint64_t eops_seen          = 0;
    uint64_t starts_overwritten = 0;  // a second START arrived on a retained key
    uint64_t ambiguous_pairs    = 0;  // EOPs dropped because their raw key was ambiguous
    uint64_t starts_evicted     = 0;  // retained STARTs aged out by the watermark

    // Stream-driven eviction watermark: the copy timestamp of the last batch that
    // triggered an evict for this GPU. Compared against batch.now_ns,
    // never the wall clock, so a backlogged processor ages nothing prematurely.
    uint64_t last_evict_ns = 0;

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
           // rptr_arr is written via __atomic_store_n, which the const check does not model.
           // NOLINTNEXTLINE(readability-non-const-parameter)
           volatile uint64_t* rptr_arr,
           ring_cursors&      cursors,
           OutT&              out)
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
        if(w < scan)
        {
            // wptr moved backwards: the stream was reset underneath us. Readiness
            // uses wptr != rptr, so leaving rptr ahead would report ready forever
            // while the w > scan drain below never runs. Snap rptr back to wptr in
            // both the local cursor and the shared rptr[] so the predicates agree.
            ++cursors.wptr_regressions;
            cursors.rptr[p] = w;
            __atomic_store_n(&rptr_arr[p], w, __ATOMIC_RELEASE);
            continue;
        }
        if(w == scan) continue;

        // The producer lapped us once dist EXCEEDS region_slots; note_overrun
        // records overruns/lost. Skip the overwritten prefix: scan = w - slots
        // (NOT w - slots + 1, which skipped one valid record).
        cursors.note_overrun(w - scan, region_slots);
        if(w - scan > region_slots) scan = w - region_slots;

        // Every copied record starts loss_free = true; the region-wide
        // verdict is gone. The untrusted back-patch below is per-record.
        const size_t region_begin = out.size();
        for(uint64_t idx = scan; idx != w; ++idx)
        {
            const uint64_t slot =
                static_cast<uint64_t>(p) * region_slots + (idx & (region_slots - 1));
            auto _out = copied_record{};
            std::memcpy(&_out.rec, records_base + slot * kFwRecBytes, sizeof(_out.rec));
            _out.region = p;
            out.emplace_back(_out);
            ++copied;
        }

        // reload wptr AFTER the copy. The producer may have advanced while
        // we memcpy'd, so any copied index the producer's next write target now
        // aliases is untrusted. w is exclusive, so index w2 aliases w2 - slots
        // (power-of-two mask); the boundary the drain sits on when it just keeps up
        // is exactly the one an acquire load of w2 cannot certify -- hence `<=`.
        const uint64_t w2 = __atomic_load_n(&wptr_arr[p], __ATOMIC_ACQUIRE);
        if(w2 >= region_slots)
        {
            const uint64_t untrusted_upto = w2 - region_slots;
            size_t         k              = region_begin;
            for(uint64_t idx = scan; idx != w; ++idx, ++k)
            {
                if(idx <= untrusted_upto)
                {
                    out[k].loss_free = false;
                    ++cursors.untrusted_records;
                }
            }
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
    // Two passes over the batch: bind every START first, then match every EOP. A
    // batch is one drain sweep of all regions in index order, and the HWS can
    // remap a queue onto a lower-numbered MEC pipe between its START and its EOP,
    // which puts the EOP in an earlier region than the START -- so in copy order
    // the EOP can precede its own START within this one batch. Firmware always
    // writes START before EOP and the sweep reads region k before region k+1, so
    // a copied EOP's START is either in an earlier batch (already consumed) or
    // later in THIS batch; binding all of this batch's STARTs before any EOP is
    // therefore exact, not a heuristic, and removes the spurious start-unknown
    // that the old single pass produced for a same-batch region reorder.
    for(int pass = 0; pass < 2; ++pass)
        for(size_t i = 0; i < count; ++i)
        {
            const auto& rec = records[i].rec;
            if(rec.record_type == kRecPadding || rec.doorbell_off == 0) continue;
            // a torn record's doorbell_off/dispatch_id are as suspect as
            // its tick, so it must never bind a START or claim an EOP. A torn START
            // is thus never retained; its later intact EOP arrives start-unknown.
            if(!records[i].loss_free) continue;
            if(pass == 0 && rec.record_type != kRecStart) continue;
            if(pass == 1 && rec.record_type == kRecStart) continue;

            const uint64_t ts =
                static_cast<uint64_t>(rec.ts_lo) | (static_cast<uint64_t>(rec.ts_hi) << 32);
            const uint64_t key = (static_cast<uint64_t>(rec.doorbell_off) << 32) |
                                 static_cast<uint64_t>(rec.dispatch_id);

            if(rec.record_type == kRecStart)
            {
                ++state.starts_seen;
                // dispatch_id is only low-32, so a raw key can recur. A second
                // outstanding START on one key makes it AMBIGUOUS: keep the first
                // START's ticks and age unchanged (so an unpairable key still
                // ages out -- refreshing seen_at_ns would make it permanent) and
                // count another outstanding. try_emplace, not [], so the first
                // START's fields survive the duplicate.
                auto [it, ins] = state.pending_starts.try_emplace(
                    key, pair_state::pending_start{ts, now_ns, 1, false});
                if(!ins)
                {
                    ++state.starts_overwritten;
                    it->second.ambiguous = true;
                    ++it->second.outstanding;
                }
                continue;
            }
            if(rec.record_type != kRecEop) continue;
            ++state.eops_seen;

            auto it = state.pending_starts.find(key);
            if(it == state.pending_starts.end())
            {
                // The START was lost (shape ii): the EOP still proves the kernel
                // finished, it just carries no interval.
                ++state.unmatched_eops;
                auto out         = drained_record{};
                out.doorbell_off = rec.doorbell_off;
                out.dispatch_id  = rec.dispatch_id;
                out.end_ticks    = ts;
                out.start_known  = false;
                on_record(out);
            }
            else if(it->second.ambiguous)
            {
                // Cannot say which dispatch this EOP belongs to. Drop it HERE --
                // forwarding it start-unknown would let the hub complete the
                // wrong dispatch (the same-window low-32-wrap shape). The key
                // stays unpairable until every outstanding START is consumed.
                ++state.ambiguous_pairs;
                if(--it->second.outstanding == 0) state.pending_starts.erase(it);
            }
            else
            {
                auto out         = drained_record{};
                out.doorbell_off = rec.doorbell_off;
                out.dispatch_id  = rec.dispatch_id;
                out.end_ticks    = ts;
                out.start_ticks  = it->second.start_ticks;
                out.start_known  = true;
                state.pending_starts.erase(it);
                ++seen;
                on_record(out);
            }
        }
    return seen;
}
}  // namespace kfd
}  // namespace rocprofiler
