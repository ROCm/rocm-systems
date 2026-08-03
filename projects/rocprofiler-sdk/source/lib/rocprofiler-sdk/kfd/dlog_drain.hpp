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

// Maximum regions the drain supports (bounds drain_state::rptr[]). Geometry with
// more regions than this is rejected rather than partially drained.
constexpr uint32_t kMaxRegions = 8;

// Dispatch-log ring size, overridable via ROCPROFILER_KFD_DISPATCH_LOG_SIZE_KB
// (KiB granularity: the useful sizes start below 1 MiB).
//
// REGISTER_BUFFER only accepts buffer_size == num_regions * 20 *
// region_record_count with region_record_count a power of two <= 2^24, and
// num_regions is ASIC-fixed (2 on gfx12, 4 on gfx9.4.x/9.5.0) but is not reported
// until STREAM_OP_INFO, i.e. AFTER the size has to be chosen. 80 * 2^k satisfies
// the rule for both counts -- region_record_count is 2^(k+1) at 2 regions and 2^k
// at 4 -- so it needs no advance knowledge of num_regions. k is capped at 23 to
// keep the 2-region count <= 2^24, which makes 80 * 2^23 == 640 MiB the largest
// accepted size. Requested sizes are snapped DOWN onto this lattice so a
// reasonable-looking value (128 KiB, say) can never become an EINVAL that
// disables the dispatch log for the whole process.
//
// Every downstream size (arr_bytes, signal_off, alloc_size, stride, aperture
// candidates) is computed in uint64 from a value <= 640 MiB and cannot overflow.
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

// Parse a ROCPROFILER_KFD_DISPATCH_LOG_SIZE_KB value. Returns the requested size
// in bytes (still to be snapped), or 0 for anything that is not a plain decimal
// integer in [1, kDlogMaxRingKb] -- empty, zero, negative, trailing junk, or too
// large. The caller warns and falls back to kDlogMinRingBytes.
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

// One completion the drain is reporting.
//
// `start_known` distinguishes the two EOP shapes: a matched START+EOP pair
// (start_ticks valid), and an EOP whose START was lost to a ring overwrite --
// which still proves the kernel completed but carries no interval.
//
// `loss_free` is the drain's verdict for the region this record came from: false
// means the producer lapped the reader before or during the scan, so the records
// around the collision may be torn and nothing may be published from them.
struct drained_record
{
    uint32_t doorbell_off = 0;
    uint32_t dispatch_id  = 0;
    uint64_t start_ticks  = 0;
    uint64_t end_ticks    = 0;
    bool     start_known  = false;
    bool     loss_free    = true;
};

// Default post-scan hook: the test seam (S1) uses this to move the producer
// between the scan and the after-scan wptr read.
struct no_post_scan_hook
{
    void operator()() const {}
};

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

    // Overrun telemetry. Phase 0 is DETECTION ONLY: the recovery in drain_pipes()
    // is unchanged, so pairing/timestamp behavior is exactly as before. The
    // producer has lapped the reader once `w - rptr` reaches region_slots -- its
    // next write target is then the slot at rptr, so that record is lost or torn.
    uint64_t overruns       = 0;  // laps observed (before or during a scan)
    uint64_t lost_records   = 0;  // records the producer advanced past the safe window
    uint64_t unmatched_eops = 0;  // EOPs whose START was lost (shape ii)

    // Staging buffer for one region's records. Reused across drains so the reader
    // does not allocate per drain; nothing is published from it until the
    // after-scan wptr read has settled the region's loss-free verdict.
    std::vector<drained_record> staged = {};

    bool note_overrun(uint64_t dist, uint32_t region_slots)
    {
        if(dist < region_slots) return false;
        ++overruns;
        lost_records += dist - region_slots + 1;
        return true;
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

// Drain new firmware records from every pipe, pairing start/eop by
// (doorbell_off, dispatch_id). Pure: no singletons, no mmap ownership, no host
// clock. on_record(const drained_record&) is invoked for every EOP, in ring
// order, AFTER that region's after-scan wptr read -- so a consumer never sees a
// record before the drain knows whether the producer lapped it (invariant 5, no
// early publication).
//
// Advances rptr_arr (the shared consumer cursor firmware/kernel read) and
// state.rptr. Returns the matched start+eop pair count, or 0 on invalid geometry;
// EOPs without a START are counted in state.unmatched_eops instead.
//
// post_scan is a test seam (S1): it runs between the scan and the after-scan wptr
// read so a producer lap during the scan can be made deterministic.
template <typename OnRecord, typename PostScanHook = no_post_scan_hook>
uint64_t
drain_pipes(const uint8_t*           records_base,
            uint32_t                 num_regions,
            uint32_t                 region_record_count,
            const volatile uint64_t* wptr_arr,
            volatile uint64_t*       rptr_arr,
            drain_state&             state,
            uint64_t                 now_ns,
            OnRecord&&               on_record,
            PostScanHook&&           post_scan = {})
{
    // Reject geometry that exceeds the cursor storage (drain_state::rptr[]) rather
    // than silently draining a subset.
    if(num_regions == 0 || num_regions > kMaxRegions) return 0;

    // Per-region slot count. Must be a power of two so idx masks to a physical slot.
    const uint32_t region_slots = region_record_count;
    if(region_slots == 0 || (region_slots & (region_slots - 1)) != 0) return 0;

    // First drain: start each pipe at the ring origin. The session buffer is
    // freshly allocated, so every written record belongs to this session and
    // should be consumed; the scan below only reads slots [rptr, wptr), never
    // past the producer, so unwritten slots are never touched. If the ring
    // already wrapped before this first drain, the overrun recovery below
    // advances past the slots the producer has physically overwritten.
    if(!state.rptr_init)
    {
        for(uint32_t p = 0; p < num_regions; ++p)
        {
            state.rptr[p] = 0;
            __atomic_store_n(&rptr_arr[p], 0, __ATOMIC_RELEASE);
        }
        state.rptr_init = true;
    }

    auto read_rec = [&](uint32_t region, uint64_t idx) {
        uint64_t slot = static_cast<uint64_t>(region) * region_slots + (idx & (region_slots - 1));
        auto     rec  = fw_record{};
        std::memcpy(&rec, records_base + slot * kFwRecBytes, sizeof(rec));
        return rec;
    };

    uint64_t seen = 0;
    for(uint32_t p = 0; p < num_regions; ++p)
    {
        uint64_t w    = __atomic_load_n(&wptr_arr[p], __ATOMIC_ACQUIRE);
        uint64_t scan = state.rptr[p];

        if(w <= scan) continue;
        const bool overran = state.note_overrun(w - scan, region_slots);
        // Overrun recovery: if the producer lapped us, resume just behind it. In a
        // power-of-two ring, w - region_slots aliases the producer's current slot,
        // so +1 keeps recovery strictly behind it.
        if(w - scan > region_slots) scan = w - region_slots + 1;

        state.staged.clear();
        for(uint64_t idx = scan; idx != w; ++idx)
        {
            auto rec = read_rec(p, idx);
            if(rec.record_type == kRecPadding || rec.doorbell_off == 0) continue;

            const uint64_t ts =
                static_cast<uint64_t>(rec.ts_lo) | (static_cast<uint64_t>(rec.ts_hi) << 32);
            const uint64_t key = (static_cast<uint64_t>(rec.doorbell_off) << 32) |
                                 static_cast<uint64_t>(rec.dispatch_id);

            if(rec.record_type == kRecStart)
            {
                // Overwrite: dispatch_id is only low-32, so a key can recur; a
                // collision means the prior start is stale.
                state.pending_starts[key] = drain_state::pending_start{ts, now_ns};
            }
            else if(rec.record_type == kRecEop)
            {
                auto out         = drained_record{};
                out.doorbell_off = rec.doorbell_off;
                out.dispatch_id  = rec.dispatch_id;
                out.end_ticks    = ts;

                auto it = state.pending_starts.find(key);
                if(it != state.pending_starts.end())
                {
                    out.start_ticks = it->second.start_ticks;
                    out.start_known = true;
                    state.pending_starts.erase(it);
                }
                // Otherwise the START was lost (shape ii): the EOP still proves the
                // kernel finished, it just carries no interval.
                state.staged.emplace_back(out);
            }
        }

        post_scan();

        // Re-read the producer frontier BEFORE publishing anything: a lap during
        // the scan means the slots just read may have been overwritten under us,
        // so the whole region's records are reported as not loss-free.
        bool overran_post = false;
        if(!overran)
            overran_post = state.note_overrun(
                __atomic_load_n(&wptr_arr[p], __ATOMIC_ACQUIRE) - state.rptr[p], region_slots);

        const bool loss_free = !overran && !overran_post;
        for(auto& rec : state.staged)
        {
            rec.loss_free = loss_free;
            if(rec.start_known)
                ++seen;
            else
                ++state.unmatched_eops;
            on_record(rec);
        }

        state.rptr[p] = w;
        __atomic_store_n(&rptr_arr[p], w, __ATOMIC_RELEASE);
    }
    return seen;
}
}  // namespace kfd
}  // namespace rocprofiler
