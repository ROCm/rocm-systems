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

#include <algorithm>
#include <atomic>
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
    bool     rptr_init         = false;  // zero rptr[] on first drain (both local and shared)

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
    uint64_t equal_tick_drops   = 0;  // EOP tick == retained START tick (fail-closed)
    uint64_t stale_eop_drops    = 0;  // EOP tick < retained START tick (belongs earlier)
    uint64_t starts_evicted     = 0;  // retained STARTs aged out by the watermark
    uint64_t starts_cap_evicted = 0;  // retained STARTs dropped by the hard size cap

    // Hard bound on pending_starts, mirroring D9's per-GPU hub cap (this map is
    // per-GPU too, and a START is of the same order in size). evict_stale is only
    // an AGE backstop -- at the 30-min default a sustained EOP loss exhausts memory
    // long before it fires -- and D9's cap does not cover this map, because
    // firmware emits records for signal-fallback dispatches the hub never saw.
    // Overridden from ROCPROFILER_KFD_DISPATCH_LOG_MAX_PENDING_STARTS by the
    // reader; the default stands for a unit test that never sets it.
    size_t max_pending_starts = 2'000'000;

    // Drop the single oldest retained START, counting its outstanding STARTs as
    // lost (each is a dispatch that will now complete start-unknown -- coverage
    // loss, not a wrong record). RESIDUAL, accepted: this is an O(live) scan, paid
    // on every insert while the map sits AT the cap. Reaching the cap already means
    // EOPs are being lost at a rate no sane workload produces, and the alternative
    // there is memory exhaustion; a cheap victim lookup would need a second index
    // ordered by seen_at_ns, the same secondary-index work D9 deferred.
    void evict_oldest_start()
    {
        auto _oldest = pending_starts.end();
        for(auto it = pending_starts.begin(); it != pending_starts.end(); ++it)
            if(_oldest == pending_starts.end() ||
               it->second.seen_at_ns < _oldest->second.seen_at_ns)
                _oldest = it;
        if(_oldest == pending_starts.end()) return;
        starts_cap_evicted += _oldest->second.outstanding;
        pending_starts.erase(_oldest);
    }

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
                // A single key can hold more than one outstanding START (a recycled
                // doorbell latched it ambiguous); each is a stranded START, so the
                // telemetry counts outstanding, not one per key.
                removed += it->second.outstanding;
                it = pending_starts.erase(it);
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
           OutT&              out,
           // Reserved per region before the first memcpy so a record is counted
           // from the instant it leaves the ring into `out`. nullptr in tests.
           std::atomic<uint64_t>* inflight = nullptr)
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
            // wptr < rptr. Per the driver contract (kfd_dlog_stream.c): wptr is
            // monotonic within a stream's lifetime -- it is zeroed only once at
            // setup, before firmware arms, and a GPU reset sets the fatal latch
            // (EPOLLERR) WITHOUT regressing wptr. So a live stream cannot legitimately
            // move wptr backwards; observing it here means a torn/corrupt read of the
            // volatile mapping (F5/F12), not a reset generation. The safe response is
            // to drop this region's un-drained span rather than trust a rewound
            // window that could mis-publish stale slots as fresh: snap rptr up to the
            // observed wptr (in both the local cursor and the shared rptr[] so
            // wptr!=rptr readiness agrees) and skip it. No new-generation drain is
            // attempted because that mid-stream reset-with-wptr-rewind state cannot
            // occur under the driver contract.
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

        // Reserve BEFORE the first memcpy: from that instant these records live in
        // `out` and will be processed while the pipe still reads empty. `w - scan`
        // is exact here (post-clamp) and is exactly what the batch's publish
        // fetch_sub removes. Release pairs with the GC gate's acquire load.
        if(inflight != nullptr) inflight->fetch_add(w - scan, std::memory_order_release);

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
        //
        // Acquire fence pairing the plain payload memcpy above with the w2 acquire
        // load below: the memcpy reads are non-atomic, and an acquire load only
        // orders operations that follow it, not ones that precede it. Without this
        // fence a weak-memory CPU (aarch64) could satisfy a payload read AFTER
        // observing w2, so the untrusted back-patch would test a stale w2 against a
        // read that actually raced the producer's overwrite. The fence keeps every
        // memcpy read sequenced before the w2 observation, so the `idx <=
        // untrusted_upto` test certifies exactly the slots that were read clean.
        std::atomic_thread_fence(std::memory_order_acquire);
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

    // Pair in TICK order, not batch/region position. Each record carries its own
    // per-agent GPU-clock tick (the stream is per-GPU, so all ticks in a batch are
    // comparable), so tick order == temporal order regardless of which region an
    // HWS remap copied a record from. Only THIS batch's records are ordered; the
    // retained cross-batch pending_starts map is the state the pass runs against
    // and is never re-fed into the work list (re-appending it would push a second
    // START transition on every retained key and latch ambiguous spuriously).
    struct work_item
    {
        uint64_t tick;         // start_ticks for a START, end_ticks for an EOP
        uint32_t kind;         // 0 = EOP, 1 = START: equal ticks run all EOPs first
        uint32_t arrival_seq;  // region-ordered merge index; ordering only, never a pairing input
        size_t   idx;          // record index in `records`
    };
    std::vector<work_item> work;
    work.reserve(count);
    for(size_t i = 0; i < count; ++i)
    {
        const auto& rec = records[i].rec;
        if(rec.record_type == kRecPadding || rec.doorbell_off == 0) continue;
        // a torn record's doorbell_off/dispatch_id are as suspect as its tick, so
        // it must never bind a START or claim an EOP. A torn START is thus never
        // retained; its later intact EOP arrives start-unknown.
        if(!records[i].loss_free) continue;
        if(rec.record_type != kRecStart && rec.record_type != kRecEop) continue;

        const uint64_t ts =
            static_cast<uint64_t>(rec.ts_lo) | (static_cast<uint64_t>(rec.ts_hi) << 32);
        work.push_back(work_item{
            ts, rec.record_type == kRecStart ? 1u : 0u, static_cast<uint32_t>(work.size()), i});
    }

    // EOP-before-START at equal ticks (the `kind` term) is what makes ties
    // order-independent AND keeps ordinary doorbell reuse intact: {E_A(t),S_B(t)}
    // pairs E_A with the outstanding S_A, then installs S_B cleanly. arrival_seq
    // only completes the strict weak ordering; it never decides a pairing.
    std::sort(work.begin(), work.end(), [](const work_item& a, const work_item& b) {
        if(a.tick != b.tick) return a.tick < b.tick;
        if(a.kind != b.kind) return a.kind < b.kind;
        return a.arrival_seq < b.arrival_seq;
    });

    for(const auto& w : work)
    {
        const auto&    rec = records[w.idx].rec;
        const uint64_t ts  = w.tick;
        const uint64_t key = (static_cast<uint64_t>(rec.doorbell_off) << 32) |
                             static_cast<uint64_t>(rec.dispatch_id);

        if(rec.record_type == kRecStart)
        {
            ++state.starts_seen;
            // Hard size cap. Only an insert that GROWS the map can breach it, so a
            // recurring key is exempt -- evicting there could pick the very key
            // about to be touched and silently reset its `ambiguous` latch. The
            // size test short-circuits, so the normal path pays no extra lookup.
            if(state.pending_starts.size() >= state.max_pending_starts &&
               state.pending_starts.find(key) == state.pending_starts.end())
                state.evict_oldest_start();
            // dispatch_id is only low-32, so a raw key can recur. A second
            // outstanding START on one key makes it AMBIGUOUS: keep the first
            // START's ticks and age unchanged (so an unpairable key still ages out
            // -- refreshing seen_at_ns would make it permanent) and count another
            // outstanding. try_emplace, not [], so the first START's fields survive.
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

        ++state.eops_seen;
        auto it = state.pending_starts.find(key);
        if(it == state.pending_starts.end())
        {
            // (1) no entry -- the START was lost: the EOP still proves the kernel
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
            // (2) ambiguous -- cannot say which dispatch this EOP belongs to.
            // Drop HERE: forwarding it start-unknown would let the hub complete the
            // wrong dispatch. The key stays unpairable until every outstanding
            // START is consumed.
            ++state.ambiguous_pairs;
            if(--it->second.outstanding == 0) state.pending_starts.erase(it);
        }
        else if(ts == it->second.start_ticks)
        {
            // (3) equal tick -- no HW contract that start_ticks < end_ticks for a
            // real pair, so fail closed: drop rather than form a zero/ambiguous pair.
            ++state.equal_tick_drops;
            if(--it->second.outstanding == 0) state.pending_starts.erase(it);
        }
        else if(ts < it->second.start_ticks)
        {
            // (4) stale EOP -- it predates the outstanding START, so it belongs to
            // an earlier dispatch whose START was lost. Drop the EOP and leave the
            // retained START untouched; consuming it would strand the EOP that
            // really belongs to it.
            ++state.stale_eop_drops;
        }
        else
        {
            // (5) outstanding==1, !ambiguous, t > start_ticks -- pair and erase.
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
