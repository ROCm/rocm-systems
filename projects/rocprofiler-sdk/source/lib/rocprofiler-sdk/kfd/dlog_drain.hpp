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
// Ring geometry: `num_regions` regions, each with its own firmware wptr[i] and a
// consumer-private rptr[i], and `region_record_count` slots (a power of two).
// Multiple queues can multiplex into one region; records carry their own
// (doorbell_off, dispatch_id).
//
// ABI v4: wptr[i] is the firmware-written WRAPPING 32-bit slot index in [0,N-1]
// (N == region_record_count). rptr[i] lives in local memory only (the BO carries
// no rptr[] and is mapped read-only). Readiness is wptr!=rptr; unread ==
// (wptr - rptr) & (N-1) in u32 arithmetic. A full lap between reads is
// undetectable and exactly-full looks empty -- accepted, documented limits.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <set>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
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
// with region_record_count a power of two, and num_regions is ASIC-fixed but not
// reported until STREAM_OP_INFO, i.e. AFTER the size must be chosen. 80 * 2^k
// satisfies the rule at both 2 and 4 regions, so it needs no advance knowledge.
// The kernel caps buffer_size at 16 MiB (KFD_DISPATCH_LOG_MAX_BUFFER_SIZE); the
// largest lattice value <= 16 MiB is 80 KiB << 7 == 10 MiB. Requests are snapped DOWN
// onto this lattice so a reasonable-looking value can never become an EINVAL that
// disables the feature.
//
// The default is the largest legal ring. ABI v4 cannot detect a producer lap, so
// region capacity IS the loss margin: the reader must revisit every region before
// the firmware writes region_record_count records into it. At ~1.8M records/s (a
// dense graph burst) a 2.5 MiB ring with 4 regions laps a region in tens of
// milliseconds -- within reach of an ordinary scheduling or page-fault stall.
// 10 MiB quadruples that margin for 7.5 MiB more GTT per GPU.
constexpr uint64_t kDlogMinRingBytes     = 80ull << 10;           // 80 KiB floor
constexpr uint64_t kDlogMaxRingBytes     = 80ull << 17;           // 10 MiB (largest <= 16 MiB cap)
constexpr uint64_t kDlogDefaultRingBytes = kDlogMaxRingBytes;     // 10 MiB (on the 80*2^k lattice)
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

// The BO carries one 8-byte wptr slot per region (u64 wptr[num_regions]); firmware
// writes only the low 32 bits, the high word stays zero. Address slot `region` by
// its BYTE offset from the wptr[] base (base + region * 8), then acquire-load the low word
// at that address (little-endian: the low word is AT the slot base, not +4). Indexing
// the array as a packed uint32_t* would use a 4-byte stride and read region i's slot
// from the wrong place -- every wptr read must go through this helper.
inline uint32_t
load_wptr_low32(const volatile void* base, uint32_t region)
{
    const auto* slot = reinterpret_cast<const volatile uint32_t*>(
        static_cast<const volatile uint8_t*>(base) +
        static_cast<uint64_t>(region) * sizeof(uint64_t));
    return __atomic_load_n(slot, __ATOMIC_ACQUIRE);
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
struct drained_record
{
    uint32_t doorbell_off = 0;
    uint32_t dispatch_id  = 0;
    uint64_t start_ticks  = 0;
    uint64_t end_ticks    = 0;
    bool     start_known  = false;
};

// One record copied out of the ring. `region` is the source region. ABI v4 has no
// reader-side loss detection (a ring lap is undetectable by contract), so there is
// no per-record loss verdict to carry.
struct copied_record
{
    fw_record rec    = {};
    uint32_t  region = 0;
};

// Reader-side state: the consumer-private ring cursors. Touched ONLY by the
// ring-copier thread, so it needs no lock. rptr[] lives here (local memory), not
// in the read-only BO. ABI v4 has no producer-visible read cursor, so there is
// no overrun/loss telemetry: a lap between reads is undetectable by contract.
struct ring_cursors
{
    uint32_t rptr[kMaxRegions] = {};     // consumer read slot index per region
    bool     rptr_init         = false;  // prime rptr[i]=wptr[i] on first drain
    // High-water of records found unread in one region at one visit. A lap is
    // undetectable, but its approach is not: this nearing region_record_count
    // means the reader came close to losing a lap of records.
    uint32_t max_unread = 0;
};

namespace detail
{
// Whether copy_pipes' output can be sized up front (std::vector can; a test sink
// that only offers emplace_back cannot).
template <typename OutT, typename = void>
struct can_reserve : std::false_type
{};

template <typename OutT>
struct can_reserve<
    OutT,
    std::void_t<decltype(std::declval<OutT&>().reserve(std::declval<OutT&>().size()))>>
: std::true_type
{};
}  // namespace detail

// Processor-side state: start/eop pairing. Touched ONLY by the processor thread,
// so it needs no lock either. Deliberately separate from ring_cursors: the whole
// point of the split is that the copier never touches this.
struct pair_state
{
    struct pending_record
    {
        copied_record record     = {};
        uint64_t      seen_at_ns = 0;
        uint64_t      epoch      = 0;
    };

    struct work_item
    {
        uint64_t key;
        uint64_t tick;
        uint32_t kind;
        size_t   index;
    };

    // Records held for one more snapshot (see pair_records), kept in pairing order.
    // The work buffers are reused across batches so steady state does not allocate.
    std::vector<pending_record> deferred_records      = {};
    std::vector<pending_record> next_deferred_records = {};
    std::vector<work_item>      pairing_work          = {};
    std::vector<work_item>      pairing_merge         = {};
    uint64_t                    deferred_epoch        = UINT64_MAX;

    struct pending_start
    {
        uint64_t start_ticks = 0;  // GPU ticks from the dispatch_start record
        uint64_t seen_at_ns  = 0;  // host clock when recorded, for aging
    };
    // dispatch_start records awaiting their matching eop, keyed by
    // (doorbell_off << 32 | dispatch_id).
    using pending_start_map = std::unordered_map<uint64_t, pending_start>;
    pending_start_map                       pending_starts = {};
    std::set<std::pair<uint64_t, uint64_t>> start_ages     = {};

    uint64_t unmatched_eops = 0;  // EOPs whose START was lost (shape ii)

    // Pairing census: starts_seen far below eops_seen means the firmware is not
    // emitting STARTs, which is a different bug from a pairing mismatch.
    uint64_t starts_seen         = 0;
    uint64_t eops_seen           = 0;
    uint64_t starts_overwritten  = 0;  // a newer START replaced a retained key's START
    uint64_t starts_out_of_order = 0;  // a START no newer than the retained one: ignored
    uint64_t equal_tick_drops    = 0;  // EOP tick == retained START tick (fail-closed)
    uint64_t stale_eop_drops     = 0;  // EOP tick < retained START tick (belongs earlier)
    uint64_t starts_evicted      = 0;  // retained STARTs aged out by the watermark
    uint64_t starts_cap_evicted  = 0;  // retained STARTs dropped by the hard size cap

    // Hard bound on pending_starts, mirroring the per-GPU hub cap (this map is
    // per-GPU too, and a START is of the same order in size). evict_stale is only
    // an AGE backstop -- at the 30-min default a sustained EOP loss exhausts memory
    // long before it fires -- and the hub cap does not cover this map, because
    // firmware emits records for signal-fallback dispatches the hub never saw.
    // Overridden from ROCPROFILER_KFD_DISPATCH_LOG_MAX_PENDING_STARTS by the
    // reader; the default stands for a unit test that never sets it.
    size_t max_pending_starts = 2'000'000;

    void erase_start(pending_start_map::iterator start)
    {
        start_ages.erase({start->second.seen_at_ns, start->first});
        pending_starts.erase(start);
    }

    // A second START on a live key is the same dispatch processed again: CP
    // replays a dispatch it discarded on a queue switch, so the newest START (by
    // tick) is the one that reached the shader engines. A START no newer than the
    // retained one is a late copy of an earlier event and is ignored.
    //
    // The size cap is NOT enforced here but by enforce_start_cap() after each
    // pairing pass, so it can never evict a START whose EOP sorts later in the
    // same batch.
    void record_start(uint64_t key, uint64_t ticks, uint64_t copied_at)
    {
        ++starts_seen;
        auto start = pending_starts.find(key);
        if(start != pending_starts.end())
        {
            if(ticks <= start->second.start_ticks)
            {
                ++starts_out_of_order;
                return;
            }
            ++starts_overwritten;
            start_ages.emplace(copied_at, key);
            if(copied_at != start->second.seen_at_ns)
                start_ages.erase({start->second.seen_at_ns, key});
            start->second = pending_start{ticks, copied_at};
            return;
        }
        auto inserted = pending_starts.emplace(key, pending_start{ticks, copied_at}).first;
        try
        {
            start_ages.emplace(copied_at, key);
        } catch(...)
        {
            pending_starts.erase(inserted);
            throw;
        }
    }

    // Drop the single oldest retained START using the same index as age cleanup.
    void evict_oldest_start()
    {
        if(start_ages.empty()) return;
        auto oldest = pending_starts.find(start_ages.begin()->second);
        ++starts_cap_evicted;
        erase_start(oldest);
    }

    // Hard bound on pending_starts, applied once a pass has paired everything it
    // can. Overshoot within a pass is bounded by one batch.
    void enforce_start_cap()
    {
        while(pending_starts.size() > max_pending_starts && !start_ages.empty())
            evict_oldest_start();
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
        while(!start_ages.empty())
        {
            const auto oldest = *start_ages.begin();
            if(now_ns <= oldest.first || now_ns - oldest.first <= max_age_ns) break;
            erase_start(pending_starts.find(oldest.second));
            ++removed;
        }
        return removed;
    }
};

// STAGE 1 (reader thread): copy raw records out of the read-only ring, as fast
// as possible and touching nothing else. This is the only code reading the
// volatile mapping -- so NO lock, no map, no pairing or timestamp math. Returns
// the number copied, or 0 on invalid geometry.
//
// ABI v4: wptr[i] is a wrapping 32-bit slot index; the consumer read cursor rptr[i]
// is local-only (no shared rptr[] to publish). On the first drain rptr is primed to
// the current wptr (discard backlog). Unread == (w - r) & (N-1) in u32 arithmetic;
// the same math handles a legacy free-running u32 producer, since only the masked
// slot and the masked delta are used. A full lap between reads is undetectable and
// exactly-full (N unread) looks empty -- accepted, documented limits.
template <typename OutT>
uint64_t
copy_pipes(const uint8_t*       records_base,
           uint32_t             num_regions,
           uint32_t             region_record_count,
           const volatile void* wptr_base,
           ring_cursors&        cursors,
           OutT&                out,
           // Reserved before the first memcpy so a record is counted
           // from the instant it leaves the ring into `out`. nullptr in tests.
           std::atomic<uint64_t>* inflight = nullptr)
{
    if(num_regions == 0 || num_regions > kMaxRegions) return 0;

    const uint32_t region_slots = region_record_count;
    if(region_slots == 0 || (region_slots & (region_slots - 1)) != 0) return 0;
    const uint32_t mask = region_slots - 1;

    if(!cursors.rptr_init)
    {
        // Prime each read cursor to the current producer index: discard any backlog
        // present at attach (a fresh KFD stream is zeroed, so this is 0 there).
        for(uint32_t p = 0; p < num_regions; ++p)
            cursors.rptr[p] = load_wptr_low32(wptr_base, p);
        cursors.rptr_init = true;
    }

    // Snapshot every region's producer index first, then copy. Sizing the output
    // once from the snapshot means the copy loop never reallocates (a growing
    // vector here was the reader's dominant cost under load, and every extra
    // microsecond in this loop is lap exposure). Records the firmware writes after
    // the snapshot are simply left for the next pass, exactly as records written
    // into an already-visited region always were.
    uint32_t wptr[kMaxRegions]   = {};
    uint32_t unread[kMaxRegions] = {};
    uint64_t total               = 0;
    for(uint32_t p = 0; p < num_regions; ++p)
    {
        wptr[p] = load_wptr_low32(wptr_base, p);
        // Unread count in the [0,N-1] wrapping domain. Wraps (w < r numerically)
        // and the N-1 -> 0 boundary are handled by the mask; exactly-full aliases
        // to empty and is indistinguishable, per the ABI's documented limits.
        unread[p] = (wptr[p] - cursors.rptr[p]) & mask;
        total += unread[p];
        cursors.max_unread = std::max(cursors.max_unread, unread[p]);
    }
    if(total == 0) return 0;

    if constexpr(detail::can_reserve<OutT>::value) out.reserve(out.size() + total);

    // Reserve BEFORE the first memcpy: from that instant these records live in
    // `out` and will be processed while the pipe still reads empty. This is
    // exactly what the batch's publish fetch_sub removes. Release pairs with
    // the GC gate's acquire load.
    if(inflight != nullptr) inflight->fetch_add(total, std::memory_order_release);

    for(uint32_t p = 0; p < num_regions; ++p)
    {
        const uint32_t r = cursors.rptr[p];
        for(uint32_t k = 0; k < unread[p]; ++k)
        {
            const uint32_t idx  = (r + k) & mask;
            const uint64_t slot = static_cast<uint64_t>(p) * region_slots + idx;
            auto           _out = copied_record{};
            std::memcpy(&_out.rec, records_base + slot * kFwRecBytes, sizeof(_out.rec));
            _out.region = p;
            out.emplace_back(_out);
        }
        cursors.rptr[p] = wptr[p];
    }
    return total;
}

// STAGE 2 (processor thread): pair start/eop out of an already-copied batch.
// No ring access, so it may take locks and do timestamp work.
template <typename OnRecord>
uint64_t
pair_records(const copied_record* records,
             size_t               count,
             pair_state&          state,
             uint64_t             now_ns,
             OnRecord&&           on_record,
             bool                 lookahead = false,
             uint64_t             epoch     = 0)
{
    uint64_t seen = 0;

    // Order each key by GPU tick across adjacent full-ring snapshots. Fresh
    // events and subsequent events on their key wait for the next snapshot:
    // both an earlier START and an earlier EOP may be in an already-sampled region.
    const size_t deferred_count = state.deferred_records.size();
    auto         record_at      = [&](size_t index) -> const copied_record& {
        return index < deferred_count ? state.deferred_records[index].record
                                                   : records[index - deferred_count];
    };
    auto& work = state.pairing_work;
    work.clear();
    work.reserve(deferred_count + count);
    size_t held_items = 0;
    for(size_t index = 0; index < deferred_count + count; ++index)
    {
        const auto& rec = record_at(index).rec;
        if(rec.record_type == kRecPadding || rec.doorbell_off == 0) continue;
        if(rec.record_type != kRecStart && rec.record_type != kRecEop) continue;

        const uint64_t ts =
            static_cast<uint64_t>(rec.ts_lo) | (static_cast<uint64_t>(rec.ts_hi) << 32);
        const uint64_t key = (static_cast<uint64_t>(rec.doorbell_off) << 32) | rec.dispatch_id;
        work.push_back(
            pair_state::work_item{key, ts, rec.record_type == kRecStart ? 1u : 0u, index});
        if(index < deferred_count) ++held_items;
    }

    // EOP-before-START at equal ticks (the `kind` term) is what makes ties
    // order-independent AND keeps ordinary doorbell reuse intact: {E_A(t),S_B(t)}
    // pairs E_A with the outstanding S_A, then installs S_B cleanly. The index
    // only completes the strict weak ordering; it never decides a pairing.
    const auto pairing_order = [](const pair_state::work_item& left,
                                  const pair_state::work_item& right) {
        if(left.key != right.key) return left.key < right.key;
        if(left.tick != right.tick) return left.tick < right.tick;
        if(left.kind != right.kind) return left.kind < right.kind;
        return left.index < right.index;
    };

    // Held records were stored in exactly this order last pass and are re-indexed
    // in storage order, so their run is already sorted: sort only this batch's
    // records, then merge the two runs. Each record is sorted once, not twice.
    const auto held_end = work.begin() + static_cast<std::ptrdiff_t>(held_items);
    std::sort(held_end, work.end(), pairing_order);
    if(held_items != 0 && held_end != work.end())
    {
        auto& merged = state.pairing_merge;
        merged.clear();
        merged.reserve(work.size());
        std::merge(
            work.begin(), held_end, held_end, work.end(), std::back_inserter(merged), pairing_order);
        work.swap(merged);
    }

    state.next_deferred_records.clear();
    state.deferred_epoch  = UINT64_MAX;
    uint64_t previous_key = 0;
    bool     defer_key    = false;
    for(const auto& w : work)
    {
        const auto&    copied       = record_at(w.index);
        const auto&    rec          = copied.rec;
        const uint64_t ts  = w.tick;
        const uint64_t key          = w.key;
        const bool     ready        = w.index < deferred_count;
        const uint64_t copied_at    = ready ? state.deferred_records[w.index].seen_at_ns : now_ns;
        const uint64_t copied_epoch = ready ? state.deferred_records[w.index].epoch : epoch;
        if(key != previous_key) defer_key = false;
        previous_key = key;
        if(defer_key || (lookahead && !ready))
        {
            defer_key = true;
            state.next_deferred_records.push_back({copied, copied_at, copied_epoch});
            state.deferred_epoch = std::min(state.deferred_epoch, copied_epoch);
            continue;
        }

        if(rec.record_type == kRecStart)
        {
            state.record_start(key, ts, copied_at);
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
        else if(ts == it->second.start_ticks)
        {
            // (2) equal tick -- no HW contract that start_ticks < end_ticks for a
            // real pair, so fail closed: drop rather than form a zero-length pair.
            ++state.equal_tick_drops;
            state.erase_start(it);
        }
        else if(ts < it->second.start_ticks)
        {
            // (3) stale EOP -- it predates the outstanding START, so it belongs to
            // an earlier dispatch whose START was lost. Drop the EOP and leave the
            // retained START untouched; consuming it would strand the EOP that
            // really belongs to it.
            ++state.stale_eop_drops;
        }
        else
        {
            // (4) t > start_ticks -- pair and erase.
            auto out         = drained_record{};
            out.doorbell_off = rec.doorbell_off;
            out.dispatch_id  = rec.dispatch_id;
            out.end_ticks    = ts;
            out.start_ticks  = it->second.start_ticks;
            out.start_known  = true;
            state.erase_start(it);
            ++seen;
            on_record(out);
        }
    }
    state.deferred_records.swap(state.next_deferred_records);
    state.next_deferred_records.clear();
    work.clear();
    state.enforce_start_cap();
    return seen;
}
}  // namespace kfd
}  // namespace rocprofiler
