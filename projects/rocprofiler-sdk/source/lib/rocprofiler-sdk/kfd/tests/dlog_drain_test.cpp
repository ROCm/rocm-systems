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

// Unit tests for the dispatch-log ring drain against a hand-built in-memory ring.
// Geometry mirrors GFX12: num_regions=2, region_record_count=2048. The reader
// stage copies into a record_stream sink (P3): a fake session drives dlog_pass
// exactly as copy_records does, and the copy-out step turns published slots back
// into a plain record vector for pair_records.

#include "lib/rocprofiler-sdk/kfd/dlog_drain.hpp"
#include "lib/rocprofiler-sdk/kfd/record_processor.hpp"
#include "lib/rocprofiler-sdk/kfd/record_stream.hpp"
#include "lib/rocprofiler-sdk/kfd/stream_geometry.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <sys/mman.h>

namespace
{
using namespace rocprofiler::kfd;
// A hand-built ring: num_regions*rrc record slots plus a per-pipe wptr array. ABI
// v4 has no rptr[] in the BO (the read cursor is consumer-private, in ring_cursors)
// and wptr[i] is a wrapping 32-bit slot index in [0,rrc-1].
//
// wptr is REAL 8-byte-stride storage (one u64 slot per region, low word holds the
// value, high word stays zero), exactly like the kernel BO. copy_pipes() must read
// it with an 8-byte stride via load_wptr_low32; a 4-byte-stride reader would pick up
// region i>0's slot from the wrong address, which is the stride bug these tests must
// catch -- so the double must not be a compact uint32 array.
struct fake_ring
{
    uint32_t              num_regions;
    uint32_t              rrc;  // region_record_count (per-region slot count)
    std::vector<uint8_t>  records;
    std::vector<uint64_t> wptr;
    fake_ring(uint32_t nreg, uint32_t region_record_count)
    : num_regions(nreg)
    , rrc(region_record_count)
    , records(static_cast<size_t>(nreg) * region_record_count * kFwRecBytes, 0)
    , wptr(nreg, 0)
    {}
    // Region r's slots are [r*rrc, (r+1)*rrc); idx masks into that region.
    void put(uint32_t region,
             uint64_t idx,
             uint32_t rtype,
             uint32_t dispatch_id,
             uint32_t doorbell_off,
             uint64_t ts)
    {
        uint64_t  slot = static_cast<uint64_t>(region) * rrc + (idx & (rrc - 1));
        fw_record rec{};
        rec.ts_lo        = static_cast<uint32_t>(ts & 0xFFFFFFFFu);
        rec.ts_hi        = static_cast<uint32_t>(ts >> 32);
        rec.record_type  = rtype;
        rec.dispatch_id  = dispatch_id;
        rec.doorbell_off = doorbell_off;
        std::memcpy(records.data() + slot * kFwRecBytes, &rec, sizeof(rec));
    }
};
// Recording sink: matched pairs and START-less EOPs (shape ii) kept apart, plus a
// count of records observed.
struct recorder
{
    std::map<std::pair<uint32_t, uint32_t>, std::pair<uint64_t, uint64_t>> pairs;  // -> (start,end)
    std::vector<std::pair<uint32_t, uint32_t>>                             eops_without_start;
    size_t                                                                 records = 0;
    auto                                                                   on_record()
    {
        return [this](const drained_record& r) {
            ++records;
            if(r.start_known)
                pairs[{r.doorbell_off, r.dispatch_id}] = {r.start_ticks, r.end_ticks};
            else
                eops_without_start.emplace_back(r.doorbell_off, r.dispatch_id);
        };
    }
};

// The counting mapping policy: counts maps/unmaps and poisons fresh pages with
// 0xA5, so reads past `published` show.
struct test_mapping
{
    static inline std::atomic<size_t> map_calls{0};
    static inline std::atomic<size_t> map_successes{0};
    static inline std::atomic<size_t> unmap_calls{0};

    static void reset()
    {
        map_calls.store(0, std::memory_order_relaxed);
        map_successes.store(0, std::memory_order_relaxed);
        unmap_calls.store(0, std::memory_order_relaxed);
    }

    static void* map(size_t bytes) noexcept
    {
        map_calls.fetch_add(1, std::memory_order_relaxed);
        void* const p =
            ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if(p == MAP_FAILED) return nullptr;
        std::memset(p, 0xA5, bytes);
        map_successes.fetch_add(1, std::memory_order_relaxed);
        return p;
    }

    static void unmap(void* p, size_t bytes) noexcept
    {
        unmap_calls.fetch_add(1, std::memory_order_relaxed);
        ::munmap(p, bytes);
    }
};

using drain_stream = basic_record_stream<test_mapping>;

// copy_out sink that flattens the published runs into one vector, the shape
// pair_records consumes.
struct sink_vector
{
    std::vector<fw_record>* out;
    void                    operator()(const fw_record* p, size_t n, uint64_t) const
    {
        out->insert(out->end(), p, p + n);
    }
};

// copy_pipes sink adapter: delegates every append to the real stream, then
// fires the producer-progress hook, so a test can advance the ring DURING the
// copy (after the snapshot) the way the production reader's producer races.
struct progress_stream
{
    drain_stream*          real    = nullptr;
    std::function<void()>* publish = nullptr;
    void                   append(const void* p, size_t n, reader_alloc_log& log)
    {
        real->append(p, n, log);
        if(publish != nullptr) (*publish)();
    }
};

// The two production stages back to back: reader copies the ring into the
// session's stream, processor pairs the copied-out records.
struct drain_state
{
    ring_cursors           cursors = {};
    pair_state             pairing = {};
    drain_stream           stream  = {};
    reader_alloc_log       log     = {};
    std::vector<fw_record> batch   = {};
};
// Copy one ring into `st.stream` (one pass of dlog_pass, without the marker
// rules -- the pairing tests below append markers explicitly), publish, copy out
// into st.batch, and return the copied count.
uint64_t
copy_ring(fake_ring& ring, drain_state& st)
{
    if(!st.stream.armed())
    {
        EXPECT_TRUE(st.stream.preallocate(4));
    }
    st.batch.clear();
    const auto copied = copy_pipes(ring.records.data(),
                                   ring.num_regions,
                                   ring.rrc,
                                   ring.wptr.data(),
                                   st.cursors,
                                   st.stream,
                                   st.log);
    st.stream.publish();
    st.stream.copy_out(sink_vector{&st.batch}, 4);
    return copied;
}
uint64_t
run_drain(fake_ring& ring, drain_state& st, recorder& rec, uint64_t now_ns = 1000)
{
    copy_ring(ring, st);
    return pair_records(st.batch.data(), st.batch.size(), st.pairing, now_ns, rec.on_record());
}
// Prime a fresh ring: first drain syncs each pipe cursor to the origin.
void
prime_ring(fake_ring& ring, drain_state& st)
{
    EXPECT_TRUE(st.stream.preallocate(4));
    recorder rec0;
    run_drain(ring, st, rec0);
}

// A fresh, already-primed drain environment: ring at [nreg,rrc], synced cursors.
struct env
{
    fake_ring   ring;
    drain_state st;
    recorder    rec;
    env(uint32_t nreg = 2, uint32_t rrc = 2048)
    : ring(nreg, rrc)
    {
        prime_ring(ring, st);
    }
    uint64_t drain() { return run_drain(ring, st, rec); }
};

// One GPU's ring + cursors + copied batch, primed to origin; copy() refills the batch.
struct gpu_ring
{
    fake_ring              ring;
    ring_cursors           cur    = {};
    drain_stream           stream = {};
    reader_alloc_log       log    = {};
    std::vector<fw_record> batch  = {};
    gpu_ring(uint32_t rrc, uint32_t nreg = 1)
    : ring(nreg, rrc)
    {
        stream.preallocate(4);
        copy();  // prime cursor to origin
    }
    uint64_t copy()
    {
        batch.clear();
        const auto copied = copy_pipes(
            ring.records.data(), ring.num_regions, ring.rrc, ring.wptr.data(), cur, stream, log);
        stream.publish();
        stream.copy_out(sink_vector{&batch}, 4);
        return copied;
    }
};
}  // namespace

// Core start/eop pairing scenarios driven through the two-stage drain.
TEST(dlog_drain, pairing_core)
{
    const uint32_t db = 4100;
    // ABI v4: the first drain PRIMES rptr := wptr (discards any backlog present at
    // attach). Records written after attach are consumed on the next drain. (Under
    // v3 the first drain zeroed rptr and consumed the backlog from the origin.)
    {
        fake_ring   ring(2, 2048);
        drain_state st;
        recorder    rec;
        ring.put(0, 0, kRecStart, 7, db, 111);
        ring.put(0, 1, kRecEop, 7, db, 222);
        ring.wptr[0] = 2;
        EXPECT_EQ(run_drain(ring, st, rec), 0u) << "backlog at attach is discarded";
        EXPECT_TRUE(rec.pairs.empty());
        EXPECT_EQ(st.cursors.rptr[0], 2u);
        EXPECT_TRUE(st.cursors.rptr_init);
        // A pair written after attach drains normally.
        ring.put(0, 2, kRecStart, 8, db, 333);
        ring.put(0, 3, kRecEop, 8, db, 444);
        ring.wptr[0] = 4;
        EXPECT_EQ(run_drain(ring, st, rec), 1u);
        ASSERT_EQ(rec.pairs.count(std::make_pair(db, 8u)), 1u);
        EXPECT_EQ(rec.pairs[std::make_pair(db, 8u)].first, 333u);
        EXPECT_EQ(st.cursors.rptr[0], 4u);
    }
    // A single pipe with N pairs: all pair, correct ticks, rptr advances.
    {
        env e;
        for(uint32_t i = 0; i < 40; ++i)
        {
            e.ring.put(0, 2 * i, kRecStart, i, db, 1000 + i);
            e.ring.put(0, 2 * i + 1, kRecEop, i, db, 2000 + i);
        }
        e.ring.wptr[0] = 80;
        EXPECT_EQ(e.drain(), 40u);
        EXPECT_EQ(e.rec.pairs.size(), 40u);
        EXPECT_EQ(e.st.cursors.rptr[0], 80u);
        for(uint32_t i = 0; i < 40; ++i)
        {
            auto it = e.rec.pairs.find({db, i});
            ASSERT_NE(it, e.rec.pairs.end());
            EXPECT_EQ(it->second.first, 1000u + i);
            EXPECT_EQ(it->second.second, 2000u + i);
        }
    }
    // Two pipes with DIFFERENT counts prove per-pipe indexing (each wptr[i] one doorbell).
    {
        env            e;
        const uint32_t dbA = 4100, dbB = 4102;
        for(uint32_t i = 0; i < 20; ++i)
        {
            e.ring.put(0, 2 * i, kRecStart, i, dbA, 100 + i);
            e.ring.put(0, 2 * i + 1, kRecEop, i, dbA, 500 + i);
        }
        e.ring.wptr[0] = 40;
        for(uint32_t i = 0; i < 40; ++i)
        {
            e.ring.put(1, 2 * i, kRecStart, i, dbB, 700 + i);
            e.ring.put(1, 2 * i + 1, kRecEop, i, dbB, 900 + i);
        }
        e.ring.wptr[1] = 80;
        EXPECT_EQ(e.drain(), 60u);
        uint32_t a = 0, b = 0;
        for(auto& kv : e.rec.pairs)
        {
            if(kv.first.first == dbA) ++a;
            if(kv.first.first == dbB) ++b;
        }
        EXPECT_EQ(a, 20u);
        EXPECT_EQ(b, 40u);
        EXPECT_EQ(e.st.cursors.rptr[0], 40u);
        EXPECT_EQ(e.st.cursors.rptr[1], 80u);
    }
    // Padding slots (type==0 or doorbell==0) are skipped, not scan-stopping.
    {
        env e;
        e.ring.put(0, 0, kRecStart, 5, db, 10);
        e.ring.put(0, 2, kRecEop, 5, db, 20);  // slot 1 left as padding
        e.ring.wptr[0] = 3;
        EXPECT_EQ(e.drain(), 1u);
        auto key = std::make_pair(db, 5u);
        ASSERT_EQ(e.rec.pairs.count(key), 1u);
        EXPECT_EQ(e.rec.pairs[key].first, 10u);
        EXPECT_EQ(e.rec.pairs[key].second, 20u);
    }
    // A start-less EOP is reported (start_known=false) but is not counted as a pair.
    {
        env e;
        e.ring.put(0, 0, kRecEop, 9, 4100, 42);
        e.ring.wptr[0] = 1;
        EXPECT_EQ(e.drain(), 0u);
        EXPECT_TRUE(e.rec.pairs.empty());
        ASSERT_EQ(e.rec.eops_without_start.size(), 1u);
        EXPECT_EQ(e.rec.eops_without_start[0].first, 4100u);
        EXPECT_EQ(e.rec.eops_without_start[0].second, 9u);
        EXPECT_EQ(e.st.pairing.unmatched_eops, 1u);
    }
    // A start in one drain pairs with its eop in a LATER drain (state persists).
    {
        env e;
        e.ring.put(0, 0, kRecStart, 3, db, 111);
        e.ring.wptr[0] = 1;
        EXPECT_EQ(e.drain(), 0u);  // start seen, not yet paired
        e.rec = recorder{};
        e.ring.put(0, 1, kRecEop, 3, db, 222);
        e.ring.wptr[0] = 2;
        EXPECT_EQ(e.drain(), 1u);
        auto key = std::make_pair(db, 3u);
        ASSERT_EQ(e.rec.pairs.count(key), 1u);
        EXPECT_EQ(e.rec.pairs[key].first, 111u);
        EXPECT_EQ(e.rec.pairs[key].second, 222u);
    }
    // Ring wrap: a pair straddling the power-of-two boundary maps to the right slots.
    {
        env e;
        e.st.cursors.rptr[0] = 2047;                 // drain [2047, 2049)
        e.ring.put(0, 2047, kRecStart, 7, db, 500);  // physical slot 2047
        e.ring.put(0, 2048, kRecEop, 7, db, 600);    // 2048 & 2047 = physical slot 0
        e.ring.wptr[0] = 2049;
        EXPECT_EQ(e.drain(), 1u);
        ASSERT_EQ(e.rec.pairs.count(std::make_pair(db, 7u)), 1u);
        EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].first, 500u);
        EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].second, 600u);
    }
}

// evict_stale drops unmatched starts older than max_age, keeps fresh ones.
TEST(dlog_drain, evict_stale_starts)
{
    drain_state st;
    st.pairing.record_start(1, 100, 1000);
    st.pairing.record_start(2, 200, 5000);
    EXPECT_EQ(st.pairing.evict_stale(/*now_ns=*/6000, /*max_age_ns=*/2000), 1u);
    EXPECT_EQ(st.pairing.pending_starts.count(1), 0u);
    EXPECT_EQ(st.pairing.pending_starts.count(2), 1u);
}

// The hard size cap bounds pending_starts even when nothing ages out: past the
// cap an insert evicts the OLDEST retained START (by seen_at_ns) and counts it.
TEST(dlog_drain, pending_starts_size_cap_evicts_oldest)
{
    auto mk_start = [](uint32_t dispatch_id, uint64_t ts) {
        auto c         = fw_record{};
        c.record_type  = kRecStart;
        c.doorbell_off = 4100;
        c.dispatch_id  = dispatch_id;
        c.ts_lo        = static_cast<uint32_t>(ts);
        return c;
    };

    pair_state pairing;
    pairing.max_pending_starts = 2;
    recorder rec;

    // Three distinct keys, each in its own batch so seen_at_ns strictly increases.
    for(uint32_t i = 1; i <= 3; ++i)
    {
        auto batch = std::vector<fw_record>{mk_start(i, 10 * i)};
        pair_records(batch.data(), batch.size(), pairing, /*now_ns=*/1000 * i, rec.on_record());
    }

    EXPECT_EQ(pairing.pending_starts.size(), 2u) << "the cap is a hard bound";
    EXPECT_EQ(pairing.starts_cap_evicted, 1u);
    const uint64_t k1 = (uint64_t{4100} << 32) | 1u;
    EXPECT_EQ(pairing.pending_starts.count(k1), 0u) << "the oldest START is the victim";

    // A recurring key is exempt: it cannot grow the map, and evicting there could
    // pick the very key being refreshed.
    const uint64_t k3    = (uint64_t{4100} << 32) | 3u;
    auto           again = std::vector<fw_record>{mk_start(3, 99)};
    pair_records(again.data(), again.size(), pairing, /*now_ns=*/4000, rec.on_record());
    EXPECT_EQ(pairing.pending_starts.size(), 2u);
    EXPECT_EQ(pairing.starts_cap_evicted, 1u) << "no eviction for a recurring key";
    ASSERT_EQ(pairing.pending_starts.count(k3), 1u);
    EXPECT_EQ(pairing.pending_starts[k3].start_ticks, 99u) << "the newest START replaces it";
}

// The cap is enforced after the pass, never mid-pass: a START retained from an
// earlier batch whose EOP sorts AFTER a new key's START in this batch must still
// pair, not be evicted first and complete start-unknown.
TEST(dlog_drain, start_cap_spares_a_start_whose_eop_is_in_the_same_batch)
{
    const uint32_t db          = 4100;
    auto           pairing     = pair_state{};
    pairing.max_pending_starts = 1;
    auto rec                   = recorder{};

    const auto retained = fw_record{100, 0, kRecStart, 9, db};
    pair_records(&retained, 1, pairing, 1000, rec.on_record());
    ASSERT_EQ(pairing.pending_starts.size(), 1u);

    // dispatch 1 sorts before dispatch 9, so its START is processed first.
    const auto batch = std::vector<fw_record>{{200, 0, kRecStart, 1, db}, {300, 0, kRecEop, 9, db}};
    EXPECT_EQ(pair_records(batch.data(), batch.size(), pairing, 2000, rec.on_record()), 1u);
    ASSERT_EQ(rec.pairs.count(std::make_pair(db, 9u)), 1u);
    EXPECT_EQ(rec.pairs[std::make_pair(db, 9u)].first, 100u);
    EXPECT_TRUE(rec.eops_without_start.empty());
    EXPECT_EQ(pairing.starts_cap_evicted, 0u);
    EXPECT_EQ(pairing.pending_starts.size(), 1u) << "the new START is retained within the cap";
}

// copy_pipes records how full a region got between reads: the only observable
// precursor of an (undetectable) lap.
TEST(dlog_drain, copy_tracks_peak_unread_per_region)
{
    gpu_ring g(8, 2);
    g.ring.wptr[0] = 5;
    g.ring.wptr[1] = 2;
    EXPECT_EQ(g.copy(), 7u);
    EXPECT_EQ(g.cur.max_unread, 5u);
    g.ring.wptr[1] = 5;
    EXPECT_EQ(g.copy(), 3u);
    EXPECT_EQ(g.cur.max_unread, 5u) << "a high-water mark, not the last visit";
}

TEST(dlog_drain, start_age_index_tracks_replay_completion_and_equal_tick_drop)
{
    auto state = pair_state{};
    auto sink  = recorder{};
    state.record_start((uint64_t{4100} << 32) | 7, 100, 1000);
    state.record_start((uint64_t{4100} << 32) | 7, 200, 1000);
    EXPECT_EQ(state.start_ages.size(), 1u);
    state.record_start((uint64_t{4100} << 32) | 7, 300, 500);
    ASSERT_EQ(state.start_ages.size(), 1u);
    EXPECT_EQ(state.start_ages.begin()->first, 500u);
    const auto completion = fw_record{400, 0, kRecEop, 7, 4100};
    EXPECT_EQ(pair_records(&completion, 1, state, 2000, sink.on_record()), 1u);
    EXPECT_TRUE(state.start_ages.empty());
    EXPECT_TRUE(state.pending_starts.empty());

    state.record_start((uint64_t{4100} << 32) | 7, 400, 3000);
    pair_records(&completion, 1, state, 4000, sink.on_record());
    EXPECT_EQ(state.equal_tick_drops, 1u);
    EXPECT_TRUE(state.start_ages.empty());
    EXPECT_TRUE(state.pending_starts.empty());
}

// Invalid geometry (0 regions, too many, or non-power-of-two rrc) is rejected.
TEST(dlog_drain, invalid_geometry_rejected)
{
    drain_state st;
    recorder    rec;
    EXPECT_EQ(copy_pipes(nullptr, 0, 2048, nullptr, st.cursors, st.stream, st.log), 0u);
    EXPECT_EQ(copy_pipes(nullptr, kMaxRegions + 1, 2048, nullptr, st.cursors, st.stream, st.log),
              0u);
    fake_ring ring(2, 3000);  // region_record_count not a power of two
    EXPECT_EQ(run_drain(ring, st, rec), 0u);
    EXPECT_TRUE(rec.pairs.empty());
}

// Stride regression guard: on a 4-region ring, put records ONLY in regions 1, 2
// and 3 (region 0 empty). The BO stores one 8-byte wptr slot per region, so the
// only way to read region i's slot is with an 8-byte stride. A 4-byte-stride
// (packed uint32_t*) reader would fetch region 1's wptr from region 0's high word
// (always 0) and drain nothing for regions >= 1 -- exactly the AIPROFSDK stride
// bug. This fails under that bug and passes only with load_wptr_low32.
TEST(dlog_drain, per_region_wptr_uses_eight_byte_stride)
{
    env            e(4, 8);
    const uint32_t db[4] = {0, 4101, 4102, 4103};
    for(uint32_t r = 1; r < 4; ++r)
    {
        e.ring.put(r, 0, kRecStart, r, db[r], 100 + r);
        e.ring.put(r, 1, kRecEop, r, db[r], 200 + r);
        e.ring.wptr[r] = 2;
    }
    // Region 0 stays empty: wptr[0] == 0. Regions 1-3 each carry one pair.
    EXPECT_EQ(e.drain(), 3u);
    EXPECT_EQ(e.rec.pairs.size(), 3u);
    for(uint32_t r = 1; r < 4; ++r)
    {
        auto key = std::make_pair(db[r], r);
        ASSERT_EQ(e.rec.pairs.count(key), 1u) << "region " << r << " drained nothing";
        EXPECT_EQ(e.rec.pairs[key].first, 100u + r);
        EXPECT_EQ(e.rec.pairs[key].second, 200u + r);
    }
    for(uint32_t r = 0; r < 4; ++r)
        EXPECT_EQ(e.st.cursors.rptr[r], e.ring.wptr[r]) << "region " << r << " cursor";
}

// ABI v4 wrap boundary: draining a span that straddles the N-1 -> 0 slot boundary
// (wptr numerically < rptr) reads the right physical slots and advances rptr to
// wptr. Under the old v3 free-running arithmetic w < rptr was treated as a wptr
// regression and the span was DROPPED (snap rptr:=w, copy nothing), so this pair
// would never drain -- this case exercises the v4 wrapping semantics directly.
TEST(dlog_drain, v4_wrap_boundary_wptr_less_than_rptr)
{
    const uint32_t db = 4100;
    fake_ring      ring(1, 8);  // N = 8, mask 7
    drain_state    st;
    recorder       rec0;
    ring.wptr[0] = 7;  // prime the read cursor to slot 7
    run_drain(ring, st, rec0);
    ASSERT_EQ(st.cursors.rptr[0], 7u);

    // Producer writes slot 7 then wraps to slot 0, publishing wptr = 1 (< rptr=7).
    ring.put(0, 7, kRecStart, 5, db, 100);  // physical slot 7
    ring.put(0, 0, kRecEop, 5, db, 200);    // wrapped: physical slot 0
    ring.wptr[0] = 1;                       // wraps: unread = (1 - 7) & 7 = 2
    recorder rec;
    EXPECT_EQ(run_drain(ring, st, rec), 1u);
    ASSERT_EQ(rec.pairs.count(std::make_pair(db, 5u)), 1u);
    EXPECT_EQ(rec.pairs[std::make_pair(db, 5u)].first, 100u);
    EXPECT_EQ(rec.pairs[std::make_pair(db, 5u)].second, 200u);
    EXPECT_EQ(st.cursors.rptr[0], 1u);
}

// ABI v4 readiness/count boundaries: empty (w==r) copies nothing; exactly one
// record; and a full N-1 unread span (the largest an in-domain wptr can express)
// drains every record without any loss accounting.
TEST(dlog_drain, v4_empty_one_and_full_span)
{
    const uint32_t db = 4100;
    // Empty: primed cursor equals wptr, so nothing is copied.
    {
        fake_ring   ring(1, 8);
        drain_state st;
        recorder    rec;
        ring.wptr[0] = 3;
        run_drain(ring, st, rec);  // prime rptr := 3
        EXPECT_EQ(run_drain(ring, st, rec), 0u);
        EXPECT_EQ(st.cursors.rptr[0], 3u);
    }
    // Exactly one record.
    {
        env e(1, 8);
        e.ring.put(0, 0, kRecEop, 9, db, 42);  // start-less EOP, still one record
        e.ring.wptr[0] = 1;
        EXPECT_EQ(e.drain(), 0u);  // no pair, but the record was drained
        EXPECT_EQ(e.rec.eops_without_start.size(), 1u);
        EXPECT_EQ(e.st.cursors.rptr[0], 1u);
    }
    // Full N-1 unread span drains all N-1 records (exactly-full N would alias to
    // empty -- indistinguishable, per the ABI's documented limit).
    {
        env            e(1, 8);
        const uint32_t db2 = 4200;
        for(uint32_t i = 0; i < 7; ++i)
            e.ring.put(0, i, kRecStart, i, db2, 100 + i);
        e.ring.wptr[0] = 7;  // unread = (7 - 0) & 7 = 7 == N-1
        EXPECT_EQ(e.drain(), 0u);
        EXPECT_EQ(e.st.pairing.starts_seen, 7u);
        EXPECT_EQ(e.st.cursors.rptr[0], 7u);
    }
}

// ABI v4: draining across the N-1 -> 0 index transition maps to physical slots
// N-1 then 0 (the wrapping mask), not to an out-of-range slot.
TEST(dlog_drain, v4_index_n_minus_1_to_zero)
{
    const uint32_t db = 4100;
    fake_ring      ring(1, 4);  // N = 4, mask 3
    drain_state    st;
    recorder       rec0;
    ring.wptr[0] = 3;  // prime to slot 3 (= N-1)
    run_drain(ring, st, rec0);

    ring.put(0, 3, kRecStart, 8, db, 111);  // physical slot 3 (N-1)
    ring.put(0, 0, kRecEop, 8, db, 222);    // index 4 & 3 = physical slot 0
    ring.wptr[0] = 1;                       // = (3+2) & 3, unread = (1-3)&3 = 2
    recorder rec;
    EXPECT_EQ(run_drain(ring, st, rec), 1u);
    ASSERT_EQ(rec.pairs.count(std::make_pair(db, 8u)), 1u);
    EXPECT_EQ(st.cursors.rptr[0], 1u);
}

// Only a plain decimal in [1, kDlogMaxRingKb] (KiB) is accepted; else 0.
TEST(dlog_ring_size, env_value_parsing)
{
    constexpr uint64_t kb = 1024;
    struct row
    {
        const char* label;
        std::string in;
        uint64_t    expect;
    };
    const row rows[] = {
        {"empty", "", 0},
        {"zero", "0", 0},
        {"double_zero", "00", 0},
        {"neg1", "-1", 0},
        {"neg80", "-80", 0},
        {"alpha", "abc", 0},
        {"leading_ws", " 80", 0},
        {"trailing_ws", "80 ", 0},
        {"trailing_junk", "80K", 0},
        {"plus", "+80", 0},
        {"hex", "0x80", 0},
        {"min_1kb", "1", 1u * kb},
        {"floor_80kb", "80", 80u * kb},
        {"floor_matches_const", "80", kDlogMinRingBytes},
        {"1024kb", "1024", 1024u * kb},
        {"max_u32_field", "4194303", kDlogMaxRingKb * kb},
        {"over_u32_field", "4194304", 0},
        {"u32_max_plus", "4294967296", 0},
        {"u64_max", "18446744073709551615", 0},
        {"u64_max_plus_1", "18446744073709551616", 0},
        {"64_nines", std::string(64, '9'), 0},
    };
    for(const auto& tc : rows)
        EXPECT_EQ(dlog_ring_bytes_from_kb_str(tc.in), tc.expect) << tc.label;
}

// snap lands any request on the 80*2^k lattice: legal for both 2- and 4-region ASICs.
TEST(dlog_ring_size, snap_yields_a_driver_legal_size)
{
    for(uint64_t want : {uint64_t{0},
                         uint64_t{1},
                         kDlogMinRingBytes - 1,
                         kDlogMinRingBytes,
                         uint64_t{100000},
                         uint64_t{131072},
                         uint64_t{1048576},
                         uint64_t{5242880},
                         kDlogMaxRingBytes - 1,
                         kDlogMaxRingBytes,
                         kDlogMaxRingBytes + 1,
                         uint64_t{0xFFFFFFFF}})
    {
        uint64_t sz = dlog_snap_ring_bytes(want);
        EXPECT_GE(sz, kDlogMinRingBytes);
        EXPECT_LE(sz, kDlogMaxRingBytes);
        EXPECT_LE(sz, 0xFFFFFFFFull);  // uint32 buffer_size field
        if(want >= kDlogMinRingBytes)
        {
            EXPECT_LE(sz, want);  // never rounds up
        }
        EXPECT_EQ(sz % 80u, 0u);
        for(uint64_t num_regions : {uint64_t{2}, uint64_t{4}})
        {
            ASSERT_EQ(sz % (num_regions * 20), 0u);
            uint64_t rrc = sz / (num_regions * 20);
            EXPECT_EQ(rrc & (rrc - 1), 0u);  // power of two
            EXPECT_LE(rrc, 1u << 24);
            EXPECT_GT(rrc, 0u);
        }
    }
}

// Specific snap cases: on-lattice unchanged, off-lattice snaps down, clamp at bounds.
TEST(dlog_ring_size, snap_boundaries)
{
    struct row
    {
        const char* label;
        uint64_t    want;
        uint64_t    expect;
    };
    const row rows[] = {
        {"floor_snaps_to_self", kDlogMinRingBytes, 81920u},
        {"128kb_snaps_down_to_80kb", 131072u, 81920u},
        {"zero_clamps_up_to_floor", 0u, kDlogMinRingBytes},
        {"one_clamps_up_to_floor", 1u, kDlogMinRingBytes},
        {"640kb_on_lattice", 655360u, 655360u},
        {"2p5mb_on_lattice", 2621440u, 2621440u},
        {"default_is_the_10mb_ceiling", kDlogDefaultRingBytes, kDlogMaxRingBytes},
        {"5mb_on_lattice", 5242880u, 5242880u},
        {"10mb_ceiling_on_lattice", kDlogMaxRingBytes, 10485760u},
        {"above_ceiling_clamps", 41943040u, kDlogMaxRingBytes},
        {"max_kb_clamps", kDlogMaxRingKb * 1024, kDlogMaxRingBytes},
        {"just_below_ceiling_snaps_down", kDlogMaxRingBytes - 1, 5242880u},
    };
    for(const auto& tc : rows)
        EXPECT_EQ(dlog_snap_ring_bytes(tc.want), tc.expect) << tc.label;
}

// ABI v4 has no reader-side loss detection, so every copied EOP is trusted as a
// completion: a matched pair forms and a START-less EOP is forwarded start-unknown.
TEST(dlog_drain, every_eop_trusted)
{
    const uint32_t db = 4100;
    env            e(1, 8);
    e.ring.put(0, 0, kRecStart, 1, db, 10);
    e.ring.put(0, 1, kRecEop, 1, db, 20);
    e.ring.put(0, 2, kRecEop, 2, db, 30);  // shape ii: no start
    e.ring.wptr[0] = 3;
    EXPECT_EQ(e.drain(), 1u);
    EXPECT_EQ(e.rec.records, 2u);
    EXPECT_EQ(e.rec.pairs.size(), 1u);
    EXPECT_EQ(e.rec.eops_without_start.size(), 1u);
}

// Reverse-region: an EOP is copied BEFORE its own START when the START lands in a
// LATER region of the same batch (copy_pipes sweeps region 0 then region 1).
// pair_records binds every START in the batch before matching any EOP, so the pair
// still forms rather than the EOP arriving start-unknown.
TEST(dlog_drain, reverse_region_eop_before_start_in_one_batch)
{
    env            e(2, 8);
    const uint32_t db = 4100;
    e.ring.put(0, 0, kRecEop, 1, db, 200);    // region 0: copied first
    e.ring.put(1, 0, kRecStart, 1, db, 100);  // region 1: its START, copied second
    e.ring.wptr[0] = 1;
    e.ring.wptr[1] = 1;
    EXPECT_EQ(e.drain(), 1u) << "the EOP pairs with a START copied after it";
    ASSERT_EQ(e.rec.pairs.count(std::make_pair(db, 1u)), 1u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 1u)].first, 100u) << "start from region 1";
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 1u)].second, 200u) << "end from region 0";
    EXPECT_TRUE(e.rec.eops_without_start.empty()) << "not forwarded start-unknown";
}

// The copier advances rptr for every region copied, freeing space before pairing runs.
TEST(dlog_drain, copier_advances_rptr_without_pairing)
{
    gpu_ring g(2048, 2);
    g.ring.put(0, 0, kRecStart, 1, 4100, 10);
    g.ring.put(0, 1, kRecEop, 1, 4100, 20);
    g.ring.wptr[0] = 2;
    g.ring.put(1, 0, kRecEop, 9, 4200, 30);
    g.ring.wptr[1] = 1;
    EXPECT_EQ(g.copy(), 3u);
    EXPECT_EQ(g.cur.rptr[0], 2u);  // freed before anything was paired
    EXPECT_EQ(g.cur.rptr[1], 1u);
    EXPECT_EQ(g.batch.size(), 3u);
    pair_state pairing;
    recorder   rec;
    EXPECT_EQ(pair_records(g.batch.data(), g.batch.size(), pairing, 1000, rec.on_record()), 1u);
    EXPECT_EQ(rec.pairs.size(), 1u);
    EXPECT_EQ(rec.eops_without_start.size(), 1u);
}

// The pairing census counts starts/eops/unmatched/overwrites.
TEST(dlog_drain, pairing_census_counts_starts_eops_and_overwrites)
{
    const uint32_t db = 4100;
    env            e(1, 2048);
    e.ring.put(0, 0, kRecStart, 1, db, 10);
    e.ring.put(0, 1, kRecEop, 1, db, 20);
    e.ring.put(0, 2, kRecStart, 2, db, 30);
    e.ring.put(0, 3, kRecEop, 2, db, 40);
    e.ring.put(0, 4, kRecEop, 9, db, 50);    // orphan: no START ever
    e.ring.put(0, 5, kRecStart, 7, db, 60);  // retained
    e.ring.put(0, 6, kRecStart, 7, db, 70);  // overwrites the live key
    e.ring.wptr[0] = 7;
    EXPECT_EQ(e.drain(), 2u);
    EXPECT_EQ(e.st.pairing.starts_seen, 4u);
    EXPECT_EQ(e.st.pairing.eops_seen, 3u);
    EXPECT_EQ(e.st.pairing.unmatched_eops, 1u);
    EXPECT_EQ(e.st.pairing.starts_overwritten, 1u);
    EXPECT_EQ(e.st.pairing.pending_starts.size(), 1u);  // id 7 still waiting
}

// A dispatch replayed after a queue switch logs START again with no EOP between;
// its EOP pairs with the NEWEST START, the one that reached the shader engines.
TEST(dlog_drain, same_key_replayed_start_pairs_with_newest)
{
    const uint32_t db = 4100;
    env            e(1, 2048);
    e.ring.put(0, 0, kRecStart, 7, db, 100);
    e.ring.put(0, 1, kRecStart, 7, db, 250);  // replay of the same dispatch
    e.ring.put(0, 2, kRecEop, 7, db, 300);
    e.ring.wptr[0] = 3;
    EXPECT_EQ(e.drain(), 1u);
    ASSERT_EQ(e.rec.pairs.count(std::make_pair(db, 7u)), 1u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].first, 250u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].second, 300u);
    EXPECT_TRUE(e.rec.eops_without_start.empty());
    EXPECT_EQ(e.st.pairing.starts_overwritten, 1u);
    EXPECT_TRUE(e.st.pairing.pending_starts.empty());
}

// Only one START is retained per key, so a second EOP on the key finds no START
// and completes start-unknown rather than pairing with a consumed START.
TEST(dlog_drain, same_key_second_eop_is_start_unknown)
{
    const uint32_t db = 4100;
    env            e(1, 2048);
    e.ring.put(0, 0, kRecStart, 7, db, 100);
    e.ring.put(0, 1, kRecStart, 7, db, 250);
    e.ring.put(0, 2, kRecEop, 7, db, 300);
    e.ring.put(0, 3, kRecEop, 7, db, 400);
    e.ring.wptr[0] = 4;
    EXPECT_EQ(e.drain(), 1u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].first, 250u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].second, 300u);
    EXPECT_EQ(e.rec.eops_without_start.size(), 1u);
    EXPECT_EQ(e.st.pairing.unmatched_eops, 1u);
    EXPECT_TRUE(e.st.pairing.pending_starts.empty());
}

// A dispatch can be replayed more than once before it runs; the last START wins.
TEST(dlog_drain, same_key_repeated_replays_pair_with_last)
{
    const uint32_t db = 4100;
    env            e(1, 2048);
    e.ring.put(0, 0, kRecStart, 7, db, 100);
    e.ring.put(0, 1, kRecStart, 7, db, 250);
    e.ring.put(0, 2, kRecStart, 7, db, 500);
    e.ring.put(0, 3, kRecEop, 7, db, 600);
    e.ring.wptr[0] = 4;
    EXPECT_EQ(e.drain(), 1u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].first, 500u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].second, 600u);
    EXPECT_EQ(e.st.pairing.starts_overwritten, 2u);
    EXPECT_TRUE(e.st.pairing.pending_starts.empty());
}

// A START retained from an earlier batch is replaced by the replay's START in a
// later batch, and the EOP pairs with the replay.
TEST(dlog_drain, retained_start_replaced_across_batches)
{
    const uint32_t db = 4100;
    env            e(1, 2048);
    e.ring.put(0, 0, kRecStart, 7, db, 100);
    e.ring.wptr[0] = 1;
    EXPECT_EQ(e.drain(), 0u);
    ASSERT_EQ(e.st.pairing.pending_starts.size(), 1u);
    e.rec = recorder{};
    e.ring.put(0, 1, kRecStart, 7, db, 250);
    e.ring.put(0, 2, kRecEop, 7, db, 300);
    e.ring.wptr[0] = 3;
    EXPECT_EQ(e.drain(), 1u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].first, 250u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].second, 300u);
    EXPECT_EQ(e.st.pairing.starts_overwritten, 1u);
    EXPECT_TRUE(e.st.pairing.pending_starts.empty());
}

TEST(dlog_drain, older_cross_batch_start_does_not_replace_newest)
{
    constexpr uint32_t doorbell    = 4100;
    auto               environment = env{2, 2048};
    environment.ring.put(1, 0, kRecStart, 7, doorbell, 200);
    environment.ring.wptr[1] = 1;
    EXPECT_EQ(environment.drain(), 0u);
    environment.ring.put(0, 0, kRecStart, 7, doorbell, 100);
    environment.ring.put(1, 1, kRecEop, 7, doorbell, 300);
    environment.ring.wptr[0] = 1;
    environment.ring.wptr[1] = 2;
    EXPECT_EQ(environment.drain(), 1u);
    EXPECT_EQ(environment.rec.pairs[std::make_pair(doorbell, 7u)].first, 200u);
    EXPECT_EQ(environment.rec.pairs[std::make_pair(doorbell, 7u)].second, 300u);
    EXPECT_EQ(environment.st.pairing.starts_out_of_order, 1u);
    EXPECT_TRUE(environment.st.pairing.pending_starts.empty());
}

TEST(dlog_drain, older_or_duplicate_start_does_not_refresh_age)
{
    auto state = pair_state{};
    auto sink  = recorder{};
    auto start = fw_record{200, 0, kRecStart, 7, 4100};
    pair_records(&start, 1, state, 1000, sink.on_record());
    pair_records(&start, 1, state, 2000, sink.on_record());
    start.ts_lo = 100;
    pair_records(&start, 1, state, 3000, sink.on_record());
    ASSERT_EQ(state.pending_starts.size(), 1u);
    EXPECT_EQ(state.pending_starts.begin()->second.start_ticks, 200u);
    EXPECT_EQ(state.pending_starts.begin()->second.seen_at_ns, 1000u);
    EXPECT_EQ(state.starts_out_of_order, 2u);
    EXPECT_EQ(state.evict_stale(4000, 2000), 1u);
}

// A replayed START refreshes the key's age along with its ticks: the key ages
// from the NEWEST START.
TEST(dlog_drain, replayed_start_ages_from_newest)
{
    pair_state st;
    auto       make_start = [](uint32_t db, uint32_t id, uint64_t ts) {
        fw_record r{};
        r.ts_lo        = static_cast<uint32_t>(ts & 0xFFFFFFFFu);
        r.ts_hi        = static_cast<uint32_t>(ts >> 32);
        r.record_type  = kRecStart;
        r.dispatch_id  = id;
        r.doorbell_off = db;
        return r;
    };
    auto           nop = [](const drained_record&) {};
    const uint32_t db  = 4100;
    // first START at now=0; replays at 1000 and 2000 move seen_at_ns to 2000.
    std::vector<fw_record> b0{make_start(db, 7, 100)};
    pair_records(b0.data(), b0.size(), st, /*now_ns=*/0, nop);
    std::vector<fw_record> b1{make_start(db, 7, 200)};
    pair_records(b1.data(), b1.size(), st, /*now_ns=*/1000, nop);
    std::vector<fw_record> b2{make_start(db, 7, 300)};
    pair_records(b2.data(), b2.size(), st, /*now_ns=*/2000, nop);
    ASSERT_EQ(st.pending_starts.size(), 1u);
    EXPECT_EQ(st.pending_starts.begin()->second.start_ticks, 300u);
    EXPECT_EQ(st.pending_starts.begin()->second.seen_at_ns, 2000u);
    EXPECT_EQ(st.evict_stale(/*now_ns=*/2500, /*max_age_ns=*/2000), 0u) << "not stale yet";
    EXPECT_EQ(st.evict_stale(/*now_ns=*/4500, /*max_age_ns=*/2000), 1u) << "one START per key";
    EXPECT_TRUE(st.pending_starts.empty());
}

// Per-GPU isolation: two rings sharing a doorbell+id each keep their own state,
// cursors, and timestamps.
TEST(dlog_drain, per_gpu_isolation)
{
    const uint32_t db = 4100;
    gpu_ring       a(2048), b(2048);
    a.ring.put(0, 0, kRecStart, 5, db, 100);
    a.ring.put(0, 1, kRecEop, 5, db, 200);
    a.ring.wptr[0] = 2;
    b.ring.put(0, 0, kRecStart, 5, db, 300);
    b.ring.put(0, 1, kRecEop, 5, db, 400);
    b.ring.wptr[0] = 2;
    a.copy();
    b.copy();
    pair_state pair_a;
    pair_state pair_b;
    recorder   rec_a;
    recorder   rec_b;
    EXPECT_EQ(pair_records(a.batch.data(), a.batch.size(), pair_a, 1000, rec_a.on_record()), 1u);
    EXPECT_EQ(pair_records(b.batch.data(), b.batch.size(), pair_b, 1000, rec_b.on_record()), 1u);
    ASSERT_EQ(rec_a.pairs.count(std::make_pair(db, 5u)), 1u);
    ASSERT_EQ(rec_b.pairs.count(std::make_pair(db, 5u)), 1u);
    EXPECT_EQ(rec_a.pairs[std::make_pair(db, 5u)].first, 100u);
    EXPECT_EQ(rec_b.pairs[std::make_pair(db, 5u)].first, 300u);
    EXPECT_EQ(pair_a.unmatched_eops, 0u);
    EXPECT_EQ(pair_b.unmatched_eops, 0u);
    EXPECT_EQ(a.cur.rptr[0], 2u);
    EXPECT_EQ(b.cur.rptr[0], 2u);
}

TEST(dlog_drain, snapshot_lookahead_pairs_eop_with_late_copied_start)
{
    auto       state      = pair_state{};
    auto       sink       = recorder{};
    const auto completion = fw_record{200, 0, kRecEop, 7, 4100};
    const auto start      = fw_record{100, 0, kRecStart, 7, 4100};

    EXPECT_EQ(pair_records(&completion, 1, state, 1000, sink.on_record(), true, 1), 0u);
    EXPECT_EQ(sink.records, 0u);
    EXPECT_EQ(state.deferred_epoch, 1u);
    EXPECT_EQ(pair_records(&start, 1, state, 2000, sink.on_record(), true, 2), 0u);
    EXPECT_EQ(pair_records(nullptr, 0, state, 3000, sink.on_record(), true, 3), 1u);
    EXPECT_EQ(sink.pairs[std::make_pair(4100u, 7u)].first, 100u);
    EXPECT_EQ(sink.pairs[std::make_pair(4100u, 7u)].second, 200u);
    EXPECT_TRUE(sink.eops_without_start.empty());
    EXPECT_TRUE(state.pending_starts.empty());
    EXPECT_TRUE(state.deferred_records.empty());
    EXPECT_EQ(state.deferred_epoch, UINT64_MAX);
}

TEST(dlog_drain, snapshot_lookahead_recovers_latest_replay_before_completion)
{
    auto       state = pair_state{};
    auto       sink  = recorder{};
    const auto first =
        std::vector<fw_record>{{100, 0, kRecStart, 7, 4100}, {300, 0, kRecEop, 7, 4100}};
    const auto replay = fw_record{200, 0, kRecStart, 7, 4100};
    pair_records(first.data(), first.size(), state, 1000, sink.on_record(), true, 1);
    EXPECT_EQ(sink.records, 0u);
    EXPECT_EQ(pair_records(&replay, 1, state, 2000, sink.on_record(), true, 2), 0u);
    EXPECT_EQ(pair_records(nullptr, 0, state, 3000, sink.on_record(), true, 3), 1u);
    EXPECT_EQ(sink.pairs[std::make_pair(4100u, 7u)].first, 200u);
    EXPECT_EQ(state.starts_seen, 2u);
    EXPECT_EQ(state.eops_seen, 1u);
    EXPECT_TRUE(state.pending_starts.empty());
}

TEST(dlog_drain, snapshot_lookahead_preserves_reused_keys_and_equal_tick_order)
{
    auto       state   = pair_state{};
    auto       emitted = std::vector<drained_record>{};
    auto       sink    = [&](const drained_record& record) { emitted.push_back(record); };
    const auto first   = std::vector<fw_record>{
        {200, 0, kRecEop, 7, 4100}, {200, 0, kRecStart, 7, 4100}, {400, 0, kRecEop, 7, 4100}};
    const auto late_start = fw_record{100, 0, kRecStart, 7, 4100};
    pair_records(first.data(), first.size(), state, 1000, sink, true, 1);
    EXPECT_TRUE(emitted.empty());
    EXPECT_EQ(pair_records(&late_start, 1, state, 2000, sink, true, 2), 0u);
    EXPECT_EQ(pair_records(nullptr, 0, state, 3000, sink, true, 3), 2u);
    ASSERT_EQ(emitted.size(), 2u);
    EXPECT_EQ(emitted[0].start_ticks, 100u);
    EXPECT_EQ(emitted[0].end_ticks, 200u);
    EXPECT_EQ(emitted[1].start_ticks, 200u);
    EXPECT_EQ(emitted[1].end_ticks, 400u);
    EXPECT_TRUE(state.pending_starts.empty());
    EXPECT_TRUE(state.deferred_records.empty());
}

TEST(dlog_drain, snapshot_lookahead_idle_marker_flushes_unknown_start_once)
{
    auto       state      = pair_state{};
    auto       sink       = recorder{};
    const auto completion = fw_record{200, 0, kRecEop, 7, 4100};
    pair_records(&completion, 1, state, 1000, sink.on_record(), true, 1);
    EXPECT_EQ(sink.records, 0u);
    pair_records(nullptr, 0, state, 2000, sink.on_record(), true, 2);
    EXPECT_EQ(sink.records, 1u);
    EXPECT_EQ(state.unmatched_eops, 1u);
    EXPECT_EQ(state.eops_seen, 1u);
    EXPECT_EQ(state.deferred_epoch, UINT64_MAX);
    pair_records(nullptr, 0, state, 3000, sink.on_record(), true, 3);
    EXPECT_EQ(sink.records, 1u);
}

TEST(dlog_drain, snapshot_lookahead_handles_producer_progress_during_copy)
{
    constexpr uint32_t dispatch_count = 4096;
    constexpr uint32_t doorbell       = 4100;
    for(uint32_t regions : {2u, 4u})
    {
        auto ring          = fake_ring{regions, 65536};
        auto cursors       = ring_cursors{};
        cursors.rptr_init  = true;
        auto     pairing   = pair_state{};
        auto     writes    = std::vector<uint64_t>(regions, 0);
        auto     emitted   = std::vector<drained_record>{};
        uint32_t published = 0;
        uint64_t copied    = 0;
        ring.put(0, writes[0]++, kRecPadding, 0, 0, 0);
        ring.wptr[0] = writes[0];

        auto publish = [&]() {
            for(uint32_t step = 0; step < 3 && published < dispatch_count * 3; ++step)
            {
                const uint32_t dispatch = published / 3;
                const uint32_t event    = published % 3;
                const uint32_t region   = (dispatch + event) % regions;
                const uint32_t kind     = event == 2 ? kRecEop : kRecStart;
                const uint64_t ticks    = uint64_t{dispatch} * 10 + (event == 2 ? 9 : event + 1);
                ring.put(region, writes[region]++, kind, dispatch % 17 + 1, doorbell, ticks);
                ring.wptr[region] = writes[region];
                ++published;
            }
        };

        // The reader stage through a stream-sink adapter that fires the
        // producer-progress hook after every append, restoring the interleaving
        // the vector-sink version of this test exercised: copy_pipes reads only
        // the wptr snapshot, so records written DURING the copy are left for
        // the next pass.
        drain_stream stream;
        ASSERT_TRUE(stream.preallocate(4));
        reader_alloc_log       log;
        std::vector<fw_record> batch;
        std::function<void()>  publish_hook = publish;
        progress_stream        sink{&stream, &publish_hook};
        for(uint64_t epoch = 1; epoch < dispatch_count * 3; ++epoch)
        {
            batch.clear();
            const auto count = copy_pipes(
                ring.records.data(), regions, ring.rrc, ring.wptr.data(), cursors, sink, log);
            stream.publish();
            stream.copy_out(sink_vector{&batch}, 4);
            copied += count;
            pair_records(
                batch.data(),
                batch.size(),
                pairing,
                epoch * 1000,
                [&](const drained_record& record) { emitted.push_back(record); },
                true,
                epoch);
            if(count == 0 && published == dispatch_count * 3) break;
        }

        EXPECT_EQ(copied, dispatch_count * 3 + 1);
        ASSERT_EQ(emitted.size(), dispatch_count) << "regions=" << regions;
        auto completions = std::set<uint64_t>{};
        for(const auto& record : emitted)
        {
            EXPECT_TRUE(record.start_known);
            EXPECT_EQ(record.start_ticks, record.end_ticks - 7);
            EXPECT_EQ(record.dispatch_id, (record.end_ticks / 10) % 17 + 1);
            EXPECT_TRUE(completions.insert(record.end_ticks).second);
        }
        EXPECT_EQ(pairing.starts_seen, dispatch_count * 2);
        EXPECT_EQ(pairing.eops_seen, dispatch_count);
        EXPECT_TRUE(pairing.pending_starts.empty());
        EXPECT_TRUE(pairing.start_ages.empty());
        EXPECT_TRUE(pairing.deferred_records.empty());
        EXPECT_EQ(pairing.unmatched_eops, 0u);
    }
}

// ---------------------------------------------------------------------------
// The reader pass (dlog_pass) and the fence over streams.
// ---------------------------------------------------------------------------

// A minimal session for dlog_pass: the drain-facing fields of dlog_session.
// Default-constructible (not movable), so it can live in the fixed arrays the
// processor-side helpers iterate.
struct pass_session
{
    struct info_t
    {
        uint32_t num_regions         = 1;
        uint32_t region_record_count = 64;
        uint64_t buffer_size         = uint64_t{1} * 64 * kFwRecBytes;
    };
    info_t           info                  = {};
    ring_cursors     cursors               = {};
    drain_stream     stream                = {};
    reader_alloc_log alloc_log             = {};
    bool             pairing_flush_pending = false;

    void setup(uint32_t nreg, uint32_t rrc, size_t spares = 4)
    {
        info.num_regions         = nreg;
        info.region_record_count = rrc;
        info.buffer_size         = uint64_t{nreg} * rrc * kFwRecBytes;
        EXPECT_TRUE(stream.preallocate(spares));
    }
    // One pass over the ring, exactly as copy_records drives it.
    uint64_t pass(const fake_ring& ring, uint64_t drain_epoch, uint64_t now_ns)
    {
        return dlog_pass(ring.records.data(), ring.wptr.data(), *this, drain_epoch, now_ns);
    }
};

// One pass = records, then one marker, then one publish: published() moves once,
// ends at the marker, and the copy-out sees a closed segment (the 6.1 assert).
TEST(dlog_pass, one_pass_publishes_records_then_marker)
{
    test_mapping::reset();
    fake_ring                   ring(1, 64);
    std::array<pass_session, 1> ss{};
    ss[0].setup(1, 64);
    auto&                         s = ss[0];
    std::array<stream_backlog, 1> backlogs{};
    // Prime pass: rptr := wptr, nothing copied, nothing pending, no marker.
    EXPECT_EQ(s.pass(ring, 0, 1000), 0u);
    EXPECT_FALSE(s.pairing_flush_pending);
    const uint64_t c0 = s.stream.published();

    ring.put(0, 0, kRecStart, 7, 4100, 10);
    ring.put(0, 1, kRecEop, 7, 4100, 20);
    ring.wptr[0] = 2;
    EXPECT_EQ(s.pass(ring, 4, 2000), 2u);
    EXPECT_TRUE(s.pairing_flush_pending);
    EXPECT_EQ(s.stream.published(), c0 + 3) << "one publish: the records plus the marker";
    EXPECT_EQ(s.cursors.rptr[0], 2u) << "rptr == the wptr snapshot";
    // The published prefix ends at the marker: the production copy-out closes the
    // segment; the 6.1 assert inside would abort on a mutant that published
    // before the marker.
    copy_out_all(ss, backlogs, 1);
    ASSERT_EQ(backlogs[0].closed.size(), 1u);
    EXPECT_EQ(backlogs[0].closed[0].records.size(), 2u);
    EXPECT_EQ(backlogs[0].closed[0].epoch, 5u);
    EXPECT_EQ(backlogs[0].closed[0].seen_at_ns, 2000u);
    EXPECT_EQ(backlogs[0].closed[0].end_pos, c0 + 3);
    EXPECT_TRUE(backlogs[0].open.records.empty());
}

// 5.2 over consecutive passes: data -> DATA; then idle -> exactly one
// IDLE_FLUSH; then idle -> nothing, no publish; pairing_flush_pending follows
// the base :684-685 rules.
TEST(dlog_pass, marker_table_over_consecutive_passes)
{
    test_mapping::reset();
    fake_ring                   ring(1, 64);
    std::array<pass_session, 1> ss{};
    ss[0].setup(1, 64);
    auto&                         s = ss[0];
    std::array<stream_backlog, 1> backlogs{};

    EXPECT_EQ(s.pass(ring, 0, 1000), 0u);  // prime
    EXPECT_FALSE(s.pairing_flush_pending);

    // Copied > 0: a DATA marker (a segment with records) closes the pass.
    ring.put(0, 0, kRecEop, 7, 4100, 10);
    ring.wptr[0] = 1;
    EXPECT_EQ(s.pass(ring, 1, 2000), 1u);
    EXPECT_TRUE(s.pairing_flush_pending);
    copy_out_all(ss, backlogs, 1);
    ASSERT_EQ(backlogs[0].closed.size(), 1u);
    EXPECT_EQ(backlogs[0].closed[0].records.size(), 1u);

    // Copied == 0 with a pending flush: exactly one IDLE_FLUSH (empty segment).
    EXPECT_EQ(s.pass(ring, 2, 3000), 0u);
    EXPECT_FALSE(s.pairing_flush_pending);
    copy_out_all(ss, backlogs, 1);
    ASSERT_EQ(backlogs[0].closed.size(), 2u);
    EXPECT_EQ(backlogs[0].closed[1].records.size(), 0u) << "the idle flush closes an empty segment";
    EXPECT_EQ(backlogs[0].closed[1].seen_at_ns, 3000u);

    // Copied == 0, nothing pending: no marker, no publish at all.
    const uint64_t published_before = s.stream.published();
    EXPECT_EQ(s.pass(ring, 3, 4000), 0u);
    EXPECT_FALSE(s.pairing_flush_pending);
    EXPECT_EQ(s.stream.published(), published_before)
        << "an idle pass with nothing pending publishes nothing";
    copy_out_all(ss, backlogs, 1);
    EXPECT_EQ(backlogs[0].closed.size(), 2u) << "no empty segment for the idle pass";
}

// Marker contents: epoch == drain_epoch + 1 sampled at pass start, seen_at_ns
// is the pass-start clock (sampled before the copy, base :671-672).
TEST(dlog_pass, marker_carries_epoch_and_pass_start_clock)
{
    test_mapping::reset();
    fake_ring    ring(1, 64);
    pass_session s;
    s.setup(1, 64);
    s.pass(ring, 0, 1000);  // prime

    ring.put(0, 0, kRecStart, 7, 4100, 10);
    ring.wptr[0] = 1;
    EXPECT_EQ(s.pass(ring, 41, 2000), 1u);
    std::vector<fw_record> out;
    s.stream.copy_out(sink_vector{&out}, 4);
    ASSERT_EQ(out.size(), 2u);
    const auto m = decode_marker(out.back());
    EXPECT_EQ(m.epoch, 42u) << "drain_epoch + 1, sampled at pass start";
    EXPECT_EQ(m.seen_at_ns, 2000u) << "the pass-start clock, sampled before the copy";
}

// 5.3: with pass_reserve spares, a maximal pass plus its marker from any fill
// makes zero map calls (2.5 MiB geometry and the 10 MiB R = 1 variant).
TEST(dlog_pass, maximal_pass_from_any_fill_needs_no_allocation)
{
    constexpr size_t                    kK      = drain_stream::kK;
    const std::pair<uint64_t, uint32_t> geoms[] = {{2621440, 4}, {10485760, 1}};
    for(const auto& [ring_bytes, regions] : geoms)
    {
        const uint32_t rrc     = static_cast<uint32_t>(ring_bytes / (uint64_t{20} * regions));
        const size_t   reserve = dlog_pass_reserve(ring_bytes, regions, kK);
        ASSERT_GT(reserve, 0u);
        SCOPED_TRACE(::testing::Message() << "ring=" << ring_bytes << " regions=" << regions);
        for(const size_t fill : {size_t{0}, size_t{1}, kK - 1})
        {
            SCOPED_TRACE(::testing::Message() << "fill=" << fill);
            test_mapping::reset();
            fake_ring    ring(regions, rrc);
            pass_session s;
            s.setup(regions, rrc, reserve);
            const size_t maps_before = test_mapping::map_calls.load();
            // Attached at ring start: rptr primed to 0, so the whole ring below
            // wptr is unread.
            s.cursors               = {};
            s.cursors.rptr_init     = true;
            s.pairing_flush_pending = false;

            // Offset the stream's fill: the pass starts mid-chunk.
            std::vector<fw_record> pad(fill);
            for(size_t i = 0; i < fill; ++i)
                pad[i] = fw_record{static_cast<uint32_t>(i), 0, kRecPadding, 0, 0};
            s.stream.append(pad.data(), pad.size(), s.alloc_log);
            s.stream.publish();

            // Maximal ring content: N-1 unread records per region.
            for(uint32_t r = 0; r < regions; ++r)
                for(uint32_t i = 0; i + 1 < rrc; ++i)
                    ring.put(r, i, kRecStart, i, 4100 + r, i);
            for(uint32_t r = 0; r < regions; ++r)
                ring.wptr[r] = rrc - 1;

            EXPECT_EQ(s.pass(ring, 0, 1000), uint64_t{regions} * (rrc - 1));
            EXPECT_EQ(s.alloc_log.reader_allocs.load(), 0u)
                << "the spares must cover a maximal pass";
            EXPECT_EQ(test_mapping::map_calls.load(), maps_before)
                << "an unexpected reader map: pass_reserve is too small";
        }
    }
}

// 7.3: processor_drained_through over streams, ported from the pipe fence tests.
// The predicate iterates sessions, so the streams sit in minimal wrappers.
struct fence_session
{
    drain_stream stream = {};
};

TEST(dlog_fence, drained_through_over_streams)
{
    test_mapping::reset();
    std::array<fence_session, 2> ss{};
    ASSERT_TRUE(ss[0].stream.preallocate(2));
    ASSERT_TRUE(ss[1].stream.preallocate(2));
    auto                         deferred_epoch = std::atomic<uint64_t>{UINT64_MAX};
    reader_alloc_log             log;
    const std::vector<fw_record> recs{fw_record{1, 0, kRecEop, 7, 4100}};

    // Capture a target; the processor is behind on processed.
    ss[0].stream.append(recs.data(), recs.size(), log);
    ss[0].stream.publish();
    std::array<uint64_t, 2> target{};
    target[0] = ss[0].stream.published();
    EXPECT_FALSE(processor_drained_through(ss, 2, target.data(), deferred_epoch, 10));
    ss[0].stream.copy_out([](const fw_record*, size_t, uint64_t) {}, 2);
    ss[0].stream.publish_processed(ss[0].stream.taken());
    // Taken and processed, but the held epoch sits at the boundary: an epoch
    // <= through still holds lookahead records, an epoch > through does not.
    deferred_epoch.store(10, std::memory_order_release);
    EXPECT_FALSE(processor_drained_through(ss, 2, target.data(), deferred_epoch, 10));
    EXPECT_TRUE(processor_drained_through(ss, 2, target.data(), deferred_epoch, 9));
    deferred_epoch.store(UINT64_MAX, std::memory_order_release);
    EXPECT_TRUE(processor_drained_through(ss, 2, target.data(), deferred_epoch, 10));

    // An uncaptured new stream is ignored: count stays at the capture size.
    ss[1].stream.append(recs.data(), recs.size(), log);
    ss[1].stream.publish();
    EXPECT_TRUE(processor_drained_through(ss, 1, target.data(), deferred_epoch, 10));
}

// A fence samples the target once; records published after that sample cannot
// extend it, even though the stream never reads empty.
TEST(dlog_fence, target_is_not_extended_by_newer_publishes)
{
    test_mapping::reset();
    std::array<fence_session, 1> ss{};
    ASSERT_TRUE(ss[0].stream.preallocate(2));
    auto                         held = std::atomic<uint64_t>{UINT64_MAX};
    reader_alloc_log             log;
    const std::vector<fw_record> recs{fw_record{1, 0, kRecEop, 7, 4100}};
    ss[0].stream.append(recs.data(), recs.size(), log);
    ss[0].stream.publish();
    std::array<uint64_t, 1> target{};
    target[0] = ss[0].stream.published();
    // Newer traffic, after the fence's sample.
    ss[0].stream.append(recs.data(), recs.size(), log);
    ss[0].stream.publish();
    ss[0].stream.copy_out([](const fw_record*, size_t, uint64_t) {}, 2);
    EXPECT_FALSE(processor_drained_through(ss, 1, target.data(), held, 5));
    ss[0].stream.publish_processed(target[0]);
    EXPECT_TRUE(processor_drained_through(ss, 1, target.data(), held, 5));
}

// The fence reads processed BEFORE the held epoch: a processor that stores the
// epoch before publishing processed must not be missed by an early epoch load.
// The scripted fake fires the epoch store between the two fence loads, so the
// correct order observes epoch 5 <= through and waits; a mutant that loads the
// epoch first sees the stale UINT64_MAX and passes the fence.
TEST(dlog_fence, processed_is_read_before_the_held_epoch)
{
    std::atomic<uint64_t> epoch{UINT64_MAX};
    std::atomic<uint64_t> processed{1};
    struct scripted_stream
    {
        std::atomic<uint64_t>* epoch_sink    = nullptr;
        std::atomic<uint64_t>* processed_src = nullptr;
        mutable bool           fired         = false;
        bool                   armed() const { return true; }
        uint64_t               processed() const
        {
            if(!fired)
            {
                fired = true;
                // The concurrent processor store lands here: between the fence's
                // two loads.
                epoch_sink->store(5, std::memory_order_release);
            }
            return processed_src->load(std::memory_order_acquire);
        }
    };
    struct scripted_session
    {
        scripted_stream stream;
    };
    std::array<scripted_session, 1> ss{};
    ss[0].stream.epoch_sink    = &epoch;
    ss[0].stream.processed_src = &processed;
    std::array<uint64_t, 1> target{1};
    EXPECT_FALSE(processor_drained_through(ss, 1, target.data(), epoch, 10));
    EXPECT_TRUE(ss[0].stream.fired) << "the store must have landed between the loads";
}
// Exact stream-geometry validation (validate_stream_geometry). Canonical layout:
//   records[buffer_size] | wptr[num_regions*8] | pad-to-page

namespace
{
constexpr uint64_t kPage = 4096;  // GFX12 default: 80 KiB, 2 regions, 4 KiB page
stream_geometry
canonical_geometry(uint64_t buffer_size, uint32_t num_regions)
{
    const uint64_t  ptr_bytes = static_cast<uint64_t>(num_regions) * 8;
    stream_geometry g;
    g.num_regions         = num_regions;
    g.region_record_count = static_cast<uint32_t>(buffer_size / (num_regions * kFwRecBytes));
    g.buffer_size         = buffer_size;
    g.records_offset      = 0;
    g.wptr_offset         = buffer_size;
    g.mmap_size           = round_up_to_page(g.wptr_offset + ptr_bytes, kPage);
    return g;
}
}  // namespace

TEST(stream_geometry, canonical_layouts_are_accepted)
{
    const uint64_t buf = kDlogMinRingBytes;  // 80 KiB, 2 regions
    auto           g   = canonical_geometry(buf, 2);
    auto           r   = validate_stream_geometry(g, buf, kPage);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.mmap_len, g.mmap_size);
    EXPECT_EQ(g.region_record_count, 2048u);
    EXPECT_EQ(r.mmap_len, round_up_to_page(buf + 2 * 8, kPage));  // single page span
    // 640 KiB, 4 regions is also legal.
    auto g4 = canonical_geometry(655360, 4);
    EXPECT_TRUE(validate_stream_geometry(g4, 655360, kPage).ok);
}

// Every rejected mutation of the canonical layout: one row per corruption.
TEST(stream_geometry, invalid_layouts_are_rejected)
{
    const uint64_t buf    = kDlogMinRingBytes;
    auto           mutate = [&](const char* label, auto fn, uint64_t req = kDlogMinRingBytes) {
        auto g = canonical_geometry(buf, 2);
        fn(g);
        EXPECT_FALSE(validate_stream_geometry(g, req, kPage).ok) << label;
    };
    mutate(
        "buffer_size_differs_from_request", [](stream_geometry&) {}, buf * 2);
    mutate("mis_routed_records_offset", [](stream_geometry& g) { g.records_offset = 64; });
    mutate("wptr_offset_off_lattice", [](stream_geometry& g) { g.wptr_offset = buf + 8; });
    mutate("mmap_too_small", [](stream_geometry& g) { g.mmap_size -= kPage; });
    mutate("mmap_too_big", [](stream_geometry& g) { g.mmap_size += kPage; });
    mutate("wrong_region_record_count", [](stream_geometry& g) { g.region_record_count = 1024; });
    mutate("zero_regions", [](stream_geometry& g) { g.num_regions = 0; });
    mutate("too_many_regions", [](stream_geometry& g) { g.num_regions = kMaxRegions + 1; });
    mutate("non_power_of_two_rrc", [](stream_geometry& g) { g.region_record_count = 2047; });
}

// The rejection reason is reported so the caller can log which check tripped.
TEST(stream_geometry, rejection_reason_is_reported)
{
    const uint64_t buf = kDlogMinRingBytes;
    auto           ok  = validate_stream_geometry(canonical_geometry(buf, 2), buf, kPage);
    EXPECT_TRUE(ok.ok);
    EXPECT_EQ(ok.reason, geometry_reason::ok);

    EXPECT_EQ(validate_stream_geometry(canonical_geometry(buf, 2), buf * 2, kPage).reason,
              geometry_reason::buffer_size_mismatch);

    auto bad_regions        = canonical_geometry(buf, 2);
    bad_regions.num_regions = 0;
    EXPECT_EQ(validate_stream_geometry(bad_regions, buf, kPage).reason,
              geometry_reason::bad_region_layout);

    auto bad_offset           = canonical_geometry(buf, 2);
    bad_offset.records_offset = 64;
    EXPECT_EQ(validate_stream_geometry(bad_offset, buf, kPage).reason,
              geometry_reason::layout_mismatch);
}

// ---------------------------------------------------------------------------
// Tick-ordered pairing. The pairer orders each batch by (tick, kind,
// arrival_seq) with EOP-before-START at equal ticks, so temporal order -- not
// region/copy position -- decides every pairing. These fail under the old
// all-STARTs-first two-pass pairer.
// ---------------------------------------------------------------------------

// In-order reuse of one raw key: {S_A, E_A, S_B, E_B} -> BOTH pair.
// The old two-pass installed S_A and S_B before any EOP, so S_B overwrote S_A;
// tick order runs S_A -> E_A(pair) -> S_B -> E_B(pair).
TEST(dlog_drain, d1_same_key_in_order_reuse_both_pair)
{
    const uint32_t db = 4100;
    env            e(1, 2048);
    e.ring.put(0, 0, kRecStart, 7, db, 100);
    e.ring.put(0, 1, kRecEop, 7, db, 200);
    e.ring.put(0, 2, kRecStart, 7, db, 300);  // same raw key, reused after E_A
    e.ring.put(0, 3, kRecEop, 7, db, 400);
    e.ring.wptr[0] = 4;
    EXPECT_EQ(e.drain(), 2u) << "both dispatches pair; key never holds 2 outstanding";
    EXPECT_EQ(e.rec.pairs.size(), 1u);  // same key -> the map holds the last pair
    EXPECT_EQ(e.st.pairing.starts_overwritten, 0u) << "in-order reuse is not a replay";
    EXPECT_EQ(e.st.pairing.equal_tick_drops, 0u);
    EXPECT_TRUE(e.st.pairing.pending_starts.empty());
}

// Reuse tie: S_A(1) retained, then {E_A(5), S_B(5), E_B(9)} in one batch.
// EOP-before-START at the equal tick 5 pairs E_A with the outstanding S_A first,
// then installs S_B cleanly; both pair, no overwrite.
TEST(dlog_drain, d1_reuse_equal_tick_both_pair_no_overwrite)
{
    const uint32_t db = 4100;
    env            e(1, 2048);
    e.ring.put(0, 0, kRecStart, 7, db, 1);
    e.ring.wptr[0] = 1;
    EXPECT_EQ(e.drain(), 0u);  // S_A retained
    ASSERT_EQ(e.st.pairing.pending_starts.size(), 1u);
    e.rec = recorder{};
    e.ring.put(0, 1, kRecEop, 7, db, 5);    // E_A at tick 5
    e.ring.put(0, 2, kRecStart, 7, db, 5);  // S_B at the same tick 5
    e.ring.put(0, 3, kRecEop, 7, db, 9);    // E_B at tick 9
    e.ring.wptr[0] = 4;
    EXPECT_EQ(e.drain(), 2u) << "E_A pairs S_A (EOP-first at equal tick), then S_B/E_B pair";
    EXPECT_EQ(e.st.pairing.starts_overwritten, 0u);
    EXPECT_EQ(e.st.pairing.equal_tick_drops, 0u);
    EXPECT_TRUE(e.st.pairing.pending_starts.empty());
}

// A START retained from batch N pairs its EOP in batch N+1, and the
// retained START is NOT counted as overwritten (it is not re-fed into N+1's sort).
TEST(dlog_drain, d1_cross_batch_pair_no_overwrite)
{
    const uint32_t db = 4100;
    env            e(1, 2048);
    e.ring.put(0, 0, kRecStart, 7, db, 100);
    e.ring.wptr[0] = 1;
    EXPECT_EQ(e.drain(), 0u);
    ASSERT_EQ(e.st.pairing.pending_starts.size(), 1u);
    e.rec = recorder{};
    e.ring.put(0, 1, kRecEop, 7, db, 200);
    e.ring.wptr[0] = 2;
    EXPECT_EQ(e.drain(), 1u) << "the retained START pairs its cross-batch EOP";
    EXPECT_EQ(e.st.pairing.starts_overwritten, 0u) << "retained START is never re-fed";
    ASSERT_EQ(e.rec.pairs.count(std::make_pair(db, 7u)), 1u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].first, 100u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].second, 200u);
}

// After a replayed key pairs, the NEXT START on the same raw key inserts a
// fresh entry and pairs normally.
TEST(dlog_drain, d1_replayed_key_then_fresh_start_pairs)
{
    const uint32_t db = 4100;
    env            e(1, 2048);
    e.ring.put(0, 0, kRecStart, 7, db, 100);
    e.ring.put(0, 1, kRecStart, 7, db, 200);
    e.ring.put(0, 2, kRecEop, 7, db, 300);
    e.ring.wptr[0] = 3;
    EXPECT_EQ(e.drain(), 1u);
    EXPECT_EQ(e.st.pairing.starts_overwritten, 1u);
    ASSERT_TRUE(e.st.pairing.pending_starts.empty());
    // A fresh START/EOP on the same raw key now pairs normally.
    e.rec = recorder{};
    e.ring.put(0, 3, kRecStart, 7, db, 500);
    e.ring.put(0, 4, kRecEop, 7, db, 600);
    e.ring.wptr[0] = 5;
    EXPECT_EQ(e.drain(), 1u) << "fresh entry after the replay pairs";
    EXPECT_EQ(e.st.pairing.starts_overwritten, 1u) << "no new overwrite";
    ASSERT_EQ(e.rec.pairs.count(std::make_pair(db, 7u)), 1u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].first, 500u);
}

// Cross-batch equal tick -- START(t) in batch N, EOP(t) in batch N+1.
// No firmware guarantee that start<end for a real pair, so an equal-tick EOP is
// dropped fail-closed (equal_tick_drops), the key drained and re-pairable.
TEST(dlog_drain, d1_cross_batch_equal_tick_dropped)
{
    const uint32_t db = 4100;
    env            e(1, 2048);
    e.ring.put(0, 0, kRecStart, 7, db, 70);
    e.ring.wptr[0] = 1;
    EXPECT_EQ(e.drain(), 0u);
    e.rec = recorder{};
    e.ring.put(0, 1, kRecEop, 7, db, 70);  // equal tick
    e.ring.wptr[0] = 2;
    EXPECT_EQ(e.drain(), 0u) << "equal-tick EOP is not a valid pair";
    EXPECT_EQ(e.st.pairing.equal_tick_drops, 1u);
    EXPECT_TRUE(e.rec.pairs.empty());
    EXPECT_TRUE(e.rec.eops_without_start.empty());
    EXPECT_TRUE(e.st.pairing.pending_starts.empty()) << "key drained to 0, re-pairable";
}

// Cross-batch stale EOP -- START(100) retained from batch N, EOP(50) in
// batch N+1 predates it (belongs to an earlier lost dispatch). The stale EOP is
// dropped and the retained START is left UNTOUCHED, so its own EOP(120) in batch
// N+2 still pairs.
TEST(dlog_drain, d1_cross_batch_stale_eop_leaves_start_and_later_eop_pairs)
{
    const uint32_t db = 4100;
    env            e(1, 2048);
    e.ring.put(0, 0, kRecStart, 7, db, 100);
    e.ring.wptr[0] = 1;
    EXPECT_EQ(e.drain(), 0u);
    ASSERT_EQ(e.st.pairing.pending_starts.size(), 1u);
    // Batch N+1: EOP(50) < retained START(100) -> stale drop, START untouched.
    e.rec = recorder{};
    e.ring.put(0, 1, kRecEop, 7, db, 50);
    e.ring.wptr[0] = 2;
    EXPECT_EQ(e.drain(), 0u);
    EXPECT_EQ(e.st.pairing.stale_eop_drops, 1u);
    EXPECT_EQ(e.st.pairing.pending_starts.size(), 1u) << "retained START still outstanding";
    EXPECT_TRUE(e.rec.eops_without_start.empty()) << "stale EOP NOT forwarded startless";
    // Batch N+2: the START's own EOP(120) pairs.
    e.rec = recorder{};
    e.ring.put(0, 2, kRecEop, 7, db, 120);
    e.ring.wptr[0] = 3;
    EXPECT_EQ(e.drain(), 1u) << "the retained START pairs its true EOP";
    ASSERT_EQ(e.rec.pairs.count(std::make_pair(db, 7u)), 1u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].first, 100u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].second, 120u);
}

// Same-batch {S(t), E(t)} on one key fed in BOTH arrival orders must give
// a byte-identical outcome -- the EOP-before-START tie rule is a property of the
// record multiset, not arrival order. Both orders: one startless EOP forwarded,
// equal_tick_drops == 0, one retained START.
TEST(dlog_drain, d1_same_batch_equal_tick_is_order_independent)
{
    const uint32_t db = 4100;
    // The envs hold a non-movable stream, so both are built in place and filled.
    env  se(1, 2048);  // {S, E}
    env  es(1, 2048);  // {E, S}
    auto fill = [&](env& e, bool eop_first) {
        if(eop_first)
        {
            e.ring.put(0, 0, kRecEop, 7, db, 55);
            e.ring.put(0, 1, kRecStart, 7, db, 55);
        }
        else
        {
            e.ring.put(0, 0, kRecStart, 7, db, 55);
            e.ring.put(0, 1, kRecEop, 7, db, 55);
        }
        e.ring.wptr[0] = 2;
        e.drain();
    };
    fill(se, false);
    fill(es, true);
    // Identical outcome: the EOP ran first against an empty key -> startless,
    // the trailing START is retained; no equal-tick drop in either order.
    EXPECT_EQ(se.rec.eops_without_start.size(), 1u);
    EXPECT_EQ(es.rec.eops_without_start.size(), 1u);
    EXPECT_EQ(se.st.pairing.equal_tick_drops, 0u);
    EXPECT_EQ(es.st.pairing.equal_tick_drops, 0u);
    EXPECT_EQ(se.st.pairing.pending_starts.size(), 1u);
    EXPECT_EQ(es.st.pairing.pending_starts.size(), 1u);
    EXPECT_TRUE(se.rec.pairs.empty());
    EXPECT_TRUE(es.rec.pairs.empty());
}
