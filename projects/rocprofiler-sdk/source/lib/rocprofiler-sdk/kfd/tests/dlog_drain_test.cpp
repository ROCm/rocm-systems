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
// Geometry mirrors GFX12: num_regions=2, region_record_count=2048.

#include "lib/rocprofiler-sdk/kfd/dlog_drain.hpp"
#include "lib/rocprofiler-sdk/kfd/record_pipe.hpp"
#include "lib/rocprofiler-sdk/kfd/stream_geometry.hpp"

#include <gtest/gtest.h>

#include <deque>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

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
// The two production stages back to back: reader copies the ring, processor pairs it.
struct drain_state
{
    ring_cursors cursors = {};
    pair_state   pairing = {};
};
// Copy one ring into `batch` (clearing it first) and return the copied count.
uint64_t
copy_ring(fake_ring& ring, ring_cursors& cur, std::vector<copied_record>& batch)
{
    batch.clear();
    // ring.wptr.data() is a uint64_t* passed as the const volatile void* wptr base;
    // copy_pipes strides it by 8 bytes per region via load_wptr_low32.
    return copy_pipes(
        ring.records.data(), ring.num_regions, ring.rrc, ring.wptr.data(), cur, batch);
}
uint64_t
run_drain(fake_ring& ring, drain_state& st, recorder& rec, uint64_t now_ns = 1000)
{
    auto batch = std::vector<copied_record>{};
    copy_ring(ring, st.cursors, batch);
    return pair_records(batch.data(), batch.size(), st.pairing, now_ns, rec.on_record());
}
// Prime a fresh ring: first drain syncs each pipe cursor to the origin.
void
prime(fake_ring& ring, drain_state& st)
{
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
        prime(ring, st);
    }
    uint64_t drain() { return run_drain(ring, st, rec); }
};

// One GPU's ring + cursors + copied batch, primed to origin; copy() refills the batch.
struct gpu_ring
{
    fake_ring                  ring;
    ring_cursors               cur;
    std::vector<copied_record> batch;
    gpu_ring(uint32_t rrc, uint32_t nreg = 1)
    : ring(nreg, rrc)
    {
        copy_ring(ring, cur, batch);  // prime cursor to origin
    }
    uint64_t copy() { return copy_ring(ring, cur, batch); }
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
    // outstanding=1: a live pending_start always carries at least one outstanding
    // START (the real emplace path seeds it 1). evict_stale returns the count of
    // stranded STARTs, not keys, so a single-start stale key contributes 1.
    st.pairing.pending_starts[1] = pair_state::pending_start{100, 1000, 1, false};  // old
    st.pairing.pending_starts[2] = pair_state::pending_start{200, 5000, 1, false};  // fresh
    EXPECT_EQ(st.pairing.evict_stale(/*now_ns=*/6000, /*max_age_ns=*/2000), 1u);
    EXPECT_EQ(st.pairing.pending_starts.count(1), 0u);
    EXPECT_EQ(st.pairing.pending_starts.count(2), 1u);
}

// The hard size cap bounds pending_starts even when nothing ages out: past the
// cap an insert evicts the OLDEST retained START (by seen_at_ns) and counts it.
TEST(dlog_drain, pending_starts_size_cap_evicts_oldest)
{
    auto mk_start = [](uint32_t dispatch_id, uint64_t ts) {
        auto c             = copied_record{};
        c.rec.record_type  = kRecStart;
        c.rec.doorbell_off = 4100;
        c.rec.dispatch_id  = dispatch_id;
        c.rec.ts_lo        = static_cast<uint32_t>(ts);
        return c;
    };

    pair_state pairing;
    pairing.max_pending_starts = 2;
    recorder rec;

    // Three distinct keys, each in its own batch so seen_at_ns strictly increases.
    for(uint32_t i = 1; i <= 3; ++i)
    {
        auto batch = std::vector<copied_record>{mk_start(i, 10 * i)};
        pair_records(batch.data(), batch.size(), pairing, /*now_ns=*/1000 * i, rec.on_record());
    }

    EXPECT_EQ(pairing.pending_starts.size(), 2u) << "the cap is a hard bound";
    EXPECT_EQ(pairing.starts_cap_evicted, 1u);
    const uint64_t k1 = (uint64_t{4100} << 32) | 1u;
    EXPECT_EQ(pairing.pending_starts.count(k1), 0u) << "the oldest START is the victim";

    // A recurring key is exempt: it cannot grow the map, and evicting there could
    // pick the very key being touched and reset its ambiguity latch.
    const uint64_t k3    = (uint64_t{4100} << 32) | 3u;
    auto           again = std::vector<copied_record>{mk_start(3, 99)};
    pair_records(again.data(), again.size(), pairing, /*now_ns=*/4000, rec.on_record());
    EXPECT_EQ(pairing.pending_starts.size(), 2u);
    EXPECT_EQ(pairing.starts_cap_evicted, 1u) << "no eviction for a recurring key";
    ASSERT_EQ(pairing.pending_starts.count(k3), 1u);
    EXPECT_TRUE(pairing.pending_starts[k3].ambiguous) << "the duplicate still latches ambiguous";
}

// Invalid geometry (0 regions, too many, or non-power-of-two rrc) is rejected.
TEST(dlog_drain, invalid_geometry_rejected)
{
    drain_state st;
    recorder    rec;
    auto        batch = std::vector<copied_record>{};
    EXPECT_EQ(copy_pipes(nullptr, 0, 2048, nullptr, st.cursors, batch), 0u);
    EXPECT_EQ(copy_pipes(nullptr, kMaxRegions + 1, 2048, nullptr, st.cursors, batch), 0u);
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
        {"default_2p5mb_on_lattice", kDlogDefaultRingBytes, 2621440u},
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

// The record's region is preserved; region 0 copies first so cross-region pairs match.
TEST(dlog_drain, copied_records_carry_their_region)
{
    gpu_ring g(2048, 2);
    g.ring.put(0, 0, kRecStart, 1, 4100, 10);
    g.ring.wptr[0] = 1;
    g.ring.put(1, 0, kRecEop, 1, 4100, 20);
    g.ring.wptr[1] = 1;
    ASSERT_EQ(g.copy(), 2u);
    ASSERT_EQ(g.batch.size(), 2u);
    EXPECT_EQ(g.batch[0].region, 0u);
    EXPECT_EQ(g.batch[1].region, 1u);
    pair_state pairing;
    recorder   rec;
    EXPECT_EQ(pair_records(g.batch.data(), g.batch.size(), pairing, 1000, rec.on_record()), 1u);
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

// Tier A (the headline): a second outstanding START on one raw key
// makes it AMBIGUOUS; every EOP on the key is then dropped AT THIS LAYER, never
// paired and never forwarded start-unknown. Falsifies the old unconditional
// overwrite, which paired E(300) with S(250) -- a wrong-dispatch emission with
// no overrun and no torn record.
TEST(dlog_drain, same_key_ambiguity_drops_both_eops)
{
    const uint32_t db = 4100;
    env            e(1, 2048);
    e.ring.put(0, 0, kRecStart, 7, db, 100);
    e.ring.put(0, 1, kRecStart, 7, db, 250);  // duplicate raw key -> ambiguous
    e.ring.put(0, 2, kRecEop, 7, db, 300);
    e.ring.put(0, 3, kRecEop, 7, db, 400);
    e.ring.wptr[0] = 4;
    EXPECT_EQ(e.drain(), 0u);
    EXPECT_TRUE(e.rec.pairs.empty());               // neither EOP paired
    EXPECT_TRUE(e.rec.eops_without_start.empty());  // neither forwarded start-unknown
    EXPECT_EQ(e.st.pairing.starts_overwritten, 1u);
    EXPECT_EQ(e.st.pairing.ambiguous_pairs, 2u);
    EXPECT_TRUE(e.st.pairing.pending_starts.empty());  // outstanding drained to 0
}

// Tier A: ambiguity is sticky for the whole burst -- three STARTs
// on one key and two EOPs must emit nothing; the key stays unpairable until
// outstanding drains.
TEST(dlog_drain, same_key_ambiguity_is_sticky)
{
    const uint32_t db = 4100;
    env            e(1, 2048);
    e.ring.put(0, 0, kRecStart, 7, db, 100);
    e.ring.put(0, 1, kRecStart, 7, db, 250);
    e.ring.put(0, 2, kRecEop, 7, db, 300);
    e.ring.put(0, 3, kRecStart, 7, db, 500);
    e.ring.put(0, 4, kRecEop, 7, db, 600);
    e.ring.wptr[0] = 5;
    EXPECT_EQ(e.drain(), 0u);
    EXPECT_TRUE(e.rec.pairs.empty());
    EXPECT_TRUE(e.rec.eops_without_start.empty());
    EXPECT_EQ(e.st.pairing.starts_overwritten, 2u);
    EXPECT_EQ(e.st.pairing.ambiguous_pairs, 2u);
    EXPECT_EQ(e.st.pairing.pending_starts.size(), 1u);  // outstanding still 1
}

// Tier A: a recycle whose new START arrives while the old START is
// still retained (across batches) marks the key ambiguous BEFORE any EOP binds,
// so E_old cannot steal S_new. Both EOPs dropped, never forwarded.
TEST(dlog_drain, retained_start_recycle_is_ambiguous_across_batches)
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
    e.ring.put(0, 3, kRecEop, 7, db, 400);
    e.ring.wptr[0] = 4;
    EXPECT_EQ(e.drain(), 0u);
    EXPECT_TRUE(e.rec.pairs.empty());
    EXPECT_TRUE(e.rec.eops_without_start.empty());
    EXPECT_EQ(e.st.pairing.ambiguous_pairs, 2u);
    EXPECT_TRUE(e.st.pairing.pending_starts.empty());
}

// Tier A: a duplicate START must NOT refresh the key's
// seen_at_ns, or an unpairable ambiguous key would never age out. It ages from
// the FIRST START.
TEST(dlog_drain, ambiguous_key_ages_from_first_start)
{
    pair_state st;
    auto       make_start = [](uint32_t db, uint32_t id, uint64_t ts) {
        copied_record cr{};
        cr.rec.ts_lo        = static_cast<uint32_t>(ts & 0xFFFFFFFFu);
        cr.rec.ts_hi        = static_cast<uint32_t>(ts >> 32);
        cr.rec.record_type  = kRecStart;
        cr.rec.dispatch_id  = id;
        cr.rec.doorbell_off = db;
        return cr;
    };
    auto           nop = [](const drained_record&) {};
    const uint32_t db  = 4100;
    // first START at now=0; two duplicates at 1000 and 2000 make the key
    // ambiguous. seen_at_ns must remain 0.
    std::vector<copied_record> b0{make_start(db, 7, 100)};
    pair_records(b0.data(), b0.size(), st, /*now_ns=*/0, nop);
    std::vector<copied_record> b1{make_start(db, 7, 200)};
    pair_records(b1.data(), b1.size(), st, /*now_ns=*/1000, nop);
    std::vector<copied_record> b2{make_start(db, 7, 300)};
    pair_records(b2.data(), b2.size(), st, /*now_ns=*/2000, nop);
    ASSERT_EQ(st.pending_starts.size(), 1u);
    EXPECT_TRUE(st.pending_starts.begin()->second.ambiguous);
    // Three STARTs are outstanding on this one ambiguous key; evict_stale counts
    // stranded STARTs, not keys, so aging the key out reports all three.
    EXPECT_EQ(st.evict_stale(/*now_ns=*/2500, /*max_age_ns=*/2000), 3u);
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

// --- Bounded SPSC handoff between the ring-copier and the record processor ---

// Batches come out in the order they went in, with their contents intact.
TEST(record_pipe, preserves_batch_order_and_contents)
{
    auto pipe = record_pipe<4>{};
    EXPECT_TRUE(pipe.empty());
    EXPECT_EQ(pipe.peek(), nullptr);
    for(uint32_t b = 0; b < 3; ++b)
    {
        auto* slot = pipe.acquire();
        ASSERT_NE(slot, nullptr);
        slot->now_ns = 1000 + b;
        for(uint32_t i = 0; i < 4; ++i)
        {
            auto r            = copied_record{};
            r.rec.dispatch_id = b * 10 + i;
            slot->records.emplace_back(r);
        }
        pipe.publish();
    }
    EXPECT_EQ(pipe.size(), 3u);
    for(uint32_t b = 0; b < 3; ++b)
    {
        auto* got = pipe.peek();
        ASSERT_NE(got, nullptr);
        EXPECT_EQ(got->now_ns, 1000u + b);
        ASSERT_EQ(got->records.size(), 4u);
        for(uint32_t i = 0; i < 4; ++i)
            EXPECT_EQ(got->records[i].rec.dispatch_id, b * 10 + i);
        pipe.pop();
    }
    EXPECT_TRUE(pipe.empty());
}

// The producer NEVER blocks: when full, acquire() returns null so the caller drops.
TEST(record_pipe, producer_never_blocks_when_the_consumer_stalls)
{
    auto pipe = record_pipe<4>{};
    for(size_t i = 0; i < pipe.capacity(); ++i)  // fill completely; consumer never runs
    {
        auto* slot = pipe.acquire();
        ASSERT_NE(slot, nullptr);
        pipe.publish();
    }
    EXPECT_EQ(pipe.size(), pipe.capacity());
    uint64_t dropped = 0;
    for(int i = 0; i < 100; ++i)
        if(pipe.acquire() == nullptr) ++dropped;
    EXPECT_EQ(dropped, 100u);
    EXPECT_EQ(pipe.size(), pipe.capacity());
    pipe.pop();  // one pop frees exactly one slot
    EXPECT_NE(pipe.acquire(), nullptr);
}

// Recycled slots arrive cleared, so a stale record is never processed twice.
TEST(record_pipe, recycled_slots_are_cleared)
{
    auto  pipe = record_pipe<2>{};
    auto* a    = pipe.acquire();
    ASSERT_NE(a, nullptr);
    a->records.emplace_back(copied_record{});
    a->records.emplace_back(copied_record{});
    pipe.publish();
    ASSERT_NE(pipe.peek(), nullptr);
    pipe.pop();
    auto* reused = pipe.acquire();
    ASSERT_NE(reused, nullptr);
    EXPECT_TRUE(reused->records.empty());
    EXPECT_EQ(reused->now_ns, 0u);
}

// Real threads: SPSC nothing lost/duplicated, order kept. Run under TSan.
TEST(record_pipe, spsc_threads_lose_and_duplicate_nothing)
{
    constexpr uint32_t kBatches = 2000;
    auto               pipe     = record_pipe<8>{};
    auto               produced = std::atomic<uint32_t>{0};
    auto               dropped  = std::atomic<uint32_t>{0};
    auto               stop     = std::atomic<bool>{false};
    auto               consumer = std::thread{[&pipe, &stop]() {
        uint32_t expect = 0;
        while(!stop.load(std::memory_order_acquire) || !pipe.empty())
        {
            auto* b = pipe.peek();
            if(!b) continue;
            if(!b->records.empty())  // contents must be exactly what was written
            {
                EXPECT_EQ(b->records[0].rec.dispatch_id, b->records.size() - 1);
            }
            EXPECT_EQ(b->now_ns, expect);
            ++expect;
            pipe.pop();
        }
    }};
    for(uint32_t i = 0; i < kBatches; ++i)
    {
        record_batch* slot = nullptr;
        while((slot = pipe.acquire()) == nullptr)  // retry rather than block
            ++dropped;
        slot->now_ns      = produced.load(std::memory_order_relaxed);
        auto r            = copied_record{};
        r.rec.dispatch_id = 0;
        slot->records.emplace_back(r);
        slot->records[0].rec.dispatch_id = static_cast<uint32_t>(slot->records.size() - 1);
        produced.fetch_add(1, std::memory_order_relaxed);
        pipe.publish();
    }
    stop.store(true, std::memory_order_release);
    consumer.join();
    EXPECT_EQ(produced.load(), kBatches);
    EXPECT_TRUE(pipe.empty());
}

// spill_overflow_to_pipe() moves as many FIFO batches as fit and reports empty-ness.
TEST(record_pipe, spill_overflow_hands_off_fifo_and_reports_remaining)
{
    auto pipe     = record_pipe<4>{};
    auto overflow = std::deque<record_batch>{};
    for(uint32_t i = 0; i < 10; ++i)
        overflow.emplace_back().gpu_id = i;
    EXPECT_FALSE(spill_overflow_to_pipe(pipe, overflow));  // fills 4, 6 remain
    EXPECT_EQ(pipe.size(), pipe.capacity());
    EXPECT_EQ(overflow.size(), 6u);
    uint32_t expected = 0;
    while(!overflow.empty() || !pipe.empty())
    {
        auto* got = pipe.peek();
        ASSERT_NE(got, nullptr);
        EXPECT_EQ(got->gpu_id, expected++);
        pipe.pop();
        spill_overflow_to_pipe(pipe, overflow);
    }
    EXPECT_EQ(expected, 10u);
    EXPECT_TRUE(overflow.empty());
    EXPECT_TRUE(spill_overflow_to_pipe(pipe, overflow));  // nothing queued -> no-op
}

// Shutdown data-loss regression: the processor reads ONLY the pipe, so overflow must
// be pumped through before reader_done or copied batches are silently lost.
TEST(record_pipe, shutdown_drains_overflow_before_declaring_done)
{
    constexpr uint32_t kTotal      = 500;
    auto               pipe        = record_pipe<8>{};
    auto               overflow    = std::deque<record_batch>{};
    auto               reader_done = std::atomic<bool>{false};
    auto               consumed    = std::atomic<uint32_t>{0};
    // Processor: exits only after reader_done AND empty pipe; never touches overflow.
    auto processor = std::thread{[&]() {
        while(true)
        {
            auto* b = pipe.peek();
            if(b == nullptr)
            {
                if(reader_done.load(std::memory_order_acquire))
                {
                    if(pipe.peek() == nullptr) break;
                    continue;
                }
                std::this_thread::sleep_for(std::chrono::microseconds{50});
                continue;
            }
            consumed.fetch_add(1, std::memory_order_relaxed);
            pipe.pop();
        }
    }};
    // Reader: copy every batch out; anything that does not fit queues in overflow.
    for(uint32_t i = 0; i < kTotal; ++i)
    {
        record_batch* dst = overflow.empty() ? pipe.acquire() : nullptr;
        if(dst != nullptr)
        {
            dst->gpu_id = i;
            pipe.publish();
        }
        else
        {
            overflow.emplace_back().gpu_id = i;
            spill_overflow_to_pipe(pipe, overflow);
        }
    }
    // The fix: pump remaining overflow into the pipe, THEN declare done.
    while(!spill_overflow_to_pipe(pipe, overflow))
        std::this_thread::sleep_for(std::chrono::microseconds{50});
    ASSERT_TRUE(overflow.empty()) << "overflow must be empty before reader_done";
    reader_done.store(true, std::memory_order_release);
    processor.join();
    EXPECT_EQ(consumed.load(), kTotal)
        << "a batch copied from the ring never reached the processor";
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
// D1 (F9) tick-ordered pairing. The pairer orders each batch by (tick, kind,
// arrival_seq) with EOP-before-START at equal ticks, so temporal order -- not
// region/copy position -- decides every pairing. These are the spec's
// fails-without-fix cases (D1 180-199); they fail under the old all-STARTs-first
// two-pass pairer.
// ---------------------------------------------------------------------------

// D1(a): {S_A, E_A, S_B, E_B} on one raw key in true temporal order -> BOTH pair.
// The old two-pass installed S_A and S_B before any EOP, latching ambiguous and
// dropping both; tick order runs S_A -> E_A(pair) -> S_B -> E_B(pair).
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
    EXPECT_EQ(e.st.pairing.ambiguous_pairs, 0u) << "no spurious ambiguity";
    EXPECT_EQ(e.st.pairing.equal_tick_drops, 0u);
    EXPECT_TRUE(e.st.pairing.pending_starts.empty());
}

// D1(i): reuse tie -- S_A(1) retained, then {E_A(5), S_B(5), E_B(9)} in one batch.
// EOP-before-START at the equal tick 5 pairs E_A with the outstanding S_A first,
// then installs S_B cleanly; both pair, no ambiguity.
TEST(dlog_drain, d1_reuse_equal_tick_both_pair_no_ambiguous)
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
    EXPECT_EQ(e.st.pairing.ambiguous_pairs, 0u);
    EXPECT_EQ(e.st.pairing.equal_tick_drops, 0u);
    EXPECT_TRUE(e.st.pairing.pending_starts.empty());
}

// D1(d): a START retained from batch N pairs its EOP in batch N+1, and the
// retained START does NOT latch ambiguous (it is not re-fed into N+1's sort).
TEST(dlog_drain, d1_cross_batch_pair_no_ambiguous)
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
    EXPECT_EQ(e.st.pairing.ambiguous_pairs, 0u) << "retained START never latches ambiguous";
    EXPECT_EQ(e.st.pairing.starts_overwritten, 0u);
    ASSERT_EQ(e.rec.pairs.count(std::make_pair(db, 7u)), 1u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].first, 100u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].second, 200u);
}

// D1(e): a latched-ambiguous key drains one outstanding START per ambiguous EOP
// to 0, then the NEXT START re-inserts a fresh non-ambiguous entry and pairs.
TEST(dlog_drain, d1_sticky_ambiguous_drains_then_fresh_start_pairs)
{
    const uint32_t db = 4100;
    env            e(1, 2048);
    // Two outstanding STARTs latch ambiguous; two EOPs drain outstanding to 0.
    e.ring.put(0, 0, kRecStart, 7, db, 100);
    e.ring.put(0, 1, kRecStart, 7, db, 200);
    e.ring.put(0, 2, kRecEop, 7, db, 300);
    e.ring.put(0, 3, kRecEop, 7, db, 400);
    e.ring.wptr[0] = 4;
    EXPECT_EQ(e.drain(), 0u);
    EXPECT_EQ(e.st.pairing.ambiguous_pairs, 2u);
    ASSERT_TRUE(e.st.pairing.pending_starts.empty()) << "drained to 0";
    // A fresh START/EOP on the same raw key now pairs normally.
    e.rec = recorder{};
    e.ring.put(0, 4, kRecStart, 7, db, 500);
    e.ring.put(0, 5, kRecEop, 7, db, 600);
    e.ring.wptr[0] = 6;
    EXPECT_EQ(e.drain(), 1u) << "fresh entry after the ambiguous run pairs";
    EXPECT_EQ(e.st.pairing.ambiguous_pairs, 2u) << "no new ambiguity";
    ASSERT_EQ(e.rec.pairs.count(std::make_pair(db, 7u)), 1u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 7u)].first, 500u);
}

// D1(g): cross-batch equal tick -- START(t) in batch N, EOP(t) in batch N+1.
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

// D1(h): cross-batch stale EOP -- START(100) retained from batch N, EOP(50) in
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

// D1(f): same-batch {S(t), E(t)} on one key fed in BOTH arrival orders must give
// a byte-identical outcome -- the EOP-before-START tie rule is a property of the
// record multiset, not arrival order. Both orders: one startless EOP forwarded,
// equal_tick_drops == 0, one retained START.
TEST(dlog_drain, d1_same_batch_equal_tick_is_order_independent)
{
    const uint32_t db  = 4100;
    auto           run = [&](bool eop_first) {
        env e(1, 2048);
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
        return e;
    };
    auto se = run(false);  // {S, E}
    auto es = run(true);   // {E, S}
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
