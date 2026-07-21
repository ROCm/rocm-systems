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

// Unit tests for the per-pipe dispatch-log ring drain (dlog_drain.hpp). These
// exercise the real production drain logic against a hand-built in-memory ring, so
// the per-pipe geometry and start/eop pairing are verified without a GPU, an mmap,
// or the reader's singletons. Geometry mirrors GFX12: num_regions=2 pipes,
// region_record_count=2048 => slots_per_pipe=1024.

#include "lib/rocprofiler-sdk/kfd/dlog_drain.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <vector>

namespace
{
using namespace rocprofiler::kfd;

// A hand-built dispatch-log ring: a records region of num_regions*rrc slots plus
// per-pipe wptr/rptr arrays, matching the mmap layout drain_pipes() consumes.
struct fake_ring
{
    uint32_t              num_regions;
    uint32_t              rrc;  // region_record_count
    uint32_t              slots_per_pipe;
    std::vector<uint8_t>  records;
    std::vector<uint64_t> wptr;
    std::vector<uint64_t> rptr;

    fake_ring(uint32_t nreg, uint32_t region_record_count)
    : num_regions(nreg)
    , rrc(region_record_count)
    , slots_per_pipe(region_record_count / nreg)
    , records(static_cast<size_t>(nreg) * region_record_count * kFwRecBytes, 0)
    , wptr(nreg, 0)
    , rptr(nreg, 0)
    {}

    // Write a record at pipe p, ring index idx (its physical slot within the pipe).
    void put(uint32_t pipe,
             uint64_t idx,
             uint32_t rtype,
             uint32_t dispatch_id,
             uint32_t doorbell_off,
             uint64_t ts)
    {
        uint64_t slot = static_cast<uint64_t>(pipe) * slots_per_pipe + (idx & (slots_per_pipe - 1));
        fw_record rec{};
        rec.ts_lo        = static_cast<uint32_t>(ts & 0xFFFFFFFFu);
        rec.ts_hi        = static_cast<uint32_t>(ts >> 32);
        rec.record_type  = rtype;
        rec.dispatch_id  = dispatch_id;
        rec.doorbell_off = doorbell_off;
        std::memcpy(records.data() + slot * kFwRecBytes, &rec, sizeof(rec));
    }
};

// Recording sinks: capture the callback stream so tests can assert on it.
struct recorder
{
    std::vector<std::pair<uint32_t, uint32_t>>                             records;  // (db, disp)
    std::map<std::pair<uint32_t, uint32_t>, std::pair<uint64_t, uint64_t>> pairs;  // -> (start,end)

    auto on_record()
    {
        return [this](uint32_t db, uint32_t disp) { records.emplace_back(db, disp); };
    }
    auto on_pair()
    {
        return [this](uint32_t db, uint32_t disp, uint64_t start, uint64_t end) {
            pairs[{db, disp}] = {start, end};
        };
    }
};

// Run one drain over the ring. now_ns defaults to a fixed value for determinism.
uint64_t
run_drain(fake_ring& ring, drain_state& st, recorder& rec, uint64_t now_ns = 1000)
{
    return drain_pipes(ring.records.data(),
                       ring.num_regions,
                       ring.rrc,
                       ring.wptr.data(),
                       ring.rptr.data(),
                       st,
                       now_ns,
                       rec.on_record(),
                       rec.on_pair());
}
}  // namespace

// First drain only syncs cursors to wptr; it must report nothing and set rptr=wptr
// so pre-existing records are never replayed.
TEST(dlog_drain, first_drain_syncs_and_reports_nothing)
{
    fake_ring   ring(2, 2048);
    drain_state st;
    recorder    rec;

    ring.put(0, 0, kRecStart, 7, 4100, 111);
    ring.wptr[0] = 1;  // pretend a record predates our attach

    uint64_t pairs = run_drain(ring, st, rec);

    EXPECT_EQ(pairs, 0u);
    EXPECT_TRUE(rec.records.empty());
    EXPECT_EQ(ring.rptr[0], 1u);  // cursor synced forward
    EXPECT_TRUE(st.rptr_init);
}

// A single pipe with N start+eop pairs: all pair, correct ticks, rptr advances.
TEST(dlog_drain, single_pipe_pairs_all)
{
    fake_ring   ring(2, 2048);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);  // sync

    const uint32_t db = 4100;
    for(uint32_t i = 0; i < 40; ++i)
    {
        ring.put(0, 2 * i, kRecStart, i, db, 1000 + i);
        ring.put(0, 2 * i + 1, kRecEop, i, db, 2000 + i);
    }
    ring.wptr[0] = 80;

    recorder rec;
    uint64_t pairs = run_drain(ring, st, rec);

    EXPECT_EQ(pairs, 40u);
    EXPECT_EQ(rec.pairs.size(), 40u);
    EXPECT_EQ(ring.rptr[0], 80u);
    for(uint32_t i = 0; i < 40; ++i)
    {
        auto it = rec.pairs.find({db, i});
        ASSERT_NE(it, rec.pairs.end());
        EXPECT_EQ(it->second.first, 1000u + i);
        EXPECT_EQ(it->second.second, 2000u + i);
    }
}

// The regression case: two pipes with DIFFERENT counts. Proves per-pipe indexing
// (each wptr[i] tracks one doorbell) — the flat/sub-block reader captured only one.
TEST(dlog_drain, two_pipes_asymmetric_capture_both)
{
    fake_ring   ring(2, 2048);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);  // sync

    const uint32_t dbA = 4100, dbB = 4102;
    // pipe 0: 20 pairs (doorbell A) at slots 0..39
    for(uint32_t i = 0; i < 20; ++i)
    {
        ring.put(0, 2 * i, kRecStart, i, dbA, 100 + i);
        ring.put(0, 2 * i + 1, kRecEop, i, dbA, 500 + i);
    }
    ring.wptr[0] = 40;
    // pipe 1: 40 pairs (doorbell B) at slots 1024..1103
    for(uint32_t i = 0; i < 40; ++i)
    {
        ring.put(1, 2 * i, kRecStart, i, dbB, 700 + i);
        ring.put(1, 2 * i + 1, kRecEop, i, dbB, 900 + i);
    }
    ring.wptr[1] = 80;

    recorder rec;
    uint64_t pairs = run_drain(ring, st, rec);

    EXPECT_EQ(pairs, 60u);  // 20 + 40, both pipes captured
    uint32_t a = 0, b = 0;
    for(auto& kv : rec.pairs)
    {
        if(kv.first.first == dbA) ++a;
        if(kv.first.first == dbB) ++b;
    }
    EXPECT_EQ(a, 20u);
    EXPECT_EQ(b, 40u);
    EXPECT_EQ(ring.rptr[0], 40u);
    EXPECT_EQ(ring.rptr[1], 80u);
}

// Padding slots (record_type==0 or doorbell_off==0) must be skipped, not stop the
// scan: a valid record after padding within [rptr,wptr) still gets drained.
TEST(dlog_drain, padding_slots_skipped)
{
    fake_ring   ring(2, 2048);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);  // sync

    const uint32_t db = 4100;
    ring.put(0, 0, kRecStart, 5, db, 10);
    // slot 1: left as padding (all zero)
    ring.put(0, 2, kRecEop, 5, db, 20);
    ring.wptr[0] = 3;

    recorder rec;
    uint64_t pairs = run_drain(ring, st, rec);

    EXPECT_EQ(pairs, 1u);
    auto key = std::make_pair(db, 5u);
    ASSERT_EQ(rec.pairs.count(key), 1u);
    EXPECT_EQ(rec.pairs[key].first, 10u);
    EXPECT_EQ(rec.pairs[key].second, 20u);
}

// An eop with no matching start is dropped (that dispatch falls back to HSA).
TEST(dlog_drain, unmatched_eop_dropped)
{
    fake_ring   ring(2, 2048);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);

    ring.put(0, 0, kRecEop, 9, 4100, 42);
    ring.wptr[0] = 1;

    recorder rec;
    EXPECT_EQ(run_drain(ring, st, rec), 0u);
    EXPECT_TRUE(rec.pairs.empty());
}

// A start in one drain pairs with its eop in a LATER drain (state persists).
TEST(dlog_drain, pair_spanning_two_drains)
{
    fake_ring   ring(2, 2048);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);

    const uint32_t db = 4100;
    ring.put(0, 0, kRecStart, 3, db, 111);
    ring.wptr[0] = 1;
    recorder rec_a;
    EXPECT_EQ(run_drain(ring, st, rec_a), 0u);  // start seen, not yet paired

    ring.put(0, 1, kRecEop, 3, db, 222);
    ring.wptr[0] = 2;
    recorder rec_b;
    EXPECT_EQ(run_drain(ring, st, rec_b), 1u);
    auto key = std::make_pair(db, 3u);
    ASSERT_EQ(rec_b.pairs.count(key), 1u);
    EXPECT_EQ(rec_b.pairs[key].first, 111u);
    EXPECT_EQ(rec_b.pairs[key].second, 222u);
}

// evict_stale drops unmatched starts older than max_age, keeps fresh ones.
TEST(dlog_drain, evict_stale_starts)
{
    drain_state st;
    st.pending_starts[1] = drain_state::pending_start{100, 1000};  // old
    st.pending_starts[2] = drain_state::pending_start{200, 5000};  // fresh

    size_t removed = st.evict_stale(/*now_ns=*/6000, /*max_age_ns=*/2000);

    EXPECT_EQ(removed, 1u);
    EXPECT_EQ(st.pending_starts.count(1), 0u);
    EXPECT_EQ(st.pending_starts.count(2), 1u);
}
