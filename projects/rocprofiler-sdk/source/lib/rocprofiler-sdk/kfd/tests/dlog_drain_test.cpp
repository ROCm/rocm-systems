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

// Unit tests for the dispatch-log ring drain (dlog_drain.hpp). These exercise the
// real production drain logic against a hand-built in-memory ring, so the region
// geometry and start/eop pairing are verified without a GPU, an mmap, or the
// reader's singletons. Geometry mirrors GFX12: num_regions=2 regions,
// region_record_count=2048 slots per region (region r at slots [r*2048,(r+1)*2048)).

#include "lib/rocprofiler-sdk/kfd/dlog_drain.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace
{
using namespace rocprofiler::kfd;

// A hand-built dispatch-log ring: a records region of num_regions*rrc slots plus
// per-pipe wptr/rptr arrays, matching the mmap layout drain_pipes() consumes.
struct fake_ring
{
    uint32_t              num_regions;
    uint32_t              rrc;  // region_record_count (per-region slot count)
    std::vector<uint8_t>  records;
    std::vector<uint64_t> wptr;
    std::vector<uint64_t> rptr;

    fake_ring(uint32_t nreg, uint32_t region_record_count)
    : num_regions(nreg)
    , rrc(region_record_count)
    , records(static_cast<size_t>(nreg) * region_record_count * kFwRecBytes, 0)
    , wptr(nreg, 0)
    , rptr(nreg, 0)
    {}

    // Write a record into region `region` at ring index idx. Region r's slots are
    // [r*rrc, (r+1)*rrc); idx masks into that region (rrc is a power of two).
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

// Recording sink: capture the record callback stream so tests can assert on it.
// Matched pairs and EOPs whose START was lost (shape ii) are kept apart, and the
// drain's loss-free verdict is captured for every record it publishes.
struct recorder
{
    std::map<std::pair<uint32_t, uint32_t>, std::pair<uint64_t, uint64_t>> pairs;  // -> (start,end)
    std::vector<std::pair<uint32_t, uint32_t>>                             eops_without_start;
    size_t                                                                 records      = 0;
    bool                                                                   all_loss_free = true;

    auto on_record()
    {
        return [this](const drained_record& r) {
            ++records;
            if(!r.loss_free) all_loss_free = false;
            if(r.start_known)
                pairs[{r.doorbell_off, r.dispatch_id}] = {r.start_ticks, r.end_ticks};
            else
                eops_without_start.emplace_back(r.doorbell_off, r.dispatch_id);
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
                       rec.on_record());
}
}  // namespace

// First drain starts each pipe at the ring origin: the session buffer is freshly
// allocated, so records already present belong to this session and must be
// consumed, not skipped. rptr ends up synced to wptr.
TEST(dlog_drain, first_drain_consumes_from_origin)
{
    fake_ring   ring(2, 2048);
    drain_state st;
    recorder    rec;

    const uint32_t db = 4100;
    ring.put(0, 0, kRecStart, 7, db, 111);
    ring.put(0, 1, kRecEop, 7, db, 222);
    ring.wptr[0] = 2;

    uint64_t pairs = run_drain(ring, st, rec);

    EXPECT_EQ(pairs, 1u);
    ASSERT_EQ(rec.pairs.count(std::make_pair(db, 7u)), 1u);
    EXPECT_EQ(rec.pairs[std::make_pair(db, 7u)].first, 111u);
    EXPECT_EQ(ring.rptr[0], 2u);  // cursor advanced to the producer
    EXPECT_TRUE(st.rptr_init);
}

// A single pipe with N start+eop pairs: all pair, correct ticks, rptr advances.
TEST(dlog_drain, single_pipe_pairs_all)
{
    fake_ring   ring(2, 2048);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);  // first drain (empty ring)

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
    run_drain(ring, st, rec0);  // first drain (empty ring)

    const uint32_t dbA = 4100, dbB = 4102;
    // region 0: 20 pairs (doorbell A) at slots 0..39
    for(uint32_t i = 0; i < 20; ++i)
    {
        ring.put(0, 2 * i, kRecStart, i, dbA, 100 + i);
        ring.put(0, 2 * i + 1, kRecEop, i, dbA, 500 + i);
    }
    ring.wptr[0] = 40;
    // region 1: 40 pairs (doorbell B) at slots 2048..2127
    for(uint32_t i = 0; i < 40; ++i)
    {
        ring.put(1, 2 * i, kRecStart, i, dbB, 700 + i);
        ring.put(1, 2 * i + 1, kRecEop, i, dbB, 900 + i);
    }
    ring.wptr[1] = 80;

    recorder rec;
    uint64_t pairs = run_drain(ring, st, rec);

    EXPECT_EQ(pairs, 60u);  // 20 + 40, both pipes captured
    uint32_t a = 0;
    uint32_t b = 0;
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
    run_drain(ring, st, rec0);  // first drain (empty ring)

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

// An eop with no matching start is not a pair, but it IS reported: the EOP still
// proves the kernel completed (shape ii), it just carries no interval. It is not
// counted as a pair and carries start_known=false so no consumer can invent one.
TEST(dlog_drain, unmatched_eop_reported_without_a_start)
{
    fake_ring   ring(2, 2048);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);

    ring.put(0, 0, kRecEop, 9, 4100, 42);
    ring.wptr[0] = 1;

    recorder rec;
    EXPECT_EQ(run_drain(ring, st, rec), 0u);  // no matched pair
    EXPECT_TRUE(rec.pairs.empty());
    ASSERT_EQ(rec.eops_without_start.size(), 1u);
    EXPECT_EQ(rec.eops_without_start[0].first, 4100u);
    EXPECT_EQ(rec.eops_without_start[0].second, 9u);
    EXPECT_EQ(st.unmatched_eops, 1u);
    EXPECT_TRUE(rec.all_loss_free);
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

// --- Degradation: the drain must stay well-behaved under bad geometry, overrun,
// and ring wrap, always falling back gracefully rather than crashing/looping. ---

// Invalid geometry (num_regions==0, num_regions beyond the cursor storage, or a
// non-power-of-two region slot count) must be rejected up front: drain reports
// nothing and never masks with a bad slot count.
TEST(dlog_drain, invalid_geometry_rejected)
{
    drain_state st;
    recorder    rec;

    // num_regions == 0 -> reject.
    EXPECT_EQ(drain_pipes(nullptr, 0, 2048, nullptr, nullptr, st, 1000, rec.on_record()), 0u);

    // num_regions beyond kMaxRegions (cursor storage) -> reject.
    EXPECT_EQ(drain_pipes(nullptr, kMaxRegions + 1, 2048, nullptr, nullptr, st, 1000,
                          rec.on_record()),
              0u);

    // region_record_count not a power of two (3000) -> reject.
    fake_ring ring(2, 3000);
    EXPECT_EQ(run_drain(ring, st, rec), 0u);
    EXPECT_TRUE(rec.pairs.empty());
}

// Overrun: firmware lapped the consumer (wptr - rptr > region slots). The drain
// must resume just behind the producer (w - region_slots + 1), drain only the
// still-valid window, and terminate — no infinite loop, no reads outside the ring,
// rptr left synced to wptr.
TEST(dlog_drain, overrun_recovery)
{
    fake_ring   ring(2, 2048);  // 2048 slots per region
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);  // first drain (empty ring) -> rptr[*]=0

    const uint32_t db = 4100;
    // Producer has run far ahead: 10000 records written to region 0 (>> 2048 slots),
    // so all but the last 2048 are already overwritten. Place a valid start/eop pair
    // in the still-live tail window to confirm recovery still pairs it.
    // Recovery point = w - region_slots + 1 = 10000 - 2048 + 1 = 7953.
    ring.put(0, 9990, kRecStart, 42, db, 111);
    ring.put(0, 9991, kRecEop, 42, db, 222);
    ring.wptr[0] = 10000;

    recorder rec;
    uint64_t pairs = run_drain(ring, st, rec);

    // Terminated (did not hang), advanced rptr to wptr, and paired the tail record.
    EXPECT_EQ(ring.rptr[0], 10000u);
    EXPECT_EQ(pairs, 1u);
    ASSERT_EQ(rec.pairs.count(std::make_pair(db, 42u)), 1u);
    EXPECT_EQ(rec.pairs[std::make_pair(db, 42u)].first, 111u);
}

// --- Phase 0: wptr overrun DETECTION (U5/U6/U7). The producer has lapped the
// reader once w - rptr reaches region_slots (>=, not >): its next write target is
// then the slot at rptr. Detection only -- pairing behavior is unchanged. ---

// U7 (safe side): w - rptr == region_slots - 1 drains fully and is NOT an overrun.
TEST(dlog_drain, boundary_one_below_full_is_safe)
{
    fake_ring   ring(1, 8);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);

    const uint32_t db = 4100;
    ring.put(0, 0, kRecStart, 1, db, 10);
    ring.put(0, 1, kRecEop, 1, db, 20);
    ring.wptr[0] = 7;  // region_slots - 1

    recorder rec;
    EXPECT_EQ(run_drain(ring, st, rec), 1u);
    EXPECT_EQ(st.overruns, 0u);
    EXPECT_EQ(st.lost_records, 0u);
    EXPECT_EQ(ring.rptr[0], 7u);
}

// U7 (overrun side) + U5: w - rptr == region_slots exactly is an OVERRUN, because
// the producer's next write target is the slot at rptr. Phase 0 detects it without
// changing what the drain publishes.
TEST(dlog_drain, boundary_exactly_full_is_overrun)
{
    fake_ring   ring(1, 8);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);

    const uint32_t db = 4100;
    ring.put(0, 0, kRecStart, 1, db, 10);
    ring.put(0, 1, kRecEop, 1, db, 20);
    ring.wptr[0] = 8;  // == region_slots

    recorder rec;
    EXPECT_EQ(run_drain(ring, st, rec), 1u);  // behavior unchanged
    EXPECT_EQ(st.overruns, 1u);
    EXPECT_EQ(st.lost_records, 1u);  // the frontier==rptr collision slot
}

// U5: a deep lap before the scan is detected, and the loss count reports every
// record past the safe window.
TEST(dlog_drain, deep_lap_before_scan_detected)
{
    fake_ring   ring(2, 2048);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);

    ring.wptr[0] = 10000;

    recorder rec;
    run_drain(ring, st, rec);

    EXPECT_EQ(st.overruns, 1u);
    EXPECT_EQ(st.lost_records, 10000u - 2048u + 1u);
}

// U6: the producer laps DURING the scan. The pre-scan wptr looked safe, so only
// the after-scan re-read can catch it. The pair callback advances wptr here,
// which is exactly the "producer moved while we were reading" schedule.
TEST(dlog_drain, mid_scan_lap_detected_by_after_scan_reread)
{
    fake_ring   ring(1, 8);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);

    const uint32_t db = 4100;
    ring.put(0, 0, kRecStart, 1, db, 10);
    ring.put(0, 1, kRecEop, 1, db, 20);
    ring.wptr[0] = 4;  // safe at the pre-scan read

    // The post-scan seam (S1) moves the producer between the scan and the
    // after-scan wptr read, which is exactly the mid-scan lap window.
    recorder rec;
    uint64_t pairs = drain_pipes(ring.records.data(),
                                 ring.num_regions,
                                 ring.rrc,
                                 ring.wptr.data(),
                                 ring.rptr.data(),
                                 st,
                                 1000,
                                 rec.on_record(),
                                 [&ring]() { ring.wptr[0] = 9; });

    EXPECT_EQ(pairs, 1u);
    EXPECT_EQ(st.overruns, 1u);
    EXPECT_EQ(st.lost_records, 9u - 8u + 1u);
    // The lap is known BEFORE anything is published, so the record is reported as
    // not loss-free and a signal-less consumer will refuse to complete on it.
    EXPECT_EQ(rec.records, 1u);
    EXPECT_FALSE(rec.all_loss_free);
}

// One lap is counted once, not twice: a pre-scan overrun must not also be counted
// by the after-scan re-read.
TEST(dlog_drain, overrun_counted_once_per_drain)
{
    fake_ring   ring(1, 8);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);

    ring.wptr[0] = 12;

    recorder rec;
    run_drain(ring, st, rec);

    EXPECT_EQ(st.overruns, 1u);
    EXPECT_EQ(st.lost_records, 12u - 8u + 1u);
}

// --- Phase 0: ring-size env-var parsing (U16). Only a plain decimal integer in
// [1, kDlogMaxRingKb] is accepted; everything else returns 0 so the caller warns
// and uses the default. The unit is KiB: the driver accepts sub-MiB rings (the
// 80 KiB default) and rejects >=1 MiB ones, so MiB granularity could not express
// a working size. ---
TEST(dlog_ring_size, env_value_parsing)
{
    constexpr uint64_t kb = 1024;

    // Rejected: empty, zero, negative, non-numeric, whitespace, trailing junk.
    EXPECT_EQ(dlog_ring_bytes_from_kb_str(""), 0u);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("0"), 0u);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("00"), 0u);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("-1"), 0u);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("-80"), 0u);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("abc"), 0u);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str(" 80"), 0u);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("80 "), 0u);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("80K"), 0u);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("+80"), 0u);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("0x80"), 0u);

    // Accepted: the smallest value, the 80 KiB default, and the largest value
    // that fits the uint32 field. The parser only bounds the uint32 field; making
    // the value driver-legal is dlog_snap_ring_bytes()'s job.
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("1"), 1u * kb);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("80"), 80u * kb);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("80"), kDlogMinRingBytes);
    EXPECT_EQ(kDlogMinRingBytes, 81920u);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("1024"), 1024u * kb);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("4194303"), kDlogMaxRingKb * kb);
    EXPECT_EQ(kDlogMaxRingKb, 4194303u);

    // Over the uint32 buffer_size field, and overflow-adjacent inputs: rejected
    // without ever wrapping (the bound is checked per digit).
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("4194304"), 0u);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("4294967296"), 0u);
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("18446744073709551615"), 0u);  // UINT64_MAX
    EXPECT_EQ(dlog_ring_bytes_from_kb_str("18446744073709551616"), 0u);  // UINT64_MAX + 1
    EXPECT_EQ(dlog_ring_bytes_from_kb_str(std::string(64, '9')), 0u);
}

// The snap makes every requested size driver-legal. REGISTER_BUFFER requires
// buffer_size == num_regions * 20 * region_record_count with a power-of-two
// region_record_count <= 2^24; the 80 * 2^k lattice satisfies that for both
// ASIC region counts (2 on gfx12, 4 on gfx9.4.x/9.5.0) without knowing which one
// applies, because num_regions is only reported after the size is chosen.
TEST(dlog_ring_size, snap_yields_a_driver_legal_size)
{
    // Any requested size, however arbitrary, lands on the lattice inside bounds
    // and is legal for a 2-region AND a 4-region ASIC.
    for(uint64_t want : {uint64_t{0},
                         uint64_t{1},
                         kDlogMinRingBytes - 1,
                         kDlogMinRingBytes,
                         uint64_t{100000},
                         uint64_t{131072},  // 128 KiB
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
        EXPECT_LE(sz, 0xFFFFFFFFull);                       // uint32 buffer_size field
        if(want >= kDlogMinRingBytes)
        {
            EXPECT_LE(sz, want);  // never rounds up
        }
        EXPECT_EQ(sz % 80u, 0u);

        // region_record_count is a power of two and within 2^24 under both counts.
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

// The specific cases that motivated the snap.
TEST(dlog_ring_size, snap_boundaries)
{
    // The default is already on the lattice: it must snap to itself, so an unset
    // env var keeps the known-good 80 KiB / 2048-record-per-region geometry.
    EXPECT_EQ(dlog_snap_ring_bytes(kDlogMinRingBytes), 81920u);
    EXPECT_EQ(81920u / (2 * 20), 2048u);  // gfx12 region_record_count

    // 128 KiB is NOT valid (131072/40 = 3276.8): snap down to 80 KiB rather than
    // let REGISTER_BUFFER EINVAL disable the dispatch log for the process.
    EXPECT_EQ(dlog_snap_ring_bytes(131072u), 81920u);

    // Anything below the floor clamps up to it.
    EXPECT_EQ(dlog_snap_ring_bytes(0u), kDlogMinRingBytes);
    EXPECT_EQ(dlog_snap_ring_bytes(1u), kDlogMinRingBytes);

    // Sizes verified as accepted on hardware sit exactly on the lattice and are
    // returned unchanged: 640 KiB, 5 MiB, 40 MiB, 640 MiB.
    EXPECT_EQ(dlog_snap_ring_bytes(655360u), 655360u);
    EXPECT_EQ(dlog_snap_ring_bytes(5242880u), 5242880u);
    EXPECT_EQ(dlog_snap_ring_bytes(41943040u), 41943040u);
    EXPECT_EQ(dlog_snap_ring_bytes(671088640u), 671088640u);
    EXPECT_EQ(kDlogMaxRingBytes, 671088640u);  // 640 MiB == 80 * 2^23

    // Above the ceiling clamps to it; just below it snaps to the next lattice
    // point down (320 MiB), never up.
    EXPECT_EQ(dlog_snap_ring_bytes(671088641u), kDlogMaxRingBytes);
    EXPECT_EQ(dlog_snap_ring_bytes(kDlogMaxRingKb * 1024), kDlogMaxRingBytes);
    EXPECT_EQ(dlog_snap_ring_bytes(kDlogMaxRingBytes - 1), 335544320u);
}

// The sizing math the session performs on the parsed value is validated BEFORE
// use: any snapped value fits the uint32 buffer_size ioctl field, and the
// derived sizes (arr_bytes, signal_off, alloc_size, stride, aperture candidate)
// cannot overflow uint64.
TEST(dlog_ring_size, accepted_sizes_keep_the_session_math_in_range)
{
    for(uint64_t k : {uint64_t{1}, kDlogMinRingBytes / 1024, uint64_t{8192}, kDlogMaxRingKb})
    {
        uint64_t buf_bytes = dlog_snap_ring_bytes(dlog_ring_bytes_from_kb_str(std::to_string(k)));
        ASSERT_LE(buf_bytes, kDlogMaxRingBytes);
        ASSERT_LE(buf_bytes, 0xFFFFFFFFull);  // uint32 buffer_size field

        // Mirrors setup_session(): arr_bytes/signal_off/alloc_size, then the
        // aperture walk's stride and its farthest (i == 127) candidate.
        const uint64_t arr_bytes  = 64;
        const uint64_t signal_off = buf_bytes + arr_bytes * 2;
        const uint64_t alloc_size = signal_off + arr_bytes + 4095;  // page round-up bound
        const uint64_t stride     = alloc_size + (8ull << 20) + 4095;
        const uint64_t cand       = (64ull << 30) + stride * 127;

        EXPECT_GT(signal_off, buf_bytes);
        EXPECT_GT(alloc_size, signal_off);
        EXPECT_GT(stride, alloc_size);
        EXPECT_GT(cand + alloc_size, cand);  // no wrap at the far end of the walk
    }
}

// Ring wrap: a pair whose indices straddle the power-of-two wrap boundary (start at
// the last slot, eop at slot 0 of the next lap) must still map to the right physical
// slots and pair correctly.
TEST(dlog_drain, ring_wrap_pairs_across_boundary)
{
    fake_ring   ring(2, 2048);  // 2048 slots per region
    drain_state st;
    recorder    rec_sync;
    run_drain(ring, st, rec_sync);  // first drain (empty ring)
    // Prime rptr to just before the power-of-two wrap so we drain [2047, 2049).
    st.rptr[0] = 2047;

    const uint32_t db = 4100;
    ring.put(0, 2047, kRecStart, 7, db, 500);  // physical slot 2047
    ring.put(0, 2048, kRecEop, 7, db, 600);    // 2048 & 2047 = physical slot 0 (wrapped)
    ring.wptr[0] = 2049;

    recorder rec;
    uint64_t pairs = run_drain(ring, st, rec);

    EXPECT_EQ(pairs, 1u);
    ASSERT_EQ(rec.pairs.count(std::make_pair(db, 7u)), 1u);
    EXPECT_EQ(rec.pairs[std::make_pair(db, 7u)].first, 500u);
    EXPECT_EQ(rec.pairs[std::make_pair(db, 7u)].second, 600u);
}

// --- Phase 2 unit 3: the drain's loss-free verdict is what lets a signal-less
// consumer decide whether an EOP may prove a completion. ---

// A normal drain reports every record as loss-free.
TEST(dlog_drain, loss_free_verdict_on_a_clean_drain)
{
    fake_ring   ring(1, 8);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);

    const uint32_t db = 4100;
    ring.put(0, 0, kRecStart, 1, db, 10);
    ring.put(0, 1, kRecEop, 1, db, 20);
    ring.put(0, 2, kRecEop, 2, db, 30);  // shape ii: no start
    ring.wptr[0] = 3;

    recorder rec;
    EXPECT_EQ(run_drain(ring, st, rec), 1u);
    EXPECT_EQ(rec.records, 2u);
    EXPECT_TRUE(rec.all_loss_free);
    EXPECT_EQ(rec.pairs.size(), 1u);
    EXPECT_EQ(rec.eops_without_start.size(), 1u);
    EXPECT_EQ(st.overruns, 0u);
}

// A pre-scan overrun (the `>=` boundary) marks the whole region's records not
// loss-free, so nothing drained from it may prove a completion.
TEST(dlog_drain, overrun_marks_records_not_loss_free)
{
    fake_ring   ring(1, 8);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);

    ring.put(0, 0, kRecStart, 1, 4100, 10);
    ring.put(0, 1, kRecEop, 1, 4100, 20);
    ring.wptr[0] = 8;  // == region_slots: the definitive overrun boundary

    recorder rec;
    run_drain(ring, st, rec);

    EXPECT_EQ(st.overruns, 1u);
    EXPECT_GT(rec.records, 0u);
    EXPECT_FALSE(rec.all_loss_free);
}

// Nothing is published before the after-scan wptr read settles the verdict, so a
// consumer never observes a record it would later have to un-publish.
TEST(dlog_drain, records_are_published_only_after_the_verdict_is_known)
{
    fake_ring   ring(1, 8);
    drain_state st;
    recorder    rec0;
    run_drain(ring, st, rec0);

    ring.put(0, 0, kRecStart, 1, 4100, 10);
    ring.put(0, 1, kRecEop, 1, 4100, 20);
    ring.wptr[0] = 2;

    // The seam laps the producer after the scan; every record published must
    // already carry the lossy verdict.
    bool     seen_any_loss_free = false;
    uint64_t pairs              = drain_pipes(ring.records.data(),
                                 ring.num_regions,
                                 ring.rrc,
                                 ring.wptr.data(),
                                 ring.rptr.data(),
                                 st,
                                 1000,
                                 [&seen_any_loss_free](const drained_record& r) {
                                     if(r.loss_free) seen_any_loss_free = true;
                                 },
                                 [&ring]() { ring.wptr[0] = 40; });

    EXPECT_EQ(pairs, 1u);
    EXPECT_FALSE(seen_any_loss_free);
}

// Queue destroy asks the reader to drop what it retains for the dead queue's
// doorbell slot, so a stale start cannot pair with a record from whatever reuses
// it. Only that slot's starts go.
TEST(dlog_drain, erase_slot_drops_only_that_slots_retained_starts)
{
    drain_state st;
    // key = (doorbell_off << 32) | dispatch_id; page slot = doorbell_off & 1023.
    const uint64_t on_slot_7    = (uint64_t{7} << 32) | 1;
    const uint64_t also_slot_7  = (uint64_t{1024 + 7} << 32) | 2;  // aliases to slot 7
    const uint64_t on_slot_8    = (uint64_t{8} << 32) | 3;

    st.pending_starts[on_slot_7]   = drain_state::pending_start{100, 1000};
    st.pending_starts[also_slot_7] = drain_state::pending_start{200, 1000};
    st.pending_starts[on_slot_8]   = drain_state::pending_start{300, 1000};

    EXPECT_EQ(st.erase_slot(/*page_slot=*/7, /*slots_per_page=*/1024), 2u);
    EXPECT_EQ(st.pending_starts.count(on_slot_7), 0u);
    EXPECT_EQ(st.pending_starts.count(also_slot_7), 0u);
    EXPECT_EQ(st.pending_starts.count(on_slot_8), 1u);

    EXPECT_EQ(st.erase_slot(7, 1024), 0u);  // idempotent
}
