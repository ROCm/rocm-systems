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
// A hand-built ring: num_regions*rrc record slots plus per-pipe wptr/rptr arrays.
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
// Recording sink: matched pairs and START-less EOPs (shape ii) kept apart, plus the
// per-record loss-free verdict.
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
    return copy_pipes(ring.records.data(),
                      ring.num_regions,
                      ring.rrc,
                      ring.wptr.data(),
                      ring.rptr.data(),
                      cur,
                      batch);
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

// An OutT that advances the shared volatile wptr[region] once a chosen number of
// records have been emplaced -- modelling a producer that laps the reader DURING
// the copy loop, so the post-copy w2 reload exceeds the w that was
// loaded once at the top. Supports the size()/operator[] the back-patch needs.
struct advancing_out
{
    std::vector<copied_record> recs;
    volatile uint64_t*         wptr          = nullptr;
    uint32_t                   region        = 0;
    size_t                     advance_after = SIZE_MAX;  // bump wptr after this many emplaces
    uint64_t                   advance_to    = 0;
    void                       emplace_back(const copied_record& r)
    {
        recs.emplace_back(r);
        if(recs.size() == advance_after && wptr != nullptr)
            __atomic_store_n(&wptr[region], advance_to, __ATOMIC_RELEASE);
    }
    size_t         size() const { return recs.size(); }
    copied_record& operator[](size_t i) { return recs[i]; }
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
    // First drain consumes from the origin: records already present are this session's.
    {
        fake_ring   ring(2, 2048);
        drain_state st;
        recorder    rec;
        ring.put(0, 0, kRecStart, 7, db, 111);
        ring.put(0, 1, kRecEop, 7, db, 222);
        ring.wptr[0] = 2;
        EXPECT_EQ(run_drain(ring, st, rec), 1u);
        ASSERT_EQ(rec.pairs.count(std::make_pair(db, 7u)), 1u);
        EXPECT_EQ(rec.pairs[std::make_pair(db, 7u)].first, 111u);
        EXPECT_EQ(ring.rptr[0], 2u);
        EXPECT_TRUE(st.cursors.rptr_init);
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
        EXPECT_EQ(e.ring.rptr[0], 80u);
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
        EXPECT_EQ(e.ring.rptr[0], 40u);
        EXPECT_EQ(e.ring.rptr[1], 80u);
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
    st.pairing.pending_starts[1] = pair_state::pending_start{100, 1000};  // old
    st.pairing.pending_starts[2] = pair_state::pending_start{200, 5000};  // fresh
    EXPECT_EQ(st.pairing.evict_stale(/*now_ns=*/6000, /*max_age_ns=*/2000), 1u);
    EXPECT_EQ(st.pairing.pending_starts.count(1), 0u);
    EXPECT_EQ(st.pairing.pending_starts.count(2), 1u);
}

// Invalid geometry (0 regions, too many, or non-power-of-two rrc) is rejected.
TEST(dlog_drain, invalid_geometry_rejected)
{
    drain_state st;
    recorder    rec;
    auto        batch = std::vector<copied_record>{};
    EXPECT_EQ(copy_pipes(nullptr, 0, 2048, nullptr, nullptr, st.cursors, batch), 0u);
    EXPECT_EQ(copy_pipes(nullptr, kMaxRegions + 1, 2048, nullptr, nullptr, st.cursors, batch), 0u);
    fake_ring ring(2, 3000);  // region_record_count not a power of two
    EXPECT_EQ(run_drain(ring, st, rec), 0u);
    EXPECT_TRUE(rec.pairs.empty());
}

// Deep overrun: drain resumes at w-region_slots (not +1), drains the
// live tail, syncs rptr.
TEST(dlog_drain, overrun_recovery)
{
    const uint32_t db = 4100;
    env            e;
    // Recovery point = w - region_slots = 10000 - 2048 = 7952; place a valid pair
    // in the still-live tail window (well past the untrusted boundary at 7952).
    e.ring.put(0, 9990, kRecStart, 42, db, 111);
    e.ring.put(0, 9991, kRecEop, 42, db, 222);
    e.ring.wptr[0] = 10000;
    EXPECT_EQ(e.drain(), 1u);
    EXPECT_EQ(e.ring.rptr[0], 10000u);
    ASSERT_EQ(e.rec.pairs.count(std::make_pair(db, 42u)), 1u);
    EXPECT_EQ(e.rec.pairs[std::make_pair(db, 42u)].first, 111u);
}

// Overrun detection: a lap is w-rptr > region_slots, STRICTLY -- exactly
// full is not an overrun. At/over full the oldest copied index aliases
// the producer's next write target, so it is untrusted and its START is dropped.
TEST(dlog_drain, overrun_detection_boundaries)
{
    struct row
    {
        const char* label;
        uint32_t    rrc;
        uint64_t    wptr;
        uint64_t    exp_pairs;
        uint64_t    exp_overruns;
        uint64_t    exp_lost;
        uint64_t    exp_untrusted;
        uint64_t    exp_rptr;
    };
    // A live start/eop pair sits at slots 0,1 of a single region.
    const row rows[] = {
        {"one_below_full_is_safe", 8, 7, 1, 0, 0, 0, 7},
        // exactly-full: not an overrun (dist == slots), but idx 0 (the START) is
        // untrusted and dropped -> its EOP is unmatched, so no pair.
        {"exactly_full_not_overrun_boundary_untrusted", 8, 8, 0, 0, 0, 1, 8},
        // deep lap: overrun at dist > slots; lost = dist - slots (no +1). The pair
        // at physical slots 0,1 is re-read via wrapped indices 8192,8193 (both past
        // the untrusted boundary 7952, so trusted) and still pairs; only the oldest
        // copied index (w-slots = 7952) is untrusted.
        {"deep_lap_detected", 2048, 10000, 1, 1, 10000u - 2048u, 1, 10000},
    };
    const uint32_t db = 4100;
    for(const auto& tc : rows)
    {
        env e(1, tc.rrc);
        e.ring.put(0, 0, kRecStart, 1, db, 10);
        e.ring.put(0, 1, kRecEop, 1, db, 20);
        e.ring.wptr[0] = tc.wptr;
        EXPECT_EQ(e.drain(), tc.exp_pairs) << tc.label;
        EXPECT_EQ(e.st.cursors.overruns, tc.exp_overruns) << tc.label;
        EXPECT_EQ(e.st.cursors.lost_records, tc.exp_lost) << tc.label;
        EXPECT_EQ(e.st.cursors.untrusted_records, tc.exp_untrusted) << tc.label;
        EXPECT_EQ(e.ring.rptr[0], tc.exp_rptr) << tc.label;
    }
}

// A lap during a copy is caught on the NEXT pass: the copier reads wptr once (w),
// so the following drain observes the overrun.
TEST(dlog_drain, lap_after_the_copy_is_caught_on_the_following_drain)
{
    const uint32_t db = 4100;
    env            e(1, 8);
    e.ring.put(0, 0, kRecStart, 1, db, 10);
    e.ring.put(0, 1, kRecEop, 1, db, 20);
    e.ring.wptr[0] = 4;
    EXPECT_EQ(e.drain(), 1u);
    EXPECT_EQ(e.st.cursors.overruns, 0u);
    EXPECT_EQ(e.st.cursors.untrusted_records, 0u);
    e.ring.wptr[0] = 4 + 9;  // producer now laps past the ring
    e.rec          = recorder{};
    e.drain();
    EXPECT_EQ(e.st.cursors.overruns, 1u);
    EXPECT_GT(e.st.cursors.untrusted_records, 0u);
}

// a producer advancing DURING the copy makes the post-copy w2 reload
// mark the aliased records untrusted -- the mid-copy verdict race a single w load
// missed. Here w is captured at 4 (dist 4, no lap), but the producer laps to 11
// mid-copy, so every copied index aliases its write frontier and is untrusted.
TEST(dlog_drain, mid_copy_advance_marks_records_untrusted)
{
    fake_ring    ring(1, 8);
    ring_cursors cur;
    {
        std::vector<copied_record> prime;
        copy_pipes(ring.records.data(), 1, 8, ring.wptr.data(), ring.rptr.data(), cur, prime);
    }
    ring.put(0, 0, kRecStart, 1, 4100, 10);
    ring.put(0, 1, kRecEop, 1, 4100, 20);
    ring.wptr[0] = 4;  // w captured = 4 -> copy [0,4), dist 4 <= slots -> no overrun
    advancing_out out;
    out.wptr          = ring.wptr.data();
    out.region        = 0;
    out.advance_after = 2;   // after 2 emplaces, the producer laps
    out.advance_to    = 11;  // w2 = 11 -> untrusted_upto = 3 -> idx 0..3 untrusted
    const uint64_t copied =
        copy_pipes(ring.records.data(), 1, 8, ring.wptr.data(), ring.rptr.data(), cur, out);
    EXPECT_EQ(copied, 4u);
    EXPECT_EQ(cur.overruns, 0u) << "the single w load saw no lap";
    EXPECT_EQ(cur.untrusted_records, 4u) << "but the reload caught all four as torn";
    for(size_t i = 0; i < out.recs.size(); ++i)
        EXPECT_FALSE(out.recs[i].loss_free);
    // the torn START(id1) is dropped, so no pair emerges.
    pair_state pairing;
    recorder   rec;
    EXPECT_EQ(pair_records(out.recs.data(), out.recs.size(), pairing, 1000, rec.on_record()), 0u);
    EXPECT_TRUE(rec.pairs.empty());
}

// Stream reset zeroes firmware wptr[] but not rptr[]: copy_pipes() must snap rptr
// back to wptr, count the regression, copy nothing (no busy spin).
TEST(dlog_drain, wptr_regression_snaps_rptr_and_converges)
{
    fake_ring   ring(2, 8);
    drain_state st;
    recorder    rec0;
    ring.wptr[0] = 5;  // consume so rptr advances past origin on both regions
    ring.wptr[1] = 3;
    run_drain(ring, st, rec0);
    ASSERT_EQ(st.cursors.rptr[0], 5u);
    ASSERT_EQ(st.cursors.rptr[1], 3u);
    ring.wptr[0] = 0;  // the reset: firmware wptr[] zeroed underneath us
    ring.wptr[1] = 0;
    recorder rec;
    EXPECT_EQ(run_drain(ring, st, rec), 0u);  // nothing to copy from a reset ring
    EXPECT_EQ(rec.records, 0u);
    EXPECT_EQ(st.cursors.wptr_regressions, 2u);  // one per regressed region
    EXPECT_EQ(st.cursors.rptr[0], 0u);
    EXPECT_EQ(st.cursors.rptr[1], 0u);
    EXPECT_EQ(ring.rptr[0], 0u);
    EXPECT_EQ(ring.rptr[1], 0u);
    EXPECT_EQ(st.cursors.overruns, 0u);  // no spurious overrun accounting
    EXPECT_EQ(st.cursors.lost_records, 0u);
    ring.put(0, 0, kRecStart, 9, 4100, 111);  // post-reset advance drains normally
    ring.put(0, 1, kRecEop, 9, 4100, 222);
    ring.wptr[0] = 2;
    recorder rec2;
    EXPECT_EQ(run_drain(ring, st, rec2), 1u);
    EXPECT_EQ(st.cursors.rptr[0], 2u);
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
        {"default_80kb", "80", 80u * kb},
        {"default_is_floor", "80", kDlogMinRingBytes},
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
        {"default_snaps_to_self", kDlogMinRingBytes, 81920u},
        {"128kb_snaps_down_to_80kb", 131072u, 81920u},
        {"zero_clamps_up_to_floor", 0u, kDlogMinRingBytes},
        {"one_clamps_up_to_floor", 1u, kDlogMinRingBytes},
        {"640kb_on_lattice", 655360u, 655360u},
        {"5mb_on_lattice", 5242880u, 5242880u},
        {"40mb_on_lattice", 41943040u, 41943040u},
        {"640mb_on_lattice", 671088640u, 671088640u},
        {"above_ceiling_clamps", 671088641u, kDlogMaxRingBytes},
        {"max_kb_clamps", kDlogMaxRingKb * 1024, kDlogMaxRingBytes},
        {"just_below_ceiling_snaps_down", kDlogMaxRingBytes - 1, 335544320u},
    };
    for(const auto& tc : rows)
        EXPECT_EQ(dlog_snap_ring_bytes(tc.want), tc.expect) << tc.label;
}

// The loss-free verdict lets a signal-less consumer trust an EOP as a completion.
TEST(dlog_drain, loss_free_verdict)
{
    const uint32_t db = 4100;
    // A normal drain, well below full, reports every record trusted.
    {
        env e(1, 8);
        e.ring.put(0, 0, kRecStart, 1, db, 10);
        e.ring.put(0, 1, kRecEop, 1, db, 20);
        e.ring.put(0, 2, kRecEop, 2, db, 30);  // shape ii: no start
        e.ring.wptr[0] = 3;                    // w2 = 3 < slots -> nothing untrusted
        EXPECT_EQ(e.drain(), 1u);
        EXPECT_EQ(e.rec.records, 2u);
        EXPECT_EQ(e.rec.pairs.size(), 1u);
        EXPECT_EQ(e.rec.eops_without_start.size(), 1u);
        EXPECT_EQ(e.st.cursors.overruns, 0u);
        EXPECT_EQ(e.st.cursors.untrusted_records, 0u) << "loss verdict lives on the copy side";
    }
    // Exactly-full: NOT an overrun, but the oldest copied index aliases
    // the producer's next write target, so it is untrusted and dropped.
    {
        env e(1, 8);
        e.ring.put(0, 0, kRecStart, 1, 4100, 10);
        e.ring.put(0, 1, kRecEop, 1, 4100, 20);
        e.ring.wptr[0] = 8;  // == region_slots
        e.drain();
        EXPECT_EQ(e.st.cursors.overruns, 0u) << "exactly-full is not an overrun";
        EXPECT_EQ(e.st.cursors.lost_records, 0u);
        EXPECT_EQ(e.st.cursors.untrusted_records, 1u);
        EXPECT_TRUE(e.rec.pairs.empty()) << "the torn START is dropped";
        EXPECT_EQ(e.rec.eops_without_start.size(), 1u) << "its EOP arrives start-unknown";
    }
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

// the loss verdict is PER RECORD (the w2 back-patch), not region-wide --
// only the indices the producer's next write target aliases are untrusted.
TEST(dlog_drain, untrusted_verdict_is_per_record_not_region_wide)
{
    gpu_ring g(8);
    for(uint32_t i = 0; i < 8; ++i)
        g.ring.put(0, i, kRecStart, i + 1, 4100, 10 + i);
    g.ring.wptr[0] = 8;  // exactly full: only idx 0 aliases the next write target
    EXPECT_EQ(g.copy(), 8u);
    EXPECT_EQ(g.cur.overruns, 0u) << "exactly-full is not a lap";
    EXPECT_EQ(g.cur.untrusted_records, 1u);
    EXPECT_FALSE(g.batch[0].loss_free) << "boundary record untrusted";
    for(size_t i = 1; i < g.batch.size(); ++i)
        EXPECT_TRUE(g.batch[i].loss_free) << "non-boundary records trusted (not region-wide)";
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
    EXPECT_EQ(g.ring.rptr[0], 2u);  // freed before anything was paired
    EXPECT_EQ(g.ring.rptr[1], 1u);
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
        cr.loss_free        = true;
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
    EXPECT_EQ(st.evict_stale(/*now_ns=*/2500, /*max_age_ns=*/2000), 1u);
    EXPECT_TRUE(st.pending_starts.empty());
}

// Per-GPU isolation: shared doorbell+id, and an overrun on one.
TEST(dlog_drain, per_gpu_isolation)
{
    const uint32_t db = 4100;
    // Two rings, SAME doorbell+id: each keeps its own state and timestamps.
    {
        gpu_ring a(2048), b(2048);
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
        EXPECT_EQ(pair_records(a.batch.data(), a.batch.size(), pair_a, 1000, rec_a.on_record()),
                  1u);
        EXPECT_EQ(pair_records(b.batch.data(), b.batch.size(), pair_b, 1000, rec_b.on_record()),
                  1u);
        ASSERT_EQ(rec_a.pairs.count(std::make_pair(db, 5u)), 1u);
        ASSERT_EQ(rec_b.pairs.count(std::make_pair(db, 5u)), 1u);
        EXPECT_EQ(rec_a.pairs[std::make_pair(db, 5u)].first, 100u);
        EXPECT_EQ(rec_b.pairs[std::make_pair(db, 5u)].first, 300u);
        EXPECT_EQ(pair_a.unmatched_eops, 0u);
        EXPECT_EQ(pair_b.unmatched_eops, 0u);
    }
    // An overrun on A's ring must not touch B's cursors or verdicts.
    {
        gpu_ring a(8), b(8);
        a.ring.put(0, 0, kRecStart, 1, 4100, 10);
        a.ring.put(0, 1, kRecEop, 1, 4100, 20);
        a.ring.wptr[0] = 40;  // A laps
        b.ring.put(0, 0, kRecStart, 1, 4100, 30);
        b.ring.put(0, 1, kRecEop, 1, 4100, 40);
        b.ring.wptr[0] = 2;  // B clean
        a.copy();
        b.copy();
        EXPECT_EQ(a.cur.overruns, 1u);
        EXPECT_EQ(b.cur.overruns, 0u) << "one ring lapping must not mark another";
        EXPECT_GT(a.cur.lost_records, 0u);
        EXPECT_EQ(b.cur.lost_records, 0u);
        for(const auto& r : b.batch)
            EXPECT_TRUE(r.loss_free) << "the clean ring's records stay usable";
    }
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
//   records[buffer_size] | wptr[num_regions*8] | rptr[num_regions*8] | pad-to-page

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
    g.rptr_offset         = buffer_size + ptr_bytes;
    g.mmap_size           = round_up_to_page(g.rptr_offset + ptr_bytes, kPage);
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
    EXPECT_EQ(r.mmap_len, round_up_to_page(buf + 4 * 8, kPage));  // single page span
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
    mutate("wptr_rptr_not_disjoint", [](stream_geometry& g) { g.rptr_offset = buf + 4; });
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
