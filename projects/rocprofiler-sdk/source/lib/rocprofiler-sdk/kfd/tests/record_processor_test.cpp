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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// P4 unit tests for the dispatch-log processor side: copy-out into private
// backlogs, segment units, the service points, the per-stream end, the eviction
// watermark, and the exit invariant. No fd, no GPU: the sessions are production
// record_stream and the segments are built by hand.

#include "lib/rocprofiler-sdk/kfd/record_processor.hpp"
#include "lib/rocprofiler-sdk/kfd/dlog_drain.hpp"
#include "lib/rocprofiler-sdk/kfd/record_stream.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace
{
using namespace rocprofiler::kfd;

using stream_t = record_stream;

// Session-like for the processor helpers: .stream plus the .info fields
// dlog_spare_target reads. The 80 KiB minimum ring gives spare target 1 + 111.
struct fake_session
{
    struct info_t
    {
        uint32_t num_regions         = 1;
        uint32_t region_record_count = 4096;
        uint64_t buffer_size         = 81920;
    };
    info_t   info = {};
    stream_t stream{};
};

constexpr size_t kK = stream_t::kK;

fw_record
mk_eop(uint64_t ts, uint32_t id, uint32_t db = 4100)
{
    return fw_record{static_cast<uint32_t>(ts), static_cast<uint32_t>(ts >> 32), kRecEop, id, db};
}

fw_record
mk_start(uint64_t ts, uint32_t id, uint32_t db = 4100)
{
    return fw_record{static_cast<uint32_t>(ts), static_cast<uint32_t>(ts >> 32), kRecStart, id, db};
}

// Append one DATA segment (records then the marker) and publish it, exactly as
// the reader's pass does.
void
emit_segment(stream_t&                     st,
             reader_alloc_log&             log,
             const std::vector<fw_record>& recs,
             uint64_t                      epoch,
             uint64_t                      seen)
{
    st.append(recs.data(), recs.size(), log);
    st.append_marker({epoch, seen}, log);
    st.publish();
}

// Pairs and emits through one segment, the way processor_loop wires process_unit
// (the caller's EOP-buffer flush runs before publication).
void
pair_segment(stream_t&                                         stream,
             stream_backlog&                                   backlog,
             pair_state&                                       pairing,
             stream_segment&                                   seg,
             const std::function<void(const drained_record&)>& on_eop)
{
    process_unit(
        stream,
        backlog,
        pairing,
        seg,
        1000,
        [&](const drained_record& rec) { on_eop(rec); },
        []() {},
        []() {});
}

// --- delayed first observation ------------------------------------------------

TEST(record_processor, delayed_first_observation_processes_segments_in_order)
{
    std::array<fake_session, 1> ss{};
    ASSERT_TRUE(ss[0].stream.preallocate(4));
    reader_alloc_log log;

    // Three published segments before the processor's first look.
    const uint64_t seg_epoch[3] = {11, 12, 13};
    for(size_t seg = 0; seg < 3; ++seg)
    {
        const std::vector<fw_record> recs{mk_start(100 + seg, static_cast<uint32_t>(seg)),
                                          mk_eop(200 + seg, static_cast<uint32_t>(seg))};
        emit_segment(ss[0].stream, log, recs, seg_epoch[seg], 1000 + seg);
    }

    // First look: one copy-out closes all three segments at once.
    std::array<stream_backlog, 1> backlogs{};
    copy_out_all(ss, backlogs, 1);
    ASSERT_EQ(backlogs[0].closed.size(), 3u);

    // Each unit processes in publish order, under its own marker epoch. With
    // lookahead, a segment's own records are held until the NEXT segment, so
    // the first two pairs emit from units 2 and 3 and the third needs the
    // closing idle flush.
    auto                  pairing = pair_state{};
    std::vector<uint64_t> seen_epochs;
    std::vector<uint64_t> end_ticks;
    size_t                rr = 0;
    for(;;)
    {
        auto u = next_unit(backlogs, 1, rr);
        if(!u.found) break;
        seen_epochs.push_back(u.seg.epoch);
        pair_segment(ss[0].stream, backlogs[0], pairing, u.seg, [&](const drained_record& rec) {
            end_ticks.push_back(rec.end_ticks);
        });
    }
    EXPECT_EQ(seen_epochs, (std::vector<uint64_t>{11, 12, 13}));
    ASSERT_EQ(end_ticks.size(), 2u);
    EXPECT_EQ(end_ticks, (std::vector<uint64_t>{200, 201}));
    // The closing idle flush releases the last segment's held pair.
    stream_segment idle{};
    idle.epoch      = 14;
    idle.seen_at_ns = 4000;
    idle.end_pos    = ss[0].stream.taken();
    pair_segment(ss[0].stream, backlogs[0], pairing, idle, [&](const drained_record& rec) {
        end_ticks.push_back(rec.end_ticks);
    });
    ASSERT_EQ(end_ticks.size(), 3u);
    EXPECT_EQ(end_ticks, (std::vector<uint64_t>{200, 201, 202}));
    EXPECT_EQ(ss[0].stream.processed(), ss[0].stream.taken());
}

// --- cross-region regression in segment form -----------------------------------

TEST(record_processor, cross_region_eop_holds_until_earlier_start_is_copied)
{
    std::array<fake_session, 1> ss{};
    ASSERT_TRUE(ss[0].stream.preallocate(4));
    reader_alloc_log              log;
    std::array<stream_backlog, 1> backlogs{};

    // Segment 1 carries the EOP (tick 200); segment 2 carries an earlier-tick
    // START on the same key. Lookahead holds the EOP until the START arrives.
    const std::vector<fw_record> seg1{mk_eop(200, 7)};
    emit_segment(ss[0].stream, log, seg1, 1, 1000);
    const std::vector<fw_record> seg2{mk_start(100, 7)};
    emit_segment(ss[0].stream, log, seg2, 2, 2000);

    copy_out_all(ss, backlogs, 1);
    ASSERT_EQ(backlogs[0].closed.size(), 2u);

    auto                        pairing = pair_state{};
    std::vector<drained_record> emitted;
    size_t                      rr = 0;
    auto                        u  = next_unit(backlogs, 1, rr);
    ASSERT_TRUE(u.found);
    pair_segment(ss[0].stream, backlogs[0], pairing, u.seg, [&](const drained_record& rec) {
        emitted.push_back(rec);
    });
    EXPECT_TRUE(emitted.empty()) << "the EOP is held, not released start-unknown";
    ASSERT_EQ(pairing.deferred_records.size(), 1u);
    EXPECT_EQ(pairing.deferred_epoch, 1u);

    // Y (the earlier-tick START) is copied. It defers first, so the held EOP is
    // re-deferred rather than released: X is not let go before Y is copied.
    u = next_unit(backlogs, 1, rr);
    ASSERT_TRUE(u.found);
    pair_segment(ss[0].stream, backlogs[0], pairing, u.seg, [&](const drained_record& rec) {
        emitted.push_back(rec);
    });
    EXPECT_TRUE(emitted.empty()) << "the held EOP stays held with the earlier START";
    ASSERT_EQ(pairing.deferred_records.size(), 2u);
    EXPECT_EQ(pairing.deferred_epoch, 1u);

    // The closing idle flush makes both ready: the START is retained, the EOP
    // pairs with it.
    stream_segment idle{};
    idle.epoch      = 3;
    idle.seen_at_ns = 3000;
    idle.end_pos    = ss[0].stream.taken();
    pair_segment(ss[0].stream, backlogs[0], pairing, idle, [&](const drained_record& rec) {
        emitted.push_back(rec);
    });
    ASSERT_EQ(emitted.size(), 1u);
    EXPECT_TRUE(emitted[0].start_known);
    EXPECT_EQ(emitted[0].start_ticks, 100u) << "paired with the START copied after it";
    EXPECT_EQ(emitted[0].end_ticks, 200u);
    EXPECT_TRUE(pairing.deferred_records.empty());
}

// --- carried deferred keys preserve their epoch across units ------------------

TEST(record_processor, carried_deferred_key_is_redeferred_with_its_epoch)
{
    std::array<fake_session, 1> ss{};
    ASSERT_TRUE(ss[0].stream.preallocate(4));
    reader_alloc_log              log;
    std::array<stream_backlog, 1> backlogs{};

    // Segment 1 holds EOP(250). Segment 2 carries an earlier-tick START on the
    // same key, which defers first and re-defers the held EOP -- now held for a
    // second unit, still carrying its original epoch and copy time.
    const std::vector<fw_record> seg1{mk_eop(250, 7)};
    emit_segment(ss[0].stream, log, seg1, 1, 1000);
    const std::vector<fw_record> seg2{mk_start(200, 7)};
    emit_segment(ss[0].stream, log, seg2, 2, 2000);
    copy_out_all(ss, backlogs, 1);
    ASSERT_EQ(backlogs[0].closed.size(), 2u);

    auto   pairing = pair_state{};
    size_t rr      = 0;
    for(int i = 0; i < 2; ++i)
    {
        auto u = next_unit(backlogs, 1, rr);
        ASSERT_TRUE(u.found);
        pair_segment(ss[0].stream, backlogs[0], pairing, u.seg, [](const drained_record&) {});
    }
    ASSERT_EQ(pairing.deferred_records.size(), 2u) << "both records carried to the third unit";
    // The carried EOP keeps its original epoch and seen_at_ns.
    bool saw_eop = false;
    for(const auto& held : pairing.deferred_records)
        if(held.rec.record_type == kRecEop)
        {
            saw_eop = true;
            EXPECT_EQ(held.epoch, 1u);
            EXPECT_EQ(held.seen_at_ns, 1000u);
        }
    EXPECT_TRUE(saw_eop);
    EXPECT_EQ(pairing.deferred_epoch, 1u) << "the oldest held epoch is published";

    // The idle segment flushes both: the START is retained, the EOP pairs.
    stream_segment idle{};
    idle.epoch      = 3;
    idle.seen_at_ns = 3000;
    idle.end_pos    = ss[0].stream.taken();
    std::vector<drained_record> emitted;
    pair_segment(ss[0].stream, backlogs[0], pairing, idle, [&](const drained_record& rec) {
        emitted.push_back(rec);
    });
    ASSERT_EQ(emitted.size(), 1u);
    EXPECT_EQ(emitted[0].start_ticks, 200u);
    EXPECT_EQ(emitted[0].end_ticks, 250u);
    EXPECT_TRUE(pairing.deferred_records.empty());
}

// --- idle flush via an empty segment ------------------------------------------

TEST(record_processor, idle_flush_via_empty_segment_releases_held_eop_once)
{
    std::array<fake_session, 1> ss{};
    ASSERT_TRUE(ss[0].stream.preallocate(4));
    reader_alloc_log              log;
    std::array<stream_backlog, 1> backlogs{};

    // A segment holding one EOP, then an idle flush (empty segment), then a
    // second idle flush: the EOP is emitted start-unknown exactly once.
    const std::vector<fw_record> seg1{mk_eop(200, 7)};
    emit_segment(ss[0].stream, log, seg1, 1, 1000);
    ss[0].stream.append_marker({2, 2000}, log);
    ss[0].stream.publish();
    ss[0].stream.append_marker({3, 3000}, log);
    ss[0].stream.publish();
    copy_out_all(ss, backlogs, 1);
    ASSERT_EQ(backlogs[0].closed.size(), 3u);

    auto                        pairing = pair_state{};
    std::vector<drained_record> emitted;
    size_t                      rr = 0;
    for(int i = 0; i < 3; ++i)
    {
        auto u = next_unit(backlogs, 1, rr);
        ASSERT_TRUE(u.found);
        pair_segment(ss[0].stream, backlogs[0], pairing, u.seg, [&](const drained_record& rec) {
            emitted.push_back(rec);
        });
    }
    ASSERT_EQ(emitted.size(), 1u);
    EXPECT_FALSE(emitted[0].start_known);
    EXPECT_EQ(pairing.unmatched_eops, 1u);
    EXPECT_EQ(pairing.eops_seen, 1u);
    EXPECT_TRUE(pairing.deferred_records.empty());
    EXPECT_EQ(pairing.deferred_epoch, UINT64_MAX);
}

// --- re-entrancy: the EOP-flush service absorbs new runs ----------------------

TEST(record_processor, eop_callback_service_absorbs_new_runs_without_corruption)
{
    std::array<fake_session, 1> ss{};
    ASSERT_TRUE(ss[0].stream.preallocate(4));
    reader_alloc_log log;

    // Segment 1 holds 2048 EOPs (dispatch 0..2047, ticks 1000+); segment 2
    // carries 2048 more on higher keys (dispatch 3000+), so during segment 2's
    // unit all of segment 1's EOPs are ready and sort BEFORE segment 2's fresh
    // records -- on_record fires past the production 256-EOP flush threshold
    // while pair_records is still reading segment 2's fresh records.
    std::vector<fw_record> held_eops;
    for(uint32_t i = 0; i < 2048; ++i)
        held_eops.push_back(mk_eop(1000 + i, i));
    emit_segment(ss[0].stream, log, held_eops, 1, 5000);
    std::vector<fw_record> fresh_eops;
    for(uint32_t i = 0; i < 2048; ++i)
        fresh_eops.push_back(mk_eop(5000 + i, 3000 + i));
    emit_segment(ss[0].stream, log, fresh_eops, 2, 6000);

    std::array<stream_backlog, 1> backlogs{};
    copy_out_all(ss, backlogs, 1);
    ASSERT_EQ(backlogs[0].closed.size(), 2u);
    // Process segment 1 with the SAME pair_state segment 2's unit will use: its
    // records are all deferred (lookahead) and become segment 2's ready set.
    size_t rr = 0;
    auto   u0 = next_unit(backlogs, 1, rr);
    ASSERT_TRUE(u0.found);
    auto pairing = pair_state{};
    pair_segment(ss[0].stream, backlogs[0], pairing, u0.seg, [](const drained_record&) {});
    ASSERT_EQ(pairing.deferred_records.size(), 2048u);
    // Two more segments the reader publishes while segment 2 is being paired,
    // published but NOT copied out until the in-pairing service runs. TWO of
    // them matter: within one absorb, segment 3's data run fills the current
    // open vector and its marker reopens the backlog from the free list, so
    // segment 4's data run is the one that writes into the segment-under-test's
    // buffer once a mutant has recycled it before pairing.
    std::vector<fw_record> later;
    for(uint32_t i = 0; i < 300; ++i)
        later.push_back(mk_eop(9000 + i, 6000 + i));
    emit_segment(ss[0].stream, log, later, 3, 7000);
    std::vector<fw_record> later2;
    for(uint32_t i = 0; i < 400; ++i)
        later2.push_back(mk_eop(12000 + i, 8000 + i));
    emit_segment(ss[0].stream, log, later2, 4, 8000);

    // Segment 2's unit: the 2048 ready EOPs emit (start-unknown); like the
    // production wiring, the caller's flush runs from inside on_record at a
    // threshold and services (copy-out) before its hub resolution.
    auto seg = std::move(backlogs[0].closed.front());
    backlogs[0].closed.pop_front();
    std::vector<uint64_t> end_ticks;
    int                   services = 0;
    auto                  service  = [&]() {
        ++services;
        copy_out_all(ss, backlogs, 1);
    };
    process_unit(
        ss[0].stream,
        backlogs[0],
        pairing,
        seg,
        1000,
        [&](const drained_record& rec) {
            end_ticks.push_back(rec.end_ticks);
            if(end_ticks.size() % 1024 == 0) service();
        },
        []() {},
        []() {});
    EXPECT_EQ(services, 2) << "one service at each 1024-EOP boundary";
    // The pairs equal the pre-pairing snapshot: start-unknown EOPs in dispatch
    // order, each with its own tick.
    ASSERT_EQ(end_ticks.size(), 2048u);
    for(uint32_t i = 0; i < 2048; ++i)
        ASSERT_EQ(end_ticks[i], 1000u + i);
    // The held set equals the snapshot too: segment 2's fresh records carried
    // over uncorrupted (the service must not overwrite storage pair_records
    // re-reads).
    ASSERT_EQ(pairing.deferred_records.size(), 2048u);
    for(uint32_t i = 0; i < 2048; ++i)
        ASSERT_EQ(pairing.deferred_records[i].rec.ts_lo, 5000u + i);
    EXPECT_EQ(pairing.deferred_epoch, 2u);
    // The service absorbed the two later segments without corrupting anything:
    // segment 4's run landed in a free-list vector, never in the segment under
    // test.
    ASSERT_EQ(backlogs[0].closed.size(), 2u);
    EXPECT_EQ(backlogs[0].closed[0].records.size(), 300u);
    EXPECT_EQ(backlogs[0].closed[1].records.size(), 400u);
}

// --- shutdown drains everything, then the exit invariant holds -----------------

TEST(record_processor, shutdown_drains_every_segment_then_exit_invariant_holds)
{
    std::array<fake_session, 2> ss{};
    ASSERT_TRUE(ss[0].stream.preallocate(4));
    ASSERT_TRUE(ss[1].stream.preallocate(4));
    reader_alloc_log log;

    // Stream A: a complete segment and a final one whose records stay held.
    // Stream B: one segment whose records stay held. After reader_done, the
    // exit drain processes every segment; the per-GPU final flush then releases
    // what each GPU still holds.
    const std::vector<fw_record> a1{mk_start(10, 1), mk_eop(20, 1)};
    emit_segment(ss[0].stream, log, a1, 1, 1000);
    const std::vector<fw_record> a2{mk_start(100, 2), mk_eop(200, 2)};
    emit_segment(ss[0].stream, log, a2, 2, 2000);
    const std::vector<fw_record> b1{mk_eop(300, 3)};
    emit_segment(ss[1].stream, log, b1, 1, 1500);

    std::array<stream_backlog, 2> backlogs{};
    auto                          pair_a = pair_state{};
    auto                          pair_b = pair_state{};
    std::vector<uint64_t>         a_ends;
    auto                          process_one = [&](size_t slot, stream_segment& seg) {
        pair_segment(ss[slot].stream,
                     backlogs[slot],
                     slot == 0 ? pair_a : pair_b,
                     seg,
                     [&](const drained_record& rec) {
                         if(slot == 0) a_ends.push_back(rec.end_ticks);
                     });
    };
    drain_for_exit(ss, backlogs, 2, process_one);
    // Every published position is copied, paired and published; nothing queued.
    for(size_t slot = 0; slot < 2; ++slot)
    {
        EXPECT_EQ(ss[slot].stream.taken(), ss[slot].stream.published());
        EXPECT_EQ(ss[slot].stream.processed(), ss[slot].stream.taken());
        EXPECT_TRUE(backlogs[slot].closed.empty());
        EXPECT_TRUE(backlogs[slot].open.records.empty());
    }
    // A's first segment paired; its LAST segment's records are still held for
    // lookahead -- exactly what the final flush is for.
    EXPECT_EQ(a_ends, (std::vector<uint64_t>{20}));

    // Base per-GPU final flush: an empty segment with a fresh now_ns releases
    // whatever each GPU still holds for lookahead.
    stream_segment final_a{};
    final_a.seen_at_ns = 9000;
    final_a.end_pos    = ss[0].stream.taken();
    std::vector<drained_record> final_emitted;
    pair_segment(ss[0].stream, backlogs[0], pair_a, final_a, [&](const drained_record& rec) {
        final_emitted.push_back(rec);
    });
    stream_segment final_b{};
    final_b.seen_at_ns = 9000;
    final_b.end_pos    = ss[1].stream.taken();
    pair_segment(ss[1].stream, backlogs[1], pair_b, final_b, [&](const drained_record& rec) {
        final_emitted.push_back(rec);
    });
    // A's last segment paired (100, 200); B's lone EOP completed start-unknown.
    ASSERT_EQ(final_emitted.size(), 2u);
    EXPECT_TRUE(final_emitted[0].start_known);
    EXPECT_EQ(final_emitted[0].start_ticks, 100u);
    EXPECT_FALSE(final_emitted[1].start_known);
    EXPECT_TRUE(pair_a.deferred_records.empty());
    EXPECT_TRUE(pair_b.deferred_records.empty());
    // The exit invariant still holds after the final flush.
    for(size_t slot = 0; slot < 2; ++slot)
    {
        EXPECT_EQ(ss[slot].stream.processed(), ss[slot].stream.published());
        EXPECT_TRUE(backlogs[slot].closed.empty());
        EXPECT_TRUE(backlogs[slot].open.records.empty());
    }
}

// --- publication ordering: buffered EOPs resolve BEFORE anything is published --

// One held EOP is released by an idle-flush unit: on_record buffers exactly one
// end, below the 256 flush threshold. The publication sequence must resolve it
// BEFORE the deferred epoch and the processed position are published, so a
// FATAL fence that observes processed >= target can never pass while an EOP
// sits unresolved in the caller's ends buffer.
TEST(record_processor, publication_flushes_buffered_eops_before_publishing)
{
    std::array<fake_session, 1> ss{};
    ASSERT_TRUE(ss[0].stream.preallocate(4));
    reader_alloc_log              log;
    std::array<stream_backlog, 1> backlogs{};

    // Segment 1 holds one EOP; segment 2 is the idle flush that releases it.
    const std::vector<fw_record> seg1{mk_eop(200, 7)};
    emit_segment(ss[0].stream, log, seg1, 1, 1000);
    ss[0].stream.append_marker({2, 2000}, log);
    ss[0].stream.publish();
    copy_out_all(ss, backlogs, 1);
    ASSERT_EQ(backlogs[0].closed.size(), 2u);

    size_t rr = 0;
    auto   u0 = next_unit(backlogs, 1, rr);
    ASSERT_TRUE(u0.found);
    auto pairing = pair_state{};
    pair_segment(ss[0].stream, backlogs[0], pairing, u0.seg, [](const drained_record&) {});
    ASSERT_EQ(pairing.deferred_records.size(), 1u);

    auto seg = std::move(backlogs[0].closed.front());
    backlogs[0].closed.pop_front();

    // Wiring like processor_loop: ends buffered per record, flushed at 256 and
    // at publication; the "fence" observes the published state. Both hooks must
    // observe processed() still at its pre-unit position: nothing may publish
    // before the final flush resolves the buffered EOP.
    const uint64_t        processed_before = ss[0].stream.processed();
    std::vector<uint64_t> ends;
    int                   flushes               = 0;
    bool                  unresolved_at_publish = false;
    bool                  handoff_complete      = false;
    process_unit(
        ss[0].stream,
        backlogs[0],
        pairing,
        seg,
        1000,
        [&](const drained_record& rec) { ends.push_back(rec.end_ticks); },
        [&]() {  // final flush: the hub resolution
            EXPECT_EQ(ss[0].stream.processed(), processed_before)
                << "processed advanced before the final flush";
            if(!ends.empty())
            {
                ++flushes;
                ends.clear();
            }
        },
        [&]() {  // publication: the fence-observable instant
            EXPECT_EQ(ss[0].stream.processed(), processed_before)
                << "processed advanced before the deferred-epoch store";
            unresolved_at_publish = !ends.empty();
        });
    handoff_complete = ends.empty();
    EXPECT_FALSE(unresolved_at_publish)
        << "the epoch/processed position were published while an EOP was unresolved";
    EXPECT_TRUE(handoff_complete) << "handoff must be complete after the unit";
    EXPECT_EQ(flushes, 1);
    EXPECT_EQ(ss[0].stream.processed(), seg.end_pos) << "the unit's end_pos is published last";
}

// --- GC gate load discipline: the session count is read AFTER pass-open --------

// Minimal session fake: the gate reads the .stream's armed/taken/published only.
struct gate_stream_fake
{
    bool     armed_     = true;
    uint64_t taken_     = 0;
    uint64_t published_ = 0;
    bool     armed() const { return armed_; }
    uint64_t taken() const { return taken_; }
    uint64_t published() const { return published_; }
};
struct gate_session_fake
{
    gate_stream_fake stream;
};

// Scripted atomics that fire a hook after their load, so the test can place a
// session publication exactly between the gate's two loads. They also assert the
// memory orders the gate must request: seq_cst for pass-open, acquire for the
// session count.
struct scripted_pass_open
{
    std::atomic<bool>*    src = nullptr;
    std::function<void()> after_load;
    bool                  load(std::memory_order order) const
    {
        EXPECT_EQ(static_cast<int>(order), static_cast<int>(std::memory_order_seq_cst))
            << "reader_pass_open must be loaded seq_cst";
        const bool v = src->load(std::memory_order_seq_cst);
        if(after_load) after_load();
        return v;
    }
};
struct scripted_count
{
    std::atomic<size_t>* src = nullptr;
    size_t               load(std::memory_order order) const
    {
        EXPECT_EQ(static_cast<int>(order), static_cast<int>(std::memory_order_acquire))
            << "session_count must be loaded acquire";
        return src->load(std::memory_order_acquire);
    }
};

TEST(record_processor, gate_reads_pass_open_before_the_session_count)
{
    std::atomic<bool>                pass_open{false};
    std::atomic<size_t>              count{1};
    std::array<gate_session_fake, 2> ss{};
    std::array<stream_backlog, 2>    backlogs{};

    scripted_pass_open po{&pass_open, [&]() {
                              // Publication + a completed reader pass on the
                              // new stream land between the old count sample
                              // and the pass-open observation.
                              count.store(2, std::memory_order_release);
                              ss[1].stream.published_ = 3;  // taken stays 0
                          }};
    scripted_count cnt{&count};

    EXPECT_FALSE(gc_gate_open_pass(po, ss, backlogs, cnt))
        << "the new stream's uncopied publish must close the gate";
    // Sanity: with the new stream fully copied and decoded, the gate opens.
    ss[1].stream.taken_ = 3;
    EXPECT_TRUE(gc_gate_open_pass(po, ss, backlogs, cnt));
}

// --- a stream's idle flush releases only its own held set ----------------------

TEST(record_processor, idle_flush_releases_only_its_own_stream_held_set)
{
    std::array<fake_session, 2> ss{};
    ASSERT_TRUE(ss[0].stream.preallocate(4));
    ASSERT_TRUE(ss[1].stream.preallocate(4));
    reader_alloc_log              log;
    std::array<stream_backlog, 2> backlogs{};

    // Both streams hold one EOP after their first segment.
    const std::vector<fw_record> a1{mk_eop(200, 1)};
    emit_segment(ss[0].stream, log, a1, 1, 1000);
    const std::vector<fw_record> b1{mk_eop(250, 2)};
    emit_segment(ss[1].stream, log, b1, 1, 1100);
    copy_out_all(ss, backlogs, 2);
    auto   pair_a = pair_state{};
    auto   pair_b = pair_state{};
    size_t rr     = 0;
    for(int i = 0; i < 2; ++i)
    {
        auto u = next_unit(backlogs, 2, rr);
        ASSERT_TRUE(u.found);
        pair_segment(ss[u.slot].stream,
                     backlogs[u.slot],
                     u.slot == 0 ? pair_a : pair_b,
                     u.seg,
                     [](const drained_record&) {});
    }
    ASSERT_EQ(pair_a.deferred_records.size(), 1u);
    ASSERT_EQ(pair_b.deferred_records.size(), 1u);

    // Stream A's idle flush: only A's held set is released. B's records stay
    // held, still carrying their epoch, until B's next segment.
    const std::vector<fw_record> a_idle{};
    emit_segment(ss[0].stream, log, a_idle, 2, 2000);
    copy_out_all(ss, backlogs, 2);
    std::vector<size_t> held_at_publish;
    for(;;)
    {
        auto u = next_unit(backlogs, 2, rr);
        if(!u.found) break;
        auto& pairing = u.slot == 0 ? pair_a : pair_b;
        process_unit(
            ss[u.slot].stream,
            backlogs[u.slot],
            pairing,
            u.seg,
            1000,
            [](const drained_record&) {},
            []() {},
            [&]() { held_at_publish.push_back(pair_a.deferred_records.size()); });
    }
    EXPECT_TRUE(pair_a.deferred_records.empty()) << "A's idle flush released A's held set";
    ASSERT_EQ(pair_b.deferred_records.size(), 1u) << "B's held records stay held";
    EXPECT_EQ(pair_b.deferred_epoch, 1u);
    // The epoch store observes the post-pairing held set.
    ASSERT_EQ(held_at_publish.size(), 1u);
    EXPECT_EQ(held_at_publish[0], 0u) << "published after pairing, not before";
}

// --- the eviction watermark is the segment's seen_at_ns ------------------------

TEST(record_processor, eviction_watermark_is_segment_seen_at_ns)
{
    std::array<fake_session, 1> ss{};
    ASSERT_TRUE(ss[0].stream.preallocate(4));
    std::array<stream_backlog, 1> backlogs{};
    auto                          pairing = pair_state{};
    // A retained START recorded at 1000 ns; age bound 2000 ns.
    pairing.record_start((uint64_t{4100} << 32) | 7, 100, 1000);

    // Segment 1 at seen_at_ns = 2000: the 1 s cadence has not passed, so the
    // eviction must not even attempt a pass over the ages.
    stream_segment seg1{};
    seg1.seen_at_ns = 2000;
    seg1.end_pos    = ss[0].stream.taken();
    pair_segment(ss[0].stream, backlogs[0], pairing, seg1, [](const drained_record&) {});
    EXPECT_EQ(pairing.starts_evicted, 0u)
        << "the watermark is the segment's copy time, never the wall clock";
    EXPECT_EQ(pairing.last_evict_ns, 0u);

    // Segment 2 past the cadence: the eviction runs with seg.seen_at_ns as the
    // watermark and ages out the retained START.
    stream_segment seg2{};
    seg2.seen_at_ns = kProcessorEvictIntervalNs + 1000;
    seg2.end_pos    = ss[0].stream.taken();
    pair_segment(ss[0].stream, backlogs[0], pairing, seg2, [](const drained_record&) {});
    EXPECT_EQ(pairing.starts_evicted, 1u);
    EXPECT_EQ(pairing.last_evict_ns, kProcessorEvictIntervalNs + 1000);
    EXPECT_TRUE(pairing.pending_starts.empty());
}

}  // namespace
