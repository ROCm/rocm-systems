// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for sample_ring_buffer<N> — Phase B data structures.

#include <gtest/gtest.h>

#include "sampling/data/backtrace_record.hpp"
#include "sampling/src/sample_ring_buffer.hpp"

using namespace rocprofsys::sampling;

TEST(sample_ring_buffer, try_push_and_pop_single_record)
{
    sample_ring_buffer<4> ring;

    backtrace_record rec;
    rec.tid          = 42;
    rec.timestamp_ns = 12345;
    rec.trigger      = trigger_type::TIMER;

    EXPECT_TRUE(ring.try_push(rec));

    auto out_opt = ring.pop();
    ASSERT_TRUE(out_opt.has_value());
    EXPECT_EQ(out_opt->tid, 42);
    EXPECT_EQ(out_opt->timestamp_ns, 12345U);
}

TEST(sample_ring_buffer, full_buffer_drops_on_try_push)
{
    sample_ring_buffer<2> ring;

    backtrace_record rec;
    rec.tid = 1;
    EXPECT_TRUE(ring.try_push(rec));
    rec.tid = 2;
    EXPECT_TRUE(ring.try_push(rec));
    // Buffer is now full (capacity=2)
    rec.tid = 3;
    EXPECT_FALSE(ring.try_push(rec)) << "try_push must return false when ring is full";
}

TEST(sample_ring_buffer, is_empty_after_construction)
{
    sample_ring_buffer<8> ring;
    EXPECT_EQ(ring.count(), 0U);
}

TEST(sample_ring_buffer, count_tracks_entries)
{
    sample_ring_buffer<4> ring;
    EXPECT_EQ(ring.count(), 0U);

    backtrace_record rec;
    ring.try_push(rec);
    EXPECT_EQ(ring.count(), 1U);

    ring.try_push(rec);
    EXPECT_EQ(ring.count(), 2U);
}

TEST(sample_ring_buffer, trivially_copyable_record)
{
    static_assert(std::is_trivially_copyable_v<backtrace_record>,
                  "backtrace_record must be trivially copyable per architecture.md § 3");
}

TEST(sample_ring_buffer, pop_returns_records_in_fifo_order)
{
    sample_ring_buffer<4> ring;

    for(int64_t i = 0; i < 3; ++i)
    {
        backtrace_record rec;
        rec.tid = i;
        ring.try_push(rec);
    }

    for(int64_t i = 0; i < 3; ++i)
    {
        auto out_opt = ring.pop();
        ASSERT_TRUE(out_opt.has_value()) << "Ring should have entry at position " << i;
        EXPECT_EQ(out_opt->tid, i) << "FIFO order violated at position " << i;
    }
}

// ─── Edge: max-depth pc_count (64 PCs) stored and retrieved intact ────────────

TEST(sample_ring_buffer, max_depth_record_stored_and_retrieved_intact)
{
    sample_ring_buffer<4> ring;

    backtrace_record rec;
    rec.tid      = 99;
    rec.pc_count = 64;
    for(size_t idx = 0; idx < 64; ++idx)
        rec.raw_pcs.at(idx) = static_cast<uintptr_t>(0x1000 + idx);

    ASSERT_TRUE(ring.try_push(rec));

    auto out_opt = ring.pop();
    ASSERT_TRUE(out_opt.has_value());
    EXPECT_EQ(out_opt->pc_count, 64);
    for(size_t idx = 0; idx < 64; ++idx)
    {
        EXPECT_EQ(out_opt->raw_pcs.at(idx), static_cast<uintptr_t>(0x1000 + idx))
            << "PC at index " << idx << " corrupted";
    }
}

// ─── C-8: dropped_count() tracks failed try_push calls ───────────────────────

TEST(sample_ring_buffer, dropped_count_starts_at_zero)
{
    sample_ring_buffer<4> ring;
    EXPECT_EQ(ring.dropped_count(), 0U);
}

TEST(sample_ring_buffer, dropped_count_increments_when_full)
{
    sample_ring_buffer<2> ring;

    backtrace_record rec;
    EXPECT_TRUE(ring.try_push(rec));
    EXPECT_TRUE(ring.try_push(rec));  // ring full

    EXPECT_EQ(ring.dropped_count(), 0U) << "No drops yet — pushes succeeded";

    EXPECT_FALSE(ring.try_push(rec));
    EXPECT_EQ(ring.dropped_count(), 1U)
        << "First dropped push must increment dropped_count";

    EXPECT_FALSE(ring.try_push(rec));
    EXPECT_EQ(ring.dropped_count(), 2U)
        << "Second dropped push must increment dropped_count";
}

TEST(sample_ring_buffer, dropped_count_does_not_decrease_after_pop)
{
    sample_ring_buffer<1> ring;

    backtrace_record rec;
    EXPECT_TRUE(ring.try_push(rec));   // full
    EXPECT_FALSE(ring.try_push(rec));  // dropped
    EXPECT_EQ(ring.dropped_count(), 1U);

    [[maybe_unused]] auto _ = ring.pop();
    EXPECT_EQ(ring.dropped_count(), 1U) << "Pop must not decrease dropped_count";
}

// ─── Edge: try_push return value tracks drops correctly ───────────────────────

TEST(sample_ring_buffer, drop_count_tracked_via_try_push_return_value)
{
    sample_ring_buffer<2> ring;

    backtrace_record rec;
    rec.tid = 1;
    EXPECT_TRUE(ring.try_push(rec));  // slot 0 used
    rec.tid = 2;
    EXPECT_TRUE(ring.try_push(rec));  // slot 1 used, ring full
    rec.tid = 3;
    EXPECT_FALSE(ring.try_push(rec))  // dropped
        << "Third push into capacity-2 ring must return false";
    rec.tid = 4;
    EXPECT_FALSE(ring.try_push(rec))  // still full
        << "Fourth push into capacity-2 ring must return false";

    // After popping one, there is room again.
    [[maybe_unused]] auto discarded = ring.pop();
    rec.tid                         = 5;
    EXPECT_TRUE(ring.try_push(rec))
        << "After one pop, ring has room and try_push must return true";
}
