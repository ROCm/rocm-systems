// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for offload round-trip — AC-15, NFR-P-1.
// Uses in_memory_offload (no /tmp I/O).
// Binary baseline fixture test is BLOCKED on Stage 1.5 artifact.

#include <gtest/gtest.h>

#include "doubles/in_memory_offload.hpp"
#include "sampling/data/backtrace_record.hpp"
#include "sampling/src/sample_ring_buffer.hpp"

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;

// Noop fatal policy for existing write()-call tests that don't exercise failure paths.
struct noop_fatal_policy
{
    template <class... Args>
    [[noreturn]] void fatal(char const*, int, std::string_view, Args const&...) noexcept
    {
        std::abort();
    }
};
static noop_fatal_policy g_noop_fatal;

// ─── AC-15: in_memory_offload round-trip ────────────────────────────────────────

TEST(offload, in_memory_round_trip_produces_same_record_sequence)
{
    in_memory_offload offload;

    const int64_t tid = 7;

    // Build a ring buffer with 3 records.
    sample_ring_buffer<8> ring;
    for(int64_t i = 0; i < 3; ++i)
    {
        backtrace_record rec;
        rec.tid          = tid;
        rec.timestamp_ns = static_cast<uint64_t>(i * 1000);
        rec.pc_count     = 2;
        rec.raw_pcs[0]   = 0xDEAD + static_cast<uintptr_t>(i);
        rec.raw_pcs[1]   = 0xBEEF + static_cast<uintptr_t>(i);
        ring.try_push(rec);
    }

    offload.write<8>(tid, ring, g_noop_fatal);
    auto records = offload.read(tid);

    ASSERT_EQ(records.size(), 3U);
    for(size_t idx = 0; idx < 3; ++idx)
    {
        EXPECT_EQ(records.at(idx).tid, tid) << "tid mismatch at index " << idx;
        EXPECT_EQ(records.at(idx).timestamp_ns, static_cast<uint64_t>(idx * 1000))
            << "timestamp mismatch at index " << idx;
        EXPECT_EQ(records.at(idx).raw_pcs[0], static_cast<uintptr_t>(0xDEAD + idx))
            << "pc[0] mismatch at index " << idx;
    }
}

TEST(offload, empty_ring_produces_empty_read)
{
    in_memory_offload     offload;
    sample_ring_buffer<8> ring;

    offload.write<8>(0, ring, g_noop_fatal);
    auto records = offload.read(0);

    EXPECT_TRUE(records.empty())
        << "Writing an empty ring must produce no records on read";
}

TEST(offload, read_unknown_tid_returns_empty)
{
    in_memory_offload offload;

    auto records = offload.read(99);
    EXPECT_TRUE(records.empty());
}

TEST(offload, reset_clears_all_tids)
{
    in_memory_offload     offload;
    sample_ring_buffer<4> ring;
    backtrace_record      rec;
    rec.tid = 1;
    ring.try_push(rec);
    offload.write<4>(1, ring, g_noop_fatal);

    offload.reset();
    EXPECT_TRUE(offload.read(1).empty()) << "reset() must clear all stored records";
}

// ─── AC-15: write preserves field ordering ────────────────────────────────────

TEST(offload, write_preserves_timestamp_ordering)
{
    in_memory_offload     offload;
    sample_ring_buffer<4> ring;

    backtrace_record r0;
    r0.tid          = 2;
    r0.timestamp_ns = 100U;
    r0.pc_count     = 1;
    r0.raw_pcs[0]   = 0x1111;

    backtrace_record r1;
    r1.tid          = 2;
    r1.timestamp_ns = 200U;
    r1.pc_count     = 1;
    r1.raw_pcs[0]   = 0x2222;

    ring.try_push(r0);
    ring.try_push(r1);
    offload.write<4>(2, ring, g_noop_fatal);

    auto records = offload.read(2);
    ASSERT_EQ(records.size(), 2U);
    EXPECT_LT(records.at(0).timestamp_ns, records.at(1).timestamp_ns)
        << "records must be returned in push order (ascending timestamp)";
    EXPECT_EQ(records.at(0).raw_pcs[0], static_cast<uintptr_t>(0x1111));
    EXPECT_EQ(records.at(1).raw_pcs[0], static_cast<uintptr_t>(0x2222));
}
