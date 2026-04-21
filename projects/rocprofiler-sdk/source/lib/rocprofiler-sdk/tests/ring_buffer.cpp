// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/hsa/ring_buffer.hpp"

#include <gtest/gtest.h>
#include <hsa/hsa.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
namespace
{
constexpr uint32_t kPktSize = 64;

// Helper: construct a RingView with a power-of-two number of 64B slots.
RingView
make_view(std::vector<char>& backing, uint32_t slot_count)
{
    backing.assign(slot_count * kPktSize, 0);
    RingView v;
    v.buf      = backing.data();
    v.size     = slot_count;
    v.mask     = slot_count - 1;
    v.pkt_size = kPktSize;
    return v;
}

TEST(RingBuffer, RingViewMaskIsSizeMinusOne)
{
    std::vector<char> backing;
    RingView          v = make_view(backing, 4096);
    EXPECT_EQ(v.size, 4096u);
    EXPECT_EQ(v.mask, 4095u);
    EXPECT_EQ(v.pkt_size, kPktSize);
}

TEST(RingBuffer, RingCursorWriteAdvancesSubmitPos)
{
    std::vector<char> backing;
    RingView          v    = make_view(backing, 8);
    uint64_t          rdid = 0;
    RingCursor        cursor{v, 0, &rdid};

    alignas(64) char pkt[kPktSize];
    memset(pkt, 0xAB, kPktSize);

    cursor.write(pkt);
    cursor.write(pkt);
    cursor.write(pkt);

    EXPECT_EQ(cursor.submit_pos(), 3u);
}

TEST(RingBuffer, RingCursorWrapAround)
{
    std::vector<char> backing;
    RingView          v    = make_view(backing, 4);  // mask = 3
    uint64_t          rdid = 0;
    RingCursor        cursor{v, 0, &rdid};

    // Keep rdid moving so we don't stall against backpressure when wrapping.
    // (After we've written 4, rdid must be >= 1 to allow the 5th write.)
    alignas(64) char pkts[6][kPktSize];
    for(int i = 0; i < 6; ++i)
    {
        memset(pkts[i], 0, kPktSize);
        // Encode payload so we can check which slot received which write.
        pkts[i][2] = static_cast<char>(0x40 + i);
    }

    // Writes 0..3 fill the ring (rdid=0, submit_pos - rdid < 4 OK for pos 0..3).
    for(int i = 0; i < 4; ++i)
    {
        cursor.write(pkts[i]);
    }
    // Before writes 4 and 5, bump rdid so they don't stall.
    __atomic_store_n(&rdid, 4, __ATOMIC_RELEASE);
    cursor.write(pkts[4]);
    cursor.write(pkts[5]);

    EXPECT_EQ(cursor.submit_pos(), 6u);

    auto slot_byte = [&](uint64_t slot_idx) -> char { return backing[slot_idx * kPktSize + 2]; };
    // Slots 0,1,2,3 overwritten by pkts 4,5,_,_ ? No: wrap order is
    // logical 0..5 -> slot (0&3,1&3,2&3,3&3,4&3,5&3) = (0,1,2,3,0,1)
    EXPECT_EQ(slot_byte(0), static_cast<char>(0x40 + 4));  // slot 0 last written by pkt 4
    EXPECT_EQ(slot_byte(1), static_cast<char>(0x40 + 5));  // slot 1 last written by pkt 5
    EXPECT_EQ(slot_byte(2), static_cast<char>(0x40 + 2));  // slot 2 untouched since pkt 2
    EXPECT_EQ(slot_byte(3), static_cast<char>(0x40 + 3));  // slot 3 untouched since pkt 3
}

TEST(RingBuffer, RingCursorBackpressure)
{
    std::vector<char> backing;
    RingView          v    = make_view(backing, 4);
    uint64_t          rdid = 0;

    // Pre-fill the ring to the brink: submit_pos starts at 4, rdid at 0,
    // so the very next write() must block until rdid advances.
    RingCursor cursor{v, 4, &rdid};

    std::atomic<bool> write_returned{false};
    auto              t_start = std::chrono::steady_clock::now();

    std::thread writer([&] {
        alignas(64) char pkt[kPktSize];
        memset(pkt, 0, kPktSize);
        cursor.write(pkt);
        write_returned.store(true, std::memory_order_release);
    });

    // Let the writer run first and confirm it's stuck.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_FALSE(write_returned.load(std::memory_order_acquire));

    // Release one slot; writer should now proceed.
    __atomic_store_n(&rdid, 1, __ATOMIC_RELEASE);
    writer.join();

    EXPECT_TRUE(write_returned.load(std::memory_order_acquire));
    EXPECT_EQ(cursor.submit_pos(), 5u);

    // Sanity: total wait was at least the 10ms sleep.
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_start);
    EXPECT_GE(elapsed.count(), 10);
}

TEST(RingBuffer, RingCursorWriteGapProducesBarrierAndHeader)
{
    std::vector<char> backing;
    RingView          v    = make_view(backing, 4);
    uint64_t          rdid = 0;
    RingCursor        cursor{v, 0, &rdid};

    // Pre-dirty the target slot so we can prove write_gap zeroes the body.
    memset(backing.data(), 0xFF, kPktSize);

    cursor.write_gap();
    EXPECT_EQ(cursor.submit_pos(), 1u);

    // Header (first 2 bytes): type == HSA_PACKET_TYPE_BARRIER_AND.
    // The header store uses __ATOMIC_RELEASE so the CP never observes a
    // zeroed type with a dirty body. We can't observe the memory-order in
    // a unit test directly; see the comment in RingCursor::write_gap.
    uint16_t header = 0;
    memcpy(&header, backing.data(), sizeof(header));
    EXPECT_EQ(header, static_cast<uint16_t>(HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE));

    // Remaining 62 bytes are zeroed.
    for(uint32_t i = 2; i < kPktSize; ++i)
    {
        EXPECT_EQ(backing[i], 0) << "byte " << i << " not zero";
    }
}

TEST(RingBuffer, RingCursorReservationRefusesOverAdvance)
{
    std::vector<char> backing;
    RingView          v    = make_view(backing, 16);
    uint64_t          rdid = 0;
    RingCursor        cursor{v, 0, &rdid};
    cursor.set_reservation_end(cursor.submit_pos() + 3);

    alignas(64) char pkts[4][kPktSize];
    for(int i = 0; i < 4; ++i)
    {
        memset(pkts[i], 0, kPktSize);
        pkts[i][2] = static_cast<char>(0x50 + i);
    }

    EXPECT_TRUE(cursor.write(pkts[0]));
    EXPECT_TRUE(cursor.write(pkts[1]));
    EXPECT_TRUE(cursor.write(pkts[2]));
    EXPECT_FALSE(cursor.write(pkts[3]));

    // 4th write must not advance submit_pos.
    EXPECT_EQ(cursor.submit_pos(), 3u);

    // First 3 slots carry the 3 written payloads; slot 3 must still be zero
    // since the 4th write was refused.
    EXPECT_EQ(backing[0 * kPktSize + 2], static_cast<char>(0x50 + 0));
    EXPECT_EQ(backing[1 * kPktSize + 2], static_cast<char>(0x50 + 1));
    EXPECT_EQ(backing[2 * kPktSize + 2], static_cast<char>(0x50 + 2));
    EXPECT_EQ(backing[3 * kPktSize + 2], 0);
}

TEST(RingBuffer, RingCursorPadToReservationWritesGaps)
{
    std::vector<char> backing;
    RingView          v    = make_view(backing, 8);
    uint64_t          rdid = 0;
    RingCursor        cursor{v, 0, &rdid};
    cursor.set_reservation_end(5);

    alignas(64) char pkt[kPktSize];
    memset(pkt, 0xAB, kPktSize);
    EXPECT_TRUE(cursor.write(pkt));
    EXPECT_TRUE(cursor.write(pkt));

    // Pre-dirty slots 2..4 so pad_to_reservation has to zero them.
    memset(backing.data() + 2 * kPktSize, 0xFF, 3 * kPktSize);

    cursor.pad_to_reservation();
    EXPECT_EQ(cursor.submit_pos(), 5u);

    const uint16_t expected_header =
        static_cast<uint16_t>(HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE);
    for(uint32_t slot = 2; slot < 5; ++slot)
    {
        // Header bytes 0,1 == 0x03, 0x00 (little-endian BARRIER_AND<<HEADER_TYPE).
        EXPECT_EQ(static_cast<unsigned char>(backing[slot * kPktSize + 0]), 0x03u)
            << "slot " << slot << " header byte 0";
        EXPECT_EQ(static_cast<unsigned char>(backing[slot * kPktSize + 1]), 0x00u)
            << "slot " << slot << " header byte 1";

        uint16_t header = 0;
        memcpy(&header, backing.data() + slot * kPktSize, sizeof(header));
        EXPECT_EQ(header, expected_header);

        // Remaining 62 bytes per slot must be zero.
        for(uint32_t i = 2; i < kPktSize; ++i)
        {
            EXPECT_EQ(backing[slot * kPktSize + i], 0)
                << "slot " << slot << " byte " << i << " not zero";
        }
    }
}

TEST(RingBuffer, RingCursorNoReservationIsUnbounded)
{
    std::vector<char> backing;
    RingView          v    = make_view(backing, 16);
    uint64_t          rdid = 0;
    RingCursor        cursor{v, 0, &rdid};

    // Keep rdid abreast so backpressure never trips; reservation is the
    // default UINT64_MAX, so write() must never refuse.
    alignas(64) char pkt[kPktSize];
    memset(pkt, 0, kPktSize);

    for(int i = 0; i < 10; ++i)
    {
        __atomic_store_n(&rdid, static_cast<uint64_t>(i), __ATOMIC_RELEASE);
        EXPECT_TRUE(cursor.write(pkt)) << "write " << i << " refused";
    }
    EXPECT_EQ(cursor.submit_pos(), 10u);
}

TEST(RingBuffer, RingCursorPadToReservationNoopIfAlreadyThere)
{
    std::vector<char> backing;
    RingView          v    = make_view(backing, 8);
    uint64_t          rdid = 0;
    RingCursor        cursor{v, 5, &rdid};
    cursor.set_reservation_end(5);

    // Pre-dirty the slot that submit_pos==5 would map to so we can detect
    // any spurious write.
    memset(backing.data() + (5 & 7) * kPktSize, 0xFF, kPktSize);

    cursor.pad_to_reservation();
    EXPECT_EQ(cursor.submit_pos(), 5u);

    // Slot untouched.
    for(uint32_t i = 0; i < kPktSize; ++i)
    {
        EXPECT_EQ(static_cast<unsigned char>(backing[(5 & 7) * kPktSize + i]), 0xFFu)
            << "byte " << i << " modified by no-op pad";
    }
}

TEST(RingBuffer, RingViewReadSlotReturnsCorrectAddress)
{
    std::vector<char> backing;
    RingView          v = make_view(backing, 8);

    // Write a recognizable pattern at slot 3 and read it through the view.
    backing[3 * kPktSize]     = 0x11;
    backing[3 * kPktSize + 1] = 0x22;

    // N & mask: read_slot(3) -> slot 3; read_slot(11) -> slot 3 too.
    const char* s3  = static_cast<const char*>(v.read_slot(3));
    const char* s11 = static_cast<const char*>(v.read_slot(11));
    EXPECT_EQ(s3, backing.data() + 3 * kPktSize);
    EXPECT_EQ(s11, s3);
    EXPECT_EQ(static_cast<unsigned char>(s3[0]), 0x11u);
    EXPECT_EQ(static_cast<unsigned char>(s3[1]), 0x22u);

    // Arbitrary N: N & mask indexing.
    const char* s17 = static_cast<const char*>(v.read_slot(17));
    EXPECT_EQ(s17, backing.data() + (17 & 7) * kPktSize);
}

}  // namespace
}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
