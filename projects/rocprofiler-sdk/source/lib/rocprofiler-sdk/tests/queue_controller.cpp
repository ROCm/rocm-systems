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

// Unit tests for the KFD dispatch-log capture seam in queue_controller.cpp.
// capture_doorbell_key() decodes an HSA-internal amd_signal_t, so these tests
// hand it hand-built signals rather than a live queue.

#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kfd/doorbell_map.hpp"

#include <gtest/gtest.h>

#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>

#include <cstdint>

#include <unistd.h>

namespace rocprofiler
{
namespace hsa
{
namespace
{
hsa_queue_t
make_intercept_queue(const amd_signal_t* signal)
{
    auto q            = hsa_queue_t{};
    q.doorbell_signal = hsa_signal_t{.handle = reinterpret_cast<uint64_t>(signal)};
    return q;
}

// Any GPU id; capture only threads it through onto the entry.
constexpr uint32_t kTestGpuId = 25567;
}  // namespace

// A legacy intercept queue exposes a USER-kind signal, whose `value` member
// aliases hardware_doorbell_ptr. Trusting it would yield a bogus correlation
// key, so capture must reject any non-doorbell kind (-> HSA fallback).
TEST(queue_controller, capture_doorbell_key_rejects_non_doorbell_signal)
{
    auto signal  = amd_signal_t{};
    signal.kind  = AMD_SIGNAL_KIND_USER;
    signal.value = 0x7f0000004010;  // aliases hardware_doorbell_ptr
    auto queue   = make_intercept_queue(&signal);

    EXPECT_FALSE(
        capture_doorbell_key(kTestGpuId, rocprofiler_queue_id_t{1}, &queue).has_value());
}

// A doorbell-kind signal yields the page-relative slot of its hardware doorbell
// pointer -- the same value the reader derives from a firmware record.
TEST(queue_controller, capture_doorbell_key_accepts_doorbell_signal)
{
    volatile uint64_t doorbell = 0;

    auto signal                  = amd_signal_t{};
    signal.kind                  = AMD_SIGNAL_KIND_DOORBELL;
    signal.hardware_doorbell_ptr = &doorbell;
    auto queue                   = make_intercept_queue(&signal);

    auto entry = capture_doorbell_key(kTestGpuId, rocprofiler_queue_id_t{2}, &queue);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->doorbell_off,
              kfd::doorbell_ptr_to_page_slot(reinterpret_cast<uint64_t>(&doorbell),
                                             static_cast<uint64_t>(sysconf(_SC_PAGESIZE))));
    EXPECT_EQ(entry->generation, 0u);
    // The entry carries the GPU it was captured for: doorbell slots repeat across
    // GPUs, so the key is only unambiguous with it.
    EXPECT_EQ(entry->gpu_id, kTestGpuId);
}

// No intercept queue and a null doorbell signal are both "unavailable" -> no key.
TEST(queue_controller, capture_doorbell_key_rejects_missing_doorbell)
{
    EXPECT_FALSE(
        capture_doorbell_key(kTestGpuId, rocprofiler_queue_id_t{3}, nullptr).has_value());

    auto queue = make_intercept_queue(nullptr);
    EXPECT_FALSE(
        capture_doorbell_key(kTestGpuId, rocprofiler_queue_id_t{3}, &queue).has_value());
}
// The inline path's stand-in for "_active_kernels == 0": the hardware has
// consumed every packet we submitted. A closing queue must wait for this before
// anything decides its dispatches produced no record -- until the read index
// catches up, their kernels are still running and their EOP records do not exist.
TEST(queue_interposition, hw_queue_drained_predicate)
{
    using queue_interposition::hw_queue_drained;

    // Nothing submitted: trivially drained.
    EXPECT_TRUE(hw_queue_drained(/*real_rdid=*/0, /*next_submit_pos=*/0));

    // Packets outstanding: NOT drained -- this is the case the old code raced.
    EXPECT_FALSE(hw_queue_drained(0, 4));
    EXPECT_FALSE(hw_queue_drained(3, 4));

    // Read index caught up, or ran past (the CP consumed packets we did not
    // submit through the interposed path): drained.
    EXPECT_TRUE(hw_queue_drained(4, 4));
    EXPECT_TRUE(hw_queue_drained(5, 4));

    // Large indices: the comparison must not be done by subtraction, which would
    // wrap. These are monotonically increasing 64-bit counters.
    constexpr uint64_t big = uint64_t{1} << 62;
    EXPECT_TRUE(hw_queue_drained(big + 1, big));
    EXPECT_FALSE(hw_queue_drained(big, big + 1));
}
}  // namespace hsa
}  // namespace rocprofiler
