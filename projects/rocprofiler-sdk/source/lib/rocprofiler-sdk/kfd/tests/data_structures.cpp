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

#include "lib/rocprofiler-sdk/kfd/correlation_types.hpp"
#include "lib/rocprofiler-sdk/kfd/doorbell_map.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_correlation.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less_gate.hpp"

#include <gtest/gtest.h>

namespace
{
using namespace rocprofiler::kfd;
rocprofiler_queue_id_t
qid(uint64_t h)
{
    return rocprofiler_queue_id_t{h};
}
}  // namespace

TEST(correlation_key, equality_hash_and_per_gpu_scoping)
{
    // (doorbell_off, dispatch_idx_low32, gpu_id) -- `generation` is deleted; a
    // recycled doorbell is now a distinct window carried beside the key, not a key
    // field.
    auto a = correlation_key{7, 100, 0};
    EXPECT_EQ(a, (correlation_key{7, 100, 0}));
    EXPECT_EQ(correlation_key_hash{}(a), correlation_key_hash{}(correlation_key{7, 100, 0}));
    // GPU id scopes the key; two records with the same slot/index on two GPUs must
    // never be equal (no cross-agent tick comparison).
    auto on_gpu0 = correlation_key{7, 100, 0};
    auto on_gpu1 = correlation_key{7, 100, 1};
    EXPECT_NE(on_gpu0, on_gpu1) << "identical slot/index on two GPUs must not be equal";
    EXPECT_NE(a, (correlation_key{8, 100, 0})) << "different slot";
    EXPECT_NE(a, (correlation_key{7, 101, 0})) << "different dispatch index";
}
// Time-as-generation window bookkeeping: open/resolve/close, max, the overlap
// verdict, first_owner, the disable latch, and the page-slot helpers.
TEST(DoorbellMap, window_open_resolve_close_and_slot_helpers)
{
    signal_less_disable_latch().store(false);  // shared process latch; start clean

    {  // max: open(100)/close(200)/open(50) -> W2.t_open == 200
        auto m  = DoorbellMap{};
        auto o1 = m.open_window(0, qid(1), 5, 100);
        ASSERT_TRUE(o1.w);
        EXPECT_FALSE(o1.overlapped);
        EXPECT_EQ(o1.w->t_open, 100u);
        EXPECT_TRUE(o1.w->first_owner) << "first ever owner of this slot";
        m.close_window(qid(1), 200, 0);
        auto o2 = m.open_window(0, qid(2), 5, 50);  // sample 50 < prev t_close 200
        ASSERT_TRUE(o2.w);
        EXPECT_EQ(o2.w->t_open, 200u) << "t_open = max(sample, prev.t_close)";
        EXPECT_FALSE(o2.w->first_owner) << "slot had a prior owner";
    }
    {  // overlap: two live owners on one slot -> overlapped, no window opened
        auto m  = DoorbellMap{};
        auto o1 = m.open_window(0, qid(1), 5, 100);
        ASSERT_TRUE(o1.w);
        auto o2 = m.open_window(0, qid(2), 5, 150);  // q1 still live
        EXPECT_TRUE(o2.overlapped);
        EXPECT_FALSE(o2.w) << "overlap opens nothing";
        EXPECT_FALSE(m.resolve(0, qid(2)).has_value()) << "no window for the overlapping queue";
    }
    {  // resolve returns the live window object; close removes it; unknown -> nullopt
        auto m = DoorbellMap{};
        EXPECT_FALSE(m.resolve(0, qid(9)).has_value());
        auto o = m.open_window(0, qid(9), 4, 100);
        auto r = m.resolve(0, qid(9));
        ASSERT_TRUE(r.has_value());
        EXPECT_EQ((*r)->slot, 4u);
        EXPECT_EQ((*r).get(), o.w.get()) << "resolve returns the same window object";
        auto w = m.close_window(qid(9), 200, 0);
        ASSERT_TRUE(w);
        EXPECT_EQ(w->t_close.load(), 200u);
        EXPECT_FALSE(m.resolve(0, qid(9)).has_value()) << "closed -> no resolve";
        EXPECT_FALSE(m.close_window(qid(9), 300, 0)) << "double close -> nullptr";
        EXPECT_FALSE(m.close_window(qid(404), 1, 0)) << "never-opened -> nullptr";
    }
    {  // first_owner is live.empty(): after many reuses it is false; the
       // predecessor is superseded before the successor opens
        auto m = DoorbellMap{};
        for(int i = 1; i <= 4; ++i)
        {
            auto o = m.open_window(0, qid(i), 5, static_cast<uint64_t>(i) * 100);
            ASSERT_TRUE(o.w);
            m.close_window(qid(i), static_cast<uint64_t>(i) * 100 + 50, 0);
        }
        auto o = m.open_window(0, qid(99), 5, 1000);
        ASSERT_TRUE(o.w);
        EXPECT_FALSE(o.w->first_owner) << "slot has had prior owners";
    }
    {  // page geometry: capture and reader agree on the fixed 4 KiB/1024 mask,
       // now with no page_size parameter, on both 4 KiB and 64 KiB-page hosts
        EXPECT_EQ(doorbell_off_to_page_slot(4100u), 4u);
        EXPECT_EQ(doorbell_off_to_page_slot(4104u), 8u);
        EXPECT_EQ(doorbell_ptr_to_page_slot(0x7f0000004010ull), 4u);
        EXPECT_EQ(doorbell_ptr_to_page_slot(0x7f0000004020ull), 8u);
        EXPECT_EQ(doorbell_off_to_page_slot(4100u), doorbell_ptr_to_page_slot(0x7f0000004010ull));
        EXPECT_EQ(doorbell_ptr_to_page_slot(0x7f0000104010ull), 4u)
            << "64 KiB-page offset still masks to the same 4 KiB slot";
    }
    {  // the process-wide disable latch makes resolve() and open_window()
       // return nothing everywhere
        auto m = DoorbellMap{};
        m.open_window(0, qid(1), 5, 100);
        ASSERT_TRUE(m.resolve(0, qid(1)).has_value());
        signal_less_disable_latch().store(true);
        EXPECT_FALSE(m.resolve(0, qid(1)).has_value()) << "disabled -> resolve refuses";
        auto o = m.open_window(0, qid(2), 6, 100);
        EXPECT_FALSE(o.w) << "disabled -> open_window opens nothing";
        signal_less_disable_latch().store(false);  // reset for other tests
    }
}
