// MIT License
//
// Copyright (c) 2024-2025 ROCm Developer Tools
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

#include "lib/rocprofiler-sdk/pc_sampling/cid_manager.hpp"
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/pc_record_interface.hpp"

#include <gtest/gtest.h>
#include <cstddef>

TEST(pc_sampling, cid_manager)
{
    using correlation_id_t = rocprofiler::context::correlation_id;
    using cid_manager_t    = rocprofiler::pc_sampling::PCSCIDManager;
    using pcs_parser_t     = PCSamplingParserContext;

    auto pcs_copy_fn = []() {};

    // Thread id
    rocprofiler_thread_id_t t1 = 1;
    // PC sampling parser
    auto pcs_parser = pcs_parser_t();
    // CID manager
    auto cid_manager = cid_manager_t(&pcs_parser);

    auto c1 = correlation_id_t(5, t1, 1);
    auto c2 = correlation_id_t(4, t1, 1);

    // Mark kernels of c1 and c2 completed
    cid_manager.cid_async_activity_completed(&c1);
    cid_manager.cid_async_activity_completed(&c2);
    // ref counts remained unchanged
    EXPECT_EQ(c1.get_ref_count(), 5);
    EXPECT_EQ(c2.get_ref_count(), 4);

    // Implicit flush happens
    cid_manager.manage_cids_implicit(pcs_copy_fn);
    // One implicit flush will not cause ref counts to decrement
    EXPECT_EQ(c1.get_ref_count(), 5);
    EXPECT_EQ(c2.get_ref_count(), 4);

    // The 2nd implicit flush happens
    cid_manager.manage_cids_implicit(pcs_copy_fn);
    // Ref counts should be decremented
    EXPECT_EQ(c1.get_ref_count(), 4);
    EXPECT_EQ(c2.get_ref_count(), 3);
    // c1 and c2 will be removed from the cid_manager,
    // so their ref counts remain the same until the end of the test
    // (see the last check)

    auto c3 = correlation_id_t(3, t1, 1);
    auto c4 = correlation_id_t(2, t1, 1);
    auto c5 = correlation_id_t(1, t1, 1);

    // kernels 3 and 4 finished
    cid_manager.cid_async_activity_completed(&c3);
    cid_manager.cid_async_activity_completed(&c4);

    // Implicit flush
    cid_manager.manage_cids_implicit(pcs_copy_fn);
    // ref counts unchanged
    EXPECT_EQ(c3.get_ref_count(), 3);
    EXPECT_EQ(c4.get_ref_count(), 2);

    // kernel 5 finished
    cid_manager.cid_async_activity_completed(&c5);

    // An explicit flush is requested
    cid_manager.manage_cids_explicit(pcs_copy_fn);
    // ref counts of c3, c4, and c5 should be decremented
    EXPECT_EQ(c3.get_ref_count(), 2);
    EXPECT_EQ(c4.get_ref_count(), 1);
    EXPECT_EQ(c5.get_ref_count(), 0);

    // Check whether c1 and c2's ref counts remained unchanged
    EXPECT_EQ(c1.get_ref_count(), 4);
    EXPECT_EQ(c2.get_ref_count(), 3);
}
