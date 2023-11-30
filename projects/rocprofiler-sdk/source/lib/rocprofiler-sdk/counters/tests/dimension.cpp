// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"

TEST(dimension, set_get)
{
    using namespace rocprofiler::counters;
    int64_t                           max_counter_val = (std::numeric_limits<uint64_t>::max() >>
                               (64 - (DIM_BIT_LENGTH / ROCPROFILER_DIMENSION_LAST)));
    rocprofiler_counter_instance_id_t test_id         = 0;
    rocprofiler_counter_id_t          test_counter{.handle = 123};

    set_counter_in_rec(test_id, test_counter);
    // 0x007B000000000000 = decimal counter id 123 << DIM_BIT_LENGTH
    EXPECT_EQ(test_id, 0x007B000000000000);

    test_counter.handle = 321;
    set_counter_in_rec(test_id, test_counter);
    // 0x0141000000000000 = decimal counter id 321 << DIM_BIT_LENGTH
    EXPECT_EQ(test_id, 0x0141000000000000);
    EXPECT_EQ(rec_to_counter_id(test_id).handle, 321);

    // Test multiples of i, setting/getting those values across all
    // dimensions
    for(size_t multi_factor = 1; multi_factor < 7; multi_factor++)
    {
        for(size_t i = 1; i < static_cast<size_t>(ROCPROFILER_DIMENSION_LAST); i++)
        {
            auto dim = static_cast<rocprofiler_profile_counter_instance_types>(i);
            set_dim_in_rec(test_id, dim, i);
            EXPECT_EQ(rec_to_dim_pos(test_id, dim), i);
            set_dim_in_rec(test_id, dim, i * multi_factor);
            for(size_t j = 1; j < static_cast<size_t>(ROCPROFILER_DIMENSION_LAST); j++)
            {
                if(i == j) continue;
                set_dim_in_rec(test_id,
                               static_cast<rocprofiler_profile_counter_instance_types>(j),
                               max_counter_val);
                EXPECT_EQ(rec_to_dim_pos(
                              test_id, static_cast<rocprofiler_profile_counter_instance_types>(j)),
                          max_counter_val);
                EXPECT_EQ(rec_to_dim_pos(test_id, dim), i * multi_factor);
            }

            for(size_t j = static_cast<size_t>(ROCPROFILER_DIMENSION_LAST - 1); j > 0; j--)
            {
                if(i == j) continue;
                set_dim_in_rec(test_id,
                               static_cast<rocprofiler_profile_counter_instance_types>(j),
                               max_counter_val);
                EXPECT_EQ(rec_to_dim_pos(test_id, (rocprofiler_profile_counter_instance_types) j),
                          max_counter_val);
                EXPECT_EQ(rec_to_dim_pos(test_id, dim), i * multi_factor);
            }

            // Check that name exists
            EXPECT_TRUE(rocprofiler::common::get_val(
                rocprofiler::counters::dimension_map(),
                static_cast<rocprofiler_profile_counter_instance_types>(i)));
        }
    }

    for(size_t i = static_cast<size_t>(ROCPROFILER_DIMENSION_LAST - 1); i > 0; i--)
    {
        auto dim = static_cast<rocprofiler_profile_counter_instance_types>(i);
        set_dim_in_rec(test_id, dim, i * 5);
        EXPECT_EQ(rec_to_dim_pos(test_id, dim), i * 5);
        set_dim_in_rec(test_id, dim, i * 3);
        EXPECT_EQ(rec_to_dim_pos(test_id, dim), i * 3);
    }

    test_counter.handle = 123;
    set_counter_in_rec(test_id, test_counter);
    EXPECT_EQ(rec_to_counter_id(test_id).handle, 123);

    // Test that all bits can be set/fetched for dims, 0xFAFBFCFDFEFF is a random
    // collection of 48 bits.
    set_dim_in_rec(test_id, ROCPROFILER_DIMENSION_NONE, 0xFAFBFCFDFEFF);
    EXPECT_EQ(rec_to_dim_pos(test_id, ROCPROFILER_DIMENSION_NONE), 0xFAFBFCFDFEFF);
    EXPECT_EQ(rec_to_counter_id(test_id).handle, 123);
}
