#include <gtest/gtest.h>

#include "lib/common/utility.hpp"
#include "lib/rocprofiler/counters/id_decode.hpp"

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