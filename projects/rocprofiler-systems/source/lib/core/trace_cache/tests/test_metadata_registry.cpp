// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/metadata_registry.hpp"

#include <gtest/gtest.h>

namespace rocprofsys::trace_cache
{
namespace
{

TEST(metadata_registry_test, gpu_perf_counter_lookup_uses_device_and_counter_id)
{
    metadata_registry registry;

    registry.set_gpu_perf_counter_counter_names(
        0,
        {
            info::gpu_perf_counter_name_entry{ 10, "SQ_WAVES", "GPU [0] SQ_WAVES (S)" },
            info::gpu_perf_counter_name_entry{ 20, "SQ_BUSY", "GPU [0] SQ_BUSY (S)" },
        });
    registry.set_gpu_perf_counter_counter_names(
        1,
        {
            info::gpu_perf_counter_name_entry{ 10, "SQ_WAVES", "GPU [1] SQ_WAVES (S)" },
        });

    auto device0_counter20 = registry.find_gpu_perf_counter_by_id(0, 20);
    ASSERT_TRUE(device0_counter20.has_value());
    EXPECT_EQ(device0_counter20->get().pmc_info_name, "SQ_BUSY");
    EXPECT_EQ(device0_counter20->get().track_name, "GPU [0] SQ_BUSY (S)");

    auto device1_counter10 = registry.find_gpu_perf_counter_by_id(1, 10);
    ASSERT_TRUE(device1_counter10.has_value());
    EXPECT_EQ(device1_counter10->get().track_name, "GPU [1] SQ_WAVES (S)");

    EXPECT_FALSE(registry.find_gpu_perf_counter_by_id(0, 99).has_value());
    EXPECT_FALSE(registry.find_gpu_perf_counter_by_id(2, 10).has_value());
}

TEST(metadata_registry_test, spm_counter_lookup_uses_device_and_counter_instance_id)
{
    metadata_registry registry;

    registry.set_spm_counter_names(
        0,
        {
            info::spm_counter_name_entry{ 7, 1001, 9001, "GPU SPM SQ_WAVES[XCC=0] [0]" },
            info::spm_counter_name_entry{ 7, 1002, 9002, "GPU SPM SQ_WAVES[XCC=1] [0]" },
        });
    registry.set_spm_counter_names(
        1,
        {
            info::spm_counter_name_entry{ 7, 1001, 9101, "GPU SPM SQ_WAVES[XCC=0] [1]" },
        });

    auto device0_counter1002 = registry.find_spm_counter_by_id(0, 1002);
    ASSERT_TRUE(device0_counter1002.has_value());
    EXPECT_EQ(device0_counter1002->get().counter_id, 7);
    EXPECT_EQ(device0_counter1002->get().track_key, 9002);
    EXPECT_EQ(device0_counter1002->get().track_name, "GPU SPM SQ_WAVES[XCC=1] [0]");

    auto device1_counter1001 = registry.find_spm_counter_by_id(1, 1001);
    ASSERT_TRUE(device1_counter1001.has_value());
    EXPECT_EQ(device1_counter1001->get().track_name, "GPU SPM SQ_WAVES[XCC=0] [1]");

    EXPECT_FALSE(registry.find_spm_counter_by_id(0, 9999).has_value());
    EXPECT_FALSE(registry.find_spm_counter_by_id(2, 1001).has_value());
}

}  // namespace
}  // namespace rocprofsys::trace_cache
