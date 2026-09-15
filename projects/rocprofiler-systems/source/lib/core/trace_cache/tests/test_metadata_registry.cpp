// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/metadata_registry.hpp"

#include <cstdint>
#include <gtest/gtest.h>

namespace rocprofsys::trace_cache
{
namespace
{
constexpr auto k_gpu_counter_id_0    = std::uint64_t{ 10 };
constexpr auto k_gpu_counter_id_1    = std::uint64_t{ 20 };
constexpr auto k_missing_gpu_counter = std::uint64_t{ 99 };
constexpr auto k_spm_counter_id_0    = std::uint64_t{ 1001 };
constexpr auto k_spm_counter_id_1    = std::uint64_t{ 1002 };
constexpr auto k_missing_spm_counter = std::uint64_t{ 9999 };

// ASSERT_TRUE aborts the test before each optional access, but clang-tidy does not
// model GoogleTest's fatal assertion macro as a control-flow guard.
// NOLINTBEGIN(bugprone-unchecked-optional-access)

TEST(metadata_registry_test, gpu_perf_counter_lookup_uses_device_and_counter_id)
{
    metadata_registry registry;

    registry.set_gpu_perf_counter_counter_names(
        0, {
               { .counter_id    = k_gpu_counter_id_0,
                 .pmc_info_name = "SQ_WAVES",
                 .track_name    = "GPU [0] SQ_WAVES (S)" },
               { .counter_id    = k_gpu_counter_id_1,
                 .pmc_info_name = "SQ_BUSY",
                 .track_name    = "GPU [0] SQ_BUSY (S)" },
           });
    registry.set_gpu_perf_counter_counter_names(
        1, {
               { .counter_id    = k_gpu_counter_id_0,
                 .pmc_info_name = "SQ_WAVES",
                 .track_name    = "GPU [1] SQ_WAVES (S)" },
           });

    auto device0_counter20 = registry.find_gpu_perf_counter_by_id(0, k_gpu_counter_id_1);
    ASSERT_TRUE(device0_counter20.has_value());
    EXPECT_EQ(device0_counter20->get().pmc_info_name, "SQ_BUSY");
    EXPECT_EQ(device0_counter20->get().track_name, "GPU [0] SQ_BUSY (S)");

    auto device1_counter10 = registry.find_gpu_perf_counter_by_id(1, k_gpu_counter_id_0);
    ASSERT_TRUE(device1_counter10.has_value());
    EXPECT_EQ(device1_counter10->get().track_name, "GPU [1] SQ_WAVES (S)");

    EXPECT_FALSE(
        registry.find_gpu_perf_counter_by_id(0, k_missing_gpu_counter).has_value());
    EXPECT_FALSE(registry.find_gpu_perf_counter_by_id(2, k_gpu_counter_id_0).has_value());
}

TEST(metadata_registry_test, spm_counter_lookup_reuses_gpu_counter_entry)
{
    metadata_registry registry;

    registry.set_spm_counter_names(0, {
                                          { .counter_id    = k_spm_counter_id_0,
                                            .pmc_info_name = "SQ_WAVES[XCC=0]",
                                            .track_name = "GPU SPM SQ_WAVES[XCC=0] [0]" },
                                          { .counter_id    = k_spm_counter_id_1,
                                            .pmc_info_name = "SQ_WAVES[XCC=1]",
                                            .track_name = "GPU SPM SQ_WAVES[XCC=1] [0]" },
                                      });
    registry.set_spm_counter_names(1, {
                                          { .counter_id    = k_spm_counter_id_0,
                                            .pmc_info_name = "SQ_WAVES[XCC=0]",
                                            .track_name = "GPU SPM SQ_WAVES[XCC=0] [1]" },
                                      });

    auto device0_counter1002 = registry.find_spm_counter_by_id(0, k_spm_counter_id_1);
    ASSERT_TRUE(device0_counter1002.has_value());
    EXPECT_EQ(device0_counter1002->get().counter_id, k_spm_counter_id_1);
    EXPECT_EQ(device0_counter1002->get().pmc_info_name, "SQ_WAVES[XCC=1]");
    EXPECT_EQ(device0_counter1002->get().track_name, "GPU SPM SQ_WAVES[XCC=1] [0]");

    auto device1_counter1001 = registry.find_spm_counter_by_id(1, k_spm_counter_id_0);
    ASSERT_TRUE(device1_counter1001.has_value());
    EXPECT_EQ(device1_counter1001->get().track_name, "GPU SPM SQ_WAVES[XCC=0] [1]");

    EXPECT_FALSE(registry.find_spm_counter_by_id(0, k_missing_spm_counter).has_value());
    EXPECT_FALSE(registry.find_spm_counter_by_id(2, k_spm_counter_id_0).has_value());
}

// NOLINTEND(bugprone-unchecked-optional-access)
}  // namespace
}  // namespace rocprofsys::trace_cache
