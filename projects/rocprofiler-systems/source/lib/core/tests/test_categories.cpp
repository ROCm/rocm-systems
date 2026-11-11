// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/categories.hpp"
#include "rocprofiler-systems/categories.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

namespace rocprofsys
{
TEST(category_info, amd_smi_name_and_description)
{
    const auto names = category_info(ROCPROFSYS_CATEGORY_AMD_SMI);

    EXPECT_EQ(names.name, "amd_smi");
    EXPECT_EQ(names.description, "AMD-SMI data");
}

TEST(category_info, cpu_freq_name_is_cpu_frequency)
{
    EXPECT_EQ(category_info(ROCPROFSYS_CATEGORY_CPU_FREQ).name, "cpu_frequency");
}

TEST(category_info, hipfile_name_and_description)
{
    const auto names = category_info(ROCPROFSYS_CATEGORY_HIPFILE);

    EXPECT_EQ(names.name, "hipfile");
    EXPECT_EQ(names.description, "hipFile's AMD Infinity Storage I/O statistics");
}

TEST(category_info, every_category_id_has_a_name_and_description)
{
    for(std::size_t category_id = ROCPROFSYS_CATEGORY_NONE + 1;
        category_id < ROCPROFSYS_CATEGORY_LAST; ++category_id)
    {
        const auto names = category_info(category_id);
        EXPECT_FALSE(names.name.empty()) << category_id;
        EXPECT_FALSE(names.description.empty()) << category_id;
    }
}

TEST(category_info, ids_outside_the_category_range_throw)
{
    EXPECT_THROW(static_cast<void>(category_info(ROCPROFSYS_CATEGORY_NONE)),
                 std::out_of_range);
    EXPECT_THROW(static_cast<void>(category_info(ROCPROFSYS_CATEGORY_LAST)),
                 std::out_of_range);
    EXPECT_THROW(static_cast<void>(category_info(ROCPROFSYS_CATEGORY_LAST + 1)),
                 std::out_of_range);
}
}  // namespace rocprofsys
