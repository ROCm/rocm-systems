// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/metadata_registry.hpp"

#include <gtest/gtest.h>

namespace rocprofsys::trace_cache
{

TEST(metadata_registry_test, ensure_track_builds_only_when_unknown)
{
    metadata_registry registry;
    int               builds = 0;
    auto const        make   = [&builds] {
        ++builds;
        return info::track{ "track", std::nullopt, "{}" };
    };

    registry.ensure_track("track", make);
    registry.ensure_track("track", make);

    EXPECT_EQ(builds, 1);
    EXPECT_EQ(registry.get_track_info_list().size(), 1U);
}

TEST(metadata_registry_test, ensure_thread_builds_only_when_unknown)
{
    metadata_registry registry;
    int               builds = 0;
    auto const        make   = [&builds] {
        ++builds;
        return info::thread{ 1, 2, 42, 0, 0, "{}" };
    };

    registry.ensure_thread(42, make);
    registry.ensure_thread(42, make);

    EXPECT_EQ(builds, 1);
    EXPECT_TRUE(registry.get_thread_info(42).has_value());
}

TEST(metadata_registry_test, add_queue_stream_string_ignore_duplicates)
{
    metadata_registry registry;
    registry.add_queue(1);
    registry.add_queue(1);
    registry.add_stream(2);
    registry.add_stream(2);
    registry.add_string("name");
    registry.add_string(std::string_view{ "name" });
    registry.add_string("other");

    EXPECT_EQ(registry.get_queue_list().size(), 1U);
    EXPECT_EQ(registry.get_stream_list().size(), 1U);
    EXPECT_EQ(registry.get_string_list().size(), 2U);
}

TEST(metadata_registry_test, add_pmc_code_object_kernel_symbol_ignore_duplicates)
{
    metadata_registry registry;
    info::pmc         pmc{};
    pmc.name = "counter";
    registry.add_pmc_info(pmc);
    registry.add_pmc_info(pmc);

    rocprofiler_callback_tracing_code_object_load_data_t code_object{};
    code_object.code_object_id = 7;
    registry.add_code_object(code_object);
    registry.add_code_object(code_object);

    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t symbol{};
    symbol.kernel_id = 9;
    registry.add_kernel_symbol(symbol);
    registry.add_kernel_symbol(symbol);

    EXPECT_EQ(registry.get_pmc_info_list().size(), 1U);
    EXPECT_EQ(registry.get_code_object_list().size(), 1U);
    EXPECT_EQ(registry.get_kernel_symbol_list().size(), 1U);
}

}  // namespace rocprofsys::trace_cache
