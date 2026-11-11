// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/agent.hpp"
#include "core/collection/schema.hpp"
#include "core/trace_cache/metadata_registry.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <rocprofiler-sdk/callback_tracing.h>
#include <thread>
#include <vector>

namespace rocprofsys::trace_cache
{

namespace
{
constexpr std::size_t   k_second_thread_id = 7;
constexpr std::size_t   k_thread_id        = 42;
constexpr std::uint64_t k_code_object_id   = 7;
constexpr std::uint64_t k_kernel_symbol_id = 9;
}  // namespace

TEST(metadata_registry_test, ensure_track_keeps_first_registration)
{
    metadata_registry registry;

    registry.ensure_track("track", std::nullopt);
    registry.ensure_track("track", k_second_thread_id, "{\"second\":1}");

    auto const tracks = registry.get_track_info_list();
    ASSERT_EQ(tracks.size(), 1U);
    EXPECT_EQ(tracks.front().track_name, "track");
    EXPECT_FALSE(tracks.front().thread_id.has_value());
    EXPECT_EQ(tracks.front().extdata, "{}");
}

TEST(metadata_registry_test, ensure_thread_builds_only_when_unknown)
{
    metadata_registry registry;
    int               builds = 0;
    auto const        make   = [&builds] {
        ++builds;
        return info::thread{ .parent_process_id = 1,
                                      .process_id        = 2,
                                      .thread_id         = k_thread_id,
                                      .start             = 0,
                                      .end               = 0,
                                      .extdata           = "{}" };
    };

    registry.ensure_thread(k_thread_id, make);
    registry.ensure_thread(k_thread_id, make);

    EXPECT_EQ(builds, 1);
    EXPECT_EQ(registry.get_thread_info_list().size(), 1U);
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
    code_object.code_object_id = k_code_object_id;
    registry.add_code_object(code_object);
    registry.add_code_object(code_object);

    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t symbol{};
    symbol.kernel_id = k_kernel_symbol_id;
    registry.add_kernel_symbol(symbol);
    registry.add_kernel_symbol(symbol);

    EXPECT_EQ(registry.get_pmc_info_list().size(), 1U);
    EXPECT_EQ(registry.get_code_object_list().size(), 1U);
    EXPECT_EQ(registry.get_kernel_symbol_list().size(), 1U);
}

namespace
{
constexpr int k_registering_threads = 8;

[[nodiscard]] collection::column_spec
busy_column()
{
    collection::column_spec column;
    column.symbol   = "busy";
    column.unit     = "%";
    column.pmc_name = "busy";
    return column;
}

[[nodiscard]] collection::schema
make_counter_schema(std::uint32_t device)
{
    return { .agent          = agent_type::gpu,
             .device         = device,
             .target_arch    = "GPU",
             .event_category = "amd_smi",
             .columns        = { busy_column() } };
}
/** Adds `description` from k_registering_threads threads at once. @return their ids */
[[nodiscard]] std::array<std::uint32_t, k_registering_threads>
add_from_threads(metadata_registry& registry, const collection::schema& description)
{
    std::array<std::uint32_t, k_registering_threads> ids{};
    ids.fill(std::numeric_limits<std::uint32_t>::max());
    std::vector<std::jthread> threads;
    threads.reserve(ids.size());
    for(auto& schema_id : ids)
    {
        threads.emplace_back([&registry, &description, &schema_id] {
            schema_id = registry.add_counter_schema(description);
        });
    }
    threads.clear();
    return ids;
}
}  // namespace

TEST(metadata_registry_test, add_counter_schema_gives_an_equal_schema_its_existing_id)
{
    metadata_registry registry;

    const auto first  = registry.add_counter_schema(make_counter_schema(0));
    const auto second = registry.add_counter_schema(make_counter_schema(1));
    const auto again  = registry.add_counter_schema(make_counter_schema(0));

    EXPECT_EQ(first, 0U);
    EXPECT_EQ(second, 1U);
    EXPECT_EQ(again, first);
    EXPECT_EQ(registry.get_counter_schemas().size(), 2U);
}

TEST(metadata_registry_test, add_counter_schema_treats_any_column_difference_as_new)
{
    metadata_registry registry;
    auto              changed = make_counter_schema(0);
    changed.columns[0].unit   = "percent";

    const auto first  = registry.add_counter_schema(make_counter_schema(0));
    const auto second = registry.add_counter_schema(changed);

    EXPECT_NE(first, second);
}

TEST(metadata_registry_test, counter_schema_id_is_its_index)
{
    metadata_registry registry;
    static_cast<void>(registry.add_counter_schema(make_counter_schema(0)));
    const auto schema_id = registry.add_counter_schema(make_counter_schema(1));

    const auto schemas = registry.get_counter_schemas();

    ASSERT_LT(schema_id, schemas.size());
    EXPECT_EQ(schemas[schema_id], make_counter_schema(1));
}

TEST(metadata_registry_test, concurrent_adds_of_one_schema_share_one_id)
{
    metadata_registry registry;

    const auto ids = add_from_threads(registry, make_counter_schema(0));

    EXPECT_EQ(registry.get_counter_schemas().size(), 1U);
    EXPECT_EQ(std::ranges::count(ids, 0U), k_registering_threads);
}
}  // namespace rocprofsys::trace_cache
