// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/config_store.hpp"
#include "parser/schema.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace rocprofsys::parser;

namespace
{
constexpr int    k_thread_count      = 4;
constexpr int    k_writes_per_thread = 20;
constexpr double k_frequency_limit   = 3.0;
constexpr double k_scale             = 2.0;

schema_param
make_param(std::string param_id, param_type type, std::string category)
{
    auto param     = schema_param{};
    param.id       = std::move(param_id);
    param.type     = type;
    param.category = std::move(category);
    return param;
}

void
add_param(schema& value, schema_param param)
{
    ASSERT_TRUE(value.add_param(std::move(param)));
}

void
add_tracing_params(schema& value)
{
    add_param(value, make_param("trace", param_type::boolean, "tracing"));
    add_param(value, make_param("profile", param_type::boolean, "tracing"));
}

void
add_sampling_params(schema& value)
{
    auto frequency  = make_param("sampling_freq", param_type::integer, "sampling");
    frequency.min   = 1.0;
    frequency.max   = k_frequency_limit;
    auto components = make_param("components", param_type::enumeration_list, "sampling");
    components.choices = { choice{ .value = "wall_clock" },
                           choice{ .value = "cpu_clock" } };
    components.env     = env_spec{ .name = "ROCPROFSYS_COMPONENTS", .separator = ";" };
    add_param(value, std::move(frequency));
    add_param(value, make_param("scale", param_type::number, "sampling"));
    add_param(value, std::move(components));
}

void
add_general_params(schema& value)
{
    auto mode    = make_param("mode", param_type::enumeration, "general");
    mode.choices = { choice{ .value = "fast" }, choice{ .value = "slow" } };
    add_param(value, make_param("output", param_type::path, "general"));
    add_param(value, std::move(mode));
}

schema
make_schema()
{
    schema value{};
    add_tracing_params(value);
    add_sampling_params(value);
    add_general_params(value);
    return value;
}

config_store
make_store()
{
    return config_store{ make_schema() };
}
}  // namespace

TEST(parser_config_store, stores_and_reads_values)
{
    auto store = make_store();
    store.set("trace", "yes", config_source::command_line);
    store.set("output", "/tmp/out", config_source::file);

    EXPECT_EQ(store.get("trace"), "yes");
    EXPECT_TRUE(store.contains("trace"));
    EXPECT_EQ(store.source_of("trace"), config_source::command_line);
    EXPECT_EQ(store.get_as<bool>("trace"), true);
    EXPECT_EQ(store.get_as<std::string>("output"), "/tmp/out");
    EXPECT_FALSE(store.contains("profile"));
    EXPECT_FALSE(store.get("missing").has_value());
}

TEST(parser_config_store, converts_numeric_and_list_values)
{
    auto store = make_store();
    store.set("sampling_freq", "2", config_source::environment);
    store.set("scale", "2", config_source::environment);
    store.set("components", "wall_clock; cpu_clock", config_source::file);
    store.set("mode", "fast", config_source::preset);

    EXPECT_EQ(store.get_as<std::int64_t>("sampling_freq"), 2);
    EXPECT_EQ(store.get_as<double>("scale"), std::optional<double>{ k_scale });
    EXPECT_EQ(store.get_as<std::string>("mode"), "fast");
    EXPECT_EQ(store.get_as<std::vector<std::string>>("components"),
              (std::vector<std::string>{ "wall_clock", "cpu_clock" }));
    EXPECT_FALSE(store.get_as<bool>("sampling_freq").has_value());
    EXPECT_FALSE(store.get_as<std::int64_t>("scale").has_value());
    EXPECT_FALSE(store.get_as<double>("missing").has_value());
}

TEST(parser_config_store, rejects_unknown_and_invalid_values)
{
    auto store = make_store();
    store.set("sampling_freq", "2", config_source::default_value);

    EXPECT_THROW(store.set("missing", "1", config_source::command_line),
                 std::runtime_error);
    EXPECT_THROW(store.set("sampling_freq", "4", config_source::command_line),
                 std::runtime_error);
    EXPECT_THROW(store.set("trace", "maybe", config_source::environment),
                 std::runtime_error);
    EXPECT_EQ(store.get("sampling_freq"), "2");
    EXPECT_EQ(store.source_of("sampling_freq"), config_source::default_value);
}

TEST(parser_config_store, source_changes_without_notifying_when_text_is_unchanged)
{
    auto store = make_store();
    int  calls = 0;
    store.on_change("trace", [&calls](std::string_view, std::string_view) { ++calls; });

    store.set("trace", "true", config_source::file);
    store.set("trace", "true", config_source::command_line);

    EXPECT_EQ(calls, 1);
    EXPECT_EQ(store.source_of("trace"), config_source::command_line);
}

TEST(parser_config_store, callback_can_read_the_store)
{
    auto                       store = make_store();
    std::optional<std::string> seen;
    store.on_change("output",
                    [&store, &seen](std::string_view param_id, std::string_view) {
                        seen = store.get(param_id);
                    });

    store.set("output", "/tmp/out", config_source::command_line);
    EXPECT_EQ(seen, "/tmp/out");
}

TEST(parser_config_store, filters_stored_ids_by_category)
{
    auto store = make_store();
    store.set("profile", "on", config_source::environment);
    store.set("trace", "off", config_source::environment);
    store.set("scale", "2", config_source::file);

    EXPECT_EQ(store.ids_in_category("tracing"),
              (std::vector<std::string>{ "trace", "profile" }));
    EXPECT_EQ(store.ids_in_category("sampling"), std::vector<std::string>{ "scale" });
    EXPECT_TRUE(store.ids_in_category("missing").empty());
}

TEST(parser_config_store, clear_drops_values_and_keeps_callbacks)
{
    auto store = make_store();
    int  calls = 0;
    store.on_change("trace", [&calls](std::string_view, std::string_view) { ++calls; });
    store.set("trace", "true", config_source::preset);
    store.clear();

    EXPECT_FALSE(store.contains("trace"));
    store.set("trace", "false", config_source::preset);
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(store.config_schema()->find_param("trace")->id, "trace");
}

TEST(parser_config_store, rejects_callback_for_unknown_parameter)
{
    auto store = make_store();
    EXPECT_THROW(store.on_change("missing", [](std::string_view, std::string_view) {}),
                 std::runtime_error);
}

TEST(parser_config_store, values_map_contains_stored_entries)
{
    auto store = make_store();
    store.set("trace", "1", config_source::environment);
    store.set("mode", "slow", config_source::file);

    const auto stored = store.values();
    EXPECT_EQ(stored.at("trace"), "1");
    EXPECT_EQ(stored.at("mode"), "slow");
    EXPECT_EQ(stored.size(), 2U);
}

TEST(parser_config_store, instance_is_resettable)
{
    auto& store = config_store::instance();
    store.reset(make_schema());
    store.set("trace", "on", config_source::command_line);

    EXPECT_EQ(&config_store::instance(), &store);
    EXPECT_EQ(store.get_as<bool>("trace"), true);

    store.reset(schema{});
    EXPECT_FALSE(store.contains("trace"));
}

void
write_slot(config_store& store, int index)
{
    const auto param_id = "slot_" + std::to_string(index);
    for(int write = 0; write < k_writes_per_thread; ++write)
    {
        store.set(param_id, std::to_string(write), config_source::environment);
    }
}

TEST(parser_config_store, concurrent_updates_keep_each_value)
{
    schema value{};
    for(int index = 0; index < k_thread_count; ++index)
    {
        add_param(value, make_param("slot_" + std::to_string(index), param_type::string,
                                    "general"));
    }
    config_store store{ std::move(value) };

    std::vector<std::thread> threads;
    threads.reserve(k_thread_count);
    for(int index = 0; index < k_thread_count; ++index)
    {
        threads.emplace_back(write_slot, std::ref(store), index);
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    const auto last = std::to_string(k_writes_per_thread - 1);
    for(int index = 0; index < k_thread_count; ++index)
    {
        EXPECT_EQ(store.get("slot_" + std::to_string(index)), last);
    }
}
