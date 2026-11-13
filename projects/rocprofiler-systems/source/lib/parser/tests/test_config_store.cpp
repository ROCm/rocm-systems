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
    components.env     = "ROCPROFSYS_COMPONENTS";
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

    EXPECT_EQ(store.get("trace"), "true");
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
    store.set("components", "wall_clock, cpu_clock", config_source::file);
    store.set("mode", "fast", config_source::preset);

    EXPECT_EQ(store.get_as<std::int64_t>("sampling_freq"), 2);
    EXPECT_EQ(store.get_as<double>("scale"), std::optional<double>{ k_scale });
    EXPECT_EQ(store.get_as<std::string>("mode"), "fast");
    EXPECT_EQ(store.get_as<std::vector<std::string>>("components"),
              (std::vector<std::string>{ "wall_clock", "cpu_clock" }));
    EXPECT_FALSE(store.get_as<bool>("sampling_freq").has_value());
    EXPECT_FALSE(store.get_as<double>("sampling_freq").has_value());
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

TEST(parser_config_store, source_changes_without_notifying_when_the_value_is_unchanged)
{
    auto store = make_store();
    int  calls = 0;
    store.on_change("trace", [&calls](std::string_view, std::string_view) { ++calls; });

    store.set("trace", "yes", config_source::file);
    store.set("trace", "TRUE", config_source::command_line);

    EXPECT_EQ(calls, 1);
    EXPECT_EQ(store.get("trace"), "true");
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
    EXPECT_EQ(stored.at("trace"), "true");
    EXPECT_EQ(stored.at("mode"), "slow");
    EXPECT_EQ(stored.size(), 2U);
}

TEST(parser_config_store, callback_can_store_another_parameter)
{
    auto store = make_store();
    store.on_change("trace", [&store](std::string_view, std::string_view value) {
        if(value == "true")
        {
            store.set("profile", "true", config_source::preset);
        }
    });

    store.set("trace", "true", config_source::command_line);
    EXPECT_EQ(store.get_as<bool>("profile"), true);

    store.clear();
    EXPECT_FALSE(store.contains("trace"));
    EXPECT_FALSE(store.contains("profile"));
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

namespace
{
schema
make_alias_schema()
{
    auto format     = make_param("format", param_type::enumeration, "output");
    format.choices  = { choice{ .value = "pftrace", .aliases = { "proto" } },
                        choice{ .value = "text", .aliases = { "txt" } } };
    auto formats    = make_param("formats", param_type::enumeration_list, "output");
    formats.choices = format.choices;

    schema value{};
    add_param(value, std::move(format));
    add_param(value, std::move(formats));
    return value;
}
}  // namespace

TEST(parser_config_store, stores_a_choice_alias_as_its_canonical_value)
{
    //  setting proto makes get() and get_as<std::string> return pftrace. Currently fails.
    auto store = config_store{ make_alias_schema() };
    store.set("format", "proto", config_source::command_line);

    EXPECT_EQ(store.get("format"), "pftrace");
    EXPECT_EQ(store.get_as<std::string>("format"), "pftrace");
}

TEST(parser_config_store, stores_aliases_in_a_list_as_canonical_values)
{
    // Setting " proto , text,txt" gives {"pftrace", "text", "text"}, or {"pftrace",
    // "text"}, if de-duped. Currently fails.
    auto store = config_store{ make_alias_schema() };
    store.set("formats", " proto , text,txt", config_source::file);

    EXPECT_EQ(store.get_as<std::vector<std::string>>("formats"),
              (std::vector<std::string>{ "pftrace", "text", "text" }));
    EXPECT_EQ(store.get("formats"), "pftrace,text,text");
}

TEST(parser_config_store, an_alias_of_the_stored_choice_is_not_a_change)
{
    // Setting "pftrace", then "proto", doesn't trigger a callback because the value is
    // the "same". Currently fails.
    auto store = config_store{ make_alias_schema() };
    int  calls = 0;
    store.on_change("format", [&](std::string_view, std::string_view) { ++calls; });

    store.set("format", "pftrace", config_source::file);
    store.set("format", "proto", config_source::command_line);

    EXPECT_EQ(calls, 1);
    EXPECT_EQ(store.source_of("format"), config_source::command_line);
}

TEST(parser_config_store, a_callback_receives_the_canonical_value_for_an_alias)
{
    // Setting "txt" makes the callback receive "text" (the canonical value). Currently
    // fails.
    auto        store = config_store{ make_alias_schema() };
    std::string seen;
    store.on_change("format",
                    [&](std::string_view, std::string_view value) { seen = value; });

    store.set("format", "txt", config_source::environment);

    EXPECT_EQ(seen, "text");
}

TEST(parser_config_store, still_rejects_text_that_is_neither_a_choice_nor_an_alias)
{
    // Setting "pb" or "proto,pb" rejects the value. Currently passes.
    auto store = config_store{ make_alias_schema() };

    EXPECT_THROW(store.set("format", "pb", config_source::command_line),
                 std::runtime_error);
    EXPECT_THROW(store.set("formats", "proto,pb", config_source::command_line),
                 std::runtime_error);
    EXPECT_FALSE(store.contains("format"));
}
