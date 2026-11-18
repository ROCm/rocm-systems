// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/cli_parser.hpp"
#include "parser/config_store.hpp"
#include "parser/env_parser.hpp"
#include "parser/file_parser.hpp"
#include "parser/parse_result.hpp"
#include "parser/precedence.hpp"
#include "parser/preset_parser.hpp"
#include "parser/schema.hpp"
#include "parser/value_text.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

using namespace rocprofsys::parser;

namespace
{
schema_param
make_param(std::string param_id, param_type type)
{
    auto param = schema_param{};
    param.id   = std::move(param_id);
    param.type = type;
    return param;
}

void
add_param(schema& value, schema_param param)
{
    ASSERT_TRUE(value.add_param(std::move(param)));
}

config_value_t
str_value(std::string value)
{
    return config_value_t{ std::move(value) };
}

config_value_t
int_value(std::int64_t value)
{
    return config_value_t{ value };
}

schema
make_schema()
{
    constexpr double k_frequency_limit = 3.0;

    schema value{};
    auto   trace        = make_param("trace", param_type::boolean);
    trace.category      = "tracing";
    trace.env           = "ROCPROFSYS_TRACE";
    auto freq           = make_param("freq", param_type::integer);
    freq.category       = "sampling";
    freq.max            = k_frequency_limit;
    freq.env            = "ROCPROFSYS_FREQ";
    auto components     = make_param("components", param_type::enumeration_list);
    components.category = "sampling";
    components.choices  = { "wall_clock", "cpu_clock" };
    components.env      = "ROCPROFSYS_COMPONENTS";
    components.arg      = "--components";
    add_param(value, std::move(trace));
    add_param(value, std::move(freq));
    add_param(value, std::move(components));
    return value;
}

class scoped_temp_file
{
public:
    explicit scoped_temp_file(const std::string& contents)
    : m_path(make_path())
    {
        std::ofstream{ m_path } << contents;
    }

    ~scoped_temp_file()
    {
        std::error_code error;
        std::filesystem::remove(m_path, error);
    }

    scoped_temp_file(const scoped_temp_file&)            = delete;
    scoped_temp_file& operator=(const scoped_temp_file&) = delete;
    scoped_temp_file(scoped_temp_file&&)                 = delete;
    scoped_temp_file& operator=(scoped_temp_file&&)      = delete;

    [[nodiscard]] std::string path() const { return m_path.string(); }

private:
    [[nodiscard]] static std::filesystem::path make_path()
    {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        const auto  name = std::string{ info == nullptr ? "parser" : info->name() };
        static std::atomic<std::uint64_t> s_serial{ 0 };
        return std::filesystem::path{ ::testing::TempDir() } /
               (name + "-" + std::to_string(s_serial.fetch_add(1)) + ".json");
    }

    std::filesystem::path m_path;
};

sourced_values
source_with(config_source source, typed_values_t values)
{
    return sourced_values{ .source = source, .values = std::move(values) };
}

void
expect_mentions(const std::vector<std::string>& errors, std::string_view fragment)
{
    const auto found =
        std::ranges::any_of(errors, [&fragment](const std::string& message) {
            return message.find(fragment) != std::string::npos;
        });
    EXPECT_TRUE(found) << fragment;
}

/// Canonical text of the value stored for @p param_id; nullopt when nothing is stored.
std::optional<std::string>
stored_text(const parse_result& parsed, std::string_view param_id)
{
    const auto found = parsed.values.find(param_id);
    return found == parsed.values.end() ? std::nullopt
                                        : std::optional{ format_value(found->second) };
}

/// What "components" stores from @p items as a JSON array and as command-line tokens.
struct list_verdicts
{
    std::optional<std::string> file;
    std::optional<std::string> command_line;
};

list_verdicts
read_items(const std::vector<std::string>& items)
{
    const auto config_schema           = make_schema();
    auto       document                = nlohmann::json::object();
    document["sampling"]["components"] = items;

    std::vector<std::string> args{ "--components" };
    args.insert(args.end(), items.begin(), items.end());
    const auto from_file = file_parser{}.parse(config_schema, document);
    const auto from_cli =
        cli_parser{ std::move(args), unknown_option::report }.parse(config_schema);
    return { .file         = stored_text(from_file, "components"),
             .command_line = stored_text(from_cli, "components") };
}

/// What a preset and the environment give for the same setting.
struct preset_and_environment
{
    parse_result preset;
    parse_result environment;
};

/// @p text as the value of @p variable in a preset and in the environment.
preset_and_environment
read_setting(const std::string& variable, const std::string& text)
{
    const auto       config_schema = make_schema();
    const env_parser environment{
        [&variable, &text](std::string_view name) -> std::optional<std::string> {
            return name == variable ? std::optional{ text } : std::nullopt;
        }
    };
    return { .preset      = preset_parser{}.parse(config_schema, { { variable, text } }),
             .environment = environment.parse(config_schema) };
}
}  // namespace

TEST(parser_file, reads_categories_scalars_and_lists)
{
    const auto document = nlohmann::json::parse(R"({
        "_comment": "ignored",
        "tracing": { "trace": true, "unused": null },
        "sampling": { "freq": 2, "components": ["wall_clock", "cpu_clock"] }
    })");

    const auto parsed = file_parser{}.parse(make_schema(), document);
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("trace"), config_value_t{ true });
    EXPECT_EQ(parsed.values.at("freq"), int_value(2));
    EXPECT_EQ(parsed.text("components"), "wall_clock,cpu_clock");
    EXPECT_EQ(parsed.values.find("unused"), parsed.values.end());
}

TEST(parser_file, reports_unknown_and_invalid_values)
{
    const auto document = nlohmann::json::parse(R"({
        "bogus": 1,
        "sampling": { "freq": 9 },
        "tracing": { "trace": { "nested": true } }
    })");

    const auto parsed = file_parser{}.parse(make_schema(), document);
    EXPECT_TRUE(parsed.values.empty());
    EXPECT_EQ(parsed.errors.size(), 3U);
}

TEST(parser_file, reads_a_file_and_reports_io_errors)
{
    const scoped_temp_file file{ R"({ "tracing": { "trace": false } })" };
    const auto             parsed = file_parser{}.parse_file(make_schema(), file.path());
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.text("trace"), "false");

    const auto missing =
        file_parser{}.parse_file(make_schema(), "/nonexistent/config.json");
    ASSERT_EQ(missing.errors.size(), 1U);
    EXPECT_NE(missing.errors.front().find("failed to open"), std::string::npos);

    const scoped_temp_file broken{ "{ " };
    const auto invalid = file_parser{}.parse_file(make_schema(), broken.path());
    ASSERT_EQ(invalid.errors.size(), 1U);
    EXPECT_NE(invalid.errors.front().find("failed to parse"), std::string::npos);
}

TEST(parser_file, reports_a_number_too_large_for_a_double)
{
    const scoped_temp_file file{ R"({ "sampling": { "freq": 1e400 } })" };
    const auto             parsed = file_parser{}.parse_file(make_schema(), file.path());
    EXPECT_TRUE(parsed.values.empty());
    ASSERT_EQ(parsed.errors.size(), 1U);
    EXPECT_NE(parsed.errors.front().find("failed to parse configuration file"),
              std::string::npos);
}

TEST(parser_file, rejects_an_array_for_a_scalar_parameter)
{
    const auto document = nlohmann::json::parse(R"({
        "tracing": { "trace": [true] },
        "sampling": { "freq": [1, 2], "components": ["wall_clock"] }
    })");

    const auto parsed = file_parser{}.parse(make_schema(), document);
    EXPECT_EQ(parsed.text("components"), "wall_clock");
    EXPECT_EQ(parsed.values.find("trace"), parsed.values.end());
    EXPECT_EQ(parsed.values.find("freq"), parsed.values.end());
    ASSERT_EQ(parsed.errors.size(), 2U);
    EXPECT_NE(parsed.errors.front().find("does not accept an array"), std::string::npos);
}

TEST(parser_file, reads_array_items_like_command_line_tokens)
{
    using items      = std::vector<std::string>;
    const auto cases = std::vector<std::pair<items, std::optional<std::string>>>{
        { items{ "wall_clock", "cpu_clock" }, "wall_clock,cpu_clock" },
        { items{ "wall_clock", "", "cpu_clock" }, "wall_clock,cpu_clock" },
        { items{ "wall_clock,", "cpu_clock" }, "wall_clock,cpu_clock" },
        { items{ "wall_clock", "," }, "wall_clock" },
        { items{ " wall_clock " }, "wall_clock" },
        { items{ "" }, "" },
        { items{ "," }, std::nullopt },
        { items{ "wall_clock", ",", ",", "cpu_clock" }, std::nullopt },
        { items{ "wall_clock", "bogus" }, std::nullopt },
    };
    for(const auto& [list, expected] : cases)
    {
        SCOPED_TRACE(nlohmann::json(list).dump());
        const auto found = read_items(list);
        EXPECT_EQ(found.file, expected);
        EXPECT_EQ(found.command_line, expected);
    }
}

TEST(parser_file, rejects_an_environment_name_and_a_misplaced_parameter)
{
    const auto document = nlohmann::json::parse(R"({
        "ROCPROFSYS_TRACE": true,
        "tracing": false,
        "sampling": { "ROCPROFSYS_FREQ": 2, "trace": true }
    })");

    const auto parsed = file_parser{}.parse(make_schema(), document);
    EXPECT_TRUE(parsed.values.empty());
    ASSERT_EQ(parsed.errors.size(), 4U);
    expect_mentions(parsed.errors, "unknown category");
    expect_mentions(parsed.errors, "must be a JSON object");
    expect_mentions(parsed.errors, "unknown parameter");
    expect_mentions(parsed.errors, "not in category");
}

TEST(parser_file, a_directory_path_is_reported)
{
    const auto directory = std::filesystem::temp_directory_path().string();
    const auto parsed    = file_parser{}.parse_file(make_schema(), directory);
    EXPECT_TRUE(parsed.values.empty());
    ASSERT_EQ(parsed.errors.size(), 1U);
    EXPECT_EQ(parsed.errors.front(),
              "configuration path '" + directory + "' is a directory");
}

TEST(parser_file, catalog_document_omits_defaults_and_round_trips)
{
    auto value     = make_schema();
    auto label     = make_param("label", param_type::string);
    label.category = "tracing";
    add_param(value, std::move(label));

    const auto resolved = resolved_config{
        { "trace", resolved_value{ .value  = config_value_t{ true },
                                   .source = config_source::file } },
        { "freq", resolved_value{ .value  = int_value(1),
                                  .source = config_source::default_value } },
        { "components", resolved_value{ .value = config_value_t{ std::vector<std::string>{
                                            "wall_clock", "cpu_clock" } },
                                        .source = config_source::environment } },
        { "label",
          resolved_value{ .value = str_value("hello"), .source = config_source::file } },
    };

    const auto document = catalog_document(value, resolved);
    EXPECT_EQ(document["tracing"]["trace"], true);
    EXPECT_EQ(document["tracing"]["label"], "hello");
    EXPECT_FALSE(document["sampling"].contains("freq"));
    EXPECT_EQ(document["sampling"]["components"],
              (nlohmann::json::array({ "wall_clock", "cpu_clock" })));

    const auto parsed = file_parser{}.parse(value, document);
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.text("trace"), "true");
    EXPECT_EQ(parsed.text("components"), "wall_clock,cpu_clock");
    EXPECT_EQ(parsed.values.find("freq"), parsed.values.end());
}

TEST(parser_file, reads_ids_that_start_with_an_underscore)
{
    auto value      = make_schema();
    auto hidden     = make_param("_hidden", param_type::integer);
    hidden.category = "sampling";
    auto shown      = make_param("shown", param_type::integer);
    shown.category  = "_internal";
    add_param(value, std::move(hidden));
    add_param(value, std::move(shown));
    const auto resolved = resolved_config{
        { "_hidden",
          resolved_value{ .value = int_value(1), .source = config_source::file } },
        { "shown",
          resolved_value{ .value = int_value(2), .source = config_source::file } },
    };

    auto document                 = catalog_document(value, resolved);
    document["_comment"]          = "ignored";
    document["sampling"]["_note"] = "ignored";
    const auto parsed             = file_parser{}.parse(value, document);
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values,
              (typed_values_t{ { "_hidden", int_value(1) }, { "shown", int_value(2) } }));
}

// A parameter with no category has no place in a document grouped by category, so
// catalog_document leaves it out instead of writing it under an empty key.
TEST(parser_file, catalog_document_omits_an_uncategorized_parameter)
{
    auto config_schema = make_schema();
    add_param(config_schema, make_param("loose", param_type::integer));
    const auto resolved = resolved_config{
        { "loose",
          resolved_value{ .value = int_value(1), .source = config_source::file } },
    };

    EXPECT_TRUE(catalog_document(config_schema, resolved).empty());
}

// A parameter with no category has an empty category string, so an empty key must not
// be taken as the name of the category that holds it.
TEST(parser_file, an_empty_category_key_does_not_name_uncategorized_parameters)
{
    auto config_schema = make_schema();
    add_param(config_schema, make_param("loose", param_type::integer));
    const auto document = nlohmann::json::parse(R"({ "": { "loose": 1 } })");

    const auto parsed = file_parser{}.parse(config_schema, document);
    EXPECT_TRUE(parsed.values.empty());
    ASSERT_EQ(parsed.errors.size(), 1U);
    expect_mentions(parsed.errors, "unknown category");
}

TEST(parser_preset, maps_environment_settings_to_parameters)
{
    const preset_parser::settings_t settings{ { "ROCPROFSYS_TRACE", "on" },
                                              { "ROCPROFSYS_OTHER", "1" },
                                              { "ROCPROFSYS_FREQ", "9" } };

    const auto parsed = preset_parser{}.parse(make_schema(), settings);
    EXPECT_EQ(parsed.values, (typed_values_t{ { "trace", config_value_t{ true } } }));
    ASSERT_EQ(parsed.errors.size(), 2U);
    EXPECT_NE(parsed.errors.front().find("invalid value '9' for parameter 'freq'"),
              std::string::npos);
}

TEST(parser_preset, leaves_a_blank_value_unset)
{
    const auto parsed =
        preset_parser{}.parse(make_schema(), { { "ROCPROFSYS_TRACE", "" },
                                               { "ROCPROFSYS_FREQ", " " },
                                               { "ROCPROFSYS_COMPONENTS", "" } });
    EXPECT_TRUE(parsed.values.empty());
    EXPECT_TRUE(parsed.errors.empty());
}

TEST(parser_preset, reads_a_value_as_the_environment_does)
{
    const auto settings = std::vector<std::pair<std::string, std::string>>{
        { "ROCPROFSYS_TRACE", "" },
        { "ROCPROFSYS_FREQ", " " },
        { "ROCPROFSYS_FREQ", " 2 " },
        { "ROCPROFSYS_FREQ", " 9 " },
        { "ROCPROFSYS_COMPONENTS", "" },
        { "ROCPROFSYS_COMPONENTS", "wall_clock," },
        { "ROCPROFSYS_COMPONENTS", " , " },
        { "ROCPROFSYS_COMPONENTS", "wall_clock,bogus" },
    };
    for(const auto& [variable, text] : settings)
    {
        SCOPED_TRACE(::testing::Message() << variable << "=" << text);
        const auto found = read_setting(variable, text);
        EXPECT_EQ(found.preset.values, found.environment.values);
        EXPECT_EQ(found.preset.errors, found.environment.errors);
    }
}

TEST(parser_precedence, higher_source_replaces_lower_source)
{
    const auto resolved = resolve_precedence({
        source_with(config_source::command_line, { { "trace", str_value("cli") } }),
        source_with(config_source::default_value,
                    { { "trace", str_value("default") }, { "freq", int_value(1) } }),
        source_with(config_source::file, { { "trace", str_value("file") } }),
        source_with(config_source::preset, { { "trace", str_value("preset") },
                                             { "output", str_value("preset") } }),
        source_with(config_source::environment, { { "trace", str_value("env") } }),
    });

    EXPECT_EQ(format_value(resolved.at("trace").value), "cli");
    EXPECT_EQ(resolved.at("trace").source, config_source::command_line);
    EXPECT_EQ(format_value(resolved.at("freq").value), "1");
    EXPECT_EQ(resolved.at("freq").source, config_source::default_value);
    EXPECT_EQ(resolved.at("output").source, config_source::preset);
}

TEST(parser_precedence, preset_overrides_file_and_environment)
{
    const auto resolved = resolve_precedence({
        source_with(config_source::file, { { "trace", str_value("file") } }),
        source_with(config_source::environment,
                    { { "trace", str_value("env") }, { "freq", int_value(2) } }),
        source_with(config_source::preset, { { "trace", str_value("preset") } }),
    });

    EXPECT_EQ(format_value(resolved.at("trace").value), "preset");
    EXPECT_EQ(resolved.at("trace").source, config_source::preset);
    EXPECT_EQ(format_value(resolved.at("freq").value), "2");
    EXPECT_EQ(resolved.at("freq").source, config_source::environment);
}

TEST(parser_precedence, later_source_of_equal_rank_wins)
{
    const auto first_then_second = resolve_precedence({
        source_with(config_source::file, { { "trace", str_value("first") } }),
        source_with(config_source::file, { { "trace", str_value("second") } }),
    });
    EXPECT_EQ(format_value(first_then_second.at("trace").value), "second");

    const auto second_then_first = resolve_precedence({
        source_with(config_source::file, { { "trace", str_value("second") } }),
        source_with(config_source::file, { { "trace", str_value("first") } }),
    });
    EXPECT_EQ(format_value(second_then_first.at("trace").value), "first");
}

// Enough equal-rank sources that an unstable sort would reorder them. A short input
// can pass by accident, because small ranges are often sorted by a stable insertion sort.
TEST(parser_precedence, equal_rank_sources_keep_their_input_order)
{
    constexpr std::int64_t k_source_count = 64;

    std::vector<sourced_values> sources;
    for(std::int64_t i = 0; i < k_source_count; ++i)
    {
        sources.push_back(
            source_with(config_source::environment, { { "freq", int_value(i) } }));
        sources.push_back(
            source_with(config_source::default_value, { { "freq", int_value(-i) } }));
    }

    const auto resolved = resolve_precedence(std::move(sources));
    EXPECT_EQ(resolved.at("freq").source, config_source::environment);
    EXPECT_EQ(resolved.at("freq").value, int_value(k_source_count - 1));
}

TEST(parser_precedence, apply_resolved_keeps_the_winning_source)
{
    auto       store    = config_store{ make_schema() };
    const auto resolved = resolve_precedence({
        source_with(config_source::environment,
                    { { "trace", config_value_t{ true } }, { "freq", int_value(2) } }),
        source_with(config_source::preset, { { "trace", config_value_t{ false } } }),
    });
    apply_resolved(store, resolved);

    EXPECT_EQ(store.get("trace"), "false");
    EXPECT_EQ(store.source_of("trace"), config_source::preset);
    EXPECT_EQ(store.source_of("freq"), config_source::environment);
}

TEST(parser_precedence, apply_resolved_writes_nothing_when_a_value_is_invalid)
{
    auto       store    = config_store{ make_schema() };
    const auto resolved = resolved_config{
        { "trace", resolved_value{ .value  = str_value("on"),
                                   .source = config_source::environment } },
        { "freq",
          resolved_value{ .value = int_value(9), .source = config_source::environment } }
    };

    EXPECT_THROW(apply_resolved(store, resolved), std::runtime_error);
    EXPECT_FALSE(store.contains("trace"));
    EXPECT_FALSE(store.contains("freq"));
}

// An id that is not in the store's schema is rejected by name. The valid value beside
// it is not written, because every value is checked before the first write.
TEST(parser_precedence, apply_resolved_rejects_an_unknown_parameter)
{
    auto       store    = config_store{ make_schema() };
    const auto resolved = resolved_config{
        { "bogus",
          resolved_value{ .value = int_value(1), .source = config_source::file } },
        { "trace", resolved_value{ .value  = config_value_t{ true },
                                   .source = config_source::file } },
    };
    try
    {
        apply_resolved(store, resolved);
        FAIL() << "an unknown parameter must throw";
    } catch(const std::runtime_error& error)
    {
        EXPECT_NE(std::string{ error.what() }.find("unknown parameter 'bogus'"),
                  std::string::npos);
    }
    EXPECT_FALSE(store.contains("trace"));
}

// The exception lists every rejected value, in id order and separated by "; ", not
// only the first one.
TEST(parser_precedence, apply_resolved_reports_every_rejection)
{
    auto       store    = config_store{ make_schema() };
    const auto resolved = resolved_config{
        { "bogus",
          resolved_value{ .value = int_value(1), .source = config_source::file } },
        { "freq",
          resolved_value{ .value = int_value(9), .source = config_source::file } },
    };
    try
    {
        apply_resolved(store, resolved);
        FAIL() << "rejected values must throw";
    } catch(const std::runtime_error& error)
    {
        EXPECT_EQ(std::string{ error.what() },
                  "unknown parameter 'bogus'; invalid value '9' for parameter 'freq'");
    }
}
