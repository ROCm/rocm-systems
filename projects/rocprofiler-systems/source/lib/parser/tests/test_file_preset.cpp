// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/config_store.hpp"
#include "parser/file_parser.hpp"
#include "parser/precedence.hpp"
#include "parser/preset_parser.hpp"
#include "parser/schema.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>  // NOLINT(misc-include-cleaner): json::parse
#include <nlohmann/json_fwd.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

using namespace rocprofsys::parser;

namespace
{
constexpr double k_frequency_limit = 3.0;

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

void
add_category(schema& value, std::string category_id)
{
    ASSERT_TRUE(value.add_category(category{ .id = std::move(category_id) }));
}

schema
make_schema()
{
    schema value{};
    add_category(value, "tracing");
    add_category(value, "sampling");
    auto trace          = make_param("trace", param_type::boolean);
    trace.category      = "tracing";
    trace.env           = env_spec{ .name = "ROCPROFSYS_TRACE" };
    auto freq           = make_param("freq", param_type::integer);
    freq.category       = "sampling";
    freq.max            = k_frequency_limit;
    freq.env            = env_spec{ .name = "ROCPROFSYS_FREQ" };
    auto components     = make_param("components", param_type::enumeration_list);
    components.category = "sampling";
    components.choices  = { choice{ .value = "wall_clock" },
                            choice{ .value = "cpu_clock" } };
    components.env      = env_spec{ .name = "ROCPROFSYS_COMPONENTS", .separator = ";" };
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
source_with(config_source source, value_map_t values)
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
    EXPECT_EQ(parsed.values.at("trace"), "true");
    EXPECT_EQ(parsed.values.at("freq"), "2");
    EXPECT_EQ(parsed.values.at("components"), "wall_clock;cpu_clock");
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
    EXPECT_EQ(parsed.values.at("trace"), "false");

    const auto missing =
        file_parser{}.parse_file(make_schema(), "/nonexistent/config.json");
    ASSERT_EQ(missing.errors.size(), 1U);
    EXPECT_NE(missing.errors.front().find("failed to open"), std::string::npos);

    const scoped_temp_file broken{ "{ " };
    const auto invalid = file_parser{}.parse_file(make_schema(), broken.path());
    ASSERT_EQ(invalid.errors.size(), 1U);
    EXPECT_NE(invalid.errors.front().find("failed to parse"), std::string::npos);
}

TEST(parser_file, stores_a_choice_alias_as_the_choice_value)
{
    schema value{};
    add_category(value, "output");
    add_category(value, "sampling");
    auto format         = make_param("format", param_type::enumeration);
    format.category     = "output";
    format.env          = env_spec{ .name = "ROCPROFSYS_FORMAT" };
    format.choices      = { choice{ .value = "pftrace", .aliases = { "proto" } } };
    auto components     = make_param("components", param_type::enumeration_list);
    components.category = "sampling";
    components.env      = env_spec{ .name = "ROCPROFSYS_COMPONENTS", .separator = ";" };
    components.choices  = { choice{ .value = "wall_clock", .aliases = { "wall" } },
                            choice{ .value = "cpu_clock" } };
    add_param(value, std::move(format));
    add_param(value, std::move(components));

    const auto document = nlohmann::json::parse(R"({
        "output": { "format": "proto" },
        "sampling": { "components": ["wall", "cpu_clock"] }
    })");
    const auto parsed   = file_parser{}.parse(value, document);
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("format"), "pftrace");
    EXPECT_EQ(parsed.values.at("components"), "wall_clock;cpu_clock");
}

TEST(parser_file, rejects_an_array_for_a_scalar_parameter)
{
    const auto document = nlohmann::json::parse(R"({
        "tracing": { "trace": [true] },
        "sampling": { "freq": [1, 2], "components": ["wall_clock"] }
    })");

    const auto parsed = file_parser{}.parse(make_schema(), document);
    EXPECT_EQ(parsed.values.at("components"), "wall_clock");
    EXPECT_EQ(parsed.values.find("trace"), parsed.values.end());
    EXPECT_EQ(parsed.values.find("freq"), parsed.values.end());
    ASSERT_EQ(parsed.errors.size(), 2U);
    EXPECT_NE(parsed.errors.front().find("does not accept an array"), std::string::npos);
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

TEST(parser_file, a_directory_path_throws)
{
    const auto directory = std::filesystem::temp_directory_path().string();
    try
    {
        const auto parsed = file_parser{}.parse_file(make_schema(), directory);
        static_cast<void>(parsed);
        FAIL() << "a directory path must throw";
    } catch(const std::runtime_error& error)
    {
        EXPECT_NE(std::string{ error.what() }.find(directory), std::string::npos);
    }
}

TEST(parser_file, catalog_document_omits_defaults_and_round_trips)
{
    auto value = make_schema();
    auto label = make_param("label", param_type::string);
    add_param(value, std::move(label));

    const auto resolved = resolved_config{
        { "trace", resolved_value{ .value = "true", .source = config_source::file } },
        { "freq",
          resolved_value{ .value = "1", .source = config_source::default_value } },
        { "components", resolved_value{ .value  = "wall_clock;cpu_clock",
                                        .source = config_source::environment } },
        { "label", resolved_value{ .value = "hello", .source = config_source::file } },
    };

    const auto document = catalog_document(value, resolved);
    EXPECT_EQ(document["tracing"]["trace"], true);
    EXPECT_FALSE(document["sampling"].contains("freq"));
    EXPECT_EQ(document["sampling"]["components"],
              (nlohmann::json::array({ "wall_clock", "cpu_clock" })));
    EXPECT_FALSE(document.contains("label"));

    const auto parsed = file_parser{}.parse(value, document);
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("trace"), "true");
    EXPECT_EQ(parsed.values.at("components"), "wall_clock;cpu_clock");
    EXPECT_EQ(parsed.values.find("freq"), parsed.values.end());
    EXPECT_EQ(parsed.values.find("label"), parsed.values.end());
}

TEST(parser_preset, stores_a_choice_alias_as_the_choice_value)
{
    schema value{};
    auto   format  = make_param("format", param_type::enumeration);
    format.env     = env_spec{ .name = "ROCPROFSYS_FORMAT" };
    format.choices = { choice{ .value = "pftrace", .aliases = { "proto" } } };
    add_param(value, std::move(format));

    const auto parsed =
        preset_parser{}.parse(value, { { "ROCPROFSYS_FORMAT", "proto" } });
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("format"), "pftrace");
}

TEST(parser_preset, maps_environment_settings_to_parameters)
{
    const preset_parser::settings_t settings{ { "ROCPROFSYS_TRACE", "on" },
                                              { "ROCPROFSYS_OTHER", "1" },
                                              { "ROCPROFSYS_FREQ", "9" } };

    const auto parsed = preset_parser{}.parse(make_schema(), settings);
    EXPECT_EQ(parsed.values, (value_map_t{ { "trace", "on" } }));
    ASSERT_EQ(parsed.errors.size(), 2U);
    EXPECT_NE(parsed.errors.front().find("invalid value '9' for parameter 'freq'"),
              std::string::npos);
}

TEST(parser_precedence, higher_source_replaces_lower_source)
{
    const auto resolved = resolve_precedence({
        source_with(config_source::command_line, { { "trace", "cli" } }),
        source_with(config_source::default_value,
                    { { "trace", "default" }, { "freq", "1" } }),
        source_with(config_source::file, { { "trace", "file" } }),
        source_with(config_source::preset,
                    { { "trace", "preset" }, { "output", "preset" } }),
        source_with(config_source::environment, { { "trace", "env" } }),
    });

    EXPECT_EQ(resolved.at("trace").value, "cli");
    EXPECT_EQ(resolved.at("trace").source, config_source::command_line);
    EXPECT_EQ(resolved.at("freq").value, "1");
    EXPECT_EQ(resolved.at("freq").source, config_source::default_value);
    EXPECT_EQ(resolved.at("output").source, config_source::preset);
}

TEST(parser_precedence, preset_overrides_file_and_environment)
{
    const auto resolved = resolve_precedence({
        source_with(config_source::file, { { "trace", "file" } }),
        source_with(config_source::environment, { { "trace", "env" }, { "freq", "2" } }),
        source_with(config_source::preset, { { "trace", "preset" } }),
    });

    EXPECT_EQ(resolved.at("trace").value, "preset");
    EXPECT_EQ(resolved.at("trace").source, config_source::preset);
    EXPECT_EQ(resolved.at("freq").value, "2");
    EXPECT_EQ(resolved.at("freq").source, config_source::environment);
}

TEST(parser_precedence, apply_resolved_keeps_the_winning_source)
{
    auto       store    = config_store{ make_schema() };
    const auto resolved = resolve_precedence({
        source_with(config_source::environment, { { "trace", "on" }, { "freq", "2" } }),
        source_with(config_source::preset, { { "trace", "off" } }),
    });
    apply_resolved(store, resolved);

    EXPECT_EQ(store.get("trace"), "off");
    EXPECT_EQ(store.source_of("trace"), config_source::preset);
    EXPECT_EQ(store.source_of("freq"), config_source::environment);
}

TEST(parser_precedence, apply_resolved_writes_nothing_when_a_value_is_invalid)
{
    auto       store    = config_store{ make_schema() };
    const auto resolved = resolved_config{
        { "trace",
          resolved_value{ .value = "on", .source = config_source::environment } },
        { "freq", resolved_value{ .value = "9", .source = config_source::environment } }
    };

    EXPECT_THROW(apply_resolved(store, resolved), std::runtime_error);
    EXPECT_FALSE(store.contains("trace"));
    EXPECT_FALSE(store.contains("freq"));
}
