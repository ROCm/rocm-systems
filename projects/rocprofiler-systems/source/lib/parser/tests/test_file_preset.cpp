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

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
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

schema
make_schema()
{
    schema value{};
    auto   trace       = make_param("trace", param_type::boolean);
    trace.env          = env_spec{ .name = "ROCPROFSYS_TRACE" };
    auto freq          = make_param("freq", param_type::integer);
    freq.max           = k_frequency_limit;
    freq.env           = env_spec{ .name = "ROCPROFSYS_FREQ" };
    auto components    = make_param("components", param_type::enumeration_list);
    components.choices = { choice{ .value = "wall_clock" },
                           choice{ .value = "cpu_clock" } };
    components.env     = env_spec{ .name = "ROCPROFSYS_COMPONENTS", .separator = ";" };
    add_param(value, std::move(trace));
    add_param(value, std::move(freq));
    add_param(value, std::move(components));
    return value;
}

class scoped_temp_file
{
public:
    explicit scoped_temp_file(const std::string& contents)
    : m_path(std::filesystem::path{ ::testing::TempDir() } /
             (std::string{
                  ::testing::UnitTest::GetInstance()->current_test_info()->name() } +
              ".json"))
    {
        std::ofstream{ m_path } << contents;
    }

    ~scoped_temp_file() { std::filesystem::remove(m_path); }

    scoped_temp_file(const scoped_temp_file&)            = delete;
    scoped_temp_file& operator=(const scoped_temp_file&) = delete;
    scoped_temp_file(scoped_temp_file&&)                 = delete;
    scoped_temp_file& operator=(scoped_temp_file&&)      = delete;

    [[nodiscard]] std::string path() const { return m_path.string(); }

private:
    std::filesystem::path m_path;
};

sourced_values
source_with(config_source source, value_map_t values)
{
    return sourced_values{ .source = source, .values = std::move(values) };
}
}  // namespace

TEST(parser_file, reads_ids_env_names_and_lists)
{
    const auto document = nlohmann::json::parse(R"({
        "_comment": "ignored",
        "trace": true,
        "ROCPROFSYS_FREQ": 2,
        "unused": null,
        "components": ["wall_clock", "cpu_clock"]
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
        "freq": 9,
        "trace": { "nested": true }
    })");

    const auto parsed = file_parser{}.parse(make_schema(), document);
    EXPECT_TRUE(parsed.values.empty());
    EXPECT_EQ(parsed.errors.size(), 3U);
}

TEST(parser_file, reads_a_file_and_reports_io_errors)
{
    const scoped_temp_file file{ R"({ "trace": false })" };
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

TEST(parser_preset, maps_environment_settings_to_parameters)
{
    const preset_parser::settings_t settings{ { "ROCPROFSYS_TRACE", "on" },
                                              { "ROCPROFSYS_OTHER", "1" },
                                              { "ROCPROFSYS_FREQ", "9" } };

    const auto parsed = preset_parser{}.parse(make_schema(), settings);
    EXPECT_EQ(parsed.values, (value_map_t{ { "trace", "on" } }));
    EXPECT_EQ(parsed.errors.size(), 2U);
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

TEST(parser_precedence, file_overrides_preset_for_the_same_parameter)
{
    const auto resolved = resolve_precedence({
        source_with(config_source::preset, { { "trace", "preset" } }),
        source_with(config_source::file, { { "trace", "file" } }),
    });

    EXPECT_EQ(resolved.at("trace").value, "file");
    EXPECT_EQ(resolved.at("trace").source, config_source::file);
}

TEST(parser_precedence, apply_resolved_keeps_the_winning_source)
{
    auto       store    = config_store{ make_schema() };
    const auto resolved = resolve_precedence({
        source_with(config_source::preset, { { "trace", "off" } }),
        source_with(config_source::environment, { { "trace", "on" }, { "freq", "2" } }),
    });
    apply_resolved(store, resolved);

    EXPECT_EQ(store.get("trace"), "on");
    EXPECT_EQ(store.source_of("trace"), config_source::environment);
    EXPECT_EQ(store.source_of("freq"), config_source::environment);
}
