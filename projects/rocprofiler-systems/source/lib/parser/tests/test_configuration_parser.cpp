// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/configuration_parser.hpp"
#include "parser/help.hpp"
#include "parser/preset_parser.hpp"
#include "parser/schema.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <ranges>
#include <stdlib.h>  // POSIX ::getenv, ::setenv, ::unsetenv
#include <string>
#include <string_view>
#include <vector>

using namespace rocprofsys::parser;

namespace
{
constexpr std::size_t k_help_param_count = 7;

constexpr std::string_view k_behavior_schema = R"json({
  "schema_version": "1.0",
  "params": [
    {
      "id": "config_file", "name": "Config file", "description": "Config file",
      "type": "path", "path_kind": "file",
      "arg": { "name": "--config", "form": "value" }
    },
    {
      "id": "preset", "name": "Preset", "description": "Preset",
      "type": "string",
      "arg": { "name": "--preset", "form": "value" }
    },
    {
      "id": "verbose", "name": "Verbose", "description": "Verbose",
      "type": "integer", "default_value": 0, "min": 0, "max": 4,
      "env": { "name": "ROCPROFSYS_VERBOSE" },
      "arg": { "name": "--verbose", "form": "value" }
    },
    {
      "id": "output_path", "name": "Output path", "description": "Output path",
      "type": "path", "path_kind": "directory", "default_value": "rocprof-sys-output",
      "env": { "name": "ROCPROFSYS_OUTPUT_PATH" },
      "arg": { "name": "--output", "form": "value" }
    },
    {
      "id": "trace", "name": "Trace", "description": "Trace",
      "type": "bool", "default_value": false,
      "env": { "name": "ROCPROFSYS_TRACE" },
      "arg": { "name": "--trace", "form": "flag" }
    },
    {
      "id": "profile", "name": "Profile", "description": "Profile",
      "type": "bool", "default_value": false,
      "env": { "name": "ROCPROFSYS_PROFILE" },
      "arg": { "name": "--profile", "form": "flag" }
    },
    {
      "id": "flat_profile", "name": "Flat profile", "description": "Flat profile",
      "type": "bool", "default_value": false,
      "arg": { "name": "--flat-profile", "form": "flag" }
    },
    {
      "id": "sample_cputime", "name": "Sample CPU time", "description": "CPU time",
      "type": "bool", "default_value": false,
      "mutually_exclusive_with": [ "sample_realtime" ],
      "arg": { "name": "--sample-cputime", "form": "flag" }
    },
    {
      "id": "sample_realtime", "name": "Sample real time", "description": "Real time",
      "type": "bool", "default_value": true,
      "mutually_exclusive_with": [ "sample_cputime" ],
      "arg": { "name": "--sample-realtime", "form": "flag" }
    },
    {
      "id": "rocm_domains", "name": "ROCm domains", "description": "ROCm domains",
      "type": "string",
      "default_value": "hip_runtime_api,kernel_dispatch,memory_copy",
      "value_provider": "rocm_domains_sdk",
      "arg": { "name": "--rocm-domains", "form": "value" }
    }
  ],
  "constraints": [
    {
      "kind": "forces",
      "when": { "param": "flat_profile", "equals": true },
      "set": { "profile": true },
      "message": "flat profiles require profiling"
    }
  ]
})json";

bool
contains_text(const std::vector<std::string>& errors, std::string_view text)
{
    return std::ranges::any_of(errors, [text](const std::string& item) {
        return item.find(text) != std::string::npos;
    });
}

configuration_parser
make_parser()
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    EXPECT_TRUE(parser.initialize()) << parser.failure();
    return parser;
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

configuration_parser
make_behavior_parser()
{
    const auto           file = scoped_temp_file{ std::string{ k_behavior_schema } };
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    EXPECT_TRUE(parser.initialize(file.path())) << parser.failure();
    return parser;
}

constexpr int k_overwrite = 1;

std::map<std::string, std::optional<std::string>>
capture_environment(const value_map_t& variables)
{
    std::map<std::string, std::optional<std::string>> previous;
    for(const auto& [name, value] : variables)
    {
        static_cast<void>(value);
        const auto* current = std::getenv(name.c_str());
        previous.emplace(name, current == nullptr
                                   ? std::nullopt
                                   : std::optional<std::string>{ current });
    }
    return previous;
}

void
restore_environment(const std::map<std::string, std::optional<std::string>>& previous)
{
    for(const auto& [name, value] : previous)
    {
        if(value.has_value())
        {
            static_cast<void>(::setenv(name.c_str(), value.value().c_str(), k_overwrite));
        }
        else
        {
            static_cast<void>(::unsetenv(name.c_str()));
        }
    }
}

TEST(parser_configuration, embedded_schema_is_help_text)
{
    auto       parser     = make_parser();
    const auto schema_ptr = parser.config_schema();
    ASSERT_NE(schema_ptr, nullptr);
    EXPECT_EQ(schema_ptr->version(), "1.0");
    EXPECT_EQ(schema_ptr->params().size(), k_help_param_count);

    const auto* preset  = schema_ptr->find_param("preset");
    const auto* verbose = schema_ptr->find_param("verbose");
    const auto* trace   = schema_ptr->find_param("trace");
    ASSERT_NE(preset, nullptr);
    ASSERT_NE(verbose, nullptr);
    ASSERT_NE(trace, nullptr);
    EXPECT_FALSE(preset->env.has_value());
    EXPECT_EQ(verbose->env.value_or(env_spec{}).name, "ROCPROFSYS_VERBOSE");
    EXPECT_EQ(trace->category, "tracing");

    const auto values = parser.parse({});
    EXPECT_EQ(values.at("verbose"), "0");
    EXPECT_EQ(parser.get<bool>("debug").value_or(true), false);
    EXPECT_EQ(parser.get<bool>("trace").value_or(true), false);
    EXPECT_EQ(values.find("output_path"), values.end());
    const auto help_text = parser.render_help();
    EXPECT_EQ(help_text.find("--help"), std::string::npos);
    EXPECT_EQ(help_text.find("General"), std::string::npos);
    EXPECT_NE(help_text.find("Configuration"), std::string::npos);
    EXPECT_EQ(help_text.find("ROCPROFSYS_PRESET"), std::string::npos);
    EXPECT_TRUE(parser.errors().empty()) << parser.errors().front();
}

std::optional<std::string>
verbose_environment(std::string_view name)
{
    if(name == "ROCPROFSYS_VERBOSE")
    {
        return "1";
    }
    return std::nullopt;
}

preset_parser::settings_t
balanced_settings(std::string_view name)
{
    if(name != "balanced")
    {
        return {};
    }
    return preset_parser::settings_t{ { "ROCPROFSYS_OUTPUT_PATH", "from-preset" },
                                      { "ROCPROFSYS_PROFILE", "true" } };
}
}  // namespace

TEST(parser_configuration, command_line_overrides_lower_sources)
{
    auto parser = make_behavior_parser();
    parser.set_environment(verbose_environment);
    parser.set_preset_lookup(balanced_settings);
    const auto file = scoped_temp_file{ R"({
        "output_path": "from-file",
        "trace": true
    })" };

    const auto values = parser.parse({ "--config", file.path(), "--preset", "balanced",
                                       "--verbose", "3", "--", "./app" });

    EXPECT_EQ(values.at("output_path"), "from-file");
    EXPECT_EQ(parser.get<bool>("trace").value_or(false), true);
    EXPECT_EQ(parser.get<bool>("profile").value_or(false), true);
    EXPECT_EQ(parser.get<std::int64_t>("verbose").value_or(-1), 3);
    EXPECT_EQ(parser.command(), std::vector<std::string>{ "./app" });
    EXPECT_TRUE(parser.errors().empty()) << parser.errors().front();
}

TEST(parser_configuration, cli_output_overrides_preset_and_file)
{
    auto parser = make_parser();
    parser.set_preset_lookup([](std::string_view) {
        return preset_parser::settings_t{ { "ROCPROFSYS_OUTPUT_PATH", "from-preset" } };
    });
    const auto file   = scoped_temp_file{ R"({ "output_path": "from-file" })" };
    const auto values = parser.parse(
        { "--config", file.path(), "--preset", "balanced", "--output", "from-cli" });
    EXPECT_EQ(values.at("output_path"), "from-cli");
}

TEST(parser_configuration, unknown_option_can_stop_parsing)
{
    auto parser = make_parser();
    parser.set_parse_mode(parse_mode::stop_at_unknown);
    const auto values = parser.parse({ "--output", "out", "--not-an-option", "--debug" });
    EXPECT_EQ(values.at("output_path"), "out");
    EXPECT_EQ(parser.get<bool>("debug").value_or(true), false);
    EXPECT_EQ(parser.unparsed(),
              (std::vector<std::string>{ "--not-an-option", "--debug" }));

    parser.set_parse_mode(parse_mode::full);
    static_cast<void>(parser.parse({ "--not-an-option" }));
    EXPECT_TRUE(contains_text(parser.errors(), "unknown option"));
}

TEST(parser_configuration, rejects_values_outside_the_schema)
{
    auto parser = make_parser();
    static_cast<void>(parser.parse({ "--verbose", "9" }));
    EXPECT_TRUE(contains_text(parser.errors(), "invalid value"));
    EXPECT_EQ(parser.get<std::int64_t>("verbose").value_or(-1), 0);
}

TEST(parser_configuration, rejects_values_outside_a_provider_catalog)
{
    auto parser = make_behavior_parser();
    static_cast<void>(parser.parse({ "--rocm-domains", "not_a_domain" }));
    EXPECT_TRUE(contains_text(parser.errors(), "not_a_domain"));
    EXPECT_EQ(parser.get<std::string>("rocm_domains").value_or(""),
              "hip_runtime_api,kernel_dispatch,memory_copy");

    static_cast<void>(parser.parse({ "--rocm-domains", "hsa_api" }));
    EXPECT_EQ(parser.get<std::string>("rocm_domains").value_or(""), "hsa_api");
    EXPECT_TRUE(parser.errors().empty()) << parser.errors().front();
}

TEST(parser_configuration, reports_constraints_and_unknown_presets)
{
    auto parser = make_behavior_parser();
    static_cast<void>(parser.parse({ "--sample-cputime", "--sample-realtime" }));
    EXPECT_TRUE(contains_text(parser.errors(), "mutually exclusive"));

    static_cast<void>(parser.parse({ "--flat-profile" }));
    EXPECT_TRUE(contains_text(parser.errors(), "flat profiles require profiling"));

    static_cast<void>(parser.parse({ "--preset", "missing" }));
    EXPECT_TRUE(contains_text(parser.errors(), "was not applied"));

    parser.set_preset_lookup(
        [](std::string_view) { return preset_parser::settings_t{}; });
    static_cast<void>(parser.parse({ "--preset", "missing" }));
    EXPECT_TRUE(contains_text(parser.errors(), "unknown preset"));
}

TEST(parser_configuration, help_filters_by_level_and_topic)
{
    auto       parser  = make_parser();
    const auto compact = parser.render_help();
    EXPECT_NE(compact.find("--output"), std::string::npos);
    EXPECT_NE(compact.find("--trace"), std::string::npos);
    EXPECT_EQ(compact.find("type:"), std::string::npos);
    EXPECT_NE(compact.find("ROCm Systems Profiler options"), std::string::npos);

    const auto full = parser.render_help(help_style::full);
    EXPECT_NE(full.find("--trace"), std::string::npos);
    EXPECT_NE(full.find("Enable Perfetto tracing"), std::string::npos);
    EXPECT_NE(full.find("type: bool"), std::string::npos);

    const auto tracing = parser.render_help(help_style::full, "tracing");
    EXPECT_NE(tracing.find("--trace"), std::string::npos);
    EXPECT_EQ(tracing.find("--output"), std::string::npos);

    const auto missing = parser.render_help(help_style::compact, "missing");
    EXPECT_NE(missing.find("No parameters match topic"), std::string::npos);
}

TEST(parser_configuration, exports_environment_and_json)
{
    auto       parser = make_parser();
    const auto values = parser.parse({ "--verbose", "4", "--output", "exported" });
    const auto env    = parser.export_as_environment();
    const auto text   = parser.export_as_json();
    EXPECT_EQ(values.at("verbose"), "4");
    EXPECT_EQ(env.at("ROCPROFSYS_VERBOSE"), "4");
    EXPECT_EQ(env.at("ROCPROFSYS_OUTPUT_PATH"), "exported");
    EXPECT_NE(text.find("\"output_path\""), std::string::npos);
    EXPECT_NE(text.find("exported"), std::string::npos);

    const auto previous = capture_environment(env);
    parser.apply_to_environment();
    EXPECT_STREQ(std::getenv("ROCPROFSYS_VERBOSE"), "4");
    EXPECT_STREQ(std::getenv("ROCPROFSYS_OUTPUT_PATH"), "exported");
    restore_environment(previous);
}

TEST(parser_configuration, reads_argc_and_reports_a_missing_schema)
{
    auto parser       = make_parser();
    auto program      = std::string{ "rocsys" };
    auto option       = std::string{ "--verbose" };
    auto option_value = std::string{ "2" };
    auto argv = std::array<char*, 4>{ program.data(), option.data(), option_value.data(),
                                      nullptr };
    const auto values = parser.parse(static_cast<int>(argv.size() - 1), argv.data());
    EXPECT_EQ(values.at("verbose"), "2");

    configuration_parser missing;
    EXPECT_FALSE(missing.initialize("/no/such/schema.json"));
    EXPECT_FALSE(missing.failure().empty());
    EXPECT_EQ(missing.render_help(), "parser is not initialized\n");
    EXPECT_FALSE(missing.get<std::string>("output_path").has_value());
    EXPECT_TRUE(missing.parse({}).empty());
    EXPECT_TRUE(contains_text(missing.errors(), "not initialized"));
}
