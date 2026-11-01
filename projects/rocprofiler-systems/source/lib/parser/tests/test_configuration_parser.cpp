// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/configuration_parser.hpp"
#include "parser/help.hpp"
#include "parser/preset_parser.hpp"
#include "parser/schema.hpp"
#include "parser/value_provider.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <stdlib.h>  // POSIX ::getenv, ::setenv, ::unsetenv
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace rocprofsys::parser;

namespace
{
constexpr std::size_t k_help_param_count = 7;
constexpr double      k_verbose_minimum  = 0.0;
constexpr double      k_verbose_maximum  = 4.0;

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

void
add_checked(schema& catalog, schema_param param)
{
    ASSERT_TRUE(catalog.add_param(std::move(param)));
}

void
add_behavior_params(schema& catalog)
{
    add_checked(catalog,
                schema_param{
                    .id          = "config_file",
                    .name        = "Config file",
                    .description = "Config file",
                    .type        = param_type::path,
                    .path        = path_kind::file,
                    .arg = arg_spec{ .name = "--config", .form = arg_form::value },
                });
    add_checked(catalog,
                schema_param{
                    .id          = "preset",
                    .name        = "Preset",
                    .description = "Preset",
                    .type        = param_type::string,
                    .arg = arg_spec{ .name = "--preset", .form = arg_form::value },
                });
    add_checked(catalog,
                schema_param{
                    .id            = "verbose",
                    .name          = "Verbose",
                    .description   = "Verbose",
                    .type          = param_type::integer,
                    .default_value = "0",
                    .min           = k_verbose_minimum,
                    .max           = k_verbose_maximum,
                    .env           = env_spec{ .name = "ROCPROFSYS_VERBOSE" },
                    .arg = arg_spec{ .name = "--verbose", .form = arg_form::value },
                });
    add_checked(catalog,
                schema_param{
                    .id            = "output_path",
                    .name          = "Output path",
                    .description   = "Output path",
                    .type          = param_type::path,
                    .default_value = "rocprof-sys-output",
                    .path          = path_kind::directory,
                    .env           = env_spec{ .name = "ROCPROFSYS_OUTPUT_PATH" },
                    .arg = arg_spec{ .name = "--output", .form = arg_form::value },
                });
    add_checked(catalog, schema_param{
                             .id            = "trace",
                             .name          = "Trace",
                             .description   = "Trace",
                             .type          = param_type::boolean,
                             .default_value = "false",
                             .env           = env_spec{ .name = "ROCPROFSYS_TRACE" },
                             .arg = arg_spec{ .name = "--trace", .form = arg_form::flag },
                         });
    add_checked(catalog,
                schema_param{
                    .id            = "profile",
                    .name          = "Profile",
                    .description   = "Profile",
                    .type          = param_type::boolean,
                    .default_value = "false",
                    .env           = env_spec{ .name = "ROCPROFSYS_PROFILE" },
                    .arg = arg_spec{ .name = "--profile", .form = arg_form::flag },
                });
    add_checked(catalog,
                schema_param{
                    .id            = "flat_profile",
                    .name          = "Flat profile",
                    .description   = "Flat profile",
                    .type          = param_type::boolean,
                    .default_value = "false",
                    .arg = arg_spec{ .name = "--flat-profile", .form = arg_form::flag },
                });
}

void
add_behavior_sampling(schema& catalog)
{
    add_checked(catalog,
                schema_param{
                    .id            = "sample_cputime",
                    .name          = "Sample CPU time",
                    .description   = "CPU time",
                    .type          = param_type::boolean,
                    .default_value = "false",
                    .arg = arg_spec{ .name = "--sample-cputime", .form = arg_form::flag },
                    .mutually_exclusive_with = { "sample_realtime" },
                });
    add_checked(catalog, schema_param{
                             .id            = "sample_realtime",
                             .name          = "Sample real time",
                             .description   = "Real time",
                             .type          = param_type::boolean,
                             .default_value = "true",
                             .arg           = arg_spec{ .name = "--sample-realtime",
                                                        .form = arg_form::flag },
                             .mutually_exclusive_with = { "sample_cputime" },
                         });
    add_checked(catalog,
                schema_param{
                    .id            = "rocm_domains",
                    .name          = "ROCm domains",
                    .description   = "ROCm domains",
                    .type          = param_type::string,
                    .default_value = "hip_runtime_api,kernel_dispatch,memory_copy",
                    .arg = arg_spec{ .name = "--rocm-domains", .form = arg_form::value },
                    .value_provider = "rocm_domains_sdk",
                });
}

schema
make_behavior_schema()
{
    schema catalog{};
    catalog.set_version("1.0");
    add_behavior_params(catalog);
    add_behavior_sampling(catalog);
    catalog.add_constraint(constraint{
        .kind          = constraint_kind::forces,
        .message       = "flat profiles require profiling",
        .when          = condition{ .terms = { param_condition{ .param  = "flat_profile",
                                                                .equals = "true" } } },
        .forced_values = { { "profile", "true" } },
    });
    return catalog;
}

configuration_parser
make_behavior_parser()
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    EXPECT_TRUE(parser.initialize(make_behavior_schema())) << parser.failure();
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

class letter_provider final : public value_provider_interface
{
public:
    [[nodiscard]] std::string_view name() const override { return "letters"; }

    [[nodiscard]] std::vector<std::string> choices(const schema_param&) const override
    {
        return { "alpha" };
    }

    [[nodiscard]] std::optional<std::string> default_value(
        const schema_param&) const override
    {
        return std::string{ "alpha" };
    }
};

schema
make_letter_schema()
{
    schema catalog{};
    static_cast<void>(catalog.add_category(category{ .id = "shown", .name = "Shown" }));
    add_checked(catalog, schema_param{
                             .id            = "trace",
                             .name          = "Trace",
                             .description   = "Trace",
                             .type          = param_type::boolean,
                             .default_value = "false",
                             .env           = env_spec{ .name = "ROCPROFSYS_TRACE" },
                         });
    add_checked(catalog,
                schema_param{
                    .id          = "letter",
                    .name        = "Letter",
                    .description = "Letter",
                    .category    = "shown",
                    .type        = param_type::enumeration,
                    .arg = arg_spec{ .name = "--letter", .form = arg_form::value },
                    .value_provider = "letters",
                });
    return catalog;
}

TEST(parser_configuration, built_in_catalog_is_help_text)
{
    auto       parser     = make_parser();
    const auto schema_ptr = parser.config_schema();
    ASSERT_NE(schema_ptr, nullptr);
    EXPECT_EQ(schema_ptr->version(), "1.0");
    EXPECT_EQ(schema_ptr->params().size(), k_help_param_count);

    const auto* preset      = schema_ptr->find_param("preset");
    const auto* config_file = schema_ptr->find_param("config_file");
    const auto* verbose     = schema_ptr->find_param("verbose");
    const auto* trace       = schema_ptr->find_param("trace");
    ASSERT_NE(preset, nullptr);
    ASSERT_NE(config_file, nullptr);
    ASSERT_NE(verbose, nullptr);
    ASSERT_NE(trace, nullptr);
    EXPECT_FALSE(preset->env.has_value());
    EXPECT_FALSE(config_file->env.has_value());
    EXPECT_EQ(verbose->env.value_or(env_spec{}).name, "ROCPROFSYS_VERBOSE");
    EXPECT_DOUBLE_EQ(verbose->min.value_or(0.0), -1.0);
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

    const auto quiet = parser.parse({ "--verbose", "-1" });
    EXPECT_EQ(quiet.at("verbose"), "-1");
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

schema
make_effect_schema()
{
    schema catalog{};
    add_checked(catalog,
                schema_param{
                    .id          = "profile",
                    .name        = "Profile",
                    .description = "Profile",
                    .type        = param_type::boolean,
                    .arg = arg_spec{ .name = "--profile", .form = arg_form::flag },
                });
    add_checked(catalog,
                schema_param{
                    .id            = "flat_profile",
                    .name          = "Flat profile",
                    .description   = "Flat profile",
                    .type          = param_type::boolean,
                    .default_value = "false",
                    .arg = arg_spec{ .name = "--flat-profile", .form = arg_form::flag },
                    .copy_to = { "profile" },
                });
    add_checked(catalog, schema_param{
                             .id          = "log_level",
                             .name        = "Log level",
                             .description = "Log level",
                             .type        = param_type::string,
                         });
    add_checked(catalog, schema_param{
                             .id            = "debug",
                             .name          = "Debug",
                             .description   = "Debug",
                             .type          = param_type::boolean,
                             .default_value = "false",
                             .arg = arg_spec{ .name = "--debug", .form = arg_form::flag },
                             .assigns = { { "log_level", "debug" } },
                         });
    add_checked(catalog,
                schema_param{
                    .id          = "verbose",
                    .name        = "Verbose",
                    .description = "Verbose",
                    .type        = param_type::integer,
                    .arg     = arg_spec{ .name = "--verbose", .form = arg_form::value },
                    .maps_to = { value_mapping{
                        .when = "2", .target = "log_level", .value = "verbose" } },
                });
    add_checked(catalog, schema_param{
                             .id          = "backend",
                             .name        = "Backend",
                             .description = "Backend",
                             .type        = param_type::string,
                         });
    add_checked(
        catalog,
        schema_param{
            .id          = "include",
            .name        = "Include",
            .description = "Include",
            .type        = param_type::enumeration,
            .choices =
                {
                    choice{ .value       = "perfetto",
                            .name        = "Perfetto",
                            .description = "Perfetto",
                            .sets        = { { "backend", "trace" } } },
                    choice{ .value       = "timemory",
                            .name        = "Timemory",
                            .description = "Timemory",
                            .sets        = { { "backend", "profile" } } },
                },
            .arg = arg_spec{ .name = "--include", .form = arg_form::value },
        });
    add_checked(catalog,
                schema_param{
                    .id            = "delay",
                    .name          = "Delay",
                    .description   = "Delay",
                    .type          = param_type::string,
                    .default_value = "0",
                    .arg = arg_spec{ .name = "--delay", .form = arg_form::value },
                });
    add_checked(catalog, schema_param{
                             .id            = "wait",
                             .name          = "Wait",
                             .description   = "Wait",
                             .type          = param_type::boolean,
                             .default_value = "false",
                             .arg = arg_spec{ .name = "--wait", .form = arg_form::flag },
                             .fills = { "delay" },
                         });
    return catalog;
}

configuration_parser
make_effect_parser()
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    EXPECT_TRUE(parser.initialize(make_effect_schema())) << parser.failure();
    return parser;
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

    static_cast<void>(parser.parse({ "--preset=" }));
    EXPECT_FALSE(contains_text(parser.errors(), "unknown preset"));
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
    EXPECT_EQ(env.find("ROCPROFSYS_TRACE"), env.end());
    EXPECT_EQ(env.find("ROCPROFSYS_DEBUG"), env.end());
    EXPECT_EQ(env.find("ROCPROFSYS_PROFILE"), env.end());
    EXPECT_NE(text.find("\"output_path\""), std::string::npos);
    EXPECT_NE(text.find("exported"), std::string::npos);

    const auto previous = capture_environment(env);
    parser.apply_to_environment();
    EXPECT_STREQ(std::getenv("ROCPROFSYS_VERBOSE"), "4");
    EXPECT_STREQ(std::getenv("ROCPROFSYS_OUTPUT_PATH"), "exported");
    restore_environment(previous);
}

TEST(parser_configuration, provider_default_keeps_the_other_values)
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    ASSERT_TRUE(parser.initialize(make_letter_schema())) << parser.failure();
    ASSERT_TRUE(parser.add_provider(std::make_unique<letter_provider>()));

    const auto values = parser.parse({});
    EXPECT_EQ(values.at("trace"), "false");
    EXPECT_EQ(values.at("letter"), "alpha");
    EXPECT_TRUE(parser.errors().empty()) << parser.errors().front();
}

TEST(parser_configuration, refused_provider_value_names_the_source)
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    ASSERT_TRUE(parser.initialize(make_letter_schema())) << parser.failure();
    ASSERT_TRUE(parser.add_provider(std::make_unique<letter_provider>()));

    static_cast<void>(parser.parse({ "--letter", "nope" }));
    EXPECT_TRUE(contains_text(parser.errors(), "command line"));
    EXPECT_EQ(parser.get<std::string>("letter").value_or(""), "alpha");
    EXPECT_EQ(parser.get<bool>("trace").value_or(true), false);
}

TEST(parser_configuration, unknown_provider_does_not_reject_every_value)
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    ASSERT_TRUE(parser.initialize(make_letter_schema())) << parser.failure();
    const auto values = parser.parse({ "--letter", "alpha" });
    EXPECT_EQ(values.at("letter"), "alpha");
    EXPECT_EQ(values.at("trace"), "false");
    EXPECT_TRUE(parser.errors().empty()) << parser.errors().front();
}

TEST(parser_configuration, failed_initialize_keeps_the_previous_catalog)
{
    auto   parser = make_parser();
    schema bad{};
    add_checked(bad, schema_param{
                         .id          = "orphan",
                         .name        = "Orphan",
                         .description = "Orphan",
                         .category    = "missing",
                         .type        = param_type::string,
                     });
    EXPECT_FALSE(parser.initialize(std::move(bad)));
    EXPECT_FALSE(parser.failure().empty());
    ASSERT_NE(parser.config_schema(), nullptr);
    EXPECT_NE(parser.config_schema()->find_param("verbose"), nullptr);
}

TEST(parser_configuration, help_shows_provider_choices_and_a_missing_category)
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    ASSERT_TRUE(parser.initialize(make_letter_schema())) << parser.failure();
    ASSERT_TRUE(parser.add_provider(std::make_unique<letter_provider>()));
    const auto full = parser.render_help(help_style::full);
    EXPECT_NE(full.find("--letter"), std::string::npos);
    EXPECT_NE(full.find("alpha"), std::string::npos);

    schema unregistered{};
    add_checked(unregistered,
                schema_param{
                    .id          = "orphan",
                    .name        = "Orphan",
                    .description = "Orphan",
                    .category    = "ghost",
                    .type        = param_type::string,
                    .arg         = arg_spec{ .name = "--orphan", .form = arg_form::flag },
                });
    const auto text = format_help(unregistered, help_style::compact, {});
    EXPECT_NE(text.find("--orphan"), std::string::npos);
    EXPECT_NE(text.find("ghost"), std::string::npos);
}

TEST(parser_configuration, effects_run_after_precedence)
{
    auto       parser    = make_effect_parser();
    const auto untouched = parser.parse({});
    EXPECT_EQ(untouched.find("profile"), untouched.end());
    EXPECT_EQ(untouched.find("log_level"), untouched.end());
    EXPECT_EQ(untouched.at("delay"), "0");

    const auto copied = parser.parse({ "--flat-profile" });
    EXPECT_EQ(copied.at("profile"), "true");
    EXPECT_EQ(parser.parse({ "--profile", "false", "--flat-profile" }).at("profile"),
              "true");

    EXPECT_EQ(parser.parse({ "--debug" }).at("log_level"), "debug");
    EXPECT_EQ(parser.parse({ "--verbose", "2" }).at("log_level"), "verbose");
    EXPECT_EQ(parser.parse({ "--include", "perfetto" }).at("backend"), "trace");

    EXPECT_EQ(parser.parse({ "--wait" }).at("delay"), "true");
    EXPECT_EQ(parser.parse({ "--wait", "--delay", "5" }).at("delay"), "5");
    EXPECT_TRUE(parser.errors().empty()) << parser.errors().front();
}

TEST(parser_configuration, reads_argc_and_reports_an_uninitialized_parser)
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
    EXPECT_TRUE(missing.failure().empty());
    EXPECT_EQ(missing.render_help(), "parser is not initialized\n");
    EXPECT_FALSE(missing.get<std::string>("output_path").has_value());
    EXPECT_TRUE(missing.parse({}).empty());
    EXPECT_TRUE(contains_text(missing.errors(), "not initialized"));
}
