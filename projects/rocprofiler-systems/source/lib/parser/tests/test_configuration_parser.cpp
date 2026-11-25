// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/backend.hpp"
#include "parser/cli_parser.hpp"
#include "parser/configuration_parser.hpp"
#include "parser/env_parser.hpp"
#include "parser/help.hpp"
#include "parser/preset_parser.hpp"
#include "parser/schema.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>  // NOLINT(misc-include-cleaner): json::parse
#include <nlohmann/json_fwd.hpp>

#include <stdlib.h>  // POSIX ::getenv and ::unsetenv

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

using namespace rocprofsys::parser;

namespace
{
constexpr std::size_t k_catalog_size = 7;

bool
contains_text(const std::vector<std::string>& errors, std::string_view text)
{
    return std::ranges::any_of(errors, [text](const std::string& item) {
        return item.find(text) != std::string::npos;
    });
}

/// An environment that sets only @p variable, to @p value.
env_parser::lookup_t
environment_with(std::string variable, std::string value)
{
    return [variable = std::move(variable), value = std::move(value)](
               std::string_view name) -> std::optional<std::string> {
        return name == variable ? std::optional{ value } : std::nullopt;
    };
}

/// A preset registry that knows only the preset @p name.
configuration_parser::preset_lookup_t
preset_with(std::string name, preset_parser::settings_t settings)
{
    return
        [name = std::move(name), settings = std::move(settings)](std::string_view asked) {
            return asked == name ? settings : preset_parser::settings_t{};
        };
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

class letter_backend final : public backend_interface
{
public:
    [[nodiscard]] std::string_view name() const override { return "letters"; }

    [[nodiscard]] std::vector<std::string> choices(const schema_param&) const override
    {
        return { "alpha", "beta" };
    }
};

schema_param
named(std::string param_id, param_type type)
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

/// The catalog most tests parse: where output goes, the preset and config file, a
/// bounded verbosity, and three switches.
schema
demo_schema()
{
    constexpr double k_verbose_minimum = -1.0;
    constexpr double k_verbose_maximum = 4.0;

    const auto row = [](std::string param_id, std::string description,
                        std::string category, param_type type) {
        auto param        = named(std::move(param_id), type);
        param.description = std::move(description);
        param.category    = std::move(category);
        return param;
    };
    auto output = row("output_path", "Directory for output files", "configuration",
                      param_type::path);
    output.env  = "ROCPROFSYS_OUTPUT_PATH";
    output.arg  = "--output";
    auto preset =
        row("preset", "Named preset to apply", "configuration", param_type::string);
    preset.arg       = "--preset";
    auto config_file = row("config_file", "JSON file of parameter values",
                           "configuration", param_type::path);
    config_file.arg  = "--config";
    auto verbose =
        row("verbose", "Verbosity level from -1 to 4", "logging", param_type::integer);
    verbose.default_value = "0";
    verbose.min           = k_verbose_minimum;
    verbose.max           = k_verbose_maximum;
    verbose.env           = "ROCPROFSYS_VERBOSE";
    verbose.arg           = "--verbose";
    auto debug = row("debug", "Enable debug mode", "logging", param_type::boolean);
    debug.default_value = "false";
    debug.env           = "ROCPROFSYS_DEBUG";
    debug.arg           = "--debug";
    auto trace = row("trace", "Enable Perfetto tracing", "tracing", param_type::boolean);
    trace.default_value = "false";
    trace.env           = "ROCPROFSYS_TRACE";
    trace.arg           = "--trace";
    auto profile =
        row("profile", "Enable call-stack profiling", "tracing", param_type::boolean);
    profile.default_value = "false";
    profile.env           = "ROCPROFSYS_PROFILE";
    profile.arg           = "--profile";

    schema catalog{};
    catalog.set_version("1.0");
    for(auto& param : std::vector<schema_param>{
            std::move(output), std::move(preset), std::move(config_file),
            std::move(verbose), std::move(debug), std::move(trace), std::move(profile) })
    {
        add_param(catalog, std::move(param));
    }
    return catalog;
}

configuration_parser
make_parser()
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    EXPECT_TRUE(parser.initialize(demo_schema())) << parser.failure();
    return parser;
}

schema
letter_schema()
{
    schema value{};
    auto   letter       = named("letter", param_type::enumeration);
    letter.category     = "demo";
    letter.backend      = "letters";
    letter.arg          = "--letter";
    auto trace          = named("trace", param_type::boolean);
    trace.category      = "tracing";
    trace.default_value = "false";
    trace.arg           = "--trace";
    add_param(value, std::move(letter));
    add_param(value, std::move(trace));
    return value;
}

/// A basic "--mode" row and an advanced "--hidden" row.
schema
level_schema()
{
    schema value{};
    auto   mode        = named("mode", param_type::enumeration);
    mode.category      = "sampling";
    mode.arg           = "--mode";
    mode.description   = "Sampling mode";
    mode.choices       = { "realtime", "cputime" };
    auto hidden        = named("hidden", param_type::boolean);
    hidden.category    = "sampling";
    hidden.level       = param_level::advanced;
    hidden.arg         = "--hidden";
    hidden.description = "Hidden option";
    add_param(value, std::move(mode));
    add_param(value, std::move(hidden));
    return value;
}

schema_param
flag_row(std::string param_id, std::string category, std::string description)
{
    auto param        = named(std::move(param_id), param_type::boolean);
    param.arg         = "--" + param.id;
    param.category    = std::move(category);
    param.description = std::move(description);
    return param;
}

/// Rows of every shape the summary prints: an alias, no description, no arg, no
/// category, an advanced row beside basic ones, and a category with only an advanced row.
schema
summary_schema()
{
    schema value{};
    auto   format        = named("output_format", param_type::enumeration);
    format.category      = "output";
    format.arg           = "--output-format";
    format.arg_aliases   = { "-f" };
    format.description   = "Output format";
    format.choices       = { "pftrace", "proto" };
    auto hidden          = flag_row("hidden", "tracing", "Hidden");
    hidden.level         = param_level::advanced;
    auto dump            = flag_row("dump", "debug", "Dump state");
    dump.level           = param_level::advanced;
    auto verbose         = named("verbose", param_type::integer);
    verbose.arg          = "--verbose";
    verbose.description  = "Verbosity";
    auto internal        = named("internal", param_type::string);
    internal.description = "Internal setting";
    add_param(value, std::move(format));
    add_param(value, flag_row("trace", "tracing", "Enable tracing"));
    add_param(value, flag_row("quiet", "tracing", ""));
    add_param(value, std::move(hidden));
    add_param(value, std::move(dump));
    add_param(value, std::move(verbose));
    add_param(value, std::move(internal));
    return value;
}

/// config_file and preset with environment names, as a catalog may give them.
schema
selector_schema()
{
    schema value{};
    auto   config_file = named("config_file", param_type::path);
    config_file.arg    = "--config";
    config_file.env    = "ROCPROFSYS_CONFIG_FILE";
    auto preset        = named("preset", param_type::string);
    preset.arg         = "--preset";
    preset.env         = "ROCPROFSYS_PRESET";
    auto verbose       = named("verbose", param_type::integer);
    verbose.arg        = "--verbose";
    verbose.env        = "ROCPROFSYS_VERBOSE";
    add_param(value, std::move(config_file));
    add_param(value, std::move(preset));
    add_param(value, std::move(verbose));
    return value;
}

/// An environment whose ROCPROFSYS_CONFIG_FILE names a missing file and whose
/// ROCPROFSYS_PRESET names "quiet".
std::optional<std::string>
selector_environment(std::string_view name)
{
    if(name == "ROCPROFSYS_CONFIG_FILE")
    {
        return "missing.json";
    }
    if(name == "ROCPROFSYS_PRESET")
    {
        return "quiet";
    }
    return std::nullopt;
}

/// "quiet" sets verbose and names the preset "loud", which would set it again.
preset_parser::settings_t
chained_presets(std::string_view name)
{
    if(name == "quiet")
    {
        return { { "ROCPROFSYS_PRESET", "loud" }, { "ROCPROFSYS_VERBOSE", "-1" } };
    }
    if(name == "loud")
    {
        return { { "ROCPROFSYS_VERBOSE", "4" } };
    }
    return {};
}

configuration_parser
make_letter_parser(bool with_backend)
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    EXPECT_TRUE(parser.initialize(letter_schema())) << parser.failure();
    if(with_backend)
    {
        EXPECT_TRUE(parser.add_backend(std::make_unique<letter_backend>()));
    }
    return parser;
}

schema
effect_schema()
{
    schema value{};
    auto   debug          = named("debug", param_type::boolean);
    debug.category        = "logging";
    debug.default_value   = "false";
    debug.arg             = "--debug";
    debug.env             = "ROCPROFSYS_DEBUG";
    debug.assigns         = { { "log_level", "debug" } };
    auto log_level        = named("log_level", param_type::string);
    log_level.category    = "logging";
    log_level.arg         = "--log-level";
    auto profile          = named("profile", param_type::boolean);
    profile.category      = "tracing";
    profile.default_value = "false";
    profile.arg           = "--profile";
    auto flat             = named("flat_profile", param_type::boolean);
    flat.category         = "tracing";
    flat.arg              = "--flat-profile";
    flat.required         = { "profile" };
    auto left             = named("left", param_type::boolean);
    left.category         = "tracing";
    left.arg              = "--left";
    left.conflicts        = { "right" };
    auto right            = named("right", param_type::boolean);
    right.category        = "tracing";
    right.arg             = "--right";
    add_param(value, std::move(debug));
    add_param(value, std::move(log_level));
    add_param(value, std::move(profile));
    add_param(value, std::move(flat));
    add_param(value, std::move(left));
    add_param(value, std::move(right));
    return value;
}
}  // namespace

TEST(parser_configuration, built_in_catalog_is_help_text)
{
    auto       parser     = make_parser();
    const auto schema_ptr = parser.config_schema();
    ASSERT_NE(schema_ptr, nullptr);
    EXPECT_EQ(schema_ptr->version(), "1.0");
    EXPECT_EQ(schema_ptr->params().size(), k_catalog_size);
    EXPECT_FALSE(schema_ptr->find_param("preset")->env.has_value());
    EXPECT_FALSE(schema_ptr->find_param("config_file")->env.has_value());
    EXPECT_EQ(schema_ptr->find_param("verbose")->env.value_or(""), "ROCPROFSYS_VERBOSE");
    EXPECT_DOUBLE_EQ(schema_ptr->find_param("verbose")->min.value_or(0.0), -1.0);
    EXPECT_EQ(schema_ptr->find_param("trace")->category, "tracing");

    const auto values = parser.parse({});
    EXPECT_EQ(values.at("verbose"), "0");
    EXPECT_EQ(values.at("debug"), "false");
    EXPECT_EQ(values.at("trace"), "false");
    EXPECT_EQ(values.find("output_path"), values.end());
    EXPECT_TRUE(parser.errors().empty());

    const auto help = parser.render_help();
    EXPECT_EQ(help.find("--help"), std::string::npos);
    EXPECT_EQ(help.find("General"), std::string::npos);
    EXPECT_NE(help.find("configuration"), std::string::npos);
    EXPECT_EQ(help.find("ROCPROFSYS_PRESET"), std::string::npos);
    EXPECT_EQ(parser.parse({ "--verbose", "-1" }).at("verbose"), "-1");
}

TEST(parser_configuration, command_line_overrides_lower_sources)
{
    const scoped_temp_file file{ R"({
        "configuration": { "output_path": "from-file" },
        "tracing": { "trace": true }
    })" };
    auto                   parser = make_parser();
    parser.set_environment(environment_with("ROCPROFSYS_VERBOSE", "1"));
    parser.set_preset_lookup(
        preset_with("balanced", { { "ROCPROFSYS_OUTPUT_PATH", "from-preset" },
                                  { "ROCPROFSYS_PROFILE", "true" } }));

    const auto values = parser.parse({ "--config", file.path(), "--preset", "balanced",
                                       "--verbose", "3", "--", "./app" });
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(values, (value_map_t{ { "config_file", file.path() },
                                    { "debug", "false" },
                                    { "output_path", "from-preset" },
                                    { "preset", "balanced" },
                                    { "profile", "true" },
                                    { "trace", "true" },
                                    { "verbose", "3" } }));
    EXPECT_EQ(parser.command(), std::vector<std::string>{ "./app" });
}

TEST(parser_configuration, cli_output_overrides_preset_and_file)
{
    const scoped_temp_file file{
        R"({ "configuration": { "output_path": "from-file" } })"
    };
    auto parser = make_parser();
    parser.set_preset_lookup([](std::string_view) {
        return preset_parser::settings_t{ { "ROCPROFSYS_OUTPUT_PATH", "from-preset" } };
    });
    const auto values = parser.parse(
        { "--config", file.path(), "--preset", "balanced", "--output", "from-cli" });
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(values.at("output_path"), "from-cli");
}

TEST(parser_configuration, preset_overrides_the_file_and_environment)
{
    const scoped_temp_file file{
        R"({ "configuration": { "output_path": "from-file" } })"
    };
    auto parser = make_parser();
    parser.set_environment(environment_with("ROCPROFSYS_OUTPUT_PATH", "from-env"));
    parser.set_preset_lookup(
        preset_with("balanced", { { "ROCPROFSYS_OUTPUT_PATH", "from-preset" } }));
    const auto values = parser.parse({ "--config", file.path(), "--preset", "balanced" });
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(values.at("output_path"), "from-preset");
}

TEST(parser_configuration, a_config_file_cannot_choose_the_preset_or_another_file)
{
    const scoped_temp_file file{ R"({
        "configuration": {
            "config_file": "missing.json",
            "output_path": "from-file",
            "preset": "balanced"
        }
    })" };
    auto                   parser = make_parser();
    parser.set_preset_lookup(
        preset_with("balanced", { { "ROCPROFSYS_PROFILE", "true" } }));
    const auto values = parser.parse({ "--config", file.path() });
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(values.at("output_path"), "from-file");
    EXPECT_EQ(values.at("profile"), "false");
}

TEST(parser_configuration, only_the_command_line_chooses_the_file_and_the_preset)
{
    configuration_parser parser;
    parser.set_environment(selector_environment);
    ASSERT_TRUE(parser.initialize(selector_schema())) << parser.failure();
    parser.set_preset_lookup(chained_presets);

    const auto from_environment = parser.parse({});
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_FALSE(from_environment.contains("verbose"));

    const auto from_command_line = parser.parse({ "--preset", "quiet" });
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(from_command_line.at("verbose"), "-1");
}

TEST(parser_configuration, an_export_after_a_preset_reads_back_as_plain_settings)
{
    auto parser = make_parser();
    parser.set_preset_lookup(
        preset_with("balanced", { { "ROCPROFSYS_PROFILE", "true" } }));
    auto values = parser.parse({ "--preset", "balanced", "--verbose", "4" });
    ASSERT_TRUE(parser.errors().empty());

    const scoped_temp_file file{ parser.export_as_json() };
    auto                   from_file = make_parser();
    auto                   read_back = from_file.parse({ "--config", file.path() });
    EXPECT_TRUE(from_file.errors().empty());
    read_back.erase("config_file");
    EXPECT_EQ(read_back, values);
}

TEST(parser_configuration, unknown_options_are_forwarded_and_parsing_continues)
{
    auto parser = make_parser();
    parser.set_unknown_option(unknown_option::forward);
    const auto forwarded =
        parser.parse({ "--gpu=temp", "--output", "kept", "--not-an-option", "value",
                       "--debug", "--", "./app" });
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(forwarded.at("output_path"), "kept");
    EXPECT_EQ(forwarded.at("debug"), "true");
    EXPECT_EQ(parser.unparsed(),
              (std::vector<std::string>{ "--gpu=temp", "--not-an-option", "value" }));
    EXPECT_EQ(parser.command(), std::vector<std::string>{ "./app" });

    parser.set_unknown_option(unknown_option::report);
    static_cast<void>(parser.parse({ "--not-an-option" }));
    EXPECT_TRUE(contains_text(parser.errors(), "unknown option"));
    EXPECT_TRUE(parser.unparsed().empty());
}

TEST(parser_configuration, rejects_values_outside_the_schema)
{
    auto parser = make_parser();
    EXPECT_EQ(parser.parse({}).at("verbose"), "0");
    const auto values = parser.parse({ "--verbose", "9" });
    EXPECT_EQ(parser.get<std::int64_t>("verbose"), 0);
    EXPECT_EQ(values.at("verbose"), "0");
    EXPECT_TRUE(contains_text(parser.errors(), "verbose"));
}

TEST(parser_configuration, rejects_values_outside_a_backend_catalog)
{
    auto       parser  = make_letter_parser(true);
    const auto refused = parser.parse({ "--letter", "nope" });
    EXPECT_TRUE(contains_text(parser.errors(), "command line"));
    EXPECT_FALSE(refused.contains("letter"));
    EXPECT_EQ(refused.at("trace"), "false");

    const auto accepted = parser.parse({ "--letter", "beta" });
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(accepted.at("letter"), "beta");
}

TEST(parser_configuration, names_the_list_element_a_backend_refuses)
{
    auto catalog     = letter_schema();
    auto letters     = named("letters", param_type::enumeration_list);
    letters.category = "demo";
    letters.arg      = "--letters";
    letters.backend  = "letters";
    add_param(catalog, std::move(letters));
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    ASSERT_TRUE(parser.initialize(std::move(catalog))) << parser.failure();
    ASSERT_TRUE(parser.add_backend(std::make_unique<letter_backend>()));

    static_cast<void>(parser.parse({ "--letters", "alpha,nope" }));
    EXPECT_EQ(parser.errors(),
              (std::vector<std::string>{
                  "invalid value 'nope' for parameter 'letters' from command line" }));
}

TEST(parser_configuration, unknown_backend_does_not_reject_every_value)
{
    auto       parser = make_letter_parser(false);
    const auto values = parser.parse({ "--letter", "alpha" });
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(values.at("letter"), "alpha");
    EXPECT_EQ(values.at("trace"), "false");
}

TEST(parser_configuration, reports_conflicts_and_requirements)
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    ASSERT_TRUE(parser.initialize(effect_schema())) << parser.failure();

    static_cast<void>(parser.parse({ "--left", "--right" }));
    EXPECT_TRUE(contains_text(parser.errors(), "mutually exclusive"));

    static_cast<void>(parser.parse({ "--flat-profile" }));
    EXPECT_TRUE(contains_text(parser.errors(), "requires"));
}

TEST(parser_configuration, reports_a_preset_that_is_not_applied_or_unknown)
{
    auto parser = make_parser();
    static_cast<void>(parser.parse({ "--preset", "missing" }));
    EXPECT_TRUE(contains_text(parser.errors(), "was not applied"));

    parser.set_preset_lookup(preset_with("balanced", {}));
    static_cast<void>(parser.parse({ "--preset", "empty" }));
    EXPECT_TRUE(contains_text(parser.errors(), "unknown preset"));

    static_cast<void>(parser.parse({ "--preset=" }));
    EXPECT_TRUE(parser.errors().empty());
}

TEST(parser_configuration, help_can_show_one_topic)
{
    auto       parser = make_parser();
    const auto basic  = parser.render_help();
    EXPECT_NE(basic.find("--output"), std::string::npos);
    EXPECT_NE(basic.find("--trace"), std::string::npos);
    EXPECT_EQ(basic.find("type:"), std::string::npos);

    const auto tracing = parser.render_help(param_level::basic, "tracing");
    EXPECT_NE(tracing.find("--trace"), std::string::npos);
    EXPECT_EQ(tracing.find("--output"), std::string::npos);

    const auto missing = parser.render_help(param_level::basic, "missing");
    EXPECT_NE(missing.find("No parameters match topic"), std::string::npos);
}

TEST(parser_configuration, help_shows_an_advanced_row_only_at_the_advanced_level)
{
    configuration_parser parser;
    ASSERT_TRUE(parser.initialize(level_schema())) << parser.failure();
    const auto basic    = parser.render_help();
    const auto advanced = parser.render_help(param_level::advanced);
    EXPECT_NE(basic.find("--mode"), std::string::npos);
    EXPECT_NE(basic.find("choices: realtime cputime"), std::string::npos);
    EXPECT_EQ(basic.find("--hidden"), std::string::npos);
    EXPECT_NE(advanced.find("--hidden"), std::string::npos);
}

TEST(parser_configuration, exports_only_values_that_are_not_defaults)
{
    auto parser = make_parser();
    parser.set_environment(environment_with("ROCPROFSYS_TRACE", "Y"));
    static_cast<void>(parser.parse({ "--verbose", "4", "--output", "exported" }));

    EXPECT_EQ(parser.export_as_environment(),
              (value_map_t{ { "ROCPROFSYS_OUTPUT_PATH", "exported" },
                            { "ROCPROFSYS_TRACE", "true" },
                            { "ROCPROFSYS_VERBOSE", "4" } }));
    EXPECT_EQ(nlohmann::json::parse(parser.export_as_json()), nlohmann::json::parse(R"({
        "configuration": { "output_path": "exported" },
        "logging": { "verbose": 4 },
        "tracing": { "trace": true }
    })"));
}

TEST(parser_configuration, applies_the_export_to_the_process_environment)
{
    auto parser = make_parser();
    static_cast<void>(parser.parse({ "--verbose", "4", "--output", "exported" }));
    parser.apply_to_environment();
    EXPECT_STREQ(::getenv("ROCPROFSYS_VERBOSE"), "4");
    EXPECT_STREQ(::getenv("ROCPROFSYS_OUTPUT_PATH"), "exported");
    ::unsetenv("ROCPROFSYS_VERBOSE");
    ::unsetenv("ROCPROFSYS_OUTPUT_PATH");
}

TEST(parser_configuration, exported_values_read_back_the_same)
{
    auto       parser = make_parser();
    const auto values =
        parser.parse({ "--verbose", "4", "--output", "exported", "--trace" });
    ASSERT_TRUE(parser.errors().empty());

    const auto exported         = parser.export_as_environment();
    auto       from_environment = make_parser();
    from_environment.set_environment(
        [&exported](std::string_view name) -> std::optional<std::string> {
            const auto found = exported.find(name);
            return found == exported.end() ? std::nullopt
                                           : std::optional{ found->second };
        });
    EXPECT_EQ(from_environment.parse({}), values);

    const scoped_temp_file file{ parser.export_as_json() };
    auto                   from_file = make_parser();
    auto                   read_back = from_file.parse({ "--config", file.path() });
    EXPECT_TRUE(from_file.errors().empty());
    read_back.erase("config_file");
    EXPECT_EQ(read_back, values);
}

TEST(parser_configuration, flag_does_not_consume_the_command)
{
    auto       parser    = make_parser();
    const auto separated = parser.parse({ "--trace", "--", "true" });
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(separated.at("trace"), "true");
    ASSERT_EQ(parser.command().size(), 1U);
    EXPECT_EQ(parser.command().front(), "true");

    const auto following = parser.parse({ "--trace", "true" });
    EXPECT_EQ(following.at("trace"), "true");
    EXPECT_TRUE(parser.command().empty());
    EXPECT_TRUE(contains_text(parser.errors(), "unexpected argument"));
}

TEST(parser_configuration, a_directory_config_path_is_reported)
{
    auto       parser    = make_parser();
    const auto directory = ::testing::TempDir();
    const auto values    = parser.parse({ "--config", directory });
    ASSERT_EQ(parser.errors().size(), 1U);
    EXPECT_EQ(parser.errors().front(),
              "configuration path '" + directory + "' is a directory");
    EXPECT_EQ(values.at("verbose"), "0");
}

TEST(parser_configuration, parse_reads_argv_after_the_program_name)
{
    constexpr int k_argc  = 2;
    auto          program = std::string{ "tool" };
    auto          option  = std::string{ "--trace" };
    auto          argv = std::array<char*, 3>{ program.data(), option.data(), nullptr };

    auto parser = make_parser();
    EXPECT_EQ(parser.parse(k_argc, argv.data()).at("trace"), "true");
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(parser.parse(0, nullptr).at("trace"), "false");
    EXPECT_TRUE(parser.errors().empty());
}

TEST(parser_configuration, refuses_a_catalog_that_assigns_an_invalid_value)
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    ASSERT_TRUE(parser.initialize(effect_schema())) << parser.failure();

    auto broken_catalog = effect_schema();
    auto broken         = named("broken", param_type::boolean);
    broken.category     = "logging";
    broken.arg          = "--broken";
    broken.assigns      = { { "profile", "not-a-bool" } };
    add_param(broken_catalog, std::move(broken));
    EXPECT_FALSE(parser.initialize(std::move(broken_catalog)));
    EXPECT_NE(
        parser.failure().find("parameter 'broken' assigns invalid value 'not-a-bool' to "
                              "'profile'"),
        std::string::npos);
    EXPECT_EQ(parser.config_schema()->find_param("broken"), nullptr);
}

TEST(parser_configuration, assigns_run_after_precedence)
{
    configuration_parser parser;
    parser.set_environment(environment_with("ROCPROFSYS_DEBUG", "true"));
    ASSERT_TRUE(parser.initialize(effect_schema())) << parser.failure();

    EXPECT_EQ(parser.parse({ "--debug" }).at("log_level"), "debug");
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(parser.parse({ "--log-level", "quiet" }).at("log_level"), "quiet");
}

TEST(parser_configuration, a_default_assigns_nothing)
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    ASSERT_TRUE(parser.initialize(effect_schema())) << parser.failure();
    EXPECT_FALSE(parser.parse({}).contains("log_level"));
}

TEST(parser_configuration, assigns_need_the_option_on)
{
    std::optional<std::string> debug_variable;
    configuration_parser       parser;
    parser.set_environment(
        [&debug_variable](std::string_view name) -> std::optional<std::string> {
            return name == "ROCPROFSYS_DEBUG" ? debug_variable : std::nullopt;
        });
    ASSERT_TRUE(parser.initialize(effect_schema())) << parser.failure();

    EXPECT_EQ(parser.parse({ "--debug" }).at("log_level"), "debug");
    EXPECT_FALSE(parser.parse({ "--debug=false" }).contains("log_level"));
    debug_variable = "false";
    EXPECT_FALSE(parser.parse({}).contains("log_level"));
    debug_variable = "true";
    EXPECT_EQ(parser.parse({}).at("log_level"), "debug");
}

TEST(parser_configuration, a_second_parse_drops_what_the_first_one_set)
{
    auto       parser = make_parser();
    const auto first  = parser.parse({ "--output", "first", "--trace" });
    ASSERT_EQ(first.at("output_path"), "first");
    ASSERT_EQ(first.at("trace"), "true");
    const auto second = parser.parse({});
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_FALSE(second.contains("output_path"));
    EXPECT_FALSE(parser.get<std::string>("output_path").has_value());
    EXPECT_EQ(second.at("trace"), "false");
    EXPECT_EQ(parser.get<bool>("trace"), std::optional<bool>{ false });
}

TEST(parser_configuration,
     an_assign_that_contradicts_a_value_from_the_same_source_is_an_error)
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    ASSERT_TRUE(parser.initialize(effect_schema())) << parser.failure();

    const auto expect_contradiction = [&parser](const std::vector<std::string>& args) {
        const auto values = parser.parse(args);
        EXPECT_EQ(parser.errors(),
                  std::vector<std::string>{
                      "parameter 'debug' assigns 'debug' to 'log_level', which the "
                      "command line sets to 'quiet'" });
        EXPECT_EQ(values.at("log_level"), "quiet");
    };
    expect_contradiction({ "--debug", "--log-level", "quiet" });
    expect_contradiction({ "--log-level", "quiet", "--debug" });
    EXPECT_EQ(parser.parse({ "--debug", "--log-level", "debug" }).at("log_level"),
              "debug");
    EXPECT_TRUE(parser.errors().empty());
}

TEST(parser_configuration, an_assigned_value_the_backend_refuses_is_reported)
{
    auto catalog = letter_schema();
    auto all     = named("all_letters", param_type::boolean);
    all.category = "demo";
    all.arg      = "--all-letters";
    all.assigns  = { { "letter", "nope" } };
    add_param(catalog, std::move(all));
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    ASSERT_TRUE(parser.initialize(std::move(catalog))) << parser.failure();
    ASSERT_TRUE(parser.add_backend(std::make_unique<letter_backend>()));
    const auto values = parser.parse({ "--all-letters" });
    EXPECT_FALSE(values.contains("letter"));
    EXPECT_TRUE(contains_text(parser.errors(), "letter"));
}

TEST(parser_configuration, failed_initialize_keeps_the_previous_catalog)
{
    auto   parser = make_parser();
    schema bad{};
    auto   flag        = named("flag", param_type::boolean);
    flag.default_value = "not-a-bool";
    ASSERT_TRUE(bad.add_param(std::move(flag)));
    EXPECT_FALSE(parser.initialize(bad));
    EXPECT_FALSE(parser.failure().empty());
    EXPECT_NE(parser.config_schema()->find_param("verbose"), nullptr);
}

TEST(parser_configuration, help_shows_backend_choices_and_a_category_word)
{
    auto   parser   = make_letter_parser(true);
    schema extra    = letter_schema();
    auto   orphan   = named("orphan", param_type::boolean);
    orphan.category = "ghost";
    orphan.arg      = "--orphan";
    ASSERT_TRUE(extra.add_param(std::move(orphan)));
    ASSERT_TRUE(parser.initialize(extra));
    ASSERT_TRUE(parser.add_backend(std::make_unique<letter_backend>()));

    const auto help = parser.render_help();
    EXPECT_NE(help.find("--letter"), std::string::npos);
    EXPECT_NE(help.find("alpha"), std::string::npos);
    EXPECT_NE(help.find("ghost"), std::string::npos);
    EXPECT_NE(help.find("--orphan"), std::string::npos);
    EXPECT_EQ(help.find("--orphan <value>"), std::string::npos);

    const auto letter = parser.render_help(param_level::basic, "--letter");
    EXPECT_NE(letter.find("choices: alpha beta\n"), std::string::npos) << letter;
    EXPECT_EQ(letter.find("--orphan"), std::string::npos) << letter;
}

TEST(parser_configuration, help_starts_with_the_first_category)
{
    const auto help = make_parser().render_help();
    EXPECT_EQ(help.rfind("configuration\n", 0), 0U) << help;
}

TEST(parser_configuration, help_lists_arg_aliases)
{
    schema catalog{};
    auto   format      = named("output_format", param_type::enumeration);
    format.category    = "output";
    format.arg         = "--output-format";
    format.arg_aliases = { "-f" };
    format.choices     = { "pftrace", "proto" };
    ASSERT_TRUE(catalog.add_param(std::move(format)));
    configuration_parser parser;
    ASSERT_TRUE(parser.initialize(std::move(catalog))) << parser.failure();

    const auto help = parser.render_help();
    EXPECT_NE(help.find("  --output-format, -f <value>\n"), std::string::npos) << help;
    EXPECT_NE(help.find("choices: pftrace proto\n"), std::string::npos) << help;
}

TEST(parser_configuration, help_lists_a_rows_choices_then_its_backends)
{
    auto catalog = letter_schema();
    auto all     = named("letters_or_all", param_type::enumeration);
    all.category = "demo";
    all.arg      = "--letters-or-all";
    all.backend  = "letters";
    all.choices  = { "all" };
    add_param(catalog, std::move(all));
    configuration_parser parser;
    ASSERT_TRUE(parser.initialize(std::move(catalog))) << parser.failure();
    ASSERT_TRUE(parser.add_backend(std::make_unique<letter_backend>()));

    const auto help = parser.render_help();
    EXPECT_NE(help.find("choices: all alpha beta\n"), std::string::npos) << help;
    EXPECT_TRUE(parser.parse({ "--letters-or-all", "all" }).contains("letters_or_all"));
    EXPECT_TRUE(parser.errors().empty());
}

TEST(parser_configuration, summary_lists_each_row_on_one_line_under_its_category)
{
    EXPECT_EQ(format_summary(summary_schema()),
              "output\n"
              "  --output-format, -f  Output format\n"
              "\n"
              "tracing\n"
              "  --trace              Enable tracing\n"
              "  --quiet\n"
              "\n"
              "  --verbose            Verbosity\n"
              "  internal             Internal setting\n"
              "\n");
}

TEST(parser_configuration, summary_shows_advanced_rows_at_the_advanced_level)
{
    EXPECT_EQ(format_summary(summary_schema(), param_level::advanced),
              "output\n"
              "  --output-format, -f  Output format\n"
              "\n"
              "tracing\n"
              "  --trace              Enable tracing\n"
              "  --quiet\n"
              "  --hidden             Hidden\n"
              "\n"
              "debug\n"
              "  --dump               Dump state\n"
              "\n"
              "  --verbose            Verbosity\n"
              "  internal             Internal setting\n"
              "\n");
}

TEST(parser_configuration, help_topic_names_one_option_by_any_of_its_spellings)
{
    configuration_parser parser;
    ASSERT_TRUE(parser.initialize(summary_schema())) << parser.failure();
    const auto* expected = "output\n"
                           "  --output-format, -f <value>\n"
                           "      Output format\n"
                           "      choices: pftrace proto\n"
                           "\n";
    EXPECT_EQ(parser.render_help(param_level::basic, "--output-format"), expected);
    EXPECT_EQ(parser.render_help(param_level::basic, "-f"), expected);
    EXPECT_EQ(parser.render_help(param_level::basic, "--hidden"), "tracing\n"
                                                                  "  --hidden\n"
                                                                  "      Hidden\n"
                                                                  "\n");
}

TEST(parser_configuration, help_leaves_out_a_category_with_no_row_shown)
{
    configuration_parser parser;
    ASSERT_TRUE(parser.initialize(summary_schema())) << parser.failure();
    EXPECT_EQ(parser.render_help().find("debug\n"), std::string::npos);
    EXPECT_NE(parser.render_help(param_level::advanced).find("debug\n  --dump\n"),
              std::string::npos);
    EXPECT_NE(parser.render_help(param_level::basic, "debug").find("No parameters match"),
              std::string::npos);
}

TEST(parser_configuration, reads_argc_and_reports_an_uninitialized_parser)
{
    configuration_parser bare;
    EXPECT_TRUE(bare.parse({}).empty());
    EXPECT_TRUE(contains_text(bare.errors(), "parser is not initialized"));
    EXPECT_FALSE(bare.add_backend(std::make_unique<letter_backend>()));

    auto                 parser = make_parser();
    std::array<char*, 4> argv{ const_cast<char*>("tool"), const_cast<char*>("--verbose"),
                               const_cast<char*>("2"), nullptr };
    const auto           values = parser.parse(3, argv.data());
    EXPECT_EQ(values.at("verbose"), "2");
}
