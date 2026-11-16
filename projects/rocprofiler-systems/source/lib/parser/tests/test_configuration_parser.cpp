// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/configuration_parser.hpp"
#include "parser/help.hpp"
#include "parser/schema.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
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
constexpr std::size_t k_catalog_size = 7;

bool
contains_text(const std::vector<std::string>& errors, std::string_view text)
{
    return std::any_of(errors.begin(), errors.end(), [text](const std::string& item) {
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

    [[nodiscard]] std::optional<std::string> default_value(
        const schema_param&) const override
    {
        return "alpha";
    }
};

schema_param
named(std::string id, param_type type)
{
    auto param = schema_param{};
    param.id   = std::move(id);
    param.type = type;
    return param;
}

void
add_param(schema& value, schema_param param)
{
    ASSERT_TRUE(value.add_param(std::move(param)));
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
    trace.flag          = true;
    trace.arg           = "--trace";
    add_param(value, std::move(letter));
    add_param(value, std::move(trace));
    return value;
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
    debug.flag            = true;
    debug.arg             = "--debug";
    debug.env             = "ROCPROFSYS_DEBUG";
    debug.assigns         = { { "log_level", "debug" } };
    auto log_level        = named("log_level", param_type::string);
    log_level.category    = "logging";
    log_level.arg         = "--log-level";
    auto profile          = named("profile", param_type::boolean);
    profile.category      = "tracing";
    profile.default_value = "false";
    profile.flag          = true;
    profile.arg           = "--profile";
    auto flat             = named("flat_profile", param_type::boolean);
    flat.category         = "tracing";
    flat.flag             = true;
    flat.arg              = "--flat-profile";
    flat.required         = { "profile" };
    auto left             = named("left", param_type::boolean);
    left.category         = "tracing";
    left.flag             = true;
    left.arg              = "--left";
    left.conflicts        = { "right" };
    auto right            = named("right", param_type::boolean);
    right.category        = "tracing";
    right.flag            = true;
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
    EXPECT_EQ(*schema_ptr->find_param("verbose")->env, "ROCPROFSYS_VERBOSE");
    EXPECT_DOUBLE_EQ(*schema_ptr->find_param("verbose")->min, -1.0);
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
    parser.set_environment([](std::string_view name) -> std::optional<std::string> {
        if(name == "ROCPROFSYS_VERBOSE")
        {
            return "1";
        }
        return std::nullopt;
    });
    parser.set_preset_lookup([](std::string_view name) {
        EXPECT_EQ(name, "balanced");
        return preset_parser::settings_t{ { "ROCPROFSYS_OUTPUT_PATH", "from-preset" },
                                          { "ROCPROFSYS_PROFILE", "true" } };
    });

    const auto values = parser.parse({ "--config", file.path(), "--preset", "balanced",
                                       "--verbose", "3", "--", "./app" });
    EXPECT_TRUE(parser.errors().empty()) << parser.errors().front();
    EXPECT_EQ(values.at("output_path"), "from-preset");
    EXPECT_EQ(values.at("trace"), "true");
    EXPECT_EQ(values.at("profile"), "true");
    EXPECT_EQ(values.at("verbose"), "3");
    ASSERT_EQ(parser.command().size(), 1U);
    EXPECT_EQ(parser.command().front(), "./app");
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
    parser.set_environment([](std::string_view name) -> std::optional<std::string> {
        if(name == "ROCPROFSYS_OUTPUT_PATH")
        {
            return "from-env";
        }
        return std::nullopt;
    });
    parser.set_preset_lookup([](std::string_view) {
        return preset_parser::settings_t{ { "ROCPROFSYS_OUTPUT_PATH", "from-preset" } };
    });
    const auto values = parser.parse({ "--config", file.path(), "--preset", "balanced" });
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(values.at("output_path"), "from-preset");
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
    EXPECT_EQ(refused.at("letter"), "alpha");
    EXPECT_EQ(refused.at("trace"), "false");

    const auto accepted = parser.parse({ "--letter", "beta" });
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(accepted.at("letter"), "beta");
}

TEST(parser_configuration, unknown_backend_does_not_reject_every_value)
{
    auto       parser = make_letter_parser(false);
    const auto values = parser.parse({ "--letter", "alpha" });
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(values.at("letter"), "alpha");
    EXPECT_EQ(values.at("trace"), "false");
}

TEST(parser_configuration, backend_default_keeps_the_other_values)
{
    auto       parser = make_letter_parser(true);
    const auto values = parser.parse({});
    EXPECT_EQ(values.at("letter"), "alpha");
    EXPECT_EQ(values.at("trace"), "false");
}

TEST(parser_configuration, reports_constraints_and_unknown_presets)
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    ASSERT_TRUE(parser.initialize(effect_schema())) << parser.failure();

    static_cast<void>(parser.parse({ "--left", "--right" }));
    EXPECT_TRUE(contains_text(parser.errors(), "mutually exclusive"));

    static_cast<void>(parser.parse({ "--flat-profile" }));
    EXPECT_TRUE(contains_text(parser.errors(), "requires"));

    auto presets = make_parser();
    static_cast<void>(presets.parse({ "--preset", "missing" }));
    EXPECT_TRUE(contains_text(presets.errors(), "was not applied"));

    presets.set_preset_lookup(
        [](std::string_view) { return preset_parser::settings_t{}; });
    static_cast<void>(presets.parse({ "--preset", "empty" }));
    EXPECT_TRUE(contains_text(presets.errors(), "unknown preset"));

    static_cast<void>(presets.parse({ "--preset=" }));
    EXPECT_FALSE(contains_text(presets.errors(), "unknown preset"));
    EXPECT_FALSE(contains_text(presets.errors(), "was not applied"));
}

TEST(parser_configuration, help_filters_by_level_and_topic)
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

    schema catalog{};
    auto   mode        = named("mode", param_type::enumeration);
    mode.category      = "sampling";
    mode.arg           = "--mode";
    mode.description   = "Sampling mode";
    mode.choices       = { "realtime", "cputime" };
    auto hidden        = named("hidden", param_type::boolean);
    hidden.category    = "sampling";
    hidden.level       = param_level::advanced;
    hidden.flag        = true;
    hidden.arg         = "--hidden";
    hidden.description = "Hidden option";
    ASSERT_TRUE(catalog.add_param(std::move(mode)));
    ASSERT_TRUE(catalog.add_param(std::move(hidden)));

    configuration_parser levels;
    levels.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    ASSERT_TRUE(levels.initialize(std::move(catalog)));
    const auto short_help = levels.render_help();
    const auto long_help  = levels.render_help(param_level::advanced);
    EXPECT_NE(short_help.find("--mode"), std::string::npos);
    EXPECT_NE(short_help.find("choices: realtime cputime"), std::string::npos);
    EXPECT_EQ(short_help.find("--hidden"), std::string::npos);
    EXPECT_EQ(short_help.find("type:"), std::string::npos);
    EXPECT_NE(long_help.find("--hidden"), std::string::npos);
    EXPECT_NE(long_help.find("choices: realtime cputime"), std::string::npos);
    EXPECT_EQ(long_help.find("type:"), std::string::npos);
}

TEST(parser_configuration, exports_environment_and_json)
{
    auto parser = make_parser();
    parser.set_environment([](std::string_view name) -> std::optional<std::string> {
        if(name == "ROCPROFSYS_TRACE")
        {
            return "Y";
        }
        return std::nullopt;
    });
    static_cast<void>(parser.parse({ "--verbose", "4", "--output", "exported" }));
    const auto exported = parser.export_as_environment();
    EXPECT_EQ(exported.at("ROCPROFSYS_VERBOSE"), "4");
    EXPECT_EQ(exported.at("ROCPROFSYS_OUTPUT_PATH"), "exported");
    EXPECT_EQ(exported.at("ROCPROFSYS_TRACE"), "true");
    EXPECT_EQ(exported.find("ROCPROFSYS_DEBUG"), exported.end());
    EXPECT_EQ(exported.find("ROCPROFSYS_PROFILE"), exported.end());

    const auto document = nlohmann::json::parse(parser.export_as_json());
    EXPECT_EQ(document.at("configuration").at("output_path"), "exported");
    EXPECT_EQ(document.at("logging").at("verbose"), 4);
    EXPECT_EQ(document.at("tracing").at("trace"), true);
    EXPECT_EQ(document.at("logging").find("debug"), document.at("logging").end());

    parser.apply_to_environment();
    EXPECT_STREQ(::getenv("ROCPROFSYS_VERBOSE"), "4");
    EXPECT_STREQ(::getenv("ROCPROFSYS_OUTPUT_PATH"), "exported");
    ::unsetenv("ROCPROFSYS_VERBOSE");
    ::unsetenv("ROCPROFSYS_OUTPUT_PATH");
    ::unsetenv("ROCPROFSYS_TRACE");
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

TEST(parser_configuration, a_directory_config_path_throws)
{
    auto parser = make_parser();
    EXPECT_EQ(parser.parse({}).at("verbose"), "0");
    EXPECT_THROW(parser.parse({ "--config", ::testing::TempDir() }), std::runtime_error);
    EXPECT_EQ(parser.get<std::int64_t>("verbose"), 0);
}

TEST(parser_configuration, refuses_a_catalog_that_assigns_an_invalid_value)
{
    configuration_parser parser;
    parser.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    ASSERT_TRUE(parser.initialize(effect_schema())) << parser.failure();

    auto broken_catalog = effect_schema();
    auto broken         = named("broken", param_type::boolean);
    broken.category     = "logging";
    broken.flag         = true;
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
    parser.set_environment([](std::string_view name) -> std::optional<std::string> {
        if(name == "ROCPROFSYS_DEBUG")
        {
            return "true";
        }
        return std::nullopt;
    });
    ASSERT_TRUE(parser.initialize(effect_schema())) << parser.failure();

    const auto from_flag = parser.parse({ "--debug" });
    EXPECT_TRUE(parser.errors().empty());
    EXPECT_EQ(from_flag.at("log_level"), "debug");

    const auto kept = parser.parse({ "--log-level", "quiet" });
    EXPECT_EQ(kept.at("log_level"), "quiet");

    configuration_parser quiet;
    quiet.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    ASSERT_TRUE(quiet.initialize(effect_schema())) << quiet.failure();
    const auto defaults = quiet.parse({});
    EXPECT_EQ(defaults.find("log_level"), defaults.end());
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
    orphan.flag     = true;
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
