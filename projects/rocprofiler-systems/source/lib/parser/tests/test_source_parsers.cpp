// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/cli_parser.hpp"
#include "parser/config_store.hpp"
#include "parser/default_parser.hpp"
#include "parser/env_parser.hpp"
#include "parser/parse_result.hpp"
#include "parser/schema.hpp"

#include <gtest/gtest.h>

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace rocprofsys::parser;

namespace
{
constexpr double k_frequency_limit = 3.0;
constexpr int    k_argc            = 2;

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
add_trace(schema& value)
{
    auto param          = make_param("trace", param_type::boolean);
    param.default_value = "true";
    param.env           = env_spec{ .name = "ROCPROFSYS_TRACE" };
    param.arg           = arg_spec{ .name = "--trace", .form = arg_form::flag };
    add_param(value, std::move(param));
}

void
add_short_flags(schema& value)
{
    auto quiet   = make_param("quiet", param_type::boolean);
    quiet.arg    = arg_spec{ .name = "-q", .form = arg_form::flag };
    auto verbose = make_param("verbose", param_type::boolean);
    verbose.arg  = arg_spec{ .name = "-v", .form = arg_form::flag };
    add_param(value, std::move(quiet));
    add_param(value, std::move(verbose));
}

void
add_values(schema& value)
{
    auto output = make_param("output", param_type::path);
    output.arg  = arg_spec{ .name = "--output", .style = assign_style::equals };
    auto freq   = make_param("freq", param_type::integer);
    freq.max    = k_frequency_limit;
    freq.env    = env_spec{ .name = "ROCPROFSYS_FREQ" };
    freq.arg    = arg_spec{ .name = "--freq", .style = assign_style::separate };
    auto label  = make_param("label", param_type::string);
    add_param(value, std::move(output));
    add_param(value, std::move(freq));
    add_param(value, std::move(label));
}

schema
make_schema()
{
    schema value{};
    add_trace(value);
    add_short_flags(value);
    add_values(value);
    return value;
}

parse_result
parse_cli(std::vector<std::string> args, unknown_option unknown = unknown_option::report)
{
    return cli_parser{ std::move(args), unknown }.parse(make_schema());
}
}  // namespace

TEST(parser_default, reads_only_parameters_with_defaults)
{
    const auto parsed = default_parser{}.parse(make_schema());

    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values, (value_map_t{ { "trace", "true" } }));
}

TEST(parser_default, reports_an_invalid_default)
{
    schema value{};
    auto   param        = make_param("freq", param_type::integer);
    param.max           = k_frequency_limit;
    param.default_value = "9";
    add_param(value, std::move(param));

    const auto parsed = default_parser{}.parse(value);
    EXPECT_TRUE(parsed.values.empty());
    ASSERT_EQ(parsed.errors.size(), 1U);
    EXPECT_NE(parsed.errors.front().find("freq"), std::string::npos);
}

TEST(parser_env, reads_set_variables_and_skips_unset_ones)
{
    const std::map<std::string, std::string> environment{ { "ROCPROFSYS_FREQ", "2" } };
    const auto                               lookup =
        [&environment](std::string_view name) -> std::optional<std::string> {
        const auto found = environment.find(std::string{ name });
        return found == environment.end() ? std::nullopt
                                          : std::optional<std::string>{ found->second };
    };

    const auto parsed = env_parser{ lookup }.parse(make_schema());
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values, (value_map_t{ { "freq", "2" } }));
}

TEST(parser_env, reports_an_invalid_variable)
{
    const auto lookup = [](std::string_view) -> std::optional<std::string> {
        return std::string{ "9" };
    };
    const auto parsed = env_parser{ lookup }.parse(make_schema());
    EXPECT_TRUE(parsed.values.empty());
    ASSERT_EQ(parsed.errors.size(), 2U);
}

TEST(parser_env, apply_parsed_records_the_environment_source)
{
    const auto lookup = [](std::string_view name) -> std::optional<std::string> {
        return name == "ROCPROFSYS_TRACE" ? std::optional<std::string>{ "yes" }
                                          : std::nullopt;
    };
    const auto parsed = env_parser{ lookup }.parse(make_schema());
    auto       store  = config_store{ make_schema() };
    apply_parsed(store, parsed, config_source::environment);

    EXPECT_EQ(store.get_as<bool>("trace"), true);
    EXPECT_EQ(store.source_of("trace"), config_source::environment);
}

TEST(parser_cli, accepts_separate_and_equals_values)
{
    const auto parsed = parse_cli({ "--output", "/tmp/out", "--freq=2" });
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("output"), "/tmp/out");
    EXPECT_EQ(parsed.values.at("freq"), "2");
}

TEST(parser_cli, flag_defaults_to_true_and_accepts_an_explicit_value)
{
    const auto enabled  = parse_cli({ "--trace" });
    const auto disabled = parse_cli({ "--trace=false" });
    EXPECT_EQ(enabled.values.at("trace"), "true");
    EXPECT_EQ(disabled.values.at("trace"), "false");
}

TEST(parser_cli, accepts_short_options_and_flag_bundles)
{
    const auto parsed = parse_cli({ "-qv" });
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("quiet"), "true");
    EXPECT_EQ(parsed.values.at("verbose"), "true");
}

TEST(parser_cli, keeps_the_command_after_the_separator)
{
    const auto parsed = parse_cli({ "--trace", "--", "./app", "--freq=9" });
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("trace"), "true");
    EXPECT_EQ(parsed.command, (std::vector<std::string>{ "./app", "--freq=9" }));
    EXPECT_EQ(parsed.values.find("freq"), parsed.values.end());
}

TEST(parser_cli, stops_at_the_first_unknown_option)
{
    const auto parsed =
        parse_cli({ "--trace", "--sampling-freq", "100" }, unknown_option::stop);
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("trace"), "true");
    EXPECT_EQ(parsed.unparsed, (std::vector<std::string>{ "--sampling-freq", "100" }));
}

TEST(parser_cli, reports_unknown_options_and_continues)
{
    const auto parsed = parse_cli({ "--bogus", "--freq", "2" });
    ASSERT_EQ(parsed.errors.size(), 1U);
    EXPECT_NE(parsed.errors.front().find("--bogus"), std::string::npos);
    EXPECT_EQ(parsed.values.at("freq"), "2");
}

TEST(parser_cli, reports_a_missing_or_invalid_value)
{
    const auto missing = parse_cli({ "--output" });
    const auto invalid = parse_cli({ "--freq", "9" });
    ASSERT_EQ(missing.errors.size(), 1U);
    EXPECT_NE(missing.errors.front().find("requires a value"), std::string::npos);
    ASSERT_EQ(invalid.errors.size(), 1U);
    EXPECT_TRUE(invalid.values.empty());
}

TEST(parser_cli, last_repeated_option_wins)
{
    const auto parsed = parse_cli({ "--freq", "1", "--freq=2" });
    EXPECT_EQ(parsed.values.at("freq"), "2");
}

TEST(parser_cli, arguments_from_skips_the_program_name)
{
    // arguments_from() takes the same char** argv a program receives.
    char  program[] = "tool";                      // NOLINT(modernize-avoid-c-arrays)
    char  flag[]    = "--trace";                   // NOLINT(modernize-avoid-c-arrays)
    char* argv[]    = { program, flag, nullptr };  // NOLINT(modernize-avoid-c-arrays)

    const auto args = cli_parser::arguments_from(k_argc, argv);
    EXPECT_EQ(args, std::vector<std::string>{ "--trace" });
    EXPECT_TRUE(cli_parser::arguments_from(0, nullptr).empty());
}
