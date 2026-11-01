// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/cli_parser.hpp"
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

TEST(parser_default, stores_the_default_without_checking_the_catalog)
{
    schema value{};
    auto   param        = make_param("freq", param_type::integer);
    param.max           = k_frequency_limit;
    param.default_value = "9";
    add_param(value, std::move(param));

    const auto parsed = default_parser{}.parse(value);
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("freq"), "9");
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
        return std::string{ "nope" };
    };
    const auto parsed = env_parser{ lookup }.parse(make_schema());
    EXPECT_TRUE(parsed.values.empty());
    ASSERT_EQ(parsed.errors.size(), 2U);
}

TEST(parser_env, trims_values_and_skips_an_empty_list)
{
    schema value{};
    auto   freq    = make_param("freq", param_type::integer);
    freq.env       = env_spec{ .name = "ROCPROFSYS_FREQ" };
    auto events    = make_param("events", param_type::enumeration_list);
    events.env     = env_spec{ .name = "ROCPROFSYS_EVENTS" };
    events.choices = { choice{ .value = "ins" } };
    auto trace     = make_param("trace", param_type::boolean);
    trace.env      = env_spec{ .name = "ROCPROFSYS_TRACE" };
    auto label     = make_param("label", param_type::string);
    label.env      = env_spec{ .name = "ROCPROFSYS_LABEL" };
    auto path      = make_param("output", param_type::path);
    path.env       = env_spec{ .name = "ROCPROFSYS_OUTPUT" };
    add_param(value, std::move(freq));
    add_param(value, std::move(events));
    add_param(value, std::move(trace));
    add_param(value, std::move(label));
    add_param(value, std::move(path));

    const auto lookup = [](std::string_view name) -> std::optional<std::string> {
        if(name == "ROCPROFSYS_FREQ")
        {
            return std::string{ "  2  " };
        }
        if(name == "ROCPROFSYS_EVENTS" || name == "ROCPROFSYS_LABEL")
        {
            return std::string{ "   " };
        }
        if(name == "ROCPROFSYS_TRACE")
        {
            return std::string{};
        }
        if(name == "ROCPROFSYS_OUTPUT")
        {
            return std::string{};
        }
        return std::nullopt;
    };

    const auto parsed = env_parser{ lookup }.parse(value);
    EXPECT_EQ(parsed.values.at("freq"), "2");
    EXPECT_EQ(parsed.values.find("events"), parsed.values.end());
    EXPECT_EQ(parsed.values.at("trace"), "");
    EXPECT_EQ(parsed.values.at("label"), "");
    ASSERT_EQ(parsed.errors.size(), 1U);
    EXPECT_NE(parsed.errors.front().find("ROCPROFSYS_OUTPUT"), std::string::npos);
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

schema
make_event_schema()
{
    schema value{};
    add_trace(value);
    auto events    = make_param("events", param_type::enumeration_list);
    events.env     = env_spec{ .name = "ROCPROFSYS_EVENTS" };
    events.arg     = arg_spec{ .name = "--events", .min_count = 1, .max_count = 3 };
    events.choices = { choice{ .value = "ins" },
                       choice{ .value = "cyc", .aliases = { "cycles" } } };
    auto format    = make_param("format", param_type::enumeration);
    format.env     = env_spec{ .name = "ROCPROFSYS_FORMAT" };
    format.arg     = arg_spec{ .name = "--format" };
    format.choices = { choice{ .value = "pftrace", .aliases = { "proto" } } };
    auto output    = make_param("output", param_type::path);
    output.arg     = arg_spec{ .name      = "--output",
                               .min_count = 1,
                               .max_count = 2,
                               .binds     = { "output", "prefix" } };
    auto prefix    = make_param("prefix", param_type::string);
    add_param(value, std::move(events));
    add_param(value, std::move(format));
    add_param(value, std::move(output));
    add_param(value, std::move(prefix));
    return value;
}

TEST(parser_cli, joins_tokens_until_the_next_option)
{
    const auto parsed =
        cli_parser{ { "--events", "ins", "cycles", "--trace" }, unknown_option::report }
            .parse(make_event_schema());

    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("events"), "ins,cyc");
    EXPECT_EQ(parsed.values.at("trace"), "true");
}

TEST(parser_cli, binds_each_token_to_its_parameter)
{
    const auto parsed =
        cli_parser{ { "--output", "/tmp/out", "run1" }, unknown_option::report }.parse(
            make_event_schema());

    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("output"), "/tmp/out");
    EXPECT_EQ(parsed.values.at("prefix"), "run1");
}

TEST(parser_cli, does_not_take_the_next_option_as_a_value)
{
    const auto parsed = parse_cli({ "--output", "--trace" });

    ASSERT_EQ(parsed.errors.size(), 1U);
    EXPECT_NE(parsed.errors.front().find("requires a value"), std::string::npos);
    EXPECT_EQ(parsed.values.find("output"), parsed.values.end());
    EXPECT_EQ(parsed.values.at("trace"), "true");
}

TEST(parser_cli, stores_a_choice_alias_as_the_choice_value)
{
    const auto parsed =
        cli_parser{ { "--format", "proto" }, unknown_option::report }.parse(
            make_event_schema());

    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("format"), "pftrace");
}

TEST(parser_env, stores_a_choice_alias_as_the_choice_value)
{
    const auto lookup = [](std::string_view name) -> std::optional<std::string> {
        if(name == "ROCPROFSYS_FORMAT")
        {
            return std::string{ "proto" };
        }
        if(name == "ROCPROFSYS_EVENTS")
        {
            return std::string{ "cycles,ins" };
        }
        return std::nullopt;
    };

    const auto parsed = env_parser{ lookup }.parse(make_event_schema());
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("format"), "pftrace");
    EXPECT_EQ(parsed.values.at("events"), "cyc,ins");
}

TEST(parser_cli, accepts_a_negative_number_and_rejects_overflow)
{
    schema value{};
    auto   param = make_param("num", param_type::integer);
    param.arg    = arg_spec{ .name = "--num" };
    add_param(value, std::move(param));

    const auto negative =
        cli_parser{ { "--num", "-5" }, unknown_option::report }.parse(value);
    EXPECT_TRUE(negative.errors.empty());
    EXPECT_EQ(negative.values.at("num"), "-5");

    const auto overflow =
        cli_parser{ { "--num", "9223372036854775808" }, unknown_option::report }.parse(
            value);
    EXPECT_TRUE(overflow.values.empty());
    ASSERT_EQ(overflow.errors.size(), 1U);
}

TEST(parser_cli, flag_accepts_a_following_boolean_token)
{
    const auto disabled = parse_cli({ "--trace", "false" });
    const auto enabled  = parse_cli({ "--trace", "--freq", "2" });

    EXPECT_TRUE(disabled.errors.empty());
    EXPECT_EQ(disabled.values.at("trace"), "false");
    EXPECT_TRUE(disabled.command.empty());
    EXPECT_EQ(enabled.values.at("trace"), "true");
    EXPECT_EQ(enabled.values.at("freq"), "2");
}

TEST(parser_cli, accepts_an_empty_equals_value)
{
    schema value{};
    auto   param = make_param("name", param_type::string);
    param.arg    = arg_spec{ .name = "--name" };
    add_param(value, std::move(param));

    const auto parsed = cli_parser{ { "--name=" }, unknown_option::report }.parse(value);
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("name"), "");
}

TEST(parser_cli, accepts_a_short_flag_bundle)
{
    schema value{};
    add_short_flags(value);
    auto json = make_param("json", param_type::boolean);
    json.arg  = arg_spec{ .name = "-j", .form = arg_form::flag };
    add_param(value, std::move(json));

    const auto parsed = cli_parser{ { "-qj" }, unknown_option::report }.parse(value);
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.at("quiet"), "true");
    EXPECT_EQ(parsed.values.at("json"), "true");
}

TEST(parser_cli, plain_argument_starts_the_command)
{
    const auto parsed = parse_cli({ "--bogus", "5" });
    ASSERT_EQ(parsed.errors.size(), 1U);
    EXPECT_NE(parsed.errors.front().find("--bogus"), std::string::npos);
    EXPECT_EQ(parsed.command, (std::vector<std::string>{ "5" }));
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
