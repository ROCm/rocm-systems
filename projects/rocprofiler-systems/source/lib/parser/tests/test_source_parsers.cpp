// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/cli_parser.hpp"
#include "parser/default_parser.hpp"
#include "parser/env_parser.hpp"
#include "parser/parse_result.hpp"
#include "parser/schema.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
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
expect_unset(const typed_values_t& values, std::initializer_list<std::string_view> ids)
{
    for(const auto param_id : ids)
    {
        EXPECT_EQ(values.find(param_id), values.end());
    }
}

template <typename ValueT>
void
expect_stored(const typed_values_t& values, std::string_view id, const ValueT& expected)
{
    const auto found = values.find(id);
    ASSERT_NE(found, values.end()) << id;
    const auto* held = std::get_if<ValueT>(&found->second);
    ASSERT_NE(held, nullptr) << id;
    EXPECT_EQ(*held, expected);
}

[[nodiscard]] std::optional<std::string>
blank_environment(std::string_view name)
{
    if(name == "ROCPROFSYS_FREQ")
    {
        return std::string{ "  2  " };
    }
    if(name == "ROCPROFSYS_EVENTS" || name == "ROCPROFSYS_LABEL")
    {
        return std::string{ "   " };
    }
    if(name == "ROCPROFSYS_TRACE" || name == "ROCPROFSYS_OUTPUT")
    {
        return std::string{};
    }
    return std::nullopt;
}

void
add_trace(schema& value)
{
    auto param          = make_param("trace", param_type::boolean);
    param.default_value = "true";
    param.env           = "ROCPROFSYS_TRACE";
    param.arg           = "--trace";
    param.flag          = true;
    add_param(value, std::move(param));
}

void
add_short_flags(schema& value)
{
    auto quiet   = make_param("quiet", param_type::boolean);
    quiet.arg    = "-q";
    quiet.flag   = true;
    auto verbose = make_param("verbose", param_type::boolean);
    verbose.arg  = "-v";
    verbose.flag = true;
    add_param(value, std::move(quiet));
    add_param(value, std::move(verbose));
}

void
add_values(schema& value)
{
    auto output = make_param("output", param_type::path);
    output.arg  = "--output";
    auto freq   = make_param("freq", param_type::integer);
    freq.max    = k_frequency_limit;
    freq.env    = "ROCPROFSYS_FREQ";
    freq.arg    = "--freq";
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
    EXPECT_EQ(parsed.values, (typed_values_t{ { "trace", true } }));
}

TEST(parser_default, rejects_a_default_the_catalog_refuses)
{
    schema value{};
    auto   param        = make_param("freq", param_type::integer);
    param.max           = k_frequency_limit;
    param.default_value = "9";
    add_param(value, std::move(param));

    const auto parsed = default_parser{}.parse(value);
    EXPECT_TRUE(parsed.values.empty());
    ASSERT_EQ(parsed.errors.size(), 1U);
    EXPECT_NE(parsed.errors.front().find("invalid value '9' for parameter 'freq'"),
              std::string::npos);
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
    EXPECT_EQ(parsed.values, (typed_values_t{ { "freq", std::int64_t{ 2 } } }));
}

TEST(parser_env, stores_a_boolean_word_as_a_bool)
{
    const auto lookup = [](std::string_view name) -> std::optional<std::string> {
        if(name == "ROCPROFSYS_TRACE")
        {
            return std::string{ "yes" };
        }
        return std::nullopt;
    };

    const auto parsed = env_parser{ lookup }.parse(make_schema());
    EXPECT_TRUE(parsed.errors.empty());
    expect_stored(parsed.values, "trace", true);
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
    freq.env       = "ROCPROFSYS_FREQ";
    auto events    = make_param("events", param_type::enumeration_list);
    events.env     = "ROCPROFSYS_EVENTS";
    events.choices = { "ins" };
    auto trace     = make_param("trace", param_type::boolean);
    trace.env      = "ROCPROFSYS_TRACE";
    auto label     = make_param("label", param_type::string);
    label.env      = "ROCPROFSYS_LABEL";
    auto path      = make_param("output", param_type::path);
    path.env       = "ROCPROFSYS_OUTPUT";
    add_param(value, std::move(freq));
    add_param(value, std::move(events));
    add_param(value, std::move(trace));
    add_param(value, std::move(label));
    add_param(value, std::move(path));

    const auto parsed = env_parser{ blank_environment }.parse(value);
    expect_stored(parsed.values, "freq", std::int64_t{ 2 });
    expect_unset(parsed.values, { "events", "trace", "label", "output" });
    EXPECT_TRUE(parsed.errors.empty());
}

TEST(parser_cli, accepts_separate_and_equals_values)
{
    const auto parsed = parse_cli({ "--output", "/tmp/out", "--freq=2" });
    EXPECT_TRUE(parsed.errors.empty());
    expect_stored(parsed.values, "output", std::string{ "/tmp/out" });
    expect_stored(parsed.values, "freq", std::int64_t{ 2 });
}

TEST(parser_cli, flag_defaults_to_true_and_accepts_an_explicit_value)
{
    const auto enabled  = parse_cli({ "--trace" });
    const auto disabled = parse_cli({ "--trace=false" });
    expect_stored(enabled.values, "trace", true);
    expect_stored(disabled.values, "trace", false);
}

TEST(parser_cli, accepts_short_options_and_flag_bundles)
{
    const auto parsed = parse_cli({ "-qv" });
    EXPECT_TRUE(parsed.errors.empty());
    expect_stored(parsed.values, "quiet", true);
    expect_stored(parsed.values, "verbose", true);
}

TEST(parser_cli, keeps_the_command_after_the_separator)
{
    const auto parsed = parse_cli({ "--trace", "--", "./app", "--freq=9" });
    EXPECT_TRUE(parsed.errors.empty());
    expect_stored(parsed.values, "trace", true);
    EXPECT_EQ(parsed.command, (std::vector<std::string>{ "./app", "--freq=9" }));
    EXPECT_EQ(parsed.values.find("freq"), parsed.values.end());
}

TEST(parser_cli, stops_at_the_first_unknown_option)
{
    const auto parsed =
        parse_cli({ "--trace", "--sampling-freq", "100" }, unknown_option::stop);
    EXPECT_TRUE(parsed.errors.empty());
    expect_stored(parsed.values, "trace", true);
    EXPECT_EQ(parsed.unparsed, (std::vector<std::string>{ "--sampling-freq", "100" }));
}

TEST(parser_cli, reports_unknown_options_and_continues)
{
    const auto parsed = parse_cli({ "--bogus", "--freq", "2" });
    ASSERT_EQ(parsed.errors.size(), 1U);
    EXPECT_NE(parsed.errors.front().find("--bogus"), std::string::npos);
    expect_stored(parsed.values, "freq", std::int64_t{ 2 });
}

TEST(parser_cli, reports_a_missing_or_invalid_value)
{
    const auto missing = parse_cli({ "--output" });
    const auto invalid = parse_cli({ "--freq", "9" });
    ASSERT_EQ(missing.errors.size(), 1U);
    EXPECT_NE(missing.errors.front().find("requires a value"), std::string::npos);
    ASSERT_EQ(invalid.errors.size(), 1U);
    EXPECT_NE(invalid.errors.front().find("invalid value '9' for parameter 'freq'"),
              std::string::npos);
    EXPECT_TRUE(invalid.values.empty());
}

TEST(parser_cli, last_repeated_option_wins)
{
    const auto parsed = parse_cli({ "--freq", "1", "--freq=2" });
    expect_stored(parsed.values, "freq", std::int64_t{ 2 });
}

schema
make_event_schema()
{
    schema value{};
    add_trace(value);
    auto events          = make_param("events", param_type::enumeration_list);
    events.env           = "ROCPROFSYS_EVENTS";
    events.arg           = "--events";
    events.arg_min_count = 1;
    events.arg_max_count = 3;
    events.choices       = { "ins", "cycles" };
    add_param(value, std::move(events));
    return value;
}

TEST(parser_cli, joins_tokens_until_the_next_option)
{
    const auto parsed =
        cli_parser{ { "--events", "ins", "cycles", "--trace" }, unknown_option::report }
            .parse(make_event_schema());

    EXPECT_TRUE(parsed.errors.empty());
    expect_stored(parsed.values, "events", std::vector<std::string>{ "ins", "cycles" });
    expect_stored(parsed.values, "trace", true);
}

TEST(parser_cli, does_not_take_the_next_option_as_a_value)
{
    const auto parsed = parse_cli({ "--output", "--trace" });

    ASSERT_EQ(parsed.errors.size(), 1U);
    EXPECT_NE(parsed.errors.front().find("requires a value"), std::string::npos);
    EXPECT_EQ(parsed.values.find("output"), parsed.values.end());
    expect_stored(parsed.values, "trace", true);
}

TEST(parser_env, reads_a_comma_separated_list)
{
    const auto lookup = [](std::string_view name) -> std::optional<std::string> {
        if(name == "ROCPROFSYS_EVENTS")
        {
            return std::string{ "cycles,ins" };
        }
        return std::nullopt;
    };

    const auto parsed = env_parser{ lookup }.parse(make_event_schema());
    EXPECT_TRUE(parsed.errors.empty());
    expect_stored(parsed.values, "events", std::vector<std::string>{ "cycles", "ins" });
}

TEST(parser_cli, accepts_a_negative_number_and_rejects_overflow)
{
    schema value{};
    auto   param = make_param("num", param_type::integer);
    param.arg    = "--num";
    add_param(value, std::move(param));

    const auto negative =
        cli_parser{ { "--num", "-5" }, unknown_option::report }.parse(value);
    EXPECT_TRUE(negative.errors.empty());
    expect_stored(negative.values, "num", std::int64_t{ -5 });

    const auto overflow =
        cli_parser{ { "--num", "9223372036854775808" }, unknown_option::report }.parse(
            value);
    EXPECT_TRUE(overflow.values.empty());
    ASSERT_EQ(overflow.errors.size(), 1U);
}

TEST(parser_cli, flag_does_not_consume_the_next_token)
{
    const auto with_word = parse_cli({ "--trace", "true" });
    expect_stored(with_word.values, "trace", true);
    EXPECT_TRUE(with_word.command.empty());
    ASSERT_EQ(with_word.errors.size(), 1U);
    EXPECT_NE(with_word.errors.front().find("unexpected argument 'true'"),
              std::string::npos);

    const auto with_command = parse_cli({ "--trace", "--", "true" });
    EXPECT_TRUE(with_command.errors.empty());
    expect_stored(with_command.values, "trace", true);
    EXPECT_EQ(with_command.command, (std::vector<std::string>{ "true" }));
}

TEST(parser_cli, accepts_an_empty_equals_value)
{
    schema value{};
    auto   param = make_param("name", param_type::string);
    param.arg    = "--name";
    add_param(value, std::move(param));

    const auto parsed = cli_parser{ { "--name=" }, unknown_option::report }.parse(value);
    EXPECT_TRUE(parsed.errors.empty());
    expect_stored(parsed.values, "name", std::string{});
}

TEST(parser_cli, accepts_a_short_flag_bundle)
{
    schema value{};
    add_short_flags(value);
    auto json = make_param("json", param_type::boolean);
    json.arg  = "-j";
    json.flag = true;
    add_param(value, std::move(json));

    const auto parsed = cli_parser{ { "-qj" }, unknown_option::report }.parse(value);
    EXPECT_TRUE(parsed.errors.empty());
    expect_stored(parsed.values, "quiet", true);
    expect_stored(parsed.values, "json", true);
}

TEST(parser_cli, plain_argument_is_unexpected)
{
    const auto parsed = parse_cli({ "--bogus", "5" });
    ASSERT_EQ(parsed.errors.size(), 2U);
    EXPECT_NE(parsed.errors.front().find("--bogus"), std::string::npos);
    EXPECT_NE(parsed.errors.back().find("unexpected argument '5'"), std::string::npos);
    EXPECT_TRUE(parsed.command.empty());
}

TEST(parser_cli, list_stops_at_the_separator_and_rejects_a_bare_command)
{
    const auto separated = cli_parser{ { "--events", "ins", "cycles", "--", "./app" },
                                       unknown_option::report }
                               .parse(make_event_schema());
    EXPECT_TRUE(separated.errors.empty());
    expect_stored(separated.values, "events",
                  std::vector<std::string>{ "ins", "cycles" });
    EXPECT_EQ(separated.command, (std::vector<std::string>{ "./app" }));

    const auto bare =
        cli_parser{ { "--events", "ins", "cycles", "./app" }, unknown_option::report }
            .parse(make_event_schema());
    expect_stored(bare.values, "events", std::vector<std::string>{ "ins", "cycles" });
    EXPECT_TRUE(bare.command.empty());
    ASSERT_EQ(bare.errors.size(), 1U);
    EXPECT_NE(bare.errors.front().find("unexpected argument './app'"), std::string::npos);
}

TEST(parser_cli, optional_value_does_not_override_the_default)
{
    schema value{};
    auto   verbose        = make_param("verbose", param_type::integer);
    verbose.default_value = "0";
    verbose.arg           = "--verbose";
    verbose.arg_min_count = 0;
    verbose.arg_max_count = 1;
    add_param(value, std::move(verbose));

    const auto parsed =
        cli_parser{ { "--verbose", "--", "./app" }, unknown_option::report }.parse(value);
    EXPECT_TRUE(parsed.errors.empty());
    EXPECT_EQ(parsed.values.find("verbose"), parsed.values.end());
    EXPECT_EQ(parsed.command, (std::vector<std::string>{ "./app" }));

    const auto defaults = default_parser{}.parse(value);
    expect_stored(defaults.values, "verbose", std::int64_t{ 0 });
}

TEST(parser_cli, required_value_stops_at_the_separator)
{
    const auto parsed = parse_cli({ "--output", "--", "./app" });
    ASSERT_EQ(parsed.errors.size(), 1U);
    EXPECT_NE(parsed.errors.front().find("requires a value"), std::string::npos);
    EXPECT_EQ(parsed.values.find("output"), parsed.values.end());
    EXPECT_EQ(parsed.command, (std::vector<std::string>{ "./app" }));
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
