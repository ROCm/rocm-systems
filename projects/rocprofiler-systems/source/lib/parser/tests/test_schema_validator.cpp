// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"
#include "parser/value_text.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using namespace rocprofsys::parser;

namespace
{
constexpr double k_integer_max       = 10.0;
constexpr double k_number_min        = 0.5;
constexpr double k_number_max        = 1.5;
constexpr double k_frequency_minimum = 1.0;
constexpr double k_frequency_maximum = 1000.0;

schema_param
make_param(param_type type)
{
    auto param = schema_param{};
    param.id   = "param";
    param.type = type;
    return param;
}

schema_param
make_enum(param_type type = param_type::enumeration)
{
    auto param    = make_param(type);
    param.choices = { choice{ .value = "alpha" }, choice{ .value = "beta" } };
    return param;
}

void
add_checked(schema& value, schema_param param)
{
    ASSERT_TRUE(value.add_param(std::move(param)));
}

void
add_profiler_params(schema& value)
{
    add_checked(value, schema_param{
                           .id          = "output",
                           .description = "Output path",
                           .type        = param_type::path,
                       });
    add_checked(value, schema_param{
                           .id            = "trace",
                           .description   = "Tracing",
                           .type          = param_type::boolean,
                           .default_value = "true",
                           .conflicts     = { "flat_profile" },
                       });
    add_checked(value, schema_param{
                           .id            = "profile",
                           .description   = "Profiling",
                           .type          = param_type::boolean,
                           .default_value = "false",
                       });
    add_checked(value, schema_param{
                           .id            = "flat_profile",
                           .description   = "Flat profile",
                           .type          = param_type::boolean,
                           .default_value = "false",
                           .required      = { "profile" },
                       });
    add_checked(value, schema_param{
                           .id          = "sampling_freq",
                           .description = "Hz",
                           .type        = param_type::number,
                           .min         = k_frequency_minimum,
                           .max         = k_frequency_maximum,
                       });
    add_checked(value,
                schema_param{
                    .id          = "mode",
                    .description = "Mode",
                    .type        = param_type::enumeration,
                    .choices = { choice{ .value = "fast" }, choice{ .value = "slow" } },
                });
    add_checked(value, schema_param{
                           .id          = "label",
                           .description = "Label",
                           .type        = param_type::string,
                       });
}

/// Parameters shared by the validate_values() tests.
schema
make_profiler_schema()
{
    schema value{};
    value.set_version("1.0");
    add_profiler_params(value);
    return value;
}

std::vector<violation>
validate(const typed_values_t& values)
{
    static const auto k_config_schema = make_profiler_schema();
    auto              with_output     = values;
    with_output.try_emplace("output", std::string{ "/tmp/out" });
    return validate_values(k_config_schema, with_output);
}

bool
has_violation(const std::vector<violation>& violations, const std::string& fragment)
{
    return std::ranges::any_of(violations, [&](const violation& item) {
        return item.message.find(fragment) != std::string::npos;
    });
}
}  // namespace

TEST(parser_schema_validator, boolean_values)
{
    const auto param = make_param(param_type::boolean);
    for(const auto* value : { "true", "FALSE", "Yes", "no", "on", "OFF", "y", "N", "t",
                              "F", "1", "0", "2", "+2" })
    {
        EXPECT_TRUE(is_valid_value(param, value)) << value;
    }
    for(const auto* value : { "", "truthy", " true", "+" })
    {
        EXPECT_FALSE(is_valid_value(param, value)) << value;
    }
}

TEST(parser_schema_validator, integer_values)
{
    const auto param = make_param(param_type::integer);
    EXPECT_TRUE(is_valid_value(param, "42"));
    EXPECT_TRUE(is_valid_value(param, "-7"));
    for(const auto* value :
        { "", "1.5", "12abc", " 12", "12 ", "0x10", "1e3", "99999999999999999999" })
    {
        EXPECT_FALSE(is_valid_value(param, value)) << value;
    }
}

TEST(parser_schema_validator, integer_bounds)
{
    auto param = make_param(param_type::integer);
    param.min  = 0.0;
    param.max  = k_integer_max;
    EXPECT_TRUE(is_valid_value(param, "0"));
    EXPECT_TRUE(is_valid_value(param, "10"));
    EXPECT_FALSE(is_valid_value(param, "-1"));
    EXPECT_FALSE(is_valid_value(param, "11"));
}

TEST(parser_schema_validator, number_values)
{
    const auto param = make_param(param_type::number);
    EXPECT_TRUE(is_valid_value(param, "3.14"));
    EXPECT_TRUE(is_valid_value(param, "-2"));
    EXPECT_TRUE(is_valid_value(param, "1e3"));
    for(const auto* value : { "", "1.5x", " 1.5", "nan", "NaN", "inf", "-inf", "1e999" })
    {
        EXPECT_FALSE(is_valid_value(param, value)) << value;
    }
}

TEST(parser_schema_validator, number_bounds)
{
    auto param = make_param(param_type::number);
    param.min  = k_number_min;
    param.max  = k_number_max;
    EXPECT_TRUE(is_valid_value(param, "0.5"));
    EXPECT_TRUE(is_valid_value(param, "1.5"));
    EXPECT_FALSE(is_valid_value(param, "0.49"));
    EXPECT_FALSE(is_valid_value(param, "1.51"));
    EXPECT_FALSE(is_valid_value(param, "nan"));
}

TEST(parser_schema_validator, string_and_path_values)
{
    const auto text = make_param(param_type::string);
    EXPECT_TRUE(is_valid_value(text, ""));
    EXPECT_TRUE(is_valid_value(text, "anything"));

    const auto path = make_param(param_type::path);
    EXPECT_TRUE(is_valid_value(path, "/tmp/output"));
    EXPECT_FALSE(is_valid_value(path, ""));
}

TEST(parser_schema_validator, enumeration_values)
{
    auto param = make_enum();
    EXPECT_TRUE(is_valid_value(param, "alpha"));
    EXPECT_FALSE(is_valid_value(param, "gamma"));
    EXPECT_FALSE(is_valid_value(param, "Alpha"));
    EXPECT_FALSE(is_valid_value(param, ""));

    param.allow_custom = true;
    EXPECT_TRUE(is_valid_value(param, "gamma"));
    EXPECT_FALSE(is_valid_value(param, ""));
}

TEST(parser_schema_validator, enumeration_list_values)
{
    auto param = make_enum(param_type::enumeration_list);
    EXPECT_TRUE(is_valid_value(param, ""));
    EXPECT_TRUE(is_valid_value(param, "alpha"));
    EXPECT_TRUE(is_valid_value(param, "alpha,beta"));
    EXPECT_TRUE(is_valid_value(param, " alpha , beta "));
    EXPECT_FALSE(is_valid_value(param, "alpha,gamma"));
    EXPECT_FALSE(is_valid_value(param, "alpha,,beta"));
    EXPECT_FALSE(is_valid_value(param, "alpha,"));

    param.env = "ROCPROFSYS_LIST";
    EXPECT_TRUE(is_valid_value(param, "alpha,beta"));

    param.allow_custom = true;
    EXPECT_TRUE(is_valid_value(param, "alpha,custom"));
}

TEST(parser_schema_validator, accepts_defaults_and_valid_values)
{
    EXPECT_TRUE(validate({}).empty());
    EXPECT_TRUE(
        validate({ { "sampling_freq", 100.0 }, { "mode", std::string{ "fast" } } })
            .empty());
}

TEST(parser_schema_validator, reports_an_unknown_parameter)
{
    const auto violations = validate({ { "bogus", std::string{ "1" } } });

    EXPECT_EQ(violations.size(), 1U);
    EXPECT_TRUE(has_violation(violations, "unknown parameter 'bogus'"));
}

TEST(parser_schema_validator, reports_mutually_exclusive_params_once)
{
    const auto violations =
        validate({ { "trace", true }, { "flat_profile", true }, { "profile", true } });

    const auto count = std::ranges::count_if(violations, [](const violation& item) {
        return item.message.find("mutually exclusive") != std::string::npos;
    });
    EXPECT_EQ(count, 1);
    EXPECT_TRUE(has_violation(violations, "'trace' and 'flat_profile'"));
}

TEST(parser_schema_validator, exclusion_uses_boolean_value)
{
    EXPECT_TRUE(
        validate({ { "trace", false }, { "flat_profile", true }, { "profile", true } })
            .empty());
}

TEST(parser_schema_validator, reports_missing_requirement)
{
    const auto violations = validate({ { "trace", false }, { "flat_profile", true } });
    EXPECT_TRUE(has_violation(violations, "'flat_profile' requires 'profile'"));
}

TEST(parser_schema_validator, a_default_does_not_enable_mutual_exclusion)
{
    const auto violations = validate({ { "flat_profile", true }, { "profile", true } });
    EXPECT_FALSE(has_violation(violations, "mutually exclusive"));
}

TEST(parser_schema_validator, reports_a_two_sided_exclusion_once)
{
    schema value{};
    add_checked(value, schema_param{
                           .id          = "left",
                           .description = "Left",
                           .type        = param_type::boolean,
                           .conflicts   = { "right" },
                       });
    add_checked(value, schema_param{
                           .id          = "right",
                           .description = "Right",
                           .type        = param_type::boolean,
                           .conflicts   = { "left" },
                       });

    const auto violations =
        validate_values(value, { { "left", true }, { "right", true } });
    const auto count = std::ranges::count_if(violations, [](const violation& item) {
        return item.message.find("mutually exclusive") != std::string::npos;
    });
    EXPECT_EQ(count, 1);
}

TEST(parser_schema_validator, integer_above_two_to_the_53_respects_its_maximum)
{
    constexpr double k_exact_integer_limit = 9007199254740992.0;
    auto             param                 = make_param(param_type::integer);
    param.max                              = k_exact_integer_limit;
    EXPECT_TRUE(is_valid_value(param, "9007199254740992"));
    EXPECT_FALSE(is_valid_value(param, "9007199254740993"));
}

TEST(parser_schema_validator, catalog_check_accepts_the_profiler_fixture)
{
    EXPECT_TRUE(validate_schema(make_profiler_schema()).empty());
}

TEST(parser_schema_validator, catalog_check_reports_structural_errors)
{
    schema value{};
    add_checked(value, schema_param{
                           .id            = "count",
                           .description   = "Count",
                           .category      = "missing",
                           .type          = param_type::integer,
                           .default_value = "maybe",
                           .min           = 2.5,
                           .max           = 1.5,
                           .env           = "ROCPROFSYS_COUNT",
                           .conflicts     = { "absent" },
                       });

    const auto errors = validate_schema(value);
    EXPECT_TRUE(has_violation(errors, "invalid default"));
    EXPECT_TRUE(has_violation(errors, "minimum greater"));
    EXPECT_TRUE(has_violation(errors, "non-integer minimum"));
    EXPECT_TRUE(has_violation(errors, "unknown parameter 'absent'"));
    EXPECT_FALSE(has_violation(errors, "unknown category"));
}

TEST(parser_schema_validator, a_choice_alias_is_a_valid_value_and_default)
{
    auto param = schema_param{
        .id            = "format",
        .type          = param_type::enumeration,
        .default_value = "proto",
        .choices       = { choice{ .value = "pftrace", .aliases = { "proto" } } },
    };

    // "proto" should be a valid value, since it's an alias for "pftrace".
    EXPECT_TRUE(is_valid_value(param, "proto"));
    EXPECT_TRUE(is_valid_value(param, "pftrace"));
    // "pb" should be invalid, since it's not an alias for "pftrace".
    EXPECT_FALSE(is_valid_value(param, "pb"));

    // The schema should be valid.
    schema value{};
    add_checked(value, std::move(param));
    EXPECT_TRUE(validate_schema(value).empty());
}
