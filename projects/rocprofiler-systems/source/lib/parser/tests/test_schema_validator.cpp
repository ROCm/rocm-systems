// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using namespace rocprofsys::parser;

namespace
{
constexpr double k_integer_max         = 10.0;
constexpr double k_number_min          = 0.5;
constexpr double k_number_max          = 1.5;
constexpr double k_frequency_minimum   = 1.0;
constexpr double k_frequency_maximum   = 1000.0;
constexpr double k_non_integer_minimum = 2.5;
constexpr double k_non_integer_maximum = 1.5;

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
                           .name        = "Output",
                           .description = "Output path",
                           .type        = param_type::path,
                           .required    = true,
                       });
    add_checked(value, schema_param{
                           .id                      = "trace",
                           .name                    = "Trace",
                           .description             = "Tracing",
                           .type                    = param_type::boolean,
                           .default_value           = "true",
                           .mutually_exclusive_with = { "flat_profile" },
                       });
    add_checked(value, schema_param{
                           .id            = "profile",
                           .name          = "Profile",
                           .description   = "Profiling",
                           .type          = param_type::boolean,
                           .default_value = "false",
                       });
    add_checked(value, schema_param{
                           .id            = "flat_profile",
                           .name          = "Flat",
                           .description   = "Flat profile",
                           .type          = param_type::boolean,
                           .default_value = "false",
                           .prerequisites = { "profile" },
                       });
    add_checked(value,
                schema_param{
                    .id          = "sampling_freq",
                    .name        = "Frequency",
                    .description = "Hz",
                    .type        = param_type::number,
                    .min         = k_frequency_minimum,
                    .max         = k_frequency_maximum,
                    .depends_on  = param_condition{ .param = "trace", .equals = "true" },
                });
    add_checked(value, schema_param{
                           .id          = "mode",
                           .name        = "Mode",
                           .description = "Mode",
                           .type        = param_type::enumeration,
                           .choices     = { choice{ .value = "fast", .name = "Fast" },
                                            choice{ .value = "slow", .name = "Slow" } },
                       });
    add_checked(value, schema_param{
                           .id          = "label",
                           .name        = "Label",
                           .description = "Label",
                           .type        = param_type::string,
                       });
}

void
add_profiler_constraints(schema& value)
{
    ASSERT_TRUE(value.add_constraint(constraint{
        .kind   = constraint_kind::mutually_exclusive,
        .params = { "label", "mode" },
    }));
    ASSERT_TRUE(value.add_constraint(constraint{
        .kind    = constraint_kind::warn,
        .level   = severity::warning,
        .message = "nothing collected",
        .when =
            condition{
                .mode  = condition_mode::all,
                .terms = { param_condition{ .param = "trace", .equals = "false" },
                           param_condition{ .param = "profile", .equals = "false" } } },
    }));
    ASSERT_TRUE(value.add_constraint(constraint{
        .kind    = constraint_kind::warn,
        .level   = severity::info,
        .message = "slow or labeled",
        .when =
            condition{
                .mode  = condition_mode::any,
                .terms = { param_condition{ .param = "mode", .equals = "slow" },
                           param_condition{ .param = "label", .equals = "debug" } } },
    }));
    ASSERT_TRUE(value.add_constraint(constraint{
        .kind          = constraint_kind::forces,
        .message       = "flat profiles need profiling",
        .when          = condition{ .terms = { param_condition{ .param  = "flat_profile",
                                                                .equals = "true" } } },
        .forced_values = { { "profile", "true" } },
    }));
}

/// Parameters and constraints shared by the validate_values() tests.
schema
make_profiler_schema()
{
    schema value{};
    value.set_version("1.0");
    add_profiler_params(value);
    add_profiler_constraints(value);
    return value;
}

std::vector<violation>
validate(const value_map_t& values)
{
    static const auto k_config_schema = make_profiler_schema();
    auto              with_output     = values;
    with_output.try_emplace("output", "/tmp/out");
    return validate_values(k_config_schema, with_output);
}

bool
has_violation(const std::vector<violation>& violations, severity level,
              const std::string& fragment)
{
    return std::ranges::any_of(violations, [&](const violation& item) {
        return item.level == level && item.message.find(fragment) != std::string::npos;
    });
}
}  // namespace

TEST(parser_schema_validator, boolean_values)
{
    const auto param = make_param(param_type::boolean);
    for(const auto* value : { "true", "FALSE", "Yes", "no", "on", "OFF", "y", "N", "t",
                              "F", "1", "0", "2", "+1", "-1" })
    {
        EXPECT_TRUE(is_valid_value(param, value)) << value;
    }
    for(const auto* value : { "truthy", " true", "", "   ", "maybe" })
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
    auto text = make_param(param_type::string);
    EXPECT_TRUE(is_valid_value(text, ""));
    EXPECT_TRUE(is_valid_value(text, "anything"));

    text.pattern = "^[a-z]+$";
    EXPECT_TRUE(is_valid_value(text, "lower"));
    EXPECT_TRUE(is_valid_value(text, "Upper"));
    EXPECT_TRUE(is_valid_value(text, "lower1"));

    auto path = make_param(param_type::path);
    EXPECT_TRUE(is_valid_value(path, "/tmp/output"));
    EXPECT_FALSE(is_valid_value(path, ""));
}

TEST(parser_schema_validator, pattern_is_not_applied)
{
    auto param    = make_param(param_type::string);
    param.pattern = "[0-9]+";
    EXPECT_TRUE(is_valid_value(param, "123"));
    EXPECT_TRUE(is_valid_value(param, "a123b"));
    EXPECT_TRUE(is_valid_value(param, std::string(4097, 'a')));
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

    param.env = env_spec{ .name = "ROCPROFSYS_LIST", .separator = ";" };
    EXPECT_TRUE(is_valid_value(param, "alpha;beta"));
    EXPECT_FALSE(is_valid_value(param, "alpha,beta"));

    param.allow_custom = true;
    EXPECT_TRUE(is_valid_value(param, "alpha;custom"));
}

TEST(parser_schema_validator, accepts_defaults_and_valid_values)
{
    EXPECT_TRUE(validate({}).empty());
    EXPECT_TRUE(validate({ { "sampling_freq", "100" }, { "mode", "fast" } }).empty());
}

TEST(parser_schema_validator, reports_unknown_and_invalid_values)
{
    const auto violations =
        validate({ { "bogus", "1" }, { "sampling_freq", "nan" }, { "mode", "medium" } });

    EXPECT_EQ(violations.size(), 3U);
    EXPECT_TRUE(has_violation(violations, severity::error, "unknown parameter 'bogus'"));
    EXPECT_TRUE(has_violation(violations, severity::error, "'nan'"));
    EXPECT_TRUE(has_violation(violations, severity::error, "'medium'"));
}

TEST(parser_schema_validator, reports_missing_required_value)
{
    const auto violations = validate_values(make_profiler_schema(), {});
    ASSERT_EQ(violations.size(), 1U);
    EXPECT_EQ(violations.front().params, std::vector<std::string>{ "output" });
}

TEST(parser_schema_validator, reports_mutually_exclusive_params_once)
{
    const auto violations = validate(
        { { "trace", "true" }, { "flat_profile", "true" }, { "profile", "true" } });

    const auto count = std::ranges::count_if(violations, [](const violation& item) {
        return item.message.find("mutually exclusive") != std::string::npos;
    });
    EXPECT_EQ(count, 1);
    EXPECT_TRUE(has_violation(violations, severity::error, "'trace' and 'flat_profile'"));
}

TEST(parser_schema_validator, exclusion_uses_boolean_value)
{
    EXPECT_TRUE(
        validate({ { "trace", "off" }, { "flat_profile", "on" }, { "profile", "yes" } })
            .empty());
}

TEST(parser_schema_validator, reports_missing_prerequisite)
{
    const auto violations =
        validate({ { "trace", "false" }, { "flat_profile", "true" } });
    EXPECT_TRUE(
        has_violation(violations, severity::error, "'flat_profile' requires 'profile'"));
}

TEST(parser_schema_validator, warns_about_unmet_dependency)
{
    const auto violations = validate(
        { { "trace", "false" }, { "profile", "true" }, { "sampling_freq", "100" } });
    ASSERT_EQ(violations.size(), 1U);
    EXPECT_EQ(violations.front().level, severity::warning);
    EXPECT_EQ(violations.front().params,
              (std::vector<std::string>{ "sampling_freq", "trace" }));
}

TEST(parser_schema_validator, schema_mutually_exclusive_constraint)
{
    const auto violations = validate({ { "label", "debug-run" }, { "mode", "fast" } });
    ASSERT_EQ(violations.size(), 1U);
    EXPECT_EQ(violations.front().message,
              "parameters label, mode are mutually exclusive");
    EXPECT_EQ(violations.front().params, (std::vector<std::string>{ "label", "mode" }));
}

TEST(parser_schema_validator, warn_constraint_with_all_terms)
{
    EXPECT_TRUE(has_violation(validate({ { "trace", "false" } }), severity::warning,
                              "nothing collected"));
    EXPECT_FALSE(has_violation(validate({ { "trace", "false" }, { "profile", "true" } }),
                               severity::warning, "nothing collected"));
}

TEST(parser_schema_validator, warn_constraint_with_any_terms)
{
    EXPECT_TRUE(has_violation(validate({ { "mode", "slow" } }), severity::info,
                              "slow or labeled"));
    EXPECT_TRUE(has_violation(validate({ { "label", "debug" } }), severity::info,
                              "slow or labeled"));
    EXPECT_FALSE(has_violation(validate({ { "mode", "fast" } }), severity::info,
                               "slow or labeled"));
}

TEST(parser_schema_validator, forces_constraint_reports_its_message)
{
    const auto violations =
        validate({ { "trace", "false" }, { "flat_profile", "true" } });
    EXPECT_TRUE(
        has_violation(violations, severity::error, "flat profiles need profiling"));
}

TEST(parser_schema_validator, forces_constraint_is_quiet_when_satisfied)
{
    const auto violations = validate(
        { { "trace", "false" }, { "flat_profile", "true" }, { "profile", "true" } });
    EXPECT_TRUE(violations.empty());
}

TEST(parser_schema_validator, a_default_does_not_enable_mutual_exclusion)
{
    const auto violations =
        validate({ { "flat_profile", "true" }, { "profile", "true" } });
    EXPECT_FALSE(has_violation(violations, severity::error, "mutually exclusive"));
}

TEST(parser_schema_validator, reports_a_two_sided_exclusion_once)
{
    schema value{};
    add_checked(value, schema_param{
                           .id                      = "left",
                           .name                    = "Left",
                           .description             = "Left",
                           .type                    = param_type::boolean,
                           .mutually_exclusive_with = { "right" },
                       });
    add_checked(value, schema_param{
                           .id                      = "right",
                           .name                    = "Right",
                           .description             = "Right",
                           .type                    = param_type::boolean,
                           .mutually_exclusive_with = { "left" },
                       });

    const auto violations =
        validate_values(value, { { "left", "true" }, { "right", "true" } });
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

TEST(parser_schema_validator, large_integers_do_not_compare_as_doubles)
{
    schema value{};
    add_checked(value, schema_param{
                           .id          = "wide",
                           .name        = "Wide",
                           .description = "Wide",
                           .type        = param_type::integer,
                           .depends_on  = param_condition{ .param  = "wide",
                                                           .equals = "9007199254740992" },
                       });

    const auto violations = validate_values(value, { { "wide", "9007199254740993" } });
    EXPECT_TRUE(has_violation(violations, severity::warning, "has no effect"));
}

TEST(parser_schema_validator, integer_text_does_not_equal_a_decimal)
{
    schema value{};
    add_checked(value,
                schema_param{
                    .id          = "count",
                    .name        = "Count",
                    .description = "Count",
                    .type        = param_type::integer,
                    .depends_on  = param_condition{ .param = "count", .equals = "1.0" },
                });
    add_checked(value,
                schema_param{
                    .id          = "scale",
                    .name        = "Scale",
                    .description = "Scale",
                    .type        = param_type::number,
                    .depends_on  = param_condition{ .param = "scale", .equals = "1.0" },
                });

    const auto violations =
        validate_values(value, { { "count", "1" }, { "scale", "1" } });
    EXPECT_TRUE(has_violation(violations, severity::warning, "count"));
    EXPECT_FALSE(has_violation(violations, severity::warning, "scale"));
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
                           .name          = "Count",
                           .description   = "Count",
                           .category      = "missing",
                           .type          = param_type::integer,
                           .default_value = "maybe",
                           .min           = k_non_integer_minimum,
                           .max           = k_non_integer_maximum,
                           .pattern       = "[",
                           .env = env_spec{ .name = "ROCPROFSYS_COUNT", .separator = "" },
                           .copy_to = { "absent" },
                       });

    const auto errors = validate_schema(value);
    EXPECT_TRUE(has_violation(errors, severity::error, "unknown category"));
    EXPECT_TRUE(has_violation(errors, severity::error, "invalid default"));
    EXPECT_TRUE(has_violation(errors, severity::error, "minimum greater"));
    EXPECT_TRUE(has_violation(errors, severity::error, "non-integer minimum"));
    EXPECT_FALSE(has_violation(errors, severity::error, "invalid pattern"));
    EXPECT_TRUE(has_violation(errors, severity::error, "empty list separator"));
    EXPECT_TRUE(has_violation(errors, severity::error, "unknown parameter 'absent'"));
}
