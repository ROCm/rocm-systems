// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema.hpp"
#include "parser/schema_loader.hpp"
#include "parser/schema_validator.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>  // NOLINT(misc-include-cleaner): json::parse
#include <nlohmann/json_fwd.hpp>

#include <algorithm>
#include <string>
#include <vector>

using namespace rocprofsys::parser;

namespace
{
constexpr double k_integer_max = 10.0;
constexpr double k_number_min  = 0.5;
constexpr double k_number_max  = 1.5;

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

/// Parameters and constraints shared by the validate_values() tests.
schema
make_profiler_schema()
{
    return load_schema(nlohmann::json::parse(R"({
        "schema_version": "1.0",
        "params": [
            { "id": "output", "name": "Output", "description": "Output path",
              "type": "path", "required": true },
            { "id": "trace", "name": "Trace", "description": "Tracing",
              "type": "bool", "default_value": true,
              "mutually_exclusive_with": [ "flat_profile" ] },
            { "id": "profile", "name": "Profile", "description": "Profiling",
              "type": "bool", "default_value": false },
            { "id": "flat_profile", "name": "Flat", "description": "Flat profile",
              "type": "bool", "default_value": false, "requires": [ "profile" ] },
            { "id": "sampling_freq", "name": "Frequency", "description": "Hz",
              "type": "double", "min": 1, "max": 1000,
              "depends_on": { "param": "trace", "equals": true } },
            { "id": "mode", "name": "Mode", "description": "Mode", "type": "enum",
              "choices": [ { "value": "fast", "name": "Fast" },
                           { "value": "slow", "name": "Slow" } ] },
            { "id": "label", "name": "Label", "description": "Label", "type": "string" }
        ],
        "constraints": [
            { "kind": "mutually_exclusive", "params": [ "label", "mode" ] },
            { "kind": "warn", "severity": "warning", "message": "nothing collected",
              "when": { "all": [ { "param": "trace", "equals": false },
                                 { "param": "profile", "equals": false } ] } },
            { "kind": "warn", "severity": "info", "message": "slow or labeled",
              "when": { "any": [ { "param": "mode", "equals": "slow" },
                                 { "param": "label", "equals": "debug" } ] } },
            { "kind": "forces", "message": "flat profiles need profiling",
              "when": { "param": "flat_profile", "equals": true },
              "set": { "profile": true } }
        ]
    })"));
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
    for(const auto* value : { "true", "FALSE", "Yes", "no", "on", "OFF", "1", "0" })
    {
        EXPECT_TRUE(is_valid_value(param, value)) << value;
    }
    for(const auto* value : { "", "2", "truthy", " true", "y" })
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
    EXPECT_FALSE(is_valid_value(text, "Upper"));
    EXPECT_FALSE(is_valid_value(text, "lower1"));

    auto path = make_param(param_type::path);
    EXPECT_TRUE(is_valid_value(path, "/tmp/output"));
    EXPECT_FALSE(is_valid_value(path, ""));
}

TEST(parser_schema_validator, regex_matches_whole_value)
{
    auto param    = make_param(param_type::string);
    param.pattern = "[0-9]+";
    EXPECT_TRUE(is_valid_value(param, "123"));
    EXPECT_FALSE(is_valid_value(param, "a123b"));
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
    const auto violations =
        validate({ { "flat_profile", "true" }, { "profile", "true" } });

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
