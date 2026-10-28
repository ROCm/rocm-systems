// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema.hpp"
#include "parser/schema_loader.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace rocprofsys::parser;
using json = nlohmann::json;

namespace
{
json
document_with(json params, json extra = json::object())
{
    auto document      = json{ { "schema_version", "1.0" } };
    document["params"] = std::move(params);
    document.update(extra);
    return document;
}

json
bool_param(const std::string& param_id)
{
    return json{ { "id", param_id },
                 { "name", param_id },
                 { "description", "test parameter" },
                 { "type", "bool" } };
}

json
typed_param(const std::string& param_id, const std::string& type)
{
    auto param    = bool_param(param_id);
    param["type"] = type;
    return param;
}

/// @return the message thrown by load_schema(), or an empty string if it succeeds.
std::string
load_error(const json& document)
{
    try
    {
        static_cast<void>(load_schema(document));
    } catch(const std::runtime_error& err)
    {
        return err.what();
    }
    return {};
}

// Writes a uniquely named file under the gtest temp directory and removes it on
// destruction so parallel test runs do not collide.
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

constexpr auto k_profiler_document = R"({
    "_comment": "condensed from system_profiler_run_params.json",
    "schema_version": "1.0",
    "tool": "rocprof-sys-run",
    "invocation": { "usage": "rocprof-sys-run [options] -- <cmd>" },
    "categories": [
        { "id": "general", "name": "General" },
        { "id": "tracing", "name": "Tracing" },
        { "id": "sampling", "name": "Sampling" }
    ],
    "params": [
        {
            "id": "output_path", "name": "Output path",
            "description": "Output directory", "category": "general",
            "type": "path", "path_kind": "directory",
            "default_value": "rocprofsys-output",
            "env": { "name": "ROCPROFSYS_OUTPUT_PATH" },
            "arg": { "name": "--output", "form": "value", "assign_style": "equals" }
        },
        {
            "id": "trace", "name": "Trace", "description": "Enable tracing",
            "category": "tracing", "type": "bool", "default_value": true,
            "env": { "name": "ROCPROFSYS_TRACE" },
            "arg": { "name": "--trace", "form": "flag" }
        },
        {
            "id": "profile", "name": "Profile", "description": "Enable profiling",
            "category": "tracing", "type": "bool", "default_value": false,
            "env": { "name": "ROCPROFSYS_PROFILE" },
            "mutually_exclusive_with": [ "flat_profile" ]
        },
        {
            "id": "flat_profile", "name": "Flat profile",
            "description": "Flatten call stacks", "category": "tracing",
            "level": "advanced", "type": "bool", "default_value": false,
            "requires": [ "profile" ]
        },
        {
            "id": "sampling_freq", "name": "Sampling frequency",
            "description": "Samples per second", "category": "sampling",
            "level": "advanced", "type": "double", "default_value": 300,
            "min": 1, "max": 10000, "units": "Hz",
            "env": { "name": "ROCPROFSYS_SAMPLING_FREQ" },
            "depends_on": { "param": "trace", "equals": true }
        },
        {
            "id": "timemory_components", "name": "Components",
            "description": "Components to collect", "category": "sampling",
            "level": "expert", "type": "enum_list",
            "default_value": [ "wall_clock", "cpu_clock" ],
            "choices": [
                { "value": "wall_clock", "name": "Wall clock" },
                { "value": "cpu_clock", "name": "CPU clock",
                  "description": "Process CPU time" }
            ],
            "allow_custom": true,
            "env": { "name": "ROCPROFSYS_TIMEMORY_COMPONENTS", "separator": ";" }
        }
    ],
    "presets": { "balanced": { "trace": true } },
    "constraints": [
        {
            "kind": "warn", "severity": "warning",
            "when": { "all": [ { "param": "trace", "equals": false },
                               { "param": "profile", "equals": false } ] },
            "message": "nothing will be collected"
        },
        {
            "kind": "forces",
            "when": { "param": "flat_profile", "equals": true },
            "set": { "profile": true },
            "message": "flat profiles require profiling"
        },
        {
            "kind": "mutually_exclusive", "severity": "info",
            "params": [ "trace", "flat_profile" ]
        }
    ]
})";
}  // namespace

TEST(parser_schema_loader, loads_minimal_schema)
{
    const auto value = load_schema(document_with(json::array({ bool_param("verbose") })));

    EXPECT_EQ(value.version(), "1.0");
    const auto* param = value.find_param("verbose");
    ASSERT_NE(param, nullptr);
    EXPECT_EQ(param->type, param_type::boolean);
    EXPECT_EQ(param->level, param_level::basic);
    EXPECT_EQ(param->description, "test parameter");
    EXPECT_FALSE(param->default_value.has_value());
    EXPECT_FALSE(param->env.has_value());
    EXPECT_FALSE(param->arg.has_value());
}

TEST(parser_schema_loader, loads_profiler_document)
{
    const auto value = load_schema(json::parse(k_profiler_document));

    ASSERT_EQ(value.categories().size(), 3U);
    EXPECT_EQ(value.categories().at(1).id, "tracing");
    ASSERT_EQ(value.params().size(), 6U);
    EXPECT_EQ(value.params().front().id, "output_path");
    EXPECT_EQ(value.params_in_category("tracing").size(), 3U);
    EXPECT_EQ(value.constraints().size(), 3U);

    const auto* output = value.find_param_by_arg("--output");
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->type, param_type::path);
    EXPECT_EQ(output->path, path_kind::directory);
    ASSERT_TRUE(output->arg.has_value());
    EXPECT_EQ(output->arg.value_or(arg_spec{}).form, arg_form::value);
    EXPECT_EQ(output->arg.value_or(arg_spec{}).style, assign_style::equals);

    const auto* trace = value.find_param_by_env("ROCPROFSYS_TRACE");
    ASSERT_NE(trace, nullptr);
    EXPECT_EQ(trace->default_value, "true");
    ASSERT_TRUE(trace->arg.has_value());
    EXPECT_EQ(trace->arg.value_or(arg_spec{}).form, arg_form::flag);
    EXPECT_EQ(trace->arg.value_or(arg_spec{}).style, assign_style::separate);

    const auto* flat = value.find_param("flat_profile");
    ASSERT_NE(flat, nullptr);
    EXPECT_EQ(flat->level, param_level::advanced);
    EXPECT_EQ(flat->prerequisites, std::vector<std::string>{ "profile" });
    EXPECT_EQ(value.find_param("profile")->mutually_exclusive_with,
              std::vector<std::string>{ "flat_profile" });

    const auto* freq = value.find_param("sampling_freq");
    ASSERT_NE(freq, nullptr);
    EXPECT_EQ(freq->type, param_type::number);
    EXPECT_EQ(freq->min, 1.0);
    EXPECT_EQ(freq->max, 10000.0);
    EXPECT_EQ(freq->units, "Hz");
    ASSERT_TRUE(freq->depends_on.has_value());
    EXPECT_EQ(freq->depends_on.value_or(param_condition{}).param, "trace");
    EXPECT_EQ(freq->depends_on.value_or(param_condition{}).equals, "true");

    const auto* components = value.find_param("timemory_components");
    ASSERT_NE(components, nullptr);
    EXPECT_EQ(components->type, param_type::enumeration_list);
    EXPECT_EQ(components->level, param_level::expert);
    EXPECT_TRUE(components->allow_custom);
    ASSERT_EQ(components->choices.size(), 2U);
    EXPECT_EQ(components->choices.at(1).description, "Process CPU time");
    EXPECT_EQ(components->default_value, "wall_clock;cpu_clock");
}

TEST(parser_schema_loader, loads_constraints)
{
    const auto  value       = load_schema(json::parse(k_profiler_document));
    const auto& constraints = value.constraints();
    ASSERT_EQ(constraints.size(), 3U);

    const auto& warn = constraints.at(0);
    EXPECT_EQ(warn.kind, constraint_kind::warn);
    EXPECT_EQ(warn.level, severity::warning);
    EXPECT_EQ(warn.when.mode, condition_mode::all);
    ASSERT_EQ(warn.when.terms.size(), 2U);
    EXPECT_EQ(warn.when.terms.at(1).param, "profile");
    EXPECT_EQ(warn.when.terms.at(1).equals, "false");

    const auto& forces = constraints.at(1);
    EXPECT_EQ(forces.kind, constraint_kind::forces);
    EXPECT_EQ(forces.level, severity::error);
    ASSERT_EQ(forces.when.terms.size(), 1U);
    EXPECT_EQ(forces.when.terms.front().param, "flat_profile");
    EXPECT_EQ(forces.forced_values.at("profile"), "true");

    const auto& exclusive = constraints.at(2);
    EXPECT_EQ(exclusive.kind, constraint_kind::mutually_exclusive);
    EXPECT_EQ(exclusive.level, severity::info);
    EXPECT_EQ(exclusive.params, (std::vector<std::string>{ "trace", "flat_profile" }));
}

TEST(parser_schema_loader, loads_any_condition)
{
    auto document = document_with(
        json::array({ bool_param("first"), bool_param("second") }),
        { { "constraints",
            json::array(
                { { { "kind", "warn" },
                    { "message", "either is set" },
                    { "when",
                      { { "any", json::array({
                                     { { "param", "first" }, { "equals", true } },
                                     { { "param", "second" }, { "equals", true } },
                                 }) } } } } }) } });

    const auto value = load_schema(document);
    ASSERT_EQ(value.constraints().size(), 1U);
    EXPECT_EQ(value.constraints().front().when.mode, condition_mode::any);
    EXPECT_EQ(value.constraints().front().when.terms.size(), 2U);
}

TEST(parser_schema_loader, formats_numeric_defaults_without_padding)
{
    const auto value = load_schema(document_with(json::parse(R"([
        { "id": "integer", "name": "n", "description": "d", "type": "integer",
          "default_value": 300 },
        { "id": "number", "name": "n", "description": "d", "type": "double",
          "default_value": 3.14 },
        { "id": "negative", "name": "n", "description": "d", "type": "integer",
          "default_value": -5 }
    ])")));

    EXPECT_EQ(value.find_param("integer")->default_value, "300");
    EXPECT_EQ(value.find_param("number")->default_value, "3.14");
    EXPECT_EQ(value.find_param("negative")->default_value, "-5");
}

TEST(parser_schema_loader, null_fields_are_absent)
{
    auto param             = typed_param("name", "string");
    param["default_value"] = nullptr;
    param["category"]      = nullptr;
    param["env"]           = nullptr;

    const auto  value = load_schema(document_with(json::array({ param })));
    const auto* found = value.find_param("name");
    ASSERT_NE(found, nullptr);
    EXPECT_FALSE(found->default_value.has_value());
    EXPECT_TRUE(found->category.empty());
    EXPECT_FALSE(found->env.has_value());
}

TEST(parser_schema_loader, rejects_invalid_documents)
{
    EXPECT_NE(load_error(json::array()), "");
    EXPECT_NE(load_error(json{ { "schema_version", "1.0" } }).find("params"),
              std::string::npos);
    EXPECT_NE(load_error(json{ { "params", json::array() } }).find("schema_version"),
              std::string::npos);
    EXPECT_NE(load_error(document_with(json::object())).find("must be an array"),
              std::string::npos);
}

TEST(parser_schema_loader, rejects_missing_or_mistyped_fields)
{
    auto missing_name = bool_param("param");
    missing_name.erase("name");
    EXPECT_NE(load_error(document_with(json::array({ missing_name }))).find("'name'"),
              std::string::npos);

    auto numeric_name    = bool_param("param");
    numeric_name["name"] = true;
    EXPECT_NE(load_error(document_with(json::array({ numeric_name }))).find("string"),
              std::string::npos);

    auto bad_type    = bool_param("param");
    bad_type["type"] = "float";
    EXPECT_NE(load_error(document_with(json::array({ bad_type }))).find("unknown type"),
              std::string::npos);

    auto bad_level     = bool_param("param");
    bad_level["level"] = "novice";
    EXPECT_NE(load_error(document_with(json::array({ bad_level }))).find("unknown level"),
              std::string::npos);
}

TEST(parser_schema_loader, errors_name_the_offending_parameter)
{
    auto bad_type    = bool_param("broken");
    bad_type["type"] = "float";

    const auto message =
        load_error(document_with(json::array({ bool_param("fine"), bad_type })));
    EXPECT_NE(message.find("parameter 'broken'"), std::string::npos) << message;
}

TEST(parser_schema_loader, rejects_inconsistent_type_fields)
{
    auto choices_on_string       = typed_param("param", "string");
    choices_on_string["choices"] = json::array({ { { "value", "a" }, { "name", "A" } } });
    EXPECT_NE(load_error(document_with(json::array({ choices_on_string }))), "");

    auto enum_without_choices = typed_param("param", "enum");
    EXPECT_NE(load_error(document_with(json::array({ enum_without_choices }))), "");

    auto bounds_on_string   = typed_param("param", "string");
    bounds_on_string["min"] = 1;
    EXPECT_NE(load_error(document_with(json::array({ bounds_on_string }))), "");

    auto inverted_bounds   = typed_param("param", "integer");
    inverted_bounds["min"] = 3;
    inverted_bounds["max"] = 2;
    EXPECT_NE(load_error(document_with(json::array({ inverted_bounds }))), "");

    auto path_kind_on_string         = typed_param("param", "string");
    path_kind_on_string["path_kind"] = "file";
    EXPECT_NE(load_error(document_with(json::array({ path_kind_on_string }))), "");

    auto flag_on_integer   = typed_param("param", "integer");
    flag_on_integer["arg"] = { { "name", "--param" }, { "form", "flag" } };
    EXPECT_NE(load_error(document_with(json::array({ flag_on_integer }))), "");
}

TEST(parser_schema_loader, rejects_invalid_patterns_and_separators)
{
    auto invalid_pattern       = typed_param("param", "string");
    invalid_pattern["pattern"] = "[unclosed";
    EXPECT_NE(load_error(document_with(json::array({ invalid_pattern }))).find("pattern"),
              std::string::npos);

    auto pattern_on_integer       = typed_param("param", "integer");
    pattern_on_integer["pattern"] = "^[0-9]+$";
    EXPECT_NE(load_error(document_with(json::array({ pattern_on_integer }))), "");

    auto empty_separator   = typed_param("param", "string");
    empty_separator["env"] = { { "name", "ROCPROFSYS_PARAM" }, { "separator", "" } };
    EXPECT_NE(load_error(document_with(json::array({ empty_separator }))), "");
}

TEST(parser_schema_loader, rejects_invalid_defaults)
{
    auto out_of_range             = typed_param("param", "integer");
    out_of_range["max"]           = 3;
    out_of_range["default_value"] = 4;
    EXPECT_NE(load_error(document_with(json::array({ out_of_range }))).find("default"),
              std::string::npos);

    auto fractional_integer             = typed_param("param", "integer");
    fractional_integer["default_value"] = json::parse("1.5");
    EXPECT_NE(load_error(document_with(json::array({ fractional_integer }))), "");

    auto unknown_choice       = typed_param("param", "enum");
    unknown_choice["choices"] = json::array({ { { "value", "a" }, { "name", "A" } } });
    unknown_choice["default_value"] = "b";
    EXPECT_NE(load_error(document_with(json::array({ unknown_choice }))), "");

    auto pattern_mismatch             = typed_param("param", "string");
    pattern_mismatch["pattern"]       = "^[a-z]+$";
    pattern_mismatch["default_value"] = "ABC";
    EXPECT_NE(load_error(document_with(json::array({ pattern_mismatch }))), "");
}

TEST(parser_schema_loader, rejects_duplicate_names)
{
    EXPECT_NE(
        load_error(document_with(json::array({ bool_param("same"), bool_param("same") })))
            .find("reuses"),
        std::string::npos);

    auto first    = bool_param("first");
    first["env"]  = { { "name", "ROCPROFSYS_SHARED" } };
    auto second   = bool_param("second");
    second["env"] = { { "name", "ROCPROFSYS_SHARED" } };
    EXPECT_NE(load_error(document_with(json::array({ first, second }))), "");

    first.erase("env");
    second.erase("env");
    first["arg"]  = { { "name", "--shared" } };
    second["arg"] = { { "name", "--shared" } };
    EXPECT_NE(load_error(document_with(json::array({ first, second }))), "");

    const auto categories = json::array({ { { "id", "dup" }, { "name", "One" } },
                                          { { "id", "dup" }, { "name", "Two" } } });
    EXPECT_NE(load_error(document_with(json::array(), { { "categories", categories } }))
                  .find("duplicate category"),
              std::string::npos);
}

TEST(parser_schema_loader, rejects_unknown_references)
{
    auto unknown_category        = bool_param("param");
    unknown_category["category"] = "missing";
    EXPECT_NE(load_error(document_with(json::array({ unknown_category })))
                  .find("unknown category"),
              std::string::npos);

    auto unknown_exclusion                       = bool_param("param");
    unknown_exclusion["mutually_exclusive_with"] = json::array({ "missing" });
    EXPECT_NE(load_error(document_with(json::array({ unknown_exclusion })))
                  .find("unknown parameter 'missing'"),
              std::string::npos);

    auto unknown_prerequisite        = bool_param("param");
    unknown_prerequisite["requires"] = json::array({ "missing" });
    EXPECT_NE(load_error(document_with(json::array({ unknown_prerequisite }))), "");

    auto self_reference        = bool_param("param");
    self_reference["requires"] = json::array({ "param" });
    EXPECT_NE(load_error(document_with(json::array({ self_reference }))).find("itself"),
              std::string::npos);

    auto bad_dependency          = bool_param("param");
    bad_dependency["depends_on"] = { { "param", "other" }, { "equals", "maybe" } };
    EXPECT_NE(
        load_error(document_with(json::array({ bad_dependency, bool_param("other") })))
            .find("invalid value 'maybe'"),
        std::string::npos);
}

namespace
{
json
with_constraint(json item)
{
    return document_with(json::array({ bool_param("first"), bool_param("second") }),
                         { { "constraints", json::array({ std::move(item) }) } });
}
}  // namespace

TEST(parser_schema_loader, rejects_invalid_constraints)
{
    EXPECT_NE(load_error(with_constraint({ { "kind", "conflicts" } })).find("unknown"),
              std::string::npos);
    EXPECT_NE(load_error(with_constraint({ { "kind", "mutually_exclusive" },
                                           { "params", json::array({ "first" }) } })),
              "");
    EXPECT_NE(
        load_error(with_constraint({ { "kind", "mutually_exclusive" },
                                     { "params", json::array({ "first", "missing" }) } }))
            .find("unknown parameter 'missing'"),
        std::string::npos);
}

TEST(parser_schema_loader, rejects_incomplete_conditional_constraints)
{
    const auto when = json{ { "param", "first" }, { "equals", true } };
    EXPECT_NE(load_error(with_constraint({ { "kind", "warn" }, { "message", "text" } })),
              "");
    EXPECT_NE(load_error(with_constraint({ { "kind", "warn" }, { "when", when } })), "");
    EXPECT_NE(load_error(with_constraint(
                  { { "kind", "forces" }, { "when", when }, { "message", "text" } })),
              "");
    EXPECT_NE(load_error(with_constraint({ { "kind", "forces" },
                                           { "when", when },
                                           { "message", "text" },
                                           { "set", { { "second", "maybe" } } } })),
              "");
    EXPECT_EQ(load_error(with_constraint({ { "kind", "forces" },
                                           { "when", when },
                                           { "message", "text" },
                                           { "set", { { "second", true } } } })),
              "");
}

TEST(parser_schema_loader, loads_schema_file)
{
    const scoped_temp_file file{ k_profiler_document };

    const auto value = load_schema_file(file.path());
    EXPECT_EQ(value.version(), "1.0");
    EXPECT_EQ(value.params().size(), 6U);
}

TEST(parser_schema_loader, reports_file_errors)
{
    EXPECT_THROW(static_cast<void>(load_schema_file("/nonexistent/schema.json")),
                 std::runtime_error);

    const scoped_temp_file file{ R"({ "schema_version": "1.0", "params": [ )" };
    try
    {
        static_cast<void>(load_schema_file(file.path()));
        FAIL() << "expected a parse error";
    } catch(const std::runtime_error& err)
    {
        EXPECT_NE(std::string{ err.what() }.find("failed to parse"), std::string::npos);
    }
}
