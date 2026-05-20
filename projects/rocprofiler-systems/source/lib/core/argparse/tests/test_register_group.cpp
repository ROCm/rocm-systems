// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Integration tests for argparse::register_group. Because the helpers in
// interpreter.cpp (strip_dashes, keys_from, delimiter_for, read_value,
// emit_env, register_flag) live in an anonymous namespace, they're covered
// here through their observable effects on parser_data after a real
// tim::argparse::argument_parser parses synthetic argv.

#include "core/argparse.hpp"
#include "core/argparse/detail/parser_engine.hpp"
#include "core/argparse/flag_descriptor.hpp"
#include "core/argparse/interpreter.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace
{
using rocprofsys::argparse::count_spec;
using rocprofsys::argparse::flag_descriptor;
using rocprofsys::argparse::flag_group;
using rocprofsys::argparse::join_with;
using rocprofsys::argparse::parser_data;
using rocprofsys::argparse::parser_t;
using rocprofsys::argparse::register_group;
using rocprofsys::argparse::value_kind;
using rocprofsys::common::update_mode;

class RegisterGroupTest : public ::testing::Test
{
protected:
    parser_t    parser{ "test" };
    parser_data data{};

    // tim::argparse::argument_parser requires enable_help() before start_group;
    // skipping it causes a segfault inside start_group's bookkeeping. This
    // mirrors the setup tool_runner does in production.
    void SetUp() override { parser.enable_help(); }

    void register_single(flag_descriptor descriptor)
    {
        flag_group group{ "TEST", "", { std::move(descriptor) } };
        register_group(parser, data, group);
    }

    void parse(std::vector<std::string> args)
    {
        args.insert(args.begin(), "test");  // argv[0]
        const auto err = parser.parse(args, 0);
        ASSERT_FALSE(err) << err.what();
    }

    [[nodiscard]] std::string env_value(std::string_view key) const
    {
        const auto needle = std::string{ key } + "=";
        for(const auto& entry : data.env.current)
        {
            if(entry.compare(0, needle.size(), needle) == 0)
                return entry.substr(needle.size());
        }
        return {};
    }
};

}  // namespace

// -----------------------------------------------------------------------------
// value_kind coverage — drives read_value / emit_env for each variant
// -----------------------------------------------------------------------------

TEST_F(RegisterGroupTest, FlagKindEmitsTrueWhenPresent)
{
    register_single({
        /* names    */ { "--my-flag" },
        /* help     */ "test flag",
        /* dtype    */ {},
        /* count    */ count_spec::at_most(1),
        /* kind     */ value_kind::flag,
        /* join     */ join_with::none,
        /* env_vars */ { "MY_FLAG_ENV" },
    });

    parse({ "--my-flag" });

    EXPECT_EQ(env_value("MY_FLAG_ENV"), "true");
}

TEST_F(RegisterGroupTest, ScalarKindEmitsRawValue)
{
    register_single({
        /* names    */ { "--scalar-opt" },
        /* help     */ "scalar option",
        /* dtype    */ "string",
        /* count    */ count_spec::exactly(1),
        /* kind     */ value_kind::scalar,
        /* join     */ join_with::none,
        /* env_vars */ { "SCALAR_ENV" },
    });

    parse({ "--scalar-opt", "hello" });

    EXPECT_EQ(env_value("SCALAR_ENV"), "hello");
}

TEST_F(RegisterGroupTest, ScalarIntEmitsIntegerString)
{
    register_single({
        /* names    */ { "--int-opt" },
        /* help     */ "int option",
        /* dtype    */ "int",
        /* count    */ count_spec::exactly(1),
        /* kind     */ value_kind::scalar_int,
        /* join     */ join_with::none,
        /* env_vars */ { "INT_ENV" },
    });

    parse({ "--int-opt", "42" });

    EXPECT_EQ(env_value("INT_ENV"), "42");
}

TEST_F(RegisterGroupTest, ScalarDoubleEmitsDoubleString)
{
    register_single({
        /* names    */ { "--double-opt" },
        /* help     */ "double option",
        /* dtype    */ "float",
        /* count    */ count_spec::exactly(1),
        /* kind     */ value_kind::scalar_double,
        /* join     */ join_with::none,
        /* env_vars */ { "DOUBLE_ENV" },
    });

    parse({ "--double-opt", "3.5" });

    // std::to_string(3.5) → "3.500000"; just check leading digits.
    EXPECT_EQ(env_value("DOUBLE_ENV").substr(0, 3), "3.5");
}

TEST_F(RegisterGroupTest, ListKindWithCommaJoinerJoinsValues)
{
    register_single({
        /* names    */ { "--list-opt" },
        /* help     */ "list option",
        /* dtype    */ "items",
        /* count    */ count_spec::at_least(1),
        /* kind     */ value_kind::list,
        /* join     */ join_with::comma,
        /* env_vars */ { "LIST_ENV" },
    });

    parse({ "--list-opt", "a", "b", "c" });

    EXPECT_EQ(env_value("LIST_ENV"), "a,b,c");
}

TEST_F(RegisterGroupTest, ListKindWithSpaceJoinerJoinsValues)
{
    register_single({
        /* names    */ { "--space-list" },
        /* help     */ "list with spaces",
        /* dtype    */ "items",
        /* count    */ count_spec::at_least(1),
        /* kind     */ value_kind::list,
        /* join     */ join_with::space,
        /* env_vars */ { "SPACE_LIST_ENV" },
    });

    parse({ "--space-list", "x", "y" });

    EXPECT_EQ(env_value("SPACE_LIST_ENV"), "x y");
}

// -----------------------------------------------------------------------------
// keys_from coverage — long-name derivation, hyphen-to-underscore env_key,
// dedup_keys promotion
// -----------------------------------------------------------------------------

TEST_F(RegisterGroupTest, LongNameDerivesParserAndEnvKeys)
{
    // names = ["-x", "--extended-name"]; parser_key derives from the LAST name.
    register_single({
        /* names    */ { "-x", "--extended-name" },
        /* help     */ "extended",
        /* dtype    */ "string",
        /* count    */ count_spec::exactly(1),
        /* kind     */ value_kind::scalar,
        /* join     */ join_with::none,
        /* env_vars */ { "EXTENDED_ENV" },
    });

    parse({ "--extended-name", "value" });

    EXPECT_EQ(env_value("EXTENDED_ENV"), "value");
    // env_key derives from parser_key with hyphens → underscores.
    EXPECT_TRUE(data.reg.processed_environs.count("extended_name") > 0);
}

TEST_F(RegisterGroupTest, DedupKeysArePromotedToProcessedEnvirons)
{
    register_single({
        /* names      */ { "--dedup-flag" },
        /* help       */ "dedup",
        /* dtype      */ {},
        /* count      */ count_spec::at_most(1),
        /* kind       */ value_kind::flag,
        /* join       */ join_with::none,
        /* env_vars   */ {},
        /* mode       */ update_mode::REPLACE,
        /* dedup_keys */ { "alias_one", "alias_two" },
    });

    EXPECT_TRUE(data.reg.processed_environs.count("dedup_flag") > 0);
    EXPECT_TRUE(data.reg.processed_environs.count("alias_one") > 0);
    EXPECT_TRUE(data.reg.processed_environs.count("alias_two") > 0);
}

// -----------------------------------------------------------------------------
// environ_filter integration — skip-on-false
// -----------------------------------------------------------------------------

TEST_F(RegisterGroupTest, EnvironFilterFalseSuppressesRegistration)
{
    data.reg.environ_filter = [](std::string_view key, const parser_data&) {
        return key != "filtered_out";
    };

    register_single({
        /* names    */ { "--filtered-out" },
        /* help     */ "should not register",
        /* dtype    */ {},
        /* count    */ count_spec::at_most(1),
        /* kind     */ value_kind::flag,
        /* join     */ join_with::none,
        /* env_vars */ { "FILTERED_ENV" },
    });

    // Filter blocked registration, so processed_environs stays clean for this key.
    EXPECT_EQ(data.reg.processed_environs.count("filtered_out"), 0u);
}

// -----------------------------------------------------------------------------
// update_mode coverage — drives emit_env behavior
// -----------------------------------------------------------------------------

TEST_F(RegisterGroupTest, UpdateModeAppendConcatenatesExisting)
{
    // Pre-seed env to make APPEND observable.
    data.env.current.emplace_back("APPEND_ENV=base");
    data.env.initial.emplace("APPEND_ENV=base");

    register_single({
        /* names    */ { "--append-opt" },
        /* help     */ "append",
        /* dtype    */ "string",
        /* count    */ count_spec::exactly(1),
        /* kind     */ value_kind::scalar,
        /* join     */ join_with::none,
        /* env_vars */ { "APPEND_ENV" },
        /* mode     */ update_mode::APPEND,
    });

    parse({ "--append-opt", "added" });

    // APPEND uses the default ":" delimiter from emit_env's scalar branch.
    EXPECT_EQ(env_value("APPEND_ENV"), "base:added");
}

// -----------------------------------------------------------------------------
// Multiple env_vars per descriptor — all should be written
// -----------------------------------------------------------------------------

TEST_F(RegisterGroupTest, MultipleEnvVarsAllWritten)
{
    register_single({
        /* names    */ { "--multi-env" },
        /* help     */ "multi env",
        /* dtype    */ "string",
        /* count    */ count_spec::exactly(1),
        /* kind     */ value_kind::scalar,
        /* join     */ join_with::none,
        /* env_vars */ { "ENV_A", "ENV_B", "ENV_C" },
    });

    parse({ "--multi-env", "shared" });

    EXPECT_EQ(env_value("ENV_A"), "shared");
    EXPECT_EQ(env_value("ENV_B"), "shared");
    EXPECT_EQ(env_value("ENV_C"), "shared");
}

// -----------------------------------------------------------------------------
// Custom action path — short-circuits default env emission
// -----------------------------------------------------------------------------

namespace
{
void
custom_writes_marker(parser_t& /*parser*/, parser_data& data)
{
    data.env.set("CUSTOM_MARKER", "yes");
}
}  // namespace

TEST_F(RegisterGroupTest, CustomActionRunsInsteadOfDefaultEnvEmission)
{
    register_single({
        /* names      */ { "--custom-flag" },
        /* help       */ "custom",
        /* dtype      */ {},
        /* count      */ count_spec::at_most(1),
        /* kind       */ value_kind::flag,
        /* join       */ join_with::none,
        /* env_vars   */ { "SHOULD_NOT_APPEAR" },
        /* mode       */ update_mode::REPLACE,
        /* dedup_keys */ {},
        /* choices    */ {},
        /* conflicts  */ {},
        /* requires_  */ {},
        /* custom     */ &custom_writes_marker,
    });

    parse({ "--custom-flag" });

    EXPECT_EQ(env_value("CUSTOM_MARKER"), "yes");
    EXPECT_EQ(env_value("SHOULD_NOT_APPEAR"), "");
}

// -----------------------------------------------------------------------------
// Multi-descriptor group — every descriptor in flags[] registers
// -----------------------------------------------------------------------------

TEST_F(RegisterGroupTest, MultipleDescriptorsAllRegister)
{
    flag_group group{
        "MULTI",
        "subtitle",
        {
            flag_descriptor{ { "--first" }, "first flag", {}, count_spec::at_most(1),
                             value_kind::flag, join_with::none, { "FIRST_ENV" } },
            flag_descriptor{ { "--second" }, "second flag", {}, count_spec::at_most(1),
                             value_kind::flag, join_with::none, { "SECOND_ENV" } },
        },
    };
    register_group(parser, data, group);

    EXPECT_GT(data.reg.processed_environs.count("first"), 0u);
    EXPECT_GT(data.reg.processed_environs.count("second"), 0u);

    parse({ "--first", "--second" });

    EXPECT_EQ(env_value("FIRST_ENV"), "true");
    EXPECT_EQ(env_value("SECOND_ENV"), "true");
}
