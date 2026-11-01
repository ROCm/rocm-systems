// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema.hpp"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

using namespace rocprofsys::parser;

namespace
{
constexpr int k_extra_params = 64;

schema_param
make_param(std::string param_id, std::string env_name = {}, std::string arg_name = {})
{
    auto param = schema_param{};
    param.id   = std::move(param_id);
    if(!env_name.empty())
    {
        param.env = env_spec{ .name = std::move(env_name) };
    }
    if(!arg_name.empty())
    {
        param.arg = arg_spec{ .name = std::move(arg_name) };
    }
    return param;
}
}  // namespace

TEST(parser_schema, empty_schema)
{
    const schema value{};

    EXPECT_TRUE(value.version().empty());
    EXPECT_TRUE(value.categories().empty());
    EXPECT_TRUE(value.params().empty());
    EXPECT_TRUE(value.constraints().empty());
    EXPECT_EQ(value.find_param("missing"), nullptr);
    EXPECT_EQ(value.find_category("missing"), nullptr);
}

TEST(parser_schema, params_keep_insertion_order)
{
    schema value{};
    EXPECT_TRUE(value.add_param(make_param("zeta")));
    EXPECT_TRUE(value.add_param(make_param("alpha")));
    EXPECT_TRUE(value.add_param(make_param("mid")));

    ASSERT_EQ(value.params().size(), 3U);
    EXPECT_EQ(value.params().at(0).id, "zeta");
    EXPECT_EQ(value.params().at(1).id, "alpha");
    EXPECT_EQ(value.params().at(2).id, "mid");
}

TEST(parser_schema, finds_param_by_id_env_and_arg)
{
    schema value{};
    ASSERT_TRUE(value.add_param(make_param("trace", "ROCPROFSYS_TRACE", "--trace")));

    const auto* by_id = value.find_param("trace");
    ASSERT_NE(by_id, nullptr);
    EXPECT_EQ(value.find_param_by_env("ROCPROFSYS_TRACE"), by_id);
    EXPECT_EQ(value.find_param_by_arg("--trace"), by_id);
    EXPECT_EQ(value.find_param_by_env("ROCPROFSYS_OTHER"), nullptr);
    EXPECT_EQ(value.find_param_by_arg("--other"), nullptr);
}

TEST(parser_schema, rejects_duplicate_id)
{
    schema value{};
    ASSERT_TRUE(value.add_param(make_param("trace", "ROCPROFSYS_TRACE")));
    EXPECT_FALSE(value.add_param(make_param("trace", "ROCPROFSYS_OTHER")));

    ASSERT_EQ(value.params().size(), 1U);
    EXPECT_EQ(value.find_param_by_env("ROCPROFSYS_OTHER"), nullptr);
}

TEST(parser_schema, rejects_duplicate_env_name)
{
    schema value{};
    ASSERT_TRUE(value.add_param(make_param("first", "ROCPROFSYS_SHARED")));
    EXPECT_FALSE(value.add_param(make_param("second", "ROCPROFSYS_SHARED")));

    EXPECT_EQ(value.find_param("second"), nullptr);
    EXPECT_EQ(value.find_param_by_env("ROCPROFSYS_SHARED"), value.find_param("first"));
}

TEST(parser_schema, rejects_duplicate_arg_name)
{
    schema value{};
    ASSERT_TRUE(value.add_param(make_param("first", "", "--shared")));
    EXPECT_FALSE(value.add_param(make_param("second", "", "--shared")));

    EXPECT_EQ(value.find_param("second"), nullptr);
    EXPECT_EQ(value.find_param_by_arg("--shared"), value.find_param("first"));
}

TEST(parser_schema, indexes_stay_valid_after_growth)
{
    schema value{};
    ASSERT_TRUE(value.add_param(make_param("first", "ROCPROFSYS_FIRST", "--first")));
    for(int idx = 0; idx < k_extra_params; ++idx)
    {
        ASSERT_TRUE(value.add_param(make_param("param_" + std::to_string(idx))));
    }

    const auto* first = value.find_param_by_env("ROCPROFSYS_FIRST");
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->id, "first");
    EXPECT_EQ(value.find_param_by_arg("--first"), first);
}

TEST(parser_schema, rejects_empty_names)
{
    schema value{};
    EXPECT_FALSE(value.add_category(category{ .id = "", .name = "Blank" }));
    EXPECT_TRUE(value.categories().empty());

    EXPECT_FALSE(value.add_param(make_param("")));
    auto empty_env = make_param("trace");
    empty_env.env  = env_spec{ .name = "" };
    EXPECT_FALSE(value.add_param(std::move(empty_env)));
    auto empty_arg = make_param("trace");
    empty_arg.arg  = arg_spec{ .name = "" };
    EXPECT_FALSE(value.add_param(std::move(empty_arg)));
    EXPECT_TRUE(value.params().empty());
}

TEST(parser_schema, rejects_a_constraint_that_names_a_bad_value)
{
    schema value{};
    auto   trace = make_param("trace");
    trace.type   = param_type::boolean;
    ASSERT_TRUE(value.add_param(std::move(trace)));

    EXPECT_FALSE(value.add_constraint(constraint{ .params = { "missing" } }));
    EXPECT_FALSE(value.add_constraint(constraint{
        .when = condition{ .terms = { param_condition{ .param  = "trace",
                                                       .equals = "ture" } } },
    }));
    EXPECT_FALSE(
        value.add_constraint(constraint{ .forced_values = { { "trace", "ture" } } }));
    EXPECT_TRUE(value.constraints().empty());

    EXPECT_TRUE(value.add_constraint(constraint{
        .when = condition{ .terms = { param_condition{ .param  = "trace",
                                                       .equals = "true" } } },
    }));
    EXPECT_EQ(value.constraints().size(), 1U);
}

TEST(parser_schema, categories_reject_duplicates)
{
    schema value{};
    EXPECT_TRUE(value.add_category({ .id = "general", .name = "General" }));
    EXPECT_FALSE(value.add_category({ .id = "general", .name = "Other" }));

    ASSERT_EQ(value.categories().size(), 1U);
    const auto* found = value.find_category("general");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->name, "General");
}

TEST(parser_schema, params_in_category)
{
    schema value{};
    auto   first    = make_param("first");
    first.category  = "tracing";
    auto second     = make_param("second");
    second.category = "sampling";
    auto third      = make_param("third");
    third.category  = "tracing";
    ASSERT_TRUE(value.add_param(first));
    ASSERT_TRUE(value.add_param(second));
    ASSERT_TRUE(value.add_param(third));

    const auto tracing = value.params_in_category("tracing");
    ASSERT_EQ(tracing.size(), 2U);
    EXPECT_EQ(tracing.at(0)->id, "first");
    EXPECT_EQ(tracing.at(1)->id, "third");
    EXPECT_TRUE(value.params_in_category("missing").empty());
}

TEST(parser_schema, effect_fields_round_trip)
{
    auto param    = make_param("flat_profile");
    param.copy_to = { "profile" };
    param.assigns = { { "log_level", "debug" } };
    param.fills   = { "trace_delay" };
    param.maps_to = { value_mapping{
        .when = "0", .target = "log_level", .value = "info" } };
    param.choices = { choice{
        .value       = "pftrace",
        .name        = "Perfetto",
        .description = "Trace",
        .sets        = { { "trace", "true" } },
        .aliases     = { "proto" },
    } };
    param.arg     = arg_spec{ .name      = "--flat-profile",
                              .min_count = 1,
                              .max_count = 2,
                              .binds     = { "output_path", "output_prefix" } };

    schema value{};
    ASSERT_TRUE(value.add_param(std::move(param)));
    const auto* stored = value.find_param("flat_profile");
    ASSERT_NE(stored, nullptr);
    EXPECT_EQ(stored->copy_to, std::vector<std::string>({ "profile" }));
    EXPECT_EQ(stored->assigns.at("log_level"), "debug");
    EXPECT_EQ(stored->fills, std::vector<std::string>({ "trace_delay" }));
    ASSERT_EQ(stored->maps_to.size(), 1U);
    EXPECT_EQ(stored->maps_to.front().target, "log_level");
    ASSERT_EQ(stored->choices.size(), 1U);
    EXPECT_EQ(stored->choices.front().sets.at("trace"), "true");
    EXPECT_EQ(stored->choices.front().aliases, std::vector<std::string>({ "proto" }));
    const auto argument = stored->arg.value_or(arg_spec{});
    EXPECT_EQ(argument.min_count.value_or(0U), 1U);
    EXPECT_EQ(argument.max_count.value_or(0U), 2U);
    EXPECT_EQ(argument.binds,
              (std::vector<std::string>{ "output_path", "output_prefix" }));
}

TEST(parser_schema, split_list_trims_and_rejects_an_empty_separator)
{
    const auto parts =
        split_list(" alpha, beta ", ",").value_or(std::vector<std::string>{});
    EXPECT_EQ(parts, (std::vector<std::string>{ "alpha", "beta" }));
    const auto empty_list = split_list("", ",").value_or(std::vector<std::string>{ "x" });
    EXPECT_TRUE(empty_list.empty());
    EXPECT_FALSE(split_list("alpha", "").has_value());
}

TEST(parser_schema, list_separator_defaults_to_comma)
{
    auto param = make_param("list");
    EXPECT_EQ(list_separator(param), ",");

    param.env = env_spec{ .name = "ROCPROFSYS_LIST", .separator = ":" };
    EXPECT_EQ(list_separator(param), ":");
}
