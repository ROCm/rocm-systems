// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/env_op.hpp"
#include "delivery.hpp"
#include "verb.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

using rocprofsys::rocsys::env_op;
using rocprofsys::rocsys::env_plan;
using rocprofsys::rocsys::k_verbs;
using rocprofsys::rocsys::merge_environ;
using rocprofsys::rocsys::plan;
using rocprofsys::rocsys::verb_spec;

namespace
{
const verb_spec*
find_verb(std::string_view name)
{
    const auto found = std::ranges::find(k_verbs, name, &verb_spec::name);
    return found == k_verbs.end() ? nullptr : &*found;
}

env_op
embedded(std::string key, std::string value)
{
    return { .key    = std::move(key),
             .value  = std::move(value),
             .origin = "--traces",
             .embed  = true };
}

env_op
inherited(std::string key, std::string value)
{
    return { .key = std::move(key), .value = std::move(value), .origin = "--traces" };
}
}  // namespace

TEST(delivery_test, profile_strips_the_subcommand_and_keeps_argv0)
{
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const auto planned = plan(
        *profile, true, { "rocsys", "profile", "--preset=quick", "--", "./app" }, {});
    ASSERT_TRUE(planned);
    EXPECT_TRUE(planned.value().in_process);
    EXPECT_EQ(planned.value().argv,
              (std::vector<std::string>{ "rocsys", "--preset=quick", "--", "./app" }));
    EXPECT_TRUE(planned.value().env.empty());
}

TEST(delivery_test, implicit_profile_keeps_the_leading_separator)
{
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const auto planned = plan(*profile, false, { "rocsys", "--", "./app" }, {});
    ASSERT_TRUE(planned);
    EXPECT_EQ(planned.value().argv,
              (std::vector<std::string>{ "rocsys", "--", "./app" }));
}

TEST(delivery_test, instrument_uses_the_sibling_name)
{
    const auto* instrument = find_verb("instrument");
    ASSERT_NE(instrument, nullptr);
    const auto planned =
        plan(*instrument, true, { "rocsys", "instrument", "--help" }, {});
    ASSERT_TRUE(planned);
    EXPECT_FALSE(planned.value().in_process);
    EXPECT_EQ(planned.value().program, "rocprof-sys-instrument");
    EXPECT_EQ(planned.value().argv,
              (std::vector<std::string>{ "rocprof-sys-instrument", "--help" }));
}

TEST(delivery_test, rewrite_inserts_output_flag_before_the_separator)
{
    const auto* rewrite = find_verb("rewrite");
    ASSERT_NE(rewrite, nullptr);
    const auto planned = plan(*rewrite, true, { "rocsys", "rewrite", "--", "./app" }, {});
    ASSERT_TRUE(planned);
    EXPECT_EQ(planned.value().argv, (std::vector<std::string>{ "rocprof-sys-instrument",
                                                               "-o", "--", "./app" }));
}

TEST(delivery_test, rewrite_keeps_a_short_output_flag)
{
    const auto* rewrite = find_verb("rewrite");
    ASSERT_NE(rewrite, nullptr);
    const auto planned = plan(
        *rewrite, true, { "rocsys", "rewrite", "-o", "app.inst", "--", "./app" }, {});
    ASSERT_TRUE(planned);
    EXPECT_EQ(planned.value().argv,
              (std::vector<std::string>{ "rocprof-sys-instrument", "-o", "app.inst", "--",
                                         "./app" }));
}

TEST(delivery_test, rewrite_keeps_a_long_output_flag)
{
    const auto* rewrite = find_verb("rewrite");
    ASSERT_NE(rewrite, nullptr);
    const auto planned = plan(
        *rewrite, true, { "rocsys", "rewrite", "--output=app.inst", "--", "./app" }, {});
    ASSERT_TRUE(planned);
    EXPECT_EQ(planned.value().argv.front(), "rocprof-sys-instrument");
    EXPECT_EQ(planned.value().argv.at(1), "--output=app.inst");
    EXPECT_EQ(std::ranges::count(planned.value().argv, "-o"), 0);
}

TEST(delivery_test, rewrite_embeds_env_and_puts_output_flag_last_before_separator)
{
    const auto* rewrite = find_verb("rewrite");
    ASSERT_NE(rewrite, nullptr);
    const auto planned = plan(*rewrite, true, { "rocsys", "rewrite", "--", "./app" },
                              { embedded("ROCPROFSYS_ROCM_DOMAINS", "kernel_dispatch") });
    ASSERT_TRUE(planned);
    EXPECT_TRUE(planned.value().env.empty());
    EXPECT_EQ(planned.value().argv,
              (std::vector<std::string>{ "rocprof-sys-instrument", "--env",
                                         "ROCPROFSYS_ROCM_DOMAINS=kernel_dispatch", "-o",
                                         "--", "./app" }));
}

TEST(delivery_test, rewrite_keeps_the_user_env_token)
{
    const auto* rewrite = find_verb("rewrite");
    ASSERT_NE(rewrite, nullptr);
    const auto planned =
        plan(*rewrite, true, { "rocsys", "rewrite", "--env", "USER=1", "--", "./app" },
             { embedded("ROCPROFSYS_ROCM_DOMAINS", "kernel_dispatch") });
    ASSERT_TRUE(planned);
    EXPECT_EQ(planned.value().argv,
              (std::vector<std::string>{ "rocprof-sys-instrument", "--env",
                                         "ROCPROFSYS_ROCM_DOMAINS=kernel_dispatch",
                                         "--env", "USER=1", "-o", "--", "./app" }));
}

TEST(delivery_test, rewrite_embeds_an_empty_value)
{
    const auto* rewrite = find_verb("rewrite");
    ASSERT_NE(rewrite, nullptr);
    const auto planned = plan(*rewrite, true, { "rocsys", "rewrite", "--", "./app" },
                              { embedded("ROCPROFSYS_ROCM_DOMAINS", "") });
    ASSERT_TRUE(planned);
    EXPECT_EQ(
        planned.value().argv,
        (std::vector<std::string>{ "rocprof-sys-instrument", "--env",
                                   "ROCPROFSYS_ROCM_DOMAINS=", "-o", "--", "./app" }));
}

TEST(delivery_test, inherited_ops_stay_out_of_argv)
{
    const auto* instrument = find_verb("instrument");
    ASSERT_NE(instrument, nullptr);
    const auto planned =
        plan(*instrument, true, { "rocsys", "instrument", "--", "./app" },
             { inherited("ROCPROFSYS_ROCM_DOMAINS", "kernel_dispatch") });
    ASSERT_TRUE(planned);
    EXPECT_EQ(planned.value().argv,
              (std::vector<std::string>{ "rocprof-sys-instrument", "--", "./app" }));
    ASSERT_EQ(planned.value().env.size(), 1U);
    EXPECT_EQ(planned.value().env.front().key, "ROCPROFSYS_ROCM_DOMAINS");
    EXPECT_FALSE(planned.value().env.front().embed);
}

TEST(delivery_test, embedding_on_a_non_rewrite_verb_is_an_error)
{
    const auto* profile = find_verb("profile");
    ASSERT_NE(profile, nullptr);
    const auto planned = plan(*profile, true, { "rocsys", "profile", "--", "./app" },
                              { embedded("ROCPROFSYS_ROCM_DOMAINS", "kernel_dispatch") });
    ASSERT_FALSE(planned);
    EXPECT_FALSE(planned.error().error.empty());
    EXPECT_FALSE(planned.error().hint.empty());
}

TEST(delivery_test, merge_environ_replaces_one_key_and_keeps_prefixes)
{
    const auto merged =
        merge_environ({ "FOO=old", "FOOBAR=keep", "OTHER=1" },
                      { inherited("FOO", "new"), inherited("EMPTY", "") });
    EXPECT_EQ(merged, (std::vector<std::string>{ "FOOBAR=keep", "OTHER=1", "FOO=new",
                                                 "EMPTY=" }));
}
