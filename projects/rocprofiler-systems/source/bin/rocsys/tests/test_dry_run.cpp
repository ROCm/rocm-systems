// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dry_run.hpp"

#include "delivery.hpp"
#include "verb.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using rocprofsys::rocsys::env_op;
using rocprofsys::rocsys::env_plan;
using rocprofsys::rocsys::k_verbs;
using rocprofsys::rocsys::plan;
using rocprofsys::rocsys::verb_spec;
using rocprofsys::rocsys::write_dry_run;

namespace
{
const verb_spec*
find_verb(std::string_view name)
{
    const auto found = std::ranges::find(k_verbs, name, &verb_spec::name);
    return found == k_verbs.end() ? nullptr : &*found;
}

[[nodiscard]] std::string
render(std::string_view verb, const std::vector<std::string>& argv, const env_plan& ops)
{
    std::ostringstream out;
    write_dry_run(out, verb, argv, ops);
    return out.str();
}
}  // namespace

TEST(dry_run_test, profile_with_no_ops_prints_the_forwarded_argv)
{
    const auto text = render("profile", { "rocsys", "--", "./app" }, {});
    EXPECT_EQ(text, "verb: profile\n"
                    "argv: rocsys -- ./app\n"
                    "rocsys overrides; legacy flags are applied later by the tool\n");
}

TEST(dry_run_test, inherited_and_embedded_ops_use_separate_labels)
{
    const env_plan ops{
        { .key = "ROCPROFSYS_FOO", .value = "1", .origin = "--alpha", .embed = false },
        { .key    = "ROCPROFSYS_ROCM_DOMAINS",
          .value  = "hip_api",
          .origin = "--traces",
          .embed  = true },
    };
    const auto text = render("rewrite",
                             { "rocprof-sys-instrument", "--env",
                               "ROCPROFSYS_ROCM_DOMAINS=hip_api", "-o", "--", "./app" },
                             ops);
    EXPECT_EQ(text,
              "verb: rewrite\n"
              "argv: rocprof-sys-instrument --env ROCPROFSYS_ROCM_DOMAINS=hip_api -o -- "
              "./app\n"
              "rocsys overrides; legacy flags are applied later by the tool\n"
              "  ROCPROFSYS_FOO=1 origin=--alpha embed=false\n"
              "default in binary\n"
              "  ROCPROFSYS_ROCM_DOMAINS=hip_api origin=--traces embed=true\n");
}

TEST(dry_run_test, rewrite_plan_shows_the_embedded_flag_and_implicit_output)
{
    const auto* rewrite = find_verb("rewrite");
    ASSERT_NE(rewrite, nullptr);
    const env_plan ops{ { .key    = "ROCPROFSYS_ROCM_DOMAINS",
                          .value  = "",
                          .origin = "--traces",
                          .embed  = true } };
    const auto     planned = plan(*rewrite, false, { "rocsys", "--", "./app" }, ops);
    ASSERT_TRUE(planned);
    const auto text = render(rewrite->name, planned.value().argv, ops);
    EXPECT_NE(text.find("verb: rewrite\n"), std::string::npos);
    EXPECT_NE(text.find("argv: rocprof-sys-instrument --env ROCPROFSYS_ROCM_DOMAINS= -o "
                        "-- ./app\n"),
              std::string::npos);
    EXPECT_NE(text.find("default in binary\n"), std::string::npos);
    EXPECT_NE(text.find("  ROCPROFSYS_ROCM_DOMAINS= origin=--traces embed=true\n"),
              std::string::npos);
}
