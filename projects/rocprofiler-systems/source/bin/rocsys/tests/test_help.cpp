// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "help/renderer.hpp"

#include "options/dry_run.hpp"
#include "options/table.hpp"
#include "verb.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using rocprofsys::rocsys::capability;
using rocprofsys::rocsys::capability_set;
using rocprofsys::rocsys::k_verbs;
using rocprofsys::rocsys::verb_spec;
using rocprofsys::rocsys::help::requests_tool_help;
using rocprofsys::rocsys::help::write_common_options;
using rocprofsys::rocsys::help::write_verb_options;
using rocprofsys::rocsys::options::k_dry_run;
using rocprofsys::rocsys::options::option_spec;
using rocprofsys::rocsys::options::option_table;

namespace
{
// Accepted long options, without "--". Attach is the usage text in
// rocprof-sys-attach.cpp (`--pid` is accepted but not printed). Python is
// rocprofsys/__main__.py. Keep in sync with tests/pytest/test_rocsys.py.
constexpr auto k_phase2_names = std::to_array<std::string_view>({ "dry-run", "traces" });

constexpr auto k_attach_long_options = std::to_array<std::string_view>({
    "detach-after",
    "format",
    "help",
    "output",
});

constexpr auto k_python_long_options = std::to_array<std::string_view>({
    "annotate-trace",
    "builtin",
    "config",
    "full-filepath",
    "function-exclude",
    "function-include",
    "function-restrict",
    "help",
    "label",
    "module-exclude",
    "module-include",
    "module-restrict",
    "setup",
    "trace-c",
    "verbosity",
});

[[nodiscard]] std::string
render_common(std::span<const option_spec> table)
{
    std::ostringstream out;
    write_common_options(out, table);
    return out.str();
}

[[nodiscard]] std::string
render_verb(const verb_spec& verb, std::span<const option_spec> table)
{
    std::ostringstream out;
    write_verb_options(out, verb, table);
    return out.str();
}

[[nodiscard]] bool
contains(std::span<const std::string_view> names, std::string_view name)
{
    return std::ranges::find(names, name) != names.end();
}
}  // namespace

TEST(help_test, common_options_lists_dry_run)
{
    const auto text = render_common(option_table());
    EXPECT_EQ(text, "Common options:\n"
                    "  --dry-run     Show the launch plan and exit\n");
}

TEST(help_test, every_verb_lists_dry_run_and_rewrite_names_the_follow_up)
{
    for(const auto& verb : k_verbs)
    {
        const auto text = render_verb(verb, option_table());
        EXPECT_NE(text.find("rocsys options for " + std::string{ verb.name } + ":\n"),
                  std::string::npos)
            << verb.name;
        EXPECT_NE(text.find("  --dry-run     Show the launch plan and exit\n"),
                  std::string::npos)
            << verb.name;
        const auto traces = text.find("--traces");
        if(verb.name == "avail")
        {
            EXPECT_NE(traces, std::string::npos);
        }
        else
        {
            EXPECT_EQ(traces, std::string::npos) << verb.name;
        }
        if(verb.name == "rewrite")
        {
            EXPECT_NE(text.find("  rocsys -- ./app.inst\n"), std::string::npos);
        }
        else
        {
            EXPECT_EQ(text.find("./app.inst"), std::string::npos) << verb.name;
        }
    }
}

TEST(help_test, verb_section_hides_options_the_verb_cannot_use)
{
    const option_spec traces{
        .name          = "traces",
        .required_caps = static_cast<capability_set>(capability::honours_rocm_tracing),
        .summary       = "Select what to trace",
    };
    const option_spec rows[] = { k_dry_run, traces };
    const auto*       causal =
        std::ranges::find(k_verbs, std::string_view{ "causal" }, &verb_spec::name);
    const auto* profile =
        std::ranges::find(k_verbs, std::string_view{ "profile" }, &verb_spec::name);
    ASSERT_NE(causal, k_verbs.end());
    ASSERT_NE(profile, k_verbs.end());

    const auto causal_text = render_verb(*causal, rows);
    EXPECT_NE(causal_text.find("--dry-run"), std::string::npos);
    EXPECT_EQ(causal_text.find("--traces"), std::string::npos);

    const auto profile_text = render_verb(*profile, rows);
    EXPECT_NE(profile_text.find("--dry-run"), std::string::npos);
    EXPECT_NE(profile_text.find("--traces"), std::string::npos);
    EXPECT_NE(profile_text.find("Select what to trace"), std::string::npos);
}

TEST(help_test, tool_help_flags_are_recognized_before_the_separator)
{
    EXPECT_TRUE(requests_tool_help(std::vector<std::string>{ "--help" }));
    EXPECT_TRUE(requests_tool_help(std::vector<std::string>{ "-h" }));
    EXPECT_TRUE(requests_tool_help(std::vector<std::string>{ "-?" }));
    EXPECT_TRUE(requests_tool_help(std::vector<std::string>{ "--help=all" }));
    EXPECT_TRUE(requests_tool_help(std::vector<std::string>{ "--verbose", "--help" }));
    EXPECT_FALSE(requests_tool_help(std::vector<std::string>{ "--verbose" }));
    EXPECT_FALSE(requests_tool_help(std::vector<std::string>{ "--helper" }));
    EXPECT_FALSE(requests_tool_help(std::span<const std::string>{}));
}

TEST(help_test, phase2_names_do_not_collide_with_attach_or_python)
{
    for(const auto name : k_phase2_names)
    {
        EXPECT_FALSE(contains(k_attach_long_options, name)) << name;
        EXPECT_FALSE(contains(k_python_long_options, name)) << name;
    }
    for(const auto& spec : option_table())
    {
        EXPECT_FALSE(contains(k_attach_long_options, spec.name)) << spec.name;
        EXPECT_FALSE(contains(k_python_long_options, spec.name)) << spec.name;
    }
    EXPECT_EQ(k_dry_run.name, "dry-run");
}
